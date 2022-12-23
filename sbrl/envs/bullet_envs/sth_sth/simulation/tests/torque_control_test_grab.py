## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np
import torch
from pybullet_planning import wrap_angle
from sbrl.envs.episode_env import EpisodeEnv

from sbrl.envs.bullet_envs.sth_sth import BulletSthGrabLift27
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.controllers import os_torque_control_panda_cfg
from sbrl.policies.sth.sth_policies import TT, GrabItemTrajectory
from sbrl.utils.plt_utils import drawLineAnimations
from sbrl.utils.python_utils import AttrDict as d, timeit
from sbrl.utils.torch_utils import concatenate, to_numpy

IMG_H = 32
IMG_W = 32
IMG_C = 3  # color
MAX_ENV_STEPS = 50

ACT_DIM = 3 + 3 + 1  # using cartesian ee pos, orn_eul, gripper
ACT_LOW = np.array([-0.05, -0.05, -0.05, -np.pi / 10, -np.pi / 10, -np.pi / 10])  # we use small pose deltas
ACT_LOW = np.append(ACT_LOW, [0]).astype(np.float32)  # final pose deformations (1cm)
ACT_HIGH = -ACT_LOW
ACT_HIGH[-1] = 255.  # gripper max (abs)


class GetDrawPts:
    def __call__(self, queue_data, *args, **kwargs):
        return np.stack(queue_data).T  # (2, num_axes) -> (num_axes, 2)


def pose_grab_fn(obs: d, goal: d, env_actions: d, idx: int):
    # all (1 x ...), returns the "delta" for this time step conditioned on current position
    out = d(action=env_actions.delta_action[:, idx].copy())
    return out


def env_reward_fn(obs: d, goals: d, action: d, next_obs: d, done) -> torch.Tensor:
    # normalized 0 to 1 across max env length
    delta = action.action[..., :6] - concatenate(obs, ['ee_position', 'ee_orientation_eul'], dim=-1)

    pos_penalty = 1. / (MAX_ENV_STEPS * np.sqrt(3) * ACT_HIGH[0]) * ((delta[..., :3] ** 2).sum(-1) ** 0.5)[
        ..., None]
    orn_penalty = 1. / (MAX_ENV_STEPS * np.sqrt(3) * ACT_HIGH[2]) * \
                  ((wrap_angle(delta[..., 3:6]) ** 2).sum(-1) ** 0.5)[..., None]

    # xy distance
    xy_dist_to_object = \
    (((next_obs.gripper_tip_position[..., :2] - next_obs.object_0.position[..., :2]) ** 2).sum(-1) ** 0.5)[..., None]

    # inputs should all be B x ...
    rew = next_obs.success + -0.1 * next_obs.table_collision \
          + -0.1 * next_obs.task_collision \
          + -0.1 * next_obs.robot_collision \
          + -0.1 * (2 * orn_penalty + pos_penalty) \
          + 0.075 * next_obs.object_0.is_gripping \
          + 0.05 * (xy_dist_to_object < 0.04)  # within 4 cm of the object, xy
    # at most for 50 steps 1. reward
    return rew  # B x 1


if __name__ == '__main__':
    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    queue = mp.Queue()

    draw_params = d(nrows=3, lines_per_axis=2, labels_per_axis=['curr', 'des'],
                    axis_titles=['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz'],
                    ylim_tolerances=[0.1] * 6 + [1] * 3, steps_to_keep=1000)

    proc = Process(target=drawLineAnimations, args=(queue, GetDrawPts(), draw_params), daemon=True)
    proc.start()

    max_steps = 5000

    params = d(
        get_action_from_env_actions=pose_grab_fn,
        env_params=d(
            cls=BulletSthGrabLift27,
            params=d(
                max_steps=50,
                time_step=.025,
                skip_n_frames_every_step=2,
                control_inner_step=True,
                task_id=27,
                render=True,
                compute_images=False,
                env_reward_fn=env_reward_fn,
                randomize_object_start_location=True,
                start_in_grasp=True,
                # position control specification for panda in sim
            )
        )
    )
    params.env_params.params.combine(os_torque_control_panda_cfg)
    params.pprint(str_max_len=100)

    # lifting env
    env = EpisodeEnv(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        # initial obs
        ("img", (IMG_H, IMG_W, IMG_C), (0, 255), np.uint8),
        ('seg', (IMG_H, IMG_W), (0, 1), np.int),

        # obs
        ("ee_position", (3,), (-np.inf, np.inf), np.float32),
        ("ee_orientation_eul", (3,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("ee_velocity", (3,), (-np.inf, np.inf), np.float32),
        ("joint_positions", (20,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("joint_velocities", (20,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("wrist_ft", (6,), (-np.inf, np.inf), np.float32),
        ("gripper_pos", (1,), (0, 255), np.float32),
        ("gripper_tip_position", (3,), (-np.inf, np.inf), np.float32),
        ("object_0/position", (3,), (-np.inf, np.inf), np.float32),
        ("object_0/is_gripping", (1,), (False, True), np.float32),

        # reward
        ('success', (1,), (False, True), np.bool),
        ('table_collision', (1,), (False, True), np.bool),
        ('task_collision', (1,), (False, True), np.bool),
        ('robot_collision', (1,), (False, True), np.bool),

        # next obs
        ("next_ee_position", (3,), (-np.inf, np.inf), np.float32),
        ("next_ee_orientation_eul", (3,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("next_ee_velocity", (3,), (-np.inf, np.inf), np.float32),
        ("next_joint_positions", (20,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("next_joint_velocities", (20,), (-2 * np.pi, 2 * np.pi), np.float32),
        ("next_wrist_ft", (6,), (-np.inf, np.inf), np.float32),
        ("next_gripper_pos", (1,), (0, 255), np.float32),
        ("next_gripper_tip_position", (3,), (-np.inf, np.inf), np.float32),
        ("object_0/next_position", (3,), (-np.inf, np.inf), np.float32),
        ("object_0/next_is_gripping", (1,), (False, True), np.float32),

        # action
        ('delta_action', (MAX_ENV_STEPS, ACT_DIM), (ACT_LOW[None], ACT_HIGH[None]), np.float32),
        # final euler angles (ee pose) + trajectory
        ('action', (ACT_DIM,), (-np.inf, -np.inf), np.float32),  # final euler angles (ee pose) + trajectory

        # mapping
        ('reward', (1,), (-np.inf, -np.inf), np.float32),
    ], observation_names=["ee_position", "ee_orientation_eul", "ee_velocity", "joint_positions", "joint_velocities",
                          "wrist_ft", "gripper_pos", "gripper_tip_position", "object_0/position", "object_0/is_gripping"],
        action_names=['delta_action'],
        output_observation_names=['success', 'task_collision', 'table_collision', 'robot_collision', "reward"],
        param_names=[], goal_names=[], final_names=[]
    )))  # gym.make('BiteTransferPanda-v0')

    policy_params = d(
        total_steps=MAX_ENV_STEPS,
        steps=35,
        cat_name="delta_action",
        trajectory_mode=TT.pose_grab,
    )

    policy = GrabItemTrajectory(policy_params, env.env_spec, env=env)
    model = Model(d(ignore_inputs=True), env.env_spec, None)

    presets = d()

    observation, goal = env.reset(presets)
    policy.reset_policy(observation, goal)

    env_action = policy.get_action(model, observation, goal)
    env_action.leaf_modify(lambda arr: to_numpy(arr, check=True))
    print("Delta:", env_action.delta_action)
    all_next_obs, all_next_goal, done = env.step(env_action)

    print(timeit)

    logger.info("Done.")
