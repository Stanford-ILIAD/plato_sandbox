## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np

from sbrl.envs.bullet_envs.sth_sth import BulletSthGrabLift27
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers import os_force_control_panda_cfg
from sbrl.utils.geometry_utils import world_frame_3D
from sbrl.utils.plt_utils import drawLineAnimations
from sbrl.utils.python_utils import AttrDict as d, timeit


class GetDrawPts:
    def __call__(self, queue_data, *args, **kwargs):
        return np.stack(queue_data).T  # (2, num_axes) -> (num_axes, 2)


if __name__ == '__main__':
    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    queue = mp.Queue()

    draw_params = d(figsize=(10, 10), nrows=4, lines_per_axis=2, labels_per_axis=['curr', 'des'], axis_titles=['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'],
                    ylim_tolerances=[0.1] * 6 + [1] * 6, steps_to_keep=1000)

    proc = Process(target=drawLineAnimations, args=(queue, GetDrawPts(), draw_params), daemon=True)
    proc.start()

    max_steps = 5000
    params = d()
    params.task_id = 27
    params.render = True
    params.compute_images = False
    params.debug = False
    params.skip_n_frames_every_step = 2  # 10Hz
    params.time_step = 0.025  # 50Hz
    params.use_gravity = True

    params.start_in_grasp = True
    params.randomize_object_start_location = True
    # sets default control params
    params.combine(os_force_control_panda_cfg)
    # params.robot_params.apply_ft_transform = True  # log scale

    # lifting env
    env = BulletSthGrabLift27(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('ee_orientation_eul', (3,), (-np.inf, np.inf), np.float32),
        ('ee_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('ee_angular_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('joint_positions', (7,), (-np.inf, np.inf), np.float32),
    ], observation_names=['wrist_ft', 'ee_orientation_eul', 'ee_position', 'ee_velocity', 'ee_angular_velocity', 'joint_positions'])))  # gym.make('BiteTransferPanda-v0')

    presets = d(mass=0.1)  # kilo

    observation, _ = env.reset(presets)
    grip = 0
    # env.robot.gripper_control(grip, step_simulation=True)
    done = False

    # (1-cos) wave in the z axis
    T = int(20 / env.dt)
    W = 6 * (2 * np.pi) / T  # N full oscillations

    # trajectory to follow
    pose = env.robot.get_end_effector_frame().as_pose(world_frame_3D)
    delta = -0.07 * (np.cos(W * np.arange(T)) - 1)
    # delta = - 0.2 * (np.sin(W * np.arange(T)))
    # delta = 0.1 * (np.sin(W * np.arange(T))) + 0.05

    # pos + orn + force + grip
    waypoints = np.stack([list(pose) + [0.] * 6 + [grip]] * T)
    waypoints[:, 0] += delta  # x
    # waypoints[:, 1] -= 0.5 * delta  # y

    plen = 12

    logger.info("Starting ctrl.")
    for i in range(waypoints.shape[0]):
        act = np.concatenate([waypoints[i, :plen], waypoints[i, -1:]])
        observation, _, done = env.step(act)
        # pose = np.concatenate([observation.ee_position[0], observation.ee_orientation_eul[0]])
        pose = np.concatenate([observation.ee_position[0], observation.ee_orientation_eul[0],
                               observation.wrist_ft[0]])
        des_pose = waypoints[i][:12]
        queue.put((pose[:12], des_pose[:12]))
        # print(observation.joint_positions)

        # states = p.getJointStates(env.robotId, env.robot.controlled_joints, physicsClientId=env.id)
        # print("outer:", [s[3] for s in states])
        # if done:
        #     logger.warn("Stopping early because done")
        #     break
        # time.sleep(0.5)

    print(timeit)

    logger.info("Done.")
