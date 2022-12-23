## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np
import pybullet as p

from sbrl.envs.bullet_envs.sth_sth import BulletSth27
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers import os_force_control_panda_cfg
from sbrl.utils.geometry_utils import world_frame_3D
from sbrl.utils.plt_utils import drawLineAnimations
from sbrl.utils.python_utils import AttrDict as d


class GetDrawPts:
    def __call__(self, queue_data, *args, **kwargs):
        return np.stack(queue_data).T  # (2, num_axes) -> (num_axes, 2)


if __name__ == '__main__':
    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    queue = mp.Queue()

    draw_params = d(nrows=3, lines_per_axis=2, labels_per_axis=['curr', 'des'], axis_titles=['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz'],
                    ylim_tolerances=[0.1] * 6 + [1] * 3, steps_to_keep=1000)

    proc = Process(target=drawLineAnimations, args=(queue, GetDrawPts(), draw_params), daemon=True)
    proc.start()

    max_steps = 5000
    params = d()
    params.task_id = 27
    params.render = True
    params.compute_images = False
    params.debug = False
    params.skip_n_frames_every_step = 1
    params.time_step = 0.05  # 20Hz
    params.use_gravity = True

    params.start_in_grasp = True
    params.randomize_object_start_location = False
    # sets default control params
    params.combine(os_force_control_panda_cfg)

    # lifting env
    env = BulletSth27(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('ee_orientation_eul', (3,), (-np.inf, np.inf), np.float32),
        ('ee_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('ee_angular_velocity', (3,), (-np.inf, np.inf), np.float32),
    ], observation_names=['wrist_ft', 'ee_orientation_eul', 'ee_position', 'ee_velocity', 'ee_angular_velocity'])))  # gym.make('BiteTransferPanda-v0')

    presets = d()

    observation, _ = env.reset(presets)
    grip = 0
    env.robot.gripper_control(grip, step_simulation=True)
    done = False

    T = 400
    # trajectory to follow
    pose = env.robot.get_end_effector_frame().as_pose(world_frame_3D)

    # desired (pos + orn + force + grip) at each step
    waypoints = np.stack([list(pose) + [0., 0., 0.] + [grip]] * T)
    waypoints[:, 2] += 0.15  # raise z up

    logger.info("Starting ctrl.")
    for i in range(waypoints.shape[0]):
        observation, _, done = env.step(waypoints[i])
        # pose = np.concatenate([observation.ee_position[0], observation.ee_orientation_eul[0]])
        pose = np.concatenate([observation.ee_position[0], observation.ee_orientation_eul[0],
                               observation.wrist_ft[0][:3]])
        des_pose = waypoints[i][:9]
        queue.put((pose[:9], des_pose[:9]))

        if 25 < i < 50:
            p.applyExternalForce(env.robotId, env.robot.gripper_left_tip_index, [0., 0., 5.], [0., 0., 0.], p.LINK_FRAME, physicsClientId=env.id)
            p.applyExternalForce(env.robotId, env.robot.gripper_right_tip_index, [0., 0., 5.], [0., 0., 0.], p.LINK_FRAME, physicsClientId=env.id)

        # states = p.getJointStates(env.robotId, env.robot.controlled_joints, physicsClientId=env.id)
        # print("outer:", [s[3] for s in states])
        # if done:
        #     logger.warn("Stopping early because done")
        #     break
        # time.sleep(0.5)

    logger.info("Done.")
