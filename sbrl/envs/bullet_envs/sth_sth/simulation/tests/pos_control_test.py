## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np

from sbrl.envs.bullet_envs.sth_sth import BulletSth27
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers import os_position_control_panda_cfg
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

    draw_params = d(lines_per_axis=2, labels_per_axis=['curr', 'des'], axis_titles=['x', 'y', 'z', 'rx', 'ry', 'rz'],
                    ylim_tolerances=[0.1] * 6, steps_to_keep=100)

    proc = Process(target=drawLineAnimations, args=(queue, GetDrawPts(), draw_params), daemon=True)
    proc.start()

    max_steps = 5000
    params = d()
    params.task_id = 27
    params.render = True
    params.compute_images = False
    params.debug = False
    params.skip_n_frames_every_step = 5
    params.time_step = 0.01  # 20Hz

    params.combine(os_position_control_panda_cfg)  # pos

    env = BulletSth27(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('ee_orientation_eul', (3,), (-np.inf, np.inf), np.float32),
        ('ee_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('ee_angular_velocity', (3,), (-np.inf, np.inf), np.float32),
    ], observation_names=['wrist_ft', 'ee_orientation_eul', 'ee_position', 'ee_velocity', 'ee_angular_velocity'])))  # gym.make('BiteTransferPanda-v0')

    presets = d()

    observation, _ = env.reset(presets)
    env.robot.gripper_control(0, step_simulation=True)
    done = False

    # (1-cos) wave in the z axis
    T = 200
    W = 3 * (2 * np.pi) / T  # two full oscillations
    # trajectory to follow
    pose = env.robot.get_end_effector_frame().as_pose(world_frame_3D)
    delta = 0.1 * (1 - np.cos(W * np.arange(T)))

    waypoints = np.stack([list(pose) + [80.]] * T)
    waypoints[:, 2] += delta  # z

    logger.info("Starting ctrl.")
    for i in range(waypoints.shape[0]):
        observation, _, done = env.step(waypoints[i])
        pose = np.concatenate([observation.ee_position[0], observation.ee_orientation_eul[0]])
        des_pose = waypoints[i][:6]
        queue.put((pose, des_pose))
        # if done:
        #     logger.warn("Stopping early because done")
        #     break
        # time.sleep(0.5)

    logger.info("Done.")
