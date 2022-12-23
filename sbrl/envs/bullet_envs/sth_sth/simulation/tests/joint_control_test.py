## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np
import pybullet as p

from sbrl.envs.bullet_envs.sth_sth import BulletSth27
from sbrl.envs.bullet_envs.sth_sth import RCM, CT
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.utils import control_utils
from sbrl.utils.geometry_utils import world_frame_3D, CoordinateFrame
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

    draw_params = d(nrows=2, lines_per_axis=2, labels_per_axis=['curr', 'des'], axis_titles=[str(i) for i in range(6)],
                    ylim_tolerances=[0.1] * 6, steps_to_keep=100)

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
    params.robot_controller_mode = RCM.xddot_with_force_pid  # forces
    params.x_controller_params = d(type=CT.PD, k_p=0.0, dim=6, difference_fn=control_utils.pose_diff_fn)  # pose difference
    params.force_controller_params = d(type=CT.PI, k_p=0.0, dim=3, difference_fn=lambda a, b: a - b)  # force difference

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
    T = 400
    W = 5 * (2 * np.pi) / T  # N full oscillations

    # trajectory to follow
    pose = env.robot.get_end_effector_frame().as_pose(world_frame_3D)
    delta = 0.03 * (1 - np.cos(W * np.arange(T)))

    # pos + orn + force + grip
    waypoints = np.stack([list(pose) + [0., 0., 0.] + [0.]] * T)
    waypoints[:, 2] += delta  # z
    qdes_all = []
    for i in range(waypoints.shape[0]):
        # keep close
        des_frame = CoordinateFrame.from_pose(waypoints[i, :6], world_frame_3D)
        qdes = env.robot.compute_frame_ik(env.robotEndEffectorIndex, des_frame, rest_pose=env.robot.rp).tolist()
        qdes_grip = env.robot.gripper_q_from_pos(0.).tolist()
        qdes_all.append(qdes + qdes_grip)

    logger.info("Starting ctrl.")
    for i in range(waypoints.shape[0]):
        # keep close
        # des_frame = CoordinateFrame.from_pose(waypoints[i, :6], world_frame_3D)
        # qdes = env.robot.compute_frame_ik(env.robotEndEffectorIndex, des_frame, rest_pose=env.robot.rp)
        qdes = np.asarray(qdes_all[i])
        q = env.robot.get_joint_values()[env.robot.dof_joints]
        qdot = env.robot.get_joint_values(velocities=True)[env.robot.dof_joints]

        # control test
        kp = np.asarray([20.] * 7 + [2.] * 6)
        M = np.asarray(p.calculateMassMatrix(env.robotId, list(env.robot.get_joint_values()[env.robot.dof_joints]), physicsClientId=env.id))
        tau = M.dot(kp * (qdes - q) + 1. * (-qdot))
        # tau = list(tau) + [0] * (env.robot.num_dof_joints - len(tau))
        Nq = np.asarray(env.robot.N_q())

        # if i < 1:
        #     p.applyExternalForce(env.robotId, env.robotEndEffectorIndex, [0, 0, 0.1], [0., 0., 0.], p.WORLD_FRAME, physicsClientId=env.id)
        env.robot.joint_torque_control(env.robot.dof_joints, tau + Nq, step_simulation=True)

        # p.stepSimulation(physicsClientId=env.id)å
        queue.put((q[:6], qdes[:6])) #, tau[:6], env.robot.N_q()[:6]))
        env.slow_time(record=True)

    logger.info("Done.")
