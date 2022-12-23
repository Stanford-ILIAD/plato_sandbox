"""
Robot controller configs for panda, etc.
"""
import numpy as np

from sbrl.envs.bullet_envs.utils_env import RobotControllerMode as RCM
from sbrl.policies.controllers.pid_controller import ControlType as CT
from sbrl.utils import control_utils
from sbrl.utils.python_utils import AttrDict as d

os_force_control_panda_cfg = d(
    robot_controller_mode=RCM.xddot_with_force_pid,
    x_controller_params=d(type=CT.PI,
                          k_p=[100] * 3 + [200] * 3, k_i=[50.] * 3 + [100.] * 3, k_d=0., dim=6, int_span=4.,
                          difference_fn=control_utils.pose_diff_fn),
    xdot_controller_params=d(type=CT.P, k_p=[6] * 3 + [12] * 3, dim=6),
    force_controller_params=d(type=CT.PI, k_p=[0.25, 0.25, 0.25] + [0., 0., 50.], k_i=[5., 5., 5.] + [0., 0., 1000.], dim=6, int_span=2.),
    posture_q_controller_params=d(type=CT.P, k_p=10.0, k_i=0.0, dim=13),
    posture_qdot_controller_params=d(type=CT.P, k_p=2.0, k_i=0.0, dim=13),
)

os_zero_force_control_panda_cfg = d(
    robot_controller_mode=RCM.xddot_with_zero_force_pid,
    x_controller_params=d(type=CT.PI,
                          k_p=[100] * 3 + [120] * 3, k_i=[0.] * 6, k_d=0., dim=6, int_span=4.,
                          difference_fn=control_utils.pose_diff_fn),
    xdot_controller_params=d(type=CT.P, k_p=[10] * 3 + [12] * 3, dim=6),
    force_controller_params=d(type=CT.PI, k_p=[0.25, 0.25, 0.25] + [0., 0., 100.], k_i=[5., 5., 5.] + [0., 0., 100.], dim=6, int_span=2.,
                              difference_fn=control_utils.get_dead_band_difference_fn(-0.1, 0.1, smooth=True)),
    posture_q_controller_params=d(type=CT.P, k_p=10.0, k_i=0.0, dim=13),
    posture_qdot_controller_params=d(type=CT.P, k_p=2.0, k_i=0.0, dim=13),
)

posture_gains = [3., 3., 3., 3., 3., 3., 3.] + 6 * [0] # [5., 5., 4., 3., 3., 3., 3.] + 6 * [0]
posture_dampings = (np.sqrt(np.asarray(posture_gains)) * 2).tolist()  # critical
os_torque_control_panda_cfg = d(
    robot_controller_mode=RCM.xddot_pid,
    x_controller_params=d(type=CT.P, k_p=[100] * 3 + [100] * 3, dim=6,
                          difference_fn=control_utils.pose_diff_fn),
    xdot_controller_params=d(type=CT.P, k_p=[20] * 3 + [20] * 3, dim=6),
    posture_q_controller_params=d(type=CT.P, k_p=posture_gains, dim=13),
    posture_qdot_controller_params=d(type=CT.P, k_p=posture_dampings, dim=13),
    # posture_qdot_controller_params=d(type=CT.P, k_p=[1., 1., 0.8, 0.6, .6, .6, .6] + 6 * [0], dim=13),
)

os_torque_control_strong_panda_cfg = d(
    robot_controller_mode=RCM.xddot_pid,
    x_controller_params=d(type=CT.P, k_p=[3000] * 3 + [2000] * 3, dim=6,
                          difference_fn=control_utils.pose_diff_fn),
    xdot_controller_params=d(type=CT.P, k_p=[100] * 3 + [80] * 3, dim=6),
    posture_q_controller_params=d(type=CT.P, k_p=posture_gains, dim=13),
    posture_qdot_controller_params=d(type=CT.P, k_p=posture_dampings, dim=13),
    # posture_qdot_controller_params=d(type=CT.P, k_p=[1., 1., 0.8, 0.6, .6, .6, .6] + 6 * [0], dim=13),
)

os_torque_control_zstiff_panda_cfg = d(
    robot_controller_mode=RCM.xddot_pid,
    x_controller_params=d(type=CT.P, k_p=[100] * 2 + [500] + [60] * 3, dim=6,
                          difference_fn=control_utils.pose_diff_fn),
    xdot_controller_params=d(type=CT.P, k_p=[20] * 3 + [10] * 3, dim=6),
    posture_q_controller_params=d(type=CT.P, k_p=posture_gains, dim=13),
    posture_qdot_controller_params=d(type=CT.P, k_p=posture_dampings, dim=13),
    # posture_qdot_controller_params=d(type=CT.P, k_p=[1., 1., 0.8, 0.6, .6, .6, .6] + 6 * [0], dim=13),
)


os_position_control_panda_cfg = d(
    robot_controller_mode=RCM.x_pid,
    x_controller_params=d(type=CT.PD, k_p=1., dim=6, difference_fn=control_utils.pose_diff_fn),
)
