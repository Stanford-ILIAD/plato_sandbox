"""
"""
import os
from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import achilles.utils.file_utils
import numpy as np
from achilles.robots.ros_panda_interface import ROSPandaInterface
from achilles.utils.yaml_config import YamlConfig
from scipy.spatial.transform import Rotation

from sbrl.envs.block_real.real_robot_env import RealRobotEnv
from sbrl.utils.config_utils import get_path_module
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import AttrDict

block_sensor_module = get_path_module("block_sensor_module",
                                      os.path.join(os.path.dirname(__file__), "multi_cam_block_sensor_config.py"))

# TIP_POS_IN_EE = np.array([0., 0., 0.0])
TIP_POS_IN_EE = np.array([0., 0., 0.15])
ROBOT_IN_SIM_GLOBAL = np.array([0.4, 0.5, 0.])  # taken from simulator (BulletRobot.start_pos)
# sim world (+x) = real world (+y)
ROBOT_REL_WORLD = CoordinateFrame(world_frame_3D, Rotation.from_euler('z', np.pi / 2), ROBOT_IN_SIM_GLOBAL)
RESET_Q = [0., -3.55341683e-01, -1.40604696e-01, -2.69942909e+00, -6.86523834e-02, 2.34899834e+00, 0.699]  # 5.10147958e-02]

object_sensor_frame = CoordinateFrame(world_frame_3D, Rotation.identity(), np.array([0.4, -0.025, 0.]))  # y used to be -0.025

def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--smoothing', type=float, default=1.)
    parser.add_argument('--no_object_sensor', action='store_true')
    block_sensor_module.declare_arguments(parser)
    return parser


def process_params(group_name, common_params):
    prms = common_params >> group_name

    if prms >> "no_object_sensor":
        block_sensor_params = None
    else:
        block_sensor_params = block_sensor_module.process_params(group_name, common_params) >> group_name

    # default achilles ros config
    ros_cfg_yaml = os.path.join(achilles.utils.file_utils.FileManager.yamls_dir, "ros2.yaml")
    ros_params = AttrDict.from_dict(YamlConfig(ros_cfg_yaml).to_attrs().as_dict())

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=RealRobotEnv,
        params=AttrDict(
            dt=prms >> "dt",
            robot_interface=AttrDict(
                cls=ROSPandaInterface,
                params=ros_params & AttrDict(
                    name="panda_controller",
                    # high level: goal setting frequency
                    policy_frequency=int(1 / (prms >> "dt")),
                    # default reset, for example
                    neutral_joint_angles=RESET_Q,
                    # neutral_joint_angles=[0.0, -0.524, 0.0, -2.617, 0.0, 2.094, 0.1],
                )
            ),
            object_sensor=block_sensor_params,
            object_sensor_frame=object_sensor_frame,
            tip_pos_in_ee=TIP_POS_IN_EE,
            robot_rel_world=ROBOT_REL_WORLD,
            reset_q=np.array(RESET_Q),
            smoothing=prms >> "smoothing",
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
