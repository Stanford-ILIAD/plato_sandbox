"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import numpy as np
from oculus_reader import OculusReader
from scipy.spatial.transform import Rotation

from sbrl.envs.bullet_envs.utils_env import target_action_postproc_fn
from sbrl.policies.vr_teleop_policy import VRPoseTeleopPolicy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--action_name', type=str, default='action')
    parser.add_argument('--gripper_pos_name', type=str, default='gripper_pos')
    parser.add_argument('--gripper_tip_pos_name', type=str, default='gripper_tip_pos')
    parser.add_argument('--action_as_delta', action='store_true', help="Delta pose for env")
    parser.add_argument('--disable_gripper', action='store_true', help="Turns off gripper")
    parser.add_argument('--continuous_gripper', action='store_true', help="Will control gripper velocity")
    parser.add_argument('--gripper_action_space', type=str, default='default', help="default or normalized.")
    parser.add_argument('--sticky_gripper', action='store_true', help='turns gripper into a toggle')
    parser.add_argument('--use_click_state', action='store_true', help='B button becomes click = 1 label')
    parser.add_argument('--spatial_gain', type=float, default=1.)
    parser.add_argument('--pos_gain', type=float, default=1.0)
    parser.add_argument('--rot_gain', type=float, default=1.0)
    parser.add_argument('--hz', type=int, default=10)
    parser.add_argument('--rot_oculus_yaw', type=float, default=None, help="z-axis rotation of oculus relative to standard ori (facing the screen).")
    parser.add_argument('--clip_ori_max', type=float, default=None,
                        help="angle (deg) to clip ee ori, relative to z-axis. None=no clip")
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    prms = common_params >> group_name

    o2r = np.asarray([[0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])

    if prms >> "rot_oculus_yaw" is not None:
        r2nr = Rotation.from_euler('z', np.deg2rad(prms >> "rot_oculus_yaw")).as_matrix()
        o2r[:3, :3] = r2nr @ o2r[:3, :3]

    an = prms >> "action_name"
    gp = prms >> "gripper_pos_name"
    gtp = prms >> "gripper_tip_pos_name"

    MAX_VEL = 0.25
    MAX_ORI_VEL = 2.5
    HZ = prms >> 'hz'

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=VRPoseTeleopPolicy,
        params=AttrDict(
            action_name=an,
            gripper_pos_name=gp,
            gripper_tip_pos_name=gtp,
            postproc_fn=None if prms >> 'action_as_delta' else 
                    lambda model, obs, goal, act: target_action_postproc_fn(obs, act.leaf_apply(lambda arr: arr[:, None]),
                                                                                action_name=an,
                                                                                gripper_pos_name=gp,
                                                                                Kv_P=(10, 10, 10.),
                                                                                Kv_O=(10., 10., 10.),
                                                                                Kv_G=10.,
                                                                                max_pos_vel=0.25,
                                                                                max_orn_vel=2.5,
                                                                                use_gripper=not prms >> 'disable_gripper'),
            action_as_delta=prms >> "action_as_delta",
            use_gripper=not prms >> 'disable_gripper',
            spatial_gain=prms >> "spatial_gain",
            pos_gain=prms >> "pos_gain",
            rot_gain=prms >> "rot_gain",
            delta_pos_max=MAX_VEL / HZ,
            delta_rot_max=MAX_ORI_VEL / HZ,
            sticky_gripper=prms >> "sticky_gripper",
            use_click_state=prms >> "use_click_state",
            gripper_action_space=prms >> 'gripper_action_space',
            continuous_gripper=prms >> 'continuous_gripper',
            clip_ori_max=np.deg2rad(prms.clip_ori_max) if prms >> "clip_ori_max" is not None else None,
            oculus_to_robot_mat_4d=o2r,
            reader=AttrDict(
                cls=OculusReader,
                params=AttrDict(),
            )
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
