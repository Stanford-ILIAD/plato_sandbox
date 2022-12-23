"""
"""
from argparse import ArgumentParser

from sbrl.envs.polymetis.polymetis_panda_env import PolymetisPandaEnv, HOMES
from sbrl.envs.sensor.camera import USBCamera
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--hz', type=int, default=10)
    parser.add_argument('--action_space', type=str, default="ee-euler-delta")
    parser.add_argument('--use_gripper', action='store_true')
    parser.add_argument('--imgs', action='store_true')
    parser.add_argument('--img_width', type=float, default=640)
    parser.add_argument('--img_height', type=float, default=480)
    return parser


def process_params(group_name, common_params):
    prms = common_params >> group_name

    imgs = prms >> 'imgs'

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=PolymetisPandaEnv,
        params=AttrDict(
            home=HOMES['default'],
            hz=prms >> 'hz',
            use_gripper=prms >> 'use_gripper',  # TODO
            franka_ip="172.16.0.1",
            action_space=prms >> 'action_space',
        )
    )

    if imgs:
        H, W = prms >> 'img_height', prms >> 'img_width'
        common_params[group_name].params.camera = AttrDict(
            cls=USBCamera,
            params=AttrDict(
                img_width=W,
                img_height=H,
            )
        )

    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
