from argparse import ArgumentParser

from sbrl.envs.gym_env import GymEnv, SUPPORTED
# declares this group's parser, and defines any sub groups we need
from sbrl.envs.vectorize_env import VectorizedEnv
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--env_type", type=str, choices=SUPPORTED.keys(), required=True)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--num_envs", type=int, default=1)
    # parser.add_argument("--img_width", type=int, default=256)
    # parser.add_argument("--img_height", type=int, default=256)
    # parser.add_argument("--img_channels", type=int, default=3)
    return parser


def process_params(group_name, common_params):
    assert "env" in group_name, f"Group name must have \'env\': {group_name}"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    env_params = d(
        cls=GymEnv,
        params=d(
            env_type=prms >> "env_type",
            render=prms >> "render",
        )
    )
    if prms >> "num_envs" > 1:
        env_params = d(
            cls=VectorizedEnv,
            params=d(
                env_params=env_params,
                num_envs=prms >> "num_envs"
            )
        )
    common_params[group_name] = common_params[group_name] & env_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
