"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.policies.teleop_policy import TeleopPolicy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--timeout', type=int, default=0, help="max number of steps per episode")
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=TeleopPolicy,
        params=AttrDict(timeout=common_params >> f"{group_name}/timeout")
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
