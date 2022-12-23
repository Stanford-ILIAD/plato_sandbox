from argparse import ArgumentParser

from sbrl.policies.blocks.block3d_goal_policies import Block3DHardcodedGoalPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    policy_params = d(
        cls=Block3DHardcodedGoalPolicy,
        params=d(),
    )

    common_params[group_name] = common_params[group_name] & policy_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
