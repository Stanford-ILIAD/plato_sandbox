from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.policies.blocks.block2d_rewards import Block2DReward
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # TODO add tolerances as arguments
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert "reward" in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    rew_params = AttrDict(
        cls=Block2DReward,
        params=AttrDict()
    )

    common_params.reward = common_params.reward & rew_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
