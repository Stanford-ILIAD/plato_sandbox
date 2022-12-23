from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.block_real.real_robot_env import get_block_real_example_spec_params
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and prefixes by goal/
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert group_name == "env_spec"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    env_spec_params = get_block_real_example_spec_params(NB=1, img_width=640, img_height=480)

    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=env_spec_params
    )

    common_params.env_spec = common_params.env_spec & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
