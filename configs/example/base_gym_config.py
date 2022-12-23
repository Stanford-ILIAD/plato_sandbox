"""
An example of how to integrate multiple separate parameterization files.
    this is the base set that doesn't change experiment to experiment.
    We can include default files by specifying loadable config files under each group here.
"""
from argparse import ArgumentParser

# this has helper functions for the current set of experiments.
from configs.example import utils
from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.python_utils import AttrDict

# Set up a parser for just the base config parameters, these should be "global" fields that many modules will need.
parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
args = parser.parse_args(get_config_args())


def get_exp_name(common_params):
    """
    The loading process will call this to figure out the experiment name (which determines where your experiment saves).
    """
    # lazily executed, gets the name from the overall common_params.
    # this function might be chained by future experiments to add on more details to the name.
    env_type = common_params >> "env_train/env_type"  # required for gym envs
    env_type_hr = env_type.lower()
    return f"gym/{env_type_hr}_b{common_params >> 'batch_size'}_h{common_params >> 'horizon'}"


# GLOBAL parameters, which will be the starting point for "common_params" seen in sub-groups
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)

# example of how to set default groups, here with the gym "env_spec" and "env_train"
# some useful flags for this parser:
#   prepend_args: Some initial arguments to pass in to the config, for more precise defaults.
#   allow_override: If False, this group's arguments will be FIXED.
params.env_spec = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_config.py")
