"""
An example of how to integrate multiple separate parameterization files.
    this is the base set that doesn't change experiment to experiment.
    We can include default files by specifying loadable config files under each group here.
"""
import os
from argparse import ArgumentParser

from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args, get_config_utils_module
from sbrl.utils.python_utils import AttrDict

# this has helper functions for the current set of experiments.
utils = get_config_utils_module(os.path.join(os.path.dirname(__file__), '../exp_lfp/base_config.py'))


parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--is_online_exp', action='store_true')
args = parser.parse_args(get_config_args())

def get_exp_name(common_params):
    # lazily executed, gets the name from the overall common_params.
    # might be chained by future experiments
    return f"stackblock2d/_b{common_params >> 'batch_size'}_h{common_params >> 'horizon'}"


# effectively global params
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)


params.env_spec = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_config.py")
