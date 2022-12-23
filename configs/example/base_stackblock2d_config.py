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
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=int, default=1e-3 / 1024)
args = parser.parse_args(get_config_args())

# parser.add_argument('--in_file_train', type=str, default="stackBlock2D/scripted_multiplay_multishape_push_pull")
# parser.add_argument('--in_file_holdout', type=str, default=None)


def get_exp_name(common_params):
    # lazily executed, gets the name from the overall common_params.
    # might be chained by future experiments
    return "test"

env_extra_args = ['--start_near_bottom']

# effectively global params
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)


params.env_spec = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_config.py", prepend_args=env_extra_args)
params.model = LoadedGroupedArgumentParser(file_name="configs/example/empty_model_config.py")
params.policy = LoadedGroupedArgumentParser(file_name="configs/example/random_policy_config.py")
params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/train_only_config.py", prepend_args=['--learning_rate', str(args.lr)])
