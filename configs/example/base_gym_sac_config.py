"""
An example of how to integrate multiple separate parameterization files.
    this is the base set that doesn't change experiment to experiment.
    We can include default files by specifying loadable config files under each group here.
"""
from argparse import ArgumentParser

from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args, get_config_utils_module
from sbrl.utils.python_utils import AttrDict

# this has helper functions for the current set of experiments.
utils = get_config_utils_module(__file__)


parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--goal_train', action='store_true')
args = parser.parse_args(get_config_args())

def get_exp_name(common_params):
    # lazily executed, gets the name from the overall common_params.
    # might be chained by future experiments
    env_type = common_params >> "env_train/env_type"  # required for gym envs
    num_envs = common_params << "env_train/num_envs"  # not required

    env_type_hr = env_type.lower()
    NAME = f"gym/{args.prefix}{env_type_hr}_b{common_params >> 'batch_size'}_h{common_params >> 'horizon'}"

    if num_envs is not None and num_envs > 1:
        NAME += f"_vec{num_envs}"

    return NAME


# effectively global params
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)

# modules to load from unique configs.
params.env_spec = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_config.py")
params.model = LoadedGroupedArgumentParser(file_name="configs/example/sac_model_config.py")
params.policy = LoadedGroupedArgumentParser(file_name="configs/example/sac_policy_config.py")
params.goal_policy = LoadedGroupedArgumentParser(file_name="configs/policies/timeout_policy_config.py")
params.dataset_train = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=["--output_file", "sac_online.npz"])
params.dataset_holdout = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=["--output_file", "sac_online_holdout.npz"])
params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/sac_train_config.py")
if args.goal_train:
    params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/sac_goaltrain_config.py")
