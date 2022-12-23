"""
Base config for Block2D experiments.
"""
import os
from argparse import ArgumentParser

from configs.exp_lfp.utils import Block2DLearningUtils
from sbrl.experiments.file_manager import FileManager
from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.python_utils import AttrDict

utils = Block2DLearningUtils()

# this has helper functions for the current set of experiments.
parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--velact', action='store_true')
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=2e-4 / 512)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--min_horizon', type=int, default=None)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--dataset_val', type=str, default=None)
# contact settings
parser.add_argument('--contact', action='store_true')
parser.add_argument('--init_horizon', type=int, default=None)
parser.add_argument('--contact_horizon', type=int, default=None)

args = parser.parse_args(get_config_args())


def get_exp_name(common_params):
    # lazily executed, gets the name from the overall common_params.
    # might be chained by future experiments
    hr_lr = str((common_params >> "lr") * (common_params >> "batch_size")).replace('.', '_')
    ac_type = "velact" if common_params >> "velact" else "posact"
    if common_params >> 'contact':
        ac_type += "_contact"
    if "seed" in common_params.leaf_keys() and common_params['seed'] is not None:
        ac_type += f"_seed{common_params >> 'seed'}"
    NAME = f"stackBlock2D/{ac_type}_b{common_params >> 'batch_size'}_lr{hr_lr}_h{common_params >> 'min_horizon'}-{common_params >> 'horizon'}"

    if common_params >> "contact_horizon" is not None:
        NAME = f"{NAME}-c{common_params >> 'contact_horizon'}"

    if common_params >> "init_horizon" is not None:
        NAME = f"{NAME}-i{common_params >> 'init_horizon'}"

    # suffix by dataset
    dataset_choice = common_params >> "dataset"
    NAME = NAME + f"_{dataset_choice}"

    return NAME


# dset default, overrides params above
data_base_name = f"stackBlock2D/{args.dataset}"
in_file = os.path.join(FileManager.data_dir, data_base_name)
in_file_holdout = (in_file + "_val") if args.dataset_val is None \
    else os.path.join(FileManager.data_dir, f'stackBlock2D/{args.dataset_val}')
args.in_file_train = in_file + ".npz"
in_file_holdout += ".npz"

env_extra_args = ['--start_near_bottom']

# effectively global params
if args.min_horizon is None:
    args.min_horizon = args.horizon
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)

prefix = "_interaction" if args.contact else ""
extra_train_dset_args = ['--fast'] if args.contact else []

params.env_spec = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_config.py",
                                               prepend_args=env_extra_args)
params.dataset_train = LoadedGroupedArgumentParser(file_name=f"configs/datasets/np{prefix}_dataset_config.py",
                                                   prepend_args=['--input_file', args.in_file_train,
                                                                 '--suffix', 'train', '--frozen'] + extra_train_dset_args)
params.dataset_holdout = LoadedGroupedArgumentParser(file_name=f"configs/datasets/np{prefix}_dataset_config.py",
                                                     prepend_args=['--input_file', in_file_holdout, '--suffix',
                                                                   'holdout', '--frozen'])
params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/train_only_config.py", 
                                             prepend_args=['--learning_rate', str(args.lr)])

goal_policy_extra = ['--use_policy_type', '--sort_primitives_by_motion', '--no_pull_directional_primitives']

# hardcoded relationships between datasets and some default goal_policy arguments, kinda hacky
if "human_play" in args.dataset:
    goal_policy_extra.extend(["--do_push", "--do_pull", "--do_tip", "--do_side_rot"])
else:
    if "_push" in args.dataset:
        goal_policy_extra.append('--do_push')
    if "_pull" in args.dataset:
        goal_policy_extra.append('--do_pull')
    if "_tip" in args.dataset:
        goal_policy_extra.append('--do_tip')
    if "srot" in args.dataset:
        goal_policy_extra.append('--do_side_rot')
    if "_rot" in args.dataset:
        goal_policy_extra.append('--do_lift_rot')


# compatibility with goal_trainer
params.goal_policy = LoadedGroupedArgumentParser(file_name="configs/data_collect/stackblock2d_goal_policy_config.py", 
                                                 prepend_args=goal_policy_extra)
params.policy = LoadedGroupedArgumentParser(file_name="configs/exp_lfp/lmp_policy_config.py")
