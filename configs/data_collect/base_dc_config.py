"""
An example of how to integrate multiple separate parameterization files.
    this is the base set that doesn't change experiment to experiment.
    We can include default files by specifying loadable config files under each group here.
"""
import os
from argparse import ArgumentParser

from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args, get_path_module

# this has helper functions for the current set of experiments.
# utils = get_config_utils_module(__file__)

parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--exp_name', type=str, default="test")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--in_file', type=str, default=None)
parser.add_argument('--out_file', type=str, default=None)
parser.add_argument('--utils_module_file', type=str, default=None)
args = parser.parse_args(get_config_args())

# parser.add_argument('--in_file_train', type=str, default="stackBlock2D/scripted_multiplay_multishape_push_pull")
# parser.add_argument('--in_file_holdout', type=str, default=None)

# dset default, overrides params above
if args.in_file is not None:
    data_base_name = f"{args.exp_name}/{args.in_file}"
    in_file = os.path.join(ExperimentFileManager.experiments_dir, data_base_name)
else:
    in_file = "null"
args.in_file = in_file + ".npz"

dc_args = ['--input_file', args.in_file]
if args.out_file is not None:
    dc_args.extend(['--output_file', args.out_file])
dc_args += ['--suffix', 'train']


params = GroupedArgumentParser.to_attrs(args)  #exp_name=get_exp_name, utils=utils)

if args.utils_module_file is not None:
    params.utils = get_path_module("utils", args.utils_module_file)

# params.env_spec = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_spec_config.py")
# params.env_train = LoadedGroupedArgumentParser(file_name="configs/stack_block2d/stack_block2d_env_config.py", prepend_args=env_extra_args)
params.model = LoadedGroupedArgumentParser(file_name="configs/example/empty_model_config.py")
params.dataset_train = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=dc_args)
# params.dataset_holdout = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=['--input_file', args.in_file_holdout, '--suffix', 'holdout', '--frozen'])
# params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/train_only_config.py", prepend_args=['--learning_rate', str(args.lr)])
