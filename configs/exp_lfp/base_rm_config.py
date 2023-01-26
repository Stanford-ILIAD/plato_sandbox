"""
"""
import os
from argparse import ArgumentParser

# this has helper functions for the current set of experiments.
from configs.exp_lfp.rm_utils import Robot3DLearningUtils
from sbrl.experiments.file_manager import FileManager
from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.math_utils import round_to_n
from sbrl.utils.python_utils import AttrDict

parser = ArgumentParser()
parser.add_argument('--eval', action='store_true')
parser.add_argument('--goal_train', action='store_true')
parser.add_argument('--env_name', type=str, required=True, choices=['square', 'tool_hang', 'buds_kitchen'],
                    help='square or tool_hang, helpful preset')
parser.add_argument('--do_posact', action='store_true')
parser.add_argument('--use_random_eval', action='store_true')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=2e-4 / 512)
parser.add_argument('--lr_absolute', type=float, default=None, help="if specified, will override lr.")
parser.add_argument('--decay', type=float, default=None, help="if specified, will override default (5e-6)")
parser.add_argument('--split_frac', type=float, default=None, help="if specified, will set train and holdout splits.")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--min_horizon', type=int, default=None)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--dataset_val', type=str, default=None)
parser.add_argument('--extra_name', type=str, default='')
parser.add_argument('--long', action='store_true', help='e.g. for ToolHang env')
parser.add_argument('--fast_dynamics', action='store_true', help='ups the speed of the waypoint following action to max norm')
parser.add_argument('--clip_l1', action='store_true', help='ups the speed of the waypoint following action to max l1 (even more)')
parser.add_argument("--augment_prefix", type=str, default="aug")
parser.add_argument("--augment_keys", type=str, nargs="*", default=[])
parser.add_argument("--augment_stds", type=str, nargs="*", default=[],
                    help="gaussian noise with these std's. broadcast-able to true shape")
args = parser.parse_args(get_config_args())

env_name = args.env_name
true_env_name = {'square': 'NutAssemblySquare', 'tool_hang': 'ToolHang', 'buds_kitchen': 'KitchenEnv'}[env_name]
true_img_size = {'square': 84, 'tool_hang': 240, 'buds_kitchen': 128}[env_name]
steps_per_rollout = {'square': 400, 'tool_hang': 700, 'buds_kitchen': 1200}[env_name]
no_ori = env_name == 'buds_kitchen'

utils = Robot3DLearningUtils(fast_dynamics=args.fast_dynamics or args.clip_l1, clip_l1=args.clip_l1, no_ori=no_ori, use_target_gripper=args.do_posact)


def get_exp_name(common_params):
    # lazily executed, gets the name from the overall common_params.
    # might be chained by future experiments
    hr_lr = str(round_to_n(common_params.lr * common_params.batch_size, n=1)).replace('.', '_')
    ac_type = "velact" if common_params >> "velact" else "posact"
    if "seed" in common_params.leaf_keys() and common_params['seed'] is not None:
        ac_type += f"_seed{common_params >> 'seed'}"
    if len(common_params >> "augment_keys") > 0:
        ac_type += f"_{common_params >> 'augment_prefix'}-{len(common_params >> 'augment_keys')}"
    ac_type += common_params >> 'extra_name'
    if common_params >> "decay" is not None:
        hr_dec = str(round_to_n(common_params.decay, n=1)).replace('.', '_') if common_params.decay > 0 else "0"
        hr_dec = f"_dec{hr_dec}"
    else:
        hr_dec = ""

    NAME = f"hvsBlock3D/{ac_type}_b{common_params >> 'batch_size'}_lr{hr_lr}{hr_dec}_h{common_params >> 'min_horizon'}-{common_params >> 'horizon'}"

    dataset_choice = common_params >> "dataset"

    # suffix by dataset
    NAME = NAME + f"_{dataset_choice}"
    if common_params >> 'split_frac' is not None and 1 > common_params.split_frac > 0:
        hr_spl = str(common_params >> 'split_frac').replace('.', '_')
        NAME += f"_split{hr_spl}"

    return NAME

# dset default, overrides params above
data_base_name = f"hvsBlock3D/{args.dataset}"
in_file = os.path.join(FileManager.data_dir, data_base_name)
in_file_holdout = (in_file + "_val") if args.dataset_val is None \
    else os.path.join(FileManager.data_dir, f'hvsBlock3D/{args.dataset_val}')
args.in_file_train = in_file + ".npz"

# do not use validation dataset if split is specified.
if args.split_frac is not None and 1 > args.split_frac > 0:
    in_file_holdout = in_file

args.in_file_holdout = in_file_holdout + ".npz"
n_rollouts = 50  # TODO 20

# NOTE this does not enable images, you have to do that manually.
env_extra_args = ['--env_name', true_env_name, '--img_height', str(true_img_size), '--img_width', str(true_img_size)]

if not args.eval:
    assert os.path.exists(args.in_file_train), f"Train file does not exist: {args.in_file_train}"
    assert os.path.exists(args.in_file_holdout), f"Holdout file does not exist: {args.in_file_holdout}"

    if args.goal_train and not args.use_random_eval and env_name != 'buds_kitchen':
        if env_name == 'tool_hang':
            n_rollouts = 36
        env_extra_args.extend(['--enable_preset_sweep_n', str(n_rollouts)])

# maximum
if args.long:
    steps_per_rollout = 700

policy_extra_args = ['--fill_extra_policy_names', '--timeout', str(steps_per_rollout)]  #  '--flush_horizon', '0'
goal_policy_extra_args = []

# effectively global params
if args.min_horizon is None:
    args.min_horizon = args.horizon
if args.lr_absolute is not None:
    args.lr = args.lr_absolute / args.batch_size  # effectively makes lr_absolute the true lr in the trainer.
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils, velact=not args.do_posact)

extra_train_args = ['--save_checkpoint_every_n_steps 20000']  # save more frequently just in case.
if len(args.augment_keys) > 0:
    extra_train_args = ['--augment_keys', *args.augment_keys, '--augment_stds', *args.augment_stds]
if args.decay is not None:
    extra_train_args.extend(['--decay', str(args.decay)])

# both train and holdout (dataset auto segments each appropriately based on name)
extra_ds_args = []
if args.split_frac is not None and 1 > args.split_frac > 0:
    extra_ds_args.extend(['--split_frac', str(args.split_frac)])

env_spec_args = ['--include_target_names', '--include_target_gripper'] if args.do_posact else []
if env_name == 'buds_kitchen':
    params.env_spec = LoadedGroupedArgumentParser(file_name="configs/robosuite/rsz_env_spec_config.py", prepend_args=env_spec_args)
else:
    params.env_spec = LoadedGroupedArgumentParser(file_name="configs/robosuite/rs_env_spec_config.py", prepend_args=env_spec_args)

params.env_train = LoadedGroupedArgumentParser(file_name="configs/robosuite/rs_env_config.py", prepend_args=env_extra_args)
params.policy = LoadedGroupedArgumentParser(file_name="configs/exp_lfp/lmp_policy_config.py", prepend_args=policy_extra_args)
params.goal_policy = LoadedGroupedArgumentParser(file_name="configs/policies/timeout_policy_config.py", prepend_args=goal_policy_extra_args)
params.dataset_train = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=['--input_file', args.in_file_train, '--suffix', 'train', '--frozen'] + extra_ds_args)
params.dataset_holdout = LoadedGroupedArgumentParser(file_name="configs/datasets/np_dataset_config.py", prepend_args=['--input_file', args.in_file_holdout, '--suffix', 'holdout', '--frozen'] + extra_ds_args)

if args.goal_train:
    extra_gt_args = ["--block_env_for_n_steps", "20000",
                     "--rollout_env_every_n_steps", "20000",
                     "--rollout_env_n_per_step", str(n_rollouts),  # e.g., 20 rollouts per 20k steps
                     "--no_data_saving",
                     "--save_best_model",  # will keep track of the model that does the best (when it can)
                     "--reward_reduction", "max",
                     "--write_returns_every_n_env_steps", "0",  # disable return writing on steps, which will write after each rollout step instead (steps vs. eps)
                     "--return_buffer_len", str(n_rollouts)]
    params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/goaltrain_config.py", prepend_args=['--learning_rate', str(args.lr), '--max_train_steps', "4e6"] + extra_train_args + extra_gt_args)
else:
    params.trainer = LoadedGroupedArgumentParser(file_name="configs/trainers/train_only_config.py", prepend_args=['--learning_rate', str(args.lr), '--max_train_steps', "4e6"] + extra_train_args)
