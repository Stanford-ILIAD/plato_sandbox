import argparse
import os

import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.file_utils import import_config, file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="The configuration script for the given dataset / env spec")
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--output_file_split', type=str, required=True)
parser.add_argument('--split_frac', type=float, default=0.5)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()


def get_split_condition(dataset: NpDataset, args):
    SPLIT_FRAC = args.split_frac
    SHUFFLE = args.shuffle

    # index based
    num_eps = dataset.get_num_episodes()
    num_to_split = int(np.round(SPLIT_FRAC * num_eps))
    assert num_to_split >= 1, "Don't split datasets with num_eps=%d, split = %f" % (num_eps, SPLIT_FRAC)

    if num_to_split < 5:
        logger.warn("Split results in few episodes for split: %d" % num_to_split)

    if SHUFFLE:
        indices = np.random.choice(num_eps, num_to_split, replace=False)  # unique to split off
    else:
        indices = np.arange(num_to_split)

    def split_condition(ep: AttrDict, i):
        # return a modified episode (map), and condition for when we keep this (filter)
        return ep, i in indices

    return split_condition


config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

out_path = file_path_with_default_dir(args.output_file, file_manager.exp_dir)
out_path_split = file_path_with_default_dir(args.output_file_split, file_manager.exp_dir)

exit_on_ctrl_c()
env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)
policy = params.policy.cls(params.policy.params, env_spec)

assert params.dataset_train.params.get("horizon", 0), "Horizon must be nonzero"
params.dataset_train.params.file = args.file
dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

split_cond = get_split_condition(dataset_input, args)
all_names = env_spec.all_names + ["done", "rollout_timestep"]

out_ls = []
out_ls_split = []

# index based
for i in range(dataset_input.get_num_episodes()):
    # two empty axes for (batch_size, horizon)
    datadict = dataset_input.get_episode(i, all_names)

    new_ep, do_split = split_cond(datadict, i)
    if do_split:
        out_ls_split.append(new_ep)
    else:
        out_ls.append(new_ep)

    if i % 10000 == 0:
        logger.debug("Loading sample %d" % i)


logger.debug("Combining the dataset to keep")
out_datadict = AttrDict.leaf_combine_and_apply(out_ls, lambda vs: np.concatenate(vs, axis=0))

logger.debug("Combining the dataset to split")
out_datadict_split = AttrDict.leaf_combine_and_apply(out_ls_split, lambda vs: np.concatenate(vs, axis=0))


logger.debug("Keys: " + str(list(out_datadict.leaf_keys())))
logger.debug("Keys split: " + str(list(out_datadict_split.leaf_keys())))
logger.debug("keep data len: %d, split data len: %d" % (len(out_datadict.done), len(out_datadict_split.done)))

to_save = dict()
to_save_split = dict()
for name in out_datadict.leaf_keys():
    to_save[name] = out_datadict[name]
    to_save_split[name] = out_datadict_split[name]

logger.debug("Saving dataset output keep to -> %s" % out_path)
np.savez_compressed(out_path, **to_save)
logger.debug("Saving dataset output split to -> %s" % out_path_split)
np.savez_compressed(out_path_split, **to_save_split)
