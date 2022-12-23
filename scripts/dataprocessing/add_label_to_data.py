"""
This file is very similar to train; it's basically a subset of train
"""

import argparse
import os

import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.config_utils import register_config_args
from sbrl.utils.file_utils import import_config, file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)  # config used to generate the dataset (used just for spec)
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--label_key', type=str, required=True)
parser.add_argument('--label_value', type=int, required=True)


args, unknown = parser.parse_known_args()
if len(unknown) > 0:
    register_config_args(unknown)


config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

out_path = file_path_with_default_dir(args.output_file, file_manager.exp_dir)


exit_on_ctrl_c()
env_spec = params.env_spec.cls(params.env_spec.params)

assert params.dataset_train.params.get("horizon", 0)

params.dataset_train.params.file = args.file
dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

out_ls = []
all_names = env_spec.all_names + ['done', 'rollout_timestep']

for i in range(dataset_input.get_num_episodes()):
    # two empty axes for (batch_size, horizon)
    datadict = dataset_input.get_episode(i, all_names)

    assert not datadict.has_leaf_keys(args.label_key), f"Key {args.label_key} already present!"
    dones = datadict >> "done"
    if isinstance(dones, torch.Tensor):
        labels = torch.ones_like(dones[..., None])
    else:
        labels = np.ones_like(dones[..., None])
    datadict[args.label_key] = args.label_value * labels
    out_ls.append(datadict)

    if i % 10000 == 0:
        logger.debug("Loading sample %d" % i)

logger.debug("Combining all datasets")
out_datadict = AttrDict.leaf_combine_and_apply(out_ls, lambda vs: np.concatenate(vs, axis=0))

logger.debug(f"Shape of {args.label_key}: {(out_datadict >> args.label_key).shape}")

logger.debug("Saving dataset output to -> %s" % out_path)

logger.debug("Keys: " + str(list(out_datadict.leaf_keys())))
logger.debug("data len: %d" % len(out_datadict.done))
to_save = dict()
for name in out_datadict.leaf_keys():
    to_save[name] = out_datadict[name]

np.savez_compressed(out_path, **to_save)

# add any other print statements here:
