import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.file_utils import import_config
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c, ipdb_on_exception

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Base config for env spec")
parser.add_argument('--key', type=str, help="Key in dataset to plot")
parser.add_argument('--file', type=str, nargs="*", default=[], help="1 or more input files")
parser.add_argument('--save_file', type=str, default=None, help="Save plot here")
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--show', action="store_true", help="show plots")
parser.add_argument('--avg', action="store_true", help="add avg line to plots")
args = parser.parse_args()

ls = ":" if args.avg else "-"  # dotted
lw = 0.5 if args.avg else 1
lw2 = 1.0  # line width for avg

exit_on_ctrl_c()  # in case of infinite waiting
ipdb_on_exception()

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

params = import_config(config_fname)

file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

env_spec = params.env_spec.cls(params.env_spec.params)

params = AttrDict(
    file=args.file,
    output_file="/tmp/empty.npz",
    capacity=1e5,
    horizon=1,
    batch_size=10,
    use_rollout_steps=False,
    # allow_missing=True
)
dataset = NpDataset(params, env_spec, file_manager)

num_eps = len(dataset.split_indices())

assert num_eps != 0, "No data to plot"
assert all([args.ep_idxs[i] < num_eps for i in range(len(args.ep_idxs))]), "Bad idxs: %s" % args.ep_idxs

num_elements_per_key = dataset.get(args.key, np.array([0]))
num_elements_per_key = num_elements_per_key.shape[1:]
num_elements_per_key = int(np.product(num_elements_per_key))
# num_elements_per_key = np.product(env_spec.names_to_shapes[args.key])

episodes = sorted(list(set(args.ep_idxs)))
if len(episodes) == 0:
    episodes = list(range(num_eps))

fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4 * num_elements_per_key))
fig.suptitle("plots for %s" % (args.key))
if num_elements_per_key == 1:
    axes = [axes]
for i in range(num_elements_per_key):
    axes[i].set_title("axis %d" % i)
for e in episodes:

    episode_e_key = dataset.get_episode(e, names=[args.key])[args.key]
    assert episode_e_key.shape[-1] >= num_elements_per_key, "%s that was read has %d elements but expected at least %d" % (
        args.key, episode_e_key.shape[1], num_elements_per_key)

    x = np.arange(episode_e_key.shape[0])
    for i in range(num_elements_per_key):
        y = episode_e_key[..., i]
        axes[i].plot(x, y, label="ep %d" % e, linewidth=lw, linestyle=ls)

if args.avg:
    all_eps = [dataset.get_episode(e, names=[args.key])[args.key] for e in episodes]
    ep_len = min([ep_data.shape[0] for ep_data in all_eps])
    big = np.stack([ep_data[:ep_len] for ep_data in all_eps])  # (num_eps, ep_len, obsdim)
    avg = big.mean(axis=0)
    std = big.std(axis=0)
    x = np.arange(ep_len)
    for i in range(num_elements_per_key):
        axes[i].plot(x, avg[..., i], label="avg", color="black", linewidth=lw2)
        axes[i].fill_between(x, avg[..., i] - std[..., i], avg[..., i] + std[..., i], color="gray", alpha=0.4)

for i in range(num_elements_per_key):
    axes[i].legend()

if args.save_file is not None:
    logger.debug("Saving to file -> %s" % args.save_file)
    fig.savefig(args.save_file)

if args.show:
    plt.show()
