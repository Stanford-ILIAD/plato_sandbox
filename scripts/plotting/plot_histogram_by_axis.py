import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils.file_utils import import_config
from sbrl.utils.input_utils import get_str_from, query_string_from_set
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Base config for env spec")
parser.add_argument('--key', type=str, required=True, help="Key in dataset to plot")
parser.add_argument('--bins', type=int, required=True, help="number of bins per histogram / axis")
parser.add_argument('--clip_low', type=float, default=-np.inf, help="min value")
parser.add_argument('--clip_high', type=float, default=np.inf, help="max value")
parser.add_argument('--file', type=str, nargs="*", default=[], help="1 or more input files")
parser.add_argument('--save_file', type=str, default=None, help="Save plot here")
# parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--show', action="store_true", help="show plots")
# parser.add_argument('--avg', action="store_true", help="add avg line to plots")
args = parser.parse_args()

# ls = ":" if args.avg else "-"  # dotted
# lw = 0.5 if args.avg else 1
# lw2 = 1.0  # line width for avg

exit_on_ctrl_c()  # in case of infinite waiting

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

params = import_config(config_fname)

file_manager = FileManager()

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

raw_data = dataset.get_datadict()

assert raw_data.has_leaf_key(args.key), get_str_from([args.key, raw_data.list_leaf_keys()])
assert args.bins > 0

our_data = raw_data[args.key]

num_elements_per_key = int(np.product(our_data.shape[1:]))
our_data = our_data.reshape([-1, num_elements_per_key])

if num_elements_per_key > 10:
    ret = query_string_from_set("Num elements for %s is %d. Continue? [y/n]", ["y", "n"])
    if ret == "n":
        sys.exit(1)

fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4 * num_elements_per_key))
fig.suptitle("plots for %s" % (args.key))

if num_elements_per_key == 1:
    axes = [axes]

for i in range(num_elements_per_key):
    axes[i].set_title("axis %d" % i)
    oob = np.logical_or(our_data[:, i] < args.clip_low, our_data[:, i] > args.clip_high)
    oob = np.logical_or(oob, np.isnan(our_data[:, i]))
    final = our_data[~oob, i]
    # print(final)
    axes[i].hist(final, args.bins)


if args.save_file is not None:
    logger.debug("Saving to file -> %s" % args.save_file)
    fig.savefig(args.save_file)

if args.show:
    plt.show()
