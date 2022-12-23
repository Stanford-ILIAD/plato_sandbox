import argparse

import matplotlib.pyplot as plt
import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils import plt_utils as pu
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.np_utils import np_split_dataset_by_key
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--key', type=str, help="Key in dataset to animate")
parser.add_argument('--save_file', type=str, default=None, help="Save animation here")
parser.add_argument('--xy_axes', type=int, nargs=2, default=[1, 2],
                    help="2 idxs in data to correspond with x,y. one indexed. negative to scale by -1.")
parser.add_argument('--temporal_idx', type=int, default=None, help='If not None, will index dim=1 with this idx')
parser.add_argument('--done_key', type=str, default=None)  # if None, all key points will be selected.
parser.add_argument('--ep_range', type=int, nargs='*', default=None, help="episodes to run, range")  # TODO
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to run, [] for all. done must be specified")  # TODO
# parser.add_argument('--ep_all_together', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--show', action="store_true", help="show plots")
parser.add_argument('--scatter', action="store_true", help="scatter instead of line")
parser.add_argument('--scale_minmax', action="store_true", help="x,y axes will be scaled to min/max.")
parser.add_argument('--clip_x', type=int, nargs=2, default=None)
parser.add_argument('--clip_y', type=int, nargs=2, default=None)
args = parser.parse_args()

exit_on_ctrl_c()  # in case of infinite waiting

assert len(args.file) >= 1

data = None

for f in args.file:
    path = file_path_with_default_dir(f, FileManager.base_dir, expand_user=True)
    logger.debug("File Path: %s" % path)
    new_data = AttrDict.from_dict(dict(np.load(path, allow_pickle=True)))
    if data is None:
        data = new_data
    else:
        common_keys = set(data.list_leaf_keys()).intersection(data.list_leaf_keys())
        data = AttrDict.leaf_combine_and_apply([data > common_keys, new_data > common_keys], lambda vs: np.concatenate(vs, axis=0))

logger.debug(list(data.leaf_keys()))

key = args.key
done_key = args.done_key
logger.debug("Key Name: %s, shape: %s" % (key, (data >> key).shape))
if done_key is not None:
    logger.debug("Episode Done Name: %s" % done_key)

to_save = {}
if done_key is not None:
    # split by key, then save each ep
    done = data >> done_key
    splits, data_ls, _ = np_split_dataset_by_key(data > [key], AttrDict(), done, complete=True)
    if args.ep_range is not None:
        if len(args.ep_range) == 1:
            args.ep_range = [0, args.ep_range[0]]
        else:
            assert len(args.ep_range) == 2
        assert args.ep_range[0] < args.ep_range[1] <= len(splits), f"Ep range is invalid: {args.ep_range}"
        ep_idxs = list(range(args.ep_range[0], args.ep_range[1]))
    else:
        ep_idxs = list(range(len(splits))) if len(args.ep_idxs) == 0 else args.ep_idxs

    # logger.debug(f"Episode indices to plot (len = {len(splits)}): {ep_idxs}")

    for ep in ep_idxs:
        to_save[f"_{ep}"] = data_ls[ep] >> key

else:
    # all of them
    to_save[''] = data >> key

# fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4*num_elements_per_key))
fig = plt.figure(figsize=(7, 6), tight_layout=True)
ax = fig.add_subplot(111)
fig.suptitle(f"Trajectory for {key}" + (f" (h_idx={args.temporal_idx})" if args.temporal_idx is not None else ""))

xy = [abs(i) - 1 for i in args.xy_axes]
xa, ya = xy
sgns = [int(i > 0) * 2 - 1 for i in args.xy_axes]
logger.debug("Using zero-indexed axes %s with scales %s" % (xy, sgns))

num_eps = len(list(to_save.keys()))
cmap = plt.cm.get_cmap("hsv", int(num_eps * 1.05) + 1)  # avoid returning back to initial color
if num_eps > 5:
    colors = [cmap(i) for i in range(num_eps)]
else:
    colors = [pu.orange, pu.teal, pu.purple, pu.grey, pu.green][:num_eps]  # specific

key_order = sorted(to_save.keys())

i = 0
ax.set_prop_cycle(color=colors)
for suffix, c in zip(key_order, colors):
    lines = to_save[suffix]
    assert lines.shape[-1] >= 2, lines.shape

    label = f"ep{suffix}"
    if args.temporal_idx is None:
        x = lines[..., xa]
        y = lines[..., ya]
    else:
        x = lines[:, args.temporal_idx, ..., xa]
        y = lines[:, args.temporal_idx, ..., ya]

    if args.scatter:
        ax.scatter(x, y, label=label)
    else:
        ax.plot(x, y, label=label)
    # ax.text2D(0.01, 0.93 - i*0.05, label, color=c, transform=ax.transAxes, fontweight='bold')
    i += 1

xmin = min([ep[:, args.temporal_idx, ..., xa].min() for ep in to_save.values()])
ymin = min([ep[:, args.temporal_idx, ..., ya].min() for ep in to_save.values()])
xmax = max([ep[:, args.temporal_idx, ..., xa].max() for ep in to_save.values()])
ymax = max([ep[:, args.temporal_idx, ..., ya].max() for ep in to_save.values()])

if args.clip_x is not None:
    xmin, xmax = args.clip_x

if args.clip_y is not None:
    ymin, ymax = args.clip_y

logger.debug(f"X ({xmin}, {xmax}) - Y ({ymin}, {ymax}) --- clip_xy = ({args.clip_x is not None}, {args.clip_y is not None})")

if args.scale_minmax:
    ax.set_xlim([xmin, xmax][::sgns[0]])
    ax.set_ylim([ymin, ymax][::sgns[1]])
else:
    all_max = np.array([xmax, ymax])
    all_min = np.array([xmin, ymin])
    max_range = (all_max - all_min).max()
    center = 0.5 * (all_max + all_min)
    e_max = center + max_range / 2
    e_min = center - max_range / 2
    ax.set_xlim([e_min[0], e_max[0]][::sgns[0]])
    ax.set_ylim([e_min[1], e_max[1]][::sgns[1]])


ax.set_xlabel("x")
ax.set_ylabel("y")


# num_elements_per_key = np.product(num_elements_per_key)
# # num_elements_per_key = np.product(env_spec.names_to_shapes[args.key])
# assert num_elements_per_key >= 3, "There must be at least 3 axes to visualize (was %d)" % num_elements_per_key
# assert all([i != 0 for i in args.xyz_axes]), "Must be one indexed: %s" % args.xyz_axes
# xyz = [abs(i) - 1 for i in args.xyz_axes]
# xa, ya, za = xyz
# sgns = [int(i > 0) * 2 - 1 for i in args.xyz_axes]
# logger.debug("Using zero-indexed axes %s with scales %s" % (xyz, sgns))
#
# assert num_eps != 0, "No data to plot"
# assert all([args.ep_idxs[i] < num_eps for i in range(len(args.ep_idxs))]), "Bad idxs: %s" % args.ep_idxs
#
# episodes = sorted(list(set(args.ep_idxs)))
# if len(episodes) == 0:
#     episodes = list(range(num_eps))

if args.show:
    plt.show()
