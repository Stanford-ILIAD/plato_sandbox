import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.file_utils import import_config
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Base config for env spec")
parser.add_argument('--key', type=str, help="Key in dataset to animate")
parser.add_argument('--file', type=str, nargs="*", default=[], help="1 or more input files")
parser.add_argument('--save_file', type=str, default=None, help="Save animation here")
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--xyz_axes', type=int, nargs=3, default=[1, 2, 3],
                    help="3 idxs in data to correspond with x,y,z. one indexed. negative to scale by -1.")
# parser.add_argument('--ep_all_together', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--show', action="store_true", help="show plots")
args = parser.parse_args()

exit_on_ctrl_c()  # in case of infinite waiting

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
    allow_missing = True
)
dataset = NpDataset(params, env_spec, file_manager)

num_eps = len(dataset.split_indices())

num_elements_per_key = dataset.get_episode(0, [args.key])[args.key].shape[1:]
num_elements_per_key = np.product(num_elements_per_key)
# num_elements_per_key = np.product(env_spec.names_to_shapes[args.key])
assert num_elements_per_key >= 3, "There must be at least 3 axes to visualize (was %d)" % num_elements_per_key
assert all([i != 0 for i in args.xyz_axes]), "Must be one indexed: %s" % args.xyz_axes
xyz = [abs(i) - 1 for i in args.xyz_axes]
xa, ya, za = xyz
sgns = [int(i > 0) * 2 - 1 for i in args.xyz_axes]
logger.debug("Using zero-indexed axes %s with scales %s" % (xyz, sgns))

assert num_eps != 0, "No data to plot"
assert all([args.ep_idxs[i] < num_eps for i in range(len(args.ep_idxs))]), "Bad idxs: %s" % args.ep_idxs

episodes = sorted(list(set(args.ep_idxs)))
if len(episodes) == 0:
    episodes = list(range(num_eps))

# fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4*num_elements_per_key))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10., azim=10.)
fig.suptitle("plots for %s, time=0.0 s" % (args.key))

cmap = plt.cm.get_cmap("hsv", len(episodes))
colors = [cmap(i) for i in range(len(episodes))]

animation_len = min([dataset.episode_length(e) for e in episodes])
ep_data = [dataset.get_episode(e, names=[args.key])[args.key] for e in episodes]  # list of AttrDict
xmin = max([ep[:, xa].min() for ep in ep_data])
ymin = max([ep[:, ya].min() for ep in ep_data])
zmin = max([ep[:, za].min() for ep in ep_data])
xmax = max([ep[:, xa].max() for ep in ep_data])
ymax = max([ep[:, ya].max() for ep in ep_data])
zmax = max([ep[:, za].max() for ep in ep_data])
xstarts = [ep[0, xa] for ep in ep_data]
ystarts = [ep[0, ya] for ep in ep_data]
zstarts = [ep[0, za] for ep in ep_data]
ax.set_xlim([xmin, xmax][::sgns[0]])
ax.set_ylim([ymin, ymax][::sgns[1]])
ax.set_zlim([zmin, zmax][::sgns[2]])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

quivs = ax.quiver(np.zeros(len(episodes)), np.zeros(len(episodes)), np.zeros(len(episodes)), np.zeros(len(episodes)), np.zeros(len(episodes)), np.zeros(len(episodes)),
                  colors=colors, normalize=False)

scat = ax.scatter(xstarts, ystarts, zstarts, c=colors)

ax.set_prop_cycle(color=colors)
lines = []
lines_xdata = []
lines_ydata = []
lines_zdata = []
for i in range(len(episodes)):
    lines_xdata.append([xstarts[i]])
    lines_ydata.append([ystarts[i]])
    lines_zdata.append([zstarts[i]])
    lines.append(ax.plot([xstarts[i]], [ystarts[i]], [zstarts[i]], linewidth=0.4, linestyle=':')[0])

# Demo 2: color
for i in range(len(episodes)):
    ax.text2D(i*0.1, 0, "ep_%d" % episodes[i], color=colors[i], transform=ax.transAxes)

def animate(frame):
    segs = []
    for i in range(len(episodes)):
        e = episodes[i]
        episode_e_key = ep_data[i]
        arr = episode_e_key[frame]  # (N-D vec)
        u = arr[xa]
        v = arr[ya]
        w = arr[za]
        xmid = xstarts[i]  # (xmax + xmin) / 2
        ymid = ystarts[i]  # (ymax + ymin) / 2
        zmid = zstarts[i]  # (zmax + zmin) / 2
        lines_xdata[i].append(u)
        lines_ydata[i].append(v)
        lines_zdata[i].append(w)
        segs.append([[xmid, ymid, zmid], [u, v, w]])
        lines[i].set_data(lines_xdata[i], lines_ydata[i])
        lines[i].set_3d_properties(lines_zdata[i])
    quivs.set_segments(segs)
    fig.suptitle("plots for %s, time=%.1f s" % (args.key, 0.1 * frame))
    return tuple([quivs] + lines)

ani = FuncAnimation(fig, animate, frames=animation_len, interval=100, blit=False, repeat=False)

# for e in episodes:
#     episode_e_key = dataset.get_episode(e, names=[args.key])[args.key]
#     assert episode_e_key.shape[
#                1] >= num_elements_per_key, "%s that was read has %d elements but expected at least %d" % (
#     args.key, episode_e_key.shape[1], num_elements_per_key)
#
#     x = np.arange(episode_e_key.shape[0])
#     for i in range(num_elements_per_key):
#         y = episode_e_key[:, i]
#
if args.save_file is not None:
    logger.debug("Saving animation to file -> %s" % args.save_file)
    ani.save(args.save_file)

if args.show:
    plt.show()
