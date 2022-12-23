"""
ND action relabeling through nested K means
"""

import argparse
import os

import numpy as np
import scipy.cluster
from scipy.stats import multivariate_normal

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.input_utils import query_string_from_set
from sbrl.utils.np_utils import np_split_dataset_by_key
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c
from sbrl.utils.torch_utils import get_horizon_chunks, combine_dims_np

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--key', type=str, help="Key in dataset to cluster by")
parser.add_argument('--action_key', type=str, help="Key in dataset to compute variance over")
parser.add_argument('--out_action_key', type=str, default=None, help="Relabeled action key (None if same as default)")
parser.add_argument('--save_file', type=str, default=None, help="Save new data here")
# parser.add_argument('--plot_axes', type=int, nargs='*', default=None,
#                     help="2-3 idxs in data to correspond with x,y... one indexed. negative to scale by -1.")
parser.add_argument('--num_iters', type=int, default=1, help='If >1, will repeat the clustering and relabeling N times')

parser.add_argument('--temporal_horizon', type=int, default=None, help='If not None, will do K means on H states instead of just 1.')
parser.add_argument('--temporal_idx', type=int, default=None, help='If not None, will index dim=1 with this idx')
parser.add_argument('--done_key', type=str, default='done')
# parser.add_argument('--ep_range', type=int, nargs='*', default=None, help="episodes to run, range")  # TODO
# parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to run, [] for all. done must be specified")  # TODO
# parser.add_argument('--ep_all_together', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
# parser.add_argument('--do_3d', action="store_true", help="do 3d version. not implemented yet")
# parser.add_argument('--show', action="store_true", help="show plots")
# parser.add_argument('--scatter', action="store_true", help="scatter instead of line")
# parser.add_argument('--scale_minmax', action="store_true", help="x,y axes will be scaled to min/max.")
# parser.add_argument('--clip_x', type=int, nargs=2, default=None)
# parser.add_argument('--clip_y', type=int, nargs=2, default=None)
parser.add_argument('--num_clusters', type=int, default=50, help='kmeans of the state space')

parser.add_argument('--num_action_clusters', type=int, default=3, help='kmeans of the actions space, ignored if filter_by_variance=True')

parser.add_argument('--filter_by_variance', action='store_true', help='if true, will skip k means of action space.')
parser.add_argument('--replace_with_nearest', action='store_true', help='if true, replaces action with sample from nearest state with good action')
parser.add_argument('--fixed_distance', type=float, default=None, help='if filtering by variance, |data - mean| < this --> relabel with cluster center action')
parser.add_argument('--epsilon', type=float, default=None, help='if filtering by variance, P(data) < epsilon --> relabel with cluster center action')

parser.add_argument('--num_action_dim', type=int, default=None, help='first ND of |A| for clustering the action space')
# parser.add_argument('--num_dim', type=int, default=None, help='for plotting the action space')
# parser.add_argument('--sum_var', action='store_true', help='sum of variance')
parser.add_argument('--nd2_theta', action='store_true', help='special case, use theta instead of each dim for action clustering')
# parser.add_argument('--max_var', type=float, default=None, help='use a constant max var for cmap')
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

save_file = args.save_file
if save_file is not None:
    save_file = file_path_with_default_dir(args.save_file, FileManager.base_dir, expand_user=True)
    logger.debug("Saving File Path: %s" % save_file)
    assert os.path.exists(os.path.dirname(save_file)), f"Missing path -> {save_file}"

# DO_3D = args.do_3d
# default_plt_axes = [1, 2, 3] if DO_3D else [1, 2]
# if args.plot_axes is None:
#     args.plot_axes = default_plt_axes

# plotting_axes = [abs(i) - 1 for i in args.plot_axes]
# sgns = [int(i > 0) * 2 - 1 for i in args.plot_axes]
#
# assert len(plotting_axes) == 3 if DO_3D else len(plotting_axes) == 2, "Wrong num axes!"

key = args.key
action_key = args.action_key
out_action_key = args.out_action_key if args.out_action_key is not None else action_key
done_key = args.done_key

logger.debug("Key Name: %s, shape: %s" % (key, (data >> key).shape))
logger.debug("Action Key Name: %s, shape: %s" % (key, (data >> action_key).shape))
logger.debug("Episode Done Name: %s" % done_key)

to_save = {}
# split by key, then save each ep
done = data >> done_key
splits, data_ls, _ = np_split_dataset_by_key(data > [key, action_key], AttrDict(), done, complete=True)
# if args.ep_range is not None:
#     if len(args.ep_range) == 1:
#         args.ep_range = [0, args.ep_range[0]]
#     else:
#         assert len(args.ep_range) == 2
#     assert args.ep_range[0] < args.ep_range[1] <= len(splits), f"Ep range is invalid: {args.ep_range}"
#     ep_idxs = list(range(args.ep_range[0], args.ep_range[1]))
# else:
#     ep_idxs = list(range(len(splits))) if len(args.ep_idxs) == 0 else args.ep_idxs

# logger.debug(f"Episode indices to plot (len = {len(splits)}): {ep_idxs}")

if args.temporal_horizon is not None:
    TH = args.temporal_horizon
    assert TH > 1, "temporal horizon must be at least 2 if specified..."
    logger.debug(f"Stacking states for {key} by temporal horizon {TH}")
    for ep in range(len(splits)):
        arr = data_ls[ep] >> key
        # pad
        arr = np.concatenate([arr[:1]] * (TH - 1) + [arr], axis=0)
        # combine dims 1&2, e.g., if arr is (B,2), then (B, TH*2) will be output
        data_ls[ep][key] = combine_dims_np(get_horizon_chunks(arr, TH, 0, len(arr) - TH, dim=0, stack_dim=0), 1)

# all of them
all_data_key = np.concatenate([data_ls[ep] >> key for ep in range(len(splits))], axis=0)
action_data = np.concatenate([data_ls[ep] >> action_key for ep in range(len(splits))], axis=0)
# logger.debug("Using zero-indexed axes %s with scales %s" % (plotting_axes, sgns))

if args.temporal_idx is not None:
    all_data_key = all_data_key[:, args.temporal_idx]
    action_data = action_data[:, args.temporal_idx]
    raise NotImplementedError

logger.debug(f"Clustering shape: {all_data_key.shape}")

for i in range(args.num_iters):
    logger.info(f"------------ Iteration: {i} ------------")

    # codebook sorts on raw data (states only)
    codebook, labels = scipy.cluster.vq.kmeans2(all_data_key, args.num_clusters)

    state_data = all_data_key.reshape(-1, all_data_key.shape[-1])
    action_data = action_data.reshape(-1, action_data.shape[-1])
    ND = args.num_action_dim if args.num_action_dim is not None else action_data.shape[-1]

    action_bins = [action_data[labels == i] for i in range(args.num_clusters)]  # action array
    action_idxs = [np.flatnonzero(labels == i) for i in range(args.num_clusters)]  # action idxs in big array (action_data)
    action_lens = np.array([len(ad) for ad in action_bins])

    # remove bins with 0 mapped values
    codebook = codebook[action_lens > 0]
    action_bins = [ad for ad in action_bins if len(ad) > 0]
    action_lens = [a for a in action_lens if a > 0]
    action_idxs = [a for a in action_idxs if len(a) > 0]

    # nd = action_data.shape[-1] if args.num_dim is None else args.num_dim
    logger.debug(f"Found Centroids: #={len(action_lens)}, Min/maxbin count: {np.min(action_lens)}/{np.max(action_lens)}")

    # now copy over the relabelled actions.
    new_action_data = action_data.copy()

    assert ND == 2 or not args.nd2_theta, "ND != 2, cannot use theta variance!"

    # helpful debugging message and asserts
    assert not args.filter_by_variance or ((args.epsilon is not None) ^ (args.fixed_distance is not None)), "Exactly one can be specified!"
    if args.filter_by_variance:
        if args.epsilon is not None:
            logger.debug(f"Filtering by action likelihood under cluster gaussian (eps = {args.epsilon})")
        else:
            logger.debug(f"Filtering by action distance from cluster mean (d = {args.fixed_distance}), "
                         f"replace with {'nearest' if args.replace_with_nearest else 'mean'}")

    num_replaced = 0
    idxs_replaced = []

    def get_actions_std(bin_ac):
        bin_ac = bin_ac[:, :ND]
        if args.nd2_theta:
            bin_ac = np.rad2deg(np.arctan2(bin_ac[:, 1], bin_ac[:, 0]))[:, None]
        action_std = bin_ac.std(0)
        return bin_ac, action_std

    # nested action kmeans
    for i in range(len(action_bins)):
        bin_ac, idxs_ac, num_ac = action_bins[i], action_idxs[i], action_lens[i]
        assert np.allclose(bin_ac, action_data[idxs_ac])
        bin_ac, action_std = get_actions_std(bin_ac)

        if args.filter_by_variance:
            action_mean = bin_ac.mean(0)
            if args.epsilon is not None:
                p = multivariate_normal.pdf(bin_ac, mean=action_mean, cov=np.diag(action_std ** 2))
                to_replace = p < args.epsilon  # low likelihood
            else:
                to_replace = np.linalg.norm(bin_ac - action_mean[None], axis=-1) > args.fixed_distance  # high distance

            good_actions = action_data[idxs_ac][~to_replace]  # ~to_replace x D
            # fill in with cluster center

            if len(good_actions) > 0:
                # only replace when there is at least one "good" action.
                if args.replace_with_nearest:
                    badac_states = state_data[idxs_ac][to_replace]
                    good_states = state_data[idxs_ac][~to_replace]
                    # to_replace x ~to_replace -> idxs (to_replace,)
                    nearest = np.argmin(np.linalg.norm(good_states[None, :] - badac_states[:, None], axis=-1), axis=-1)
                    closest_ac = np.take_along_axis(good_actions, nearest[:, None], axis=0)
                    new_action_data[idxs_ac[to_replace]] = closest_ac
                else:
                    new_action_data[idxs_ac[to_replace]] = good_actions.mean(0)[None]

                # keep track
                num_replaced += np.count_nonzero(to_replace)
                idxs_replaced.append(idxs_ac[to_replace])

        else:
            if args.nd2_theta:
                raise NotImplementedError('dims is wrong')
            dims = np.flatnonzero(action_std > 1e-11)  # check for nonzero std (messes up kmeans)
            if len(dims) > 0 and len(bin_ac) > args.num_action_clusters:
                action_codebook, action_labels = scipy.cluster.vq.kmeans2(bin_ac[:, dims], args.num_action_clusters, minit='points')
                this_action_data_copy = new_action_data[idxs_ac]
                this_action_data_copy[:, dims] = action_codebook[action_labels]  # go through bins, copying over the cluster center.
                new_action_data[idxs_ac] = this_action_data_copy  # copy back
                num_replaced += len(idxs_ac)
                idxs_replaced.append(idxs_ac)

        bin_ac_new, action_std_new = get_actions_std(new_action_data[idxs_ac])
        logger.debug(f"Before: {action_std.sum()}, after: {action_std_new.sum()}")

    all_replaced = np.concatenate(idxs_replaced)
    assert len(np.unique(all_replaced)) == num_replaced, "Replace was nonunique!!"
    logger.debug(f"Replaced {num_replaced} / {len(action_data)} actions")

    # override for next iteration
    action_data = new_action_data.copy()

# save output actions
if data.has_leaf_key(out_action_key):
    logger.warn(f"Overriding key {out_action_key} with new data!")
data[out_action_key] = new_action_data

# save
if save_file is not None:
    data.leaf_shapes().pprint()
    exists = os.path.exists(save_file)
    do_save = query_string_from_set(f'Save to {save_file}, exists = {exists}? (y/n)', ['y', 'n']) == 'y'

    if do_save:
        dc = {}
        for name in data.leaf_keys():
            dc[name] = data[name]
        logger.debug('Saving...')
        np.savez_compressed(save_file, **dc)
        logger.debug('Saved.')

# assert nd == 2 or not args.nd2_theta, "ND != 2, cannot use theta variance!"
# if args.nd2_theta or args.sum_var:
#     nd = 1
#
# # fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4*num_elements_per_key))
# fig = plt.figure(figsize=(nd * 5, 6), tight_layout=True)
# gs = fig.add_gridspec(6, nd)
# axes = [[fig.add_subplot(gs[:5, i], projection='3d' if DO_3D else None), fig.add_subplot(gs[5, i])] for i in range(nd)]
# # axes = fig.subplots(1, nd, subplot_kw={'projection': '3d'} if DO_3D else None)
# fig.suptitle(f"Var in action = {action_key}, given state = {key}" + (f" (h_idx={args.temporal_idx})" if args.temporal_idx is not None else ""))
#
# if args.nd2_theta:
#     action_variance_per_bin = [np.rad2deg(np.arctan2(ad[:, 1], ad[:, 0])).std(0) for ad in action_bins]
# elif args.sum_var:
#     action_variance_per_bin = [ad.std(0).mean() for ad in action_bins]
# else:
#     action_variance_per_bin = [ad.std(0) for ad in action_bins]
#
# cmap = mpl.cm.cool
#
# for dim in range(nd):
#     if args.nd2_theta or args.sum_var:
#         dim_var = action_variance_per_bin
#     else:
#         dim_var = [av[dim] for av in action_variance_per_bin]
#
#     ax_top = axes[dim][0]
#     ax_bot = axes[dim][1]
#
#     min_var = 0 if args.max_var is not None else min(dim_var)
#     max_var = args.max_var if args.max_var is not None else max(dim_var)
#
#     colors = [cmap((av - min_var) / (max_var - min_var)) for av in dim_var]
#
#     codebook_axes = np.split(codebook[:, plotting_axes], len(plotting_axes), axis=-1)
#     ax_top.scatter(*codebook_axes, c=colors)
#
#     suff = f'dims {list(plotting_axes)}' if args.sum_var else f"{dim}"
#     label = "Var[Theta]" if args.nd2_theta else f'Var[{suff}]'
#     norm = mpl.colors.Normalize(vmin=min_var, vmax=max_var)
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                  cax=ax_bot, orientation='horizontal', label=label)
#
#     if not DO_3D:
#         ax_top.set_aspect('equal')
#     else:
#         # manual
#         plt_utils.equal_aspect_3d(ax_top, *codebook_axes, sgns=sgns)
#         # First remove fill
#         ax_top.xaxis.pane.fill = False
#         ax_top.yaxis.pane.fill = False
#         ax_top.zaxis.pane.fill = False
#         plt.rcParams['grid.color'] = "0.93"
#
#         # Now set color to white (or whatever is "invisible")
#         ax_top.xaxis.pane.set_edgecolor('w')
#         ax_top.yaxis.pane.set_edgecolor('w')
#         ax_top.zaxis.pane.set_edgecolor('w')
#
#     ax_top.set_xlabel("x")
#     ax_top.set_ylabel("y")
#     if DO_3D:
#         ax_top.set_zlabel("z")
#
#     ax_bot.set_aspect(0.1)

# xmin = min([ep[:, args.temporal_idx, ..., xa].min() for ep in to_save.values()])
# ymin = min([ep[:, args.temporal_idx, ..., ya].min() for ep in to_save.values()])
# xmax = max([ep[:, args.temporal_idx, ..., xa].max() for ep in to_save.values()])
# ymax = max([ep[:, args.temporal_idx, ..., ya].max() for ep in to_save.values()])
#
# if args.clip_x is not None:
#     xmin, xmax = args.clip_x
#
# if args.clip_y is not None:
#     ymin, ymax = args.clip_y
#
# logger.debug(f"X ({xmin}, {xmax}) - Y ({ymin}, {ymax}) --- clip_xy = ({args.clip_x is not None}, {args.clip_y is not None})")

# if args.scale_minmax:
#     ax.set_xlim([xmin, xmax][::sgns[0]])
#     ax.set_ylim([ymin, ymax][::sgns[1]])
# else:
#     all_max = np.array([xmax, ymax])
#     all_min = np.array([xmin, ymin])
#     max_range = (all_max - all_min).max()
#     center = 0.5 * (all_max + all_min)
#     e_max = center + max_range / 2
#     e_min = center - max_range / 2
#     ax.set_xlim([e_min[0], e_max[0]][::sgns[0]])
#     ax.set_ylim([e_min[1], e_max[1]][::sgns[1]])



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

# if args.show:
#     plt.show()
