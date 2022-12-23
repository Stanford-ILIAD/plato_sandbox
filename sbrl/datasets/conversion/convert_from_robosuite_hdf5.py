"""
This script takes robosuite configs
"""
import os
from argparse import ArgumentParser
from typing import List

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.robosuite.robosuite_env import get_rs_example_spec_params
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict as d
from sbrl.utils.python_utils import get_with_default


def get_hdf5_leaf_names(node, name=''):
    if isinstance(node, h5py.Dataset):
        return [name]
    else:
        names = []
        for child_name, child in node.items():
            names += get_hdf5_leaf_names(child, name=name + '/' + child_name)
        return names


def parse_hdf5(key, value):
    new_value = np.array(value)
    #     if type(new_value[0]) == np.bytes_:
    #         new_value = np_utils.bytes2im(new_value)
    if new_value.dtype == np.float64:
        new_value = new_value.astype(np.float32)
    if len(new_value.shape) == 1:
        new_value = new_value[:, np.newaxis]
    return new_value


def from_episodic_hdf5(node, episodic_names, flat_names=None, prefix="/data/", ep_prefix="demo_", get_obs_extra=True,
                       get_ac_extra=True, integer_sort_episodes=True, mask=None):
    """
    Will get episodic_names, stacked, and flat_names, not stacked, from the given hdf5 node.

    :param node: the hdf5 node
    :param episodic_names: names that will be looked for under prefix+ep_prefix in node
    :param flat_names: names that are outside and do not need to be modified (shape is already correct in hdf5)
    :param prefix: the hdf5 prefix. must start with '/'. we only get keys under this prefix.
    :param ep_prefix: the episode prefix. the integer episode number will follow this prefix.
    :param get_obs_extra: if True, will also return 'done' and 'rollout_timestep' keys.
    :param get_ac_extra: if True, will also return 'policy_type', 'policy_name', and 'policy_switch' keys.
    :param integer_sort_episodes: if True, will sort episodes as integers, else by string
    """

    assert prefix.startswith("/"), "Hdf5 format starts with /, but that's not compatible, pls remove"
    assert len(episodic_names) > 0, "Provide some names!"

    all_hdf5_names = get_hdf5_leaf_names(node)
    depref_hdf5_names = [n[len(prefix):] for n in all_hdf5_names if n.startswith(prefix)]

    if flat_names is None:
        # if not specified, flat_names will be any name starting with prefix, but outside of the prefix+ep_prefix
        flat_names = [n for n in depref_hdf5_names if not n.startswith(ep_prefix)]

    assert len(set(flat_names).intersection(episodic_names)) == 0, [flat_names, episodic_names]

    data = d()
    for n in flat_names:
        hd_name = prefix + n
        data[n] = parse_hdf5(hd_name, node[hd_name])

    # placeholder
    zero_d = d.from_kvs(depref_hdf5_names, [0] * len(depref_hdf5_names))
    if integer_sort_episodes:
        sorted_names = sorted([n for n in zero_d.keys() if n.startswith(ep_prefix)],
                              key=lambda k: int(k[len(ep_prefix):]))
    else:
        sorted_names = sorted([n for n in zero_d.keys() if n.startswith(ep_prefix)])

    if mask is not None:
        assert isinstance(mask, List)
        allowed_keys = []
        for m in list(mask):
            allowed_keys += [elem.decode("utf-8") for elem in np.array(node["mask/{}".format(m)][:])]
        # print(len(sorted_names) - len(allowed_keys))
        sorted_names = [s for s in sorted_names if s in allowed_keys]

    # print("SORTED N AMES:", sorted_names)
    assert len(sorted_names) > 0, f"No names found to match ep prefix!: {ep_prefix}" \
                                  + (f", with masks: {mask}" if mask is not None else "")

    all_episode_keys = (zero_d >> sorted_names[0]).list_leaf_keys()
    print(all_episode_keys)
    optional_keys = []
    if get_obs_extra:
        optional_keys.extend(['done', 'rollout_timestep'])
    if get_ac_extra:
        optional_keys.extend(['policy_type', 'policy_name', 'policy_switch'])

    nonspecified_required_keys = list(set(optional_keys).difference(all_episode_keys))  # nonspecified keys

    assert set(episodic_names).issubset(all_episode_keys + optional_keys), list(
        set(episodic_names).difference(all_episode_keys + optional_keys))

    all_episodes_filtered = []

    for n in sorted_names:
        this_ep = d()
        # for all the keys that we need to get AND can actually get...
        for key in set(episodic_names).difference(nonspecified_required_keys):
            hd_name = os.path.join(prefix + n, key)
            this_ep[key] = parse_hdf5(hd_name, node[hd_name])

        # fill in the others as needed
        ep_len = len(this_ep[episodic_names[0]])
        if get_obs_extra:
            this_ep['done'] = np.array([False] * ep_len, dtype=np.bool)
            this_ep['done'][-1] = True
            this_ep['rollout_timestep'] = np.arange(ep_len)
        if get_ac_extra:
            this_ep['policy_type'] = get_with_default(this_ep, 'policy_type', np.array([-1] * ep_len)[:, None])
            this_ep['policy_name'] = get_with_default(this_ep, 'policy_name',
                                                      np.array(["unknown" for _ in range(ep_len)], dtype=np.object)[:,
                                                      None])
            this_ep['policy_switch'] = get_with_default(this_ep, 'policy_switch',
                                                        np.array([False] * ep_len, dtype=np.bool)[:, None])

        all_episodes_filtered.append(this_ep)

    return data & d.leaf_combine_and_apply(all_episodes_filtered, lambda vs: np.concatenate(vs, axis=0))


def from_robosuite_hdf5(node, obs_names, ac_names, out_obs_names, flat_names=None, prefix="/data/", ep_prefix="demo_",
                        get_obs_extra=True, get_ac_extra=True, integer_sort_episodes=True, mask=None, img_key='obs/agentview_image'):
    """
    loads episodic obs / ac, but maps some weird naming conventions back.
    """
    mapped_obs_names = [img_key if n == 'image' else 'obs/' + n for n in obs_names]
    mapped_ac_names = ac_names.copy()
    if "action" in mapped_ac_names:
        idx = mapped_ac_names.index("action")
        mapped_ac_names[idx] = "actions"  # stupid
    mapped_out_obs_names = ["rewards" if n == "reward" else 'next_obs/' + n for n in out_obs_names]

    episodic_names = mapped_obs_names + mapped_ac_names + mapped_out_obs_names

    raw_data = from_episodic_hdf5(node, episodic_names, flat_names=flat_names, prefix=prefix, ep_prefix=ep_prefix,
                                  get_obs_extra=get_obs_extra, get_ac_extra=get_ac_extra,
                                  integer_sort_episodes=integer_sort_episodes, mask=mask)

    data = raw_data.leaf_copy()

    # mapping back
    for mo, o in zip(mapped_obs_names + mapped_ac_names + mapped_out_obs_names, obs_names + ac_names + out_obs_names):
        data[o] = raw_data[mo]

    data.leaf_shapes().pprint()

    return data



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help='hdf5 file to load from')
    parser.add_argument('env_name', type=str, help='what type of environment is this? [NutAssemblySquare, ToolHang] supported so far')
    parser.add_argument('--save_file', type=str, default=None, help='Optional npz file to output to')
    parser.add_argument('--mask', type=str, nargs="*", default=None, help='Optional episode mask(s) (will look under mask/{}) by default')
    parser.add_argument('--add_ee_ori', action='store_true')
    parser.add_argument('--imgs', action='store_true')
    args = parser.parse_args()

    # loading robosuite data
    file = args.file  # "data/robosuite/human_square_low_dim.hdf5", for example
    env_name = args.env_name  # "NutAssemblySquare", for example
    # file = "human_tool_hang_low_dim.hdf5"

    es_prms = get_rs_example_spec_params(env_name, raw=True)
    if args.imgs:
        es_prms.observation_names.append('image')
    env_spec = ParamEnvSpec(es_prms)

    img_key = 'obs/sideview_image' if env_name == 'ToolHang' else 'obs/agentview_image'

    if args.mask is not None:
        logger.debug(f"Mask keys: {[f'mask/{m}' for m in args.mask]}")

    with h5py.File(file, 'r') as node:
        # will load
        data = from_robosuite_hdf5(node, env_spec.observation_names, env_spec.action_names,
                                   env_spec.output_observation_names, mask=args.mask, img_key=img_key)

        data = (data > (env_spec.all_names + ['done', 'rollout_timestep']))

    if args.add_ee_ori:
        new_data = d()
        for key in data.leaf_keys():
            new_data[key] = data[key]
            if "_eef_quat" in key:
                # euler
                new_data[key.replace("_eef_quat", "_eef_eul")] = Rotation.from_quat(data[key]).as_euler("xyz")
        data = new_data

    logger.debug(f"Done loading. Dataset length = {len(data['done'])}")

    if args.save_file is None:
        data.leaf_shapes().pprint()
    else:
        logger.warn(f"Saving to --> {args.save_file}")
        np.savez_compressed(args.save_file, **data.as_dict())