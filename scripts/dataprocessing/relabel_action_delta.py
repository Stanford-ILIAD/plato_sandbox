"""
(optional) merge.py + relabeling of an action key

Merge multiple datasets, at the key level, then action becomes delta (difference, or pose_difference)

Specifically, if pose_diff:
- action = pose_diff((obs.pos, obs.ori_eul), action)
"""

import argparse
import os

import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils import transform_utils as T
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.input_utils import query_string_from_set
from sbrl.utils.python_utils import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--output_file', type=str, required=True)

# either specify pose name or pos & ori (eul or quat)
parser.add_argument('--action_name', type=str, required=True)
parser.add_argument('--base_name', type=str, default=None)
parser.add_argument('--pos_name', type=str, default=None)
parser.add_argument('--ori_quat_name', type=str, default=None)
parser.add_argument('--ori_eul_name', type=str, default=None)

parser.add_argument('--use_first_keys', action='store_true', help="uses the first dataset keys")
parser.add_argument('--pose_diff', action='store_true',
                    help="computes delta on first 6 elements in action. else uses \'minus\'")
parser.add_argument('--skip_keys', type=str, nargs="*", help="which keys to ignore in all datasets", default=[])

args = parser.parse_args()


if args.pose_diff:
    # argument specification for pose.
    if args.base_name is not None:
        assert args.pos_name is None and args.ori_quat_name is None and args.ori_eul_name is None
    else:
        assert args.pos_name is not None
        assert (args.ori_quat_name is None) ^ (args.ori_eul_name is None)
else:
    assert args.base_name is not None

file_manager = FileManager()

out_path = file_path_with_default_dir(args.output_file, file_manager.data_dir)

for infile in args.file:
    assert os.path.isfile(infile)

""" MERGE """

input_data = []
input_data_keys = None
for inp_file in args.file:
    data = np.load(inp_file, allow_pickle=True)
    keys = list(data.keys())

    if len(args.skip_keys) > 0:
        # removing some keys (if present)
        keys = list(set(keys).difference(args.skip_keys))

    if input_data_keys is None:
        input_data_keys = keys

    if args.use_first_keys:
        keys = list(set(input_data_keys).intersection(keys))

    assert set(input_data_keys) == set(keys), f"Key difference for {inp_file}: {set(input_data_keys).symmetric_difference(keys)}"
    # faster loading pattern
    dd = AttrDict()
    for k in keys:
        dd[k] = data[k]
    input_data.append(dd)

assert input_data_keys is not None

logger.info(f"Keys to merge: {input_data_keys}")

out_datadict = AttrDict.leaf_combine_and_apply(input_data, lambda vs: np.concatenate(vs))

""" EXTRACT BASE & ACTION and COMPUTE DIFF """

action = out_datadict >> args.action_name
logger.debug(f"Action shape: {action.shape}")

if args.pose_diff:
    logger.debug("Getting pose...")
    if args.base_name is not None:
        pose = out_datadict >> args.base_name
        pos = pose[..., :3]
        ori = pose[..., 3:6]
        ori_q = T.fast_euler2quat_ext(ori)
    else:
        pos = out_datadict >> args.pos_name
        if args.ori_quat_name is not None:
            ori_q = out_datadict >> args.ori_quat_name
        else:
            ori_q = T.fast_euler2quat_ext(out_datadict >> args.ori_eul_name)

    act_q = T.fast_euler2quat_ext(action[..., 3:6])

    p_diff = action[..., :3] - pos
    q_diff = T.quat_difference(act_q, ori_q)
    eul_diff = T.fast_quat2euler_ext(q_diff)

    # concatenate pos, ori_eul, and any other action dimensions
    delta_action = np.concatenate([p_diff, eul_diff, action[..., 6:]], axis=-1)
else:
    base = out_datadict >> args.base_name
    delta_action = action - base

assert list(delta_action.shape) == list(action.shape), [action.shape, delta_action.shape]

# assign delta action in place of action
out_datadict[args.action_name] = delta_action
out_datadict[f"old/{args.action_name}"] = action

logger.debug("Saving dataset output to -> %s" % out_path)

logger.debug("new shapes:\n%s" % out_datadict.leaf_shapes().pprint(ret_string=True))

if query_string_from_set("Save ok? [y, n]", ['y', 'n']) == 'y':
    to_save = dict()
    for name in out_datadict.leaf_keys():
        to_save[name] = out_datadict[name]

    np.savez_compressed(out_path, **to_save)
