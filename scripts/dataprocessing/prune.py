"""
Reduce the number of episodes in a dataset
"""

import argparse
import os

import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.input_utils import query_string_from_set
from sbrl.utils.np_utils import np_split_dataset_by_key
from sbrl.utils.python_utils import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help="A file to prune")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--start_offset_ep', type=int, default=0)
parser.add_argument('--max_steps', type=int, required=True)
parser.add_argument('--in_order', action='store_true', help="Add episodes in the order they appear")
parser.add_argument('--done_key', type=str, default='done', help='Used for segmenting into episodes')
parser.add_argument('--allow_onetime', action='store_true', help="Enables checking of onetime keys. Otherwise, all keys should be the same length")
parser.add_argument('--skip_ask', action='store_true', help="Enables checking of onetime keys. Otherwise, all keys should be the same length")

args = parser.parse_args()

file_manager = FileManager()

out_path = file_path_with_default_dir(args.output_file, file_manager.data_dir)

assert os.path.isfile(args.file)

assert args.max_steps > 0

dd = np.load(args.file, allow_pickle=True, mmap_mode='r')
logger.debug("Loaded.")
data = AttrDict.from_dict(dd)
keys = list(data.keys())
dk = args.done_key
assert dk in keys, f"{dk} not in {keys}"
ep_len = len(data >> dk)

onetime_data = AttrDict()
if args.allow_onetime:
    onetime_data, data = data.leaf_partition(lambda k, v: len(v) < ep_len)
    if not onetime_data.is_empty():
        logger.debug(f"Onetime keys:\n{onetime_data.leaf_shapes().pprint(ret_string=True)}")

splits, data_ep, onetime_data_ep = np_split_dataset_by_key(data, onetime_data, data >> dk)

logger.info(f"Num episodes: {len(data_ep)}")
logger.info(f"Num steps: {len(data[dk])}")

assert args.start_offset_ep < len(data_ep)

if args.in_order:
    data_ep_order = np.arange(args.start_offset_ep, len(data_ep))
else:
    data_ep_order = args.start_offset_ep + np.random.choice(len(data_ep) - args.start_offset_ep, len(data_ep) - args.start_offset_ep, replace=False)

count = 0
to_keep = []
to_keep_onetime = []
for ep_idx in data_ep_order:
    ep_len = len(data_ep[ep_idx] >> dk)
    if count + ep_len > args.max_steps:
        break

    count += ep_len
    to_keep.append(data_ep[ep_idx])
    to_keep_onetime.append(onetime_data_ep[ep_idx])

logger.debug(f"Keeping {len(to_keep)} episodes with length: {count}")
out_datadict = AttrDict.leaf_combine_and_apply(to_keep, np.concatenate)
if args.allow_onetime:
    out_datadict.combine(AttrDict.leaf_combine_and_apply(to_keep_onetime, np.concatenate))

ep_lens = np.array([len(ep >> dk) for ep in to_keep])

logger.debug("Saving dataset output to -> %s" % out_path)

logger.debug("new shapes:\n%s" % out_datadict.leaf_shapes().pprint(ret_string=True))

if args.skip_ask or query_string_from_set("Save ok? [y, n]", ['y', 'n']) == 'y':
    to_save = dict()
    for name in out_datadict.leaf_keys():
        to_save[name] = out_datadict[name]

    np.savez_compressed(out_path, **to_save)
