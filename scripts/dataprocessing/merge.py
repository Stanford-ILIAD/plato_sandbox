"""
Merge multiple datasets, at the key level
"""

import argparse
import os

import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.input_utils import query_string_from_set
from sbrl.utils.python_utils import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--use_first_keys', action='store_true', help="uses the first dataset keys")
parser.add_argument('--skip_keys', type=str, nargs="*", help="which keys to ignore in all datasets", default=[])

args = parser.parse_args()

file_manager = FileManager()

out_path = file_path_with_default_dir(args.output_file, file_manager.data_dir)

for infile in args.file:
    assert os.path.isfile(infile)

input_data = []
input_data_keys = None
for inp_file in args.file:
    logger.debug(f"Loading: {inp_file}...")
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

logger.debug("Saving dataset output to -> %s" % out_path)

logger.debug("new shapes:\n%s" % out_datadict.leaf_shapes().pprint(ret_string=True))

if query_string_from_set("Save ok? [y, n]", ['y', 'n']) == 'y':
    to_save = dict()
    for name in out_datadict.leaf_keys():
        to_save[name] = out_datadict[name]

    np.savez_compressed(out_path, **to_save)
