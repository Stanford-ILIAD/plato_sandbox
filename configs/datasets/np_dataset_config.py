from argparse import ArgumentParser

from sbrl.datasets.dataset import SequentialSampler, WeightedSequentialSampler
from sbrl.datasets.np_dataset import NpDataset
from sbrl.datasets.np_sequence_dataset import NpSequenceDataset
from sbrl.datasets.preprocess.data_augment_keys import DataAugmentKeys
from sbrl.experiments import logger
from sbrl.experiments.grouped_parser import LoadableGroupedArgumentParser
from sbrl.utils.python_utils import AttrDict as d, get_with_default

submodules = d(
    data_preprocessor=LoadableGroupedArgumentParser(optional=True),
)


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="/tmp/null.npz")
    parser.add_argument("--capacity", type=float, default=1e6)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--mmap_mode", type=str, default=None)
    parser.add_argument("--exclude_names", type=str, nargs="*", default=[], help="will be removed from batch names")
    parser.add_argument("--load_ignore_prefixes", type=str, nargs="*", default=[],
                        help="will be removed from loading names if prefixed by these")
    parser.add_argument("--initial_load_episodes", type=int, default=0)
    parser.add_argument("--split_frac", type=float, default=None)
    parser.add_argument("--frozen", action='store_true')
    parser.add_argument("--allow_padding", action='store_true')
    parser.add_argument("--pad_end", action='store_true')
    parser.add_argument("--index_all_keys", action='store_true')
    parser.add_argument("--async", action='store_true', help="batches will be loaded in the background.")

    parser.add_argument("--load_preprocessor", action='store_true', help="preprocessor submodule will be used.")

    parser.add_argument("--sequential", action='store_true',
                        help="batches will be loaded as iterated, shuffled sequences")
    parser.add_argument("--seq_no_shuffle", action='store_true',
                        help="if sequential, will not shuffle the loading order.")
    parser.add_argument("--seq_mode_key", type=str, default=None,
                        help="if sequential, will used weighted sampler using this key.")
    parser.add_argument("--seq_mtw", type=str, default='first',
                        help="if sequential, method for mode-to-weights (see WeightedSampler)")
    parser.add_argument("--seq_mode_weights", type=float, nargs='*', default=[1],
                        help="if sequential, will used weighted sampler using these weights for index key value.")

    parser.add_argument("--add_noise_keys", type=str, nargs="*", default=[])
    parser.add_argument("--add_noise_stds", type=float, nargs="*", default=[])
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert 'dataset' in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    # access to all the params for the current experiment here.
    prms = common_params >> group_name

    preprocessors = []
    if prms >> 'load_preprocessor':
        # load the type of data preprocessor (adds noise)
        common_params = (submodules['data_preprocessor'].params >> "process_params")(f"{group_name}/data_preprocessor",
                                                                                     common_params)
        dp = common_params >> f"{group_name}/data_preprocessor"
        if dp.is_empty():
            raise ValueError('Data processor is empty but one is required.')
        preprocessors.append(dp)  # will be loaded
    elif 'holdout' in group_name:
        dt_group_name = group_name.replace('holdout', 'train')
        if dt_group_name in common_params.keys():
            preprocessors = common_params << f"{dt_group_name}/params/data_preprocessors"
            if preprocessors is not None:
                logger.warn(f"NpDataset {group_name} using dp from {dt_group_name}")
            else:
                preprocessors = []

    if common_params.has_leaf_key("allow_padding"):
        prms.allow_padding = common_params >> "allow_padding"  # padding can be set by model.

    if "batch_names_to_get" in common_params.leaf_keys():
        batch_names_to_get = list(common_params >> "batch_names_to_get")
        logger.debug(f"NpDataset: Common Params supplied batch names: {batch_names_to_get}")
    elif len(prms >> "exclude_names") > 0:
        batch_names_to_get = (common_params >> "env_spec/params/observation_names") + \
                             (common_params >> "env_spec/params/output_observation_names") + \
                             (common_params >> "env_spec/params/action_names") + \
                             get_with_default(common_params, "env_spec/params/final_names", []) + \
                             get_with_default(common_params, "env_spec/params/goal_names", []) + \
                             get_with_default(common_params, "env_spec/params/param_names", [])
    else:
        batch_names_to_get = None

    if batch_names_to_get is not None:
        logger.debug(f"Batch names to get: {batch_names_to_get}")

    if batch_names_to_get is not None and len(prms >> "exclude_names") > 0:
        batch_names_to_get = list(set(batch_names_to_get).difference(prms >> "exclude_names"))

    sampler = None
    if prms >> "sequential":
        if prms >> "seq_mode_key" is None:
            sampler = d(cls=SequentialSampler, params=d(shuffle=not prms >> "seq_no_shuffle"))
        else:
            def_weights = list(prms >> 'seq_mode_weights')
            assert len(def_weights) > 0
            logger.debug(f"Using weighted sampling for key {prms >> 'seq_mode_key'}, weights = {def_weights}")
            # uses default mode -> weight function (see class for the method)
            sampler = d(cls=WeightedSequentialSampler, params=d(mode_key=prms >> 'seq_mode_key',
                                                                default_mtw=prms >> "seq_mtw",
                                                                num_modes=len(def_weights),
                                                                default_weights=def_weights))
    else:
        assert not prms >> "seq_no_shuffle", "Cannot disable shuffle for non-sequence loading."

    # split
    load_episode_range = None
    if prms >> "split_frac" is not None:
        if "train" in group_name:
            load_episode_range = [0., prms.split_frac]
        elif "holdout" in group_name:
            load_episode_range = [prms.split_frac, 1.]
        else:
            raise NotImplementedError(f"splitting dataset by fraction not implemented for dataset named: {group_name}")

    common_params[group_name] = common_params[group_name] & d(
        cls=NpSequenceDataset if prms >> "sequential" else NpDataset,
        params=d(
            file=prms >> "input_file",
            output_file=prms >> "output_file",
            save_every_n_steps=prms >> "save_every_n_steps",  # not used
            index_all_keys=prms >> "index_all_keys",
            initial_load_episodes=prms >> "initial_load_episodes",  # not used
            load_episode_range=load_episode_range,
            horizon=common_params >> "horizon",
            capacity=int(prms >> "capacity"),
            mmap_mode=prms >> "mmap_mode",
            batch_size=common_params >> "batch_size",
            asynchronous_get_batch=prms >> "async",  # careful with memory here...
            frozen=prms >> "frozen",  # careful with memory here...
            allow_padding=prms >> "allow_padding",  # computes padding mask per batch...
            pad_end_sequence=prms >> "pad_end",  # duplicates the last obs H-1 times for sampling
            batch_names_to_get=batch_names_to_get,
            # default is None, meaning all names, but earlier configs can change this
            load_ignore_prefixes=prms >> "load_ignore_prefixes",
            sampler=sampler,
            data_preprocessors=preprocessors,
        )
    )

    if len(prms >> "add_noise_keys") > 0 and len(prms >> "add_noise_keys") > 0:
        common_params[group_name].params.batch_processor = d(
            cls=DataAugmentKeys,
            params=d(
                add_noise_keys=prms >> "add_noise_keys",
                add_noise_stds=prms >> "add_noise_stds",
            )
        )
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
    submodules=submodules,
)
