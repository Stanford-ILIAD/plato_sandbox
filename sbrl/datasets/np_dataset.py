import math
import multiprocessing as mp
import os

import numpy as np
import sharedmem
import torch
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pad_sequence

from sbrl.datasets.dataset import Dataset
from sbrl.datasets.preprocess.batch_processor import BatchProcessor
from sbrl.datasets.preprocess.data_preprocessor import DataPreprocessor
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.file_utils import file_path_with_default_dir, postpend_to_base_name
from sbrl.utils.np_utils import np_pad_sequence
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit, get_required, get_or_instantiate_cls
from sbrl.utils.torch_utils import split_dim_np, split_dim, broadcast_dims, broadcast_dims_np, pad_dims, \
    combine_after_dim, to_torch, concatenate


class NpDataset(Dataset):

    # # @abstract.overrides
    def _init_params_to_attrs(self, params):
        super(NpDataset, self)._init_params_to_attrs(params)

        self._input_files = params.get("file",
                                       None)  # None if we are starting a new file, o/w where we load from (can be a list)
        self._output_file = params >> "output_file"  # Cannot be none, where we save data to
        assert params.output_file is not None

        # set to inf to keep recording everything
        # max length of dataset (transitions) before overriding (useful for replay buffer)
        self._capacity = int(params.capacity)
        assert self._capacity < np.inf
        self._frozen = get_with_default(params, "frozen", False)

        # temporal dilation when getting batch
        self._horizon_spacing = get_with_default(params, "horizon_skip", 1)
        assert self._horizon_spacing >= 1

        self._done_key = get_with_default(params, "done_key", "done", map_fn=str)

        self._dynamic_capacity = 0  # grows dynamically as needed
        self._dynamic_data_len = 0

        self._promised_inputs = None
        self._promised_outputs = None
        self._promised_meta = None

        self._use_rollout_steps = get_with_default(params, "use_rollout_steps", True)
        self._mmap_mode = get_with_default(params, "mmap_mode", None)
        if self._mmap_mode is not None:
            logger.info("Dataset using mmap mode: %s" % self._mmap_mode)

        self._step_names = get_with_default(params, "step_names", self._env_spec.names)
        self._onetime_names = get_with_default(params, "onetime_names",
                                               self._env_spec.param_names + self._env_spec.final_names)

        print(self._step_names, self._onetime_names)

        self._step_names = self._step_names + [self._done_key]
        if self._use_rollout_steps:
            self._step_names.append('rollout_timestep')

        self._batch_names_to_get = get_with_default(params, "batch_names_to_get", None)
        if self._batch_names_to_get is not None:
            logger.debug(f"Using batch names: {self._batch_names_to_get}")

        self._load_ignore_prefixes = get_with_default(params, "load_ignore_prefixes", [])

        assert len(self._step_names + self._onetime_names) == len(list(set(self._step_names + self._onetime_names))), \
            "non unique names in: %s" % (self._step_names + self._onetime_names)

        # load only this many episodes
        self._initial_load_episodes = get_with_default(params, "initial_load_episodes", 0)

        # fraction based loading
        self._load_episode_range = params << "load_episode_range"
        self._min_frac = 0  # round up, inclusive
        self._max_frac = 1  # round up, exclusive
        if self._load_episode_range is not None:
            assert len(self._load_episode_range) == 2, self._load_episode_range
            assert all(1. >= r >= 0. for r in self._load_episode_range), self._load_episode_range
            self._min_frac, self._max_frac = self._load_episode_range
            assert self._min_frac < self._max_frac, f"{self._min_frac} must be less than {self._max_frac}"

        self._allow_missing = get_with_default(params, "allow_missing", False)

        # pads episodes to length H if they are not this length already
        self._allow_padding = get_with_default(params, "allow_padding", False)
        # self._skip_short = get_with_default(params, "skip_short", True)

        # will pad the end of a sequence (H > 1) with the last value (H-1) times
        self._pad_end_sequence = get_with_default(params, "pad_end_sequence", False)

        # otherwise, will sample proportionally to the episode length (longer sequences get more samples)
        self._uniform_sample_eps = get_with_default(params, "uniform_sample_eps", False)

        if self._allow_padding:
            logger.info("Dataset allows padding.")

        if self._pad_end_sequence:
            logger.info("Dataset will pad ends of sequences for sampling.")

        # no duplicates!!
        assert len(set(self._step_names)) == len(self._step_names)

        self._data_len = 0  # number of filled elements
        self._add_index = 0  # add index

        self._asynchronous_get_batch = get_with_default(params, "asynchronous_get_batch", False)
        if self._asynchronous_get_batch:
            logger.warn("Batches will be loaded asynchronously!")

        # modifies the whole dictionary before loading
        self._data_preprocessors = get_with_default(params, "data_preprocessors", [])
        for i, dp in enumerate(self._data_preprocessors):
            if isinstance(dp, AttrDict):
                self._data_preprocessors[i] = get_or_instantiate_cls(dp, None, DataPreprocessor,
                                                                     constructor=lambda cls, prms: cls(prms,
                                                                                                       self._env_spec,
                                                                                                       self))
            else:
                assert isinstance(dp, DataPreprocessor)
            logger.debug(f"Dataset using preprocessor: {self._data_preprocessors[i].name}")

        # modifies batches
        self._batch_processor = params << "batch_processor"
        if params << "batch_processor" is not None:
            logger.debug(f"Using batch processor: {params >> 'batch_processor/cls'}")
            self._batch_processor = get_or_instantiate_cls(params, "batch_processor", BatchProcessor)

        # names to add to final_names (prepended by goal/ or param/), when loading (in case they were not saved in original dataset, e.g. goal conditioning).
        self._allow_missing_onetime = get_with_default(params, "allow_missing_onetime", True)

        # if True, all keys will be merged into one mega np dataset for indexing. this only works if dataset is frozen.
        self._index_all_keys = self._frozen and get_with_default(params, "index_all_keys", False)
        # if True, final names are loaded to "inputs", else "outputs".
        self._final_names_as_inputs = get_with_default(params, "final_names_as_inputs", True)

        # when to run batch processor in _get_batch
        self._batch_processor_before_torch = get_with_default(params, "batch_processor_before_torch",
                                                              not self._index_all_keys)

    def torchify(self, device):
        """
        be careful here, move whole dataset to torch
        NOTE: you can't add data if the dataset was torchified, not supported yet TODO
        :return:
        """
        logger.warn("Torch-ifying dataset to device: %s ..." % device)
        self._datadict = self._datadict.leaf_apply(lambda arr: torch.from_numpy(arr).to(device))
        self._onetime_datadict = self._onetime_datadict.leaf_apply(lambda arr: torch.from_numpy(arr).to(device))
        self._is_torch = True
        self._torchify_device = device
        logger.warn("Data in torch format...")

    def share_memory(self):
        # requires https://github.com/dillonalaird/shared_numpy
        if self._is_shared:
            return
        # import shared_numpy as snp
        # self._datadict.leaf_modify(lambda arr: snp.from_array(arr))
        for lk in self._datadict.leaf_keys():
            old = self._datadict[lk]
            self._datadict[lk] = sharedmem.copy(old)
            del old
        # moves to shared memory. this will be easier with python 3.9
        self._is_shared = True

    @property
    def save_dir(self):
        return self._output_file

    @property
    def start_idx(self):
        # underlying
        return (self._add_index - self._data_len) % self._capacity

    @property
    def effective_horizon(self):
        return self._horizon * self._horizon_spacing - (self._horizon_spacing - 1)

    # @abstract.overrides
    def _init_setup(self):
        self._datadict, self._onetime_datadict, self._split_indices = self._load_np()
        self._onetime_buffer = AttrDict()

        self._data_setup()

    def _data_setup(self):
        # static sampling likelihoods
        self._period_lengths = np.array([self.period_length(i) for i in range(self.get_num_periods())])
        # if padding end, use period_lengths, else period_lengths - H : for sampling padding.
        self._sampling_period_lengths = np.maximum(
            self._period_lengths - int(not self._pad_end_sequence) * self.effective_horizon, 1).astype(np.int64)
        self._num_valid_samples = sum(self._sampling_period_lengths)
        if self._uniform_sample_eps:
            self._period_probs = np.ones_like(self._period_lengths, dtype=np.float32)  # uniform
        else:
            # number of valid samples, sample proportional to period length
            self._period_probs = 1 / self._sampling_period_lengths
        self._period_probs /= self._period_probs.sum()

        # setup the batch processor with this dataset.
        if self._batch_processor is not None:
            self._batch_processor.setup(self)

        if self._index_all_keys:
            logger.debug("Merging datasets...")
            self._merged_datadict_keys = self._datadict.list_leaf_keys()
            self._merged_onetime_datadict_keys = self._onetime_datadict.list_leaf_keys()
            if self._batch_names_to_get is not None:
                names_to_get = self._batch_names_to_get + [self._done_key] + \
                               (["rollout_timestep"] if self._use_rollout_steps else [])
                self._merged_datadict_keys = list(set(names_to_get).intersection(self._merged_datadict_keys))
                self._merged_onetime_datadict_keys = list(
                    set(names_to_get).intersection(self._merged_onetime_datadict_keys))
                diff = list(set(names_to_get).difference(self._merged_datadict_keys))
                onetime_diff = list(set(names_to_get).intersection(self._merged_onetime_datadict_keys))
                if len(diff) > 0:
                    logger.warn(f"Batch names includes non-present names: {diff}")
                if len(onetime_diff) > 0:
                    logger.warn(f"Batch names includes non-present onetime names: {onetime_diff}")

            self._merged_datadict = concatenate(
                self._datadict.leaf_apply(lambda arr: combine_after_dim(arr, start_dim=1, allow_no_dim=True)),
                self._merged_datadict_keys, dim=-1)
            if len(self._onetime_datadict.list_leaf_keys()) > 0 and len(self._merged_onetime_datadict_keys) > 0:
                self._merged_onetime_datadict = concatenate(self._onetime_datadict.leaf_apply(
                    lambda arr: combine_after_dim(arr, start_dim=1, allow_no_dim=True)),
                                                            self._merged_onetime_datadict_keys, dim=-1)
            else:
                self._merged_onetime_datadict = None
            logger.debug("Datasets have been merged")

    def _init_empty(self, local_dict: AttrDict, onetime_dict):
        self._data_len = 0  # number of filled elements
        self._add_index = 0  # add index

        # static allocation prevents failures of memory!
        for key in self._step_names:
            local_dict[key] = np.empty((self._capacity, *(self._env_spec.names_to_shapes >> key)),
                                       dtype=self._env_spec.names_to_dtypes >> key)
        # these names are dynamically allocated since it's just once per episode
        for key in self._onetime_names:
            onetime_dict[key] = np.empty([0] + list(self._env_spec.names_to_shapes >> key),
                                         dtype=self._env_spec.names_to_dtypes >> key)  # empty to begin
        split_indices = np.array([], dtype=np.long)

        return local_dict, onetime_dict, split_indices

    def _load_np(self):
        local_dict = AttrDict()
        onetime_dict = AttrDict()
        local_dict, onetime_dict, split_indices = self._init_empty(local_dict, onetime_dict)

        if self._input_files is None:
            return local_dict, onetime_dict, split_indices

        # iterate through all files, loading each
        self._input_files = [self._input_files] if isinstance(self._input_files, str) else self._input_files

        if len(self._input_files) == 0:
            logger.warn("Np dataset received no files!")

        i = 0
        num_eps = 0
        data_len = 0
        for file in self._input_files:
            i += 1
            logger.debug('Loading ' + file)
            if os.path.isfile(file):
                path = file
            else:
                if isinstance(self._file_manager, ExperimentFileManager):
                    path = os.path.join(self._file_manager.exp_dir, file)
                else:
                    path = os.path.join(self._file_manager.data_dir, file)

                # fail safe (invalid dataset passed in)
                if not os.path.isfile(path):
                    logger.warn("Could not find dataset: %s" % file)
                    return self._init_empty(local_dict, onetime_dict)

            # load the dictionary from memory
            datadict = np.load(path, allow_pickle=True, mmap_mode=self._mmap_mode)  # , mmap_mode="r")
            datadict = dict(datadict)

            if len(self._load_ignore_prefixes) > 0:
                # keep keys that aren't ignored
                datadict = AttrDict.from_dict(datadict) \
                    .leaf_filter(lambda k, v: not any(k.startswith(pr) for pr in self._load_ignore_prefixes)) \
                    .as_dict()

            # this is the number of new elements to add
            logger.debug("keys: " + str(list(datadict.keys())))
            new_data_len = len(datadict[self._done_key])

            new_data_num_eps = len(np.where(datadict[self._done_key])[0])
            num_eps += new_data_num_eps

            # if we are loading more than we want / enough, end early
            finished_loading = False
            if num_eps >= self._initial_load_episodes > 0:
                delta = self._initial_load_episodes - (num_eps - new_data_num_eps)
                logger.warn("Loading only %d / %d episodes from file"
                            % (delta, new_data_num_eps))
                assert delta > 0
                # setting things appropriately
                old_new_data_num_eps = new_data_num_eps
                new_data_num_eps = delta
                num_eps = self._initial_load_episodes
                new_data_len = np.where(datadict[self._done_key])[0][delta - 1] + 1

                datadict[self._done_key] = datadict[self._done_key][:new_data_len]
                for key in set(self._step_names).intersection(datadict.keys()):
                    datadict[key] = datadict[key][:new_data_len]

                #  and key.startswith("final/") and key[6:] in self._add_final_names
                for key in set(self._onetime_names).intersection(datadict.keys()):
                    # one per ep, or one per step
                    if datadict[key].shape[0] == old_new_data_num_eps:
                        datadict[key] = datadict[key][:new_data_num_eps]
                    else:
                        datadict[key] = datadict[key][:new_data_len]

                finished_loading = True

            # make sure the dataset ends with a 'done' flag
            assert datadict[self._done_key][-1], "Dataset %s must end with done == True" % file

            if data_len + new_data_len > self._capacity:
                logger.warn(
                    "Data (%d) exceeds the capacity in allocated storage (%d), increase capacity param in NpDataset" % (
                        data_len + new_data_len, self._capacity))
                raise NotImplementedError

            # input files missing keys, reduce overall dataset to the intersection of keys, update all_names
            if self._allow_missing:  # TODO @suneel fix this with onetime_dict
                raise NotImplementedError
                if i == 1:
                    self._step_names = list(datadict.keys())
                    local_dict, onetime_dict, split_indices = self._init_empty(local_dict, onetime_dict)
                else:
                    self._step_names = list(set(self._step_names).intersection(datadict.keys()))
                local_dict = local_dict.leaf_filter_keys(self._step_names)

            if self._allow_missing_onetime:
                missing_onetime_names = list(set(self._onetime_names).difference(datadict.keys()))
                # names to get from last part of each episode (will warn if present already)
                for name in missing_onetime_names:
                    logger.debug(f"FILLING IN NAME: {name}")
                    assert name.startswith("goal/") or name.startswith("param/"), name
                    prefix, inner_name = (name[:5], name[5:]) if name.startswith("goal/") else (name[:6], name[6:])
                    all_vals = datadict[inner_name]
                    # make sure it was truncated and is not onetime. (will be properly parsed later)
                    assert all_vals.shape[0] in [new_data_len,
                                                 num_eps], f"{name} must be a regular (parsed) step name: {all_vals.shape}"
                    datadict[name] = all_vals

            # assign each key of step sample keys in our local dictionary based on the np input
            for key in self._step_names:
                assert key in datadict, '%s not in np file' % key
                assert len(datadict[key]) == new_data_len, [key, len(datadict[key]), new_data_len]
                # fill in the chunk of the data set
                try:
                    local_dict[key][data_len:data_len + new_data_len] = datadict[key]
                except ValueError as e:
                    logger.debug([e, key])
                    raise e

            # assign each key of one-per-episode keys in our local dictionary based on the np input
            for key in self._onetime_names:
                assert key in datadict, '%s not in np file' % key
                # assert len(datadict[key])
                # fill in the chunk of the data set
                onetime_dict[key] = np.concatenate([onetime_dict[key], datadict[key]], axis=0)

            data_len += new_data_len
            logger.debug('Dataset \"{}\" length: {}'.format(file, new_data_len))

            if finished_loading:
                logger.warn("Finished loading: loaded length = %d" % data_len)
                break

        # if no elements, this is []
        # if N >= 1 episodes, this has [L1, L1+L2, ... L1+.+LN] (human ordering is maintained
        #  regardless of wrapping)
        split_indices = np.where(local_dict[self._done_key][:data_len])[0] + 1  # one after each episode ends

        # if self._skip_short:
        #     ep_lengths = np.concatenate([ [split_indices[0]], np.diff(split_indices) ])
        #     short_idxs = (ep_lengths < self.horizon).nonzero()[0]
        #     remove_len = ep_lengths[short_idxs].sum()
        #     keep_onetime_mask = ~(ep_lengths < self.horizon)
        #     # episode index
        #     episode_idx_for_element = np.argmax(np.arange(self._data_len)[:, None] < split_indices[None], axis=-1)
        #     episode_idx_for_element
        #     local_dict.leaf_modify(lambda arr: arr[keep_batch_mask])
        #

        # make sure the number of episodes is consistent in dataset
        for key in self._onetime_names:
            final = key in self._env_spec.final_names
            if len(onetime_dict[key]) == data_len:
                # more data was passed in than needed, just get the episode starts, SPECIAL CASE
                logger.warn(f"Truncating onetime key {key} to {'last' if final else 'first'} value in each episode, "
                            f"since it is same shape as normal data")
                first_pts = [0] + list(split_indices)[:-1]  # all the true episode starts
                last_pts = list(split_indices - 1)  # all the true episode ends
                onetime_dict[key] = onetime_dict[key][last_pts] if final else onetime_dict[key][first_pts]
            assert len(onetime_dict[key]) == len(split_indices), "Key[%s] needs [%d] samples, had [%d]" % (
                key, len(onetime_dict[key]), len(split_indices))

        # keeping only some range of the dataset
        if self._load_episode_range is not None:
            ep_lower = int(math.ceil(self._min_frac * num_eps))
            ep_upper = int(math.ceil(self._max_frac * num_eps))
            logger.warn(
                f"Dataset keeping range [{ep_lower}, {ep_upper}), or {ep_upper - ep_lower} / {num_eps} episodes.")

            step_start = split_indices[ep_lower - 1] if ep_lower > 0 else 0
            step_end = split_indices[ep_upper - 1]  # non inclusive

            local_dict = local_dict.leaf_apply(lambda arr: arr[step_start:step_end])
            onetime_dict = onetime_dict.leaf_apply(lambda arr: arr[ep_lower:ep_upper])

            data_len = step_end - step_start
            split_indices = split_indices[ep_lower:ep_upper] - step_start

        # other preprocessing for data loading
        for data_preprocessor in self._data_preprocessors:
            local_dict, onetime_dict, split_indices = data_preprocessor.forward(self, local_dict, onetime_dict,
                                                                                split_indices)
            logger.debug(f"[Preprocessor]: {data_preprocessor.name}, length: {data_len} -> {len(local_dict >> 'done')}")
            logger.debug(f"New keys: {local_dict.list_leaf_keys()}, onetime: {onetime_dict.list_leaf_keys()}")
            data_len = len(local_dict >> "done")

        self._data_len = data_len

        # parsing the dynamic capacity (number of episodes)
        if len(self._onetime_names) > 0:
            self._dynamic_capacity = len(onetime_dict[self._onetime_names[-1]])
            self._dynamic_data_len = self._dynamic_capacity  # to start, the dynamic array is full

        # NOTE: we assumes all datasets have been stored sequentially!
        self._add_index = self._data_len % self._capacity

        logger.debug('Dataset Total length: {}'.format(self._data_len))
        logger.debug('Dataset Starting Idx: {}'.format(self._add_index))
        logger.debug('Dataset Number of Eps: {}'.format(len(split_indices)))

        if self._frozen and len(split_indices) > 0:
            local_dict.leaf_modify(lambda arr: arr[:self._data_len])
            onetime_dict.leaf_modify(lambda arr: arr[:len(split_indices)])

            # per episode, where does it start, how long is it, and where is the last position we can start.
            self.ep_starts = np.concatenate([[0], split_indices[:-1]])
            self.ep_lengths = np.concatenate([[split_indices[0]], np.diff(split_indices)])
            logger.debug(
                'Dataset Ep Lengths: min {}, max {}, med {}'.format(np.min(self.ep_lengths), np.max(self.ep_lengths),
                                                                    np.median(self.ep_lengths)))
            # long_enough = (ep_lengths >= self.horizon).nonzero()[0]
            # np.where(long_enough)
            # first startable is always 0
            # last startable is either L-1 (if padding the end) or L-1-H
            self.last_startable = np.maximum(
                self.ep_lengths - 1 - int(not self._pad_end_sequence) * self.effective_horizon, 0)

        return local_dict, onetime_dict, split_indices

    def get_statistics(self, names):
        assert len(self._datadict) > 0, "Empty datadict"
        filtered = self.get_datadict().leaf_filter_keys(
            names)  # be careful with this fn call, it computes full data ordering to retrieve the data properly, O(N)
        filtered.combine(self.get_onetime_datadict().leaf_filter_keys(names))
        assert filtered.has_leaf_keys(names), \
            f"Datadict is missing the following keys: {set(names).difference(filtered.leaf_keys())}"
        means = filtered.leaf_apply(lambda arr: np.mean(arr, axis=0))
        means.leaf_modify(lambda arr: np.where(np.isnan(arr), 0, arr))
        stds = filtered.leaf_apply(lambda arr: np.std(arr, axis=0))
        stds.leaf_modify(
            lambda arr: np.where(np.logical_or(np.isnan(arr), arr == 0), 1, arr))  # stdev should not be zero or inf

        mins = filtered.leaf_apply(lambda arr: np.min(arr, axis=0))
        maxs = filtered.leaf_apply(lambda arr: np.max(arr, axis=0))
        out = AttrDict(mean=means, std=stds, min=mins, max=maxs)
        return out

    def _dynamic_expand(self, datadict, old_size, new_size):
        for key in datadict.leaf_keys():
            data = datadict[key]
            assert new_size > len(data) >= old_size, "%s: new_size: %d / old_size: %d, data_size: %d" % (
                key, new_size, old_size, len(data))
            new_data = np.empty((new_size, *data.shape[1:]), dtype=data.dtype)
            new_data[:old_size] = data[:old_size]
            datadict[key] = new_data

    def _dynamic_add(self, datadict, new_datadict):
        # check keys are all there (add all at once)
        new_part_len = 0
        for key in datadict.leaf_keys():
            assert key in new_datadict.leaf_keys(), key
            assert new_part_len == 0 or new_datadict[key].shape[0] == new_part_len, "%s has length %d instead of %d" % (
                key, new_datadict[key].shape[0], new_part_len)
            new_part_len = new_datadict[key].shape[0]

        # figure out new capacity
        new_len = self._dynamic_data_len + new_part_len  # how much will we have
        copy_over = new_len > self._dynamic_capacity
        while new_len > self._dynamic_capacity:
            self._dynamic_capacity = self._dynamic_capacity * 2 + 1  # 1 for good measure, since we start with 0
        if copy_over:
            logger.debug(f"{self._dynamic_data_len} -> {self._dynamic_capacity}")
            logger.debug(datadict.leaf_apply(lambda arr: len(arr)).pprint(ret_string=True))
            self._dynamic_expand(datadict, self._dynamic_data_len, self._dynamic_capacity)  # expands to capacity

        for key in datadict.leaf_keys():
            # copy new data in, since we know there is enough space now
            datadict[key][self._dynamic_data_len:new_len] = new_datadict[key]  # new data assignment

        self._dynamic_data_len = new_len  # the new amount of data <= capacity
        assert self._dynamic_data_len <= self._dynamic_capacity, "%d / %d" % (
            self._dynamic_data_len, self._dynamic_capacity)

    def episode_length(self, i):
        prev = self._split_indices[i - 1] if i > 0 else 0
        return self._split_indices[i] - prev  # length of episode i

    def period_length(self, i):
        return self.episode_length(i)

    def period_weights(self, indices=None):
        if indices is None:
            return self._period_probs
        return self._period_probs[indices]

    def get_num_episodes(self):
        return len(self._split_indices)

    # getting the indices for a chunk from episode i (human ordering)
    def sample_indices_for_episode(self, i, horizon=None, horizon_spacing=None):
        prev = self._split_indices[i - 1] if i > 0 else 0  # first idx
        ep_len = self._split_indices[i] - prev  # length of episode i
        horizon = self._horizon if horizon is None else horizon  # allow user override
        horizon_spacing = self._horizon_spacing if horizon_spacing is None else horizon_spacing
        # how many elements does a horizon span.
        effective_horizon = horizon * horizon_spacing - (horizon_spacing - 1)

        if effective_horizon > ep_len:
            # "Padding must be enabled when retrieving episodes smaller than effective_horizon length"
            if self._allow_padding:
                # get the exact sequence, will be zero padded later
                return np.arange(prev, prev + ep_len)
            else:
                # get the last sample N times
                in_bounds = list(range(prev, prev + ep_len, horizon_spacing))
                return np.asarray(in_bounds + [prev + ep_len - 1] * (horizon - len(in_bounds)))
        else:
            # pick a random chunk of the episode of the right length
            rg = max(ep_len - effective_horizon + 1, 1)
            idx = np.random.choice(rg)  # pick a starting point
            return np.arange(prev + idx, prev + idx + horizon * horizon_spacing, step=horizon_spacing)

    # getting the indices for a chunk from episode i (human ordering)
    def batch_sample_indices_for_episode(self, indices):
        # print("len", len(indices))
        assert self._frozen, "not implemented for dynamic datasets"
        relevant_ep_starts = self.ep_starts[indices]
        relevant_ep_lengths = self.ep_lengths[indices]
        relevant_ep_last_starts = self.last_startable[indices]
        random_start_offset = np.random.randint(0, relevant_ep_last_starts + 1)

        # start = relevant_ep_starts + random_start_offset
        # horizon forward (B x H) -> list of BxH, make sure they don't go outside ep boundary
        within_ep_idxs = random_start_offset[:, None] + np.arange(self.horizon)[None] * self._horizon_spacing

        # if self._allow_padding and np.any(relevant_ep_lengths < self.horizon):
        #     raise NotImplementedError("need to implement padding the sequences")

        bounded_ep_idxs = np.minimum(within_ep_idxs, (relevant_ep_lengths - 1)[:, None])
        indices = relevant_ep_starts[:, None] + bounded_ep_idxs

        # how long each sample really is
        chunk_lengths = np.minimum(relevant_ep_lengths, self.effective_horizon)

        return indices.reshape(-1), chunk_lengths

    def get_episode(self, i, names, split=False, torch_device=None,
                    include_done=True, include_rt=True, onetime_broadcast=True, **kwargs):
        # sampled datadict has Ni samples, onetime has 1 sample per key
        if names is None:
            names = self._batch_names_to_get if self._batch_names_to_get is not None else self._env_spec.all_names
        names = names + [self._done_key] if include_done else names
        names = names + ["rollout_timestep"] if include_rt else names
        chunk_idxs = self.sample_indices_for_episode(i, horizon=self.episode_length(i), horizon_spacing=1)
        real_idxs = self.idx_map_to_real_idx(chunk_idxs)
        sampled_datadict = self._datadict.leaf_filter_keys(names).leaf_apply(lambda arr: arr[real_idxs])
        onetime_datadict = self._onetime_datadict.leaf_filter_keys(names).leaf_apply(lambda arr: arr[i:i + 1])

        # ep_idxs is sparse (1,..) -> (H,..)
        if onetime_broadcast:
            if self._is_torch:
                onetime_datadict.leaf_modify(lambda arr: broadcast_dims(arr, [0], [len(real_idxs)]))
            else:
                onetime_datadict.leaf_modify(lambda arr: broadcast_dims_np(arr, [0], [len(real_idxs)]))

        sampled_datadict.combine(onetime_datadict)
        all_ds = []
        if split:
            inputs = AttrDict()
            outputs = AttrDict()
            intersect_in = set(
                self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names + self._env_spec.param_names) \
                .intersection(set(names))
            intersect_out = set(self._env_spec.output_observation_names + self._env_spec.final_names) \
                .intersection(set(names))
            for key in intersect_in:
                inputs[key] = sampled_datadict[key]
            for key in intersect_out:
                outputs[key] = sampled_datadict[key]

            if include_done:
                if self._is_torch:
                    outputs[self._done_key] = sampled_datadict[self._done_key].to(dtype=bool)
                else:
                    outputs[self._done_key] = sampled_datadict[self._done_key].astype(bool)
            # inputs/outputs get torchified and returned
            all_ds.extend([inputs, outputs])
        else:
            # raw gets torchified and returned
            all_ds.append(sampled_datadict)

        if torch_device is not None:
            for d in all_ds:
                if self._is_torch:
                    d.leaf_modify(lambda x: x.to(torch_device))
                else:
                    d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return all_ds[0] if len(all_ds) == 1 else tuple(all_ds)

    # human mapped indices
    def get_ep_indices_for_step_indices(self, indices):
        # ep_idxs = np.argmax(np.asarray(indices)[:, None] < self._split_indices[None], axis=-1)
        return np.searchsorted(self._split_indices, indices,
                               side='right')  # split indices are the one past the last idx for that ep

    def _resolve_indices(self, indices=None, min_idx=0, max_idx=0, np_pad=False, local_batch_size=None, **kwargs):
        assert self._data_len > 0
        # if no indices provided, sample randomly from [min_idx, max_idx),
        # either sampling
        data_length = self._data_len if self._horizon == 1 else self._split_indices.size
        if indices is None:
            probs = None if self._horizon == 1 else self._period_probs
            if self._horizon > 1:
                assert len(self._period_probs) == data_length
            # indices stop at max (exclusive), <= 0, wrap backwards from end
            # TODO current samples eating in?
            if max_idx <= 0:
                max_idx += data_length
            assert 0 <= min_idx < max_idx <= data_length

            indices = np.random.choice(max_idx - min_idx, self._batch_size,
                                       p=probs[min_idx:max_idx] if probs is not None else None,
                                       replace=max_idx - min_idx < self._batch_size)
            indices += min_idx  # base index to consider in dataset
        else:
            local_batch_size = indices.shape[0]
            assert np.max(indices) < data_length and np.min(indices) >= 0

        with timeit("get_batch/get_episodes"):
            if self._horizon > 1:
                # all the contiguous episode indices are mapped contiguous horizon ranges:
                episode_indices = indices
                indices, chunk_lengths = self.batch_sample_indices_for_episode(episode_indices)
                # with timeit("more_specific"):
                #     indices = [self.sample_indices_for_episode(i) for i in indices]  # all the sample idxs
                # chunk_lengths = [len(ls) for ls in indices]  # lengths of each episode sample

                do_padding = self._allow_padding and np.any(
                    chunk_lengths < self.horizon)  # pad if some sequences are not len() = horizon
                # indices = np.concatenate(indices)
            else:
                episode_indices = self.get_ep_indices_for_step_indices(indices)  # all the episode idxs
                chunk_lengths = [1] * len(episode_indices)
                do_padding = False

        assert do_padding or len(indices) == local_batch_size * self._horizon, \
            "contiguous indices should be of size B*H in get_batch"

        return indices, episode_indices, chunk_lengths, do_padding

    def _get_batch(self, indices=None, torch_device=None, min_idx=0, max_idx=0, non_blocking=False, np_pad=False,
                   local_batch_size=None, **kwargs):
        """

        :param indices: if HORIZON > 1, this is the episodes to sample chunks from
        :param torch_device:
        :param min_idx:
        :param max_idx:
        :param non_blocking:
        :param kwargs:
        :return:
        """
        # import ipdb; ipdb.set_trace()

        if local_batch_size is None:
            local_batch_size = self._batch_size
        indices, episode_indices, chunk_lengths, do_padding = self._resolve_indices(indices, min_idx=min_idx,
                                                                                    max_idx=max_idx, np_pad=np_pad,
                                                                                    local_batch_size=local_batch_size,
                                                                                    **kwargs)

        # first parse the keys we care about
        if self._batch_names_to_get is not None and len(self._batch_names_to_get) > 0:
            names_to_get = self._batch_names_to_get
        else:
            names_to_get = self._env_spec.all_names

        names_to_get = names_to_get + [self._done_key] + \
                       (["rollout_timestep"] if self._use_rollout_steps else [])

        meta = AttrDict()
        if self._index_all_keys:
            assert self._batch_processor is None or not self._batch_processor_before_torch, "Batch processor must be after torch if specified."
            # this indexes by the combined dataset (all keys at once, names_to_get will be parsed from this)
            inputs, outputs = self._get_batch_index_into_merged_datadict(indices, episode_indices, names_to_get,
                                                                         local_batch_size,
                                                                         chunk_lengths, do_padding, self.horizon,
                                                                         np_pad=np_pad, torch_device=torch_device)

        else:
            inputs, outputs = self._get_batch_index_into_datadict(indices, episode_indices, names_to_get,
                                                                  local_batch_size,
                                                                  chunk_lengths, do_padding, self.horizon,
                                                                  np_pad=np_pad)

            if self._batch_processor is not None and self._batch_processor_before_torch:
                # default behavior is episode_indices == period_indices
                inputs, outputs, meta = self._batch_processor.forward(inputs, outputs, indices, episode_indices,
                                                                      episode_indices, names_to_get, chunk_lengths)

            with timeit("get_batch/to_device"):
                if torch_device is not None:
                    for d in (inputs, outputs, meta):
                        if self._is_torch:
                            d.leaf_modify(lambda x: x.to(torch_device, non_blocking=non_blocking))
                        else:
                            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device, non_blocking=non_blocking))

        if self._batch_processor is not None and not self._batch_processor_before_torch:
            inputs, outputs, meta = self._batch_processor.forward(inputs, outputs, indices, episode_indices,
                                                                  episode_indices, names_to_get, chunk_lengths)

        return inputs, outputs, meta

    def _get_batch_index_into_datadict(self, indices, episode_indices, names_to_get, local_batch_size, chunk_lengths,
                                       do_padding, horizon, np_pad=False):
        """
        Indexing routine,
        """
        with timeit("get_batch/indexing"):
            # get the idxs in the underlying data
            with timeit("get_batch/true_idx"):
                dd = self._datadict.leaf_filter_keys(names_to_get)
                true_indices = self.idx_map_to_real_idx(indices)

            if self._is_torch:
                # Torch sampling (slower, usually no need for this)
                true_indices = torch.from_numpy(true_indices).to(self._torchify_device)

                with timeit("get_batch/inner_indexing"):
                    # actually get indices of data
                    sampled_datadict = dd.leaf_apply(lambda arr: torch.index_select(arr, 0, true_indices))

                # pad to the right horizon shape
                if do_padding:
                    # B x Tmin x ...
                    sampled_datadict.leaf_modify(lambda arr: pad_sequence(torch.split(arr, chunk_lengths, dim=0),
                                                                          batch_first=True, padding_value=0.))
                    sampled_datadict.leaf_modify(
                        lambda arr: pad_dims(arr, [1], [horizon], delta=False) if arr.shape[
                                                                                      1] < horizon else arr)
                else:
                    # B x horizon
                    sampled_datadict.leaf_modify(lambda arr: split_dim(torch_in=arr, dim=0,
                                                                       new_shape=(local_batch_size, self._horizon)))

            else:
                # actually sample
                with timeit("get_batch/inner_indexing"):
                    sampled_datadict = dd.leaf_apply(lambda arr: arr[true_indices])

                # pad horizon length
                if do_padding:
                    with timeit("get_batch/indexing/pad"):
                        # group_ends = np.cumsum(chunk_lengths)[:-1]
                        order = sampled_datadict.list_leaf_keys()
                        dtypes = sampled_datadict.leaf_apply(lambda arr: arr.dtype)
                        arrays = sampled_datadict.leaf_apply(
                            lambda arr: combine_after_dim(arr, 1) if len(arr.shape) > 1 else arr[:,
                                                                                             None]).get_keys_required(
                            order)
                        joint_dim = sum(int(np.prod(arr.shape[1:])) for arr in arrays)

                        # pre-allocate big array memory once (or if a bigger array is requested)
                        if not hasattr(self, "_saved_padded_array") or self._saved_padded_array.shape[0] < \
                                arrays[0].shape[0]:
                            self._saved_padded_array = np.empty((arrays[0].shape[0], joint_dim), dtype=np.float32)

                        # write the output of concatenate to pre-alloc array
                        all_cat = self._saved_padded_array[:arrays[0].shape[0]]
                        # with timeit("cat_array"):
                        all_cat = np.concatenate(arrays, -1, out=all_cat)

                        # move to torch before pad split if not padding at numpy array
                        if not np_pad:
                            all_cat = torch.from_numpy(all_cat)

                        # splitting and padding (these can be quite slow)
                        with timeit("get_batch_indexing/split_and_pad_seq"):
                            if np_pad:
                                stacked = np_pad_sequence(
                                    np.split(all_cat, np.cumsum(chunk_lengths)[:-1].astype(int), axis=0))
                            else:
                                with torch.no_grad():
                                    split_seq = torch.split(all_cat, chunk_lengths.tolist(), dim=0)
                                    # B x maxH
                                    stacked = pad_sequence(split_seq, batch_first=True)
                                    stacked = stacked.numpy()

                            # pre allocate all inner padded arrays
                            if not hasattr(self, "_save_split_padded_dc") \
                                    or self._save_split_padded_dc.get_one().shape[1] < stacked.shape[1]:
                                self._saved_split_padded_dc = AttrDict.from_dict({
                                    name: np.empty((*stacked.shape[:2], *sampled_datadict[name].shape[1:]),
                                                   dtype=dtypes[name]) for name in order
                                })

                        # copy over the stacked padded arrays into a dict
                        with timeit("parse"):
                            sampled_datadict = self._env_spec.parse_from_concatenated_flat(
                                stacked, order, outs=self._saved_split_padded_dc.leaf_apply(
                                    lambda arr: arr[:stacked.shape[0], :stacked.shape[1]]),
                                copy=True
                            )
                else:
                    sampled_datadict.leaf_modify(lambda arr: split_dim_np(np_in=arr, axis=0,
                                                                          new_shape=(local_batch_size, horizon)))

        with timeit("get_batch/onetime"):
            od = self._onetime_datadict.leaf_filter_keys(names_to_get)

            sampled_onetime_datadict = od.leaf_apply(lambda arr: arr[episode_indices, None])
            if self._horizon > 1:
                if self._is_torch:
                    sampled_onetime_datadict.leaf_modify(lambda arr: broadcast_dims(arr, [1], [horizon]))
                else:
                    sampled_onetime_datadict.leaf_modify(lambda arr: broadcast_dims_np(arr, [1], [horizon]))

            sampled_datadict.combine(sampled_onetime_datadict)

        inputs = AttrDict()
        outputs = AttrDict()

        names_set = set(names_to_get)
        for key in names_set.intersection(
                self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names):
            inputs[key] = sampled_datadict[key]
        for key in names_set.intersection(self._env_spec.output_observation_names + self._env_spec.output_goal_names):
            outputs[key] = sampled_datadict[key]

        # params are inputs because they are known ahead of time
        for key in names_set.intersection(self._env_spec.param_names):
            inputs[key] = sampled_onetime_datadict[key]

        # put missing names in inputs by default.
        extra_names = names_set.difference(inputs.list_leaf_keys() + outputs.list_leaf_keys())
        for key in extra_names:
            inputs[key] = sampled_datadict >> key

        # default should be use as inputs
        src = inputs if self._final_names_as_inputs else outputs

        # finals are outputs because they are known only after the episode is over
        for key in names_set.intersection(self._env_spec.final_names):
            src[key] = sampled_onetime_datadict[key]

        if 'rollout_timestep' in sampled_datadict.leaf_keys():
            inputs['rollout_timestep'] = sampled_datadict['rollout_timestep']

        if self._is_torch:
            outputs[self._done_key] = sampled_datadict[self._done_key].to(dtype=bool)
        else:
            outputs[self._done_key] = sampled_datadict[self._done_key].astype(bool)

        if do_padding:
            assert not inputs.has_leaf_key('padding_mask')
            if self._is_torch:
                inputs.padding_mask = torch.zeros(local_batch_size, horizon, dtype=torch.bool,
                                                  device=inputs[self._done_key].device)
            else:
                inputs.padding_mask = np.zeros((local_batch_size, horizon), dtype=bool)

            # basically, this tells us which steps are padded. if any
            for i, clen in enumerate(chunk_lengths):
                inputs.padding_mask[i, clen:] = True

            # logger.debug(f"\n{inputs.leaf_apply(lambda arr: arr.dtype).pprint(ret_string=True)}")
            # logger.debug(f"\n{outputs.leaf_apply(lambda arr: arr.dtype).pprint(ret_string=True)}")
            # logger.debug(f"Padded to lengths:\n{chunk_lengths}")
            # logger.debug(f"Pad Mask SUM:\n{(~inputs.padding_mask).sum(-1)}")

        return inputs, outputs  # each shape is (batch, horizon, name_dim...)

    def _get_batch_index_into_merged_datadict(self, indices, episode_indices, names_to_get, local_batch_size,
                                              chunk_lengths,
                                              do_padding, horizon, np_pad=False, torch_device=None):

        is_torch = self._is_torch  # keeping track of if we are in torch or not

        with timeit("get_batch/indexing"):
            # idxs in underlying data
            with timeit('maptoreal'):
                true_indices = self.idx_map_to_real_idx(indices)

            # if data is in torch format
            if self._is_torch:
                true_indices = torch.from_numpy(true_indices).to(self._torchify_device)

                # get indices of data
                sampled_datadict_arr = torch.index_select(self._merged_datadict, 0, true_indices)

                # pad to the right shape
                if do_padding:
                    # B x Tmin x ...
                    sampled_datadict_arr = pad_sequence(torch.split(sampled_datadict_arr, chunk_lengths, dim=0),
                                                        batch_first=True, padding_value=0.)
                    if sampled_datadict_arr.shape[1] < horizon:
                        sampled_datadict_arr = pad_dims(sampled_datadict_arr, [1], [horizon], delta=False)
                else:
                    # B x horizon
                    sampled_datadict_arr = split_dim(torch_in=sampled_datadict_arr, dim=0,
                                                     new_shape=(local_batch_size, horizon))
            else:
                # actually sampling
                with timeit("get_batch/inner_indexing"):
                    sampled_datadict_arr = self._merged_datadict[true_indices]

                if do_padding:
                    # move array to numpy
                    if not np_pad:
                        sampled_datadict_arr = torch.from_numpy(sampled_datadict_arr)

                    # padding array and splitting it into its chunks.
                    with timeit("get_batch_indexing/split_and_pad_seq"):
                        if np_pad:
                            sampled_datadict_arr = np_pad_sequence(
                                np.split(sampled_datadict_arr, np.cumsum(chunk_lengths)[:-1].astype(int), axis=0))
                        else:
                            with torch.no_grad():
                                split_seq = torch.split(sampled_datadict_arr, chunk_lengths.tolist(), dim=0)
                                # B x maxH
                                stacked = pad_sequence(split_seq, batch_first=True)
                                sampled_datadict_arr = stacked.numpy()

                else:
                    sampled_datadict_arr = split_dim_np(np_in=sampled_datadict_arr, axis=0,
                                                        new_shape=(local_batch_size, horizon))

            if torch_device is not None:
                is_torch = True
                if not self._is_torch:
                    with timeit("get_batch/to_torch"):
                        sampled_datadict_arr = torch.from_numpy(sampled_datadict_arr).to(device=torch_device)

        with timeit("get_batch/onetime_indexing"):
            sampled_onetime_datadict_arr = None
            if self._merged_onetime_datadict is not None:
                sampled_onetime_datadict_arr = self._merged_onetime_datadict[episode_indices, None]

                # TO TORCH
                if torch_device is not None:
                    sampled_onetime_datadict_arr = to_torch(sampled_onetime_datadict_arr, device=torch_device,
                                                            check=True)

                if self._horizon > 1:
                    if is_torch:
                        sampled_onetime_datadict_arr = broadcast_dims(sampled_onetime_datadict_arr, [1], [horizon])
                    else:
                        sampled_onetime_datadict_arr = broadcast_dims_np(sampled_onetime_datadict_arr, [1], [horizon])

            # PARSE
            sampled_datadict = self._env_spec.parse_view_from_concatenated_flat(sampled_datadict_arr,
                                                                                self._merged_datadict_keys)
            if sampled_onetime_datadict_arr is not None:
                sampled_onetime_datadict = self._env_spec.parse_view_from_concatenated_flat(
                    sampled_onetime_datadict_arr,
                    self._merged_onetime_datadict_keys)

        inputs = AttrDict()
        outputs = AttrDict()

        names_set = set(names_to_get)
        for key in names_set.intersection(
                self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names):
            inputs[key] = sampled_datadict[key]
        for key in names_set.intersection(self._env_spec.output_observation_names + self._env_spec.output_goal_names):
            outputs[key] = sampled_datadict[key]
        # params are inputs because they are known ahead of time
        for key in names_set.intersection(self._env_spec.param_names):
            inputs[key] = sampled_onetime_datadict[key]

        # put missing names in inputs by default.
        extra_names = names_set.difference(inputs.list_leaf_keys() + outputs.list_leaf_keys())
        for key in extra_names:
            inputs[key] = sampled_datadict >> key

        # default should be use as inputs
        src = inputs if self._final_names_as_inputs else outputs

        # finals are outputs because they are known only after the episode is over
        for key in names_set.intersection(self._env_spec.final_names):
            src[key] = sampled_onetime_datadict[key]

        if 'rollout_timestep' in sampled_datadict.leaf_keys():
            inputs['rollout_timestep'] = sampled_datadict['rollout_timestep']

        if is_torch:
            outputs[self._done_key] = sampled_datadict[self._done_key].to(dtype=bool)
        else:
            outputs[self._done_key] = sampled_datadict[self._done_key].astype(bool)

        if do_padding:
            assert not inputs.has_leaf_key('padding_mask')
            if is_torch:
                inputs.padding_mask = torch.zeros(local_batch_size, horizon, dtype=torch.bool,
                                                  device=inputs[self._done_key].device)
            else:
                inputs.padding_mask = np.zeros((local_batch_size, horizon), dtype=bool)

            # basically, this tells us which steps are padded. if any
            for i, clen in enumerate(chunk_lengths):
                inputs.padding_mask[i, clen:] = True

            # logger.debug(f"\n{inputs.leaf_apply(lambda arr: arr.dtype).pprint(ret_string=True)}")
            # logger.debug(f"\n{outputs.leaf_apply(lambda arr: arr.dtype).pprint(ret_string=True)}")
            # logger.debug(f"Padded to lengths:\n{chunk_lengths}")
            # logger.debug(f"Pad Mask SUM:\n{(~inputs.padding_mask).sum(-1)}")

        return inputs, outputs  # each shape is (batch, horizon, name_dim...)

    def __iter__(self):
        while True:
            yield self._get_batch()

    # @abstract.overrides
    def get_batch(self, indices=None, torch_device=None, min_idx=0, max_idx=0, do_async=True, local_batch_size=None,
                  **kwargs):
        """
        :param indices: ONE STEP BEHIND if asynchronous
        :param torch_device:
        :param min_idx: ONE STEP BEHIND
        :param max_idx: ONE STEP BEHIND
        :param async: preload data after execution of get batch (i.e. load while training)
        :param kwargs:
        :return:
        """
        # directly get the batch
        if not do_async or not self._asynchronous_get_batch:
            return self._get_batch(indices=indices, torch_device=torch_device, min_idx=min_idx, max_idx=max_idx,
                                   non_blocking=False, local_batch_size=local_batch_size, **kwargs)
        else:
            if self._promised_inputs is None:
                res = self._get_batch(indices=indices, torch_device=torch_device, min_idx=min_idx,
                                      max_idx=max_idx, non_blocking=False, local_batch_size=local_batch_size, **kwargs)
                last_ins, last_outs = res[:2]
                last_meta = res[2] if len(res) == 3 else AttrDict()
            else:
                last_ins, last_outs, last_meta = self._promised_inputs, self._promised_outputs, self._promised_meta

            res = self._get_batch(indices=indices, torch_device=torch_device, local_batch_size=local_batch_size,
                                  min_idx=min_idx, max_idx=max_idx,
                                  non_blocking=True, **kwargs)
            self._promised_inputs, self._promised_outputs = res[:2]
            self._promised_meta = res[2] if len(res) == 3 else AttrDict()

            if last_meta.is_empty():
                return last_ins, last_outs
            else:
                return last_ins, last_outs, last_meta

    # get number of episodes to remove, etc given new L datapoints
    def get_overwrite_details(self, L):
        # if we are going to overwrite things, null any overwritten episodes
        num_removed_elements = 0
        overwrite_how_many_samples = 0
        overwrite_how_many_episodes = 0
        if L + self._data_len > self._capacity:
            overwrite_how_many_samples = (self._data_len + L) - self._capacity
            i = 0
            # first where we are overwriting no further than this episode
            for i in range(len(self._split_indices)):
                if self._split_indices[i] >= overwrite_how_many_samples:
                    break
            # prune split indices (these episodes do not exist (remove at least 1)
            num_removed_elements = int(self._split_indices[i])
            overwrite_how_many_episodes = i + 1

        return AttrDict(
            num_removed_elements=num_removed_elements,
            overwrite_how_many_samples=overwrite_how_many_samples,
            overwrite_how_many_episodes=overwrite_how_many_episodes,
        )

    def add_episode(self, inputs, outputs, **kwargs):
        if self._frozen:
            logger.warn("Dataset is frozen but add_episode() was called! Nothing will happen")
            return

        # fill in rollout steps if not there.
        L = (outputs >> self._done_key).shape[0]
        if not inputs.has_leaf_key('rollout_timestep'):
            inputs.rollout_timestep = np.arange(L)

        # other preprocessing for data loading
        joint = inputs & outputs

        # checking that shapes of keys are good.
        for key in self._step_names:
            assert (joint >> key).shape[0] == L, "%s, shape: %s, expected: %d" % (key, joint[key].shape, L)
        for key in self._onetime_names:
            assert (joint >> key).shape[0] >= 1, "%s, shape: %s, expected: 1" % (key, joint[key].shape)

        # filtering into step names and onetime names
        local_dict = joint > self._step_names
        onetime_dict = (joint > self._env_spec.param_names).leaf_apply(lambda arr: arr[:1]) \
                       & (joint > self._env_spec.final_names).leaf_apply(lambda arr: arr[-1:])

        # preprocess before adding
        for data_preprocessor in self._data_preprocessors:
            local_dict, onetime_dict, _ = data_preprocessor.forward(self, local_dict, onetime_dict, np.array([L]))

        # compute where to add in static dataset
        to_add_idxs = np.arange(self._add_index, self._add_index + L)
        to_add_idxs = (to_add_idxs % self._capacity)

        # compute where to add in dynamic dataset
        od = self.get_overwrite_details(L)
        self._split_indices = self._split_indices[od.overwrite_how_many_episodes:] - od.num_removed_elements

        for key in self._onetime_names:
            self._onetime_datadict[key] = self._onetime_datadict[key][od.overwrite_how_many_episodes:]

        # we are deleting whole episodes from dynamic list
        if len(self._onetime_names) > 0:
            self._dynamic_data_len -= od.overwrite_how_many_episodes
            # capacity decreased by this amount.
            self._dynamic_capacity -= od.overwrite_how_many_episodes
            assert self._dynamic_data_len >= 0
            assert self._dynamic_capacity >= 0

        if self._data_len > 0:
            new_data_len = int(self._split_indices[-1]) + L  # how much are we keeping plus new data
        else:
            new_data_len = L
        self._split_indices = np.append(self._split_indices, new_data_len)  # new episode has this length

        # actually assign everything
        for key in self._step_names:
            self._datadict[key][to_add_idxs] = local_dict[key]

        # add to onetime (single entry)
        self._dynamic_add(self._onetime_datadict, onetime_dict)

        self._datadict[self._done_key][to_add_idxs] = outputs[self._done_key]

        # set these to the new positions
        self._data_len = new_data_len
        self._add_index = (self._add_index + L) % self._capacity

        # if the add index is looped and we removed new elements, make sure the delta is maintained
        if self.start_idx > self._add_index and od.num_removed_elements > 0:
            if (self.start_idx - self._add_index) != od.num_removed_elements - od.overwrite_how_many_samples:
                import ipdb;
                ipdb.set_trace()

    def add(self, inputs, outputs, rollout_step=0, **kwargs):

        if self._frozen:
            logger.warn("Dataset is frozen but add() was called! Nothing will happen")
            return

        # inputs are B x .. where B = 1
        od = self.get_overwrite_details(1)
        if od.overwrite_how_many_samples > 0:
            diff = self._data_len - self._split_indices[-1]  # how many elements are in the current ep so far.
            self._split_indices = self._split_indices[od.overwrite_how_many_episodes:] - od.num_removed_elements
            new_data_len = int(self._split_indices[-1] + diff) + 1  # how much are we keeping plus new data
            for key in self._onetime_names:
                self._onetime_datadict[key] = self._onetime_datadict[key][od.overwrite_how_many_episodes:]
            if len(self._onetime_names) > 0:
                self._dynamic_data_len -= od.overwrite_how_many_episodes
                assert self._dynamic_data_len >= 0, self._dynamic_data_len
        else:
            new_data_len = self._data_len + 1  # we must be below capacity if we aren't overwriting anything
            assert new_data_len <= self._capacity

        # add individual observations
        for key in self._env_spec.observation_names + self._env_spec.action_names:
            assert inputs[key].shape[0] == 1, key
            self._datadict[key][self._add_index] = inputs[key][0]
        for key in self._env_spec.output_observation_names:
            assert outputs[key].shape[0] == 1, key
            self._datadict[key][self._add_index] = outputs[key][0]

        # ONETIME stuff
        if rollout_step == 0:
            assert self._onetime_buffer.is_empty()
            for key in self._env_spec.param_names:  # param
                assert inputs[key].shape[0] == 1, key
                self._onetime_buffer[key] = inputs[key]
        okeys = list(outputs.leaf_keys())
        for key in self._env_spec.final_names:  # final
            if key in okeys:
                assert outputs[key].shape[0] == 1, key
                self._onetime_buffer[key] = outputs[key]  # override what was there, potentially
        # TODO add everything from buffer (see below)

        self._datadict[self._done_key][self._add_index] = outputs[self._done_key][0]
        self._datadict.rollout_timestep[self._add_index] = rollout_step

        # record the end of the new data here
        if (outputs >> self._done_key)[0]:
            self._split_indices = np.append(self._split_indices, new_data_len)  # new episode has this length
            # do data checks here to make sure we received the all the data we needed
            for key in self._onetime_names:
                assert key in self._onetime_buffer.leaf_keys(), key
            # TODO theres some bug here
            self._dynamic_add(self._onetime_datadict, self._onetime_buffer)
            self._onetime_buffer = AttrDict()

        # advance
        self._data_len = new_data_len
        self._add_index = (self._add_index + 1) % self._capacity

    def terminate_episode(self):
        assert self._data_len > 0, "No data at all!"
        assert len(self._split_indices) == 0 or self._split_indices[-1] < self._data_len, "No new data added.."
        assert len(self._onetime_names) == 0, "Onetime not supported for terminate yet!"
        self._split_indices = np.append(self._split_indices, self._data_len)
        self._datadict[self._done_key][(self._add_index - 1) % self._capacity] = True  # last added item
        # print(self._datadict.done)
        # print(self._split_indices)
        self._onetime_buffer = AttrDict()

    # don't call this function too often
    def reorder_underlying_data(self):
        # if we have non contiguous data, reorder underlying data to be "contiguous"
        # data:
        # [ add - - st - - - -]
        # import ipdb; ipdb.set_trace()
        if self.start_idx > 0:
            current_order = self.idx_map_to_real_idx(np.arange(self._data_len))
            for key in self._step_names:
                self._datadict[key][:self._data_len] = self._datadict[key][current_order]

            self._add_index = self._data_len % self._capacity

    # don't call this function too often
    def save(self, extra_save_data=AttrDict(), local=False, suffix=None, ep_range=None, **kwargs):
        # np.save(path, save_dict)
        output_file = self._output_file
        if suffix is not None:
            output_file = postpend_to_base_name(output_file, suffix)
        logger.debug("Saving NP dataset to %s" % output_file)
        logger.debug('-> Dataset length: {}'.format(self._data_len))
        # print((self._datadict >> "done")[self._add_index - 1])  # last
        # print((self._datadict >> "done")[self._split_indices[-1] - 1])
        self.reorder_underlying_data()
        # print((self._datadict >> "done")[self._split_indices[-1] - 1])

        full_episode_only_data_len = int(self._split_indices[-1])
        assert full_episode_only_data_len <= self._data_len

        # episodes
        start=0
        end=self._dynamic_data_len

        # steps
        ep_start=0 
        ep_end=full_episode_only_data_len
        
        if ep_range is not None:
            start, end = ep_range  # exclusive, start can wrap, but be careful wrapping end
            if start is None:
                start = 0
            if end is None:
                end = self._dynamic_data_len
            ep_start = np.concatenate([[0], self._split_indices[:-1]])[start]
            ep_end = self._split_indices[end - 1]
            logger.debug(f"-> Saving from Episode {start} - {end}, for steps {ep_start} - {ep_end}")

        smaller = dict()
        for name in self._step_names:
            smaller[name] = self._datadict[name][ep_start:ep_end]

        for name in self._onetime_names:
            smaller[name] = self._onetime_datadict[name][start:end]  # dynamic length is correct here

        assert smaller[self._done_key][-1], "TODO: Save must end with done == True"

        # allows for saving of extra data (e.g. user might pass in labels, something not in the spec)
        for name in extra_save_data.leaf_keys():
            smaller[name] = extra_save_data[name]

        logger.debug('-> Dataset keys: {}'.format(list(smaller.keys())))
        if local:
            path = output_file
            assert os.path.exists(os.path.dirname(output_file)), output_file
        else:
            path = file_path_with_default_dir(output_file, self._file_manager.exp_dir)
        np.savez_compressed(path, **smaller)

    def __len__(self):
        return self._data_len

    def split_indices(self):
        return self._split_indices

    def get_datadict(self) -> AttrDict:
        # don't call this too often
        current_order = self.idx_map_to_real_idx(np.arange(self._data_len))
        return self._datadict.leaf_apply(lambda arr: arr[current_order])

    def get_onetime_datadict(self) -> AttrDict:
        # don't call this too often
        return self._onetime_datadict.leaf_apply(lambda arr: arr[:self._dynamic_data_len])

    def idx_map_to_real_idx(self, idxs):
        wrapped_contiguous_idxs = np.asarray(idxs) % max(self._data_len, 1)
        if self.start_idx == 0:
            return wrapped_contiguous_idxs
        else:
            return (self.start_idx + wrapped_contiguous_idxs) % self._capacity

    # wraps from the starting point (0 is the oldest point, -1 is the newest) in all directions
    def get(self, name, idxs):
        true_idxs = self.idx_map_to_real_idx(idxs)
        return self._datadict[name][true_idxs]

    # wraps from the starting point (0 is the oldest point, -1 is the newest) in all directions
    def set(self, name, idxs, values):
        true_idxs = self.idx_map_to_real_idx(idxs)
        assert len(true_idxs) == len(values)
        self._datadict[name][true_idxs] = values

    def reset(self):
        self._datadict, self._onetime_datadict, self._split_indices = self._load_np()
        self._onetime_buffer = AttrDict()

    def create_save_dir(self):
        file_path = os.path.join(self._file_manager.exp_dir, self._output_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def __delete__(self, instance):
        if self._is_shared:
            # snp close
            self._datadict.leaf_apply(lambda arr: arr.close())


if __name__ == '__main__':

    params = AttrDict(
        file=None,
        output_file="test_save.npz",
        save_every_n_steps=0,  # not used for horizon 0
        horizon=1,
        capacity=30,
        batch_size=5,
    )

    env_spec = ParamEnvSpec(AttrDict(
        names_shapes_limits_dtypes=[
            ('obs', (3, 5), (-np.inf, np.inf), np.float32),
            ('next_obs', (3, 5), (-np.inf, np.inf), np.float32),

            ('reward', (1,), (-np.inf, np.inf), np.float32),

            ('action', (4,), (-1, 1), np.float32),
        ],
        output_observation_names=['next_obs', 'reward'],
        observation_names=['obs'],
        action_names=['action'],
        goal_names=[],
    ))

    file_manager = ExperimentFileManager("test_stuff",
                                         is_continue=os.path.exists(
                                             os.path.join(ExperimentFileManager.experiments_dir, "test_stuff")),
                                         log_fname='log_train.txt',
                                         config_fname=None)

    dataset = NpDataset(params, env_spec, file_manager)

    ###### TEST ADD SEQUENTIAL #######

    obs_set = np.ones((41, 3, 5)).astype(np.float32)
    obs_set = obs_set.cumsum(axis=0)
    dones_set = np.zeros(40).astype(bool)
    dones_set[9] = True
    dones_set[19] = True
    dones_set[29] = True
    dones_set[39] = True
    reward_set = np.arange(40).reshape((40, 1)).astype(np.float32)
    action_set = -np.ones((40, 4)).astype(np.float32)
    action_set = action_set.cumsum(axis=0)

    for i in range(30):
        inputs = AttrDict(
            obs=obs_set[i:i + 1],
            action=action_set[i:i + 1],
        )
        outputs = AttrDict(
            next_obs=obs_set[i + 1:i + 2],
            done=dones_set[i:i + 1],
            reward=reward_set[i:i + 1],
        )

        dataset.add(inputs, outputs, rollout_step=i % 10)
        # print("i = %d" % i, dataset.get("obs", i))

    assert 30 == len(dataset)
    assert (dataset.split_indices() == np.array([10, 20, 30])).all(), dataset.split_indices()

    # sample
    inputs, outputs = dataset.get_batch()
    assert inputs.obs.shape == (5, 1, 3, 5)
    assert inputs.action.shape == (5, 1, 4)
    assert outputs.next_obs.shape == (5, 1, 3, 5)
    assert outputs.reward.shape == (5, 1, 1)
    assert outputs.done.shape == (5, 1)

    # passed in idxs
    idxs = np.array([2, 3, 4, 5, 6])
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert np.allclose(inputs.obs, obs_set[idxs, None]), [inputs.obs, obs_set[idxs, None]]
    assert np.allclose(inputs.action, action_set[idxs, None])
    assert np.allclose(outputs.next_obs, obs_set[idxs + 1, None])
    assert np.allclose(outputs.reward, reward_set[idxs, None])
    assert np.allclose(outputs.done, dones_set[idxs, None])

    ###### TESTING OVERWRITING #######

    inputs = AttrDict(
        obs=obs_set[30:31],
        action=action_set[30:31],
    )
    outputs = AttrDict(
        next_obs=obs_set[31:32],
        done=dones_set[30:31],
        reward=reward_set[30:31],
    )

    dataset.add(inputs, outputs, rollout_step=0)

    assert (dataset.split_indices() == np.array([10, 20])).all()
    assert len(dataset) == 21
    assert dataset.start_idx == 10
    assert dataset._add_index == 1

    for i in range(31, 40):
        inputs = AttrDict(
            obs=obs_set[i:i + 1],
            action=action_set[i:i + 1],
        )
        outputs = AttrDict(
            next_obs=obs_set[i + 1:i + 2],
            done=dones_set[i:i + 1],
            reward=reward_set[i:i + 1],
        )

        dataset.add(inputs, outputs, rollout_step=i - 30)

    assert (dataset.split_indices() == np.array([10, 20, 30])).all()
    assert len(dataset) == 30
    assert dataset.start_idx == 10
    assert dataset._add_index == 10

    idxs = np.array([2, 3, 4, 5, 6, 7])
    real_idxs = idxs + 10
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    ###### REORDER #######

    idxs = np.arange(30)
    # idxs 0 -> 9 are real 10->19
    # idxs 10 -> 19 are real 20->29
    # idxs 20 -> 29 are real 30->39
    real_idxs = np.arange(10, 40)
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    dataset.reorder_underlying_data()
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    assert len(dataset) == 30
    assert dataset.start_idx == 0
    assert dataset._add_index == 0

    ###### REORDER: checking add episode ######

    inputs = AttrDict(
        obs=obs_set[:10],
        action=action_set[:10],
    )
    outputs = AttrDict(
        next_obs=obs_set[1:11],
        done=dones_set[:10],
        reward=reward_set[:10],
    )

    dataset.add_episode(inputs, outputs)

    assert (dataset.split_indices() == np.array([10, 20, 30])).all()
    assert len(dataset) == 30
    assert dataset.start_idx == 10
    assert dataset._add_index == 10

    idxs = np.arange(30)
    # 10->19 are idxs 0 -> 9 are real 20->29
    # 20->29 are idxs 10 -> 19 are real 30->39
    # 0->9 are idxs 20 -> 29 are real 0->9
    real_idxs = np.concatenate([np.arange(20, 40), np.arange(10)])
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    ###### REORDER partial ######

    inputs = AttrDict(
        obs=obs_set[10:11],
        action=action_set[10:11],
    )
    outputs = AttrDict(
        next_obs=obs_set[11:12],
        done=dones_set[10:11],
        reward=reward_set[10:11],
    )

    dataset.add(inputs, outputs, rollout_step=0)

    idxs = np.arange(21)
    # 20->29 are idxs 0 -> 9 are real 30->39
    # 0->9 are idxs 10 -> 19 are real 0->9
    # 10->11 are idxs 20 -> 21 are real 10->11
    real_idxs = np.concatenate([np.arange(30, 40), np.arange(11)])
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    dataset.reorder_underlying_data()
    inputs, outputs = dataset.get_batch(indices=idxs)
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs + 1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    assert len(dataset) == 21
    assert dataset._add_index == 21
    assert dataset.start_idx == 0

    ###### SAVE & LOAD ######
    dataset.save()

    params.file = "test_save.npz"
    dataset2 = NpDataset(params, env_spec, file_manager)

    assert len(dataset) == len(dataset2) + 1
    assert dataset2._add_index == 20
    assert (dataset.split_indices() == np.array([10, 20])).all()

    for name in dataset._step_names:
        assert (dataset._datadict[name][:len(dataset2)] == dataset2._datadict[name][
                                                           :len(dataset2)]).all(), "%s does not match" % name
