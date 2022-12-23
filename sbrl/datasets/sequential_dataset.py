"""
Holds an underlying data type in an episodic manner, split into

"""

import os

import numpy as np
import torch

from sbrl.datasets.dataset import Dataset
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit, get_required
from sbrl.utils.torch_utils import split_dim_np, split_dim, broadcast_dims, broadcast_dims_np


# TODO
class SequentialDataset(Dataset):

    # # @abstract.overrides
    def _init_params_to_attrs(self, params):
        super(SequentialDataset, self)._init_params_to_attrs(params)
        self._input_files = get_with_default(params, "file", None)  # None if we are starting a new file, o/w where we load from (can be a list)
        self._output_file = get_required(params, "output_file")  # Cannot be none, where we save data to

        # set to inf to keep recording everything
        # max length of dataset (transitions) before overriding (useful for replay buffer)
        self._capacity = int(get_required(params, "capacity"))
        assert self._capacity < np.inf

        self._done_key = get_with_default(params, "done_key", "done", map_fn=str)

        self._dynamic_capacity = 0  # this type of data grows dynamically as needed
        self._dynamic_data_len = 0

        self._promised_inputs = None
        self._promised_outputs = None

        self._use_rollout_steps = get_with_default(params, "use_rollout_steps", True)
        self._mmap_mode = get_with_default(params, "mmap_mode", None)
        if self._mmap_mode is not None:
            logger.info("Dataset using mmap mode: %s" % self._mmap_mode)

        self._step_names = self._env_spec.names + [self._done_key]
        if self._use_rollout_steps:
            self._step_names.append('rollout_timestep')

        self._onetime_names = self._env_spec.param_names + self._env_spec.final_names

        assert len(self._step_names + self._onetime_names) == len(list(set(self._step_names + self._onetime_names))), \
            "non unique names in: %s" % (self._step_names + self._onetime_names)

        self._allow_missing = get_with_default(params, "allow_missing", False)

        # no duplicates!!
        assert len(set(self._step_names)) == len(self._step_names)

        self._data_len = 0  # number of filled elements
        self._add_index = 0  # add index

        self._asynchronous_get_batch = get_with_default(params, "asynchronous_get_batch", False)

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

    @property
    def start_idx(self):
        # underlying
        return (self._add_index - self._data_len) % self._capacity

    # @abstract.overrides
    def _init_setup(self):
        self._datadict, self._onetime_datadict, self._split_indices = self._load_np()
        self._onetime_buffer = AttrDict()

    def _init_empty(self, local_dict: AttrDict, onetime_dict):
        self._data_len = 0  # number of filled elements
        self._add_index = 0  # add index

        # static allocation prevents failures of memory!
        for key in self._step_names:
            local_dict[key] = np.empty((self._capacity, *self._env_spec.names_to_shapes[key]),
                                       dtype=self._env_spec.names_to_dtypes[key])
        # these names are dynamically allocated since it's just once per episode
        for key in self._onetime_names:
            onetime_dict[key] = np.empty([0] + list(self._env_spec.names_to_shapes[key]), dtype=self._env_spec.names_to_dtypes[key])  # empty to begin
        split_indices = np.array([])

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
        for file in self._input_files:
            i += 1
            logger.debug('Loading ' + file)
            if os.path.isfile(file):
                path = file
            else:
                path = os.path.join(self._file_manager.exp_dir, file)
                # fail safe (invalid dataset passed in)
                if not os.path.isfile(path):
                    logger.warn("Could not find dataset: %s" % file)
                    return self._init_empty(local_dict, onetime_dict)

            # load the dictionary from memory
            datadict = np.load(path, allow_pickle=True, mmap_mode=self._mmap_mode)  # , mmap_mode="r")

            # this is the number of new elements to add
            print("keys: " + str(list(datadict.keys())))
            new_data_len = len(datadict[self._done_key])
            # make sure the dataset ends with a 'done' flag
            assert datadict[self._done_key][-1], "Dataset %s must end with done == True" % file

            if self._data_len + new_data_len > self._capacity:
                logger.warn("Data exceeds the capacity in allocated storage, increase capacity param in NpDataset")
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

            # assign each key of step sample keys in our local dictionary based on the np input
            for key in self._step_names:
                assert key in datadict, '%s not in np file' % key
                assert len(datadict[key]) == new_data_len, [key, len(datadict[key]), new_data_len]
                # fill in the chunk of the data set
                try:
                    local_dict[key][self._data_len:self._data_len + new_data_len] = datadict[key]
                except ValueError as e:
                    logger.debug([e, key])
                    raise e

            # assign each key of one-per-episode keys in our local dictionary based on the np input
            for key in self._onetime_names:
                assert key in datadict, '%s not in np file' % key
                # assert len(datadict[key])
                # fill in the chunk of the data set
                onetime_dict[key] = np.concatenate([onetime_dict[key], datadict[key]], axis=0)

            self._data_len += new_data_len
            logger.debug('Dataset \"{}\" length: {}'.format(file, new_data_len))

        # if no elements, this is []
        # if N >= 1 episodes, this has [L1, L1+L2, ... L1+.+LN] (human ordering is maintained
        #  regardless of wrapping)
        split_indices = np.where(local_dict[self._done_key][:self._data_len])[0] + 1  # one after each episode ends
        # make sure the number of episodes is consistent in dataset
        for key in self._onetime_names:
            assert len(onetime_dict[key]) == len(split_indices), "Key[%s] needs [%d] samples, had [%d]" % (key, len(onetime_dict[key]), len(split_indices))
            self._dynamic_capacity = len(onetime_dict[key])
            self._dynamic_data_len = self._dynamic_capacity  # to start, the dynamic array is full

        # NOTE: we assumes all datasets have been stored sequentially!
        if self._data_len == self._capacity:
            self._add_index = 0
        else:
            self._add_index = self._data_len

        logger.debug('Dataset Total length: {}'.format(self._data_len))
        logger.debug('Dataset Starting Idx: {}'.format(self._add_index))
        logger.debug('Dataset Number of Eps: {}'.format(len(split_indices)))

        return local_dict, onetime_dict, split_indices

    def get_statistics(self, names):
        assert len(self._datadict) > 0, "Empty datadict"
        filtered = self.get_datadict().leaf_filter_keys(names)  # be careful with this fn call, it computes full data ordering to retrieve the data properly, O(N)
        filtered.combine(self.get_onetime_datadict().leaf_filter_keys(names))
        means = filtered.leaf_apply(lambda arr: np.mean(arr, axis=0))
        means.leaf_modify(lambda arr: np.where(np.isnan(arr), 0, arr))
        stds = filtered.leaf_apply(lambda arr: np.std(arr, axis=0))
        stds.leaf_modify(lambda arr: np.where(np.logical_or(np.isnan(arr), arr == 0), 1, arr))  # stdev should not be zero or inf

        out = AttrDict(mean=means, std=stds)
        return out

    def _dynamic_expand(self, datadict, old_size, new_size):
        for key in datadict.leaf_keys():
            data = datadict[key]
            assert new_size > len(data) >= old_size, "%s: %d / %d / %d" % (key, new_size, len(data), old_size)
            new_data = np.empty((new_size, *data.shape[1:]), dtype=data.dtype)
            new_data[:old_size] = data[:old_size]
            datadict[key] = new_data

    def _dynamic_add(self, datadict, new_datadict):
        # check keys are all there (add all at once)
        new_part_len = 0
        for key in datadict.leaf_keys():
            assert key in new_datadict.leaf_keys(), key
            assert new_part_len == 0 or new_datadict[key].shape[0] == new_part_len, "%s has length %d instead of %d" % (key, new_datadict[key].shape[0], new_part_len)
            new_part_len = new_datadict[key].shape[0]

        # figure out new capacity
        new_len = self._dynamic_data_len + new_part_len  # how much will we have
        copy_over = new_len > self._dynamic_capacity
        while new_len > self._dynamic_capacity:
            self._dynamic_capacity = self._dynamic_capacity * 2 + 1  # 1 for good measure, since we start with 0
        if copy_over:
            self._dynamic_expand(datadict, self._dynamic_data_len, self._dynamic_capacity)  # expands to capacity

        for key in datadict.leaf_keys():
            # copy new data in, since we know there is enough space now
            datadict[key][self._dynamic_data_len:new_len] = new_datadict[key]  # new data assignment

        self._dynamic_data_len = new_len  # the new amount of data <= capacity
        assert self._dynamic_data_len <= self._dynamic_capacity, "%d / %d" % (self._dynamic_data_len, self._dynamic_capacity)

    def episode_length(self, i):
        prev = self._split_indices[i - 1] if i > 0 else 0
        return self._split_indices[i] - prev  # length of episode i

    def get_num_episodes(self):
        return len(self._split_indices)

    # getting the indices for a chunk from episode i (human ordering)
    def sample_indices_for_episode(self, i, horizon=None):
        prev = self._split_indices[i - 1] if i > 0 else 0
        ep_len = self._split_indices[i] - prev  # length of episode i
        horizon = self._horizon if horizon is None else horizon  # allow user override

        if horizon > ep_len:
            raise NotImplementedError

        rg = max(ep_len - horizon + 1, 1)
        idx = np.random.choice(rg)  # pick a starting point
        return np.arange(prev + idx, prev + idx + horizon)

    def get_episode(self, i, names, split=False, torch_device=None, include_done=True, **kwargs):
        # sampled datadict has Ni samples, onetime has 1 sample per key
        names = names + [self._done_key] if include_done else names
        chunk_idxs = self.sample_indices_for_episode(i, horizon=self.episode_length(i))
        real_idxs = self.idx_map_to_real_idx(chunk_idxs)
        sampled_datadict = self._datadict.leaf_filter_keys(names).leaf_apply(lambda arr: arr[real_idxs])
        onetime_datadict = self._onetime_datadict.leaf_filter_keys(names).leaf_apply(lambda arr: arr[i:i+1])

        # ep_idxs is sparse (1,..) -> (H,..)
        if self._is_torch:
            onetime_datadict.leaf_modify(lambda arr: broadcast_dims(arr, [0], [len(real_idxs)]))
        else:
            onetime_datadict.leaf_modify(lambda arr: broadcast_dims_np(arr, [0], [len(real_idxs)]))

        sampled_datadict.combine(onetime_datadict)
        all_ds = []
        if split:
            inputs = AttrDict()
            outputs = AttrDict()
            intersect_in = set(self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names + self._env_spec.param_names)\
                .intersection(set(names))
            intersect_out = set(self._env_spec.output_observation_names + self._env_spec.final_names)\
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
        return np.searchsorted(self._split_indices, indices, side='right')  # split indices are the one past the last idx for that ep

    def _get_batch(self, indices=None, torch_device=None, min_idx=0, max_idx=0, non_blocking=False, **kwargs):
        assert self._data_len > 0
        # if no indices provided, sample randomly from [min_idx, max_idx),
        local_batch_size = self._batch_size
        # either sampling
        data_length = self._data_len if self._horizon == 1 else self._split_indices.size
        if indices is None:
            # indices stop at max (exclusive), <= 0, wrap backwards from end
            # TODO current samples eating in?
            if max_idx <= 0:
                max_idx += data_length
            assert 0 <= min_idx < max_idx <= data_length
            indices = np.random.choice(max_idx - min_idx, self._batch_size,
                                       replace=max_idx - min_idx < self._batch_size)
            indices += min_idx  # base index to consider in dataset
        else:
            local_batch_size = indices.shape[0]
            assert np.max(indices) < data_length and np.min(indices) >= 0

        if self._horizon > 1:
            # all the contiguous episode indices are mapped contiguous horizon ranges:
            episode_indices = indices
            indices = np.concatenate([self.sample_indices_for_episode(i) for i in indices])  # all the sample idxs
        else:
            episode_indices = self.get_ep_indices_for_step_indices(indices)  # all the episode idxs

        assert len(indices) == local_batch_size * self._horizon, "contiguous indices should be of size B*H in get_batch"

        # first parse the keys we care about
        names_to_get = self._env_spec.all_names + [self._done_key] + \
                       ([] if self._use_rollout_steps else ["rollout_timesteps"])

        with timeit("get_batch/indexing"):
            dd = self._datadict.leaf_filter_keys(names_to_get)
            true_indices = self.idx_map_to_real_idx(indices)
            # each is (batch, horizon, name...)
            if self._is_torch:
                true_indices = torch.from_numpy(true_indices).to(self._torchify_device)
                sampled_datadict = dd.leaf_apply(lambda arr: split_dim(torch_in=torch.index_select(arr, 0, true_indices), dim=0,
                                                                       new_shape=(
                                                                           local_batch_size, self._horizon)))
            else:
                sampled_datadict = dd.leaf_apply(lambda arr: split_dim_np(np_in=arr[true_indices], axis=0,
                                                                          new_shape=(
                                                                              local_batch_size, self._horizon)))

        with timeit("get_batch/onetime"):
            od = self._onetime_datadict.leaf_filter_keys(names_to_get)

            sampled_onetime_datadict = od.leaf_apply(lambda arr: arr[episode_indices, None])
            if self._horizon > 1:
                if self._is_torch:
                    sampled_onetime_datadict.leaf_modify(lambda arr: broadcast_dims(arr, [1], [self._horizon]))
                else:
                    sampled_onetime_datadict.leaf_modify(lambda arr: broadcast_dims_np(arr, [1], [self._horizon]))

            sampled_datadict.combine(sampled_onetime_datadict)

        inputs = AttrDict()
        outputs = AttrDict()
        for key in self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names:
            inputs[key] = sampled_datadict[key]
        for key in self._env_spec.output_observation_names:
            outputs[key] = sampled_datadict[key]

        # params are inputs because they are known ahead of time
        for key in self._env_spec.param_names:
            inputs[key] = sampled_onetime_datadict[key]
        # finals are outputs because they are known only after the episode is over
        for key in self._env_spec.final_names:
            outputs[key] = sampled_onetime_datadict[key]

        with timeit("get_batch/to_device"):
            if self._is_torch:
                outputs[self._done_key] = sampled_datadict[self._done_key].to(dtype=bool)
            else:
                outputs[self._done_key] = sampled_datadict[self._done_key].astype(bool)

            if torch_device is not None:
                for d in (inputs, outputs):
                    if self._is_torch:
                        d.leaf_modify(lambda x: x.to(torch_device, non_blocking=non_blocking))
                    else:
                        d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device, non_blocking=non_blocking))

        return inputs, outputs  # each shape is (batch, horizon, name_dim...)

    # @abstract.overrides
    def get_batch(self, indices=None, torch_device=None, min_idx=0, max_idx=0, async=True, **kwargs):
        """
        :param indices: ONE STEP BEHIND if asynchronous
        :param torch_device:
        :param min_idx: ONE STEP BEHIND
        :param max_idx: ONE STEP BEHIND
        :param async: preload data after execution of get batch (i.e. load while training)
        :param kwargs:
        :return:
        """
        if not async or not self._asynchronous_get_batch:
            return self._get_batch(indices=indices, torch_device=torch_device, min_idx=min_idx, max_idx=max_idx, non_blocking=False, **kwargs)
        else:
            if self._promised_inputs is None:
                last_ins, last_outs = self._get_batch(indices=indices, torch_device=torch_device, min_idx=min_idx, max_idx=max_idx, non_blocking=False, **kwargs)
            else:
                last_ins, last_outs = self._promised_inputs, self._promised_outputs

            self._promised_inputs, self._promised_outputs = self._get_batch(indices=indices, torch_device=torch_device, min_idx=min_idx, max_idx=max_idx, non_blocking=True, **kwargs)
            return last_ins, last_outs

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
        # "allocate" the space
        L = outputs[self._done_key].shape[0]
        to_add_idxs = np.arange(self._add_index, self._add_index + L)
        to_add_idxs = (to_add_idxs % self._capacity)

        od = self.get_overwrite_details(L)
        self._split_indices = self._split_indices[od.overwrite_how_many_episodes:] - od.num_removed_elements
        for key in self._onetime_names:
            self._onetime_datadict[key] = self._onetime_datadict[key][od.overwrite_how_many_episodes:]
        self._dynamic_data_len -= od.overwrite_how_many_episodes
        assert self._dynamic_data_len >= 0

        if self._data_len > 0:
            new_data_len = int(self._split_indices[-1]) + L  # how much are we keeping plus new data
        else:
            new_data_len = L
        self._split_indices = np.append(self._split_indices, new_data_len)  # new episode has this length

        # actually assign everything
        for key in self._env_spec.observation_names + self._env_spec.action_names:
            assert inputs[key].shape[0] == L, "%s, shape: %s" % (key, inputs[key].shape)
            self._datadict[key][to_add_idxs] = inputs[key]

        for key in self._env_spec.output_observation_names:
            assert outputs[key].shape[0] == L, "%s, shape: %s" % (key, outputs[key].shape)
            self._datadict[key][to_add_idxs] = outputs[key]

        # get the first input
        to_add_to_onetime = AttrDict()
        for key in self._env_spec.param_names:
            assert inputs[key].shape[0] >= 1, "%s, shape: %s" % (key, inputs[key].shape)
            # self._onetime_datadict[key] = np.concatenate([self._onetime_datadict[key], inputs[key][:1]], axis=0)
            to_add_to_onetime[key] = inputs[key][:1]

        # get the last output
        for key in self._env_spec.final_names:
            assert outputs[key].shape[0] >= 1, "%s, shape: %s" % (key, outputs[key].shape)
            # self._onetime_datadict[key] = np.concatenate([self._onetime_datadict[key], outputs[key][-1:]], axis=0)
            to_add_to_onetime[key] = outputs[key][-1:]

        # add to onetime (single entry)
        self._dynamic_add(self._onetime_datadict, to_add_to_onetime)

        self._datadict[self._done_key][to_add_idxs] = outputs[self._done_key]
        self._datadict.rollout_timestep[to_add_idxs] = np.arange(L)

        # set these to the new positions
        self._data_len = new_data_len
        self._add_index = (self._add_index + L) % self._capacity

        # if the add index is looped
        if self.start_idx > self._add_index:
            assert (self.start_idx - self._add_index) == od.num_removed_elements

    def add(self, inputs, outputs, rollout_step=0, **kwargs):

        # inputs are B x .. where B = 1
        od = self.get_overwrite_details(1)
        if od.overwrite_how_many_samples > 0:
            self._split_indices = self._split_indices[od.overwrite_how_many_episodes:] - od.num_removed_elements
            new_data_len = int(self._split_indices[-1]) + 1  # how much are we keeping plus new data
            for key in self._onetime_names:
                self._onetime_datadict[key] = self._onetime_datadict[key][od.overwrite_how_many_episodes:]
            self._dynamic_data_len -= od.overwrite_how_many_episodes
            assert self._dynamic_data_len >= 0
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
        if outputs[self._done_key][0]:
            self._split_indices = np.append(self._split_indices, new_data_len)  # new episode has this length
            # do data checks here to make sure we received the all the data we needed
            for key in self._onetime_names:
                assert key in self._onetime_buffer.leaf_keys(), key
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
        if self.start_idx > 0:
            current_order = self.idx_map_to_real_idx(np.arange(self._data_len))
            for key in self._step_names:
                self._datadict[key][:self._data_len] = self._datadict[key][current_order]

            self._add_index = self._data_len % self._capacity

    # don't call this function too often
    def save(self, extra_save_data=AttrDict(), **kwargs):
        # np.save(path, save_dict)
        logger.debug("Saving NP dataset to %s" % self._output_file)
        logger.debug('-> Dataset length: {}'.format(self._data_len))
        self.reorder_underlying_data()

        full_episode_only_data_len = int(self._split_indices[-1])
        assert full_episode_only_data_len <= self._data_len
        smaller = dict()
        for name in self._step_names:
            smaller[name] = self._datadict[name][:full_episode_only_data_len]

        for name in self._onetime_names:
            smaller[name] = self._onetime_datadict[name][:self._dynamic_data_len]  # dynamic length is correct here

        assert smaller[self._done_key][-1], "TODO: Save must end with done == True"

        # allows for saving of extra data (e.g. user might pass in labels, something not in the spec)
        for name in extra_save_data.leaf_keys():
            smaller[name] = extra_save_data[name]

        path = os.path.join(self._file_manager.exp_dir, self._output_file)
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
        wrapped_contiguous_idxs = np.array(idxs) % max(self._data_len, 1)
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


# TODO unit tests


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
                                         is_continue=os.path.exists(os.path.join(ExperimentFileManager.experiments_dir, "test_stuff")),
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
    inputs, outputs = dataset.get_batch()[:2]
    assert inputs.obs.shape == (5, 1, 3, 5)
    assert inputs.action.shape == (5, 1, 4)
    assert outputs.next_obs.shape == (5, 1, 3, 5)
    assert outputs.reward.shape == (5, 1, 1)
    assert outputs.done.shape == (5, 1)

    # passed in idxs
    idxs = np.array([2, 3, 4, 5, 6])
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[idxs, None]).all()
    assert (inputs.action == action_set[idxs, None]).all()
    assert (outputs.next_obs == obs_set[idxs+1, None]).all()
    assert (outputs.reward == reward_set[idxs, None]).all()
    assert (outputs.done == dones_set[idxs, None]).all()

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

        dataset.add(inputs, outputs, rollout_step=i-30)

    assert (dataset.split_indices() == np.array([10, 20, 30])).all()
    assert len(dataset) == 30
    assert dataset.start_idx == 10
    assert dataset._add_index == 10

    idxs = np.array([2, 3, 4, 5, 6, 7])
    real_idxs = idxs + 10
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    ###### REORDER #######

    idxs = np.arange(30)
    # idxs 0 -> 9 are real 10->19
    # idxs 10 -> 19 are real 20->29
    # idxs 20 -> 29 are real 30->39
    real_idxs = np.arange(10, 40)
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    dataset.reorder_underlying_data()
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
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
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
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
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
    assert (outputs.reward == reward_set[real_idxs, None]).all()
    assert (outputs.done == dones_set[real_idxs, None]).all()

    dataset.reorder_underlying_data()
    inputs, outputs = dataset.get_batch(indices=idxs)[:2]
    assert (inputs.obs == obs_set[real_idxs, None]).all()
    assert (inputs.action == action_set[real_idxs, None]).all()
    assert (outputs.next_obs == obs_set[real_idxs+1, None]).all()
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
        assert (dataset._datadict[name][:len(dataset2)] == dataset2._datadict[name][:len(dataset2)]).all(), "%s does not match" % name
