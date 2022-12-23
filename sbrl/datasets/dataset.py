import abc
from typing import Sized

import numpy as np
import torch.utils.data as TD

from sbrl.envs.env_spec import EnvSpec
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_with_default


class Dataset(abc.ABC, Sized, TD.IterableDataset):

    def __init__(self, params, env_spec, file_manager):
        assert isinstance(params, AttrDict)
        assert isinstance(env_spec, EnvSpec)
        self._file_manager = file_manager
        self._env_spec = env_spec
        self._params = params

        self._is_torch = False
        self._is_shared = False

        # Note: H = 1 reduces to sampling transitions.
        self._batch_size = int(params >> "batch_size")  # number of episodes per batch
        self._horizon = int(params >> "horizon")  # number of steps per episode per batch

        self._init_params_to_attrs(params)

        assert self._batch_size >= 1
        assert self._horizon >= 1

        self._init_setup()

    def torchify(self, device):
        """
        Moves underlying dataset to torch (for better speeds possibly).
        :param device: to store torch data
        :return: nothing, but should set _is_torch = True
        """
        raise NotImplementedError

    def share_memory(self):
        """
        Moves underlying dataset to shared memory.
        :return: nothing, will set _is_shared to True
        """
        raise NotImplementedError

    def _init_params_to_attrs(self, params):
        self._sampler_config = get_with_default(params, "sampler", AttrDict(cls=Sampler, params=AttrDict()))
        self._sampler_cls = self._sampler_config >> "cls"
        self._sampler_prms = self._sampler_config >> "params"

    def _init_setup(self):
        pass

    def get_batch(self, indices=None, torch_device=None, **kwargs):
        """
        :param torch_device:
        :param indices: iterable indices list (N,) in sequential array format
        :return inputs: (AttrDict) each np array value is (B x H x ...)
        :return outputs: (AttrDict) each np array value is (B x H x ...)
        :return meta: (AttrDict) non standard shapes. could be batched. computed from data usually.
        """
        raise NotImplementedError

    def get_episode(self, i, names, split=True, torch_device=None, **kwargs):
        raise NotImplementedError

    def idx_map_to_real_idx(self, idxs):
        """
        :param idxs: iterable indices list (N,) in sequential array format [0...len(dataset))
        :return real_idxs: iterable indices list (N,) of idxs in underlying representation
        """
        raise NotImplementedError

    def get(self, name, idxs):
        """
        :param name: key in DataSet
        :param idxs: iterable indices list (N,)

        :return array: np array of size (N, obs_dim ...)
        """
        raise NotImplementedError

    def set(self, name, idxs, values):
        """
        :param name: key in DataSet
        :param idxs: iterable indices list (N,)
        :param values: iterable indices list (N, obs_dim ...)
        """
        raise NotImplementedError

    def add(self, inputs, outputs, **kwargs):
        """
        Adding a single transition to data set. B = 1

        :param inputs: (AttrDict) each np array value is (B x ...)
        :param outputs: (AttrDict) each np array value is (B x ...)
        TODO meta
        """
        raise NotImplementedError

    def add_episode(self, inputs, outputs, **kwargs):
        """
        Adding an entire episode (length L) to the data set, only required
        for some algorithms. Cannot do this batched.

        :param inputs: (AttrDict) each np array value is (L x ...)
        :param outputs: (AttrDict) each np array value is (L x ...)
        TODO meta
        """
        raise NotImplementedError

    def reset(self):
        """
        Return data set to its "original" state (or reset in some other manner).
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def horizon(self):
        return self._horizon

    @property
    def params(self) -> AttrDict:
        return self._params.leaf_copy()

    def get_statistics(self, names):
        raise NotImplementedError

    def get_num_episodes(self):
        raise NotImplementedError

    def get_num_periods(self):
        # e.g., interactions
        return self.get_num_episodes()

    def episode_length(self, i):
        raise NotImplementedError

    def period_length(self, i):
        raise NotImplementedError

    def period_weights(self, indices=None):
        raise NotImplementedError

    def create_save_dir(self):
        raise NotImplementedError

    def save(self, fname=None, ep_range=None, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    @property
    def save_dir(self):
        return None

    @property
    def sampler(self):
        # creates a new sampler and returns it
        return self._sampler_cls(self, self._sampler_prms)

class Sampler:
    def __init__(self, dataset, params):
        self._ds = dataset

    def get_indices(self, **kwargs):
        return None


class SequentialSampler(Sampler):
    def __init__(self, dataset, params):
        super(SequentialSampler, self).__init__(dataset, params)
        self._curr_idx = None
        self._bs = self._ds.batch_size
        self._shuffle = get_with_default(params, "shuffle", True)
        self._order = None
        self._num_batches = None
        self._sample_weights = params << "sample_weights"
        assert self._sample_weights is None or len(self._sample_weights) == len(self._ds)
        self._reset()

    def _reset(self):
        self._curr_idx = 0

        if len(self._ds) > self._bs:
            num_pad_to_order = self._bs - (len(self._ds) % self._bs)  # pad to be a multiple of batch_size
        else:
            num_pad_to_order = self._bs - len(self._ds)  # pad to have 1 batch

        if self._sample_weights is not None:
            if not self._shuffle:
                logger.debug("SequentialSampler: sampling always by weights, "
                             "but shuffle=False (which will be ignored)...")
            self._order = np.random.choice(len(self._ds), len(self._ds) + num_pad_to_order, replace=True, p=self._sample_weights)
        else:
            self._order = np.arange(len(self._ds))
            if num_pad_to_order > 0:
                # pad up to batch size with some random idxs.
                self._order = np.concatenate([self._order,
                                              np.random.choice(len(self._ds), num_pad_to_order, replace=False)])
            if self._shuffle:
                self._order = np.random.permutation(self._order)

        self._num_batches = len(self._order) // self._bs  # will be at least one

    def get_indices(self, **kwargs):
        # predetermined order
        if self._curr_idx >= self._num_batches:
            self._reset()
        indices = self._order[self._curr_idx * self._bs: (self._curr_idx + 1) * self._bs]
        assert len(indices) == self._bs, "Bug, this shouldn't happen"
        self._curr_idx += 1
        return indices


class WeightedSequentialSampler(SequentialSampler):

    def __init__(self, dataset, params):
        super(WeightedSequentialSampler, self).__init__(dataset, params)
        self._mode_key = get_with_default(params, "mode_key", "mode")
        # for each value.
        self._num_modes = get_with_default(params, "num_modes", 2)
        # default uniform
        self._default_weights = get_with_default(params, "default_weights", np.ones(self._num_modes), map_fn=np.asarray)
        assert len(self._default_weights) == self._num_modes

        self._mtw = {
            'max': self.max_mode_to_weight,
            'first': self.nth_mode_to_weight,
            'last': lambda mode: self.nth_mode_to_weight(mode, n=-1),
        }
        self._default_mtw = get_with_default(params, "default_mtw", 'first')

        self._mode_to_weight_fn = get_with_default(params, "mode_to_weight_fn", self._mtw[self._default_mtw])

        assert self._mode_key in self._ds._env_spec.all_names

        # initial pass through the data to determine all weights (goes through get batch)
        all_weights = []
        for i in range(0, len(self._ds), self._bs):
            idxs = np.minimum(np.arange(i, i + self._bs), len(self._ds) - 1)
            res = self._ds.get_batch(indices=idxs, local_batch_size=len(idxs))
            mode = (res[0] & res[1]) >> self._mode_key
            all_weights.append(self._mode_to_weight_fn(mode))

        self._sample_weights = np.concatenate(all_weights)[:len(self._ds)]  # remove the last extra things

        unique, counts = np.unique(self._sample_weights, return_counts=True)
        logger.debug(f"Before normalizing: sample weight -> min = {min(self._sample_weights)}, max = {max(self._sample_weights)}")
        logger.debug(f"  unique = {unique}, counts = {counts}")
        self._sample_weights = self._sample_weights / self._sample_weights.sum()
        self._reset()  # do this again now that we've updated weights

    def nth_mode_to_weight(self, mode, n=0):
        mode = mode.astype(np.long) % self._num_modes  # mode -> index, then wrap around in case its long
        if mode.size == self._bs:
            mode_idx = mode.reshape(self._bs)
        else:
            mode_idx = mode.reshape(self._bs, self._ds.horizon)[:, n]  # n'th element

        return self._default_weights[mode_idx]  # weight is 0 -> num_modes, shape (B,)

    def max_mode_to_weight(self, mode):
        mode = mode.astype(np.long) % self._num_modes  # mode -> index, then wrap around in case its long
        if mode.size == self._bs:
            mode_idx = mode.reshape(self._bs)
        else:
            mode_idx = mode.reshape(self._bs, self._ds.horizon).max(axis=-1)

        return self._default_weights[mode_idx]  # weight is 0 -> num_modes, shape (B,)
