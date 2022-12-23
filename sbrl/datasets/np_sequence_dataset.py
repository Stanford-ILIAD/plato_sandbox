"""
Indices here refer to the indexable sample sequences, not the episodes or the exact data points.

This enables sampling of sequences of data more efficiently and uniformly.
"""
import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger


class NpSequenceDataset(NpDataset):
    def _data_setup(self):
        super(NpSequenceDataset, self)._data_setup()
        # indexing into this will give you the episode to sample
        if len(self._sampling_period_lengths) > 0:
            self._seq_idx_to_ep_idx = np.repeat(np.arange(len(self._sampling_period_lengths)), self._sampling_period_lengths)
            self._seq_idx_to_start_in_ep_idx = np.concatenate([np.arange(l) for l in self._sampling_period_lengths])
            period_starts = np.concatenate([[0] + np.cumsum(self._period_lengths)[:-1]]) if len(self._period_lengths) > 1 else [0]
            self._seq_idx_to_true_idx = np.concatenate([ps + np.arange(l) for l, ps in zip(self._sampling_period_lengths, period_starts)])
        else:
            logger.warn("No sequences present, do not call get_batch.")

        # self._used_bins = np.zeros(len(self), dtype=np.int)

    def _resolve_indices(self, indices=None, min_idx=0, max_idx=0, np_pad=False, local_batch_size=None, **kwargs):
        # index resolution for sequences of length H by "sequence" idx
        assert self._data_len > 0
        # assert self.horizon > 1 TODO horizon = 1? does that work
        # import ipdb; ipdb.set_trace()
        if max_idx <= 0:
            max_idx += self._num_valid_samples
        assert 0 <= min_idx < max_idx <= self._num_valid_samples

        # now indices will refer to the indexable **sample sequences** not the episodes.
        if indices is None:
            # equally ranked sequences
            indices = np.random.choice(max_idx - min_idx, local_batch_size, replace=max_idx - min_idx < local_batch_size)
        else:
            assert len(indices) == local_batch_size, [local_batch_size, len(indices), indices]

        episode_indices = self._seq_idx_to_ep_idx[indices]
        start_indices_within_ep = self._seq_idx_to_start_in_ep_idx[indices]
        ep_lens = self._period_lengths[episode_indices]
        ep_start_idxs = self.ep_starts[episode_indices]

        start_indices = ep_start_idxs + start_indices_within_ep
        # B x H
        unclipped_seq_indices = start_indices[:, None] + np.arange(self.horizon)[None]
        max_indices = ep_start_idxs + ep_lens - 1
        # clipping indices to not overflow (will pad end depending on self._pad_end_sequences)
        indices = np.minimum(unclipped_seq_indices, max_indices[:, None])

        chunk_lengths = np.minimum(ep_lens, self.horizon)
        do_padding = self._allow_padding and np.any(
            chunk_lengths < self.horizon)  # pad if some sequences are not len() = horizon

        # self._used_bins[indices.reshape(-1)] += 1
        # print(np.bincount(self._used_bins))

        return indices.reshape(-1), episode_indices, chunk_lengths, do_padding

    def __len__(self):
        if self._frozen:
            return self._num_valid_samples
        else:
            return super(NpSequenceDataset, self).__len__()  # not implemented...