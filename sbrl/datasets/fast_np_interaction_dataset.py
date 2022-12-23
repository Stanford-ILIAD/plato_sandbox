import numpy as np
import torch

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.utils.np_utils import np_idx_range_between
from sbrl.utils.python_utils import get_required, get_with_default, timeit, AttrDict
from sbrl.utils.torch_utils import combine_dims_np


class NpInteractionDataset(NpDataset):

    def _init_params_to_attrs(self, params):
        super(NpInteractionDataset, self)._init_params_to_attrs(params)
        # returns list of idx (begin, cstart, cend, end) per window, and number of internal skips
        # begin <= cstart < cend <= end
        self._parse_interaction_bounds_from_episode_fn = get_required(params, "parse_interaction_bounds_from_episode_fn")
        # gets the points of contact within a window
        self._pad_to_horizon = get_with_default(params, "pad_to_horizon", True)  # pads last element for shorter than max len interactions.
        self._sample_pre_window = get_with_default(params, "sample_pre_window", True)  # samples horizon length window for pre-interactions.
        self._sample_goals = get_with_default(params, "sample_goals", True)  # toggles on goal selection (default samples between window_end and end)

        # interaction goals sampled from some subset of: [window_end | cend | end] based on following three bools. default is [window_end -> end] (whole range)
        self._sample_goal_start = get_with_default(params, "sample_goal_start", True)  # goal will be window_end ONLY (fixed goal)
        self._sample_post_goals_only = get_with_default(params, "sample_post_goals_only", False)  # samples goals only in [cend -> end]
        self._sample_interaction_goals_only = get_with_default(params, "sample_interaction_goals_only", False)  # samples goals only in [window_end -> cend]

        assert not self._sample_post_goals_only or not self._sample_interaction_goals_only, "Only one goal_sampling method can be specified!"

        self._all_interactions = []  # stacked list per episode
        self._total_num_interactions = 0

        self._soft_boundary_length = get_with_default(params, "soft_boundary_length", 0)
        self._init_soft_boundary_length = get_with_default(params, "init_soft_boundary_length", self._soft_boundary_length)

        # these all get populated by refresh_interaction_stable()
        self._stacked_interactions = None
        self._interaction_episode_map = None
        self._interaction_table_is_stale = True
        self._pre_window_key_prefix = get_with_default(params, "pre_window_key_prefix", "initiation")
        self._goal_key_prefix = get_with_default(params, "goal_key_prefix", "goal_states")

        assert self._frozen, "Not Implemented (dynamic dataset)"

    def _data_setup(self):
        num_skipped = 0
        num_skipped_interactions = 0
        num_saved_interactions = 0
        for i in range(self.get_num_episodes()):
            # (Hi x ...)
            inputs, outputs = self.get_episode(i, None, split=True)
            next_interactions, n_skip = self._compute_interactions_for_episode(inputs, outputs)
            self._all_interactions.append(next_interactions)
            if self._all_interactions[-1].shape[0] == 0:
                num_skipped += 1
            num_skipped_interactions += n_skip
            num_saved_interactions += self._all_interactions[-1].shape[0]

            # contact = np.logical_or(np.any(inputs >> "block_contact", axis=-1).reshape(-1),
            #                         np.any(inputs >> "grab_binary", axis=-1).reshape(-1))
            # for i in range(next_interactions.shape[0]):
            #     s, e = next_interactions[i]
            #     assert contact[s:e+1].any()

        # import ipdb; ipdb.set_trace()
        if num_skipped > 0:
            logger.warn(f"Skipped {num_skipped}/{self.get_num_episodes()} episodes during loading")
            logger.warn(f"Skipped {num_skipped_interactions}/{num_skipped_interactions + num_saved_interactions} interactions during loading")

        # called after initial episode set
        self.refresh_interaction_table()

        logger.warn(
            f"Interaction Lengths: min {np.min(self._interaction_lengths)}, max {np.max(self._interaction_lengths)}, med {np.median(self._interaction_lengths)}, <H {np.sum(self._interaction_lengths < self._horizon)} ")

        # for i in range(self.get_num_periods()):
        #     inputs, outputs, data_idxs, indices, all_eps_to_fetch, local_batch_size, chunk_lengths, names_to_get = self.get_periods([i], do_index=True)
        #     contact = np.logical_or(np.any(inputs >> "block_contact", axis=-1).reshape(-1),
        #                             np.any(inputs >> "grab_binary", axis=-1).reshape(-1))
        #     if not contact.any():
        #         import ipdb; ipdb.set_trace()

        assert self._horizon > 1, "interaction batch requires horizon > 1"

        super(NpInteractionDataset, self)._data_setup()

    def add_episode(self, inputs, outputs, **kwargs):
        # (Hi x ...)
        super(NpInteractionDataset, self).add_episode(inputs, outputs, **kwargs)
        inputs, outputs = self.get_episode(self.get_num_episodes() - 1, None, split=True)
        self._all_interactions.append(self._compute_interactions_for_episode(inputs, outputs))
        self._interaction_table_is_stale = True

    def add(self, inputs, outputs, rollout_step=0, **kwargs):
        raise NotImplementedError("single step add not allowed right now")

    def _compute_interactions_for_episode(self, inputs, outputs):
        # list of "distinct" interaction boundaries [start, end), varying time length
        interaction_list, n_skip = self._parse_interaction_bounds_from_episode_fn(inputs, outputs)

        # for s,e in interaction_list:
        #     if (inputs >> "objects/contact")[s:e].sum() < 1:
        #         import ipdb; ipdb.set_trace()

        # check that these idxs are valid.
        interaction_idxs = np.asarray(interaction_list)
        assert len(interaction_idxs.shape) == 2 and interaction_idxs.shape[1] == 4, interaction_idxs.shape
        assert np.all(interaction_idxs <= len(outputs.done))  # end idx might be equal to ep length
        assert np.all(interaction_idxs >= 0)
        assert np.all(interaction_idxs[:, 0] < interaction_idxs[:, 3]), f"No zero length interactions! {interaction_idxs.shape}"
        assert np.all(interaction_idxs[:, 0] <= interaction_idxs[:, 1]), f"Invalid start / contact_start: {interaction_idxs[:, :2]}"
        assert np.all(interaction_idxs[:, 1] <= interaction_idxs[:, 2]), f"Invalid contact_start / contact_end: {interaction_idxs[:, 1:3]}"
        assert np.all(interaction_idxs[:, 2] <= interaction_idxs[:, 3]), f"Invalid contact_end / end: {interaction_idxs[:, 2:]}"
        assert n_skip >= 0, "No negative skips!"

        return interaction_idxs, n_skip

    def _get_batch(self, indices=None, torch_device=None, min_idx=0, max_idx=0, non_blocking=False, np_pad=False, local_batch_size=None, **kwargs):
        # gets interaction level batches, of variable length
        self.refresh_interaction_table()

        assert min_idx == 0 and max_idx == 0, "Not implemented"

        if local_batch_size is None:
            if indices is not None:
                local_batch_size = len(indices)
            else:
                local_batch_size = self.batch_size

        with timeit("get_batch/idx_sample"):
            # First get the period, default weighted by the length of the interaction (roughly number of sample-able windows).
            if indices is None:
                indices = np.random.choice(self.get_num_periods(), local_batch_size, p=self.period_weights(), replace=True)

            # local_batch_size = len(indices)
            # (B,) interactions
            all_critical_idxs = self._stacked_interactions[indices]
            all_eps_to_fetch = self._interaction_episode_map[indices]
            prev = np.where(all_eps_to_fetch > 0, self._split_indices[np.maximum(all_eps_to_fetch - 1, 0).astype(int)],
                            0)

            start_idxs = (prev + all_critical_idxs[:, 0]).astype(int)  # (N,)
            raw_cstart_idxs = (prev + all_critical_idxs[:, 1]).astype(int)
            raw_cend_idxs = (prev + all_critical_idxs[:, 2]).astype(int)
            end_idxs = (prev + all_critical_idxs[:, 3]).astype(int)

            # padding for contact region
            if self._soft_boundary_length > 0:
                cstart_idxs = np.maximum(raw_cstart_idxs - self._soft_boundary_length, start_idxs)
                cend_idxs = np.minimum(raw_cend_idxs + self._soft_boundary_length, end_idxs)
            else:
                cstart_idxs = raw_cstart_idxs
                cend_idxs = raw_cend_idxs

            all_ep_end_idxs = self.ep_starts[all_eps_to_fetch] + self.ep_lengths[all_eps_to_fetch] - 1
            chunk_lengths = np.ones_like(start_idxs) * self.horizon

            # sample length-H between

        with timeit("get_batch/get_idx_chunks"):
            interaction_window = np_idx_range_between(cstart_idxs, cend_idxs, self.horizon, spill_end_idxs=all_ep_end_idxs)
            all_frames = [interaction_window]
            if self._sample_pre_window:
                pre_window_end_idxs = np.minimum(np.maximum(raw_cstart_idxs - 1 + self._init_soft_boundary_length, start_idxs), cend_idxs)
                pre_interaction_window = np_idx_range_between(start_idxs, pre_window_end_idxs, self.horizon, spill_end_idxs=all_ep_end_idxs)
                all_frames.append(pre_interaction_window)

            if self._sample_goals:
                if self._sample_post_goals_only:
                    gstart = cend_idxs  # end of CONTACT
                else:
                    gstart = interaction_window[:, -1]  # end of WINDOW (includes interaction)

                if self._sample_goal_start:
                    # one fixed goal per interaction (at the goal window start).
                    all_frames.append(gstart.reshape(-1, 1))
                else:
                    # one random goal per interaction.
                    if self._sample_interaction_goals_only:
                        goal_idxs = np_idx_range_between(gstart, cend_idxs, 1)  # no goals from post
                    else:
                        goal_idxs = np_idx_range_between(gstart, end_idxs, 1)  # uses posterior goals.
                    all_frames.append(goal_idxs.reshape(-1, 1))

        # first parse the keys we care about
        if self._batch_names_to_get is not None and len(self._batch_names_to_get) > 0:
            names_to_get = self._batch_names_to_get
        else:
            names_to_get = self._env_spec.all_names

        names_to_get = names_to_get + [self._done_key] + \
                       (["rollout_timestep"] if self._use_rollout_steps else [])

        # logger.debug(chunk_lengths.max())
        # logger.debug((~(inputs >> "padding_mask")).sum(-1))

        with timeit("get_batch/index_into_data"):
            horizon_chunk_lengths = [fr.shape[1] for fr in all_frames]  # for splitting later.
            all_data_idxs = np.concatenate(all_frames, axis=1)  # concat along horizon
            # no padding, fetch all the data first, chunk_lengths will be ignored.

            if self._index_all_keys:
                concat_inputs, concat_outputs = self._get_batch_index_into_merged_datadict(combine_dims_np(all_data_idxs, 0), all_eps_to_fetch, names_to_get, local_batch_size,
                                                                  chunk_lengths, False, sum(horizon_chunk_lengths), np_pad=np_pad, torch_device=torch_device)
            else:
                concat_inputs, concat_outputs = self._get_batch_index_into_datadict(combine_dims_np(all_data_idxs, 0), all_eps_to_fetch, names_to_get, local_batch_size,
                                                                  chunk_lengths, False, sum(horizon_chunk_lengths), np_pad=np_pad)

            # split along horizon
            start = 0
            all_inputs, all_outputs = [], []
            for Hi in horizon_chunk_lengths:
                all_inputs.append(concat_inputs.leaf_apply(lambda arr: arr[:, start:start+Hi]))
                all_outputs.append(concat_outputs.leaf_apply(lambda arr: arr[:, start:start+Hi]))
                start += Hi

            # interaction_window
            inputs = all_inputs[0]
            outputs = all_outputs[0]

            if self._sample_pre_window:
                inputs[self._pre_window_key_prefix] = all_inputs[1]
                outputs[self._pre_window_key_prefix] = all_outputs[1]

            if self._sample_goals:
                inputs[self._goal_key_prefix] = all_inputs[-1]
                outputs[self._goal_key_prefix] = all_outputs[-1]

        meta = AttrDict()
        if self._index_all_keys:
            assert self._batch_processor is None or not self._batch_processor_before_torch, "Things are already torchified!"

        else:
            meta = AttrDict()
            if self._batch_processor is not None and self._batch_processor_before_torch:
                inputs, outputs, meta = self._batch_processor.forward(inputs, outputs, all_data_idxs, indices,
                                                                      all_eps_to_fetch, names_to_get, chunk_lengths)

            with timeit("get_batch/to_device"):
                if torch_device is not None:
                    for d in (inputs, outputs, meta):
                        if self._is_torch:
                            d.leaf_modify(lambda x: x.to(torch_device, non_blocking=non_blocking))
                        else:
                            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device, non_blocking=non_blocking))

        if self._batch_processor is not None and not self._batch_processor_before_torch:
            inputs, outputs, meta = self._batch_processor.forward(inputs, outputs, all_data_idxs, indices, all_eps_to_fetch, names_to_get, chunk_lengths)

        return inputs, outputs, meta

    def get_num_periods(self):
        # overriding, this gets used in TorchDataset, for example.
        self.refresh_interaction_table()
        return self._total_num_interactions

    def get_periods(self, indices=None, do_index=True, local_batch_size=None, **kwargs):
        if local_batch_size is None:
            local_batch_size = self._batch_size

        self.refresh_interaction_table()

        # which interaction to choose. default weighted by the length of the interaction (roughly number of sample-able windows).
        if indices is None:
            indices = np.random.choice(self.get_num_periods(), local_batch_size, p=self.period_weights(), replace=True)
            # print(self.period_weights())

        with timeit("get_period"):
            local_batch_size = len(indices)
            # (B,) interactions
            all_critical_idxs = self._stacked_interactions[indices]
            all_eps_to_fetch = self._interaction_episode_map[indices]
            prev = np.where(all_eps_to_fetch > 0, self._split_indices[np.maximum(all_eps_to_fetch - 1, 0).astype(int)], 0)

            # print(all_start_end_idxs, all_eps_to_fetch)
            # print(prev)

            start_idxs = (prev + all_critical_idxs[:, 0]).astype(int)
            end_idxs = (prev + all_critical_idxs[:, 3]).astype(int)

            all_ep_end_idxs = self.ep_starts[all_eps_to_fetch] + self.ep_lengths[all_eps_to_fetch] - 1
            chunk_lengths = end_idxs - start_idxs  # cumulative lengths

            if self._pad_to_horizon:
                # end horizon gets padded
                true_end_idxs = end_idxs.copy()
                end_idxs = np.where(chunk_lengths < self.horizon, start_idxs + self.horizon, end_idxs)
                chunk_lengths = end_idxs - start_idxs
                assert np.all(chunk_lengths >= self.horizon), "bug"

            all_internal_idxs = []

            for i in range(len(start_idxs)):
                all_internal_idxs.append(np.arange(start_idxs[i], end_idxs[i]))

            data_idxs = np.concatenate(all_internal_idxs).astype(int)
            if self._pad_to_horizon:
                # e.g., 0,1,2,3,4,5, ep_len=4  --> 0,1,2,3,3,3
                data_idxs = np.minimum(data_idxs, np.repeat(all_ep_end_idxs, chunk_lengths))

        # first parse the keys we care about
        if self._batch_names_to_get is not None and len(self._batch_names_to_get) > 0:
            names_to_get = self._batch_names_to_get
        else:
            names_to_get = self._env_spec.all_names

        names_to_get = names_to_get + [self._done_key] + \
                       (["rollout_timestep"] if self._use_rollout_steps else [])

        if do_index:
            # B x max(Hi) x ...)
            chunk_lengths = chunk_lengths.astype(int)
            max_horizon = int(max(chunk_lengths))
            # print(max_horizon, data_idxs.shape, chunk_lengths.shape)

            inputs, outputs = self._get_batch_index_into_datadict(data_idxs, all_eps_to_fetch, names_to_get, local_batch_size,
                                                                  chunk_lengths, do_padding=True, horizon=max_horizon, **kwargs)
            # print("np_interaction: lengths and num contacts\n", np.stack([chunk_lengths, ctc.sum(-1)], axis=-1))
            return inputs, outputs, data_idxs, indices, all_eps_to_fetch, local_batch_size, chunk_lengths, names_to_get

        return data_idxs, indices, all_eps_to_fetch, local_batch_size, chunk_lengths, names_to_get

    def period_length(self, i):
        self.refresh_interaction_table()
        return self._interaction_lengths[i]

    def period_weights(self, indices=None):
        if indices is None:
            return np.ones((self.get_num_periods(),), dtype=np.float32) / self.get_num_periods()
        return np.ones_like(indices, dtype=np.float32) / len(indices)

    def refresh_interaction_table(self):
        if self._interaction_table_is_stale:
            with timeit("update_interaction_table"):
                self._stacked_interactions = np.concatenate(self._all_interactions)
                self._interaction_episode_map = np.concatenate([[i] * len(interact) for i, interact in enumerate(self._all_interactions)]).astype(int)
                self._total_num_interactions = len(self._stacked_interactions)
                self._interaction_lengths = (self._stacked_interactions[:, -1] - self._stacked_interactions[:, 0]).astype(int)
                self._interaction_table_is_stale = False

