import torch

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.python_utils import timeit, AttrDict, get_with_default, get_required
from sbrl.utils.torch_utils import concatenate, combine_after_dim, combine_dims, unconcatenate, randint_between


class ContactSampler(Model):

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def _init_params_to_attrs(self, params):
        super(ContactSampler, self)._init_params_to_attrs(params)
        self._variable_horizon = get_with_default(params, "variable_horizon", True)
        # if True, we will be rigid about contact start. sampled windows contain
        self._return_seq_from_start = get_with_default(params, "return_seq_from_start", False)
        # only applies for return_seq_from_start = True, will use the true contact start / end, (except soft boundary allowed)
        self._use_true_contact = get_with_default(params, "use_true_contact", False)
        self._get_contact_start_ends = get_required(params, "get_contact_start_ends")
        self._soft_boundary_length = get_with_default(params, "soft_boundary_length", 0)
        self._init_soft_boundary_length = get_with_default(params, "init_soft_boundary_length", self._soft_boundary_length)
        self._init_skew_sampling = get_with_default(params, "init_skew_sampling", False)  # sample more likely near the end.
        self._init_window_size = get_with_default(params, "init_window_size", None)  # how far before the affordance to sample the init window.
        self._contact_horizon = get_with_default(params, "contact_horizon", None)  # how long should the init sequence be. default is same as contact
        self._init_horizon = get_with_default(params, "init_horizon", None)  # how long should the init sequence be. default is same as contact
        self._do_goal_select = get_with_default(params, "do_goal_select", False)
        self._far_out_goals = get_with_default(params, "far_out_goals", True)  # only applies if do_goal_select
        self._do_initiation_sampling = get_with_default(params, "do_initiation_sampling", False)
        # if True, will use the true contact end as the sampling boundary
        self._do_initiation_sampling_until_contact_end = get_with_default(params, "do_initiation_sampling_until_contact_end", False)
        # if True when above ^ is True, this = True means we will use the sampled contact end instead of true_contact length (causal)
        self._do_initiation_sampling_until_contact_sample_end = get_with_default(params, "do_initiation_sampling_until_contact_sample_end", False)
        self._initiation_name = get_with_default(params, "initiation_name", "initiation")

        assert self._contact_horizon is None or self._init_horizon is None, "Cannot specify both."

        assert self._do_initiation_sampling or not self._do_initiation_sampling_until_contact_end, "Cannot init sample past start if not init sampling"
        assert self._do_initiation_sampling_until_contact_end or not self._do_initiation_sampling_until_contact_sample_end, "Cannot sample til the end of the contact sample unless sampling past contact start."

        logger.debug(
             f"ContactSampler: sb[{self._soft_boundary_length}], goal_select[{self._do_goal_select}], init_sample[{self._do_initiation_sampling}], init_skew[{self._init_skew_sampling}, init_until_end[{self._do_initiation_sampling_until_contact_end} | {self._do_initiation_sampling_until_contact_sample_end}]")
        if self._init_window_size is not None:
            logger.debug(f"                iw[{self._init_window_size}]")

    def _select_goals(self, inputs, model_outs):
        # default behavior is sample goal from end of window to end of all.
        end_of_window_idxs = model_outs >> "sample_window_ends"  # where sampling window ends
        end_of_contact_idxs = model_outs >> "sampling_contact_ends"  # where contact ends (soft)
        ep_lengths = model_outs >> "ep_lengths"  # where interaction ends

        return randint_between(end_of_window_idxs, ep_lengths) \
            if self._far_out_goals else \
            randint_between(end_of_window_idxs, end_of_contact_idxs + 1)  # random between end and contact end

    def forward(self, inputs, preproc=True, posproc=True, outputs=None, current_horizon=None, init_horizon=None, **kwargs):
        # samples a horizon length contact sequence from an interaction
        if current_horizon is None:
            return inputs  # do nothing since we don't know how to sample

        current_horizon: int = current_horizon
        init_horizon: int = init_horizon if init_horizon is not None else current_horizon

        if self._contact_horizon is not None:
            current_horizon = self._contact_horizon
        elif self._init_horizon is not None:
            init_horizon = self._init_horizon

        if outputs is not None:
            # safe combine
            assert set(outputs.leaf_keys()).isdisjoint(inputs.leaf_keys())
            inputs = inputs & outputs

        if preproc:
            inputs = self._preproc_fn(inputs)

        model_outs = AttrDict()
        names = inputs.list_leaf_keys()
        if 'padding_mask' in names:
            names.remove('padding_mask')

        with timeit("contact_sampling/concatenate"):
            # B x maxH x sum(D)
            ins: AttrDict = inputs > names
            element_shapes = ins.leaf_apply(lambda arr: list(arr.shape)[2:])
            element_dtypes = ins.leaf_apply(lambda arr: arr.dtype)
            # ins.leaf_shapes().pprint()
            flat = concatenate(ins.leaf_apply(lambda arr: combine_after_dim(arr, 2)), names, dim=-1)
            B, H = flat.shape[:2]

        with timeit("contact_sampling/get_periods"):
            # (1) compute the reach, contact, and detach periods

            with timeit("contact_sampling/get_start_ends"):
                # each is (B,) [0, 0, 0, 10, 0]   [4, 5, 7, 12, 2]
                ep_horizons = H * torch.ones(B, dtype=torch.int, device=flat.device)
                if self._variable_horizon:
                    # B x H, which idxs to keep
                    mask = ~(inputs >> "padding_mask")
                    ep_lengths = mask.count_nonzero(dim=1)  # (B,)
                else:
                    # (B, ), all the same length
                    ep_lengths = ep_horizons.copy()

                assert torch.all(ep_lengths >= current_horizon), "episode lengths must all be greater than curr_horizon."
                if init_horizon != current_horizon:
                    assert torch.all(ep_lengths >= init_horizon), "episode lengths must all be greater than init_horizon."
                # (B, ), ordered, concat_starts < contact_ends <= ep_lengths
                true_contact_starts, true_contact_ends, contact_mask = self._get_contact_start_ends(inputs,
                                                                                                    current_horizon=current_horizon,
                                                                                                    ep_lengths=ep_lengths)
                # print((true_contact_ends - true_contact_starts).max())
                # the interaction should have at least curr_horizon steps
                # assert torch.all(ep_lengths >= current_horizon)

            # ep_start = H * torch.arange(B, device=flat.device)

            with timeit("contact_sampling/smooth_contact_region"):
                # pad the interaction region to the right length
                # region will be smaller if contact interaction doesn't occur for long enough.
                if self._return_seq_from_start and self._use_true_contact:
                    contact_starts = true_contact_starts
                    contact_ends = true_contact_ends
                else:
                    max_viable_start = ep_lengths - current_horizon
                    contact_starts = torch.where(true_contact_starts > max_viable_start, max_viable_start, true_contact_starts)
                    min_viable_end = contact_starts + current_horizon - 1
                    contact_ends = torch.where(true_contact_ends < min_viable_end, min_viable_end, true_contact_ends)

                # clen = contact_ends - contact_starts
                # if not self._fixed_contact_start:
                #     # contact start will be "movable"
                #     clen = torch.where(clen < current_horizon, current_horizon, clen)

                # contact_lengths = contact_ends - contact_starts
                ep_starts = H * torch.arange(B, device=flat.device)
                # ep_ends = ep_starts + ep_lengths

                # sample length curr_horizon within the batch.
                if self._soft_boundary_length > 0:
                    # allow for some "wiggle" room in when the contact starts / ends
                    contact_starts = torch.maximum(contact_starts - self._soft_boundary_length,
                                                   torch.zeros(1, dtype=contact_starts.dtype, device=contact_starts.device))
                    contact_ends = torch.minimum(contact_ends + self._soft_boundary_length, ep_lengths - 1)

                if self._return_seq_from_start:
                    sampled_contact_start = contact_starts
                else:
                    max_allowed_starts = contact_ends + 1 - current_horizon  # cends + 1 - ch >= cstarts
                    sampled_contact_start = randint_between(contact_starts, max_allowed_starts + 1)
                    # sampled_contact_end = sampled_contact_start + current_horizon  # will necessarily be within ep bounds.

                c_lengths = contact_ends - sampled_contact_start + 1 if self._return_seq_from_start else current_horizon

                # initiation
                if self._do_initiation_sampling:
                    # sample length `curr_horizon` window for initiation, ending at the contact start window, or after 1 horizon, whichever is bigger
                    if self._do_initiation_sampling_until_contact_end:
                        # TODO do we want to only end at the sampling end? rather than true end.
                        c_ends = sampled_contact_start + c_lengths - 1 if self._do_initiation_sampling_until_contact_sample_end else true_contact_ends
                        max_viable_init_end = torch.where(c_ends < init_horizon - 1, init_horizon - 1,
                                                          c_ends)
                    else:
                        max_viable_init_end = torch.where(true_contact_starts < init_horizon - 1, init_horizon - 1,
                                                          true_contact_starts)

                    if self._init_soft_boundary_length > 0:
                        # allow for some "wiggle" room in when the contact starts / ends
                        max_viable_init_end = torch.minimum(max_viable_init_end + self._init_soft_boundary_length,
                                                            ep_lengths - 1)

                    # max_viable_end + 1 - current_horizon is the max viable starting idx.
                    max_viable_init_start = (max_viable_init_end + 1 - init_horizon).to(dtype=torch.int)
                    earliest_init_start = torch.zeros_like(max_viable_init_end, dtype=max_viable_init_end.dtype)

                    if self._init_window_size is not None:
                        assert self._init_window_size >= init_horizon, f"Init window size = {self._init_window_size} < horizon = {init_horizon}"
                        # move up the start, so that we don't consider after.
                        earliest_init_start = torch.maximum(earliest_init_start, max_viable_init_end + 1 - self._init_window_size)

                    if self._init_skew_sampling:
                        # skew "right"
                        # places exponentially more weight on sampling idxs closer to the affordance. no longer uniform sampling
                        sampled_initiation_start = randint_between(earliest_init_start, max_viable_init_start + 1, fn=lambda x: x**0.5)
                    else:
                        sampled_initiation_start = randint_between(earliest_init_start, max_viable_init_start + 1)

            # print("-----------")
            # print(true_contact_starts)
            # print(true_contact_ends)
            # print(ep_lengths)

            # window range for learning (along batch dim.

            with timeit("contact_sampling/get_sampling_window"):
                CURR_H = current_horizon
                if self._return_seq_from_start:
                    CURR_H = int(torch.max(c_lengths))
                    sampling_window_idxs = ((ep_starts + sampled_contact_start)[:, None] +
                                            torch.arange(CURR_H, device=flat.device)[None])
                    # bound it to not go past the end per episode
                    sampling_pad_mask = sampling_window_idxs > (ep_starts + contact_ends)[:, None]
                    sampling_window_idxs = torch.minimum(sampling_window_idxs, (ep_starts + contact_ends)[:, None])
                    sample_window_ends = contact_ends
                    sampling_window_idxs = sampling_window_idxs.reshape(-1)
                    # print(sampling_pad_mask.shape, CURR_H)
                else:
                    sampling_window_idxs = (
                                (ep_starts + sampled_contact_start)[:, None] + torch.arange(CURR_H, device=flat.device)[
                            None])
                    # sample_window_starts = sampling_window_idxs[:, 0] - ep_starts
                    sample_window_ends = sampling_window_idxs[:, -1] - ep_starts  # last time step
                    # print("true_cstart", true_contact_starts)
                    # print("true_cend", true_contact_ends)
                    # print("sample_cstart", sampled_contact_start)
                    # print("sample_cend", sample_window_ends)
                    sampling_window_idxs = sampling_window_idxs.reshape(-1)
                    sampling_pad_mask = None

                # print("sampling", sampling_window_idxs)
                # B x curr_horizon x sumD
                sampled_flat = combine_dims(flat, 0, 2)[sampling_window_idxs].reshape(B, CURR_H, -1)
                # print(sampled_flat.shape)
                model_outs.combine(unconcatenate(sampled_flat, names, element_shapes, dtypes=element_dtypes))

                if self._do_initiation_sampling:
                    # uses possibly different horizon.
                    # populate keys
                    initiation_window_idxs = ((ep_starts + sampled_initiation_start)[:, None] +
                                              torch.arange(init_horizon, device=flat.device)[None])
                    initiation_sample_window_ends = initiation_window_idxs[:, -1] - ep_starts  # last time step
                    initiation_window_idxs = initiation_window_idxs.reshape(-1)
                    initiation_flat = combine_dims(flat, 0, 2)[initiation_window_idxs].reshape(B, init_horizon, -1)
                    # initiation key set here, with all unconcatenated inputs.
                    model_outs[self._initiation_name].combine(
                        unconcatenate(initiation_flat, names, element_shapes, dtypes=element_dtypes))
                    model_outs[f"{self._initiation_name}_sampling_window_indices"] = initiation_window_idxs
                    model_outs[f"{self._initiation_name}_sample_window_starts"] = sampled_initiation_start
                    model_outs[f"{self._initiation_name}_sample_window_ends"] = initiation_sample_window_ends

            # idxs (B*curr_horizon,) to sample
            model_outs.sampling_window_indices = sampling_window_idxs
            # idxs (B,)
            model_outs.sample_window_starts = sampled_contact_start
            model_outs.sample_window_ends = sample_window_ends
            model_outs.contact_starts = true_contact_starts
            model_outs.contact_ends = true_contact_ends
            model_outs.contact_mask = contact_mask
            model_outs.sampling_contact_starts = contact_starts
            model_outs.sampling_contact_ends = contact_ends
            model_outs.ep_lengths = ep_lengths
            model_outs.ep_starts = ep_starts

            # there might be padding on the sampled data, depending on the sampling method
            model_outs.sampling_padding_mask = sampling_pad_mask

            # flat arrays
            model_outs.concatenated_full = flat
            model_outs.concatenated_sample = sampled_flat

        if self._do_goal_select:
            with timeit("contact_sampling/select_goals"):
                # indices per batch axis
                model_outs.goal_state_idxs = self._select_goals(inputs, model_outs)

        return self._postproc_fn(inputs, model_outs) if posproc else model_outs
