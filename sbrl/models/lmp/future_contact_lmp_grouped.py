import torch

from sbrl.datasets.fast_np_interaction_dataset import NpInteractionDataset
from sbrl.models.lmp.lmp_grouped import LMPGroupedModel
from sbrl.utils.python_utils import AttrDict as d, get_with_default, timeit
from sbrl.utils.torch_utils import broadcast_dims, unsqueeze_then_gather


class FutureContactLMPGroupedModel(LMPGroupedModel):
    """
    A contact interaction is defined as follows:
    [period of non contact]
    [grasp]
    [period of semi-continuous contact]
    [release]
    [optional period of non contact afterwards]

    This model requires inputs to have all of these segments in the horizon.

    Unlike ContactLMPGroupedModel, we

    """

    required_models = LMPGroupedModel.required_models + [
        'contact_sampler',
    ]

    def _init_params_to_attrs(self, params: d):

        self._no_policy = get_with_default(params, "no_policy", False)
        if self._no_policy:
            params.optimize_policy = False  # disable regular policy update.

        super(FutureContactLMPGroupedModel, self)._init_params_to_attrs(params)

        self._contact_horizon = get_with_default(params, "contact_horizon", self.horizon)
        self._beta_contact = get_with_default(params, "beta_contact", 0.)
        self._do_contact_policy = self._beta_contact > 0
        # # right-aligned sampling of contact window TODO in ContactSampler
        # self._align_contact_sample = get_with_default(params, "contact_sample_end", False)

        self._goal_sampling = get_with_default(params, "goal_sampling",
                                               False)  # goal select happens within contact_sampler

        # assert isinstance(self._get_contact_start_ends, Callable)
        if self._dataset_train is not None:
            assert isinstance(self._dataset_train, NpInteractionDataset), \
                f"Dataset must be compatible with reading interactions: {type(self._dataset_train)}"

    def forward(self, inputs, preproc=True, postproc=True, run_prepare=True, current_horizon=None, sample_first=True,
                run_plan=True,
                plan_posterior=False, run_all=False, run_policy=True, run_contact_policy=True,
                model_outs=d(), meta=d(), **kwargs):

        # if run_policy is None:
        #     run_policy = run_all or not self._no_policy  # default during training.

        with timeit("contact_lmp/prepare_inputs"):
            model_outs = model_outs.leaf_copy()
            if run_prepare:
                inputs = self._prepare_inputs(inputs, preproc=preproc, current_horizon=None)  # do NOT truncate

        with timeit("contact_lmp/sampler"):
            contact_current_horizon = self._contact_horizon
            # returns sampled contact window in the future, and optionally goals,
            model_outs.sampler = self.contact_sampler.forward(inputs,
                                                              current_horizon=contact_current_horizon if sample_first else None,
                                                              init_horizon=current_horizon,
                                                              meta=meta)
            to_combine = list(
                set(inputs.list_leaf_keys()).intersection(model_outs.sampler.list_leaf_keys()))  # override
            # print(model_outs.sampler.list_leaf_keys())
            contact_lmp_inputs = inputs & (model_outs.sampler > to_combine)  # override with the shared keys
            contact_current_horizon = min(contact_current_horizon, contact_lmp_inputs.get_one().shape[1])
            if model_outs << "sampler/sampling_padding_mask" is not None:
                # method can choose to use this or not (contact region)
                contact_lmp_inputs["padding_mask"] = model_outs.sampler.sampling_padding_mask
                contact_current_horizon = model_outs.sampler.sampling_padding_mask.shape[1]

            if self._goal_sampling and sample_first:
                # things for the sampler to use, required. relative to each batch start idx.
                goal_state_idxs = model_outs.sampler >> "goal_state_idxs"  # (B x H x ...) goals
                # gather horizon elements along batch dim, after broadcasting index to right shape
                goal_bcast_horizon = (model_outs.sampler >> to_combine[0]).shape[1]
                goal_states = inputs.leaf_apply(
                    lambda arr: broadcast_dims(unsqueeze_then_gather(arr, goal_state_idxs, dim=1)[:, None],
                                               [1], [goal_bcast_horizon]))
                contact_lmp_inputs.goal_states = goal_states  # add goal states for future use, broadcasted to match new inputs

            if sample_first:
                # sampler needs to return "initiation" key, with window for grasp policy.
                assert model_outs.has_node_leaf_key("sampler/initiation")

                init_lmp_inputs = ((model_outs >> "sampler/initiation") > to_combine)
                init_lmp_inputs = inputs & init_lmp_inputs

                # init sequence should not be padded.
                if init_lmp_inputs.has_leaf_key("padding_mask"):
                    init_lmp_inputs.__delattr__("padding_mask")

                if self._goal_sampling and sample_first:
                    init_lmp_inputs.goal_states = contact_lmp_inputs >> "goal_states"
            else:
                init_lmp_inputs = contact_lmp_inputs.leaf_copy()

        # PLAN(s)
        if run_plan:
            with timeit("contact_lmp/combine"):
                # contact window, and initiation window get combined TODO
                ic_inputs = self._combine_initiation_contact(init_lmp_inputs, contact_lmp_inputs,
                                                             contact_horizon=contact_current_horizon, init_horizon=current_horizon)

            with timeit("contact_lmp/combined_plans"):
                # lmp_inputs.leaf_apply(lambda arr: arr.shape).pprint()
                ic_outputs = super(FutureContactLMPGroupedModel, self).forward(ic_inputs, preproc=False,
                                                                               postproc=False,
                                                                               run_prepare=False, run_all=False,
                                                                               run_plan=True,
                                                                               plan_posterior=plan_posterior,
                                                                               model_outs=model_outs,
                                                                               run_policy=False,
                                                                               # just run plan and/or prior
                                                                               current_horizon=contact_current_horizon,
                                                                               run_goal_select=not self._goal_sampling or not sample_first)
                model_outs.combine(ic_outputs)

        # POLICY for PLAN(s)
        if run_policy:
            with timeit("contact_lmp/initiation_policy"):
                # run lfp policy, but condition on the latent above.
                init_current_horizon = init_lmp_inputs.get_one().shape[1]
                init_lmp_outs = (model_outs < ['embedding', 'plan_posterior_sample',
                                               'plan_prior_sample']).leaf_copy()  # at least one will be here.

                # run just the initiation sequence policy
                init_lmp_outs = super(FutureContactLMPGroupedModel, self).forward(init_lmp_inputs, preproc=False,
                                                                                  postproc=False,
                                                                                  run_prepare=False, run_plan=False,
                                                                                  # read plans from contact region
                                                                                  run_all=run_all,
                                                                                  run_policy=True,
                                                                                  plan_posterior=plan_posterior,
                                                                                  model_outs=init_lmp_outs,
                                                                                  current_horizon=init_current_horizon,
                                                                                  run_goal_select=not self._goal_sampling or not sample_first, **kwargs)
                model_outs.combine(init_lmp_outs)

        # POLICY for PLAN(s)
        if run_all or (run_contact_policy and self._do_contact_policy):
            with timeit("contact_lmp/contact_policy"):
                # run lfp policy, but condition on the latent above.
                contact_current_horizon = contact_lmp_inputs.get_one().shape[1]
                contact_lmp_outs = (model_outs < ['embedding', 'plan_posterior_sample',
                                               'plan_prior_sample']).leaf_copy()  # at least one will be here.

                # run just the initiation sequence policy
                contact_lmp_outs = super(FutureContactLMPGroupedModel, self).forward(contact_lmp_inputs, preproc=False,
                                                                                  postproc=False,
                                                                                  run_prepare=False, run_plan=False,
                                                                                  # read plans from contact region
                                                                                  run_all=run_all,
                                                                                  run_policy=True,
                                                                                  plan_posterior=plan_posterior,
                                                                                  model_outs=contact_lmp_outs,
                                                                                  current_horizon=contact_current_horizon,
                                                                                  run_goal_select=not self._goal_sampling or not sample_first, **kwargs)
                model_outs['contact'].combine(contact_lmp_outs)

        return self._postproc_fn(inputs, model_outs) if postproc else model_outs

    def _combine_initiation_contact(self, init_inputs, contact_inputs, contact_horizon=None, init_horizon=None):
        shared_keys = set(init_inputs.list_leaf_keys()).intersection(contact_inputs.list_leaf_keys())
        # leftover_keys = set(init_inputs.list_leaf_keys()).symmetric_difference(contact_inputs.list_leaf_keys())

        # stacked along horizon
        return d.leaf_combine_and_apply([init_inputs > shared_keys, contact_inputs > shared_keys], lambda vs: torch.cat(vs, dim=1))

    def _get_policy_outputs(self, inputs, outputs, model_outputs, base_key="sampler/initiation", current_horizon=None):
        """
        returns the policy outputs (current_horizon - 1), aligned with the contact sample

        :param inputs: raw inputs (B x maxH)
        :param outputs:
        :param model_outputs:
        :param current_horizon:
        :return:
        """
        new_outs = (model_outputs >> base_key) > list(self.action_names)
        # new_outs.leaf_apply(lambda arr: arr.shape).pprint()
        new_outs = new_outs.leaf_apply(lambda arr: arr[:, :-1])  # skip last step
        return outputs & new_outs

    def additional_losses(self, model_outs, inputs, outputs, i=0, writer=None, writer_prefix="", current_horizon=None):
        losses, extra_scalars = super(FutureContactLMPGroupedModel, self).additional_losses(model_outs, inputs, outputs, i=i,
                                                                                      writer=writer,
                                                                                      writer_prefix=writer_prefix,
                                                                                      current_horizon=current_horizon)

        if self._do_contact_policy:
            # model_outs.leaf_apply(lambda arr: None).pprint()
            contact_model_outs = (model_outs >> "contact").leaf_copy()
            contact_model_outs.sampler = model_outs >> "sampler"  # for action_loss to know the true input window if it cares.
            contact_model_outs.combine(contact_model_outs >> "plan_posterior_policy")

            # logger.debug(init_model_outs.leaf_shapes().pprint(ret_string=True))

            # initiation policy outputs, don't require padding mask over loss.
            outs = self._get_policy_outputs(inputs, outputs, model_outs, base_key="sampler")

            # these inputs are wrong!
            contact_policy_posterior_loss = self.action_loss_fn(self, contact_model_outs, inputs, outs,
                                                                   i=i, writer=writer,
                                                                   writer_prefix=writer_prefix + "posterior/contact/")

            # write the prior loss here, since it doesn't get optimized directly.
            if writer is not None:
                contact_model_outs.combine(contact_model_outs >> "plan_prior_policy")

                # write the prior policy
                contact_policy_prior_loss = self.action_loss_fn(self, contact_model_outs, inputs, outs,
                                                                   i=i, writer=writer,
                                                                   writer_prefix=writer_prefix + "prior/contact/")

                writer.add_scalar(writer_prefix + "prior/contact/policy_loss", (contact_policy_prior_loss).mean().item(), i)

            # gets added
            losses['posterior/contact/policy_loss'] = (self._beta_contact, contact_policy_posterior_loss)
            # returns losses, extra_scalars

        return losses, extra_scalars

    @property
    def contact_sampler(self):
        return self._contact_sampler


if __name__ == '__main__':
    pass
