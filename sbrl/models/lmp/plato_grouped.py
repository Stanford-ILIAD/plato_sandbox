import torch

from sbrl.datasets.fast_np_interaction_dataset import NpInteractionDataset
from sbrl.experiments import logger
from sbrl.models.lmp.lmp_grouped import LMPGroupedModel
from sbrl.utils.dist_utils import detach_normal
from sbrl.utils.python_utils import AttrDict as d, get_with_default, timeit
from sbrl.utils.torch_utils import broadcast_dims


class PLATOGroupedModel(LMPGroupedModel):
    """
    A contact interaction is defined as follows:
    [period of non contact]
    [grasp]
    [period of semi-continuous contact]
    [release]
    [optional period of non contact afterwards]

    This model requires inputs to have all of these segments in the horizon.
    LMP is used to learn the affordances, from [a subset] of the contact period

    """

    def _init_params_to_attrs(self, params: d):
        self._no_contact_policy = get_with_default(params, "no_contact_policy", False)
        if self._no_contact_policy:
            params.optimize_policy = False  # disable regular policy update.

        super(PLATOGroupedModel, self)._init_params_to_attrs(params)
        # self._get_contact_start_ends = get_required(params, "get_contact_start_ends")
        # self._variable_horizon = get_with_default(params, "variable_horizon", True)

        # run policy on init window
        self._do_init_policy = get_with_default(params, "do_init_policy", False)
        # weight of initiation reconstruction
        self._beta_init = get_with_default(params, "beta_init", 1.)
        # no gradients from initiation policy to encoder
        self._detach_init_plan = get_with_default(params, "detach_init_plan", False)
        # discounting action reconstruction seq based on how far away from affordance.
        self._init_discount = get_with_default(params, "init_discount", 1.)

        assert not self._detach_init_plan or self._do_init_policy, "Cannot detach init plan without enabling init pol."
        self._goal_sampling = get_with_default(params, "goal_sampling",
                                               False)  # goal select happens within contact_sampler
        # assert isinstance(self._get_contact_start_ends, Callable)
        if self._dataset_train is not None:
            assert isinstance(self._dataset_train, NpInteractionDataset), \
                    "Dataset must be compatible with reading interactions: {type(self._dataset_train)}"

        if self._do_init_policy:
            logger.info(f"ContactLMP using initiation policy. beta_init = {self._beta_init}, "
                        "dt-plan = {self._detach_init_plan}, gamma_init = {self._init_discount}")

        if self._no_contact_policy:
            logger.info("ContactLMP using no contact policy! Make sure init window sampling behavior knows this...")
            assert self._do_init_policy, "Init policy must train if contact is not training."

    def compute_policy_outs(self, pl_inputs, model_outs, plan_sample_name, sample=False, **kwargs):
        # add the sampled plan + inputs + encodings, optional sampling the output
        policy_ins = self.policy_input_selector(pl_inputs)
        policy_outs = self.policy(policy_ins, **kwargs)
        model_outs[plan_sample_name + "_policy"].combine(policy_outs)

        if sample:
            policy_sample = self.action_sample_fn(model_outs[plan_sample_name + "_policy"])
            assert policy_sample.has_leaf_keys(self.action_names)
            model_outs[plan_sample_name + "_policy"].combine(policy_sample)  # sampled action will be at the root level

        return model_outs[plan_sample_name + "_policy"]

    def forward(self, inputs, preproc=True, postproc=True, run_prepare=True, current_horizon=None, sample_first=True,
                do_init=True, plan_posterior=False, run_all=False, run_policy=None,
                model_outs=d(), meta=d(), **kwargs):

        if run_policy is None:
            run_policy = run_all or not self._no_contact_policy  # default during training.

        with timeit("contact_lmp/prepare_inputs"):
            model_outs = model_outs.leaf_copy()

            if run_prepare:
                # make sure normalization inputs are in all of these.
                inputs = self._prepare_inputs(inputs, preproc=preproc, current_horizon=None)
                # all keys that aren't initiation keys
                lmp_inputs = inputs.node_leaf_filter_keys([k for k in list(inputs.keys()) if k != "initiation"])
                # max over the shape[1] dims
                contact_current_horizon = lmp_inputs.leaf_reduce(lambda red, val: max(red, val.shape[1]), seed=1)

                if "initiation" in inputs.keys():
                    # do NOT truncate
                    inputs.initiation = self._prepare_inputs(inputs.initiation, preproc=preproc, current_horizon=None)

                if "goal_states" in inputs.keys():
                    # do NOT truncate
                    inputs.goal_states = self._prepare_inputs(inputs.goal_states, preproc=preproc,
                                                              current_horizon=None)
                    inputs.goal_states = inputs.goal_states.leaf_apply(lambda arr: 
                                                                       broadcast_dims(arr, [1], [contact_current_horizon]))

            if self._do_init_policy and do_init:
                init_lmp_inputs = inputs >> "initiation"
                # copy over goals
                if "goal_states" not in init_lmp_inputs.keys():
                    init_lmp_inputs.goal_states = lmp_inputs >> "goal_states"
                init_current_horizon = lmp_inputs.leaf_reduce(lambda red, val: max(red, val.shape[1]), seed=1)

        with timeit("contact_lmp/lmp_forward"):
            # lmp_inputs.leaf_apply(lambda arr: arr.shape).pprint()
            if "run_goal_select" not in kwargs.keys():
                kwargs['run_goal_select'] = not self._goal_sampling or not sample_first
            lmp_outputs = super(PLATOGroupedModel, self).forward(lmp_inputs, preproc=preproc, postproc=postproc,
                                                                 run_prepare=False, run_all=run_all,
                                                                 plan_posterior=plan_posterior,
                                                                 model_outs=model_outs, run_policy=run_policy,
                                                                 current_horizon=contact_current_horizon,
                                                                 **kwargs)
            model_outs.combine(lmp_outputs)

        if self._do_init_policy and do_init:
            with timeit("contact_lmp/initiation_lmp_forward"):
                # run lfp policy, but condition on desired contact affordance.
                # init_current_horizon = init_lmp_inputs.leaf_reduce(max_shape1, seed=1)
                # at least one will be here.
                init_lmp_outs = (model_outs < ['embedding', 'plan_posterior_sample', 'plan_prior_sample']).leaf_copy()
                if self._detach_init_plan:
                    for key, arr in (init_lmp_outs < ['plan_posterior_sample', 'plan_prior_sample']).leaf_items():
                        if isinstance(arr, torch.distributions.Distribution):
                            arr = detach_normal(arr)  # only supports Normal or independent normals
                        else:
                            # tensor
                            arr = arr.detach()
                        init_lmp_outs[key] = arr
                init_lmp_outs = super(PLATOGroupedModel, self).forward(init_lmp_inputs, preproc=preproc,
                                                                       postproc=postproc,
                                                                       run_prepare=False, run_plan=False,
                                                                       run_all=run_all,
                                                                       plan_posterior=plan_posterior,
                                                                       model_outs=init_lmp_outs,
                                                                       current_horizon=init_current_horizon,
                                                                       run_goal_select=not self._goal_sampling \
                                                                                       or not sample_first)
                model_outs.initiation.combine(init_lmp_outs)

        return self._postproc_fn(inputs, model_outs) if postproc else model_outs

    def _get_policy_outputs(self, inputs, outputs, model_outputs, current_horizon=None, use_mask=True):
        """
        returns the policy outputs (current_horizon - 1), aligned with the contact sample

        :param inputs: raw inputs (B x maxH)
        :param outputs:
        :param model_outputs:
        :param current_horizon:
        :return:
        """
        outs = outputs.leaf_copy()

        # sampling_idxs = model_outputs >> sampling_idxs_key  # (B x current_horizon)
        # assert current_horizon is not None
        # assert sampling_idxs.shape[1] == current_horizon, "Incorrect sampling window horizon dimension"

        # flatten_fn = lambda arr: split_dim(combine_dims(arr, 0, 2)[sampling_idxs], 0,
        #               (arr.shape[0], len(sampling_idxs) // arr.shape[0]))
        new_outs = inputs > list(self.action_names)
        # new_outs.leaf_apply(lambda arr: arr.shape).pprint()
        new_outs = new_outs.leaf_apply(lambda arr: arr[:, :-1])  # skip last step
        if current_horizon is not None:
            new_outs.leaf_assert(lambda arr: arr.shape[1] == current_horizon - 1)
        # if use_mask and inputs.has_leaf_key("padding_mask") and \
        #         model_outputs >> "sampler/ndding_mask" is not None:
        #     new_outs["padding_mask"] = (model_outputs >> "sampler/sampling_padding_mask")[:, :-1]  # skip last step to match (even if not really last step)
        return outs & new_outs

    def additional_losses(self, model_outs, inputs, outputs, i=0, writer=None, writer_prefix="", current_horizon=None):
        losses, extra_scalars = super(PLATOGroupedModel, self).additional_losses(model_outs, inputs, outputs, i=i,
                                                                                 writer=writer,
                                                                                 writer_prefix=writer_prefix,
                                                                                 current_horizon=current_horizon)

        if self._do_init_policy:
            # model_outs.leaf_apply(lambda arr: None).pprint()
            init_model_outs = (model_outs >> "initiation").leaf_copy()
            # init_model_outs.sampler = model_outs >> "sampler/initiation"  # for action_loss to know the true input window if it cares.
            init_model_outs.combine(init_model_outs >> "plan_posterior_policy")

            # logger.debug(init_model_outs.leaf_shapes().pprint(ret_string=True))

            # initiation policy outputs, don't require padding mask over loss.
            outs = self._get_policy_outputs(inputs >> "initiation", outputs, model_outs, current_horizon=current_horizon)

            # these inputs are wrong!
            initiation_policy_posterior_loss = self.action_loss_fn(self, init_model_outs, inputs, outs,
                                                                   i=i, writer=writer,
                                                                   writer_prefix=writer_prefix + "posterior/initiation/")

            weights = 1
            if self._init_discount < 1.:
                # (B, H)
                raise NotImplementedError

            # write the prior loss here, since it doesn't get optimized directly.
            if writer is not None:
                init_model_outs.combine(init_model_outs >> "plan_prior_policy")

                # write the prior policy
                initiation_policy_prior_loss = self.action_loss_fn(self, init_model_outs, inputs, outs,
                                                                   i=i, writer=writer,
                                                                   writer_prefix=writer_prefix + "prior/initiation/")
                if isinstance(weights, torch.Tensor):
                    writer.add_scalar(writer_prefix + "posterior/initiation/unweighted_policy_loss",
                                      initiation_policy_posterior_loss.mean().item(), i)
                    writer.add_scalar(writer_prefix + "prior/initiation/unweighted_policy_loss",
                                      initiation_policy_prior_loss.mean().item(), i)

                writer.add_scalar(writer_prefix + "prior/initiation/policy_loss", (weights * initiation_policy_prior_loss).mean().item(), i)

            # gets added
            losses['posterior/initiation/policy_loss'] = (self._beta_init, weights * initiation_policy_posterior_loss)
            # returns losses, extra_scalars

        return losses, extra_scalars

    # @property
    # def grasp_policy(self):
    #     return self._grasp_policy
    #
    # @property
    # def contact_sampler(self):
    #     return self._contact_sampler


if __name__ == '__main__':
    pass
