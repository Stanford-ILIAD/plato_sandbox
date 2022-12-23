"""
Default architectures for models to be used by configs.
"""
import torch

from sbrl.models.basic_model import BasicModel
from sbrl.models.function_model import FunctionModel
from sbrl.models.learned_init_rnn_model import LearnedInitRnnModel
from sbrl.models.lmp.play_helpers import get_goal_preproc_fn
from sbrl.models.rnn_model import RnnModel
from sbrl.utils.param_utils import LayerParams, SequentialParams, build_mlp_param_list
from sbrl.utils.python_utils import AttrDict as d
from sbrl.utils.torch_utils import combine_after_dim, concatenate


def check_names_and_sizes(lmp_names_and_sizes: d, required):
    # check for lmp required names
    assert lmp_names_and_sizes.has_leaf_keys(required), lmp_names_and_sizes.leaf_key_missing(required)


def get_default_lmp_posterior_params(ns, device, plan_name, plan_size, hidden_size,
                                     posterior_takes_goal=False, attention_posterior=False,
                                     goal_stack_posterior=False, dropout=0,
                                     posterior_hidden_size=None, vq_layer=None):
    """
    POSTERIOR (encoder) parameters.

    Default is a bidirectional RNN, which optionally takes the goal in addition to TRAJECTORY_NAMES
    Outputs "Z" the latent plan
    """
    # make sure all required keys are there
    check_names_and_sizes(ns, ['TRAJECTORY_NAMES', 'TRAJECTORY_SIZE', 'PRIOR_GOAL_STATE_NAMES', 'PRIOR_GOAL_IN_SIZE'])

    if posterior_hidden_size is None:
        posterior_hidden_size = hidden_size

    # output space is either Gaussian or VectorQuantize
    # quantize layer, might be shared which is why it should be declared elsewhere.
    out_layers = [vq_layer] if vq_layer is not None else \
        [LayerParams("gaussian_dist_cap", params=d(use_log_sig=False, event_dim=1))]
    plan_intermediate_size = plan_size if vq_layer is not None else plan_size * 2

    posterior = d(
        cls=RnnModel,
        params=d(
            model_inputs=ns.TRAJECTORY_NAMES,  # requires the action
            model_output=plan_name + "_dist",
            device=device,
            preproc_fn=lambda inputs: inputs & \
                                      inputs.leaf_filter_keys(ns.TRAJECTORY_NAMES)
                                          .leaf_apply(lambda arr: combine_after_dim(arr, 2)),  # (B x H x D)
            rnn_output_name="rnn_output_plan_recog",
            hidden_name="hidden_plan_recog",
            rnn_before_net=True,

            recurrent_network=LayerParams('gru', input_size=ns.TRAJECTORY_SIZE,
                                          hidden_size=posterior_hidden_size, num_layers=2,
                                          bidirectional=True, batch_first=True, dropout=dropout),
            # rnn outputs (B x Seq x Hidden)
            network=SequentialParams([
                LayerParams("list_select", list_index=-1, dim=1),
                # last sequence element, (B x hidden_size*2)
                LayerParams("linear", in_features=2 * posterior_hidden_size,
                            out_features=2 * posterior_hidden_size, bias=True),
                LayerParams("relu"),  # TODO
                LayerParams("linear", in_features=2 * posterior_hidden_size,
                            out_features=plan_intermediate_size, bias=True),
                *out_layers
            ]),
        ),
    )

    if attention_posterior:
        raise NotImplementedError

    if posterior_takes_goal:
        posterior_in_names = ns.TRAJECTORY_NAMES + [f'goal_states/{name}' for name in ns.PRIOR_GOAL_STATE_NAMES]

        if goal_stack_posterior:
            posterior.params.combine(d(
                preproc_fn=get_goal_preproc_fn(ns.TRAJECTORY_NAMES, ns.PRIOR_GOAL_STATE_NAMES),
                model_inputs=posterior_in_names,
                recurrent_network=LayerParams('gru', input_size=ns.TRAJECTORY_SIZE + ns.PRIOR_GOAL_IN_SIZE,
                                              hidden_size=hidden_size, num_layers=2,
                                              bidirectional=True, batch_first=True, dropout=dropout)
            ))
        else:
            # concats the goal state after RNN pass.
            def merge_fn(rnn_out, pm_outs):
                # rnn out will be tuple
                # print(rnn_out.shape)
                # pm_outs.leaf_apply(lambda arr: arr.shape).pprint()
                last_rnn_out = rnn_out[:, -1]  # (B x hidden_size*2)
                # goals are B x H x ... to start
                last_goals = pm_outs.leaf_apply(lambda arr: combine_after_dim(arr[:, -1], 1))  # (B x pr_goal_dim)
                flat_goal = concatenate(last_goals, ns.PRIOR_GOAL_STATE_NAMES, dim=-1)  # (B x sum of pr_goal_dim)

                return torch.cat([last_rnn_out, flat_goal], dim=-1)

            posterior.params.combine(d(
                preproc_fn=lambda inputs: inputs & \
                                          (inputs > posterior_in_names)
                                              .leaf_apply(lambda arr: combine_after_dim(arr, 2)),  # (B x H x D)
                parallel_model=d(cls=FunctionModel,
                                 params=d(device=device,
                                          forward_fn=lambda model, ins:
                                          (ins >> "goal_states") > ns.PRIOR_GOAL_STATE_NAMES)),
                merge_parallel_outputs_fn=merge_fn,
                network=SequentialParams(
                    build_mlp_param_list(2 * hidden_size + ns.PRIOR_GOAL_IN_SIZE,
                                         [2 * hidden_size, hidden_size, hidden_size, plan_size * 2],
                                         dropout_p=dropout) + out_layers
                ),
            ))
    return posterior


def get_default_lmp_prior_params(ns, DEVICE, plan_name, plan_size, proposal_width, prior_extra_layers=0, dropout=0,
                                 prior_mix=1, prior_mix_temp=0., vq_layer=None):
    """
    PRIOR (encoder)

    Default is MLP that takes (state start and state goal) and outputs (plan)
    """

    # make sure all required keys are there
    check_names_and_sizes(ns, ['PRIOR_NAMES', 'PRIOR_IN_SIZE'])

    if prior_mix > 1:
        raise NotImplementedError('GMM for prior is not implemented yet')

    # output space is either Gaussian or VectorQuantize
    # quantize layer, might be shared which is why it should be declared elsewhere.
    out_layers = [vq_layer] if vq_layer is not None else \
        [LayerParams("gaussian_dist_cap", params=d(use_log_sig=False, event_dim=1))]
    plan_intermediate_size = plan_size if vq_layer is not None else plan_size * 2

    net_extra_layers = [proposal_width] * prior_extra_layers
    prior = d(
        cls=BasicModel,
        params=d(
            normalize_inputs=False,
            normalization_inputs=[],
            model_inputs=ns.PRIOR_NAMES,
            model_output=plan_name + "_dist",
            preproc_fn=lambda inputs: inputs.leaf_filter_keys(ns.PRIOR_NAMES)
                .leaf_apply(lambda arr: combine_after_dim(arr, 1)),  # # (B x D)
            device=DEVICE,
            network=SequentialParams(
                build_mlp_param_list(ns.PRIOR_IN_SIZE,
                                     [proposal_width, proposal_width, proposal_width] + net_extra_layers + [
                                         plan_intermediate_size],
                                     dropout_p=dropout) + out_layers),
        ),
    )
    return prior


def get_default_lmp_policy_params(ns, DEVICE, hidden_size, proposal_width,
                                  policy_preproc_fn, policy_postproc_fn, use_policy_dist=False, learned_rnn_init=False,
                                  dropout=0,
                                  no_rnn_policy=False):
    """
    POLICY (decoder)

    Default is an RNN (with option for MLP) that takes (state, plan, goal) and outputs (action)
    """

    # make sure all required keys are there
    check_names_and_sizes(ns, ['POLICY_NAMES', 'POLICY_IN_SIZE', 'policy_out_size'])

    if not no_rnn_policy:
        policy = d(
            cls=LearnedInitRnnModel if learned_rnn_init else RnnModel,
            params=d(
                model_inputs=ns.POLICY_NAMES,
                model_output="policy_raw",
                preproc_fn=policy_preproc_fn,
                device=DEVICE,
                rnn_output_name="rnn_output_policy",
                hidden_name="hidden_policy",
                rnn_before_net=True,
                # used only by LearnedInitRnnModel
                init_network=SequentialParams(
                    build_mlp_param_list(ns.POLICY_IN_SIZE, [hidden_size // 2, hidden_size, 2 * hidden_size],
                                         dropout_p=dropout) + [
                        LayerParams("split_dim", dim=-1, new_shape=[2, hidden_size]),  # (B x L x H)
                    ]),
                recurrent_network=LayerParams('gru', input_size=ns.POLICY_IN_SIZE,
                                              hidden_size=hidden_size, num_layers=2,
                                              bidirectional=False, batch_first=True, dropout=dropout),
                # outputs (B x Seq x Hidden)
                network=SequentialParams([
                    LayerParams("linear", in_features=hidden_size,
                                out_features=hidden_size, bias=True),  # outputs (B x Seq x A*nbins)
                    LayerParams("relu"),
                    LayerParams("linear", in_features=hidden_size, out_features=ns.policy_out_size, bias=True)
                ]),
                postproc_fn=policy_postproc_fn,
            ),
        )

        if use_policy_dist:
            policy.params.network = SequentialParams([
                LayerParams("linear", in_features=hidden_size,
                            out_features=hidden_size, bias=True),  # outputs (B x Seq x A*nbins)
                LayerParams("relu"),
                LayerParams("linear", in_features=hidden_size, out_features=2 * ns.policy_out_size, bias=True),
                LayerParams("gaussian_dist_cap", params=d(use_log_sig=False, event_dim=0)),  # each dim is independent
            ])
    else:
        assert not use_policy_dist, "Not implemented"
        policy = d(
            cls=BasicModel,
            params=d(
                model_inputs=ns.POLICY_NAMES,
                model_output="policy_raw",
                preproc_fn=policy_preproc_fn,
                device=DEVICE,
                # outputs (B x Seq x Hidden)
                network=SequentialParams(
                    build_mlp_param_list(ns.POLICY_IN_SIZE, [2 * proposal_width] * 3 + [ns.policy_out_size],
                                         dropout_p=dropout)),
                postproc_fn=policy_postproc_fn,
            ),
        )
    return policy
