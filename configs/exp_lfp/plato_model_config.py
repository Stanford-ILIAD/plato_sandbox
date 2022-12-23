from argparse import ArgumentParser

from sbrl.models.lmp.plato_grouped import PLATOGroupedModel
from sbrl.models.lmp.model_architectures import get_default_lmp_posterior_params, get_default_lmp_prior_params, \
    get_default_lmp_policy_params
from sbrl.models.lmp.play_helpers import get_play_lmp_selector_models, get_fixed_posterior_plan_dist_fn, \
    get_plan_dist_fn, get_policy_preproc_fn
from sbrl.utils.config_utils import bool_cond_add_to_exp_name
from sbrl.utils.python_utils import AttrDict as d, get_with_default
from sbrl.utils.loss_utils import pose_mae_err_fn, mae_err_fn, get_default_nll_loss_fn


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--plan_name", type=str, default='plan')
    parser.add_argument("--plan_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--posterior_hidden_size", type=int, default=None)
    parser.add_argument("--proposal_width", type=int, default=128)
    parser.add_argument("--prior_num_mix", type=int, default=1)
    parser.add_argument("--prior_mix_temp", type=int, default=0)  # default, no analysis of the "best"
    parser.add_argument("--prior_extra_layers", type=int, default=0)  # default, no analysis of the "best"
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--include_goal_proprio", action='store_true')
    parser.add_argument("--single_grab", action='store_true')
    parser.add_argument("--do_grab_norm", action='store_true')
    parser.add_argument("--do_pose_err", action='store_true')
    parser.add_argument("--use_targ_quat", action='store_true')
    parser.add_argument("--no_encode_actions", action='store_true')
    parser.add_argument("--include_block_sizes", action='store_true')
    parser.add_argument("--use_real_inputs", action='store_true', help="uses the inputs available in the real world")
    parser.add_argument("--no_encode_objects", action='store_true')
    parser.add_argument("--prior_objects_only", action='store_true')
    parser.add_argument("--relative_actions", action='store_true')
    parser.add_argument("--no_single_head", action='store_true')
    parser.add_argument("--use_policy_dist", action='store_true')
    parser.add_argument("--no_policy_goals", action='store_true')
    parser.add_argument("--no_rnn_policy", action='store_true')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--block_kl_training_steps", type=int, default=0)

    # detached posterior distance fn
    parser.add_argument("--detached_posterior", action='store_true')
    parser.add_argument("--fixed_beta", type=float, default=1e-3, help="posterior needs some regularization...")
    parser.add_argument("--block_prior_training_steps", type=int, default=0)
    parser.add_argument("--use_prior_logprob", action='store_true')

    # contact specific args
    parser.add_argument("--beta_init", type=float, default=None)
    parser.add_argument("--init_discount", type=float, default=1.)
    # parser.add_argument("--no_far_out_goals", action='store_true')
    parser.add_argument("--goal_stack_posterior", action='store_true')
    parser.add_argument("--do_initiation", action='store_true')
    parser.add_argument("--detach_init_plan", action='store_true')
    # parser.add_argument("--full_contact_seq", action='store_true')
    # parser.add_argument("--use_true_contact", action='store_true')
    parser.add_argument("--no_contact_policy", action='store_true')

    # contact stuff that the dataset will use

    parser.add_argument("--sample_goal_start", action='store_true',
                        help="goal will be the end of the sampled window (no sampling)")
    parser.add_argument("--sample_post_goals_only", action='store_true')
    parser.add_argument("--sample_interaction_goals_only", action='store_true')
    parser.add_argument("--soft_boundary_length", type=int, default=0)
    parser.add_argument("--init_soft_boundary_length", type=int, default=None)

    # TODO below
    # parser.add_argument("--init_skew_sampling", action='store_true', default=None)
    # parser.add_argument("--init_window_size", type=int, default=None)

    return parser


def wrap_get_exp_name(group_name, exp_name_fn):
    def get_exp_name(common_params):
        prms = common_params >> group_name
        NAME = exp_name_fn(common_params) + "_fast-clfp"

        if prms >> "detached_posterior":
            hrfb = str(prms >> 'fixed_beta').replace('.', '_')
            NAME += f"_detpost-fb{hrfb}"
            if prms >> "block_prior_training_steps" > 0:
                NAME += f"-waitpr{prms >> 'block_prior_training_steps'}"
            if prms >> "use_prior_logprob":
                NAME += "-prnll"

        if prms >> "block_kl_training_steps" > 0:
            bmt = str(prms >> "block_kl_training_steps")
            NAME += f"_klwait{bmt}"

        if prms >> "beta_init" is not None:
            hr_binit = str(prms >> "beta_init").replace('.', '_')
            NAME += f"_betainit{hr_binit}"

        if (prms >> "init_discount") < 1.:
            hr_dinit = str(prms >> "init_discount").replace('.', '_')
            NAME += f"_gammainit{hr_dinit}"

        if prms >> "soft_boundary_length" > 0:
            NAME += f"_softbnd{prms >> 'soft_boundary_length'}"

        if prms >> "init_soft_boundary_length" is not None:
            NAME += f"_initbnd{prms >> 'init_soft_boundary_length'}"

        if prms >> "do_initiation":
            NAME += "_initpol"
            if prms >> "detach_init_plan":
                NAME += "-dtplan"

        if prms >> "no_contact_policy":
            NAME += "_nocpol"

        if not prms >> "sample_goal_start":
            NAME += "_goalsample"

            if prms >> "sample_post_goals_only":
                NAME += "-c2e"  # contact_end to end
            elif prms >> "sample_interaction_goals_only":
                NAME += "-w2c"  # window_end to contact_end
            else:
                NAME += "-w2e"  # window_end to end

            # only makes sense if goal sampling (usually far-out)
            if prms >> "goal_stack_posterior":
                NAME += "_stackpost"

        NAME = bool_cond_add_to_exp_name(NAME, prms, [
            ("include_goal_proprio", "goalproprio"),
            ("no_policy_goals", "nopolicygoal"),
            ("no_rnn_policy", "mlppolicy"),
            ("use_policy_dist", "policydist"),
            ("no_encode_actions", "noactenc"),
            ("no_encode_objects", "noobjenc"),
            ("prior_objects_only", "probjonly"),
            ("relative_actions", "relac"),
            ("single_grab", "singlegrab"),
            ("do_grab_norm", "normgrabil"),
            ("include_block_sizes", "bsz"),
            ("use_real_inputs", "realin"),
            ("do_pose_err", "perr"),
            ("use_targ_quat", "tqt")])

        if (prms >> "prior_num_mix") > 1:
            hr_pmt = str(prms >> "prior_mix_temp").replace('.', '_')
            NAME += f"_prmix{prms >> 'prior_num_mix'}_prtmp{hr_pmt}"
            if prms >> "no_single_head":
                NAME += "_prnsh"

        if (prms >> "prior_extra_layers") > 0:
            NAME += f"_prextra{prms >> 'prior_extra_layers'}"

        if (prms >> "dropout") > 0:
            hr_dr = str(prms >> "dropout").replace('.', '_')
            NAME += f"_dr{hr_dr}"

        hr_beta = str(prms >> "beta").replace('.', '_')
        NAME += f"_p{prms >> 'plan_size'}_prop{prms >> 'proposal_width'}"

        if (prms >> "hidden_size") != 64:
            NAME += f"_hs{prms >> 'hidden_size'}"

        if (prms >> "posterior_hidden_size") is not None and (prms >> "posterior_hidden_size") != (
                prms >> "hidden_size"):
            NAME += f"_phs{prms >> 'posterior_hidden_size'}"

        return f"{NAME}_beta{hr_beta}"

    return get_exp_name


def process_params(group_name, common_params):
    assert "model" in group_name, "This is a model spec."

    prms = common_params >> group_name

    utils = common_params >> "utils"  # module
    env_spec_params = common_params >> "env_spec/params"
    env_prms = common_params >> "env_train"

    DEVICE = common_params >> 'device'

    # check block sizes are specified in spec if we are going to include them
    assert not (prms >> "include_block_sizes") or \
           'block_sizes' in (env_spec_params >> "param_names") or \
           'objects/size' in (env_spec_params >> "param_names")

    dset = common_params >> "dataset"
    lmp_names_and_sizes = utils.get_default_lmp_names_and_sizes(env_spec_params, prms >> "plan_name",
                                                                prms >> "plan_size",
                                                                prms >> "include_goal_proprio", prms >> "single_grab",
                                                                prms >> "do_grab_norm",
                                                                ENCODE_ACTIONS=not prms >> "no_encode_actions",
                                                                INCLUDE_BLOCK_SIZES=prms >> "include_block_sizes",
                                                                ENCODE_OBJECTS=not prms >> "no_encode_objects",
                                                                PRIOR_OBJECTS_ONLY=prms >> "prior_objects_only",
                                                                POLICY_GOALS=not prms >> "no_policy_goals",
                                                                USE_DRAWER=env_prms << "use_drawer" and (
                                                                        "objonly" not in dset),
                                                                NO_OBJECTS="noobj" in dset,
                                                                REAL_INPUTS=prms >> "use_real_inputs",
                                                                TARG_USE_QUAT=prms >> 'use_targ_quat')

    if prms >> "use_policy_dist":
        # TODO mask
        policy_loss_fn = get_default_nll_loss_fn(lmp_names_and_sizes >> "policy_out_names", use_outs=True,
                                                 relative=prms >> "relative_actions")
    else:
        err_fn = pose_mae_err_fn if prms >> "do_pose_err" else mae_err_fn
        policy_loss_fn = utils.get_action_loss_fn(lmp_names_and_sizes >> "policy_out_names", prms >> "single_grab",
                                                  prms >> 'do_grab_norm',
                                                  use_outs=True, relative=prms >> "relative_actions", err_fn=err_fn,
                                                  mask_name=None)  # mask_name if prms >> "full_contact_seq" else None

    if not (prms >> "detached_posterior"):
        plan_dist_fn = get_plan_dist_fn(prms >> "plan_name",
                                        use_gmm_prior=(prms >> "prior_num_mix") > 1)
    else:
        # posterior is detached from KL term.
        assert prms >> "use_prior_logprob" or (prms >> "prior_num_mix") > 1, "prior logprob only makes sense for GMM"
        plan_dist_fn = get_fixed_posterior_plan_dist_fn(prms >> "plan_name", fixed_beta=prms >> "fixed_beta",
                                                        use_gmm_prior=(prms >> "prior_num_mix") > 1,
                                                        block_prior_steps=prms >> "block_prior_training_steps",
                                                        gmm_logprob=prms >> "use_prior_logprob")

    # models that select the correct inputs and broadcast things, etc
    selector_models = get_play_lmp_selector_models(DEVICE, lmp_names_and_sizes.TRAJECTORY_NAMES,
                                                   lmp_names_and_sizes.PRIOR_GOAL_STATE_NAMES,
                                                   lmp_names_and_sizes.PRIOR_NAMES,
                                                   lmp_names_and_sizes.POLICY_GOAL_STATE_NAMES,
                                                   lmp_names_and_sizes.POLICY_NAMES)

    # posterior, prior, policy
    posterior = get_default_lmp_posterior_params(lmp_names_and_sizes, DEVICE, prms >> "plan_name",
                                                 prms >> "plan_size", prms >> "hidden_size",
                                                 dropout=prms >> 'dropout',
                                                 posterior_hidden_size=prms >> 'posterior_hidden_size')

    prior = get_default_lmp_prior_params(lmp_names_and_sizes, DEVICE, prms >> 'plan_name',
                                         prms >> 'plan_size', prms >> 'proposal_width',
                                         prior_mix=prms >> "prior_num_mix", prior_mix_temp=prms >> "prior_mix_temp",
                                         prior_extra_layers=0, dropout=prms >> 'dropout')

    policy_preproc_fn = get_policy_preproc_fn(prms >> "plan_name", lmp_names_and_sizes >> "POLICY_NAMES")

    policy_postproc_fn = utils.get_policy_postproc_fn(env_spec_params,
                                                      lmp_names_and_sizes >> "policy_out_names",
                                                      prms >> "use_policy_dist",
                                                      do_orn_norm=prms >> "do_pose_err")

    policy = get_default_lmp_policy_params(lmp_names_and_sizes, DEVICE, prms >> "hidden_size", prms >> "proposal_width",
                                           policy_preproc_fn, policy_postproc_fn,
                                           use_policy_dist=prms >> 'use_policy_dist',
                                           dropout=prms >> 'dropout',
                                           no_rnn_policy=prms >> 'no_rnn_policy')
    # put them together
    model_params = d(
        batch_names_to_get=list(
            set(lmp_names_and_sizes.SAVE_NORMALIZATION_NAMES + ["policy_type", "action", "policy_switch"])),
        cls=PLATOGroupedModel,
        params=d(
            horizon=common_params >> 'horizon',
            min_horizon=common_params >> 'min_horizon',
            device=DEVICE,
            normalize_inputs=True,
            normalization_inputs=lmp_names_and_sizes.NORMALIZATION_NAMES,
            save_normalization_inputs=lmp_names_and_sizes.SAVE_NORMALIZATION_NAMES,
            action_names=lmp_names_and_sizes.policy_out_names,  # outputs of the model
            models=d(
                goal_selector=selector_models >> "goal_selector",
                posterior_input_selector=selector_models >> "posterior_input_selector",
                prior_input_selector=selector_models >> "prior_input_selector",
                policy_input_selector=selector_models >> "policy_input_selector",
                posterior=posterior,
                prior=prior,
                policy=policy,
            ),
            plan_sample_fn=lambda out: d(plan=out[prms.plan_name + "_dist"].rsample()),
            plan_dist_fn=plan_dist_fn,
            plan_name=prms.plan_name,
            plan_size=prms.plan_size,
            action_sample_fn=lambda out: d(action=None),
            action_loss_fn=policy_loss_fn,
            beta=prms >> 'beta',

            # some training options
            block_kl_training_steps=prms >> "block_kl_training_steps",

            # contact specific options
            beta_init=prms >> "beta_init",
            do_init_policy=prms >> "do_initiation",  # initiation and grasp are the same thing i guess
            detach_init_plan=prms >> "detach_init_plan",  # whether or not plan allows gradients for init policy
            init_discount=prms >> "init_discount",  # loss weighting for init window, based on dist from contact start.
            no_contact_policy=prms >> "no_contact_policy",

            # dataset uses these parameters to know how to sample, make sure to set these if goal_sampling = True
            sampling_params=d(
                sample_goals=True,  # we always want to get the goal.
                sample_pre_window=prms >> "do_initiation",
                sample_goal_start=prms >> "sample_goal_start",
                sample_post_goals_only=prms >> "sample_post_goals_only",
                sample_interaction_goals_only=prms >> "sample_interaction_goals_only",
                pre_window_key_prefix="initiation",
                goal_key_prefix="goal_states",
                soft_boundary_length=prms >> "soft_boundary_length",
                init_soft_boundary_length=get_with_default(prms, "init_soft_boundary_length",
                                                           prms >> "soft_boundary_length"),
            )
        ),
    )

    common_params[group_name] = common_params[group_name] & model_params
    common_params[group_name].names_and_sizes = lmp_names_and_sizes  # for policy, for example

    if "batch_names_to_get" in model_params.leaf_keys():
        # copy up for dataset, and we need to add block_contact for contact based parsing
        common_params["batch_names_to_get"] = (model_params >> "batch_names_to_get") + utils.contact_names

    # adds more info to the name
    common_params.exp_name = wrap_get_exp_name(group_name, common_params >> "exp_name")

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
