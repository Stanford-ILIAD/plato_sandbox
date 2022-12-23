from argparse import ArgumentParser

from sbrl.models.lmp.lmp_grouped import LMPGroupedModel
from sbrl.models.lmp.model_architectures import get_default_lmp_posterior_params, get_default_lmp_prior_params, \
    get_default_lmp_policy_params
from sbrl.models.lmp.play_helpers import get_play_lmp_selector_models, get_policy_preproc_fn, get_vq_plan_dist_fn, \
    get_plan_dist_fn, get_fixed_posterior_plan_dist_fn
from sbrl.utils.config_utils import bool_cond_add_to_exp_name
from sbrl.utils.loss_utils import mae_err_fn, pose_mae_err_fn, get_default_nll_loss_fn
from sbrl.utils.param_utils import LayerParams
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--plan_name", type=str, default='plan')
    parser.add_argument("--plan_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--posterior_hidden_size", type=int, default=None)
    parser.add_argument("--proposal_width", type=int, default=128)
    parser.add_argument("--prior_num_mix", type=int, default=1)
    parser.add_argument("--prior_mix_temp", type=int, default=0)  # default, no analysis of the "best"
    parser.add_argument("--beta", type=float, default=1e-2, help="weight of plan reg. loss relative to BC objective.")
    parser.add_argument("--include_goal_proprio", action='store_true')
    parser.add_argument("--single_grab", action='store_true')
    parser.add_argument("--do_grab_norm", action='store_true')
    parser.add_argument("--do_pose_err", action='store_true')
    parser.add_argument("--no_encode_actions", action='store_true')
    parser.add_argument("--relative_actions", action='store_true')
    parser.add_argument("--include_block_sizes", action='store_true')
    parser.add_argument("--use_real_inputs", action='store_true', help="uses the inputs available in the real world")
    parser.add_argument("--no_encode_objects", action='store_true')
    parser.add_argument("--no_single_head", action='store_true')
    parser.add_argument("--use_policy_dist", action='store_true')
    parser.add_argument("--learned_rnn_init", action='store_true')
    parser.add_argument("--attention_posterior", action='store_true')
    parser.add_argument("--optimize_prior", action='store_true')
    parser.add_argument("--variable_length", action='store_true', help="will do padded_sequences.")
    parser.add_argument("--block_model_training_steps", type=int, default=0)
    parser.add_argument("--block_kl_training_steps", type=int, default=0)
    parser.add_argument("--no_policy_goals", action='store_true')
    parser.add_argument("--no_rnn_policy", action='store_true')
    parser.add_argument("--max_std_actions", action='store_true')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--beta_info", type=float, default=0.)

    # detached posterior distance
    parser.add_argument("--detached_posterior", action='store_true')
    parser.add_argument("--fixed_beta", type=float, default=1e-3)
    parser.add_argument("--block_prior_training_steps", type=int, default=0)
    parser.add_argument("--use_prior_logprob", action='store_true')

    ## VQ
    parser.add_argument("--vq", action='store_true')
    parser.add_argument("--vq_num_codes", type=int, default=128, help="codebook size for VQ")
    parser.add_argument("--vq_num_parts", type=int, default=1, help="num codes per plan")
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--vq_beta_prior", type=float, default=None)
    parser.add_argument("--vq_beta_posterior", type=float, default=1.,
                        help="weight of posterior (VQ) unsupervised objective in plan dist fn")

    return parser


def wrap_get_exp_name(group_name, exp_name_fn):
    def get_exp_name(common_params):
        prms = common_params >> group_name
        NAME = exp_name_fn(common_params)

        if prms >> "max_std_actions":
            NAME += "_maxstd-ac"

        if prms >> "detached_posterior":
            hrfb = str(prms >> 'fixed_beta').replace('.', '_')
            NAME += f"_detpost-fb{hrfb}"
            if prms >> "block_prior_training_steps" > 0:
                NAME += f"-waitpr{prms >> 'block_prior_training_steps'}"
            if prms >> "use_prior_logprob":
                NAME += f"-prnll"

        if prms >> "variable_length":
            NAME += "_varh"

        if prms >> "vq":
            nc = str(prms >> "vq_num_codes")
            bt = str(prms >> "vq_beta").replace('.', '_')
            btp = str(prms >> "vq_beta_posterior").replace('.', '_')
            NAME += f"_vq{nc}"
            if prms >> "vq_num_parts" > 1:
                NAME += f"-p{prms >> 'vq_num_parts'}"

            NAME += f"-bt{bt}-btp{btp}"
            if prms >> "vq_beta_prior" is not None:
                btpr = str(prms >> "vq_beta_prior").replace('.', '_')
                NAME += f"-btpr{btpr}"

        if prms >> "block_model_training_steps" > 0:
            bmt = str(prms >> "block_model_training_steps")
            NAME += f"_trainwait{bmt}"

        if prms >> "block_kl_training_steps" > 0:
            bmt = str(prms >> "block_kl_training_steps")
            NAME += f"_klwait{bmt}"

        bool_cond_add_to_exp_name(NAME, prms, [
            ("optimize_prior", "OPTPR"),
            ("learned_rnn_init", "rnninit"),
            ("include_goal_proprio", "goalproprio"),
            ("no_policy_goals", "nopolicygoal"),
            ("no_rnn_policy", "mlppolicy"),
            ("use_policy_dist", "policydist"),
            ("no_encode_actions", "noactenc"),
            ("relative_actions", "relac"),
            ("no_encode_objects", "noobjenc"),
            ("single_grab", "singlegrab"),
            ("do_grab_norm", "normgrabil"),
            ("include_block_sizes", "bsz"),
            ("use_real_inputs", "realin"),
            ("do_pose_err", "perr"),
        ])

        if (prms >> "prior_num_mix") > 1:
            hr_pmt = str(prms >> "prior_mix_temp").replace('.', '_')
            NAME += f"_prmix{prms >> 'prior_num_mix'}_prtmp{hr_pmt}"
            if prms >> "no_single_head":
                NAME += "_prnsh"

        if (prms >> "dropout") > 0:
            hr_dr = str(prms >> "dropout").replace('.', '_')
            NAME += f"_dr{hr_dr}"

        hr_beta = str(prms >> "beta").replace('.', '_')
        NAME = NAME + f"_p{prms >> 'plan_size'}_prop{prms >> 'proposal_width'}"
        if (prms >> "hidden_size") != 64:
            NAME += f"_hs{prms >> 'hidden_size'}"

        if (prms >> "posterior_hidden_size") is not None and (prms >> "posterior_hidden_size") != (
                prms >> "hidden_size"):
            NAME += f"_phs{prms >> 'posterior_hidden_size'}"

        if (prms >> "beta_info") > 0:
            hr_bi = str(prms >> "beta_info").replace('.', '_')
            NAME += f"_betainf{hr_bi}"

        return NAME + f"_beta{hr_beta}"

    return get_exp_name


def process_params(group_name, common_params):
    assert "model" in group_name, "This is a model spec."

    prms = common_params >> group_name

    utils = common_params >> "utils"  # module
    env_spec_params = common_params >> "env_spec/params"
    env_params = common_params >> "env_train"  # raw

    vel_act = common_params >> "velact"

    # check block sizes are specified in spec if we are going to include them
    assert not (prms >> "include_block_sizes") or \
           'block_sizes' in (env_spec_params >> "param_names") or \
           'objects/size' in (env_spec_params >> "param_names")  # 2D & 3D

    dset = common_params >> "dataset"
    lmp_names_and_sizes = utils.get_default_lmp_names_and_sizes(env_spec_params, prms >> "plan_name",
                                                                prms >> "plan_size",
                                                                prms >> "include_goal_proprio", prms >> "single_grab",
                                                                prms >> "do_grab_norm",
                                                                ENCODE_ACTIONS=not prms >> "no_encode_actions",
                                                                INCLUDE_BLOCK_SIZES=prms >> "include_block_sizes",
                                                                ENCODE_OBJECTS=not prms >> "no_encode_objects",
                                                                POLICY_GOALS=not prms >> "no_policy_goals",
                                                                VEL_ACT=vel_act,
                                                                USE_DRAWER=env_params << "use_drawer" and (
                                                                        "objonly" not in dset),
                                                                NO_OBJECTS="noobj" in dset,
                                                                REAL_INPUTS=prms >> "use_real_inputs")

    if prms >> "use_policy_dist":
        policy_loss_fn = get_default_nll_loss_fn(lmp_names_and_sizes >> "policy_out_names",
                                                       relative=prms >> "relative_actions")
    else:
        err_fn = pose_mae_err_fn if prms >> "do_pose_err" else mae_err_fn
        policy_loss_fn = utils.get_action_loss_fn(lmp_names_and_sizes >> "policy_out_names", prms >> "single_grab",
                                                  prms >> 'do_grab_norm', relative=prms >> "relative_actions",
                                                  vel_act=vel_act, err_fn=err_fn)

    extra_args = {}
    vq_layer = None

    if prms >> "vq":
        # sample from plan "distribution" (get curr vector quantized)
        plan_sample_fn = lambda out: d(plan=(out >> "plan_vq_tuple")[2])

        assert not prms >> "detached_posterior", "not implemented"
        extra_args['plan_num_codes'] = prms >> "vq_num_codes"
        if prms >> "vq_num_parts" > 1:
            extra_args['plan_num_parts'] = prms >> "vq_num_parts"

        # shared, it will be initialized a single time but used multiple times.
        if prms.vq_num_codes > 1:
            assert prms.plan_size % prms.vq_num_parts == 0, f"Not evenly divisible {prms.plan_size}, {prms.vq_num_parts}"
            vq_layer = LayerParams("vq", shared=True, e_dim=prms.plan_size // prms.vq_num_parts, vec_dim=prms.plan_size,
                                   n_embed=prms.vq_num_codes)
        else:
            vq_layer = LayerParams("vq", shared=True, e_dim=prms.plan_size, n_embed=prms.vq_num_codes)

        plan_dist_fn = get_vq_plan_dist_fn(prms >> "plan_name", vq_beta=prms >> "vq_beta",
                                                 beta_post=prms >> "vq_beta_posterior", beta_pr=prms >> "vq_beta_prior")
        assert prms >> "prior_num_mix" == 1, "Not valid: GMM w/ vq vae"
    else:
        # sample from plan distribution
        plan_sample_fn = lambda out: d(plan=out[prms.plan_name + "_dist"].rsample())

        if not prms >> "detached_posterior":
            plan_dist_fn = get_plan_dist_fn(prms >> "plan_name",
                                                  use_gmm_prior=(prms >> "prior_num_mix") > 1)
        else:
            assert prms >> "use_prior_logprob" or (
                    prms >> "prior_num_mix") > 1, "prior logprob only makes sense for GMM"
            plan_dist_fn = get_fixed_posterior_plan_dist_fn(prms >> "plan_name", fixed_beta=prms >> "fixed_beta",
                                                                  use_gmm_prior=(prms >> "prior_num_mix") > 1,
                                                                  block_prior_steps=prms >> "block_prior_training_steps",
                                                                  gmm_logprob=prms >> "use_prior_logprob")
        extra_args['attention_posterior'] = prms >> "attention_posterior"

    DEVICE = common_params >> 'device'

    # models that select the correct inputs and broadcast things, etc
    selector_models = get_play_lmp_selector_models(DEVICE, lmp_names_and_sizes.TRAJECTORY_NAMES,
                                                   lmp_names_and_sizes.PRIOR_GOAL_STATE_NAMES,
                                                   lmp_names_and_sizes.PRIOR_NAMES,
                                                   lmp_names_and_sizes.POLICY_GOAL_STATE_NAMES,
                                                   lmp_names_and_sizes.POLICY_NAMES)

    # posterior, prior, policy
    posterior = get_default_lmp_posterior_params(lmp_names_and_sizes, DEVICE, prms >> "plan_name",
                                                 prms >> "plan_size", prms >> "hidden_size",
                                                 posterior_takes_goal=False,
                                                 attention_posterior=prms >> 'attention_posterior',
                                                 goal_stack_posterior=False, dropout=prms >> 'dropout',
                                                 posterior_hidden_size=prms >> 'posterior_hidden_size',
                                                 vq_layer=vq_layer)

    prior = get_default_lmp_prior_params(lmp_names_and_sizes, DEVICE, prms >> 'plan_name',
                                         prms >> 'plan_size', prms >> 'proposal_width',
                                         prior_mix=prms >> "prior_num_mix", prior_mix_temp=prms >> "prior_mix_temp",
                                         prior_extra_layers=0, dropout=prms >> 'dropout',
                                         vq_layer=vq_layer)

    policy_preproc_fn = get_policy_preproc_fn(prms >> "plan_name",
                                                    lmp_names_and_sizes >> "POLICY_NAMES")

    policy_postproc_fn = utils.get_policy_postproc_fn(env_spec_params,
                                                      lmp_names_and_sizes >> "policy_out_names",
                                                      prms >> "use_policy_dist",
                                                      do_orn_norm=prms >> "do_pose_err")

    policy = get_default_lmp_policy_params(lmp_names_and_sizes, DEVICE, prms >> "hidden_size", prms >> "proposal_width",
                                           policy_preproc_fn, policy_postproc_fn,
                                           use_policy_dist=prms >> 'use_policy_dist',
                                           learned_rnn_init=prms >> 'learned_rnn_init',
                                           dropout=prms >> 'dropout',
                                           no_rnn_policy=prms >> 'no_rnn_policy')

    # put them together
    model_params = d(
        batch_names_to_get=list(
            set(lmp_names_and_sizes.SAVE_NORMALIZATION_NAMES + ["policy_type", "action", "policy_switch"])),
        cls=LMPGroupedModel,
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
            plan_sample_fn=plan_sample_fn,
            plan_dist_fn=plan_dist_fn,
            plan_name=prms.plan_name,
            plan_size=prms.plan_size,
            action_sample_fn=lambda out: d(action=None),
            action_loss_fn=policy_loss_fn,
            beta=prms >> 'beta',
            optimize_prior=prms >> "optimize_prior",
            beta_info=prms >> "beta_info",

            # some training options
            block_model_training_steps=prms >> "block_model_training_steps",
            block_kl_training_steps=prms >> "block_kl_training_steps",
        ),
    )

    if prms >> "variable_length":
        model_params.params.models.posterior.params.mask_name = "padding_mask"
        model_params.params.models.policy.params.mask_name = "padding_mask"
        common_params["allow_padding"] = True  # dataset will use this

    if prms >> "max_std_actions":
        # each action name will have a single std value to normalize each dimension. somewhat risky.
        model_params.params.max_std_normalization_inputs = list(lmp_names_and_sizes >> "policy_out_names")

    common_params[group_name] = common_params[group_name] & model_params
    common_params[group_name].names_and_sizes = lmp_names_and_sizes  # for policy, for example

    if "batch_names_to_get" in model_params.leaf_keys():
        # copy up for dataset
        common_params["batch_names_to_get"] = model_params >> "batch_names_to_get"

    if common_params.has_leaf_key("batch_names_to_get") and "policy_name" in common_params["batch_names_to_get"]:
        common_params["batch_names_to_get"].remove("policy_name")  # this should not be retrieved every time

    # adds more info to the name
    common_params.exp_name = wrap_get_exp_name(group_name, common_params >> "exp_name")

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
