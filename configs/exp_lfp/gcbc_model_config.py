from argparse import ArgumentParser

from sbrl.experiments import logger
from sbrl.models.basic_model import BasicModel
from sbrl.models.lmp.play_helpers import get_gcbc_preproc_fn
from sbrl.models.rnn_model import RnnModel
from sbrl.utils import config_utils
from sbrl.utils.config_utils import bool_cond_add_to_exp_name
from sbrl.utils.loss_utils import get_default_nll_loss_fn, pose_mae_err_fn, mae_err_fn, mse_err_fn
from sbrl.utils.param_utils import LayerParams, SequentialParams, build_mlp_param_list, add_policy_dist_cap
from sbrl.utils.python_utils import AttrDict as d, get_with_default


def declare_arguments(parser=ArgumentParser(), prefix=""):
    parser.add_argument("--" + prefix + "no_rnn_policy", action='store_true', help="uses mlp network for policy")
    parser.add_argument("--" + prefix + "use_lstm", action='store_true', help="uses lstm network for policy, else gru")
    parser.add_argument("--" + prefix + "use_tanh_out", action='store_true',
                        help="uses tanh for policy output (make sure actions are normalized), else gru")
    parser.add_argument("--" + prefix + "hidden_size", type=int, default=64)
    parser.add_argument("--" + prefix + "policy_size", type=int, default=None,
                        help='mlp after rnn size, same as hidden by default. set to 0 for no mlp')
    parser.add_argument("--" + prefix + "include_goal_proprio", action='store_true')
    parser.add_argument("--" + prefix + "no_objects", action='store_true')
    parser.add_argument("--" + prefix + "use_quat", action='store_true', help='end effector quat, else eul')
    parser.add_argument("--" + prefix + "single_grab", action='store_true')
    parser.add_argument("--" + prefix + "do_grab_norm", action='store_true')
    parser.add_argument("--" + prefix + "relative_actions", action='store_true')
    parser.add_argument("--" + prefix + "include_block_sizes", action='store_true')
    parser.add_argument("--" + prefix + "exclude_velocities", action='store_true')
    parser.add_argument("--" + prefix + "use_real_inputs", action='store_true',
                        help="uses the inputs available in the real world")
    parser.add_argument("--" + prefix + "no_single_head", action='store_true')
    parser.add_argument("--" + prefix + "use_policy_dist", action='store_true')
    parser.add_argument("--" + prefix + "no_sample", action='store_true')
    parser.add_argument("--" + prefix + "policy_sig_min", type=float, default=1e-5)
    parser.add_argument("--" + prefix + "policy_sig_max", type=float, default=1e5)
    parser.add_argument("--" + prefix + "policy_num_mix", type=int, default=1)
    parser.add_argument("--" + prefix + "do_mse_err", action='store_true')
    parser.add_argument("--" + prefix + "do_pose_err", action='store_true')
    parser.add_argument("--" + prefix + "no_goal", action='store_true')
    parser.add_argument("--" + prefix + "use_final_goal", action='store_true')
    parser.add_argument("--" + prefix + "ignore_existing_goal", action='store_true')
    parser.add_argument("--" + prefix + "disable_norm", action='store_true')
    parser.add_argument("--" + prefix + "disable_input_norm", action='store_true')
    parser.add_argument("--" + prefix + "dropout", type=float, default=0)
    return parser


def wrap_get_exp_name(group_name, exp_name_fn):
    def get_exp_name(common_params):
        prms = common_params >> group_name
        NAME = exp_name_fn(common_params) + ("_bc" if prms >> "no_goal" else "_gcbc")

        if prms >> "no_rnn_policy":
            NAME += "-mlp"
        elif prms >> "use_lstm":
            NAME += "-lstm"

        if prms >> "use_tanh_out":
            NAME += "-tanh"

        if prms >> "use_final_goal":
            NAME += "_goalfinal"
            if prms >> "ignore_existing_goal":
                NAME += "-ig"

        NAME = bool_cond_add_to_exp_name(NAME, prms, [("disable_norm", "nonorm"),
                                                      ("disable_input_norm", "noinnorm"),
                                                      ("no_objects", "noobj"),
                                                      ("include_goal_proprio", "goalproprio"),
                                                      ("use_quat", "qt")])

        if prms >> "use_policy_dist":
            hr_min = str(prms >> "policy_sig_min").replace('.', '_')
            hr_max = str(prms >> "policy_sig_max").replace('.', '_')
            NAME += f"_policydist-sig{hr_min}-{hr_max}"

            if (prms >> "policy_num_mix") > 1:
                NAME += f"-gmm{prms.policy_num_mix}"

            if prms >> 'no_sample':
                NAME += f"-nosample"

        NAME = bool_cond_add_to_exp_name(NAME, prms, [("relative_actions", "relac"),
                                                      ("single_grab", "singlegrab"),
                                                      ("do_grab_norm", "normgrabil"),
                                                      ("include_block_sizes", "bsz"),
                                                      ("exclude_velocities", "_no-el"),
                                                      ("use_real_inputs", "realin")])

        if prms >> "do_pose_err":
            NAME += "_perr"
        elif prms >> "do_mse_err":
            NAME += "_l2err"

        if prms >> "hidden_size" != 64:
            NAME += f"_hs{prms >> 'hidden_size'}"
        if prms >> "policy_size" is not None and prms.policy_size != prms.hidden_size:
            NAME += f"_ps{prms >> 'policy_size'}"

        if (prms >> "dropout") > 0:
            hr_dr = str(prms >> "dropout").replace('.', '_')
            NAME += f"_dr{hr_dr}"

        return NAME

    return get_exp_name


def process_params(group_name, common_params):
    assert "model" in group_name, "This is a model spec."

    prms = common_params >> group_name

    utils = common_params >> "utils"  # module
    env_spec_params = common_params >> "env_spec/params"
    env_params = common_params >> "env_train"

    NOOBJ = "noobj" in (common_params >> "dataset") or prms >> 'no_objects'

    # get all the names for the model to use (env specific)
    lmp_names_and_sizes = utils.get_default_lmp_names_and_sizes(env_spec_params, "plan", 32,
                                                                prms >> "include_goal_proprio", prms >> "single_grab",
                                                                prms >> "do_grab_norm",
                                                                VEL_ACT=common_params >> "velact",
                                                                ENCODE_ACTIONS=False,
                                                                ENCODE_OBJECTS=not NOOBJ,
                                                                INCLUDE_BLOCK_SIZES=prms >> "include_block_sizes",
                                                                USE_DRAWER=get_with_default(env_params, "use_drawer",
                                                                                            False),
                                                                NO_OBJECTS=NOOBJ,
                                                                REAL_INPUTS=prms >> "use_real_inputs",
                                                                EXCLUDE_VEL=prms >> "exclude_velocities",
                                                                OBS_USE_QUAT=prms >> "use_quat")

    assert not prms.disable_only_act_norm and not prms.disable_only_obs_norm

    # determine the inputs to the policy
    nsld = env_spec_params >> "names_shapes_limits_dtypes"
    POLICY_NAMES = lmp_names_and_sizes >> "POLICY_NAMES"
    POLICY_NAMES.remove("plan")  # there is no plan in GC BC or BC
    if prms >> "no_goal":
        POLICY_GOAL_STATE_NAMES = []
        POLICY_IN_SIZE = config_utils.nsld_get_dims_for_keys(nsld, POLICY_NAMES)
    else:
        POLICY_GOAL_STATE_NAMES = lmp_names_and_sizes >> "POLICY_GOAL_STATE_NAMES"
        POLICY_IN_SIZE = (lmp_names_and_sizes >> "POLICY_IN_SIZE") - 32

    # outputs of the policy
    policy_out_names = lmp_names_and_sizes >> "policy_out_names"
    policy_out_size = lmp_names_and_sizes >> "policy_out_size"

    # normalization of various keys
    SAVE_NORMALIZE_IN = POLICY_NAMES + policy_out_names
    NORMALIZE_IN = list(POLICY_NAMES)

    logger.debug(f"POLICY_IN_NAMES: {POLICY_NAMES}")

    # prefix w/ {goal} instead of {goal_states}, must be present in data.
    if prms >> "use_final_goal":
        SAVE_NORMALIZE_IN = SAVE_NORMALIZE_IN + [f"goal/{n}" for n in POLICY_GOAL_STATE_NAMES]
        NORMALIZE_IN = NORMALIZE_IN + [f"goal/{n}" for n in POLICY_GOAL_STATE_NAMES]

    # do NOT normalize inputs to policy
    if prms >> "disable_input_norm":
        shared = list(set(lmp_names_and_sizes.NORMALIZATION_NAMES).intersection(POLICY_NAMES))
        for n in shared:
            lmp_names_and_sizes.NORMALIZATION_NAMES.remove(n)
        for n in POLICY_NAMES:
            NORMALIZE_IN.remove(n)
            SAVE_NORMALIZE_IN.remove(n)

    # do NOT normalize outputs of policy (e.g., when using -1 -> 1 action space)
    if prms >> "disable_norm":
        shared = list(set(lmp_names_and_sizes.NORMALIZATION_NAMES).intersection(policy_out_names))
        for n in shared:
            lmp_names_and_sizes.NORMALIZATION_NAMES.remove(n)
        for n in policy_out_names:
            SAVE_NORMALIZE_IN.remove(n)

    if prms >> "use_policy_dist":
        # outputting an action distribution, requires NLL loss
        policy_out_norm_names = [] if prms >> "disable_norm" else list(policy_out_names)
        policy_loss_fn = get_default_nll_loss_fn(lmp_names_and_sizes >> "policy_out_names",
                                                 relative=prms >> "relative_actions",
                                                 policy_out_norm_names=policy_out_norm_names,
                                                 vel_act=common_params >> "velact")
    else:
        # outputting deterministic action, loss is either (1) mae (2) mse or (3) pose
        assert (prms >> "policy_num_mix") <= 1, "Cannot use GMM if policy is deterministic"
        err_fn = pose_mae_err_fn if prms >> "do_pose_err" else mae_err_fn
        err_fn = mse_err_fn if prms >> "do_mse_err" else err_fn
        assert not prms.do_mse_err or not prms.do_pose_err, "mse+pose at once not supported"
        policy_out_norm_names = [] if prms >> "disable_norm" else list(policy_out_names)

        # pose error does its own normalization for orientations.
        if not prms >> "disable_norm" and prms >> "do_pose_err":
            if "target/orientation_eul" in policy_out_norm_names:
                policy_out_norm_names.remove("target/orientation_eul")
            elif "target/ee_orientation_eul" in policy_out_norm_names:
                policy_out_norm_names.remove("target/ee_orientation_eul")
            else:
                raise NotImplementedError(policy_out_norm_names)

        policy_loss_fn = utils.get_action_loss_fn(lmp_names_and_sizes >> "policy_out_names", prms >> "single_grab",
                                                  prms >> 'do_grab_norm', relative=prms >> "relative_actions",
                                                  err_fn=err_fn, vel_act=common_params >> "velact",
                                                  policy_out_norm_names=policy_out_norm_names)

    lmp_names_and_sizes.policy_out_norm_names = policy_out_norm_names

    # parses the goal from the input window.
    policy_preproc_fn = get_gcbc_preproc_fn(prms >> 'no_goal', prms >> 'use_final_goal', common_params >> "device",
                                            POLICY_NAMES, POLICY_GOAL_STATE_NAMES)

    # will extract policy names from the raw model output
    policy_postproc_fn = utils.get_default_policy_postproc_fn(nsld, policy_out_names, raw_out_name="policy_raw",
                                                              use_policy_dist=prms >> "use_policy_dist",
                                                              use_policy_dist_mean=prms >> "no_sample",
                                                              relative=prms >> "relative_actions",
                                                              do_orn_norm=prms >> "do_pose_err")

    hidden_size = prms >> "hidden_size"
    policy_size = hidden_size if prms >> "policy_size" is None else prms.policy_size

    end_layers = [LayerParams('tanh')] if prms >> "use_tanh_out" else []

    if prms >> "no_rnn_policy":
        assert policy_size > 0
        common_params[group_name] = common_params[group_name] & d(
            cls=BasicModel,
            params=d(
                normalize_inputs=not prms >> "disable_input_norm",
                save_normalization_inputs=SAVE_NORMALIZE_IN,
                normalization_inputs=NORMALIZE_IN,
                model_inputs=POLICY_NAMES,
                model_output="policy_raw",
                preproc_fn=policy_preproc_fn,
                device=common_params >> 'device',
                # outputs (B x Seq x Hidden)
                network=SequentialParams(
                    build_mlp_param_list(POLICY_IN_SIZE, [policy_size, policy_size, policy_size, policy_out_size],
                                         dropout_p=prms >> "dropout") + end_layers),
                postproc_fn=policy_postproc_fn,
                loss_fn=policy_loss_fn,
            ),
        )
    else:
        rnn_type = 'lstm' if prms >> "use_lstm" else 'gru'
        mlp_after_rnn_dims = [policy_size, policy_size, policy_out_size]
        if policy_size == 0:
            mlp_after_rnn_dims = [policy_out_size]  # no mlp.
        common_params[group_name] = common_params[group_name] & d(
            cls=RnnModel,
            params=d(
                normalize_inputs=not prms >> "disable_input_norm",
                save_normalization_inputs=SAVE_NORMALIZE_IN,
                normalization_inputs=NORMALIZE_IN,
                model_inputs=POLICY_NAMES,
                model_output="policy_raw",
                preproc_fn=policy_preproc_fn,
                device=common_params >> 'device',

                rnn_output_name="rnn_output_policy",
                hidden_name="hidden_policy",
                rnn_before_net=True,
                tuple_hidden=rnn_type == "lstm",
                recurrent_network=LayerParams(rnn_type, input_size=POLICY_IN_SIZE,
                                              hidden_size=hidden_size, num_layers=2,
                                              bidirectional=False, batch_first=True,
                                              dropout=prms >> "dropout"),
                # outputs (B x Seq x Hidden)
                network=SequentialParams(build_mlp_param_list(hidden_size, mlp_after_rnn_dims,
                                                              dropout_p=prms >> "dropout") + end_layers),
                postproc_fn=policy_postproc_fn,
                loss_fn=policy_loss_fn,
            ),
        )

    # defining the policy distribution "cap" at the output of the model, which might be GMM.
    num_mix = prms >> "policy_num_mix"
    if prms >> "use_policy_dist":
        common_params[group_name].params.network = add_policy_dist_cap(common_params[group_name].params.network,
                                                                       num_mix, prms.use_tanh_out, hidden_size,
                                                                       policy_out_size, prms.policy_sig_min,
                                                                       prms.policy_sig_max)

    # add model details to experiment name (wraps the function)
    common_params.exp_name = wrap_get_exp_name(group_name, common_params >> "exp_name")
    common_params[group_name].names_and_sizes = lmp_names_and_sizes  # for policy, for example

    SAVE_NORMALIZATION_NAMES = lmp_names_and_sizes >> "SAVE_NORMALIZATION_NAMES"
    extra_action_names = lmp_names_and_sizes < ["action_names", "waypoint_names"]
    extra_action_names = list(v for x in extra_action_names.leaf_values() for v in x)

    # setting common_params.batch_names_to_get, which a dataset can use to know what limited set of keys to extract.
    common_params["batch_names_to_get"] = list(
        set(SAVE_NORMALIZATION_NAMES + ["policy_type", "action", "policy_switch"] + extra_action_names))

    if prms >> "use_final_goal":
        common_params["batch_names_to_get"] = common_params["batch_names_to_get"] + env_spec_params.final_names

        if prms >> "ignore_existing_goal" and common_params.has_leaf_key("dataset_train/load_ignore_prefixes"):
            common_params["dataset_train/load_ignore_prefixes"].append("goal/")

    if "policy_name" in common_params["batch_names_to_get"]:
        common_params["batch_names_to_get"].remove("policy_name")  # this should not be retrieved every time

    if "image" in env_spec_params.observation_names:
        common_params["batch_names_to_get"].append("image")  # careful with this

    if "mode" in env_spec_params.observation_names:
        common_params['batch_names_to_get'].append("mode")

    if "real" in env_spec_params.param_names:
        common_params['batch_names_to_get'].append("real")

    # target is like long term action
    target_names = [tn for tn in env_spec_params.action_names if tn.startswith('target/')]

    common_params['batch_names_to_get'].extend(target_names)

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
