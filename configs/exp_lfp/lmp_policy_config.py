from argparse import ArgumentParser

import numpy as np
import torch

from sbrl.envs.block_real.real_robot_env import RealRobotEnv
from sbrl.experiments import logger
from sbrl.models.lmp.plato_grouped import PLATOGroupedModel
from sbrl.models.lmp.future_contact_lmp_grouped import FutureContactLMPGroupedModel
from sbrl.models.lmp.lmp_grouped import LMPGroupedModel
from sbrl.policies.memory_policy import MemoryPolicy
from sbrl.utils.python_utils import AttrDict as d, get_with_default


def declare_arguments(parser=ArgumentParser(), prefix=''):
    # oclmp
    parser.add_argument("--" + prefix + "do_object_prior", action='store_true')
    parser.add_argument("--" + prefix + "sample_plans", action='store_true')
    parser.add_argument("--" + prefix + "free_orientation", action='store_true')
    parser.add_argument("--" + prefix + "replan_horizon", type=int, default=None)
    parser.add_argument("--" + prefix + "flush_horizon", type=int, default=None)
    parser.add_argument("--" + prefix + "timeout", type=int, default=0)
    parser.add_argument("--" + prefix + "use_setpoint", action='store_true')
    parser.add_argument("--" + prefix + "use_fast_kp", action='store_true')
    parser.add_argument("--" + prefix + "fill_extra_policy_names", action='store_true')
    parser.add_argument("--" + prefix + "add_goals_in_hor", action='store_true')

    return parser


def process_params(group_name, common_params):
    lmp_model_path = get_with_default(common_params, "lmp_model_path", "model")
    assert (common_params >> lmp_model_path).has_node_leaf_key("names_and_sizes"), "Missing model information"

    prms = common_params >> group_name

    extra_model_kwargs_by_cls = {
        PLATOGroupedModel: {'do_init': False},
        FutureContactLMPGroupedModel: {'run_contact_policy': False},
    }

    cls = common_params >> lmp_model_path + "/cls"
    model_kwargs = extra_model_kwargs_by_cls[cls] if cls in extra_model_kwargs_by_cls.keys() else {}
    logger.debug("extra model args: %s" % model_kwargs)

    replan_horizon = prms >> "replan_horizon"
    flush_horizon = prms >> "flush_horizon"
    replan_horizon = replan_horizon if replan_horizon is not None else common_params >> "horizon"
    flush_horizon = flush_horizon if flush_horizon is not None else replan_horizon
    sample = prms >> "sample_plans"
    # either None or False, the model is recurrent
    recurrent = not (common_params << lmp_model_path + "/no_rnn_policy")
    logger.debug(f"Using replan horizon = {replan_horizon}, "
                 f"recurrent: {recurrent}, flush_horizon = {flush_horizon}")
    
    # this uses memory to progress an (optionally) recurrent policy, using underlying model. classes define these.
    model_cls = (common_params >> lmp_model_path + "/cls")
    lmp_mem_forward_fn = model_cls.get_default_mem_policy_forward_fn(replan_horizon,
                                                                     common_params >> lmp_model_path + 
                                                                     "/names_and_sizes/policy_out_names",
                                                                     recurrent=recurrent,
                                                                     sample_plan=sample,
                                                                     flush_horizon=flush_horizon,
                                                                     add_goals_in_hor=prms >> "add_goals_in_hor")

    utils = common_params >> "utils"

    model_prms = common_params >> lmp_model_path
    policy_out_names = model_prms >> "names_and_sizes/policy_out_names"
    policy_out_norm_names = (model_prms >> "names_and_sizes") << "policy_out_norm_names"
    logger.debug(f"Online policy normalization names: {policy_out_norm_names}")
    
    # see contact_lmp_model for an example, False if not present
    relative = bool(model_prms << "relative_actions")
    if relative:
        logger.info("Policy using relative actions, as specified by the model!")

    mode_key = model_prms << "mode_key"  # This sets the mode automatically.
    if mode_key is not None:
        logger.info(f"Policy will toggle based on mode key = {mode_key}!")

    # Kp is the gain term used when action space is target/position
    kp = utils.ONLINE_FAST_KP if common_params >> f"{group_name}/use_fast_kp" else utils.ONLINE_KP
    logger.info(f"KP: {kp}")
    real = issubclass(common_params >> "env_train/cls", RealRobotEnv)
    if real:
        logger.debug("Using real environment!")

    # policy first rolls out model, then postprocess action(s) as defined by utils
    def mem_policy_model_forward_fn(model: LMPGroupedModel, obs: d, goal: d, memory: d, known_sequence=None, **kwargs):
        obs = obs.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))
        goal = goal.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))
        
        # model/rnn forward
        out = lmp_mem_forward_fn(model, obs, goal, memory, known_sequence=known_sequence, **kwargs, **model_kwargs)

        # optional mode switching between velocity & target/position action
        if mode_key is None:
            vel_act = common_params >> "velact"
        else:
            vel_act = (out >> mode_key).item() > 0.5  # mode is 1 for velact, 0 for posact.

        # postprocessing of action (e.g. target action)
        utils.default_online_action_postproc_fn(model, obs, out, policy_out_names,
                                                relative=relative, vel_act=vel_act, memory=memory,
                                                max_gripper_vel=1000 if real else 150.,
                                                policy_out_norm_names=policy_out_norm_names,
                                                max_orn_vel=5. if "drawer" in obs.keys() else 10.,
                                                use_setpoint=common_params >> f"{group_name}/use_setpoint", Kv_P=kp,
                                                free_orientation=prms >> "free_orientation",
                                                **kwargs)

        # adding in extra info
        if prms >> "fill_extra_policy_names":
            shp = list((out >> policy_out_names[0]).shape[:1])
            if not out.has_leaf_key("policy_type"):
                out['policy_type'] = np.broadcast_to([253], shp + [1])
            if not out.has_leaf_key("policy_name"):
                out['policy_name'] = np.broadcast_to(["lmp_policy"], shp + [1])
            if not out.has_leaf_key("policy_switch"):
                out['policy_switch'] = np.broadcast_to([False], shp + [1])

        return out

    # timeout termination
    is_terminated_fn = None
    if prms >> "timeout" > 0:
        # requires mem policy to keep track of count
        is_terminated_fn = lambda model, obs, goal, mem, **kwargs: \
            False if mem.is_empty() else mem >> "count" >= prms.timeout

    policy_params = d(
        cls=MemoryPolicy,
        params=d(
            policy_model_forward_fn=mem_policy_model_forward_fn,
            is_terminated_fn=is_terminated_fn,
        )
    )
    common_params[group_name] = common_params[group_name] & policy_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
