"""
SAC model, works with *all* gym environments (and some other specs)
"""

from argparse import ArgumentParser

import numpy as np

# declares this group's parser, and defines any sub groups we need
from sbrl.experiments import logger
from sbrl.models.basic_model import BasicModel
from sbrl.models.sac.critic import DoubleQCritic, REDQCritic
from sbrl.models.sac.sac import SACModel
from sbrl.utils import config_utils
from sbrl.utils.config_utils import nsld_get_row
from sbrl.utils.param_utils import SequentialParams, build_mlp_param_list, LayerParams
from sbrl.utils.python_utils import AttrDict as d


#
#
# def loss_fn(model, model_outputs: d, inputs: d, outputs: d, i=0, writer=None, writer_prefix="", ret_dict=False, **kwargs):
#     B, H = model_outputs.get_one().shape[:2]
#


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--sac_discount", type=float, default=0.99)
    parser.add_argument("--init_temp", type=float, default=0.1)
    parser.add_argument("--actor_size", type=int, default=200)
    parser.add_argument("--critic_size", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--use_redq", action='store_true')
    parser.add_argument("--no_learnable_temp", action='store_true')
    parser.add_argument("--use_next_prefix", action='store_true')
    parser.add_argument("--use_goals", action='store_true')
    parser.add_argument("--action_name", type=str, default='action')
    parser.add_argument("--obs_names_path", type=str, default=None,
                        help='where to look in common_params for the observation_names. if None, will default to using env_spec observations')
    return parser


def wrap_get_exp_name(group_name, exp_name_fn):
    # modifies the experiment name with specific things from this level
    def get_exp_name(common_params):
        prms = common_params >> group_name
        NAME = exp_name_fn(common_params) + "_sac"

        if prms >> "no_learnable_temp":
            NAME += "_nolearntemp"

        if prms >> "use_goals":
            NAME += "_goals"

        if prms >> "use_redq":
            NAME += "_redq"

        if prms >> "actor_size" != 200:
            NAME += f"_as{prms.actor_size}"

        if prms >> "critic_size" != 200:
            NAME += f"_cs{prms.critic_size}"

        if prms >> "num_layers" != 2:
            NAME += f"_depth{prms.num_layers}"

        if prms >> "action_name" != "action":
            NAME += f"_ac-{prms.action_name}"

        return NAME
    return get_exp_name


def process_params(group_name, common_params):
    assert 'model' in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    env_spec_params = common_params >> "env_spec/params"
    prms = common_params >> group_name

    device = common_params >> "device"
    action_name = prms >> 'action_name'
    assert action_name in (env_spec_params >> "action_names"), f"Env spec actions must contain specified action name ({action_name})"

    ACTOR_IN_NAMES = list(env_spec_params >> "observation_names" if prms >> 'obs_names_path' is None else common_params >> prms.obs_names_path)
    for n in ACTOR_IN_NAMES:
        if n not in (env_spec_params.observation_names + env_spec_params.param_names):
            logger.debug(f"Name {n} missing from spec but required by SAC. Adding to observation names.")
            env_spec_params.observation_names.append(n)
    ACTOR_IN_NAMES += list(env_spec_params >> "goal_names") if prms >> "use_goals" else []
    ACTOR_IN_SIZE = config_utils.nsld_get_dims_for_keys(env_spec_params >> "names_shapes_limits_dtypes", ACTOR_IN_NAMES)
    ACTOR_OUT_SIZE = config_utils.nsld_get_dims_for_keys(env_spec_params >> "names_shapes_limits_dtypes", [action_name])
    CRITIC_IN_NAMES = ACTOR_IN_NAMES + [action_name]
    CRITIC_IN_SIZE = ACTOR_IN_SIZE + ACTOR_OUT_SIZE

    logger.debug(f"Using actor input names: {ACTOR_IN_NAMES}")

    oon = env_spec_params.output_observation_names
    nsld = env_spec_params.names_shapes_limits_dtypes
    names = [tup[0] for tup in nsld]
    for k in ACTOR_IN_NAMES:
        # add to nsld if not there
        next_name = f'next/{k}'
        if next_name not in names:
            row = list(nsld_get_row(nsld, k))
            row[0] = next_name
            nsld.append(tuple(row))
        if next_name not in oon:
            oon = oon + [next_name]
            logger.debug(f"--> Filling in {next_name} in env_spec")

    env_spec_params.names_shapes_limits_dtypes = nsld
    env_spec_params.output_observation_names = oon

    N_LAYERS = prms >> 'num_layers'
    USE_REDQ = prms >> 'use_redq'

    # fill in the model class and params (instantiated later)
    common_params[group_name] = common_params[group_name] & d(
        cls=SACModel,
        params=d(
            device=device,
            discount=prms >> "sac_discount",
            learnable_temperature=not prms >> "no_learnable_temp",
            init_temperature=prms >> "init_temp",
            use_nested_next=not prms >> "use_next_prefix",
            action_name=action_name,
            # env_reward_fn=env_reward_fn,
            # actor takes in images, outputs an action
            actor=d(
                cls=BasicModel,
                params=d(
                    normalize_inputs=True,
                    normalization_inputs=ACTOR_IN_NAMES,
                    device=device,
                    model_inputs=ACTOR_IN_NAMES,
                    model_output=f"{action_name}_dist",  # deterministic actor
                    network=SequentialParams(build_mlp_param_list(ACTOR_IN_SIZE, [prms >> 'actor_size'] * N_LAYERS + [2*ACTOR_OUT_SIZE]) + [
                        LayerParams("squashed_gaussian_dist_cap", params=d(sig_min=np.exp(-5), sig_max=np.exp(2)))  # event_dim=0 assumed by sac
                    ])
                )
            ),
            # critic takes in images and action, outputs value fn
            critic=d(
                cls=REDQCritic if USE_REDQ else DoubleQCritic,  # redq --> uses default M=2, N=10
                params=d(
                    device=device,
                    # q1 gets duplicated in each double Q critic (nested models)
                    q1=d(
                        cls=BasicModel,
                        params=d(
                            normalize_inputs=True,
                            normalization_inputs=CRITIC_IN_NAMES,
                            device=device,
                            model_inputs=CRITIC_IN_NAMES,
                            model_output="qval",
                            network=SequentialParams(build_mlp_param_list(CRITIC_IN_SIZE, [prms >> 'critic_size'] * N_LAYERS + [1]))
                        )
                    ),
                ),
            ),
        ),
    )

    # adds more info to the name
    common_params.exp_name = wrap_get_exp_name(group_name, common_params >> "exp_name")

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
