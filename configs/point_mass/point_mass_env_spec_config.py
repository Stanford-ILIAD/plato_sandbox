from argparse import ArgumentParser

import numpy as np

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.models.gpt.bet import logger
from sbrl.utils.python_utils import AttrDict, get_with_default


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument('--use_next_prefix', action='store_true')
    parser.add_argument('--use_goal_obs', action='store_true')
    parser.add_argument('--use_modality', action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert group_name == "env_spec"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name
    N = get_with_default(common_params >> "env_train", "num_obstacles", 0)
    obstacle_names = []
    obstacle_param_names = []
    if N > 0:
        logger.debug("Using object position in spec!")
        obstacle_names.append('objects/position')
        obstacle_param_names.append('objects/size')
        # obstacle_param_names.append('objects/contact') # TODO

    extra_names = []
    if prms >> "use_modality":
        extra_names.append("modality")

    # make the env to get the sizes... avoids hard coding per env but is a bit dumb since we do this twice.
    if prms >> "use_next_prefix":
        next_name = "next_obs"
    else:
        next_name = "next/obs"

    base_nsld = [
        ('obs', (4,), (0, 1), np.float32),
        ('objects/position', (N, 2), (0, 1), np.float32),
        ('objects/size', (N,), (0, 1), np.float32),
        ('objects/contact', (N,), (False, True), np.bool),
        (next_name, (4,), (0, 1), np.float32),
        ('reward', (1,), (-np.inf, np.inf), np.float32),
        ('action', (2,), (-1, 1), np.float32),
        ('modality', (1,), (-np.inf, np.inf), np.float32),
    ]

    # we record these names in dataset  TOD0 'image'
    obs_names = ['obs'] + obstacle_names
    output_obs_names = [next_name, 'reward']
    action_names = ['action']
    goal_names = ["goal/obs"] if prms >> "use_goal_obs" else []
    param_names = [] + obstacle_param_names + extra_names
    final_names = []
    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=AttrDict(
            names_shapes_limits_dtypes=base_nsld,
            output_observation_names=output_obs_names,
            observation_names=obs_names,
            action_names=action_names,
            goal_names=goal_names,
            param_names=param_names,
            final_names=final_names,
        )
    )

    common_params.env_spec = common_params.env_spec & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
