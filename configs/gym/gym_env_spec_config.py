from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import gym
import numpy as np

from sbrl.envs.gym_env import SUPPORTED
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument('--use_next_prefix', action='store_true')
    parser.add_argument('--use_goal_obs', action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert group_name == "env_spec"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    # access to all the params for the current experiment here.
    env_prms = common_params >> "env_train"

    gym_env_spec = SUPPORTED[env_prms >> "env_type"]

    # make the env to get the sizes... avoids hard coding per env but is a bit dumb since we do this twice.
    env_wrap = gym.make(gym_env_spec.id)
    if prms >> "use_next_prefix":
        next_name = "next_obs"
    else:
        next_name = "next/obs"

    base_nsld = [
        ('obs', tuple(env_wrap.observation_space.shape), (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
        ('goal/obs', tuple(env_wrap.observation_space.shape), (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
        (next_name, tuple(env_wrap.observation_space.shape), (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
        ('reward', (1,), (-np.inf, np.inf), np.float32),
        ('action', tuple(env_wrap.action_space.shape), (env_wrap.action_space.low, env_wrap.action_space.high), env_wrap.action_space.dtype.type),
    ]
    del env_wrap

    # we record these names in dataset  TOD0 'image'
    obs_names = ['obs']
    output_obs_names = [next_name, 'reward']
    action_names = ['action']
    goal_names = ["goal/obs"] if prms >> "use_goal_obs" else []
    param_names = []
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
