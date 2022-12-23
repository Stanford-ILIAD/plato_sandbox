"""
Goal distance reward for the observation.
"""

from argparse import ArgumentParser

from sbrl.sandbox.new_trainer.goal_dist_reward import GoalDistReward
from sbrl.utils.config_utils import nsld_get_lims_for_keys
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--normalize_reward", action='store_true')
    return parser


def process_params(group_name, common_params):
    assert 'reward' in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name
    nsld = common_params >> "env_spec/params/names_shapes_limits_dtypes"
    low, high = nsld_get_lims_for_keys(nsld, ['obs'])[0]

    obs_range = high - low + 1e-11

    normalization_weights = list(obs_range) if prms >> "normalize_reward" else None
    # fill in the model class and params (instantiated later)
    common_params[group_name] = common_params[group_name] & d(
        cls=GoalDistReward,
        params=d(normalization_weights=normalization_weights)
    )
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
