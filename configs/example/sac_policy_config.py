"""
SAC policy
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.models.sac.sac import SACModel
from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--actor_no_sample_action", action='store_true', help="False means sample from action_dist, else get mean")
    parser.add_argument("--timeout", type=int, default=0)
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    # training policy forward
    actor_policy_model_forward_fn = SACModel.get_default_policy_model_forward_fn(d(SAMPLE_ACTION=not prms >> "actor_no_sample_action"))

    # fill in the model class and params (instantiated later)
    common_params[group_name] = common_params[group_name] & d(
        cls=BasicPolicy,
        params=d(
            timeout=prms >> "timeout",  # nonzero will cause terminated = True after this many get_action calls.
            policy_model_forward_fn=actor_policy_model_forward_fn,
        ),
    )
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
