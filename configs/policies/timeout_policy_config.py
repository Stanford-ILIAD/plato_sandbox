"""
times out after n steps.
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--timeout", type=int, default=2)
    parser.add_argument("--run_basic_policy", action='store_true')
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    policy_model_forward_fn = lambda m, o, g, **kwargs: AttrDict()
    if common_params >> f"{group_name}/run_basic_policy":
        policy_model_forward_fn = None

    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=BasicPolicy,
        params=AttrDict(
            policy_model_forward_fn=policy_model_forward_fn,
            timeout=common_params >> f"{group_name}/timeout",
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
