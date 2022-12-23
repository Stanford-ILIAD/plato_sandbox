"""
Multinomial CEM
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.experiments import logger
from sbrl.policies.multinomial_cem_policy import MultinomialCEMPolicy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--num_action_samples', type=int, default=1024)
    parser.add_argument('--noise_scale', type=float, default=0.33)
    parser.add_argument('--shrink_noise_scale', type=float, default=0.5)
    parser.add_argument('--energy_key', type=str, default="energy")
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--autoregressive', action='store_true')
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    max_iters = prms >> "max_iters"

    # model should set this
    optimize_action_names = common_params << 'optimize_action_names'
    if optimize_action_names is not None:
        logger.debug(f"MultinomialCEM will optimize names: {optimize_action_names}")


    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=MultinomialCEMPolicy,
        params=AttrDict(
            max_iters=max_iters,
            num_action_samples=prms >> "num_action_samples",
            optimize_action_names=optimize_action_names,
            noise_scale=prms >> "noise_scale",
            shrink_noise_scale=prms >> "shrink_noise_scale",
            energy_key=prms >> "energy_key",
            autoregressive=prms >> "autoregressive",
            timeout=prms >> "timeout",
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
