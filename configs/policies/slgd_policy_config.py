"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.experiments import logger
from sbrl.policies.slgd_policy import SLGDPolicy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_action_samples', type=int, default=256)
    parser.add_argument('--noise_scale', type=float, default=0.5)
    parser.add_argument('--init_l_lambda', type=float, default=0.1)
    parser.add_argument('--l_lambda_schedule', type=str, default="poly")
    parser.add_argument('--energy_key', type=str, default="energy")
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    max_iters = prms >> "max_iters"
    sched_type = prms >> "l_lambda_schedule"

    # model should set this
    optimize_action_names = common_params << 'optimize_action_names'
    if optimize_action_names is not None:
        logger.debug(f"SLGD will optimize names: {optimize_action_names}")

    def poly_scale(l_lambda, i):
        # returns SCALE on l_lambda
        l_lambda_final = l_lambda * 1e-4  # TODO parameterize this
        power = 2.0
        new_l = ((1 - i / max_iters) ** power) * (l_lambda - l_lambda_final) + l_lambda_final
        return new_l / l_lambda

    if sched_type == 'none':
        scale_fn = None
    elif sched_type == 'poly':
        scale_fn = poly_scale
    else:
        raise NotImplementedError

    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=SLGDPolicy,
        params=AttrDict(
            max_iters=max_iters,
            num_action_samples=prms >> "num_action_samples",
            optimize_action_names=optimize_action_names,
            noise_scale=prms >> "noise_scale",
            init_l_lambda=prms >> "init_l_lambda",
            l_lambda_scale_fn=scale_fn,
            energy_key=prms >> "energy_key",
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
