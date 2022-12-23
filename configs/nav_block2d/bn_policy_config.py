from argparse import ArgumentParser

import numpy as np

from sbrl.policies.blocks.maze_navigation_policies import get_basic_nav_policy_params, Waypoint2DPolicy
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # parser.add_argument('--single_policy', action='store_true', help="no retreat, one policy per eval")
    parser.add_argument('--random_motion', action='store_true', help="will move with added noise")
    parser.add_argument('--smooth_noise', type=float, default=0.)
    parser.add_argument('--random_slow_prob', type=float, default=0.)
    parser.add_argument('--stop_prob', type=float, default=0.33)
    parser.add_argument('--no_clip_norm', action='store_true', help='if true, will clip by action env_spec instead of by max_vel.')
    # # parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])

    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    prms = common_params >> group_name
    env_prms = common_params >> "env_train"

    mmp = [1, 2]

    polprms = d(clip_norm=not prms >> "no_clip_norm")

    # polprms.max_vel = 150  # otherwise things are way too fast

    def policy_next_params_fn(policy_idx: int, model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(*mmp)

        if memory.reset_count < memory.max_iters:
            memory.reset_count += 1

            return 0, get_basic_nav_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                                  goal.leaf_apply(lambda arr: arr[:, 0]),
                                                  env=env,
                                                  random_motion=prms >> "random_motion",
                                                  smooth_noise=prms >> "smooth_noise",
                                                  random_slow_prob=prms >> "random_slow_prob",
                                                  stop_prob=prms >> "stop_prob",
                                                  )
        else:
            return None, d()

    ALL_POLICIES = [
        d(cls=Waypoint2DPolicy, params=polprms),
    ]

    policy_params = d(
        cls=MetaPolicy,
        params=d(
            all_policies=ALL_POLICIES,
            next_param_fn=policy_next_params_fn,
        )
    )
    common_params[group_name] = common_params[group_name] & policy_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
