from argparse import ArgumentParser

import numpy as np

from sbrl.policies.meta_policy import MetaPolicy
from sbrl.policies.robosuite.robosuite_policies import WaypointPolicy, get_nut_assembly_square_policy_params, \
    get_tool_hang_policy_params
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # parser.add_argument('--use_rotate', action='store_true')
    # parser.add_argument('--no_push_pull', action='store_true', help="disable push pull")
    # parser.add_argument('--no_lift_rot', action='store_true', help="disable lift rotate")
    parser.add_argument('--single_policy', action='store_true', help="no retreat, one policy per eval")
    parser.add_argument('--single_do_retreat', action='store_true', help="use retreat bounds for single policy eval.")
    # parser.add_argument('--use_intermediate_targets', action='store_true',
    #                     help="target action is intermediate, not final, for position")

    parser.add_argument('--random_motion', action='store_true', help="will move with added noise")
    # parser.add_argument('--less_rot', action='store_true', help="sample rot even less.")
    # parser.add_argument('--mug_diverse_grasp', action='store_true', help="diverse grasp for mug.")
    # parser.add_argument('--mug_rot_allow_wall', action='store_true', help="rot should stop at the wall.")
    # # parser.add_argument('--oversample_rot', action='store_true', help="oversample rotation actions")
    # # parser.add_argument('--oversample_tip', action='store_true', help="oversample tip actions, only for rotate_only")
    # # parser.add_argument('--undersample_lift', action='store_true', help="undersample lift actions, to compensate for it being the only upward primitive.")
    # # only relevant if using primitives
    parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[5, 12])
    parser.add_argument('--min_max_policies', type=int, nargs=2, default=[3, 6])
    # parser.add_argument('--smooth_noise', type=float, default=0.)
    # parser.add_argument('--random_slow_prob', type=float, default=0.)
    # parser.add_argument('--stop_prob', type=float, default=0.33)
    # # parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])

    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    prms = common_params >> group_name
    env_prms = common_params >> "env_train"

    mmp = (prms >> "min_max_policies")
    mmr = (prms >> "min_max_retreat")

    # if prms >> "directional" and mmr == [15, 25]:
    #     # changing the default, kinda
    #     mmr = [5, 12]

    if prms >> "single_policy":
        # overrides
        mmp = [1, 2]
        if not prms >> "single_do_retreat":
            mmr = [0, 1]

    polprms = d()
    polprms.max_pos_vel = 0.4  # otherwise things are way too fast

    env_name = env_prms >> "env_name"
    if env_name == "NutAssemblySquare":
        policy_params_fn = get_nut_assembly_square_policy_params
    elif env_name == "ToolHang":
        policy_params_fn = get_tool_hang_policy_params
        polprms.max_pos_vel = 1.0
        polprms.max_ori_vel = 10.0
    else:
        raise NotImplementedError(env_name)

    def policy_next_params_fn(policy_idx: int, model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(*mmp)

        if memory.reset_count < memory.max_iters:
            memory.reset_count += 1

            return 0, policy_params_fn(obs.leaf_apply(lambda arr: arr[:, 0]), goal.leaf_apply(lambda arr: arr[:, 0]),
                                       env=env,
                                       random_motion=prms >> "random_motion")
        else:
            return None, d()

    ALL_POLICIES = [
        d(cls=WaypointPolicy, params=polprms),
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
