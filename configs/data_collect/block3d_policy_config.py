from argparse import ArgumentParser

from sbrl.policies.blocks.block3d_meta_policies import get_push_pull_meta_policy_params_fn, \
    get_push_pull_toprot_lift_meta_policy_params_fn
from sbrl.policies.blocks.block3d_policies import PushPrimitive, PullPrimitive, TopRotatePrimitive, LiftPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # parser.add_argument('--use_rotate', action='store_true')
    # parser.add_argument('--no_push_pull', action='store_true', help="disable push pull")
    # parser.add_argument('--no_lift_rot', action='store_true', help="disable lift rotate")
    parser.add_argument('--single_policy', action='store_true', help="no retreat, one policy per eval")
    parser.add_argument('--single_do_retreat', action='store_true', help="use retreat bounds for single policy eval.")
    parser.add_argument('--use_intermediate_targets', action='store_true',
                        help="target action is intermediate, not final, for position")
    parser.add_argument('--push_only', action='store_true', help="push prim only")
    parser.add_argument('--no_push', action='store_true', help="remove push prim")
    parser.add_argument('--no_pull', action='store_true', help="remove pull prim")
    parser.add_argument('--do_lift', action='store_true', help="lift primitive with platform env")
    parser.add_argument('--lift_sample', action='store_true', help="lift, sample the direction of motion")
    parser.add_argument('--top_rot_only', action='store_true', help="top_rot prim only")
    parser.add_argument('--use_rotate', action='store_true', help="includes rotation primitive")
    parser.add_argument('--one_dim_only', action='store_true', help="motion along 1 axis only")
    parser.add_argument('--random_motion', action='store_true', help="randomized velocity and time")
    parser.add_argument('--shorter_motion', action='store_true', help="divides time of each motion by two")
    parser.add_argument('--retreat_first', action='store_true', help="retreats first and only if it needs to")
    parser.add_argument('--retreat_xy', action='store_true',
                        help="retreats with variation along x/y, and less so on z axis")
    parser.add_argument('--directional', action='store_true', help="randomized velocity and time")
    parser.add_argument('--uniform_velocity', action='store_true', help="uses true postproc velocities, e.g. setpoint -> vel is a fixed transform")

    parser.add_argument('--more_lift', action='store_true', help="sample lift even more.")
    parser.add_argument('--less_rot', action='store_true', help="sample rot even less.")
    parser.add_argument('--mug_diverse_grasp', action='store_true', help="diverse grasp for mug.")
    parser.add_argument('--mug_rot_allow_wall', action='store_true', help="rot should stop at the wall.")
    # parser.add_argument('--oversample_rot', action='store_true', help="oversample rotation actions")
    # parser.add_argument('--oversample_tip', action='store_true', help="oversample tip actions, only for rotate_only")
    # parser.add_argument('--undersample_lift', action='store_true', help="undersample lift actions, to compensate for it being the only upward primitive.")
    # only relevant if using primitives
    parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[5, 12])
    parser.add_argument('--min_max_policies', type=int, nargs=2, default=[3, 6])
    parser.add_argument('--smooth_noise', type=float, default=0.)
    parser.add_argument('--random_slow_prob', type=float, default=0.)
    parser.add_argument('--stop_prob', type=float, default=0.33)
    # parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])

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

    polprms = d(vel_noise=0., use_intermediate_targets=prms >> "use_intermediate_targets")
    if prms >> "uniform_velocity":
        polprms.max_pos_vel = 0.4  # otherwise things are way too fast

    if prms >> "use_rotate" or prms >> "do_lift":
        policy_next_params_fn = get_push_pull_toprot_lift_meta_policy_params_fn(*mmp, *mmr,
                                                                                do_pull=not prms >> "no_pull" and (not prms >> "push_only") and (
                                                                                    not prms >> "top_rot_only"),
                                                                                do_push=not prms >> "no_push" and not prms >> "top_rot_only",
                                                                                do_toprot=prms >> "use_rotate",
                                                                                do_lift=prms >> "do_lift",
                                                                                one_dim_only=prms >> "one_dim_only",
                                                                                random_motion=prms >> "random_motion",
                                                                                shorter_motion=prms >> "shorter_motion",
                                                                                retreat_first=prms >> "retreat_first",
                                                                                retreat_xy=prms >> "retreat_xy",
                                                                                uniform_velocity=prms >> "uniform_velocity",
                                                                                directional=prms >> "directional",
                                                                                lift_sample_directions=prms >> "lift_sample",
                                                                                smooth_noise=prms >> "smooth_noise",
                                                                                use_mug=env_prms >> "use_mug",
                                                                                more_lift=prms >> "more_lift",
                                                                                less_rot=prms >> "less_rot",
                                                                                mug_diverse_grasp=prms >> "mug_diverse_grasp",
                                                                                mug_rot_check_wall=not env_prms >> "use_mug" or not prms >> "mug_rot_allow_wall")
    else:
        assert not env_prms >> "use_mug", "not implemented"
        policy_next_params_fn = get_push_pull_meta_policy_params_fn(*mmp, *mmr, do_pull=not prms >> "push_only",
                                                                    one_dim_only=prms >> "one_dim_only",
                                                                    random_motion=prms >> "random_motion",
                                                                    directional=prms >> "directional",
                                                                    retreat_first=prms >> "retreat_first",
                                                                    uniform_velocity=prms >> "uniform_velocity",
                                                                    retreat_xy=prms >> "retreat_xy",
                                                                    smooth_noise=prms >> "smooth_noise",
                                                                    random_slow_prob=prms >> "random_slow_prob",
                                                                    stop_prob=prms >> "stop_prob")

    ALL_POLICIES = [
        d(cls=PushPrimitive, params=polprms),
        d(cls=PullPrimitive, params=polprms),
        d(cls=TopRotatePrimitive, params=polprms),
        d(cls=LiftPrimitive, params=polprms),
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
