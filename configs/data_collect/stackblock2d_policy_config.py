from argparse import ArgumentParser

from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.blocks.stack_block2d_policies import get_push_pull_lift_rotate_memory_meta_policy_params_fn, \
    get_push_pull_lift_memory_meta_policy_params_fn, TipBlockPrimitive, RotateBlockPrimitive, \
    get_rotate_only_memory_meta_policy_params_fn, SideRotateBlockPrimitive, \
    get_push_pull_lift_tip_srot_rot_memory_meta_policy_params_fn
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--use_rotate', action='store_true')
    parser.add_argument('--no_push_pull', action='store_true', help="disable push pull")
    parser.add_argument('--no_lift_rot', action='store_true', help="disable lift rotate")
    parser.add_argument('--no_tip', action='store_true', help="disable tip")
    parser.add_argument('--no_pull', action='store_true', help="disable pull")
    parser.add_argument('--no_push', action='store_true', help="disable push")
    parser.add_argument('--no_save_names', action='store_true')
    parser.add_argument('--single_policy', action='store_true', help="no retreat, one policy per eval")
    parser.add_argument('--oversample_rot', action='store_true', help="oversample rotation actions")
    parser.add_argument('--oversample_tip', action='store_true', help="oversample tip actions, only for rotate_only")
    parser.add_argument('--undersample_lift', action='store_true', help="undersample lift actions, to compensate for it being the only upward primitive.")
    # only relevant if using primitives
    parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[5, 12])
    parser.add_argument('--min_max_policies', type=int, nargs=2, default=[3, 6])
    parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])

    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    prms = common_params >> group_name

    env_spec_prms = common_params >> "env_spec/params"
    if not prms >> "no_save_names":
        (env_spec_prms >> "action_names").append('policy_name')
    (env_spec_prms >> "action_names").append('policy_switch')

    mmp = (prms >> "min_max_policies")
    mmr = (prms >> "min_max_retreat")
    oversample_rot = (prms >> "oversample_rot")
    oversample_tip = (prms >> "oversample_tip")
    undersample_lift = (prms >> "undersample_lift")
    no_lift_rot = (prms >> "no_lift_rot")
    no_push = (prms >> "no_push")
    no_pull = (prms >> "no_pull")
    no_tip = (prms >> "no_tip")
    pref_idxs = prms >> "prefer_idxs"

    if prms >> "single_policy":
        # overrides
        mmp = [1, 2]
        mmr = [0, 1]

    if prms >> "use_rotate":
        if prms >> "no_push_pull":
            assert not undersample_lift
            policy_next_params_fn = get_rotate_only_memory_meta_policy_params_fn(*mmp, *mmr,
                                                                                 random_side=True,
                                                                                 randomize_offset=True,
                                                                                 oversample_tip=oversample_tip,
                                                                                 no_lift_rot=no_lift_rot,
                                                                                 prefer_idxs=pref_idxs)
        else:
            if no_lift_rot:
                policy_next_params_fn = get_push_pull_lift_tip_srot_rot_memory_meta_policy_params_fn(*mmp, *mmr,
                                                                                                     random_side=True,
                                                                                                     randomize_offset=True,
                                                                                                     oversample_tip=oversample_tip,
                                                                                                     undersample_lift=undersample_lift,
                                                                                                     no_lift_rot=no_lift_rot,
                                                                                                     no_push=no_push,
                                                                                                     no_pull=no_pull,
                                                                                                     no_tip=no_tip,
                                                                                                     prefer_idxs=pref_idxs)
            else:
                assert not undersample_lift, "not implemented"
                policy_next_params_fn = get_push_pull_lift_rotate_memory_meta_policy_params_fn(*mmp, *mmr,
                                                                                               random_side=True,
                                                                                               randomize_offset=True,
                                                                                               oversample_rot=oversample_rot,
                                                                                               prefer_idxs=pref_idxs)
    else:
        assert not no_lift_rot, "Rot was not enabled, can't disabled lift rot"
        assert not undersample_lift, "not implemented"
        policy_next_params_fn = get_push_pull_lift_memory_meta_policy_params_fn(*mmp, *mmr, random_side=True)
        assert not oversample_rot, "Cannot specify oversample if rotations not included"

    ALL_POLICIES = [
        d(cls=PushPrimitive, params=d(vel_noise=0.)),
        d(cls=PullPrimitive, params=d(vel_noise=0.)),
    ]

    if prms >> "use_rotate":
        ALL_POLICIES.extend([
            d(cls=TipBlockPrimitive, params=d(vel_noise=0.)),
            d(cls=RotateBlockPrimitive, params=d(vel_noise=0.)),
            d(cls=SideRotateBlockPrimitive, params=d(vel_noise=0.)),
        ])

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
