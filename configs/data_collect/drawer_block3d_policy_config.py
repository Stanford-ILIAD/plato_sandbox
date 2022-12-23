from argparse import ArgumentParser

from sbrl.policies.blocks.block3d_extra_policies import DrawerMovePrimitive, ButtonPressPrimitive
from sbrl.policies.blocks.block3d_meta_policies import get_push_toprot_drawer_cab_meta_policy_params_fn, \
    get_sequential_push_toprot_drawer_cab_meta_policy_params_fn
from sbrl.policies.blocks.block3d_policies import PushPrimitive, PullPrimitive, TopRotatePrimitive, LiftPrimitive, \
    Reach3DPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # parser.add_argument('--use_rotate', action='store_true')
    # parser.add_argument('--no_push_pull', action='store_true', help="disable push pull")
    # parser.add_argument('--no_lift_rot', action='store_true', help="disable lift rotate")
    parser.add_argument('--single_policy', action='store_true', help="no retreat, one policy per eval")
    # parser.add_argument('--use_intermediate_targets', action='store_true',
    #                     help="target action is intermediate, not final, for position")
    parser.add_argument('--do_push', action='store_true', help="remove push prim")
    parser.add_argument('--do_toprot', action='store_true', help="includes rotation primitive")
    parser.add_argument('--do_drawer', action='store_true', help="includes rotation primitive")
    parser.add_argument('--do_buttons', action='store_true', help="button pressing")
    parser.add_argument('--object_only', action='store_true', help="True = no drawer/cabinet primitives for sequential env. make sure they start open.")
    parser.add_argument('--no_object_move', action='store_true', help="False = includes to/from cabinet/drawer actions in sequential env.")
    parser.add_argument('--one_dim_only', action='store_true', help="motion along 1 axis only")
    parser.add_argument('--sequential', action='store_true', help="Use the sequential task next_params fn")
    parser.add_argument('--random_motion', action='store_true', help="randomized velocity and time")
    # parser.add_argument('--retreat_first', action='store_true', help="retreats first and only if it needs to")
    # parser.add_argument('--retreat_xy', action='store_true',
    #                     help="retreats with variation along x/y, and less so on z axis")
    parser.add_argument('--directional', action='store_true', help="push in direction block is facing, not just cardinal axes")
    parser.add_argument('--safe_open', action='store_true', help="safely grasp to open cabinet (also more tolerance)")
    parser.add_argument('--diverse', action='store_true', help="grasping diversity, and some initial motion diversity as well.")
    parser.add_argument('--use_cab_waypoints', action='store_true', help="use waypoints to and from cabinet pulling.")
    parser.add_argument('--pull_xy_tol', action='store_true', help="xy tolerance only (ignore z) for to/from drawer/cabinet. helps to remove pauses.")
    # parser.add_argument('--uniform_velocity', action='store_true', help="uses true postproc velocities, e.g. setpoint -> vel is a fixed transform")
    # parser.add_argument('--oversample_rot', action='store_true', help="oversample rotation actions")
    # parser.add_argument('--oversample_tip', action='store_true', help="oversample tip actions, only for rotate_only")
    # parser.add_argument('--undersample_lift', action='store_true', help="undersample lift actions, to compensate for it being the only upward primitive.")
    # only relevant if using primitives
    parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[6, 12])
    parser.add_argument('--min_max_policies', type=int, nargs=2, default=[4, 10])
    # parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])

    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    prms = common_params >> group_name

    mmp = (prms >> "min_max_policies")
    mmr = (prms >> "min_max_retreat")

    # if prms >> "directional" and mmr == [15, 25]:
    #     # changing the default, kinda
    #     mmr = [5, 12]

    if prms >> "single_policy":
        # overrides
        mmp = [1, 2]
        mmr = [0, 1]

    polprms = d(vel_noise=0., max_pos_vel=0.4, max_ori_vel=5.0)  # use_intermediate_targets=prms >> "use_intermediate_targets"
    # if prms >> "uniform_velocity":
    #     polprms.max_pos_vel = 0.4  # otherwise things are way too fast

    if prms >> 'sequential':
        assert not prms >> "object_only" or not prms >> "no_object_move", "Contradictory: select only one of these options"
        policy_next_params_fn = get_sequential_push_toprot_drawer_cab_meta_policy_params_fn(*mmp, *mmr,
                                                                                            do_push=prms >> "do_push",
                                                                                            do_toprot=prms >> "do_toprot",
                                                                                            do_object_move=not prms >> "no_object_move",  # enabled by default.
                                                                                            random_motion=prms >> "random_motion",
                                                                                            directional=prms >> "directional",
                                                                                            safe_open=prms >> "safe_open",
                                                                                            diverse=prms >> "diverse",
                                                                                            do_drawer=not prms >> "object_only",
                                                                                            do_cabinet=not prms >> "object_only",
                                                                                            use_cab_waypoints=prms >> "use_cab_waypoints",
                                                                                            pull_xy_tol=prms >> "pull_xy_tol",
                                                                                            do_buttons=prms >> "do_buttons",
                                                                                            retreat_first=False,
                                                                                            retreat_xy=False)
    else:
        policy_next_params_fn = get_push_toprot_drawer_cab_meta_policy_params_fn(*mmp, *mmr,
                                                                                 do_push=prms >> "do_push",
                                                                                 do_toprot=prms >> "do_toprot",
                                                                                 do_drawer=prms >> "do_drawer",
                                                                                 one_dim_only=prms >> "one_dim_only",
                                                                                 random_motion=prms >> "random_motion",
                                                                                 directional=prms >> "directional",
                                                                                 safe_open=prms >> "safe_open",
                                                                                 diverse=prms >> "diverse",
                                                                                 retreat_first=False,
                                                                                 retreat_xy=False)

    ALL_POLICIES = [
        d(cls=PushPrimitive, params=polprms),
        d(cls=PullPrimitive, params=polprms),
        d(cls=TopRotatePrimitive, params=polprms),
        d(cls=LiftPrimitive, params=polprms),
        d(cls=TopRotatePrimitive, params=polprms),  # cabinet
        d(cls=DrawerMovePrimitive, params=polprms),  # drawer
        d(cls=Reach3DPrimitive, params=polprms),  # drawer
        d(cls=ButtonPressPrimitive, params=polprms),  # drawer
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
