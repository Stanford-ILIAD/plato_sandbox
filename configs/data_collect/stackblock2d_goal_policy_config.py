from argparse import ArgumentParser

from sbrl.policies.blocks.block2d_goal_policies import Block2DGoalPolicy
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--use_intermediate_targets', action='store_true', help="enable intermediate targets")
    parser.add_argument('--do_push', action='store_true', help="enable push")
    parser.add_argument('--do_pull', action='store_true', help="enable pull")
    parser.add_argument('--do_side_rot', action='store_true', help="enable side rotate")
    parser.add_argument('--do_lift_rot', action='store_true', help="enable lift rotate")
    parser.add_argument('--do_tip', action='store_true', help="enable tip")
    parser.add_argument('--oversample_rot', action='store_true', help="oversample rotation actions")
    parser.add_argument('--oversample_tip', action='store_true', help="oversample tip actions, only for rotate_only")
    parser.add_argument('--undersample_lift', action='store_true', help="undersample lift actions, to compensate for it being the only upward primitive.")

    # only relevant if using primitives
    parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[5, 12])

    parser.add_argument('--use_policy_type', action="store_true")
    parser.add_argument('--sort_primitives_by_motion', action="store_true")
    parser.add_argument('--no_pull_directional_primitives', action="store_true")

    return parser


def process_params(group_name, common_params):
    # utils = common_params >> "utils"

    prms = common_params >> group_name

    policy_params = d(
        cls=Block2DGoalPolicy,
        params=prms
    )

    common_params[group_name] = common_params[group_name] & policy_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
