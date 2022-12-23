"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import numpy as np

from sbrl.envs.bullet_envs.utils_env import target_action_postproc_fn
from sbrl.policies.memory_policy import MemoryPolicy
from sbrl.utils.python_utils import AttrDict, get_with_default
from sbrl.utils.torch_utils import to_numpy


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--action_name', type=str, default="action")
    parser.add_argument('--ee_pos_name', type=str, default="ee_position")
    parser.add_argument('--ee_ori_name', type=str, default="ee_orientation_eul")
    parser.add_argument('--gripper_name', type=str, default="gripper_pos")
    parser.add_argument('--edge_steps', type=int, default=20)
    parser.add_argument('--action_as_delta', action='store_true')
    parser.add_argument('--spatial_gain', type=float, default=1.)
    parser.add_argument('--pos_gain', type=float, default=1.)
    parser.add_argument('--rot_gain', type=float, default=.2)
    parser.add_argument('--dx', type=float, default=0.1)
    parser.add_argument('--dy', type=float, default=None)
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    prms = common_params >> group_name

    pn = prms >> "ee_pos_name"
    on = prms >> "ee_ori_name"
    gn = prms >> "gripper_name"
    an = prms >> "action_name"

    dx = prms >> "dx"
    dy = get_with_default(prms, "dy", dx)

    edge_steps = prms >> "edge_steps"
    waypoints = np.array([[dx, 0.0, 0.],
                          [dx, 0, dy],
                          [0., 0, dy],
                          [0., 0., 0.]])
    # waypoints = np.array([[dx, 0.0, 0.],
    #                       [dx, dy, 0.],
    #                       [0., dy, 0.],
    #                       [0., 0., 0.]])

    def move_ee_in_square(model, obs, goal, memory: AttrDict = AttrDict(), **kwargs):
        if not memory.has_leaf_key("count"):
            memory.count = 0
            pos, ori, grip = obs.get_keys_required([pn, on, gn])
            pos, ori, grip = to_numpy(pos[0, 0], check=True).reshape(-1), to_numpy(ori[0, 0], check=True).reshape(-1), to_numpy(
                grip[0, 0], check=True).reshape(-1)
            memory.base_pos = pos.copy()
            memory.base_ori = ori.copy()
            memory.base_grip = grip.copy()

        phase = (memory.count // edge_steps) % len(waypoints)
        # step = memory.count % edge_steps

        targ_pos = memory.base_pos.copy()
        targ_ori = memory.base_ori.copy()
        targ_grip = memory.base_grip.copy()
        targ_pos += waypoints[phase]

        memory.count += 1

        act = AttrDict.from_dict({
            'target': AttrDict.from_dict({
                pn: targ_pos,
                on: targ_ori,
                gn: targ_grip,
            }),
            'policy_name': np.array(['test_move']),
            'policy_type': np.array([1000]),
            'policy_switch': np.array([False]),
        }).leaf_apply(lambda arr: arr[None])

        # act[an] = np.concatenate([targ_pos, targ_ori, targ_grip])[None]
        # return act
        # # TODO put back...
        return target_action_postproc_fn(obs, act, action_name=an)

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=MemoryPolicy,
        params=AttrDict(
            action_name=an,
            policy_model_forward_fn=move_ee_in_square,
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
