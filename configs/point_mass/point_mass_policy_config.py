
"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import numpy as np

from sbrl.policies.memory_policy import MemoryPolicy
from sbrl.utils.np_utils import clip_norm
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import to_numpy


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--kp", type=float, default=10.)
    parser.add_argument("--max_action", type=float, default=1.)
    parser.add_argument("--curvature_noise", type=float, default=None, help='orthogonal offset, as a fraction of delta_xy')
    parser.add_argument("--fixed_side", type=int, choices=[-1, 1], default=None)
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--max_steps", type=int, default=np.inf)
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    prms = common_params >> group_name

    KP = np.broadcast_to(prms >> "kp", (2,))
    MAX_ACTION = prms >> "max_action"
    TOL = prms >> "tolerance"
    MAX_STEPS = prms >> "max_steps"
    CURV_NS = prms >> "curvature_noise"
    SIDE = prms >> "fixed_side"

    def forward_fn(model, obs, goal, memory, **kwargs):
        obs = to_numpy(obs >> "obs", check=True).reshape((4,))

        curr = obs[:2]
        targ = obs[2:]
        dpos = targ - curr

        if memory.is_empty():
            memory.count = 0
            memory.sign = np.random.choice([-1, 1]) if SIDE is None else SIDE
            if CURV_NS is not None:
                memory.curvature = memory.sign * np.random.uniform(0, CURV_NS)
            else:
                memory.curvature = 0.

            memory.init_dpos_orth = np.array([dpos[1], -dpos[0]])

        t = 1 - np.linalg.norm(dpos) / np.linalg.norm(memory.init_dpos_orth)  # goes from 0 -> 1
        scale = (1 - t) ** 2  # np.cos(np.pi/2 * t)
        # print(scale)
        dpos_orth = memory.curvature * scale * memory.init_dpos_orth  # same norm

        dpos = dpos + dpos_orth

        vel = clip_norm(dpos * KP, MAX_ACTION)

        memory.count += 1

        return AttrDict(
            action=vel[None],
            modality=np.array([memory.sign])[None]
        )

    def is_terminated(model, obs, goal, memory, **kwargs):
        return not memory.is_empty() and (memory >> "count") > MAX_STEPS

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=MemoryPolicy,
        params=AttrDict(
            policy_model_forward_fn=forward_fn,
            is_terminated_fn=is_terminated,
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
