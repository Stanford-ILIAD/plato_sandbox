import numpy as np
import pybullet as p

from sbrl.policies.policy import Policy
from sbrl.utils import transform_utils
from sbrl.utils.np_utils import clip_norm
from sbrl.utils.pybullet_utils import draw_box_gui
from sbrl.utils.python_utils import get_with_default, AttrDict
from sbrl.utils.torch_utils import to_numpy


class Block3DHardcodedGoalPolicy(Policy):

    def _init_params_to_attrs(self, params):
        super(Block3DHardcodedGoalPolicy, self)._init_params_to_attrs(params)
        # the only option supported
        self._do_push = get_with_default(params, "do_push", True)
        self._aligned_push = get_with_default(params, "aligned_push", True)
        assert self._do_push, "No other options yet.."

    def _init_setup(self):
        super(Block3DHardcodedGoalPolicy, self)._init_setup()

        self._goal_vis = None

    def warm_start(self, model, observation, goal):
        pass

    def get_action(self, model, observation, goal, **kwargs):
        get_keys = ['ee_position', 'gripper_pos', 'ee_orientation_eul', 'objects/position', 'objects/orientation_eul', 'objects/size']
        pos, gripper_pos, ori, blocks_pos, block_ori, block_size = (observation > get_keys).leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(get_keys)

        sc = self._env.surface_center.copy()
        hl = 0.8 * self._env.surface_bounds.copy() / 2
        bc = np.random.uniform(sc[:2] - hl, sc[:2] + hl, (blocks_pos.shape[0], 2))

        if self._aligned_push:
            delta = bc - blocks_pos[:, :2]
            delta_x = delta * np.array([1., 0])[None]
            delta_y = delta * np.array([0., 1.])[None]
            delta = np.where(np.linalg.norm(delta_x, axis=-1) > np.linalg.norm(delta_y, axis=-1), delta_x, delta_y)
            delta = clip_norm(delta, 0.3)
            bc = blocks_pos[:, :2] + delta

        bc = np.append(bc, blocks_pos[:, 2:3], axis=-1)  # 1,3
        bo = block_ori.copy()

        if self._goal_vis is None:
            # TODO support multiple blocks
            _, self._goal_vis = draw_box_gui(bc[0], transform_utils.euler2quat_ext(bo[0].tolist()), block_size[0], [0.5, 0.5, 0.5, 1.], 0)
        else:
            p.resetBasePositionAndOrientation(self._goal_vis, bc[0], bo[0], [0.5, 0.5, 0.5, 1.], 0)

        ptype = 0

        # (1, ...) the last state that was reached.
        goal_ac = AttrDict(goal=AttrDict.from_dict({
            "objects/position": bc,
            "objects/velocity": np.zeros_like(bc),
            "objects/orientation_eul": bo,
            "objects/angular_velocity": np.zeros_like(bo),
            "objects/size": block_size,
            "policy_type": np.array([ptype], dtype=np.int16),
        })).leaf_apply(lambda arr: arr[None])

        return goal_ac

    def is_terminated(self, model, observation, goal, env_memory=None, **kwargs):
        return False

    def reset_policy(self, **kwargs):
        if self._goal_vis is not None:
            p.removeBody(self._goal_vis, 0)
            self._goal_vis = None

