import numpy as np
import pymunk
from pymunk.space_debug_draw_options import SpaceDebugColor

from sbrl.experiments import logger
from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.blocks.stack_block2d_policies import get_push_pull_lift_rotate_memory_meta_policy_params_fn, \
    get_push_pull_lift_tip_srot_rot_memory_meta_policy_params_fn, get_rotate_only_memory_meta_policy_params_fn, \
    get_push_pull_lift_memory_meta_policy_params_fn, TipBlockPrimitive, RotateBlockPrimitive, SideRotateBlockPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_with_default, AttrDict as d
from sbrl.utils.torch_utils import to_numpy


class Block2DGoalPolicy(Policy):
    def _init_params_to_attrs(self, params):
        self._sub_env_params = self._env.params
        if self._sub_env_params.has_leaf_key("render"):
            self._sub_env_params.render = False

        self._use_intermediate_targets = get_with_default(params, "use_intermediate_targets", False)
        self._do_push = get_with_default(params, "do_push", False)
        self._do_pull = get_with_default(params, "do_pull", False)
        self._do_tip = get_with_default(params, "do_tip", False)
        self._do_side_rot = get_with_default(params, "do_side_rot", False)
        self._do_lift_rot = get_with_default(params, "do_lift_rot", False)
        self._oversample_rot = get_with_default(params, "oversample_rot", False)
        self._oversample_tip = get_with_default(params, "oversample_tip", False)
        self._undersample_lift = get_with_default(params, "undersample_lift", False)

        self._vel_noise = get_with_default(params, "vel_noise", 0.)

        # single rollout
        self._mmr = get_with_default(params, "min_max_retreat", (0, 1))

        # any of the rotates
        self._do_rot = self._do_side_rot or self._do_lift_rot or self._do_tip

        logger.debug(f"Goal Policy -- Primitives: [push: {self._do_push}, pull: {self._do_pull}, tip: {self._do_tip}, srot: {self._do_side_rot}, liftrot: {self._do_lift_rot}]")

        # check validity of args.
        if self._do_rot:
            if not self._do_pull and not self._do_push:
                # rotate only
                assert not self._oversample_rot
                assert self._do_side_rot and self._do_tip, "tip and side-rot are minimal set for rotate only"
                policy_next_params_fn = get_rotate_only_memory_meta_policy_params_fn(1, 2, *self._mmr,
                                                                                     random_side=True,
                                                                                     randomize_offset=True,
                                                                                     oversample_tip=self._oversample_tip,
                                                                                     no_lift_rot=not self._do_lift_rot,)
            else:
                if not self._do_lift_rot:
                    assert self._do_side_rot, "disabling side-rot has not been implemented"
                    policy_next_params_fn = get_push_pull_lift_tip_srot_rot_memory_meta_policy_params_fn(1, 2, *self._mmr,
                                                                                                         random_side=True,
                                                                                                         randomize_offset=True,
                                                                                                         oversample_tip=self._oversample_tip,
                                                                                                         undersample_lift=self._undersample_lift,
                                                                                                         no_lift_rot=not self._do_lift_rot,
                                                                                                         no_push=not self._do_push,
                                                                                                         no_pull=not self._do_pull,
                                                                                                         no_tip=not self._do_tip)
                else:
                    assert not self._undersample_lift, "not implemented"
                    assert not self._do_side_rot, "side rotate is not implemented here..."
                    policy_next_params_fn = get_push_pull_lift_rotate_memory_meta_policy_params_fn(1, 2, *self._mmr,
                                                                                                   random_side=True,
                                                                                                   randomize_offset=True,
                                                                                                   oversample_rot=self._oversample_rot)
        else:
            assert self._do_lift_rot, "Rot was not enabled, can't disabled lift rot"
            assert not self._undersample_lift, "not implemented"
            policy_next_params_fn = get_push_pull_lift_memory_meta_policy_params_fn(1, 2, *self._mmr, random_side=True)
            assert not self._oversample_rot, "Cannot specify oversample if rotations not included"

        ALL_POLICIES = [
            d(cls=PushPrimitive, params=d(vel_noise=self._vel_noise)),
            d(cls=PullPrimitive, params=d(vel_noise=self._vel_noise)),
        ]

        if self._do_rot:
            ALL_POLICIES.extend([
                d(cls=TipBlockPrimitive, params=d(vel_noise=self._vel_noise)),
                d(cls=RotateBlockPrimitive, params=d(vel_noise=self._vel_noise)),
                d(cls=SideRotateBlockPrimitive, params=d(vel_noise=self._vel_noise)),
            ])

        self._sub_policy_params = d(
            all_policies=ALL_POLICIES,
            next_param_fn=policy_next_params_fn,
        )

        # other params
        self._use_policy_type = get_with_default(params, "use_policy_type", True)
        self._no_pull_directional_primitives = get_with_default(params, "no_pull_directional_primitives", True)
        self._sort_primitives_by_motion = get_with_default(params, "sort_primitives_by_motion", True)

    def _init_setup(self):
        assert self._is_goal

        # initialize sub-env as a copy
        self._sub_env = self._env.__class__(self._sub_env_params, self._env_spec)

        # initialize sub-policy
        self._sub_policy = MetaPolicy(self._sub_policy_params, self._env_spec, file_manager=self._file_manager, env=self._env)

        self._sub_rollout_length = 0

    def get_action(self, model, observation, goal, **kwargs):
        # this is going to get goals for block2d (by rollout of 2D env), starting from the current env state.
        good_rollout = False
        num_tries = 0
        while not good_rollout and num_tries <= 10:
            self._sub_env.reset(observation.leaf_apply(lambda arr: arr[0, 0]))
            self._sub_env.set_state(observation.leaf_apply(lambda arr: arr[0]))
            o, g = observation.leaf_apply(lambda arr: arr[0]), goal.leaf_apply(lambda arr: arr[0])
            self._sub_policy.reset_policy(next_obs=o, next_goal=g)
            action_sequence = []
            sequence = [o]
            ptype = -1

            for i in range(1000):
                act = self._sub_policy.get_action(model, o.leaf_apply(lambda arr: arr[None]),
                                                  g.leaf_apply(lambda arr: arr[None]))
                action_sequence.append(act.leaf_copy())

                if i == 0 and self._use_policy_type:
                    if self._sub_policy.curr_policy_idx == -1:
                        logger.warn(f"Policy type is -1! skipping this rollout. num_tries = {num_tries}")
                        break

                    ptype = self._sub_policy.curr_policy_idx
                    if not self._no_pull_directional_primitives and self._sub_policy.curr_policy_idx == 1:
                        ptype = 2 + self._sub_policy.curr_policy._pull_tangent

                o, g, done = self._sub_env.step(act)
                sequence.append(o)

                self._sub_rollout_length += 1

                if self._sub_policy.is_terminated(model, o, g):
                    break

            # make sure a rollout was recorded.
            if self._sub_rollout_length > 0:
                good_rollout = True

            num_tries += 1

        # if timeout, ptype = -1, we need to set the sub rollout length to something small but nonzero
        if num_tries > 10:
            self._sub_rollout_length = 2
        else:
            if self._use_policy_type and self._sort_primitives_by_motion:
                block_ps_initial = to_numpy(sequence[0] >> "block_positions", check=True)
                block_ps_final = to_numpy(sequence[-1] >> "block_positions", check=True)
                dir = block_ps_final[0] - block_ps_initial[0]  # final minus initial  (N x 2)
                angles = np.arctan2(dir[:, 1], dir[:, 0]) % (2 * np.pi)  # y (up), x (right)
                # wrap around (now we have four rotated quadrants, counterclockwise starting at rightwards)
                quantized_angles = np.digitize((angles + np.pi/4) % (2 * np.pi), [np.pi/2, np.pi, 3 * np.pi / 2, 2 * np.pi], right=True)
                num_blocks = block_ps_initial.shape[1]
                # width = num_blocks * 4  # directions per block
                # unique per block
                # print(ptype, quantized_angles, num_blocks * 4 * ptype + np.dot(quantized_angles, np.arange(1, num_blocks + 1)).astype(int))

                if ptype < 2:
                    ptype = num_blocks * 4 * ptype + np.dot(quantized_angles, np.arange(1, num_blocks + 1)).astype(int)
                else:
                    ptype = num_blocks * 4 * ptype  # group all together for tip / rotate

        # (1, ...) the last state that was reached.
        goal_ac = (sequence[-1] > ["block_positions", "block_velocities", "block_angles", "block_angular_velocities", "block_sizes"]) & d(
            policy_type=np.array([[ptype]], dtype=np.int16)
        )

        # goal_ac.leaf_modify(lambda arr: to_torch(arr, check=True))

        return goal_ac

    def is_terminated(self, model, observation, goal, env_memory=None, **kwargs):
        if env_memory is None or self._sub_rollout_length == 0:
            return False

        return env_memory >> "rollout_count" > self._sub_rollout_length * 2

    def reset_policy(self, **kwargs):
        self._sub_rollout_length = 0


def do_draw(env, obs, action, color=SpaceDebugColor(r=0., g=255.0, b=0.0, a=255.0), goal=None, goal_color=SpaceDebugColor(r=255., g=255.0, b=255.0, a=100.0)):
    action = action.copy()
    extra_draw_actions = []
    if action.has_leaf_key('target/position'):
        act = action.target.position[0]
    elif action.has_leaf_key('action'):
        pos = obs >> "position"
        act = pos[0] + action.action[0, :2]
    else:
        act = np.asarray((0, 0))

    if action.has_leaf_key('target/grab_binary'):
        add_blue = max(int(action.target.grab_binary.reshape(-1)[0]), 0)
    else:
        add_blue = 0

    color = SpaceDebugColor(r=color.r, g=color.g, b=min(color.b + add_blue * 255, 255), a=color.a)
    # print(color)
    extra_draw_actions.append(lambda ops: ops.draw_circle(to_numpy(act, check=True), angle=0, radius=5.,
                                                          outline_color=color, fill_color=color))
    if goal is not None:
        # (1 x N x ...) -> (N x ...)
        bp, bs, ba = goal.leaf_apply(lambda arr: arr[0]).get_keys_required(['block_positions', 'block_sizes', 'block_angles'])
        # (4, 2, N, 1), corners dist from center
        vec_center_to_corner = np.stack([bs[:, 0] / 2, bs[:, 1] / 2,  # top right
                                -bs[:, 0] / 2, bs[:, 1] / 2,  # top left
                                -bs[:, 0] / 2, -bs[:, 1] / 2,  # bottom left
                                bs[:, 0] / 2, -bs[:, 1] / 2]).reshape((4, 2, -1, 1))  # bottom right
        # (4, N, 2, 1)
        vec_center_to_corner = np.transpose(vec_center_to_corner, [0, 2, 1, 3])

        # (1, N, 2, 2), rotate by angle
        rot_mat = np.stack([np.cos(ba), -np.sin(ba), np.sin(ba), np.cos(ba)], axis=-1).reshape((1, -1, 2, 2))

        rotated = np.matmul(rot_mat, vec_center_to_corner)[..., 0]  # -> (4, N, 2, 1)
        true_corners = bp[None] + rotated  # (4, N, 2), centered around block

        def draw_goals(ops):
            for bi in range(bp.shape[0]):
                first = [true_corners[j, bi] for j in range(4)]
                next = first[1:] + [first[0]]
                for a, b in zip(first, next):
                    ops.draw_segment(pymunk.Vec2d(a[0], a[1]), pymunk.Vec2d(b[0], b[1]), color=goal_color)

        extra_draw_actions.append(draw_goals)
    env.set_draw_shapes(extra_draw_actions)
