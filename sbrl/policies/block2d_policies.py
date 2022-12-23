"""
A set of hardcoded policies for a block environment

"""

import numpy as np
import pygame

from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.experiments import logger
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_with_default, AttrDict
from sbrl.utils.torch_utils import to_numpy


class BlockPrimitive(Policy):
    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, block_idx=0, offset=np.array([0., 0.]), tolerance=0.1, **kwargs):
        assert self._env.num_blocks > block_idx, [block_idx, self._env.num_blocks]
        assert np.sum(np.abs(offset)) > np.min(self._env.block_size) and len(offset) == 2, offset
        self._num_blocks = self._env.num_blocks
        self._block_size = self._env.block_size
        self._block_idx = block_idx
        self._target_offset = offset
        self._tolerance = tolerance

    def _init_params_to_attrs(self, params):
        self._max_ego_speed = get_with_default(params, "max_ego_speed", 60, map_fn=np.asarray)  # in each direction
        self._smooth_vel_coef = get_with_default(params, "smooth_vel_coef", 0.8, map_fn=float)  # smooths controller out
        self._vel_noise = get_with_default(params, "vel_noise", 2., map_fn=float)  # smooths controller out

    def _init_setup(self):
        if self._env is not None:
            assert isinstance(self._env, BlockEnv2D), type(self._env)


    @property
    def curr_name(self) -> str:
        # returns a string identifier for the policy, rather than ints.
        raise NotImplementedError

class ReachPrimitive(BlockPrimitive):

    def reset_policy(self, kp_vel=1.0, timeout=20, **kwargs):
        super(ReachPrimitive, self).reset_policy(**kwargs)
        self._kp_vel = np.array(kp_vel)
        self._timeout = timeout

        self.num_steps = 0

    def get_action(self, model, observation, goal, **kwargs):
        # get the pos, vel, and block pos's
        # print(observation.leaf_filter(lambda k, v: k != "image"))
        pos, vel, blocks_pos = observation.leaf_apply(lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(['position', 'velocity', 'block_positions'])

        delta = blocks_pos[self._block_idx] + self._target_offset - pos
        v = np.clip(self._kp_vel * delta, -self._max_ego_speed, self._max_ego_speed)

        v = self._smooth_vel_coef * v + (1 - self._smooth_vel_coef) * vel
        v += np.random.randn(2) * self._vel_noise

        v = np.append(v, 0)  # grab
        # print(blocks_pos[self._block_idx], pos, "ACT:", v)
        self.num_steps += 1
        setpoint_position = blocks_pos[self._block_idx] + self._target_offset
        setpoint_grab = np.array([0] * self._num_blocks)
        # the desired waypoint
        out = AttrDict(action=v[None])
        out.policy_type = np.array([4])[None]
        out.policy_name = np.array([self.curr_name])[None]
        out['target/position'] = setpoint_position[None]
        out['target/grab_binary'] = setpoint_grab[None]
        return out

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        pos, vel, blocks_pos = observation.leaf_apply(lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(['position', 'velocity', 'block_positions'])
        return self.num_steps >= self._timeout or \
               np.linalg.norm(
                   blocks_pos[self._block_idx] + self._target_offset - pos) < np.min(self._block_size) * self._tolerance

    @property
    def curr_name(self) -> str:
        return 'reach'

class PushPrimitive(ReachPrimitive):

    def reset_policy(self, push_steps=10, wait_steps=5, speed_scale=1., retreat_steps=0, retreat_direction=None, **kwargs):
        super(PushPrimitive, self).reset_policy(**kwargs)
        self._push_steps = push_steps
        self._push_direction = self._target_offset.copy()
        self._wait_steps = wait_steps
        self._speed_scale = speed_scale
        self._curr_step = 0
        self._retreat_steps = retreat_steps
        self._retreat_direction = retreat_direction
        self._stage = 0

    def get_action(self, model, observation, goal, **kwargs):
        observation = observation.leaf_apply(lambda arr: to_numpy(arr, check=True))
        if self._stage == 0:
            # first reach
            ac = super(PushPrimitive, self).get_action(model, observation, goal, **kwargs)
            if super(PushPrimitive, self).is_terminated(model, observation, goal, **kwargs):
                self._stage += 1
            setpoint_position = ac['target/position'].copy()
            setpoint_grab = ac['target/grab_binary'].copy()
        elif self._stage == 1:
            # push forward
            v = - self._speed_scale * self._push_direction.copy()  # move towards block
            setpoint_position = (observation.position[0, 0] + v * self._env.dt * (self._push_steps - self._curr_step))[None]
            setpoint_grab = np.zeros((1, self._num_blocks))
            v = np.where(np.abs(v) < 0.1, self._vel_noise * np.random.randn(2), v)  # noise in nonzero
            v = np.append(v, 0)  # no grabbing
            ac = AttrDict(action=v[None])
            self._curr_step += 1
            if self._curr_step > self._push_steps:
                self._curr_step = 0
                self._stage += 1

            self.num_steps += 1
        elif self._stage == 2:
            # wait
            v = np.array([0, 0, 0])
            ac = AttrDict(action=v[None])
            self._curr_step += 1
            if self._curr_step > self._wait_steps:
                self._curr_step = 0
                self._stage += 1
            setpoint_position = observation.position[:, 0].copy()
            setpoint_grab = np.zeros((1, self._num_blocks))
            self.num_steps += 1
        else:
            # retreat
            if self._retreat_steps > 0:
                if self._retreat_direction is not None:
                    v = + self._speed_scale * self._retreat_direction.copy()
                else:
                    v = + self._speed_scale * self._target_offset.copy()
                setpoint_position = (observation.position[0, 0] + v * self._env.dt * (
                        self._retreat_steps - self._curr_step))[None]
                # much larger lateral noise on retreat
                if not self._env.is_in_bounds():
                    v = -v  # bang bang control kinda
                else:
                    v = np.where(np.abs(v) < 0.1, 10 * self._vel_noise * np.random.randn(2), v)
                v = np.append(v, 0)
                self._curr_step += 1
                if self._curr_step > self._retreat_steps:
                    self._curr_step = 0
                    self._stage += 1
            else:
                setpoint_position = observation.position[:, 0].copy()
                logger.warn("Push Policy is done but action requested...")
                v = np.array([0, 0, 0])

            ac = AttrDict(action=v[None])
            setpoint_grab = np.zeros((1, self._num_blocks))
            self.num_steps += 1

        ac.policy_type = np.array([0])[None]
        ac.policy_name = np.array([self.curr_name])[None]
        ac['target/position'] = setpoint_position.copy()
        ac['target/grab_binary'] = setpoint_grab.copy()

        # ac.pprint()
        return ac

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self._stage == 3 + int(self._retreat_steps > 0)

    @property
    def curr_name(self) -> str:
        v = -self._target_offset
        motion_dir = "right" if v[0] > 0 else "left"
        if v[0] == 0:
            motion_dir = "unknown"
        return f'push_{motion_dir}'


class PullPrimitive(ReachPrimitive):

    def reset_policy(self, pull_steps=10, wait_steps=5, speed_scale=1., pull_tangent=0, retreat_steps=0, retreat_direction=None, grab_force=500,
                     grab_binary=False, **kwargs):
        super(PullPrimitive, self).reset_policy(**kwargs)
        self._pull_steps = pull_steps
        self._wait_steps = wait_steps
        self._speed_scale = speed_scale
        self._curr_step = 0
        self._stage = 0
        self._pull_tangent = pull_tangent
        self._retreat_steps = retreat_steps
        self._retreat_direction = retreat_direction
        self._grab_force = grab_force
        self._grab_binary = grab_binary

    def get_action(self, model, observation, goal, **kwargs):
        observation = observation.leaf_apply(lambda arr: to_numpy(arr, check=True))
        if self._stage == 0:
            # first reach
            ac = super(PullPrimitive, self).get_action(model, observation, goal, **kwargs)
            if super(PullPrimitive, self).is_terminated(model, observation, goal, **kwargs):
                self._stage += 1
            setpoint_position = ac['target/position'].copy()
            setpoint_grab = ac['target/grab_binary'].copy()
        elif self._stage == 1:
            # pull in direction (pull_tangent)
            if self._pull_tangent == 0:
                v = self._target_offset.copy()
            else:
                v = self._pull_tangent * np.flip(self._target_offset).copy()
            v = self._speed_scale * v  # move away from block
            setpoint_position = (observation.position[0, 0] + v * self._env.dt * (self._pull_steps - self._curr_step))[None]
            setpoint_grab = np.ones((1, self._num_blocks)) if self._grab_binary else np.asarray([self._grab_force] * self._num_blocks)[None]
            v = np.where(np.abs(v) < 0.1, self._vel_noise * np.random.randn(2), v)  # noise in nonzero
            v = np.append(v, 50 * np.random.rand() + self._grab_force)  # while grabbing
            ac = AttrDict(action=v[None])
            self._curr_step += 1
            if self._curr_step > self._pull_steps:
                self._curr_step = 0
                self._stage += 1
            self.num_steps += 1
        elif self._stage == 2:
            # wait and disconnect
            v = np.array([0, 0, 0])
            ac = AttrDict(action=v[None])
            self._curr_step += 1
            if self._curr_step > self._wait_steps:
                self._curr_step = 0
                self._stage += 1
            setpoint_position = observation.position[:, 0].copy()
            setpoint_grab = np.ones((1, self._num_blocks)) if self._grab_binary else np.asarray([self._grab_force] * self._num_blocks)[None]
            self.num_steps += 1
        else:
            # retreat
            if self._retreat_steps > 0:
                if self._retreat_direction is not None:
                    v = + self._speed_scale * self._retreat_direction.copy()
                else:
                    v = + self._speed_scale * self._target_offset.copy()
                setpoint_position = (observation.position[0, 0] + v * self._env.dt * (
                        self._retreat_steps - self._curr_step))[None]

                # much larger lateral noise on retreat
                if not self._env.is_in_bounds():
                    if 0 > self._env.player_body.position.x or self._env.player_body.position.x > self._env.grid_size[0]:
                        v[0] = -v[0]
                    if 0 > self._env.player_body.position.y or self._env.player_body.position.y > self._env.grid_size[1]:
                        v[1] = -v[1]
                    # v = -v  # bang bang control kinda
                else:
                    v = np.where(np.abs(v) < 0.1, 10 * self._vel_noise * np.random.randn(2), v)
                v = np.append(v, 0)
                self._curr_step += 1
                if self._curr_step > self._retreat_steps:
                    self._curr_step = 0
                    self._stage += 1
            else:
                setpoint_position = observation.position[:, 0].copy()
                logger.warn("Pull Policy is done but action requested...")
                v = np.array([0, 0, 0])
            setpoint_grab = np.zeros((1, self._num_blocks))
            ac = AttrDict(action=v[None])
            self.num_steps += 1
        #
        # if self.curr_name == "pull_back_move_up":
        #     ac.policy_type = np.array([3])[None]  # overloaded (TODO remove)
        # else:
        ac.policy_type = np.array([1])[None]
        ac.policy_name = np.array([self.curr_name])[None]
        ac['target/position'] = setpoint_position.copy()
        ac['target/grab_binary'] = setpoint_grab.copy()

        # ac.pprint()
        return ac

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self._stage == 3 + int(self._retreat_steps > 0)

    @property
    def curr_name(self) -> str:
        v = self._target_offset.copy() if self._pull_tangent == 0 else self._pull_tangent * np.flip(self._target_offset).copy()
        tangent_direction = {-1: "tang_left", 0: "back", 1: "tang_right"}[self._pull_tangent]
        angle = np.arctan2(v[1], v[0])  # x right y up
        angle = (angle + np.pi/4) % (2 * np.pi) - np.pi/4
        if angle < np.pi/4:
            motion_dir = "move_right"
        elif angle < 3*np.pi/4:
            motion_dir = "move_up"
        elif angle < 5*np.pi/4:
            motion_dir = "move_left"
        else:
            motion_dir = "move_down"

        return f'pull_{tangent_direction}_{motion_dir}'


# user should specify these.
policy_types_ = AttrDict()


def register_policy_types(ptypes):
    policy_types_.safe_combine(ptypes, warn_conflicting=True)


def get_policy_types():
    return policy_types_.leaf_copy()

######## TELEOP POLICIES #########

# TODO
def get_block2d_keyboard_teleop_policy_params(smooth=True):
    def model_forward_fn(model, obs, goal, env=None, **kwargs):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and (
                    event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                act[1] = env._default_teleop_speed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                act[1] = -env._default_teleop_speed
            elif event.type == pygame.KEYUP and event.key in (pygame.K_i, pygame.K_k):
                act[1] = 0.

            if event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                act[0] = -block._default_teleop_speed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                act[0] = block._default_teleop_speed
            elif event.type == pygame.KEYUP and event.key in (pygame.K_j, pygame.K_l):
                act[0] = 0.

            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                act[2] = 500.
            elif event.type == pygame.KEYUP and event.key == pygame.K_g:
                act[2] = 0.

        # for FunctionalPolicy & subclasses
        return AttrDict(
            policy_model_forward_fn=model_forward_fn,
            provide_policy_env_in_forward=True,
            is_terminated_fn=None,
            # reset_policy_fn=None,
        )
