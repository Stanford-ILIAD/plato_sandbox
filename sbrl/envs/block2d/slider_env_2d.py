"""
BLOCK MAZE ENVIRONMENT, 2D

A randomly generated maze, with pymunk physics. some blocks will be created, which are dynamic but cannot move between maze walls

you control the RED block, which can move freely between walls. Colliding with a dynamic block will cause it to move!

Controls:
i: move up
k: move down
j: move left
l: move right

g: grab objects within some distance of you
"""

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.utils.python_utils import get_with_default, AttrDict


class SliderBlockEnv2D(BlockEnv2D):

    def _init_params_to_attrs(self, params):
        params.num_maze_cells = get_with_default(params, "num_maze_cells", 5, map_fn=int)
        params.num_blocks = get_with_default(params, "num_blocks", 2, map_fn=int)
        params.fixed_np_maze = get_with_default(params, "fixed_np_maze",
                                                np.zeros((params.num_maze_cells, params.num_maze_cells)))

        super(SliderBlockEnv2D, self)._init_params_to_attrs(params)

        self.slider_x_center = self.grid_size / 2 - self.grid_size / 5.
        self.slider_x_range = np.asarray([-self.block_size * 1.5, self.block_size * 1.5])
        self.slider_y_center = self.grid_size / 2 + self.grid_size / 5.
        self.slider_y_range = np.asarray([-self.block_size * 1.5, self.block_size * 1.5])

        if self.num_blocks == 1:
            self.slider_x_center = self.grid_size / 2
            self.slider_x_range = np.asarray([-self.block_size * 2, self.block_size * 2])

        assert self.num_blocks in [1, 2]

    def create_world(self, presets: AttrDict = AttrDict()):
        super(SliderBlockEnv2D, self).create_world(presets)
        self._slider_x = pymunk.Body(body_type=pymunk.Body.STATIC)
        self._slider_x.position = Vec2d(*self.slider_x_center)
        self._slider_y = pymunk.Body(body_type=pymunk.Body.STATIC)
        self._slider_y.position = Vec2d(*self.slider_y_center)
        if self.num_blocks == 1:
            self.world.add(self._slider_x)
        else:
            self.world.add(self._slider_x, self._slider_y)

        self._constraint_x_rot = pymunk.RotaryLimitJoint(self._slider_x, self.bodies[0], 0, 0)
        self._constraint_x = pymunk.GrooveJoint(self._slider_x, self.bodies[0], (self.slider_x_range[0], 0), (self.slider_x_range[1], 0), (0, 0))
        if self.num_blocks == 1:
            self.world.add(self._constraint_x, self._constraint_x_rot)
        else:
            self._constraint_y_rot = pymunk.RotaryLimitJoint(self._slider_y, self.bodies[1], 0, 0)
            self._constraint_y = pymunk.GrooveJoint(self._slider_y, self.bodies[1], (0, self.slider_y_range[0]), (0, self.slider_y_range[1]), (0, 0))
            self.world.add(self._constraint_x, self._constraint_y, self._constraint_x_rot, self._constraint_y_rot)

    def get_block_positions(self, presets: AttrDict = AttrDict()):
        # returns loc idx for each block, and blocks per axis to compute chunk size

        # slider 1
        sl1 = np.random.uniform(self.slider_x_range[0], self.slider_x_range[1])
        sl2 = np.random.uniform(self.slider_y_range[0], self.slider_y_range[1])

        b1 = self.slider_x_center + np.asarray([sl1, 0])
        b2 = self.slider_y_center + np.asarray([0, sl2])

        # generate again
        b_ego = np.random.uniform([0, 0], self.grid_size - self.block_size, 2)
        max_iters = 50
        i = 0
        while i < max_iters and np.linalg.norm(b_ego - b1) < 1.5 * self.block_size or np.linalg.norm(b_ego - b2) < 1.5 * self.block_size:
            b_ego = np.random.uniform([0, 0], self.grid_size - self.block_size, 2)
            i += 1

        if self.num_blocks == 1:
            return np.stack([b1, b_ego])
        else:
            return np.stack([b1, b2, b_ego])

    def _get_obs(self):
        obs = super(SliderBlockEnv2D, self)._get_obs()
        vec1 = self.bodies[0].position
        lin1 = np.sum(np.asarray([vec1.x, vec1.y]) - self.slider_x_center)
        # 1, 2
        if self.num_blocks == 1:
            obs.block_linear_positions = np.asarray([lin1])[None]
        else:
            vec2 = self.bodies[1].position
            lin2 = np.sum(np.asarray([vec2.x, vec2.y]) - self.slider_y_center)
            obs.block_linear_positions = np.asarray([lin1, lin2])[None]
        return obs

    # def set_state(self, obs):
    #     # TODO
    #     su


if __name__ == '__main__':
    env = SliderBlockEnv2D(AttrDict(num_blocks=1, block_size=80, grab_action_binary=True, grab_action_max_force=10000), ParamEnvSpec(AttrDict()))
    env.user_input_reset(1)  # trolling with a fake UI

    running = True
    gamma = 0.99
    last_act = np.zeros(3)
    act = np.zeros(3)
    while running:
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
                act[1] = 75.
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                act[1] = -75.
            elif event.type == pygame.KEYUP and event.key in (pygame.K_i, pygame.K_k):
                act[1] = 0.

            if event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                act[0] = -75.
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                act[0] = 75.
            elif event.type == pygame.KEYUP and event.key in (pygame.K_j, pygame.K_l):
                act[0] = 0.

            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                act[2] = 1000.
            elif event.type == pygame.KEYUP and event.key == pygame.K_g:
                act[2] = 0.

        # print(act)
        smoothed_act = gamma * act[:2] + (1 - gamma) * last_act[:2]
        smoothed_act = np.concatenate([smoothed_act, act[2:]])

        act_dict = AttrDict(action=smoothed_act[None])
        last_act = smoothed_act

        # print(smoothed_act)
        obs, goal, done = env.step(act_dict)

        # cv2.imshow("image_test", env.get_img())
        # cv2.waitKey(1)