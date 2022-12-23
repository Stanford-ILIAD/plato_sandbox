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

from sbrl.envs.block2d.block_env_2d import BlockEnv2D, get_block2d_example_params
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.utils.python_utils import get_with_default, AttrDict


class StackBlockEnv2D(BlockEnv2D):

    def _init_params_to_attrs(self, params):
        params.num_maze_cells = get_with_default(params, "num_maze_cells", 8, map_fn=int)
        params.grid_size = get_with_default(params, "grid_size", np.array([400., 400.]), map_fn=np.array)
        params.gravity = get_with_default(params, "gravity", (0., -100.))
        params.damping = get_with_default(params, "damping", 0.75)
        params.ego_block_size = get_with_default(params, "ego_block_size", 40)
        params.block_size = get_with_default(params, "block_size", (40, 80))
        params.block_mass = get_with_default(params, "block_mass", 40)
        params.block_corner_radius = get_with_default(params, "block_corner_radius", 3.)
        params.block_grabbing_frac = get_with_default(params, "block_grabbing_frac", 1.5)
        params.block_friction = get_with_default(params, "block_friction", 0.2)
        params.block_bbox = get_with_default(params, "block_bbox", True)
        params.static_line_friction = get_with_default(params, "static_line_friction", 0.4)
        params.num_blocks = get_with_default(params, "num_blocks", 1)
        params.default_teleop_speed = get_with_default(params, "default_teleop_speed", 120.)
        params.initialization_steps = get_with_default(params, "initialization_steps", 30)
        params.grab_add_rotary_limit_joint = get_with_default(params, "grab_add_rotary_limit_joint", False)
        params.break_constraints_on_large_impulse = get_with_default(params, "break_constraints_on_large_impulse", True)
        params.grab_slider_min_frac = get_with_default(params, "grab_slider_min_frac", 0.5)
        params.max_velocity = get_with_default(params, "max_velocity", 100, map_fn=float)
        # maze
        maze = np.zeros((params.num_maze_cells, params.num_maze_cells))
        maze[params.num_maze_cells - 3, 0] = 8  # left most ledge
        maze[params.num_maze_cells - 3, 1] = 8  # left most ledge
        maze[params.num_maze_cells - 5, params.num_maze_cells - 1] = 8  # right most ledge (lowest)
        maze[params.num_maze_cells - 5, params.num_maze_cells - 2] = 8  # right most ledge (lowest)
        maze[params.num_maze_cells - 2, params.num_maze_cells // 2] = 8  # middle ledge (highest)
        maze[params.num_maze_cells - 2, params.num_maze_cells // 2 + 1] = 8  # middle ledge (highest)
        params.fixed_np_maze = get_with_default(params, "fixed_np_maze", maze)
        # bottom half only
        # vs_default = np.meshgrid(range(params.num_maze_cells), range(params.num_maze_cells // 2))
        # vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
        # params.valid_start_idxs = get_with_default(params, "valid_start_idxs", vs_default)
        super(StackBlockEnv2D, self)._init_params_to_attrs(params)

    def reset(self, presets: AttrDict = AttrDict()):
        obs, goal = super(StackBlockEnv2D, self).reset(presets)
        invalid_angle = (np.abs([self.bodies[i].angle for i in range(self.num_blocks)]) > np.pi/10).any()
        invalid_block_pos = any([body.position.x < 0 or body.position.x > self.grid_size[0] or body.position.y < 0 or
                                body.position.y > self.grid_size[1] for body in (self.bodies + [self.player_body])])
        invalid_reset = invalid_angle or invalid_block_pos
        while invalid_reset:
            obs, goal = super(StackBlockEnv2D, self).reset(presets)
            invalid_angle = (np.abs([self.bodies[i].angle for i in range(self.num_blocks)]) > np.pi / 10).any()
            invalid_block_pos = any(
                [body.position.x < 0 or body.position.x > self.grid_size[0] or body.position.y < 0 or
                 body.position.y > self.grid_size[1] for body in (self.bodies + [self.player_body])])
            invalid_reset = invalid_angle or invalid_block_pos
        return obs, goal

    def get_block_positions(self, presets):
        locations = super(StackBlockEnv2D, self).get_block_positions(presets)
        # ego block should always be the highest to start. (permute the order accordingly)
        highest_idx = np.argmax(locations[:, 1])
        return np.concatenate([
            locations[:highest_idx],
            locations[highest_idx+1:],
            locations[highest_idx:highest_idx+1]  # ego last
        ], axis=0)


def get_stack_block2d_example_params(grid_size=(400, 400), block_max_size=(40, 80), block_lower=None, block_upper=None, start_near_bottom=True):
    env_params = AttrDict()
    env_params.render = True
    env_params.realtime = True
    env_params.keep_in_bounds = True
    env_params.grab_action_binary = True
    env_params.grid_size = np.asarray(grid_size)
    env_params.block_lower = block_lower
    env_params.block_upper = block_upper
    if block_upper is not None:
        assert block_lower is not None, block_lower
        block_max_size = block_upper
    else:
        env_params.block_size = block_max_size

    num_blocks = (np.asarray(grid_size) / np.asarray(block_max_size)).astype(int)

    if start_near_bottom:
        vs_default = np.meshgrid(range(num_blocks[0]), range(num_blocks[1] // 2))
        vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
        env_params.valid_start_idxs = vs_default
    return env_params


if __name__ == "__main__":

    _, env_spec_params = get_block2d_example_params()

    env_params = AttrDict()
    env_params.render = True
    env_params.realtime = True
    env_params.keep_in_bounds = True
    env_params.grab_action_binary = True

    env_spec = ParamEnvSpec(env_spec_params)
    block = StackBlockEnv2D(env_params, env_spec)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    block.user_input_reset(1)  # trolling with a fake UI

    running = True
    gamma = 0.5
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
                block.reset()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                act[1] = block._default_teleop_speed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                act[1] = -block._default_teleop_speed
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

        # print(act)
        smoothed_act = gamma * act[:2] + (1-gamma) * last_act[:2]
        smoothed_act = np.concatenate([smoothed_act, act[2:]])

        act_dict = AttrDict(action=smoothed_act[None])
        last_act = smoothed_act

        # print(smoothed_act)
        obs, goal, done = block.step(act_dict)

        # cv2.imshow("image_test", block.get_img())
        # cv2.waitKey(1)
