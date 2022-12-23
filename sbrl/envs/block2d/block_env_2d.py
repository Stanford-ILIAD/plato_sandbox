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

from typing import List

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import torch
from pymunk import Vec2d
from scipy.spatial import distance_matrix

from sbrl.envs.block2d import teleop_functions
from sbrl.envs.env import Env
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.utils import plt_utils
from sbrl.utils.cv_utils import cv2
from sbrl.utils.geometry_utils import Maze
from sbrl.utils.input_utils import UserInput
from sbrl.utils.pygame_utils import PygameDisplayWrapper
from sbrl.utils.python_utils import get_with_default, AttrDict
from sbrl.utils.torch_utils import to_numpy


class BlockEnv2D(Env):

    def __init__(self, params, env_spec):
        super().__init__(params, env_spec)
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        self.num_blocks = get_with_default(params, "num_blocks", 10)
        self.dt = get_with_default(params, "dt", 0.1)
        # noisy dt for each step, dt will be scaled by uniform[1 - scale, 1 + scale)
        self._dt_scale = get_with_default(params, "dt_scale", 0)
        assert self.dt > 1. / 250
        assert 1 > self._dt_scale >= 0

        # when adding action noise (w.p. action_noise_prob), will be dtheta ~ N(0, action_noise_theta)
        self._action_noise_theta = get_with_default(params, "action_noise_theta", np.deg2rad(10))
        self._action_noise_prob = get_with_default(params, "action_noise_prob", 0)

        if self._action_noise_prob > 0:
            logger.debug(f"Adding action noise std={np.rad2deg(self._action_noise_theta)} with prob={self._action_noise_prob}")

        self.grid_size = get_with_default(params, "grid_size", np.array([600., 600.]), map_fn=np.array)
        self.image_size = get_with_default(params, "image_size", self.grid_size.astype(int), map_fn=np.array)
        self._default_teleop_speed = get_with_default(params, "default_teleop_speed", 75., map_fn=float)

        self._gravity = get_with_default(params, "gravity", (0, 0), map_fn=tuple)
        self._damping = get_with_default(params, "damping", 0.5, map_fn=float)
        self._max_velocity = get_with_default(params, "max_velocity", 100, map_fn=float)
        self._static_line_friction = get_with_default(params, "static_line_friction", 0., map_fn=float)

        self.block_size = get_with_default(params, "block_size", 20, map_fn=np.asarray)
        self.ego_block_size = get_with_default(params, "ego_block_size", self.block_size, map_fn=np.asarray)
        self._block_grabbing_frac = get_with_default(params, "block_grabbing_frac", 1.15, map_fn=float)
        self.block_mass = get_with_default(params, "block_mass", 10., map_fn=float)
        self.block_corner_radius = get_with_default(params, "block_corner_radius", 0., map_fn=float)
        self._block_friction = get_with_default(params, "block_friction", 0.3, map_fn=float)
        self._block_bbox = get_with_default(params, "block_bbox", False)
        self._grab_one_only = get_with_default(params, "grab_one_only", False)

        self._block_size_lower = get_with_default(params, "block_size_lower", None)
        self._block_size_upper = get_with_default(params, "block_size_upper", None)
        assert (self._block_size_lower is None) == (self._block_size_upper is None), "Both or None must be specified"
        if self._block_size_upper is not None:
            self._block_size_lower = np.broadcast_to(self._block_size_lower, (2,))
            self._block_size_upper = np.broadcast_to(self._block_size_upper, (2,))
            assert np.all(self._block_size_upper >= self._block_size_lower), [self._block_size_lower, "not <=", self._block_size_upper]
            self.num_blocks_per_grid_axis = np.floor(self.grid_size / self._block_size_upper).astype(int)
            self.num_blocks_per_grid_axis = np.floor(self.grid_size / self._block_size_upper).astype(int)
        else:
            self.num_blocks_per_grid_axis = np.floor(self.grid_size / self.block_size).astype(int)
            self.num_blocks_per_grid_axis = np.floor(self.grid_size / self.block_size).astype(int)

        self.all_block_sizes = None

        self.render = get_with_default(params, "render", True)
        self.num_maze_cells = get_with_default(params, "num_maze_cells", 5, map_fn=int)
        self.fixed_np_maze = get_with_default(params, "fixed_np_maze", None)
        if self.fixed_np_maze is not None:
            self.fixed_np_maze = np.asarray(self.fixed_np_maze)
            assert self.fixed_np_maze.shape[0] == self.fixed_np_maze.shape[
                1] == self.num_maze_cells, self.fixed_np_maze.shape
        self.valid_start_idxs = get_with_default(params, "valid_start_idxs", None)
        assert self.valid_start_idxs is None or np.all(self.valid_start_idxs < self.num_blocks_per_grid_axis[None])

        # if not None, will treat all goals as
        self.valid_goal_idxs = get_with_default(params, "valid_goal_idxs", None)
        assert self.valid_goal_idxs is None or np.all(self.valid_goal_idxs < self.num_blocks_per_grid_axis[None])

        self.keep_in_bounds = get_with_default(params, "keep_in_bounds", False)

        self._t = 0
        self._horizon = get_with_default(params, "horizon", np.float("inf"))
        self.realtime = get_with_default(params, "realtime", self.render)
        self.disable_images = get_with_default(params, "disable_images", False)
        if self.disable_images:
            assert not self.render, "Render must not be enabled with images disabled"

        self.initialization_steps = get_with_default(params, "initialization_steps", 0)
        if self.initialization_steps > 0:
            logger.debug(f"Initialization steps: {self.initialization_steps}")

        # if this is true, grab is 0/1 toggle with 1=action_force at the link distance
        self.grab_action_binary = get_with_default(params, "grab_action_binary", False)
        self._break_constraints_on_large_impulse = get_with_default(params, "break_constraints_on_large_impulse", False)
        self.grab_add_rotary_limit_joint = get_with_default(params, "grab_add_rotary_limit_joint", True)
        # if non None, will construct slider joint on grab with min a multiple of starting dist
        self.grab_slider_min_frac = get_with_default(params, "grab_slider_min_frac", None)
        self.grab_action_max_force = get_with_default(params, "grab_action_max_force", np.inf)

        self.do_wall_collisions = get_with_default(params, "do_wall_collisions", False)

        self.maximize_agent_block_distance = get_with_default(params, "maximize_agent_block_distance", False)

        logger.info("Block2D initialized with [render = %s, disabled_img = %s, realtime = %s" % (
        self.render, self.disable_images, self.realtime))

        self._teleop_fn = get_with_default(params, "teleop_fn", teleop_functions.get_pygame_mouse_teleop_fn())

        self._done_on_success = get_with_default(params, "done_on_success", False)

    def _init_setup(self):
        self.extra_memory = AttrDict()
        self.cell_width = self.grid_size / self.num_maze_cells
        self.create_world()

        pygame.font.init()  # you have to call this at the start,
        # if you want to use this module.
        self.font = pygame.font.SysFont('Comic Sans MS', 15)

        if self.render:
            # setting env level display, calls create_render once
            self._display = PygameDisplayWrapper(AttrDict(create_display_fn=self.create_render))
            self._display.create_display()
        elif not self.disable_images:
            pygame.init()
            self.screen = pygame.Surface(np.ceil(self.grid_size).astype(int))
            self.clock = pygame.time.Clock()
            pymunk.pygame_util.positive_y_is_up = True
            self.reset_render()

    def grid_index_to_corner_pos(self, idxs):
        cell_size = self.grid_size / self.num_maze_cells
        return idxs * cell_size

    def get_random_positions(self, count, valid_idxs, clip_radius=None):
        # gets count allowed positions from indices in valid_idxs, using grid indexing.
        if valid_idxs is not None:
            blks_per_cell = np.maximum(np.floor(self.num_blocks_per_grid_axis / self.num_maze_cells), 1)
            blks_per_dim = blks_per_cell * self.num_maze_cells
            # these are MAZE coordinates passed in, blocks can be generated by knowing the mult factor
            final_list = []
            for idx in valid_idxs:
                for i in range(int(blks_per_cell[0])):
                    for j in range(int(blks_per_cell[1])):
                        final_list.append([idx[0] * blks_per_cell[0] + i, idx[1] * blks_per_cell[1] + j])
            idxs = np.random.choice(len(final_list), count, replace=False)
            locs = np.array(final_list)[idxs]
        else:
            blks_per_dim = self.num_blocks_per_grid_axis
            choice = np.arange(int(np.prod(self.num_blocks_per_grid_axis)))
            packed_idxs = np.random.choice(choice, count, replace=False)
            locs = np.column_stack(
                [packed_idxs // self.num_blocks_per_grid_axis[1], packed_idxs % self.num_blocks_per_grid_axis[1]])

        chunk_size = self.grid_size / blks_per_dim
        positions = locs * chunk_size + 0.5 * self.block_size

        if clip_radius is not None:
            positions = np.clip(positions, clip_radius, self.grid_size - clip_radius)

        return positions

    def get_goal_position(self):
        return self.get_random_positions(1, self.valid_goal_idxs, clip_radius=self.ego_block_size)

    def get_block_positions(self, presets):
        return self.get_random_positions(self.num_blocks + 1, self.valid_start_idxs)

    def create_world(self, presets: AttrDict = AttrDict()):
        self.world = pymunk.Space()
        self.world.gravity = get_with_default(presets, "gravity", self._gravity)
        self.world.damping = self._damping
        self._stop_counter = 0

        def limit_velocity(body, gravity, damping, dt):
            max_velocity = self._max_velocity
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            l = body.velocity.length
            if l > max_velocity:
                scale = max_velocity / l
                body.velocity = body.velocity * scale

        # world boundary
        self.static_lines = [
            pymunk.Segment(self.world.static_body, Vec2d(0, 0), Vec2d(0, self.grid_size[1]), 3),
            pymunk.Segment(self.world.static_body, Vec2d(0, self.grid_size[1]), Vec2d(*self.grid_size), 3),
            pymunk.Segment(self.world.static_body, Vec2d(*self.grid_size), Vec2d(self.grid_size[0], 0), 3),
            pymunk.Segment(self.world.static_body, Vec2d(self.grid_size[0], 0), Vec2d(0, 0), 3),
        ]

        self.maze = Maze(self.num_maze_cells, self.num_maze_cells)

        # option to load in predefined maze
        if presets.has_leaf_key("maze"):
            self.maze.from_numpy(np.asarray(presets.maze))
        elif self.fixed_np_maze is not None:
            self.maze.from_numpy(self.fixed_np_maze)
        else:
            self.maze.make_maze()

        self.maze_np = self.maze.to_numpy()
        # self.maze.write_svg("test.svg")

        # columns
        for i in range(self.num_maze_cells + 1):
            xi_walls = self.maze.get_contiguous_x_walls(i)
            yi_walls = self.maze.get_contiguous_y_walls(i)

            for x1, x2 in xi_walls:
                self.static_lines.append(
                    pymunk.Segment(self.world.static_body, Vec2d(x1 * self.cell_width[0], i * self.cell_width[1]),
                                   Vec2d((x2 + 1) * self.cell_width[0], i * self.cell_width[1]), 3),
                )

            for y1, y2 in yi_walls:
                self.static_lines.append(
                    pymunk.Segment(self.world.static_body, Vec2d(i * self.cell_width[0], y1 * self.cell_width[1]),
                                   Vec2d(i * self.cell_width[0], (y2 + 1) * self.cell_width[1]), 3),
                )

        for l in self.static_lines:
            l.friction = self._static_line_friction
            l.elasticity = 1.0
        self.world.add(*self.static_lines)

        if self.valid_goal_idxs is not None:
            self.goal_pos = get_with_default(presets, "goal_position", self.get_goal_position()).reshape(2)
            self.goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            # self.goal_body.position = list(self.goal_pos)  # static
            self.goal_body.position = list(self.goal_pos)  # static
            self.goal_shape = pymunk.Circle(self.goal_body, radius=5)
            self.goal_shape.color = (10, 255, 10, 255)
            self.world.add(self.goal_body, self.goal_shape)

        # N + 1
        body_positions = self.get_block_positions(presets)
        if presets.has_leaf_key('block_positions'):
            body_positions[:-1] = presets >> "block_positions"
        if presets.has_leaf_key('position'):
            body_positions[-1] = presets >> "position"

        assert body_positions.shape[0] == self.num_blocks + 1
        if self.maximize_agent_block_distance:
            num_candidates = 40
            block_positions = body_positions[:-1]
            candidate_agent_positions = np.random.uniform(low=np.array([0.0, 0.0]), high=self.grid_size, size=(num_candidates, 2))
            min_candidate_distances = distance_matrix(candidate_agent_positions, block_positions).min(-1)
            farthest_candidate_position = np.argmax(min_candidate_distances)
            agent_pos = candidate_agent_positions[farthest_candidate_position][None]
            body_positions = np.concatenate((block_positions, agent_pos), 0)
        self.bodies = []

        ## COLORS
        if presets.has_leaf_key('block_colors'):
            self._colors = np.asarray(presets.block_colors)
        else:
            self._colors = np.random.randint(50, 255, size=(self.num_blocks, 4))
            self._colors[:, 0] = 100.
            self._colors[:, -1] = 255.
        assert list(self._colors.shape) == [self.num_blocks, 4], self._colors

        ## MASS
        if presets.has_leaf_key('block_masses'):
            self._masses = np.asarray(presets.block_masses)
        else:
            self._masses = np.ones((self.num_blocks,)) * self.block_mass
        assert list(self._masses.shape) == [self.num_blocks] and np.all(self._masses > 0), self._masses

        if presets.has_leaf_key("block_sizes"):
            self.all_block_sizes = np.asarray(presets >> "block_sizes")
            assert list(self.all_block_sizes.shape) == [self.num_blocks, 2]
        else:
            self.all_block_sizes = None

        all_block_sizes = []
        for i, color, mass, location in zip(range(self.num_blocks), self._colors, self._masses, body_positions[:-1]):
            # idx_loc = np.array([pack_id // self.num_blocks_per_grid_axis[1], pack_id % self.num_blocks_per_grid_axis[1]])
            # location = idx_loc * chunk_size + 0.5 * self.block_size
            if self.all_block_sizes is not None:
                xy_block_size = self.all_block_sizes[i]
            elif self._block_size_upper is not None:
                xy_block_size = np.random.uniform(self._block_size_lower, self._block_size_upper)
            else:
                xy_block_size = np.broadcast_to(self.block_size, (2,))
            moment = pymunk.moment_for_box(mass, tuple(xy_block_size))
            body = pymunk.Body(mass, moment)
            body.position = Vec2d(*location)
            body.velocity_func = limit_velocity

            shape = pymunk.Poly.create_box(body, tuple(xy_block_size), radius=self.block_corner_radius)
            shape.friction = self._block_friction
            shape.color = color.tolist()
            self.world.add(body, shape)
            self.bodies.append(body)

            all_block_sizes.append(xy_block_size)

        # idx_loc = np.array([packed_idxs[-1] // self.num_blocks_per_grid_axis[1], packed_idxs[-1] % self.num_blocks_per_grid_axis[1]])
        # location = corner_locs[-1] * chunk_size + 0.5 * self.block_size
        location = body_positions[-1]

        my_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        my_body.position = Vec2d(*location)
        my_shape = pymunk.Poly.create_box(my_body, tuple(np.broadcast_to(self.ego_block_size, (2,))))
        my_shape.friction = self._block_friction
        my_shape.color = (255, 10, 10, 255)
        self.world.add(my_body, my_shape)
        self.player_body = my_body

        self.added_constraints = [None for _ in range(self.num_blocks)]
        self.extra_constraints = [[] for _ in range(self.num_blocks)]

        self.all_block_sizes = np.stack(all_block_sizes) if len(all_block_sizes) > 0 else np.zeros((0, 2))

        # visualization extra shapes
        self.draw_shapes = []

    def create_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode(np.ceil(self.grid_size).astype(int))
        self.clock = pygame.time.Clock()
        pymunk.pygame_util.positive_y_is_up = True

        self.reset_render()

        return self.screen

    # called on reset
    def reset_render(self):
        self.screen.fill(pygame.Color("black"))
        self.options = pymunk.pygame_util.DrawOptions(self.screen)
        self.options.flags = self.options.DRAW_SHAPES + self.options.DRAW_CONSTRAINTS
        self.world.debug_draw(self.options)

    def get_draw_shapes(self) -> List:
        return self.draw_shapes

    def set_draw_shapes(self, ls):
        self.draw_shapes = list(ls)

    def add_draw_shapes(self, ls):
        self.draw_shapes.extend(list(ls))

    ## extra helpers for drawing random stuff (called on reset)
    def setup_draw_reward(self, get_reward_fn, color=None, **kwargs):
        if color is None:
            color = (plt_utils.cadet_blue[:3] * 255).astype(int).tolist()

        def draw_rew(ops):
            tcolor = list(color)
            self.screen.blit(self.font.render(f"Reward: {get_reward_fn(self)}", False, tuple(tcolor)), (25, 25))

        extra_draw_actions = [draw_rew]
        self.set_draw_shapes(extra_draw_actions)

    def step_render(self):
        if not self.disable_images:
            self.screen.fill(pygame.Color("black"))
            ### Draw stuff
            self.world.debug_draw(self.options)
            ### add extras
            [draw_fn(self.options) for draw_fn in self.draw_shapes]

        if self.render:
            pygame.display.flip()

    def get_img(self):
        # Get the entire image
        # if self.render:
        image = pygame.surfarray.pixels3d(self.screen)
        # image = np.zeros(list(self.image_size) + [3])
        # Swap the axes as the X and Y axes in Pygame and Scipy are opposite
        image_rotated = np.swapaxes(image, 0, 1)
        image_rotated = image_rotated[:, :, ::-1]  # channel flip bgr -> rgb
        image_rotated = cv2.resize(image_rotated, dsize=tuple(self.image_size))
        # Copy the array, otherwise the surface will be locked
        return np.asarray(image_rotated)

    # def update(self, dt):
    #     # Here we use a very basic way to keep a set space.step dt.
    #     # For a real game its probably best to do something more complicated.
    #     step_dt = 1 / 250.0
    #     x = 0
    #     while x < dt:
    #         x += step_dt
    #         self.space.step(step_dt)

    # ONLY CALL WHEN YOU DON'T NEED TO SIMULATE
    def set_state(self, obs):
        position, velocity, angle, angular_velocity, force, torque, \
        block_positions, block_velocities, block_angles, block_angular_velocities, block_forces, block_torques, \
        active_constraints = obs.leaf_apply(lambda arr: arr[0]) \
            .get_keys_required(['position', 'velocity', 'angle', 'angular_velocity', 'force', 'torque',
                                'block_positions', 'block_velocities', 'block_angles', 'block_angular_velocities',
                                'block_forces', 'block_torques',
                                'active_constraints'])
        self.player_body.position = position.tolist()
        self.player_body.velocity = velocity.tolist()
        self.player_body.force = force.tolist()
        self.player_body.angle = angle[0]
        self.player_body.angular_velocity = angular_velocity[0]
        self.player_body.torque = torque[0]

        if self.grab_action_binary:
            grab_vector, grab_distance = obs.leaf_apply(lambda arr: arr[0]).get_keys_required(
                ['grab_vector', 'grab_distance'])

        if obs.has_leaf_key('target/position'):
            pass  # TODO

        for i in range(self.num_blocks):
            self.bodies[i].position = block_positions[i].tolist()
            self.bodies[i].velocity = block_velocities[i].tolist()
            self.bodies[i].force = block_forces[i].tolist()
            self.bodies[i].angle = block_angles[i]
            self.bodies[i].angular_velocity = block_angular_velocities[i]
            self.bodies[i].torque = block_torques[i]

            if obs.has_leaf_key('block_colors'):
                c = obs.block_colors[0, i]
                list(self.bodies[i].shapes)[0].color = c.tolist()

            if self.added_constraints[i] is not None:
                self.world.remove(self.added_constraints[i])
                self.added_constraints[i] = None
            for j in range(len(self.extra_constraints[i])):
                self.world.remove(self.extra_constraints[i][j])
            self.extra_constraints[i] = []

            if active_constraints[i] > 0:
                if self.grab_action_binary:
                    to_add = []
                    if self.grab_add_rotary_limit_joint:
                        dx = grab_vector[i, 0]
                        dy = grab_vector[i, 1]
                        cstr2 = pymunk.GrooveJoint(self.player_body, self.bodies[i], groove_a=(0, 0), groove_b=(dx, dy),
                                                   anchor_b=(0, 0))
                        # cstr2 = pymunk.PinJoint(self.player_body, self.bodies[i], anchor_a=(self.block_size / 4, self.block_size / 4),
                        #                         anchor_b=(self.block_size / 4, self.block_size / 4))
                        cstr2.max_force = self.grab_action_max_force
                        # cstr2.distance = active_constraints[i]  # TODO is this ok?
                        to_add = [cstr2]
                        cstr = pymunk.PinJoint(self.player_body, self.bodies[i])
                        cstr.max_force = self.grab_action_max_force
                        cstr.distance = grab_distance[i]  # TODO is this ok?
                        cstr3 = pymunk.RotaryLimitJoint(self.player_body, self.bodies[i], 0, 0)
                        to_add.append(cstr3)
                    else:
                        cstr = pymunk.SlideJoint(self.player_body, self.bodies[i], (0, 0), (0, 0),
                                                 min=grab_distance[i] * 0.77, max=grab_distance[i] * 1.41)
                        cstr.max_force = self.grab_action_max_force
                        # cstr.distance = grab_distance[i]  # TODO is this ok?

                    if len(to_add) > 0:
                        self.world.add(*to_add)
                        self.extra_constraints[i].extend(to_add)
                else:
                    cstr = pymunk.PinJoint(self.player_body, self.bodies[i])
                    cstr.max_force = 500. * self.bodies[i].mass
                    cstr.distance = 1. * np.max(self.block_size)  # TODO is this ok?
                self.world.add(cstr)
                self.added_constraints[i] = cstr

        self.world.step(1. / 1000.)  # just to update the renderer
        self.step_render()

    def activate_grab(self, i, dist, action):
        # turn on the joint
        if self.grab_action_binary:
            # apply a binary (rigid) joint between bodies
            if self.grab_slider_min_frac is not None:
                cstr = pymunk.SlideJoint(self.player_body, self.bodies[i], (0, 0), (0, 0),
                                         self.grab_slider_min_frac * dist, dist)
            else:
                cstr = pymunk.PinJoint(self.player_body, self.bodies[i])
            cstr.max_force = self.grab_action_max_force
            cstr.distance = dist
            self.world.add(cstr)
            self.added_constraints[i] = cstr
            if self.grab_add_rotary_limit_joint:
                dx = self.bodies[i].position.x - self.player_body.position.x
                dy = self.bodies[i].position.y - self.player_body.position.y
                cstr2 = pymunk.GrooveJoint(self.player_body, self.bodies[i], groove_a=(0, 0),
                                           groove_b=(dx, dy),
                                           anchor_b=(0, 0))
                cstr2.max_force = self.grab_action_max_force
                # cstr2.distance = dist
                to_add = [cstr2]
                cstr3 = pymunk.RotaryLimitJoint(self.player_body, self.bodies[i], 0, 0)
                to_add.append(cstr3)
                self.world.add(*to_add)
                self.extra_constraints[i].extend(to_add)
        else:
            # apply this amount of acceleration to the body
            cstr = pymunk.PinJoint(self.player_body, self.bodies[i])
            cstr.max_force = np.max(action.action[0, 2:]) * self.bodies[i].mass
            cstr.distance = dist
            self.world.add(cstr)
            self.added_constraints[i] = cstr

    def step(self, action):
        action = action.leaf_apply(lambda arr: to_numpy(arr) if isinstance(arr, torch.Tensor) else arr)
        # print(action.action)

        ## Grabbing (can be specified by 1 or more values)
        if np.any(action.action[0, 2:] > 0):
            all_dists = []
            for i in range(len(self.bodies)):
                dist = np.linalg.norm(self.bodies[i].position - self.player_body.position)
                all_dists.append(dist)
                thresh = self._block_grabbing_frac * np.maximum(np.max(self.ego_block_size),
                                                                np.max(self.all_block_sizes[i]))
                if not self._grab_one_only and self.added_constraints[i] is None and dist <= thresh:
                    # turn on this joint
                    self.activate_grab(i, dist, action)
            if self._grab_one_only:
                # grab the closest one, if nothing is active right now.
                block_id_closest = np.argmin(all_dists)
                any_active = any([c is not None for c in self.added_constraints])
                thresh = self._block_grabbing_frac * np.maximum(1.4 * np.max(self.ego_block_size), np.max(self.all_block_sizes[block_id_closest]))
                if not any_active and all_dists[block_id_closest] <= thresh:
                    self.activate_grab(block_id_closest, all_dists[block_id_closest], action)
        else:
            # clear all holds when action becomes zero
            for i in range(len(self.bodies)):
                if self.added_constraints[i] is not None:
                    self.world.remove(self.added_constraints[i])
                    self.added_constraints[i] = None
                for j in range(len(self.extra_constraints[i])):
                    self.world.remove(self.extra_constraints[i][j])
                self.extra_constraints[i] = []

        # PLAYER VELOCITY

        vel = action.action[0, :2].copy()
        if np.random.random() < self._action_noise_prob:
            old_vel = vel
            # magnitude unchanged, theta is changed
            theta = np.arctan2(vel[1], vel[0]) + self._action_noise_theta * np.random.randn()
            vel = np.linalg.norm(vel) * np.asarray([np.cos(theta), np.sin(theta)])
            # print(old_vel, vel)

        def limit_force(arbiter, *args, **kwargs):
            nonlocal too_much_force, impulse_dir
            if np.linalg.norm(arbiter.total_impulse) > 50000:  # too much impulse on this object
                too_much_force = True
                impulse_dir += -arbiter.total_impulse  # the impulse the body applies (negative)

        step_dt = 1 / 250.0
        effective_dt = self.dt
        if self._dt_scale > 0:
            effective_dt *= (1 + np.random.uniform(-self._dt_scale, self._dt_scale))

        x = 0
        while x < effective_dt:
            too_much_force = False
            impulse_dir = np.array([0., 0.])

            self.player_body.each_arbiter(limit_force)
            if self._break_constraints_on_large_impulse:
                for i in range(len(self.bodies)):
                    if self.added_constraints[i] is not None:
                        total_cstr_impulse = self.added_constraints[i].impulse + sum(cstr.impulse for cstr in self.extra_constraints[i])
                        if total_cstr_impulse > 50000:
                            self.world.remove(self.added_constraints[i])
                            self.added_constraints[i] = None
                            for j in range(len(self.extra_constraints[i])):
                                self.world.remove(self.extra_constraints[i][j])
                            self.extra_constraints[i] = []

            if too_much_force:
                targ_vel = vel.copy()
                # if velocity and impulse in same half plane, subtract out
                unit_impulse = impulse_dir / np.linalg.norm(impulse_dir)
                # print(unit_impulse, unit_impulse.dot(vel))
                if unit_impulse.dot(vel) > 0:
                    # recoil by 50%
                    targ_vel -= 1.5 * unit_impulse.dot(vel) * unit_impulse

                vel[:] = vel * 0.3 + targ_vel * (1 - 0.3)  # smoothly & quickly slow down while too much force applied
            # else:
            #     vel[:] = action.action[0, :2]

            if self.keep_in_bounds:
                pos = np.asarray([self.player_body.position.x, self.player_body.position.y])
                bynd = np.maximum(pos - self.grid_size + 1, 0)  # upper bound
                bhnd = np.minimum(pos - 1, 0)  # lower bound
                dist_outside_grid_edges = bynd + bhnd
                dnorm = np.linalg.norm(dist_outside_grid_edges)
                if dnorm > 0:
                    # limit the velocity to prevent out of bounds
                    vel[:] = vel - max(vel.dot(dist_outside_grid_edges / dnorm), 0) * (dist_outside_grid_edges / dnorm)

            if self.do_wall_collisions:
                coll_points = [sl.shapes_collide(list(self.player_body.shapes)[0]).points for sl in self.static_lines]
                if any(len(cp) > 0 for cp in coll_points):
                    for i in range(len(self.static_lines)):
                        if len(coll_points[i]) > 0:
                            for j in range(len(coll_points[i])):
                                pt = coll_points[i][j]
                                collision_direction = pt.point_b - self.player_body.position
                                vel = vel - max(vel.dot(collision_direction), 0) * collision_direction / (np.linalg.norm(collision_direction) ** 2)

            self.player_body.velocity = vel.tolist()
            x += step_dt
            self.world.step(step_dt)

        # print(self.player_body.force)

        obs = self._get_obs()

        if not self.disable_images:
            if self.user_input is None:
                # flush the event queue if no ui provided but pygame is initialized
                pygame.event.get()

        self.step_render()

        if self.realtime:
            # realtime
            self.clock.tick(int(1. / self.dt))

        # else:
        #     raise NotImplementedError("Need to implement image rendering w/ no display...")
        done = np.array([False])
        self._t += 1
        if self._t >= self._horizon:
            done = np.array([True])

        if self._done_on_success and (self.is_success() or self._stop_counter > 0):
            self._stop_counter += 1
            # stop after success and + N-1 additional steps
            if self._stop_counter >= 2:  # needs one step to register the reward I think
                done = np.array([True])

        return obs, AttrDict(), done

    def _get_obs(self):
        block_pos = np.zeros((self.num_blocks, 2))
        block_vel = np.zeros((self.num_blocks, 2))
        block_force = np.zeros((self.num_blocks, 2))
        block_ang = np.zeros((self.num_blocks,))
        block_angvel = np.zeros((self.num_blocks,))
        block_torque = np.zeros((self.num_blocks,))
        block_mass = np.zeros((self.num_blocks,))
        block_color = np.zeros((self.num_blocks, 4))
        block_bbox = np.zeros((self.num_blocks, 4))  # left right top bottom
        block_contact = np.zeros((self.num_blocks,), dtype=bool)
        block_contact_normal = np.zeros((self.num_blocks, 2), dtype=bool)
        block_contact_points = np.zeros((self.num_blocks, 4), dtype=bool)  # x,y ego, x,y block

        # true force applied on each block
        active_constraints = np.array(
            [(cstr.impulse / self.dt if cstr is not None else 0.) for cstr in self.added_constraints])

        for i in range(self.num_blocks):
            info = list(self.bodies[i].shapes)[0].shapes_collide(list(self.player_body.shapes)[0])
            block_contact[i] = len(info.points) > 0
            block_contact_normal[i] = info.normal.x, info.normal.y
            if block_contact[i]:
                # first point
                block_contact_points[i] = info.points[0].point_b.x, info.points[0].point_b.y, info.points[0].point_a.x, info.points[0].point_a.y

        # print(block_contact)

        # dup
        grab_force = np.array([(cstr.impulse / self.dt if cstr is not None else 0.) for cstr in self.added_constraints])
        grab_dist = np.array([(cstr.distance if cstr is not None else 0.) for cstr in self.added_constraints])
        grab_binary = np.array([int(cstr is not None) for cstr in self.added_constraints])
        grab_vec = np.array(
            [((all[0].groove_b.x, all[0].groove_b.y) if len(all) > 0 else (0., 0.)) for all in self.extra_constraints])
        # active_constraints = np.array([(cstr.distance if cstr is not None else 0.) for cstr in self.added_constraints])

        for i in range(self.num_blocks):
            p = self.bodies[i].position
            v = self.bodies[i].velocity
            f = self.bodies[i].force
            block_ang[i] = self.bodies[i].angle
            block_angvel[i] = self.bodies[i].angular_velocity
            block_torque[i] = self.bodies[i].torque
            block_pos[i] = p.x, p.y
            block_vel[i] = v.x, v.y
            block_force[i] = f.x, f.y
            block_mass[i] = self.bodies[i].mass
            block_color[i] = list(self.bodies[i].shapes)[0].color
            if self._block_bbox:
                bbox = list(self.bodies[i].shapes)[0].bb
                block_bbox[i] = (bbox.left, bbox.right, bbox.top, bbox.bottom)

        # print(self.player_body.force)

        obs = AttrDict(
            position=np.array(self.player_body.position)[None],  # (1, 2)
            velocity=np.array(self.player_body.velocity)[None],  # (1, 2)
            angle=np.array([self.player_body.angle])[None],  # (1,)
            angular_velocity=np.array([self.player_body.angular_velocity])[None],  # (1,)
            force=np.array(self.player_body.force)[None],  # (1, 2)
            torque=np.array([self.player_body.torque])[None],  # (1,)
            block_positions=block_pos[None],
            block_sizes=self.all_block_sizes[None],
            block_velocities=block_vel[None],
            block_angles=block_ang[None],
            block_angular_velocities=block_angvel[None],
            block_forces=block_force[None],
            block_torques=block_torque[None],
            block_masses=block_mass[None],
            block_colors=block_color[None],
            block_contact=block_contact[None],
            block_contact_points=block_contact_points[None],
            block_contact_normal=block_contact_normal[None],
            maze=self.maze_np.copy()[None],
            active_constraints=active_constraints[None],
            grab_binary=grab_binary[None],
            grab_force=grab_force[None],
            grab_distance=grab_dist[None],  # distance along groove
            grab_vector=grab_vec[None],  # groove vector (x,y)
        )
        if self._block_bbox:
            obs['block_bounding_boxes'] = block_bbox[None]
            bbox = list(self.player_body.shapes)[0].bb
            obs['bounding_box'] = np.array([bbox.left, bbox.right, bbox.top, bbox.bottom])[None]

        if self.valid_goal_idxs is not None:
            obs['goal_position'] = np.array(self.goal_pos)[None]

        if not self.disable_images:
            obs.image = self.get_img()[None]  # (1, imgw, imgh, 3)
        obs = obs.leaf_filter_keys(
            self.env_spec.observation_names + self.env_spec.output_observation_names + self.env_spec.param_names + self.env_spec.final_names)
        return obs

    def reset(self, presets: AttrDict = AttrDict()):
        self.create_world(presets=presets)

        if self.render:
            self.reset_render()

        initialization_steps = get_with_default(presets, "initialization_steps", self.initialization_steps)
        assert initialization_steps > 0
        for t in range(initialization_steps):
            self.world.step(self.dt)

        self._t = 0
        self.extra_memory.clear()
        return self._get_obs(), AttrDict()

    # def user_input_reset(self, user_input: UserInput, reset_action_fn=None, presets: AttrDict = AttrDict()):
    #     self._user_input = user_input
    #     if reset_action_fn is not None:
    #         reset_action_fn()
    #     return self.reset(presets)

    def is_in_bounds(self):
        return 0 <= self.player_body.position.x <= self.grid_size[0] and \
               0 <= self.player_body.position.y <= self.grid_size[1]


    def get_default_teleop_model_forward_fn(self, user_input: UserInput):  # TODO add this to parents
        return self._teleop_fn(self, user_input)


def get_block2d_example_params(num_blocks=10, num_maze_cells=5):
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_C = 3

    BLOCK_SIZE = 30
    BLOCK_MASS = 10.
    NUM_MAZE_CELLS = num_maze_cells
    NUM_BLOCKS = num_blocks
    DT = 0.1

    GRID_SIZE = np.array([600, 600])

    nsld = [
        # param
        ("image", (IMG_HEIGHT, IMG_WIDTH, 3), (0, 255), np.uint8),
        ("position", (2,), (-np.inf, np.inf), np.float32),
        ("goal_position", (2,), (-np.inf, np.inf), np.float32),
        ("velocity", (2,), (-np.inf, np.inf), np.float32),
        ("bounding_box", (4,), (-np.inf, np.inf), np.float32),
        ("block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_bounding_boxes", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),

        ("maze", (NUM_MAZE_CELLS, NUM_MAZE_CELLS), (0, 32), np.uint8),

        ("action", (3,), ((-GRID_SIZE / 10.).tolist() + [0], (GRID_SIZE / 10.).tolist() + [500.]), np.float32),
    ]
    obs_names = ['image', 'position', 'velocity', 'block_positions', 'block_velocities', 'bounding_box', 'block_bounding_boxes']
    output_obs_names = []

    # for name in obs_names:
    #     output_obs_names.append("next_" + name)

    action_names = ['action']
    goal_names = []
    param_names = ['maze']
    final_names = []
    env_spec_params = AttrDict(
        names_shapes_limits_dtypes=nsld,
        output_observation_names=output_obs_names,
        observation_names=obs_names,
        action_names=action_names,
        goal_names=goal_names,
        param_names=param_names,
        final_names=final_names,
    )

    env_params = AttrDict(
        num_blocks=NUM_BLOCKS,
        dt=DT,
        grid_size=GRID_SIZE,
        image_size=np.array([IMG_HEIGHT, IMG_WIDTH]),
        block_size=BLOCK_SIZE,
        block_mass=BLOCK_MASS,
        num_maze_cells=NUM_MAZE_CELLS,
    )

    return env_params, env_spec_params


if __name__ == "__main__":

    env_params, env_spec_params = get_block2d_example_params()

    env_params.render = True
    env_params.realtime = True
    env_params.keep_in_bounds = True
    env_params.grab_action_binary = True
    env_params.initialization_steps = 5
    env_params.do_wall_collisions = False

    env_spec = ParamEnvSpec(env_spec_params)
    block = BlockEnv2D(env_params, env_spec)

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
        obs, goal, done = block.step(act_dict)

        # cv2.imshow("image_test", block.get_img())
        # cv2.waitKey(1)
