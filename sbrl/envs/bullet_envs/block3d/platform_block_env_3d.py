#!/usr/bin/env python3
import argparse
## plotting forces
import multiprocessing as mp
import sys

import numpy as np
import pybullet as p

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D, get_block3d_example_spec_params, \
    get_block3d_example_params
from sbrl.envs.bullet_envs.utils_env import RobotControllerMode
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers.pid_controller import ControlType
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.python_utils import AttrDict as d, get_with_default

RCM = RobotControllerMode
CT = ControlType

DEFAULT_PLAT_HEIGHT = 0.16/3

class PlatformBlockEnv3D(BlockEnv3D):
    def _init_params_to_attrs(self, params: d):
        # change object start bounds
        self._platform_height = get_with_default(params, "platform_height", DEFAULT_PLAT_HEIGHT)
        self._platform_extent = get_with_default(params, "platform_extent", 0.08)
        self._init_obj_on_platform = get_with_default(params, "init_obj_on_platform", False)  # TODO implement
        params.object_start_bounds = get_with_default(params, "object_start_bounds", {
            'block': (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
            'mug': (np.array([-0.45, -0.45]), np.array([0.45, 0.45]))
            # % [-1, 1] is the range for each, object initializes inside this percentage of w/2
        })

        assert not self._init_obj_on_platform, "not implemented yet"

        super(PlatformBlockEnv3D, self)._init_params_to_attrs(params)

    def load_surfaces(self):
        super(PlatformBlockEnv3D, self).load_surfaces()

        h = self._platform_height / 2  # third of the cabinet height TODO param
        ext = self._platform_extent / 2  # protrudes this much into the table TODO param

        # four edge platforms for lifting
        w1 = self._create_cabinet_fn(halfExtents=[self.surface_bounds[0] / 2, ext, h],
                                     location=self.surface_center + np.array([0, self.surface_bounds[1] / 2 - ext, h / 2]))
        w2 = self._create_cabinet_fn(halfExtents=[self.surface_bounds[0] / 2, ext, h],
                                     location=self.surface_center + np.array([0, -self.surface_bounds[1] / 2 + ext, h / 2]))
        w3 = self._create_cabinet_fn(halfExtents=[ext, self.surface_bounds[1] / 2, h],
                                     location=self.surface_center + np.array([self.surface_bounds[0] / 2 - ext, 0, h / 2]))
        w4 = self._create_cabinet_fn(halfExtents=[ext, self.surface_bounds[1] / 2, h],
                                     location=self.surface_center + np.array([-self.surface_bounds[0] / 2 + ext, 0, h / 2]))

        _, aabbmax = p.getAABB(w1, -1, physicsClientId=self.id)
        self.platform_z = aabbmax[2]

        self.cabinet_obj_ids.extend([w1, w2, w3, w4])

    def get_nearest_platform(self, obj_obs, return_all=True, margin=0):
        # distances to each platform
        obj_pos = (obj_obs >> "position")[0]
        obj_aabb = (obj_obs >> "aabb")[0]
        obj_height = obj_aabb[5] - obj_aabb[2]
        closest_points = np.array([
            [self.surface_center[0] + self.surface_bounds[0] / 2 - self._platform_extent/2 - margin, obj_pos[1], obj_pos[2]],
            [self.surface_center[0] - self.surface_bounds[0] / 2 + self._platform_extent/2 + margin, obj_pos[1], obj_pos[2]],
            [obj_pos[0], self.surface_center[1] + self.surface_bounds[1] / 2 - self._platform_extent/2 - margin, obj_pos[2]],
            [obj_pos[0], self.surface_center[1] - self.surface_bounds[1] / 2 + self._platform_extent/2 + margin, obj_pos[2]],
        ])
        closest_dist = np.abs(closest_points - obj_pos).sum(-1)

        close_idx = np.argmin(closest_dist)
        cp = closest_points[close_idx]

        # z will be such that object is flush with platform
        cp[2] = obj_height/2 + self.platform_z

        if return_all:
            # close_pt (3,), idx, distances (4,), points (4,3)
            return cp, close_idx, closest_dist, closest_points
        return cp



# teleop code as a test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mug', action='store_true')
    args = parser.parse_args()

    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    params = get_block3d_example_params(use_mug=args.use_mug, render=True)

    params.debug_cam_dist = 0.25
    params.debug_cam_p = -45
    params.debug_cam_y = 0
    params.debug_cam_target_pos = [0.4, 0, 0.45]

    env_spec_params = get_block3d_example_spec_params()
    env = PlatformBlockEnv3D(params, ParamEnvSpec(env_spec_params))

    presets = d()
    max_steps = 5000
    # presets = d(objects=d(position=np.array([0.4, 0.1, 0.35])[None], orientation_eul=np.array([0., 0., 0.])[None], size=np.array([0.032, 0.043, 0.03])[None]))

    keys_actions = {
        p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]),
        p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]),
        p.B3G_UP_ARROW: np.array([0, 0.01, 0]),
        p.B3G_DOWN_ARROW: np.array([0, -0.01, 0]),
        ord('i'): np.array([0, 0, 0.01]),
        ord('k'): np.array([0, 0, -0.01])
    }

    keys_orient_actions = {  # in rpt space
        ord('='): np.array([0, 0, 0.01]),
        ord('-'): np.array([0, 0, -0.01]),
        ord('['): np.array([0, 0.02, 0]),
        ord(']'): np.array([0, -0.02, 0]),
        ord(';'): np.array([0.01, 0, 0]),
        ord('\''): np.array([-0.01, 0, 0]),
    }

    observation, _ = env.reset(presets)
    # Get the position and orientation of the end effector
    target_position, target_orientation = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
    target_rpt_orientation = convert_quat_to_rpt(target_orientation)  # rpt rotation about default

    grip_state = 0
    done = False

    i = 0

    while True:
        i += 1
        keys = p.getKeyboardEvents(physicsClientId=env.id)
        if done or i >= max_steps or ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            logger.debug(
                "Resetting (after %d iters)! done = %s" % (i, done))
            i = 0
            observation, _ = env.reset(presets)
            target_position, target_orientation = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
            target_rpt_orientation = convert_quat_to_rpt(target_orientation)

        for key, action in keys_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                target_position += action

        for key, action in keys_orient_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                # orientation = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(orientation) + action) % (2 * np.pi))
                target_rpt_orientation = (target_rpt_orientation + action) % (2 * np.pi)

        # open w/ >
        if ord('.') in keys and keys[ord('.')] & p.KEY_IS_DOWN:
            grip_state = max(grip_state - 0.05, 0)
        # close w/ <
        if ord(',') in keys and keys[ord(',')] & p.KEY_IS_DOWN:
            grip_state = min(grip_state + 0.05, 1.)

        curr_pos, curr_orn = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
        curr_rpt = convert_quat_to_rpt(curr_orn)

        # decaying target position
        target_position = target_position * 0.9 + curr_pos * 0.1
        # target_rpt_orientation = target_rpt_orientation * 0.9 + curr_rpt * 0.1
        target_orientation, target_orientation_eul = convert_rpt(*target_rpt_orientation)

        # target end effector state
        # targ_frame = CoordinateFrame(world_frame_3D, R.from_quat(orientation).inv(), np.asarray(position))
        act = np.concatenate([np.asarray(target_position), np.asarray(target_orientation_eul), [grip_state * 255.]])

        observation, _, done = env.step(act)
