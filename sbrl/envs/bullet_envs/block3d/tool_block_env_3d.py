#!/usr/bin/env python3

## plotting forces
import multiprocessing as mp
import sys

import numpy as np
import pybullet as p

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D, get_block3d_example_spec_params
from sbrl.envs.bullet_envs.utils_env import RobotControllerMode
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers.pid_controller import ControlType
from sbrl.policies.controllers.robot_config import os_torque_control_panda_cfg
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.pybullet_utils import Hook
from sbrl.utils.python_utils import AttrDict as d, get_with_default
from sbrl.utils.transform_utils import quat2euler_ext

RCM = RobotControllerMode
CT = ControlType


class ToolBlockEnv3D(BlockEnv3D):
    def _init_params_to_attrs(self, params: d):
        # hook default
        self._tool_shape_dim = 3
        self._init_tool_shape = get_with_default(params, "initial_tool_shape", np.array([0.3, 0.1, 0.02]))
        self._random_tool_shape = get_with_default(params, "random_tool_shape", False)

        # if random
        self._tool_shape_bounds = get_with_default(params, "tool_shape_bounds", np.array([[0.3, 0.1, 0.02],
                                                                                          [0.5, 0.2, 0.03]]),
                                                   map_fn=np.asarray)

        super(ToolBlockEnv3D, self)._init_params_to_attrs(params)

    def _load_asset_objects(self, presets: d = d()):
        super(ToolBlockEnv3D, self)._load_asset_objects(presets)

        default_shape = np.random.uniform(self._tool_shape_bounds[0], self._tool_shape_bounds[1]) \
            if self._random_tool_shape else self._init_tool_shape.copy()

        self.hook_width, self.hook_length1, self.hook_length2 = self.tool_shape = \
            get_with_default(presets, "tool_shape", default_shape).reshape(3)

        self.tool = Hook(self.hook_width, self.hook_length1, self.hook_length2)
        self.tool.load(clientId=self.id)

    def reset_assets(self, presets: d = d()):
        super(ToolBlockEnv3D, self).reset_assets(presets)

        # TODO set position to be random
        self.tool.set_position(self.surface_center)

    def cleanup(self):
        super(ToolBlockEnv3D, self).cleanup()
        self.tool.clean(clientId=self.id)

    def _get_obs(self, **kwargs):
        obs = super(ToolBlockEnv3D, self)._get_obs(**kwargs)

        obs.tool.shape = self.tool_shape.copy()[None]
        obs.tool.position = np.asarray(self.tool.get_position())[None]
        obs.tool.orientation = np.asarray(self.tool.get_orientation())[None]
        obs.tool.orientation_eul = quat2euler_ext(obs.tool.orientation[0])[None]

        return obs


# teleop code as a test
if __name__ == '__main__':

    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 5000
    params = d()
    params.num_blocks = NB = 1
    params.render = True
    params.compute_images = False
    params.debug = False
    params.object_spec = ['mug']
    params.object_rotation_bounds = {'block': (-np.pi/4, np.pi/4), 'mug': (-np.pi, np.pi)}
    params.skip_n_frames_every_step = 5
    params.time_step = 0.02  # 20Hz
    params.combine(os_torque_control_panda_cfg)

    params.debug_cam_dist = 0.25
    params.debug_cam_p = -45
    params.debug_cam_y = 0
    params.debug_cam_target_pos = [0.4, 0, 0.45]

    env_spec_params = get_block3d_example_spec_params()
    env = ToolBlockEnv3D(params, ParamEnvSpec(env_spec_params))

    presets = d()
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

    # shared_ft = shm.empty((6,), dtype=np.float32)
    # shared_ft[:] = 0

    queue = mp.Queue()

    # proc = Process(target=drawProcess, args=(queue,), daemon=True)
    # proc.start()

    # vel = np.array([0., -0.01, 0.])
    # for i in range(30):
    #     observation, _, done = env.step(np.concatenate([target_position + i * vel, target_orientation, np.array([0])]))

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
        queue.put(observation.wrist_ft[0])
        # queue.put(np.concatenate([observation.contact_force[0], np.zeros(3)]))
        # print(observation['joint_positions'])
