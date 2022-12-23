"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
import time

import numpy as np
# import pybullet
from oculus_reader import OculusReader

from sbrl.envs.interfaces import VRInterface
from sbrl.policies.policy import Policy
from sbrl.utils.control_utils import orientation_error
from sbrl.utils.geometry_utils import clip_ee_orientation_conical
from sbrl.utils.np_utils import clip_norm
from sbrl.utils.python_utils import AttrDict, get_with_default, get_or_instantiate_cls
from sbrl.utils.torch_utils import cat_any, to_numpy
from sbrl.utils.transform_utils import quat2mat, mat2quat, euler2mat, quat_difference, quat_multiply, \
    quat2euler_ext, quat2axisangle, axisangle2quat, fast_euler2quat_ext
from sbrl.utils.vr_utils import controller_off_message


class VRPoseTeleopPolicy(Policy):
    def _init_params_to_attrs(self, params):

        self.action_name = get_with_default(params, "action_name", "action")  # this will be the pose (euler)
        self.gripper_pos_name = get_with_default(params, "gripper_pos_name", "gripper_pos")
        self.gripper_tip_pos_name = get_with_default(params, "gripper_tip_pos_name", "gripper_tip_pos")


        self.use_click_state = get_with_default(params, "use_click_state", False)  # remaps B -> mode label

        self.use_gripper = get_with_default(params, "use_gripper", True)
        # default = 0->255
        # normalized = 0->1
        self.gripper_action_space = get_with_default(params, "gripper_action_space", "default")
        self._continuous_gripper = get_with_default(params, "continuous_gripper", True) # will use clipped delta gripper
        if self.gripper_action_space == "default":
            self._gripper_max = 250. # max = closed
        elif self.gripper_action_space == "normalized":
            self._gripper_max = 1.

        # parses obs -> pose 7d
        self.get_pose_from_obs_fn = get_with_default(params, "get_pose_from_obs_fn", lambda obs: cat_any(
            [(obs >> "ee_position").reshape(-1), fast_euler2quat_ext((obs >> "ee_orientation_eul").reshape(-1))],
            dim=-1))  # this will be the pose.
        self.get_gripper_from_obs_fn = get_with_default(params, "get_gripper_from_obs_fn",
                                                        lambda obs: (obs >> self.gripper_pos_name).reshape(-1))
        self.get_gripper_tip_pose_from_obs_fn = get_with_default(params, "get_gripper_tip_pose_from_obs_fn",
                                                                 lambda obs: cat_any(
                                                                     [(obs >> self.gripper_tip_pos_name).reshape(-1),
                                                                      fast_euler2quat_ext(
                                                                          (obs >> "ee_orientation_eul").reshape(-1))],
                                                                     dim=-1))
        # parses obs -> base pose 7d (usually doesn't change, so set this to return a constant)
        self.get_base_pose_from_obs_fn = get_with_default(params, "get_base_pose_from_obs_fn", lambda obs: np.array(
            [0, 0, 0, 0, 0, 0, 1]))  # this will be the pose.

        self.read_delta = get_with_default(params, "read_delta", True)  # read deltas from VR controller
        self.action_as_delta = get_with_default(params, "action_as_delta", False)  # action space is absolute.

        self.spatial_gain = get_with_default(params, "spatial_gain", 1.)
        self.pos_gain = get_with_default(params, "pos_gain", .1)
        self.rot_gain = get_with_default(params, "rot_gain", 1.)
        self.delta_pos_max = get_with_default(params, "delta_pos_max", 0.5 / 10)  # m, xyz, per step
        self.delta_rot_max = get_with_default(params, "delta_rot_max", np.pi/6 / 10)  # rad, euler angles
        self.delta_clip_norm = get_with_default(params, "delta_clip_norm", True)  # treats above as max_l2 instead of max_l1

        self._freeze_on_pause = get_with_default(params, "freeze_on_pause", True)
        self.sticky_gripper = get_with_default(params, "sticky_gripper", False)
        # self.sticky_gripper = get_with_default(params, "sticky_gripper", False)

        self.oculus_to_robot_mat_4d = get_with_default(params, "oculus_to_robot_mat_4d", np.asarray(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32))

        self.tip_safe_bounds = np.asarray(
            get_with_default(params, "tip_safe_bounds", self._env.get_safenet()))  # make sure this is implemented in your env.
        # conical clipping of orientation
        self._clip_ori_max = params << "clip_ori_max"  # if non None, determines the max allowable angle of the end effector from downward-z
        assert list(self.tip_safe_bounds.shape) == [2, 3], self.tip_safe_bounds.shape

        super(VRPoseTeleopPolicy, self)._init_params_to_attrs(params)

    def reset_policy(self, **kwargs):
        self.reset_origin = True
        self._target_robot_pose = None
        self._targ_gripper = 0
        self.initial_pose = None
        self._done = False

        self._step = 0  # TODO TESTING

    def _init_setup(self):
        super(VRPoseTeleopPolicy, self)._init_setup()

        # TODO TESTING
        self.reader = get_or_instantiate_cls(self._params, "reader", OculusReader,
                                             constructor=lambda cls, prms: cls(**prms.as_dict()))

        self.hand_origin = AttrDict.from_dict({'pos': None, 'quat': None})
        self.vr_origin = AttrDict.from_dict({'pos': None, 'quat': None})

        self.reset_origin = True
        self._done = False
        self._trig_pressed = False
        self._targ_gripper = 0.

        self.yaw = 0  # pybullet only

        assert issubclass(type(self._env), VRInterface), "Env must inherit from VR interface!"

    def warm_start(self, model, observation, goal):
        pass

    def _set_robot_orientation(self, obs):
        # For all of them Det == 1 => proper rotation matrices.
        # sets internal pose estimate.
        base_pose = self.get_base_pose_from_obs_fn(obs).reshape(-1)
        self.robot_to_global_rmat = quat2mat(base_pose[3:])[:3, :3]
        self.robot_to_global_mat_4d = np.eye(4)
        self.robot_to_global_mat_4d[:3, :3] = self.robot_to_global_rmat
        self.global_to_robot_mat_4d = np.linalg.inv(
            self.robot_to_global_mat_4d)
        self.global_to_robot_mat_rmat = self.global_to_robot_mat_4d[:3, :3]

    def _read_sensor(self, controller_id='r'):
        # Read Controller
        start_time = time.time()
        while True:
            poses, buttons = self.reader.get_transformations_and_buttons()
            if poses == {} and time.time() - start_time > 5:
                input(controller_off_message)
            if poses != {} and not buttons['RG']:
                self.reset_origin = True
                if buttons['A'] or not self._freeze_on_pause:
                    break  # this allows sim to run even when VR controller not active.
            if poses != {} and buttons['RG']:
                break

        # logger.debug(f"RIGHT: {mat2euler_ext(poses[controller_id][:3, :3])}")

        # Put Rotation Matrix In Robot Frame
        rot_mat = self.oculus_to_robot_mat_4d @ np.asarray(poses[controller_id])
        orientation_shift = np.array([0, 0, 0, 1])  
        # R.from_euler('z', -np.pi).as_quat()  # [0,0,0,1]  # (R.from_euler('z', -np.pi/2) * R.from_euler('x', -np.pi)).as_quat()
        vr_pos, vr_quat = rot_mat[:3, 3], quat_multiply(orientation_shift, mat2quat(rot_mat))
        vr_pos *= self.spatial_gain

        # Get Handle Orientation For Visualization
        vis_rot_mat = self.robot_to_global_mat_4d[:3, :3] @ rot_mat[:3, :3] @ euler2mat([-0.25 * np.pi, np.pi, 0])
        vis_quat = mat2quat(vis_rot_mat)

        return vr_pos, vr_quat, buttons, {'visual_quat': vis_quat}

    def _read_observation(self, obs):
        # Read Environment Observation
        self.robot_7d_pose = to_numpy(self.get_pose_from_obs_fn(obs).reshape(-1), check=True)
        if self.use_gripper:
            self.gripper = to_numpy(self.get_gripper_from_obs_fn(obs).reshape(-1), check=True)[0]
        else:
            self.gripper = 0
        self.gripper_tip_pose = to_numpy(self.get_gripper_tip_pose_from_obs_fn(obs).reshape(-1), check=True)
        return self.robot_7d_pose, self.gripper, self.gripper_tip_pose

    def get_action(self, model, observation, goal, **kwargs):
        self._set_robot_orientation(observation)

        # Read Sensor TODO TESTING
        vr_pos, vr_quat, buttons, oculus_info = self._read_sensor()

        # Read Observation
        robot_pose, curr_gripper, curr_gripper_tip_pose = self._read_observation(observation)

        if self.sticky_gripper:
            if buttons['rightTrig'][0] > 0.75 and not self._trig_pressed:  # mostly closed
                self._targ_gripper = gripper = 0. if curr_gripper.item() > self._gripper_max / 2. else self._gripper_max
                self._trig_pressed = True
            else:
                gripper = self._targ_gripper
                self._trig_pressed = False
            gripper = np.array([gripper], dtype=np.float32)
        else:
            gripper = self._gripper_max * np.array([buttons['rightTrig'][0]],
                                     dtype=np.float32)  # pressed = closed

        x_joy = buttons['rightJS'][0]

        # # TODO TESTING
        # vr_pos = np.zeros(3)
        # # vr_pos += 0.2 * (1 - np.cos(self._step * (2 * np.pi) / 40))
        # vr_quat = R.from_euler('x', 0 * np.pi / 4 * np.sin(self._step * (2 * np.pi) / 40)).as_quat()
        # gripper = np.array([0.])
        # x_joy = 0.
        # buttons = {'B': False, 'A': False}
        # oculus_info = {'visual_quat': np.array([0, 0, 0, 1])}
        # self._step += 1
        #
        # if self._step % 40 == 0:
        #     phase = (self._step // 40) % 5
        #     base = R.from_quat(np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2, 0, 0]))
        #     self.hand_origin['quat'] = (base * R.from_euler('z', -np.pi / 2 * phase / 4)).as_quat()

        if self._continuous_gripper:
            delta_gripper = gripper - curr_gripper
            delta_gripper = np.sign(delta_gripper) * min(abs(delta_gripper), self._gripper_max * 15. / 255.)  # max delta is 15 / 255 %
            gripper = curr_gripper + delta_gripper
            gripper = np.maximum(np.minimum(self._gripper_max, gripper), 0.05 * self._gripper_max)

        # Set Origins When Button Released Or Episode Started
        if self.reset_origin:
            if self.initial_pose is None:
                self.initial_pose = robot_pose.copy()
            self.hand_origin = AttrDict(pos=robot_pose[:3], quat=robot_pose[3:])
            self.vr_origin = AttrDict(pos=vr_pos, quat=vr_quat)
            self.reset_origin = False

        curr_click_label = 0.
        if buttons['B']:
            if self.use_click_state:
                curr_click_label = 1.
            else:
                # make sure this is implemented
                self._env.change_view(delta_yaw=-self.yaw)
                self.yaw = 0

        if buttons['A']:
            # RESET
            self._done = True

        if abs(x_joy) > 0.03:
            delta = x_joy * 6 * np.pi * self._env.dt  # something is off here
            self.yaw += delta  # tracking internal
            # make sure this is implemented
            self._env.change_view(delta_yaw=delta)

        # if abs(x_joy) > 0.03:
        #     delta = x_joy * 6 * np.pi * self._env.dt  # something is off here
        #     self.yaw += delta  # tracking internal
        #     # make sure this is implemented
        #     self._env.change_view(delta_yaw=delta)

        # Calculate Positional Action
        hand_pos_offset = robot_pose[:3] - self.hand_origin['pos']
        target_pos_offset = vr_pos - self.vr_origin['pos']
        pos_action = target_pos_offset - hand_pos_offset

        # safenet clipping.
        new_pos = curr_gripper_tip_pose[:3] + pos_action  # target tip pose
        new_pos = np.clip(new_pos, self.tip_safe_bounds[0], self.tip_safe_bounds[1])
        pos_action = new_pos - curr_gripper_tip_pose[:3]

        # # Calculate Euler Action
        # if False:
        target_quat_offset = quat_difference(vr_quat, self.vr_origin['quat'])
        target_quat_offset = axisangle2quat(quat2axisangle(target_quat_offset) * self.rot_gain)

        desired_quat = quat_multiply(target_quat_offset, self.hand_origin['quat'])


        desired_euler = quat2euler_ext(desired_quat)
        if self._clip_ori_max is not None:
            desired_euler = clip_ee_orientation_conical(desired_euler, ee_axis=np.array([0, 0, 1.]),
                                                        world_axis=np.array([0, 0, -1.]), max_theta=self._clip_ori_max)

        scale_pos_action = pos_action * self.pos_gain

        if self.action_as_delta:
            delta_euler = orientation_error(quat2mat(desired_quat), quat2mat(robot_pose[3:]))
            if self.delta_clip_norm:
                #l2 clipping (as vector)
                scale_pos_action = clip_norm(scale_pos_action, self.delta_pos_max)
                delta_euler = clip_norm(delta_euler, self.delta_rot_max)
            else:
                #l1 clipping (individual)
                scale_pos_action = scale_pos_action.clip(-self.delta_pos_max, self.delta_pos_max)
                delta_euler = delta_euler.clip(-self.delta_rot_max, self.delta_rot_max)
            command = np.concatenate([scale_pos_action, delta_euler])
        else:
            command = np.concatenate([robot_pose[:3] + scale_pos_action, desired_euler])

        # postprocess
        return self._postproc_fn(model, observation, goal, AttrDict.from_dict({
            self.action_name: np.concatenate([command, gripper]),
            'target': {
                'ee_position': robot_pose[:3] + pos_action,
                'ee_orientation_eul': quat2euler_ext(desired_quat),
                self.gripper_pos_name: gripper,
            },
            'vr_pos': self.robot_to_global_rmat @ scale_pos_action,
            'vr_quat': oculus_info['visual_quat'],
            'policy_type': np.array([254]),  # VR
            'policy_name': np.array(["vr_teleop"]),  # VR
            'policy_switch': np.array([False]),  # VR
            'click_state': np.array([curr_click_label])
        }).leaf_apply(lambda arr: arr[None]))

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done
