#!/usr/bin/env python3

## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.sth_sth import BulletSth
from sbrl.envs.bullet_envs.sth_sth import drawProcess
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.python_utils import AttrDict as d, get_with_default


class BulletSth46(BulletSth):
    def _init_params_to_attrs(self, params: d):
        super(BulletSth46, self)._init_params_to_attrs(params)
        assert self.task_id == 46
        # self.obj_friction_ceof = get_with_default(params, "object_friction_coef", 100.0)
        # self.obj_linear_damping = get_with_default(params, "object_linear_damping", 1.0)
        # self.obj_angular_damping = get_with_default(params, "object_angular_damping", 1.0)
        # self.obj_contact_stiffness = get_with_default(params, "object_contact_stiffness", 1.0)
        # self.obj_contact_damping = get_with_default(params, "object_contact_damping", 1.0)

        self.drawer_mass = get_with_default(params, "drawer_mass", 0.5, map_fn=float)

        # randomizes the drawer position
        self.randomize_object_start_location = get_with_default(params, "randomize_object_start_location", True)
        self.randomize_drawer_joint = get_with_default(params, "randomize_drawer_joint", True)
        # this just means, start inside the drawer, or close to it
        self.start_in_grasp = get_with_default(params, "start_in_grasp", True)

    def _init_setup(self):
        super(BulletSth46, self)._init_setup()

    def _load_asset_objects(self):
        self.object = self.init_obj(d(object_cls='drawer'))
        p.changeVisualShape(self.object.id, -1, rgbaColor=[1., 0., 0., 1], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, -1, rgbaColor=[1., 0., 0., 1], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, 0, rgbaColor=[0, 0, 1, 1.0], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, 1, rgbaColor=[0, 0, 1, 1.0], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, 2, rgbaColor=[0, 0, 1, 1.0], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, 3, rgbaColor=[0, 0, 1, 1.0], physicsClientId=self.id)
        p.changeVisualShape(self.object.id, 4, rgbaColor=[0, 0, 1, 1.0], physicsClientId=self.id)

        # opening drawer
        p.resetJointState(self.object.id, 0, 0.05)

        # registering
        self.objects.append(self.object)

    def _load_dynamics(self):
        super(BulletSth46, self)._load_dynamics()

        self.robot.set_gripper_max_force(10.0)
        arm_damp = [0.1, 0.05, 0.04, 0.01, 0.01, 0.01, 0.01]
        self.robot.set_joint_damping(arm_damp + [0.01] * 6)
        # make the drawer heavy
        p.changeDynamics(self.object.id, 0, self.drawer_mass, physicsClientId=self.id)

        # p.changeDynamics(self.object.id, -1,
        #                  lateralFriction=self.obj_friction_ceof,
        #                  rollingFriction=self.obj_friction_ceof,
        #                  spinningFriction=self.obj_friction_ceof,
        #                  linearDamping=self.obj_linear_damping,
        #                  angularDamping=self.obj_angular_damping,
        #                  contactStiffness=self.obj_contact_stiffness,
        #                  contactDamping=self.obj_contact_damping, physicsClientId=self.id)

    def reset_assets(self, presets: d = d()):
        super(BulletSth46, self).reset_assets(presets)

        # base_frame will be set again for our object
        pos = np.asarray([0.38, 0.0, 0.35])

        orn = p.getQuaternionFromEuler([np.pi / 2.0, 0, np.pi])
        rotation_degree = 0.
        if self.randomize_object_start_location:
            robot_base_frame = self.robot.get_link_frame(0)
            pos[:2] += np.random.uniform(-.1, .1, size=(2,))
            r = R.from_quat(orn)
            h_trans = np.zeros((4, 4))
            h_trans[:3, :3] = r.as_dcm()

            rotation_degree = np.random.uniform(-0.5, 0.5)
            add_rot = R.from_rotvec(rotation_degree * np.array([0, 0, 1]))
            add_h_trans = np.zeros((4, 4))
            add_h_trans[:3, :3] = add_rot.as_dcm()
            new_h_trans = add_h_trans.dot(h_trans)
            orn = R.from_dcm(new_h_trans[:3, :3]).as_quat()

        init_drawer_joint = 0.05
        if self.randomize_drawer_joint:
            init_drawer_joint = np.random.uniform(0.015, 0.05)

        self.set_obj_pose(self.object, pos, orn,
                          assign_base_frame=True)

        p.resetJointState(self.object.id, 0, init_drawer_joint)

        self.object.rotation_angle = rotation_degree

        if self.start_in_grasp:
            self.init_grasp(self.object)

    def init_grasp(self, object: d):
        self.robot.gripper_control(220)

        above = np.asarray(object.position) + np.asarray([0., 0.05, 0.23])
        down_orn = R.from_euler("xyz", [np.pi, 0, 0]).inv()
        above_frame = CoordinateFrame(world_frame_3D, down_orn, above)
        rp = self.robot.rp
        for i in range(2):  # N chances for IK solver
            q_desired = self.robot.compute_frame_ik(self.robotEndEffectorIndex, above_frame, rest_pose=rp)
            # keep robot at q_desired
            self.robot.set_joint_values(q_desired, 220)
            self.robot.joint_position_control(self.robot.controlled_arm_joints, q_desired, gripper_pos=220)
            rp = q_desired

        self.set_initial_joint_positions()  # sets to current location

    def get_success(self, seg=None):
        # open drawer
        drawer_joint_info = p.getJointState(self.object.id, 0, physicsClientId=self.id)
        if drawer_joint_info[0] > 0.1:
            return True
        else:
            return False

    def _get_object_obs(self, object: d):
        # make sure this the drawer... lol
        obs = super(BulletSth46, self)._get_object_obs(object)
        djs = p.getJointState(object.id, 0, physicsClientId=self.id)
        # extra fields
        obs.drawer_joint_position = np.asarray([djs[0]])[None]
        obs.drawer_joint_velocity = np.asarray([djs[1]])[None]
        # technically you can retrieve this from obs.orientation, but nah
        obs.rotation_angle = np.asarray([object.rotation_angle])[None]
        return obs


# teleop code as a test
if __name__ == '__main__':

    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 5000
    params = d()
    params.task_id = 46
    params.render = True
    params.compute_images = False
    params.debug = False
    params.skip_n_frames_every_step = 1
    params.time_step = 0.05  # 20Hz
    params.start_in_grasp = True
    env = BulletSth46(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
    ], observation_names=['wrist_ft'])))  # gym.make('BiteTransferPanda-v0')

    presets = d()

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

    grip_state = 220 / 255
    done = False

    i = 0

    queue = mp.Queue()
    proc = Process(target=drawProcess, args=(queue,), daemon=True)
    proc.start()

    while True:
        i += 1
        keys = p.getKeyboardEvents(physicsClientId=env.id)
        if done or i >= max_steps or ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            logger.debug("Resetting (after %d iters)! done = %s" % (i, done))
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
