"""
Contains the robot, with robot model to query things like IK, ee position, orientation, joint angles, etc.

Also implements low level controllers:
- joint_vel/pos/torque_control
- os_vel/pos/torque_control

"""
import os
from collections import OrderedDict
from numbers import Number
from typing import Union, Tuple, List

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.experiments import logger
# sys.path.append('./')
# sys.path.insert(0,"../rllib/a3c")
# from sbrl.policies.controllers.controller import Controller
from sbrl.policies.controllers.pid_controller import ControlType, PIDController
from sbrl.utils import control_utils
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import get_with_default, get_required, AttrDict as d
from sbrl.utils.transform_utils import mat2euler


class BulletRobot(object):
    def __init__(self, id, params):
        self.id = id
        self.params = params

        self.resources_dir = get_required(params, "resources_dir")

        self.gripper_max_force = get_with_default(params, "gripper_max_force", None)
        self.gripper_max_delta = get_with_default(params, "gripper_max_delta", 255.)  # can open whole gripper per step (no limit).
        self.arm_max_force = get_with_default(params, "armMaxForce", None)
        self.end_effector_index = get_with_default(params, "endEffectorIndex", 7)  # TODO 8?
        self.start_pos = get_with_default(params, "start_pos", np.asarray([0.4, 0.5, 0.]), map_fn=np.asarray)

        self._collision_detection_links = get_with_default(params, "collision_detection_links", [2, 3, 4, 5], map_fn=list)
        self._apply_ft_transform = get_with_default(params, "apply_ft_transform", False)
        if self._apply_ft_transform:
            logger.warn("Robot will return transformed f/t readings!")
        # simple mass gravity force compensation (force only, TODO torque)
        self.wrist_ft_compensate = get_with_default(params, "compensateWristFT", True)

        # lower limits for null space
        self.ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, -0.0001, -0.0001, -0.0001, 0.0, 0.0,
                   -3.14, -3.14, 0.0, 0.0, 0.0, 0.0, -0.0001, -0.0001]
        # upper limits for null space
        self.ul = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0001, 0.0001, 0.0001, 0.81, 0.81, 3.14, 3.14,
                   0.8757, 0.8757, -0.8, -0.8, 0.0001, 0.0001]

        # joint ranges for null space
        self.jr = [(u - l) for (u, l) in zip(self.ul, self.ll)]

        # restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp = get_with_default(params, "reset_q", [-np.pi / 2, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, 0])

        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]


        self.num_controlled_arm_joints = 7
        self.controlled_arm_joints = [0, 1, 2, 3, 4, 5, 6]

        self.num_controlled_gripper_joints = 6
        self.controlled_gripper_joints = [10, 12, 14, 16, 18, 19]

        self.dof_joints = self.controlled_arm_joints + self.controlled_gripper_joints
        self.num_dof_joints = len(self.dof_joints)

        self.gripper_left_tip_index = 13
        self.gripper_right_tip_index = 17

        self.wrist_joint_index = 8  # panda_robotiq_coupling, for example
        self.finger_joint_index_left = 13  # robotiq_2f_85_left_inner_finger_pad_joint
        self.finger_joint_index_right = 17  # robotiq_2f_85_right_inner_finger_pad_joint

        self.urdf_dir = os.path.join(self.resources_dir, 'urdf')
        self._use_infinite_joint7 = get_with_default(params, "use_infinite_joint7", False)

        urdf_file = get_with_default(params, "urdf_file", "franka_panda/panda_robotiq.urdf")

        if self._use_infinite_joint7:
            # infinite
            model_path = os.path.join(self.urdf_dir, "franka_panda/panda_robotiq_free.urdf")
        else:
            model_path = os.path.join(self.urdf_dir, urdf_file)

        # model_path = os.path.join(self.urdf_dir,"panda_robotiq.urdf")
        logger.debug("Robot URDF Path: %s" % model_path)

        self.robot_id = p.loadURDF(model_path, [0, 0, 0], useFixedBase=True,
                                   flags=p.URDF_USE_SELF_COLLISION and p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
                                   physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.robot_id, self.start_pos, [0, 0, 0, 1])

        # f/t sensors
        # print(p.getJointInfo(self.robot_id, self.wrist_joint_index)[1],
        #       p.getJointInfo(self.robot_id, self.finger_joint_index_left)[1],
        #       p.getJointInfo(self.robot_id, self.finger_joint_index_right)[1],
        #       )

        # TODO parametrize
        p.enableJointForceTorqueSensor(self.robot_id, self.wrist_joint_index,
                                       enableSensor=1)  # 'getJointState' returns external f/t
        self.wrist_joint_impulse = np.zeros(6)
        self.wrist_joint_ft = np.zeros(6)
        p.enableJointForceTorqueSensor(self.robot_id, self.finger_joint_index_left,
                                       enableSensor=1)  # 'getJointState' returns external f/t
        self.finger_joint_left_impulse = np.zeros(6)
        self.finger_joint_left_ft = np.zeros(6)
        p.enableJointForceTorqueSensor(self.robot_id, self.finger_joint_index_right,
                                       enableSensor=1)  # 'getJointState' returns external f/t
        self.finger_joint_right_impulse = np.zeros(6)
        self.finger_joint_right_ft = np.zeros(6)

        self.target_velocities = [0] * self.num_controlled_arm_joints
        self.target_gripper_velocities = [0] * len(self.controlled_gripper_joints)
        self.position_gains = [0.03] * self.num_controlled_arm_joints
        self.velocity_gains = [1] * self.num_controlled_arm_joints

        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.id)

        self.gripper_ll = []
        self.gripper_ul = []

        self.joint_infos = []
        for j_idx in range(self.num_joints):
            j_info = p.getJointInfo(self.robot_id, j_idx, physicsClientId=self.id)
            #     print(self.p.getJointInfo(self.robotId,jointIndex))
            if j_idx in self.controlled_gripper_joints:
                self.gripper_ll.append(j_info[8])
                self.gripper_ul.append(j_info[9])

            self.joint_infos.append(j_info)

        # parse things from joint infos
        self.max_joint_forces = [ji[10] for ji in self.joint_infos]
        self.max_joint_velocities = [ji[11] for ji in self.joint_infos]
        self.joint_ll = [ji[8] for ji in self.joint_infos]
        self.joint_ul = [ji[9] for ji in self.joint_infos]

        if self.arm_max_force is not None:
            self.arm_max_force_list = [self.arm_max_force] * len(self.controlled_arm_joints)
        else:
            self.arm_max_force_list = [self.max_joint_forces[i] for i in self.controlled_arm_joints]

        if self.gripper_max_force is not None:
            self.gripper_max_force_list = [self.gripper_max_force] * len(self.controlled_gripper_joints)
        else:
            self.gripper_max_force_list = [self.max_joint_forces[i] for i in self.controlled_gripper_joints]

        all_f = np.asarray(self.arm_max_force_list + self.gripper_max_force_list)
        idxs_where = (all_f < 1e-15).nonzero()[0]
        if len(idxs_where) > 0:
            logger.warn("Zero max force for DOF joint idxs %s" % np.asarray(self.dof_joints)[idxs_where])

        self.set_joint_damping(self.jd)
        self.last_joint_velocities = np.zeros(len(self.dof_joints))
        self.joint_accelerations = np.zeros(len(self.dof_joints))
        self.joint_gravity_torques = np.zeros(len(self.dof_joints))
        self.joint_coriolis_torques = np.zeros(len(self.dof_joints))
        self.wrist_linear_jacobian = np.zeros((3, len(self.dof_joints)))
        self.wrist_angular_jacobian = np.zeros((3, len(self.dof_joints)))

        self.last_applied_torques = np.zeros(self.num_joints)

        self.controllers = d()
        # True = torque control, False = velocity controller (default)
        # self._torque_control_enabled = np.array([False] * self.num_joints)

    def reset(self, joints=None, ee_frame=None):

        # reset the base pos
        self.set_robot_base((self.start_pos, np.asarray([0, 0, 0, 1])))

        ####### Set Dynamic Parameters for the gripper pad######
        if joints is None and ee_frame is None:
            joints = self.rp
        elif ee_frame is not None:
            self.set_joint_values(self.rp[:self.num_controlled_arm_joints], 0.)
            self.joint_position_control(self.controlled_arm_joints, self.rp, gripper_pos=0)
            joints = self.compute_frame_ik(self.end_effector_index, ee_frame, rest_pose=self.rp).tolist()
            joints.extend(self.rp[self.num_controlled_arm_joints:])
        else:
            assert len(joints) == self.num_dof_joints

        friction_ceof = 100.0  # TODO
        p.changeDynamics(self.robot_id, self.gripper_left_tip_index, lateralFriction=friction_ceof,
                         rollingFriction=friction_ceof,
                         spinningFriction=friction_ceof,
                         physicsClientId=self.id)

        p.changeDynamics(self.robot_id, self.gripper_right_tip_index, lateralFriction=friction_ceof,
                         rollingFriction=friction_ceof,
                         spinningFriction=friction_ceof,
                         physicsClientId=self.id)

        # reset the joint values
        self.set_joint_values(joints[:self.num_controlled_arm_joints], 0.)
        # fill the motor control constraints to keep it there (for stepSim later)
        # print(joints)
        self.joint_position_control(self.controlled_arm_joints, joints[:self.num_controlled_arm_joints], gripper_pos=0)
        # print(self.get_joint_values()[self.controlled_arm_joints])

        self.wrist_joint_impulse[:] = 0
        self.finger_joint_left_impulse[:] = 0
        self.finger_joint_right_impulse[:] = 0
        self.wrist_joint_ft[:] = 0
        self.finger_joint_left_ft[:] = 0
        self.finger_joint_right_ft[:] = 0

        # reset the state terms (finite difference)
        self.last_joint_velocities[:] = 0
        self.joint_accelerations[:] = 0
        self.joint_gravity_torques[:] = 0
        self.joint_coriolis_torques[:] = 0
        self.wrist_linear_jacobian[:] = 0

        self.last_applied_torques[:] = 0

        self.wrist_joint_frame = None
        self.ee_link_com_frame = None  # ee link
        self.ee_p1_link_com_frame = None  # link after ee
        self.wrist_hanging_mass = None  # for rough force comp TODO do this properly
        self.counter = 0

        self.setpoint_gripper_angle = None  # for gripper smoothing

        # reset all controllers
        self.controllers.leaf_call(lambda controller: controller.reset())

    def update_state_in_between_step(self, time_step):
        # this is a place to define accumulators when simulator is stepped but control hasn't changed
        self.wrist_joint_impulse += self._get_ft(world_frame_3D, self.wrist_joint_index) * time_step
        self.finger_joint_right_impulse += self._get_ft(world_frame_3D, self.finger_joint_index_right) * time_step
        self.finger_joint_left_impulse += self._get_ft(world_frame_3D, self.finger_joint_index_left) * time_step

        # # also during torque control, you have to re-apply the last cached torques
        # if not np.allclose(self.last_applied_torques, 0):
        #     print("btwn", self.counter, self.last_applied_torques[self.dof_joints])
        #     self.joint_torque_control(self.dof_joints, self.last_applied_torques[self.dof_joints], save=False)

    def update_state_after_step(self, dt):
        # print("after", self.counter, self.last_applied_torques[self.dof_joints])
        # this is a place to define actions that you do before controllers will be called again
        # e.g. turn accumulators into states for this "dt" where dt > time_step
        # e.g., [update_btwn, update_btwn, update_btwn, update_after] * ...

        # e.g. finite diff joint accelerations
        new_joint_velocities = self.get_joint_values(velocities=True)[self.dof_joints]
        fd_acc = (new_joint_velocities - self.last_joint_velocities) / dt
        self.last_joint_velocities[:] = new_joint_velocities
        self.joint_accelerations[:] = fd_acc

        jvals = self.get_joint_values()
        # TODO keep track of velocities and accelerations
        dof_jvals = jvals[self.dof_joints]
        dof_jvals_vel = self.last_joint_velocities
        # print(len(jvals), len(jvals_vel), len(self.joint_accelerations))

        # update force torque sensors
        self.wrist_joint_ft[:] = self.wrist_joint_impulse / dt
        self.finger_joint_right_ft[:] = self.finger_joint_right_impulse / dt
        self.finger_joint_left_ft[:] = self.finger_joint_left_impulse / dt
        # clear impulse memory
        self.wrist_joint_impulse[:] = 0
        self.finger_joint_right_impulse[:] = 0
        self.finger_joint_left_impulse[:] = 0

        self.counter += 1

    # links
    def get_link_frame(self, link_index, compute_fk=0, com=True):
        all = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=compute_fk, physicsClientId=self.id)
        if com:
            pos, orn = all[:2]
        else:
            pos, orn = all[4:6]  # urdf link frame
        return CoordinateFrame(world_frame_3D, R.from_quat(orn).inv(), np.asarray(pos))

    # joints
    def get_joint_frame(self, joint_index, compute_fk=False):
        j_info = self.joint_infos[joint_index]
        # j_state = p.getJointState(self.robot_id, joint_index, physicsClientId=self.id)
        pf_pos, pf_orn, p_idx = j_info[-3:]
        # frame of the parent
        p_frame = self.get_link_frame(p_idx, compute_fk=compute_fk)
        return CoordinateFrame(p_frame, R.from_quat(pf_orn).inv(), np.asarray(pf_pos))

    # EE state
    def get_end_effector_pos(self):
        return np.array(p.getLinkState(self.robot_id, self.end_effector_index, physicsClientId=self.id)[0])

    def get_end_effector_vel(self):
        return np.array(
            p.getLinkState(self.robot_id, self.end_effector_index, computeLinkVelocity=1, physicsClientId=self.id)[6])

    def get_end_effector_orn(self):
        return np.array(p.getLinkState(self.robot_id, self.end_effector_index, physicsClientId=self.id)[1])

    def get_end_effector_ang_vel(self):
        return np.array(
            p.getLinkState(self.robot_id, self.end_effector_index, computeLinkVelocity=1, physicsClientId=self.id)[7])

    def get_end_effector_frame(self):
        all = p.getLinkState(self.robot_id, self.end_effector_index, physicsClientId=self.id)
        pos = all[0]
        orn = all[1]
        frame = CoordinateFrame(world_frame_3D, R.from_quat(orn).inv(), np.asarray(pos))
        return frame

    def _get_ft(self, frame: CoordinateFrame = None, joint=None):
        if joint is None:
            joint = self.wrist_joint_index

        external_ft = p.getJointState(self.robot_id, joint, physicsClientId=self.id)[2]
        external_ft = np.array(external_ft).copy()
        if frame is not None:
            jframe = self.get_joint_frame(joint)
            j2f, _ = CoordinateFrame.transform_from_a_to_b(jframe, frame)
            external_ft[:3] = j2f.apply(external_ft[:3])
            external_ft[3:] = j2f.apply(external_ft[3:])

        # if joint == self.wrist_joint_index:
        #     print(external_ft[2] / 9.81)
        if joint == self.wrist_joint_index and self.wrist_ft_compensate and frame is not None:
            if not self.wrist_hanging_mass:
                # print("setting")
                self.wrist_hanging_mass = abs(external_ft[2] / 9.81)  # force / acc
                # logger.warn("Mass estimating (ext_ft = %f): %f" % (external_ft[2], self.wrist_hanging_mass))

            external_ft[2] -= self.wrist_hanging_mass * 9.81  # sign here is confusing

        return external_ft

    def get_contact_force(self, link=None):
        f_total = np.zeros(3)
        if link is None:
            cpts = p.getContactPoints(bodyA=self.robot_id)
        else:
            cpts = p.getContactPoints(bodyA=self.robot_id, linkA=link)

        for pt in cpts:
            if pt[3] >= self.end_effector_index:
                normal = np.array(pt[7]) * pt[9]
                lat_f1 = np.array(pt[11]) * pt[10]
                lat_f2 = np.array(pt[13]) * pt[12]
                f_total += normal + lat_f1 + lat_f2

        return f_total

    def set_gripper_max_force(self, force):
        # friction coef is pretty high so chill
        if isinstance(force, Number):
            self.gripper_max_force = force
            self.gripper_max_force_list = [force] * len(self.gripper_max_force_list)
        else:
            self.gripper_max_force = max(force)
            self.gripper_max_force_list = np.broadcast_to(force, (len(self.gripper_max_force_list),)).tolist()

    def set_arm_max_force(self, force):
        self.arm_max_force = force
        self.arm_max_force_list = [force] * len(self.arm_max_force_list)

    def set_joint_damping(self, jd, indices=None):
        if indices is None:
            indices = self.dof_joints
        self.jd = jd
        for i, ji in enumerate(indices):
            p.changeDynamics(self.robot_id, ji, jointDamping=self.jd[i], physicsClientId=self.id)

    def gripper_q_from_pos(self, pos):
        """
        pos is the commanded gripper state (1D), q is the resulting joint state

        """
        gripper_alpha_open = 1.0 - pos / 255.0
        # linear interp from low -> high
        gripper_pos = np.array(self.gripper_ul) * (1 - gripper_alpha_open) + np.array(
            self.gripper_ll) * gripper_alpha_open
        return gripper_pos

    """ GETTERS """

    def get_obs(self):
        out = d()
        out['ee_position'] = self.get_end_effector_pos()
        out['ee_orientation'] = self.get_end_effector_orn()
        out['ee_orientation_eul'] = R.from_quat(out['ee_orientation']).as_euler("xyz")
        out['ee_velocity'] = self.get_end_effector_vel()
        out['ee_angular_velocity'] = self.get_end_effector_ang_vel()
        out['joint_positions'] = np.asarray(self.get_joint_values()[self.dof_joints])
        out['joint_velocities'] = np.asarray(self.get_joint_values(velocities=True)[self.dof_joints])
        out['gripper_tip_pos'] = self.get_gripper_tip_pos()
        out['gripper_pos'] = np.array([self.get_gripper_pos()])  # angle (0 -> 255)
        out['wrist_ft'] = self.wrist_joint_ft.copy()
        out['finger_left_ft'] = self.finger_joint_right_ft.copy()
        out['finger_right_ft'] = self.finger_joint_left_ft.copy()
        out['finger_left_contact'] = np.array([self.any_in_contact(link=self.gripper_left_tip_index)])
        out['finger_right_contact'] = np.array([self.any_in_contact(link=self.gripper_right_tip_index)])
        out['contact_force'] = np.array(self.get_contact_force())

        if self._apply_ft_transform:
            for key in ['wrist_ft', 'finger_left_ft', 'finger_right_ft']:
                out[key] = BulletRobot.ft_transform(out[key])

        # controller outputs added in
        out.combine(self.controllers.leaf_kv_apply(lambda name, ctrl: d(name=ctrl.get_obs())))

        return out.leaf_apply(lambda arr: arr[None])  # 1 x for each

    def get_gripper_tip_pos(self):
        left_tip_pos = p.getLinkState(self.robot_id, self.gripper_left_tip_index, physicsClientId=self.id)[0]
        right_tip_pos = p.getLinkState(self.robot_id, self.gripper_right_tip_index, physicsClientId=self.id)[0]
        # average
        gripper_tip_pos = 0.5 * np.array(left_tip_pos) + 0.5 * np.array(right_tip_pos)
        return gripper_tip_pos

    def set_robot_base(self, frame: Union[CoordinateFrame, Tuple[np.ndarray, np.ndarray]]):
        if isinstance(frame, tuple):
            # first np array is pos, second is orn
            p.resetBasePositionAndOrientation(self.robot_id, frame[0], frame[1], physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.robot_id, frame.pos, frame.orn, physicsClientId=self.id)

    def set_joint_values(self, q, gripper):
        for idx, j in enumerate(self.controlled_arm_joints):
            p.resetJointState(self.robot_id, jointIndex=j, targetValue=q[idx],
                              targetVelocity=0.0, physicsClientId=self.id)
        self.gripperOpen = 1 - gripper / 255.0
        self.gripperPos = np.array(self.gripper_ul) * (1 - self.gripperOpen) + np.array(
            self.gripper_ll) * self.gripperOpen
        for idx, j in enumerate(self.controlled_gripper_joints):
            p.resetJointState(self.robot_id, jointIndex=j, targetValue=self.gripperPos[idx],
                              targetVelocity=0.0, physicsClientId=self.id)

    def set_joint_value(self, joint, value, vel=0.0):
        p.resetJointState(self.robot_id, joint, value, targetVelocity=vel, physicsClientId=self.id)

    def get_joint_values(self, velocities=False):
        idx = int(velocities)  # what values to return
        jointList = []
        for jointIndex in range(self.num_joints):
            js = p.getJointState(self.robot_id, jointIndex, physicsClientId=self.id)
            jointList.append(js[idx])
        return np.array(jointList)

    def get_gripper_pos(self):
        jointInfo = p.getJointState(self.robot_id, self.controlled_gripper_joints[-1], physicsClientId=self.id)
        angle = jointInfo[0]
        angle = (angle - self.gripper_ll[-1]) / (
                self.gripper_ul[-1] - self.gripper_ll[-1]) * 255.0
        return angle

    ### robot agnostic controllers ###

    def create_controller(self, type: ControlType, name: str, params: d):
        assert not self.controllers.has_leaf_key(name), "Controller of name %s already created" % name
        params = params.copy()
        params.type = type

        # PID variants, create controller in our controller dict
        if type in [ControlType.P, ControlType.PD, ControlType.PI, ControlType.PID]:
            self.controllers[name] = PIDController(params)
        else:
            raise NotImplementedError(type.name)

    def joint_position_control(self, joint_indices, joint_target_positions, joint_target_velocities=None,
                               gripper_pos=None, max_velocity=None, step_simulation=False):
        """
        pass in array of joint **positions**, for the controllable robot joints

        :param joint_indices: joint idxs
        :param joint_target_positions: joint target positions
        :param joint_target_velocities: joint target velocities (additional error term used if specified)
        :param gripper_pos: uint8, None if no gripper pos control
        :param max_velocity: (TODO)
        :param step_simulation:
        """

        assert len(joint_indices) == len(joint_target_positions) and len(
            joint_indices) <= self.num_controlled_arm_joints
        all_js = list(joint_target_positions)
        #
        # if joint_target_velocities is not None:
        #     assert len(joint_target_velocities) == len(all_js)
        #     all_js_vels = list(joint_target_velocities)
        # else:
        #     all_js_vels = None

        kwargs = OrderedDict()
        kwargs['bodyUniqueId'] = self.robot_id
        kwargs['jointIndices'] = self.controlled_arm_joints
        kwargs['controlMode'] = p.POSITION_CONTROL
        kwargs['targetPositions'] = all_js
        if joint_target_velocities is not None:
            assert len(joint_target_velocities) == len(all_js)
            kwargs['targetVelocities'] = list(joint_target_velocities)
        kwargs['forces'] = self.arm_max_force_list
        if max_velocity is not None:
            kwargs['maxVelocity'] = max_velocity
        kwargs['physicsClientId'] = self.id

        if gripper_pos is None:
            p.setJointMotorControlArray(**kwargs)
        else:
            gripper_q = self.gripper_q_from_pos(gripper_pos).tolist()
            p.setJointMotorControlArray(**kwargs)
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.controlled_gripper_joints,
                                        controlMode=p.POSITION_CONTROL, targetPositions=gripper_q,
                                        forces=self.gripper_max_force_list,
                                        physicsClientId=self.id)
        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

        return all_js

    def joint_velocity_control(self, joint_indices, joint_target_velocities,
                               step_simulation=False):
        """
        pass in array of joint **positions**, for the controllable robot joints

        :param joint_indices: joint idxs
        :param joint_target_velocities: joint target velocities
        :param step_simulation:
        """
        assert len(joint_indices) == len(joint_target_velocities) and len(
            joint_indices) <= self.num_controlled_arm_joints

        all_js_vels = list(joint_target_velocities)

        # TODO gripper vel support here
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.controlled_arm_joints,
                                    controlMode=p.VELOCITY_CONTROL, targetVelocities=all_js_vels,
                                    physicsClientId=self.id)

        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

    def joint_torque_control(self, joint_indices, joint_torques, max_torques=np.inf,
                             step_simulation=False, save=True):
        """
        pass in array of joint **torques**, for the controllable robot joints

        :param joint_indices: joint idxs
        :param joint_torques: joint torques
        :param step_simulation:
        """
        # assert len(joint_indices) == len(joint_torques) and len(joint_indices) <= self.num_controlled_arm_joints

        clipped_torques = np.clip(
            joint_torques, -np.asarray(max_torques), np.asarray(max_torques))

        internal_friction = [0.0] * len(joint_indices)
        # first disable velocity motors

        mode = p.VELOCITY_CONTROL
        for i in range(len(joint_indices)):
            p.setJointMotorControl2(self.robot_id, joint_indices[i], mode,
                                    force=internal_friction[i], physicsClientId=self.id)

        # TODO gripper vel support here
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=joint_indices,
                                    controlMode=p.TORQUE_CONTROL, forces=list(clipped_torques),
                                    physicsClientId=self.id)

        if save:
            self.last_applied_torques[joint_indices] = clipped_torques

        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

    def compute_frame_ik(self, link_idx, target_frame: CoordinateFrame, offset_frame: CoordinateFrame = world_frame_3D,
                         rest_pose=None, curr_position=None, skip_orn=False):
        # com_pos, com_orn = p.getLinkState(self.robot_id, link_idx, physicsClientId=self.id)[:2]
        # curr_joint_frame = CoordinateFrame(world_frame_3D, R.from_quat(com_orn).inv(), np.array(com_pos))
        target_joint_frame = target_frame.apply_a_to_b(offset_frame, world_frame_3D)

        kwargs = OrderedDict()
        args = [self.robot_id, link_idx, target_joint_frame.pos]
        # kwargs['bodyUniqueId'] = self.robot_id
        # kwargs['endEffectorLinkIndex'] = link_idx
        # kwargs['targetPosition'] = target_joint_frame.pos

        if rest_pose is not None:
            kwargs['lowerLimits'] = list(self.ll)
            kwargs['upperLimits'] = list(self.ul)
            kwargs['jointRanges'] = list(self.jr)
            kwargs['restPoses'] = list(rest_pose)

        # if curr_position is not None:
        #     kwargs['currPosition'] = list(curr_position)

        if not skip_orn:
            args.append(target_joint_frame.orn)

        kwargs['maxNumIterations'] = 100
        kwargs['residualThreshold'] = 1e-6
        kwargs['jointDamping'] = list(self.jd)
        kwargs['physicsClientId'] = self.id

        # IK
        return np.asarray(p.calculateInverseKinematics(*args, **kwargs)[:self.num_controlled_arm_joints])

    def compute_forward_kinematics(self, q):
        def RotX(q):
            return np.array(
                [[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])

        def RotZ(q):
            return np.array(
                [[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        def TransX(q, x, y, z):
            return np.array(
                [[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])

        def TransZ(q, x, y, z):
            return np.array(
                [[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi / 2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi / 2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi / 2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi / 2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi / 2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi / 2, 0.088, 0, 0), RotZ(q[6]))
        H_panda_hand = TransZ(-np.pi / 4, 0, 0, 0.2105)
        H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        p, R = H[:, 3][:3], H[0:3, 0:3]
        return p + self.start_pos, mat2euler(R)

    def os_position_control(self, link_idx, target_frame: CoordinateFrame,
                            offset_frame: CoordinateFrame = world_frame_3D, rest_pose=None, grip_pos=None,
                            skip_orn=False, step_simulation=False):
        """
        Operation space control, in some control frame, for the robot joints + maybe gripper
            to reach a target ** velocity **

        :param link_idx: link idx that the pose is relative to
        :param target_frame: set-point frame
        :param offset_frame: pose of frame pt in frame of joint (idx)
        :param rest_pose: the rest pose for ik (optional), num_dof entries
        :param grip_pos: the grip 1D position [ closed (0) -> open (255) ] for this class
        :param skip_orn: control only target position
        :param step_simulation: steps self.id sim

        """

        q_desired = self.compute_frame_ik(link_idx, target_frame, offset_frame=offset_frame, rest_pose=rest_pose,
                                          skip_orn=skip_orn)

        # damping velocities effect
        return self.joint_position_control(self.controlled_arm_joints, joint_target_positions=q_desired,
                                           joint_target_velocities=np.zeros(len(q_desired)), gripper_pos=grip_pos,
                                           step_simulation=step_simulation)

    def os_velocity_control(self, link_idx, linear_velocity: np.ndarray, angular_velocity: np.ndarray = None,
                            offset_frame: CoordinateFrame = world_frame_3D, step_simulation=False):
        """
        :param link_idx: link idx that the pose is relative to
        :param linear_velocity: desired linear velocity in offset frame
        :param angular_velocity: desired angular velocity in offset frame
        :param offset_frame: pose of frame pt in frame of joint (idx)
        :param step_simulation: steps self.id sim
        """

        o2l = offset_frame.rot
        o_in_l = offset_frame.pos

        jpos = self.get_joint_values()[self.dof_joints]
        jvel = self.get_joint_values(velocities=True)[self.dof_joints]

        if angular_velocity is None:
            angular_velocity = np.zeros(3)

        # for velocity control, we compute q_dot_desired from the jacobians (linear and rotational)
        o_jl, o_jr = p.calculateJacobian(self.robot_id,
                                         linkIndex=link_idx,
                                         localPosition=list(o_in_l),
                                         objPositions=list(jpos),
                                         objVelocities=list(jvel),
                                         objAccelerations=list(self.joint_accelerations),
                                         physicsClientId=self.id)

        # qdot = ([Jl.T Jr.T].T).inv * v_r
        J = np.concatenate([o_jl, o_jr], axis=0)
        xdot = np.concatenate([o2l.apply(linear_velocity), o2l.apply(angular_velocity)])
        q_dot_desired = np.linalg.pinv(J) @ xdot

        # only control arm
        q_dot_desired = q_dot_desired[:self.num_controlled_arm_joints]

        return self.joint_velocity_control(self.controlled_arm_joints, q_dot_desired,
                                           step_simulation=step_simulation)

    def os_force_control(self, link_idx, force_feedback_inputs: d, x_feedback_inputs: d = None,
                         xdot_feedback_inputs: d = None, posture_q_inputs: d = None, posture_qdot_inputs: d = None,
                         offset_frame: CoordinateFrame = world_frame_3D, xddot_direct: np.ndarray = None, grip_pos=0,
                         inertia_compensation=False, uncoupled=True,
                         step_simulation=False):
        """
        :param link_idx: link idx that the pose is relative to
        :param force_feedback_inputs: inputs to compute feedback for "force"
        :param x_feedback_inputs: inputs to compute feedback for "x"
        :param xdot_feedback_inputs: inputs to compute feedback for "x"
        :param posture_q_inputs: inputs to compute "q" related feedback (in task null space)
        :param posture_qdot_inputs: inputs to compute "qdot" related feedback (in task null space)
        # :param torque: true torques in offset frame (optional), defaults to having no effect
        # :param xddot: acceleration term in offset frame (optional), defaults to all zero
        :param offset_frame: pose of frame pt in frame of joint (idx)
        :param xddot_direct: This gets directly added to the xddot input, e.g. the output of a DMP policy
        :param grip_pos: gripper q to try to reach
        :param inertia_compensation: add the force "error" term to xddot if True, otherwise add to Mee @ xddot
        :param uncoupled: if True, force and torque induced tau can be computed separately and added together
        :param step_simulation: steps self.id sim
        """
        assert self.controllers.has_leaf_key("force"), \
            "Controller (force) must already be instantiated before calling os_controllers"

        # two feedback terms in offset frame
        force_feedback = self.controllers.force.forward(force_feedback_inputs)
        if force_feedback.shape[0] == 3:
            force_feedback = np.concatenate([force_feedback, np.zeros(3)])  # no torque inputs yet
        # force_feedback = np.zeros(6)

        # task (OS) terms
        x_feedback = (self.controllers >> "x").forward(x_feedback_inputs) if x_feedback_inputs is not None else np.zeros(6)
        xdot_feedback = (self.controllers >> "xdot").forward(
            xdot_feedback_inputs) if xdot_feedback_inputs is not None else np.zeros(6)

        # posture (JS) terms
        posture_q_term = (self.controllers >> "posture_q").forward(
            posture_q_inputs) if posture_q_inputs is not None else np.zeros(self.num_dof_joints)
        posture_qdot_term = (self.controllers >> "posture_qdot").forward(
            posture_qdot_inputs) if posture_qdot_inputs is not None else np.zeros(self.num_dof_joints)

        o2l = offset_frame.rot
        o_in_l = offset_frame.pos

        jpos = self.get_joint_values()[self.dof_joints]
        jvel = self.get_joint_values(velocities=True)[self.dof_joints]

        # for xdot based pos control, we compute q_desired from the jacobians (linear and rotational)
        o_jl, o_jr = p.calculateJacobian(self.robot_id,
                                         linkIndex=link_idx,
                                         localPosition=list(o_in_l),
                                         objPositions=list(jpos),
                                         objVelocities=[0] * len(jpos),
                                         objAccelerations=[0] * len(jpos),
                                         physicsClientId=self.id)

        M = np.asarray(p.calculateMassMatrix(self.robot_id, list(jpos), physicsClientId=self.id))
        # M_inv = np.linalg.inv(M)

        Jl_op = o2l.inv().as_matrix() @ np.asarray(o_jl)
        Jr_op = o2l.inv().as_matrix() @ np.asarray(o_jr)
        J_op = np.concatenate([Jl_op, Jr_op], axis=0)

        # operational space matrices
        M_op, Ml_op, Mr_op, N_op = control_utils.opspace_matrices(M, J_op, Jl_op, Jr_op)

        desired_acc = x_feedback + xdot_feedback
        posture_desired_acc = posture_q_term + posture_qdot_term

        # direct inputs to xddot adjustment term
        if xddot_direct is not None:
            desired_acc = desired_acc + xddot_direct

        # inertia compensation for force feedback as well
        if inertia_compensation:
            desired_acc = desired_acc + force_feedback

        # force command
        f_command = M_op @ desired_acc  # M_op @

        # forces directly contribute to joint forces
        if not inertia_compensation:
            f_command += force_feedback

        # uncoupled
        if uncoupled:
            tau = Jl_op.T @ f_command[:3] + Jr_op.T @ f_command[3:]
        else:
            tau = J_op.T @ f_command

        # posture setting is important
        posture_tau = N_op.T @ (M @ posture_desired_acc)

        # # final joint command \tau = u(task_feedback) + u(posture_feedback) + N(q, qdot)
        joint_torque = tau + posture_tau + self.N_q(jpos, jvel)

        self.joint_torque_control(self.dof_joints, joint_torque,
                                  max_torques=self.arm_max_force_list + self.gripper_max_force_list,
                                  step_simulation=False)

        # grip control is requested, overrides above command for last set of joints
        if grip_pos is not None:
            self.gripper_control(grip_pos, step_simulation=False)

        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

    def os_torque_control(self, link_idx, x_feedback_inputs: d = None,
                          xdot_feedback_inputs: d = None, posture_q_inputs: d = None, posture_qdot_inputs: d = None,
                          offset_frame: CoordinateFrame = world_frame_3D, grip_pos=0,
                          uncoupled=True, mass_compensation=True, joint_ori_check=True,
                          step_simulation=False):
        """
        :param link_idx: link idx that the pose is relative to
        :param x_feedback_inputs: inputs to compute feedback for "x"
        :param xdot_feedback_inputs: inputs to compute feedback for "x"
        :param posture_q_inputs: inputs to compute "q" related feedback (in task null space)
        :param posture_qdot_inputs: inputs to compute "qdot" related feedback (in task null space)
        # :param torque: true torques in offset frame (optional), defaults to having no effect
        # :param xddot: acceleration term in offset frame (optional), defaults to all zero
        :param offset_frame: pose of frame pt in frame of joint (idx)
        :param grip_pos: gripper q to try to reach
        :param uncoupled: if True, force and torque induced tau can be computed separately and added together
        :param step_simulation: steps self.id sim
        """

        # logger.debug(x_feedback_inputs.desired - x_feedback_inputs.current)
        xdot_feedback = (self.controllers >> "xdot").forward(
            xdot_feedback_inputs) if xdot_feedback_inputs is not None else np.zeros(6)
        # logger.debug(">>" + str(xdot_feedback))

        # posture (JS) terms
        posture_q_term = (self.controllers >> "posture_q").forward(
            posture_q_inputs) if posture_q_inputs is not None else np.zeros(self.num_dof_joints)
        posture_qdot_term = (self.controllers >> "posture_qdot").forward(
            posture_qdot_inputs) if posture_qdot_inputs is not None else np.zeros(self.num_dof_joints)

        o2l = offset_frame.rot
        o_in_l = offset_frame.pos

        jpos = self.get_joint_values()[self.dof_joints]
        jvel = self.get_joint_values(velocities=True)[self.dof_joints]

        # at_low_boundary = jpos - np.asarray(self.ll)[self.dof_joints] <= 0.01
        # at_high_boundary = np.asarray(self.ul)[self.dof_joints] - jpos <= 0.01
        # print(at_low_boundary, at_high_boundary)
        # print(x_feedback_inputs >> "desired", x_feedback_inputs >> "current")

        # for xdot based pos control, we compute q_desired from the jacobians (linear and rotational)
        o_jl, o_jr = p.calculateJacobian(self.robot_id,
                                         linkIndex=link_idx,
                                         localPosition=list(o_in_l),
                                         objPositions=list(jpos),
                                         objVelocities=list(jvel),
                                         objAccelerations=[0] * len(jpos),
                                         physicsClientId=self.id)

        M = np.asarray(p.calculateMassMatrix(self.robot_id, list(jpos), physicsClientId=self.id))
        # M_inv = np.linalg.inv(M)

        Jl_op = o2l.inv().as_matrix() @ np.asarray(o_jl)
        Jr_op = o2l.inv().as_matrix() @ np.asarray(o_jr)
        J_op = np.concatenate([Jl_op, Jr_op], axis=0)

        # operational space matrices
        M_op, Ml_op, Mr_op, N_op = control_utils.opspace_matrices(M, J_op, Jl_op, Jr_op)

        # TODO joint limit, need to figure out why orientation error term is low for large delta_rot. some singularity?
        # if joint_ori_check and x_feedback_inputs is not None:
        #     des, curr = x_feedback_inputs.get_keys_required(['desired', 'current'])
        #     diff_x = (self.controllers >> "x")._difference_fn(des, curr)
        #     diff_ori = diff_x[3:]
        #     diff_q_ori = Jr_op.T @ diff_ori  # unscaled
        #     # choose the new q direction to be in range (one direct
        #     sign_ul = np.where(diff_q_ori[self.controlled_arm_joints] > np.asarray(self.joint_ul)[self.controlled_arm_joints] - jpos[:self.num_controlled_arm_joints], -1, 1)
        #     sign_ll = np.where(diff_q_ori[self.controlled_arm_joints] < np.asarray(self.joint_ll)[self.controlled_arm_joints] - jpos[:self.num_controlled_arm_joints], -1, 1)
        #
        #     if np.any(np.logical_and(sign_ul == -1, sign_ll == -1)):
        #         print(sign_ul)
        #         print(sign_ll)
        #         import ipdb; ipdb.set_trace()
        #
        #     diff_x[3:] = Jr_op @ (np.concatenate([sign_ul * sign_ll, np.ones(self.num_controlled_gripper_joints)]) * diff_q_ori)
        #     x_feedback = (self.controllers >> "x").forward(x_feedback_inputs, err=diff_x)
        # else:
        #     # task (OS) terms
        #     x_feedback = (self.controllers >> "x").forward(x_feedback_inputs) if x_feedback_inputs is not None else np.zeros(6)
        x_feedback = (self.controllers >> "x").forward(x_feedback_inputs) if x_feedback_inputs is not None else np.zeros(6)

        desired_acc = x_feedback + xdot_feedback
        posture_desired_acc = posture_q_term + posture_qdot_term


        # force command
        if mass_compensation:
            f_command = M_op @ desired_acc  # M_op @
        else:
            f_command = desired_acc

        # uncoupled
        if uncoupled:
            tau = Jl_op.T @ f_command[:3] + Jr_op.T @ f_command[3:]
        else:
            tau = J_op.T @ f_command

        # posture setting is important
        posture_tau = N_op.T @ (M @ posture_desired_acc)

        # # final joint command \tau = u(task_feedback) + u(posture_feedback) + N(q, qdot)
        joint_torque = tau + posture_tau + self.N_q(jpos, jvel)

        self.joint_torque_control(self.dof_joints, joint_torque,
                                  max_torques=self.arm_max_force_list + self.gripper_max_force_list,
                                  step_simulation=False)

        # grip control is requested, overrides above command for last set of joints
        if grip_pos is not None:
            self.gripper_control(grip_pos, step_simulation=False)

        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

    def N_q(self, joint_positions=None, joint_velocities=None):
        if joint_positions is None:
            joint_positions = self.get_joint_values()[self.dof_joints]
        if joint_velocities is None:
            joint_velocities = self.get_joint_values(velocities=True)[self.dof_joints]
        return np.asarray(p.calculateInverseDynamics(self.robot_id, list(joint_positions),
                                                     list(joint_velocities),
                                                     [0] * len(joint_positions), physicsClientId=self.id))

    def gripper_control(self, gripper_pos, control_mode=p.POSITION_CONTROL, step_simulation=True):
        """
        gripper control

        :param gripper_pos: the gripper position
        :param control_mode: joint controller mode
        :param step_simulation: steps self.id sim

        """
        self.target_gripper_angle = gripper_pos
        curr_gripper_pos = self.get_gripper_pos()
        if self.setpoint_gripper_angle is None:
            self.setpoint_gripper_angle = curr_gripper_pos

        # internal interpolation for gripper angle
        delta = np.clip(self.target_gripper_angle - self.setpoint_gripper_angle, -self.gripper_max_delta, self.gripper_max_delta)

        self.setpoint_gripper_angle = self.setpoint_gripper_angle + delta  # update the setpoint

        self.gripperPos = self.gripper_q_from_pos(self.setpoint_gripper_angle).tolist()

        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.controlled_gripper_joints,
                                    controlMode=control_mode, targetPositions=self.gripperPos,
                                    forces=self.gripper_max_force_list,
                                    physicsClientId=self.id)
        if step_simulation:
            p.stepSimulation(physicsClientId=self.id)

    def is_colliding(self):
        for x in self.controlled_gripper_joints:
            for y in [0, 1, 2, 3, 4, 5, 6]:
                # c = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, linkIndexA=x, linkIndexB=y)
                cl = p.getClosestPoints(bodyA=self.robot_id, bodyB=self.robot_id, distance=100, linkIndexA=x,
                                        linkIndexB=y)
                if cl is not None and len(cl) > 0:
                    if cl[0][8] < 0.02:
                        return True
        return False

    def any_in_contact(self, link: Union[List[int], int]):
        # return if any of the passed in links is in contact
        if not isinstance(link, List):
            link = [link]

        return any(len(p.getContactPoints(bodyA=self.robot_id, linkIndexA=l)) > 0 for l in link)

    def all_in_contact(self, link: Union[List[int], int]):
        # return if any of the passed in links is in contact
        if not isinstance(link, List):
            link = [link]

        return all(len(p.getContactPoints(bodyA=self.robot_id, linkIndexA=l)) > 0 for l in link)

    def collision_detection_links(self):
        # these links should be collision free with external objects (e.g. table)
        return self._collision_detection_links

    @staticmethod
    def ft_transform(ft, log_fn=np.log2):
        abs = np.abs(ft)
        sgn = np.sign(ft)
        return sgn * log_fn(1 + abs)
