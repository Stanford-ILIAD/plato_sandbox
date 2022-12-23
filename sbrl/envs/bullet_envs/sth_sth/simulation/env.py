#!/usr/bin/env python3
import math
import os

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.robot import BulletRobot
from sbrl.envs.bullet_envs.robot_bullet_env import RobotBulletEnv
from sbrl.envs.bullet_envs.utils_env import point2traj, get_view, RobotControllerMode
from sbrl.experiments import logger
from sbrl.policies.controllers.pid_controller import ControlType
from sbrl.utils import control_utils
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import get_with_default, AttrDict as d, get_required
################################
from sbrl.utils.script_utils import is_next_cycle

#################################
# np.set_printoptions(precision=4, suppress=True, linewidth=300)
# sys.path.insert(0, "../external/something-something-v2-baseline.git")
# sys.path.insert(0, "../classification/image")
# sys.path.insert(0, "../../")

RCM = RobotControllerMode
CT = ControlType


#
# def pose_diff_fn(pose1, pose2):
#     return np.array(
#         [circular_difference(pose1[i], pose2[i]) if i >= 3 else pose1[i] - pose2[i] for i in range(len(pose1))])
#


class BulletSth(RobotBulletEnv):
    def _init_params_to_attrs(self, params: d):
        super(BulletSth, self)._init_params_to_attrs(params)
        self.resources_dir = os.path.join(self.asset_directory, 'sth')
        assert os.path.exists(self.resources_dir)
        # assert self.skip_n_frames_every_step == 1, "Skipping frames doesn't work with f/t sensors"

        # self.max_steps = get_with_default(params, "max_steps", 23)
        self.task_id = get_with_default(params, "task_id", 15)
        self.robot_params = get_with_default(params, "robot_params", d())
        self.robot_model = get_with_default(params, "robot_model", BulletRobot)
        self.dt = self.time_step * self.skip_n_frames_every_step  # get_with_default(params, "dt", 1. / 100, map_fn=float)

        self.robot_controller_mode = get_with_default(params, "robot_controller_mode", RCM.x_pid)

        # controller parameter specification
        if self.robot_controller_mode in [RCM.xddot_with_force_pid, RCM.xddot_with_zero_force_pid, RCM.xddot_pid]:
            # separate clause to accomodate the 2 controllers used within the robot, special case
            self.x_controller_params = get_with_default(params, "x_controller_params",
                                                        d(type=CT.PD, dim=6, difference_fn=control_utils.pose_diff_fn))
            self.posture_q_controller_params = get_with_default(params, "posture_q_controller_params",
                                                                d(type=CT.PD, dim=None))  # filled in setup
            self.xdot_controller_params = get_with_default(params, "xdot_controller_params", d(type=CT.P, dim=6))
            self.posture_qdot_controller_params = get_with_default(params, "posture_qdot_controller_params",
                                                                   d(type=CT.P, dim=None))  # filled in setup
            # not always used
            self.force_controller_params = get_with_default(params, "force_controller_params", d(type=CT.PI, dim=6))
            for p in [self.x_controller_params, self.posture_q_controller_params,
                      self.xdot_controller_params, self.posture_qdot_controller_params, self.force_controller_params]:
                p.dt = self.time_step if self._control_inner_step else self.dt
        else:
            self.robot_controller_params = get_with_default(params, "robot_controller_params", d(type=CT.PID, dim=6,
                                                                                                 difference_fn=control_utils.pose_diff_fn))  # pose otherwise
            self.robot_controller_params.dt = self.time_step if self._control_inner_step else self.dt

        if not self.robot_params.has_leaf_key('resources_dir'):
            self.robot_params.resources_dir = self.resources_dir
        self.robot_params.task_id = self.task_id

        self.view_matrix, self.proj_matrix = get_view(params)
        self.q_home = np.array((0., -np.pi / 6., 0., -5. / 6. * np.pi, 0., 2. / 3. * np.pi, 0.))
        self._initial_joint_positions = np.zeros(7)
        # assert self._use_gravity TODO put back

        self.num_resets = 0
        self.reset_full_every_n_resets = get_with_default(params, "reset_full_every_n_resets", 20, map_fn=int)  # memory leaks otherwise :(

        self.table_friction_ceof = get_with_default(params, "table_friction_coef", 0.4)

        # ADD TO THIS if you subclass
        self.objects = []

    def _init_setup(self):
        super(BulletSth, self)._init_setup()

        if self.task_id == int(49):
            p.setPhysicsEngineParameter(numSolverIterations=20, physicsClientId=self.id)
            p.setPhysicsEngineParameter(numSubSteps=10, physicsClientId=self.id)
        else:
            p.setPhysicsEngineParameter(numSolverIterations=40, physicsClientId=self.id)
            p.setPhysicsEngineParameter(numSubSteps=40, physicsClientId=self.id)

        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.id)

        # self.robot_recordings_dir = os.path.join(self.file, 'data', 'robot_recordings')
        # self.log_root = os.path.join(opti.project_root,'logs')
        # self.log_root = safe_path(self.log_root+'/td3_log/test{}'.format(self.test_id))
        # self.sim_dir = os.path.join(self.opti.project_dir, 'simulation')
        # print("self.opti.project_dir", self.opti.project_dir)
        # self.env_dir = self.sim_dir

        self.env_step = 0
        self.success_flag = False

        self.obs_history = d()

    def _load_robot(self):
        self.robot = self.robot_model(self.id, self.robot_params.leaf_copy())
        self.robotId = self.robot.robot_id
        self.robotEndEffectorIndex = self.robot.end_effector_index
        # 1.35433812e-01 - 3.25247350e-01 - 9.98125170e-02 - 1.95713799e+00
        # -3.19091457e-02
        # 1.63336107e+00
        # 4.28182175e-02
        # 0.00000000e+00

        # making sure joint related controllers have the right dof
        for cp_name in ["posture_q_controller_params", "posture_qdot_controller_params"]:
            if hasattr(self, cp_name) and getattr(self, cp_name).dim is None:
                getattr(self, cp_name).dim = self.robot.num_dof_joints

        # register the controller(s) for the robot to use
        if self.robot_controller_mode == RCM.xddot_pid:
            self.robot.create_controller(get_required(self.x_controller_params, "type"), "x", self.x_controller_params)
            self.robot.create_controller(get_required(self.posture_q_controller_params, "type"), "posture_q",
                                         self.posture_q_controller_params)
            self.robot.create_controller(get_required(self.xdot_controller_params, "type"), "xdot",
                                         self.xdot_controller_params)
            self.robot.create_controller(get_required(self.posture_qdot_controller_params, "type"), "posture_qdot",
                                         self.posture_qdot_controller_params)
        elif self.robot_controller_mode == RCM.x_pid:
            # TODO actually use this
            self.robot.create_controller(get_required(self.robot_controller_params, "type"), "x",
                                         self.robot_controller_params)
        elif self.robot_controller_mode == RCM.xdot_pid:
            # TODO actually use this
            self.robot.create_controller(get_required(self.robot_controller_params, "type"), "xdot",
                                         self.robot_controller_params)
        elif self.robot_controller_mode in [RCM.xddot_with_force_pid, RCM.xddot_with_zero_force_pid]:
            self.robot.create_controller(get_required(self.force_controller_params, "type"), "force",
                                         self.force_controller_params)
            self.robot.create_controller(get_required(self.x_controller_params, "type"), "x", self.x_controller_params)
            self.robot.create_controller(get_required(self.posture_q_controller_params, "type"), "posture_q",
                                         self.posture_q_controller_params)
            self.robot.create_controller(get_required(self.xdot_controller_params, "type"), "xdot",
                                         self.xdot_controller_params)
            self.robot.create_controller(get_required(self.posture_qdot_controller_params, "type"), "posture_qdot",
                                         self.posture_qdot_controller_params)
        else:
            raise NotImplementedError

    def _load_assets(self):
        """ loading various assets into scene """

        """ TABLE """
        self.load_table()

        """ OBJECTS for the scene """
        self._load_asset_objects()

    def _load_asset_objects(self):
        # ADD TO THIS if you subclass
        pass

    def _load_dynamics(self):
        p.setPhysicsEngineParameter(enableConeFriction=1, physicsClientId=self.id)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001, physicsClientId=self.id)
        p.setPhysicsEngineParameter(allowedCcdPenetration=0.0, physicsClientId=self.id)

        p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG, globalCFM=0.000001,
                                    physicsClientId=self.id)
        # table
        p.changeDynamics(self.table_id, -1,
                         lateralFriction=self.table_friction_ceof,
                         rollingFriction=self.table_friction_ceof,
                         spinningFriction=self.table_friction_ceof,
                         contactStiffness=1.0,
                         contactDamping=0.9,
                         physicsClientId=self.id)

    def load_table(self):
        table_path = os.path.join(self.resources_dir, 'urdf/table/table.urdf')
        self.table_id = p.loadURDF(table_path, [0.42, 0, 0], [0, 0, math.pi * 0.32, 1],
                                   globalScaling=0.6, physicsClientId=self.id)
        texture_path = os.path.join(self.resources_dir, 'textures/table_textures/table_texture.jpg')
        self.table_textid = p.loadTexture(texture_path, physicsClientId=self.id)
        p.changeVisualShape(self.table_id, -1, textureUniqueId=self.table_textid, physicsClientId=self.id)

        self.table_aabb = self.get_aabb(self.table_id)

    # def init_motion(self):
    #     # TODO: use json
    #     self.data_q = np.load(os.path.join(self.configs_dir, 'init', 'q.npy'))
    #     self.data_dq = np.load(os.path.join(self.configs_dir, 'init', 'dq.npy'))
    #     self.data_gripper = np.load(os.path.join(self.configs_dir, 'init', 'gripper.npy'))

    # def save_video(self, img_info, i):
    #     img = img_info[2][:, :, :3]
    #     mask = (img_info[4] > 10000000)
    #     mask_id_label = [234881025, 301989889, 285212673, 268435457, 318767105, 335544321, 201326593, 218103809,
    #                      167772161]
    #     for item in mask_id_label:
    #         mask = mask * (img_info[4] != item)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img[mask] = [127, 151, 182]
    #     cv2.imwrite(os.path.join(self.output_path, '%06d.jpg' % (i)), img)
    #
    #     try:
    #         img = cv2.imread(os.path.join(self.frames_path, '%06d.jpg' % (i + 1)))
    #         img[mask] = [127, 151, 182]
    #         cv2.imwrite(os.path.join(self.mask_frames_path, '%06d.jpg' % (i)), img)
    #     except:
    #         print('no video frame:{}'.format(i))

    def init_obj(self, obj_params: d):
        object_cls, = obj_params.get_keys_required(['object_cls'])
        object_scale, = obj_params.get_keys_optional(['object_scaling'], [None])

        obj = d(
            cls=object_cls,
            file=None,
            position=None,
            orientation=None,
            scaling=None,
            id=None,
        )

        if object_cls == 'bottle':
            obj.file = os.path.join(self.resources_dir, "urdf/objmodels/urdfs/bottle1.urdf")
            obj.position = [0.4, -0.15, 0.42]
            obj.orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            obj.scaling = 1.4 if object_scale is None else object_scale
            obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
                                baseOrientation=obj.orientation,
                                globalScaling=obj.scaling, physicsClientId=self.id)

        if object_cls == 'cup':
            obj.file = os.path.join(self.resources_dir, "urdf/objmodels/urdfs/cup.urdf")
            obj.position = [0.45, -0.18, 0.34]
            obj.orientation = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
            obj.scaling = 0.11 if object_scale is None else object_scale
            obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
                                baseOrientation=obj.orientation,
                                globalScaling=obj.scaling, physicsClientId=self.id)

        if object_cls == 'nut':
            obj.file = os.path.join(self.resources_dir, "urdf/objmodels/nut.urdf")
            obj.position = [0.4, -0.15, 0.34]
            obj.scaling = 2 if object_scale is None else object_scale
            obj.orientation = p.getQuaternionFromEuler([math.pi / 2, -math.pi / 2, 0])
            obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
                                baseOrientation=obj.orientation,
                                globalScaling=obj.scaling, physicsClientId=self.id)
            p.changeVisualShape(obj.id, -1, rgbaColor=[0.3, 0.3, 0.9, 1], physicsClientId=self.id)

        if object_cls == 'bottle_b1':
            obj.file = os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf")
            obj.position = [0.4, -0.15, 0.42]
            obj.orientation = p.getQuaternionFromEuler([0, 0, 0])
            obj.scaling = 1.4 if object_scale is None else object_scale
            obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
                                baseOrientation=obj.orientation,
                                globalScaling=obj.scaling, physicsClientId=self.id)

        if object_cls == 'drawer':
            obj.file = os.path.join(self.resources_dir, "urdf/obj_libs/drawers/d4/d4.urdf")
            obj.position = [0.38, 0.0, 0.35]
            obj.orientation = p.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi])
            obj.scaling = 1.0 if object_scale is None else object_scale
            obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
                                baseOrientation=obj.orientation, useFixedBase=1,
                                globalScaling=obj.scaling, physicsClientId=self.id)

        # the hard-coded frame for the object to start
        obj.base_frame = CoordinateFrame(world_frame_3D, R.from_quat(obj.orientation).inv(), np.array(obj.position))

        return obj

    def set_obj_pose(self, obj: d, position, orientation=None, assign_base_frame=False):
        # set without stepping sim
        if orientation is None:
            orientation = obj.orientation

        if assign_base_frame:
            obj.base_frame = CoordinateFrame(world_frame_3D, R.from_quat(orientation).inv(), np.array(position))

        p.resetBasePositionAndOrientation(obj.id, position, orientation, physicsClientId=self.id)
        p.resetBaseVelocity(obj.id, np.zeros(3), np.zeros(3), physicsClientId=self.id)
        obj.position = position
        obj.orientation = orientation

    def pre_reset(self, presets: d = d()):
        # occasionally we need to actually reset and reload assets to clear some pybullet caches
        self.num_resets += 1
        if is_next_cycle(self.num_resets, self.reset_full_every_n_resets):
            logger.warn("Reloading environment from scratch!")
            p.resetSimulation(physicsClientId=self.id)
            self.objects.clear()
            self.load()

    def reset_obj(self, obj: d, frame: CoordinateFrame = None):
        object_cls, obj_id = obj.get_keys_required(['cls', 'id'])

        # default is reset the obj to its base frame
        if frame is None:
            frame = obj.base_frame
        p.resetBasePositionAndOrientation(obj_id, frame.pos, frame.orn, physicsClientId=self.id)
        p.resetBaseVelocity(obj_id, np.zeros(3), np.zeros(3), physicsClientId=self.id)

    def _step_simulation(self):
        super(BulletSth, self)._step_simulation()
        self.robot.update_state_in_between_step(self.time_step)

    def _after_step_simulation(self):
        self.robot.update_state_after_step(self.dt)

    # def run(self):
    #     for i in range(self.data_q.shape[0]):
    #         jointPoses = self.data_q[i]
    #         for j in range(self.robotEndEffectorIndex):
    #             p.resetJointState(self.robotId, j, jointPoses[j], self.data_dq[i][j])
    #
    #         gripper = self.data_gripper[i]
    #         self.gripperOpen = 1 - gripper / 255.0
    #         self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(
    #             self.gripperLowerLimitList) * self.gripperOpen
    #         for j in range(6):
    #             index_ = self.activeGripperJointIndexList[j]
    #             p.resetJointState(self.robotId, index_, self.gripperPos[j], 0)
    #
    #         img_info = self.p.getCameraImage(width=self.w,
    #                                          height=self.h,
    #                                          viewMatrix=self.view_matrix,
    #                                          projectionMatrix=self.proj_matrix,
    #                                          shadow=-1,
    #                                          flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    #                                          renderer=p.ER_TINY_RENDERER)
    #         self.save_video(img_info, i)
    #         self.p.stepSimulation()

    # def get_traj(self):
    #     pos_traj, orn_traj = [], []
    #     for i in range(self.data_q.shape[0]):
    #         poses = self.data_q[i]
    #         for j in range(7):
    #             p.resetJointState(self.robotId, j, poses[j], self.data_dq[i][j])
    #
    #         state = p.getLinkState(self.robotId, 7)
    #         pos = state[0]
    #         orn = state[1]
    #         pos_traj.append(pos)
    #         orn_traj.append(orn)

    def init_grasp(self, object, presets):
        """
        Grasps an object in the scene

        :param object: AttrDict, object to grasp (holds id, position, etc)
        -------

        """
        raise NotImplementedError
        # pos_traj = np.load(os.path.join(self.config_dir, 'init', 'pos.npy'))
        # orn_traj = np.load(os.path.join(self.config_dir, 'init', 'orn.npy'))
        # self.fix_orn = np.load(os.path.join(self.config_dir, 'init', 'orn.npy'))

        # for j in range(7):
        #     self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
        #
        # for init_t in range(100):
        #     box = self.p.get_aabb(self.obj_id)
        #     center = [(x + y) * 0.5 for x, y in zip(box[0], box[1])]
        #     center[0] -= 0.05
        #     center[1] -= 0.05
        #     center[2] += 0.03
        #     # center = (box[0]+box[1])*0.5
        # points = np.array([pos_traj[0], center])
        #
        # start_id = 0
        # init_traj = point2traj(points)
        # start_id = self.move(init_traj, orn_traj, start_id)
        #
        # self.p.stepSimulation()
        #
        # # grasping
        # grasp_stage_num = 10
        # for grasp_t in range(grasp_stage_num):
        #     gripperPos = (grasp_t + 1.) / float(grasp_stage_num) * 250.0 + 0.0
        #     self.robot.gripperControl(gripperPos)
        #
        #     start_id += 1
        #
        # pos = p.getLinkState(self.robotId, 7)[0]
        # left_traj = point2traj([pos, [pos[0], pos[1] + 0.14, pos[2] + 0.05]])
        # start_id = self.move(left_traj, orn_traj, start_id)
        #
        # self.start_pos = p.getLinkState(self.robotId, 7)[0]

    def move_up(self):
        # move in z-axis direction
        orn_traj = np.load(os.path.join(self.configs_dir, 'init', 'orn.npy'))
        pos = p.getLinkState(self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2] + 0.3]], delta=0.005)
        start_id = self.move(up_traj, orn_traj, 0)

    def explore(self, traj):
        orn_traj = np.load('orn.npy')
        start_id = self.move(traj, orn_traj, 0)

    # def move(self, pos_traj, orn_traj, start_id=0):
    #     for i in range(int(len(pos_traj))):
    #         pos = pos_traj[i]
    #         orn = orn_traj[i]
    #         self.robot.operationSpacePositionControl(pos=pos, orn=orn, null_pose=self.data_q[i])
    #
    #         img_info = self.p.getCameraImage(width=self.w,
    #                                          height=self.h,
    #                                          viewMatrix=self.view_matrix,
    #                                          projectionMatrix=self.proj_matrix,
    #                                          shadow=-1,
    #                                          flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    #                                          renderer=self.p.ER_TINY_RENDERER)
    #         # self.save_video(img_info,start_id+i)
    #     return start_id + len(pos_traj)

    def reset_assets(self, presets: d = d()):
        for obj in self.objects:
            self.reset_obj(obj)

    def reset_robot(self, presets: d = d()):
        self.robot.reset()
        # self.init_grasp()
        self.set_initial_joint_positions()  # recording where we start

    def set_initial_joint_positions(self, q=None):
        if q is None:
            q = self.robot.get_joint_values()[self.robot.controlled_arm_joints]
        self._initial_joint_positions[:] = q

    def reset_dynamics(self, presets: d = d()):
        pass

    def _register_obs(self, obs: d, done: np.ndarray):
        super(BulletSth, self)._register_obs(obs, done)
        # registering happens at the end of the ep, and after reset
        if self.obs_history.is_empty():
            self.obs_history.combine(obs.leaf_apply(lambda arr: [arr]))
        else:
            self.obs_history = d.leaf_combine_and_apply([self.obs_history, obs], lambda vs: vs[0] + [vs[1]])

    def is_outside_bounds(self):
        # are we within the table bounds
        pos_gripper = self.robot.get_gripper_tip_pos()
        tableAABB = p.getAABB(self.table_id, physicsClientId=self.id)
        if pos_gripper[0] > tableAABB[1][0] or pos_gripper[1] < tableAABB[0][1] or pos_gripper[1] > tableAABB[1][1] or \
                pos_gripper[0] < tableAABB[0][0]:
            return True
        else:
            return False

    #
    # def step_dmp(self, action, f_w, coupling, reset, test=False):
    #     if reset:
    #         action = action.squeeze()
    #         self.start_pos = self.robot.getEndEffectorPos()
    #         self.start_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
    #         self.start_gripper_pos = self.robot.getGripperPos()
    #         self.start_status = np.array(
    #             [self.start_pos[0], self.start_pos[1], self.start_pos[2], self.start_orn[0], self.start_orn[1],
    #              self.start_orn[2], 0.0]).reshape((-1,))
    #         self.dmp.set_start(np.array(self.start_status)[:self.dmp.n_dmps])
    #         dmp_end_pos = [x + y for x, y in zip(self.start_status, action)]
    #         self.dmp.set_goal(dmp_end_pos)
    #         if f_w is not None:
    #             self.dmp.set_force(f_w)
    #         self.dmp.reset_state()
    #         self.dmp.rollout()
    #         self.dmp.reset_state()
    #         self.actual_traj = []
    #         p1 = self.start_pos
    #         p1 = np.array(p1)
    #         self.dmp.timestep = 0
    #         small_observation = self.step_within_dmp(coupling)
    #         lenT = len(self.dmp.force[:, 0])
    #     else:
    #         small_observation = self.step_within_dmp(coupling)
    #     seg = None
    #     observation_next, seg = self.get_observation(segFlag=True)
    #     ft_next = self.robot.getEndEffectorForceTorque(world=True)  # in world frame
    #     self.ft_list.append(ft_next)
    #     reward = 0
    #     done = False
    #     suc = False
    #     if self.dmp.timestep >= self.dmp.timesteps:
    #         if test:
    #             self.success_flag = self.get_success(seg)
    #             reward = self.get_reward(seg)
    #             self.termination_flag = True
    #         else:
    #             self.termination_flag = True
    #             if np.sum(seg == 167772162) < 1:
    #                 self.success_flag = False
    #                 reward = -0.1
    #             elif self.robot.colliDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #             elif self.tableDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #             elif self.taskColliDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #             else:
    #                 self.success_flag = self.get_success(seg)
    #                 reward = self.get_reward(seg)
    #     else:
    #         if test:
    #             self.success_flag = self.get_success(seg)
    #             # if self.success_flag:
    #             #  self.termination_flag = True
    #             reward = self.get_reward(seg)
    #             self.termination_flag = False
    #         else:
    #             if np.sum(seg == 167772162) < 1:
    #                 self.success_flag = False
    #                 reward = -0.1
    #                 self.termination_flag = True
    #             elif self.robot.colliDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #                 self.termination_flag = True
    #             elif self.tableDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #                 self.termination_flag = True
    #             elif self.taskColliDet():
    #                 self.success_flag = False
    #                 reward = -0.1
    #                 self.termination_flag = True
    #             else:
    #                 self.success_flag = self.get_success(seg)
    #                 if self.success_flag:
    #                     self.termination_flag = True
    #                     reward = self.get_reward(seg)
    #                 else:
    #                     self.termination_flag = False
    #                     if self.classifier == 'image' and self.opti.only_coupling_term == True:
    #                         reward = self.get_reward(seg)
    # #     return observation_next, reward, self.termination_flag, self.success_flag
    #
    # def get_current_traj_pos(self):
    #     cur_pos = self.robot.getEndEffectorPos()
    #     cur_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
    #     cur_gripper_pos = self.robot.getGripperPos()
    #     return np.array([cur_pos[0], cur_pos[1], cur_pos[2], cur_orn[0], cur_orn[1], cur_orn[2], cur_gripper_pos])
    #
    # def step_within_dmp(self, coupling):
    #     action = self.dmp.step(coupling, )[0]
    #     pos = np.array(action[:3])
    #     if len(action) > 3:
    #         orn = angleaxis2quaternion(action[3:6])
    #     else:
    #         orn = self.fix_orn[0]
    #     if len(action) == 7:
    #         if action[6] < 0:
    #             gripperPos = self.start_gripper_pos + int(action[6] * 255)  # + 127.5
    #         else:
    #             gripperPos = None
    #     else:
    #         gripperPos = None
    #     self.robot.operationSpacePositionControl(pos, orn=orn, null_pose=self.data_q[0], gripperPos=gripperPos)
    #     observation_next = None
    #     self.real_traj_list.append(self.robotCurrentStatus())
    #     return observation_next
    #
    # def step_without_dmp(self, action):
    #     action = action.squeeze()
    #     pos = np.array(self.robot.getEndEffectorPos()) + np.array(action)
    #     self.robot.operationSpacePositionControl(pos, self.data_q[0])
    #     seg = None
    #     observation_next, seg = self.get_observation(segFlag=True)
    #     reward = 0
    #     done = False
    #     reward, done, suc = self.get_reward(seg)
    #     return observation_next, reward, done, suc

    def get_initial_joint_positions(self):
        return self._initial_joint_positions

    def _get_obs(self, ret_images=False, seg_flag=False, **kwargs):
        assert not seg_flag or ret_images

        obs = d()
        obs.combine(self.robot.get_obs())

        # print(obs.gripper_pos)

        # reward terms
        obs.success = np.asarray(self.get_success(seg=seg_flag))[None, None]
        obs.table_collision = np.asarray(self.is_outside_bounds())[None, None]
        obs.task_collision = np.asarray(self.taskColliDet())[None, None]
        obs.robot_collision = np.asarray(self.robot.is_colliding())[None, None]

        if ret_images:
            # get images
            img_info = p.getCameraImage(width=self.img_width,
                                        height=self.img_height,
                                        viewMatrix=self.view_matrix,
                                        projectionMatrix=self.proj_matrix,
                                        shadow=-1,
                                        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                        renderer=p.ER_TINY_RENDERER)
            obs.img = img_info[2][None, :, :, :3]  # rgb image

            if seg_flag:
                obs.seg = img_info[4][None]  # seg map

        # obs.leaf_apply(lambda arr: arr.shape).pprint()
        for i, obj in enumerate(self.objects):
            obs["object_%d" % i] = self._get_object_obs(obj)

        return obs.leaf_filter_keys(
            self.env_spec.observation_names + self.env_spec.output_observation_names + self.env_spec.param_names + self.env_spec.final_names)

    def _get_done(self, obs: d = None):
        sup_done = super(BulletSth, self)._get_done(obs)
        # all (1,1)
        s, tbc, tkc, rc, seg = obs.get_keys_optional(
            ['success', 'table_collision', 'task_collision', 'robot_collision', 'seg'],
            [None] * 5)
        s = self.get_success(seg=seg) if s is None else s[0, 0]
        tbc = self.is_outside_bounds() if tbc is None else tbc[0, 0]
        tkc = self.taskColliDet() if tkc is None else tkc[0, 0]
        rc = self.robot.is_colliding() if rc is None else rc[0, 0]
        # if any([s, tbc, tkc, rc]):
        #     print(s, tkc, tbc, rc)
        # return np.array([s or sup_done[0]])  # TODO
        return np.array([s or tkc or tbc or rc or sup_done[0]])

    def _get_object_obs(self, object: d):
        id, = object.get_keys_required(['id'])
        pos, orn = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
        vel, avel = p.getBaseVelocity(id, physicsClientId=self.id)
        return d(position=np.asarray(pos),
                 orientation=np.asarray(orn),
                 orientation_eul=np.asarray(p.getEulerFromQuaternion(orn)),
                 velocity=np.asarray(vel),
                 angular_velocity=np.asarray(avel)).leaf_apply(lambda arr: arr[None])

    def _control(self, action, **kwargs):
        # last element is always the "grip" command
        grip = action[-1]

        # position control
        if self.robot_controller_mode == RCM.x_pid:
            assert len(action) == 7, action.shape
            # interpret action as pos + orn(eul) (+ grip)
            pos = action[:3]
            orn = R.from_euler("xyz", action[3:6])

            target_frame = CoordinateFrame(world_frame_3D, orn.inv(), pos)
            self.robot.os_position_control(self.robotEndEffectorIndex, target_frame, rest_pose=self.robot.rp,
                                           grip_pos=grip)

        elif self.robot_controller_mode == RCM.xdot_pid:
            assert len(action) == 7, action.shape
            # TODO grip here
            # interpret action as lin_vel + ang_vel(eul) (+ grip)
            lin_vel, ang_vel = action[:3], action[3:6]
            self.robot.os_velocity_control(self.robotEndEffectorIndex, lin_vel, angular_velocity=ang_vel)

        elif self.robot_controller_mode == RCM.xddot_pid:
            assert len(action) == 7, action.shape
            # interpret action as pos + orn(eul) + forces (+ grip)
            pose = action[:6]

            # forces come in at the wrist f/t sensor, convert to world frame
            ee_link_frame = self.robot.get_end_effector_frame()
            posture_q_inps = d(desired=np.concatenate(
                [self.get_initial_joint_positions(), [0] * self.robot.num_controlled_gripper_joints]),
                current=self.robot.get_joint_values()[self.robot.dof_joints])
            posture_qdot_inps = d(desired=np.zeros(self.robot.num_dof_joints),
                                  current=self.robot.get_joint_values(velocities=True)[self.robot.dof_joints])
            # set x to desired
            x_inps = d(desired=pose, current=ee_link_frame.as_pose(world_frame_3D))

            # set vel to zero
            vel = np.concatenate([self.robot.get_end_effector_vel(), self.robot.get_end_effector_ang_vel()])

            xdot_inps = d(desired=np.zeros(6), current=vel)
            self.robot.os_torque_control(self.robotEndEffectorIndex, x_inps, xdot_inps,
                                         posture_q_inps, posture_qdot_inps,
                                         offset_frame=world_frame_3D, uncoupled=False,
                                         grip_pos=grip)

        elif self.robot_controller_mode in [RCM.xddot_with_force_pid, RCM.direct_xddot_with_force_pid, RCM.xddot_with_zero_force_pid]:
            if self.robot_controller_mode == RCM.xddot_with_zero_force_pid:
                assert len(action) == 7, action.shape
                pose = action[:6]
                forces = np.zeros(6)  # compliant
            else:
                assert len(action) == 13, action.shape
                # interpret action as pos + orn(eul) + forces (+ grip)
                pose, forces = action[:6], action[6:12]
            # forces come in at the wrist f/t sensor, convert to world frame
            ee_link_frame = self.robot.get_end_effector_frame()
            # desired_ee_frame = CoordinateFrame.from_pose(pose, world_frame_3D)
            # q_ik = self.robot.compute_frame_ik(self.robotEndEffectorIndex, desired_ee_frame, rest_pose=self.robot.rp)
            offset_frame = world_frame_3D.apply_a_to_b(ee_link_frame,
                                                       self.robot.get_joint_frame(self.robot.wrist_joint_index))
            current_force = self.robot.wrist_joint_ft

            # multiple controllers added together to set xddot
            force_inps = d(desired=forces, current=current_force)
            posture_q_inps = d(desired=np.concatenate(
                [self.get_initial_joint_positions(), [0] * self.robot.num_controlled_gripper_joints]),
                current=self.robot.get_joint_values()[self.robot.dof_joints])
            posture_qdot_inps = d(desired=np.zeros(self.robot.num_dof_joints),
                                  current=self.robot.get_joint_values(velocities=True)[self.robot.dof_joints])

            if self.robot_controller_mode in [RCM.xddot_with_force_pid, RCM.xddot_with_zero_force_pid]:
                # compute xddot feedback term using the inputs (e.g. pose setpoint)

                # set x to desired
                x_inps = d(desired=pose, current=ee_link_frame.as_pose(world_frame_3D))

                # set vel to zero
                vel = np.concatenate([self.robot.get_end_effector_vel(), self.robot.get_end_effector_ang_vel()])

                xdot_inps = d(desired=np.zeros(6), current=vel)
                self.robot.os_force_control(self.robotEndEffectorIndex, force_inps, x_inps, xdot_inps,
                                            posture_q_inps, posture_qdot_inps,
                                            offset_frame=offset_frame, inertia_compensation=True, uncoupled=False,
                                            grip_pos=grip)
            elif self.robot_controller_mode == RCM.direct_xddot_with_force_pid:
                # assume pose is pose_ddot, e.g. direct inputs to xddot
                self.robot.os_force_control(self.robotEndEffectorIndex, force_inps,
                                            posture_q_inputs=posture_q_inps, posture_qdot_inputs=posture_qdot_inps,
                                            offset_frame=offset_frame, xddot_direct=pose, inertia_compensation=True,
                                            uncoupled=False,
                                            grip_pos=grip)

    # def robotCurrentStatus(self):
    #     self.curr_pos = self.robot.getEndEffectorPos()
    #     self.curr_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
    #     self.curr_gripper_pos = self.robot.getGripperPos()
    #     self.curr_status = np.array(
    #         [self.curr_pos[0], self.curr_pos[1], self.curr_pos[2], self.curr_orn[0], self.curr_orn[1], self.curr_orn[2],
    #          0.0]).reshape((-1,))
    #     return self.curr_status

    def taskColliDet(self):
        robot_links = self.robot.collision_detection_links()
        for rl in robot_links:
            table_coll_pts = p.getContactPoints(bodyA=self.robotId, bodyB=self.table_id,
                                                linkIndexA=rl, physicsClientId=self.id)
            if len(table_coll_pts) > 0:
                return True
        return False

    # def get_reward(self, seg=None):
    #     if self.cReward:
    #         if self.classifier in ('video', 'image'):
    #             return self.get_video_reward(seg)
    #         elif self.classifier == 'tsm_video':
    #             return self.get_tsm_video_reward()
    #     else:
    #         return float(self.get_success(seg))

    def is_gripping(self, object: d):
        # false if any grip link not in contact with the object
        for lnk in [self.robot.gripper_left_tip_index, self.robot.gripper_right_tip_index]:
            contact_pts = p.getContactPoints(bodyA=self.robotId, bodyB=object.id,
                                             linkIndexA=lnk, physicsClientId=self.id)
            if contact_pts is None or len(contact_pts) == 0:
                return False
        return True

    def get_success(self, seg=None):
        return False

    # def get_handcraft_reward(self, seg=None):
    #     return 1.0, False, None
