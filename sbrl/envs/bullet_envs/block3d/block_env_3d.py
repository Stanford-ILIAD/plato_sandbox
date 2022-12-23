#!/usr/bin/env python3
import argparse
## plotting forces
import multiprocessing as mp
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.robot import BulletRobot
from sbrl.envs.bullet_envs.robot_bullet_env import RobotBulletEnv
from sbrl.envs.bullet_envs.utils_env import RobotControllerMode, get_view
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers.pid_controller import ControlType
from sbrl.policies.controllers.robot_config import os_torque_control_panda_cfg
from sbrl.utils import control_utils
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D, clip_ee_orientation_conical
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.pybullet_utils import draw_cyl_gui, draw_point_gui
from sbrl.utils.python_utils import AttrDict as d, get_with_default, get_required, timeit
from sbrl.utils.script_utils import is_next_cycle

RCM = RobotControllerMode
CT = ControlType


class BlockEnv3D(RobotBulletEnv):
    def _init_params_to_attrs(self, params: d):
        # set new defaults
        params.debug_cam_dist = get_with_default(params, "debug_cam_dist", 0.8)
        params.debug_cam_p = get_with_default(params, "debug_cam_p", -35)
        params.debug_cam_y = get_with_default(params, "debug_cam_y", 80)
        params.debug_cam_target_pos = get_with_default(params, "debug_cam_target_pos", [0.2, 0, 0.4])
        super(BlockEnv3D, self)._init_params_to_attrs(params)

        self.num_blocks = get_with_default(params, "num_blocks", 1)
        assert self.num_blocks <= 1, "Not implemented #block > 1"

        self.resources_dir = os.path.join(self.asset_directory, 'sth')
        assert os.path.exists(self.resources_dir)
        # assert self.skip_n_frames_every_step == 1, "Skipping frames doesn't work with f/t sensors"

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
            for pr in [self.x_controller_params, self.posture_q_controller_params,
                       self.xdot_controller_params, self.posture_qdot_controller_params, self.force_controller_params]:
                pr.dt = self.time_step if self._control_inner_step else self.dt
        else:
            self.robot_controller_params = get_with_default(params, "robot_controller_params", d(type=CT.PID, dim=6,
                                                                                                 difference_fn=control_utils.pose_diff_fn))  # pose otherwise
            self.robot_controller_params.dt = self.time_step if self._control_inner_step else self.dt

        if not self.robot_params.has_leaf_key('resources_dir'):
            self.robot_params.resources_dir = self.resources_dir

        view_params = get_with_default(params, "view", d(view_point="third"))
        self.view_matrix, self.proj_matrix = get_view(view_params)
        self.q_home = np.array((0., -np.pi / 6., 0., -5. / 6. * np.pi, 0., 2. / 3. * np.pi, 0.))
        self._initial_joint_positions = np.zeros(7)
        # assert self._use_gravity TODO put back

        # robot start range
        self._do_random_ee_position = get_with_default(params, "do_random_ee_position", False)

        self._clip_ee_ori = get_with_default(params, "clip_ee_ori", False)
        self._draw_ee_pointer = get_with_default(params, "draw_ee_pointer", self._render)  # no GUI default

        self.num_resets = 0
        self._reinit_objects_on_reset = get_with_default(params, "reinit_objects_on_reset",
                                                         True)  # needed for randomization
        self.reset_full_every_n_resets = get_with_default(params, "reset_full_every_n_resets", 2,
                                                          map_fn=int)  # memory leaks otherwise (should really be a func of env steps) :(

        self.table_friction_ceof = get_with_default(params, "table_friction_coef", 0.4)
        self.table_z = get_with_default(params, "table_z", 0.)
        self.table_xy_offset = get_with_default(params, "table_xy_offset", np.zeros(2))

        # ADD TO THIS if you subclass
        self.objects = []

        # this defines what object types are loaded in the env (and the order for obs.objects)
        self._object_spec = get_with_default(params, "object_spec", ["block"])

        self.obj_friction_ceof = get_with_default(params, "object_friction_coef", 2.0)  # 200.
        self.obj_other_friction_coef = get_with_default(params, "object_other_friction_coef", 2.0)  # 200.
        self.obj_linear_damping = get_with_default(params, "object_linear_damping", 1.0)
        self.obj_angular_damping = get_with_default(params, "object_angular_damping", 1.0)
        self.obj_contact_stiffness = get_with_default(params, "object_contact_stiffness", 100.0)  # 1.
        self.obj_contact_damping = get_with_default(params, "object_contact_damping", 0.9)

        # types map to underlying loaders; [box, mug] are implemented
        self._object_sources: dict = get_with_default(params, "object_sources", {
            'block': "box",  # this means generate a geom_box.
            'mug': "mug",  # this means generate a mug.
        })

        self._object_shape_bounds: dict = get_with_default(params, "object_shape_bounds", {
            'block': (np.array([0.025, 0.025, 0.025]), np.array([0.053, 0.053, 0.053])),  # m
            'mug': (np.array([0.05]), np.array([0.09])),  # mug scale can only be randomized equally along each axis
        })
        self._object_mass_bounds = get_with_default(params, "object_mass_bounds", {
            'block': (0.05, 0.15),  # kg
            'mug': (0.01, 0.05),  # kg
        })

        # proportional to the surface bounds (table)
        self._object_start_bounds = get_with_default(params, "object_start_bounds", {
            'block': (np.array([-0.7, -0.7]), np.array([0.7, 0.7])),
            'mug': (np.array([-0.7, -0.7]), np.array([0.7, 0.7]))
            # % [-1, 1] is the range for each, object initializes inside this percentage of w/2
        })

        self._object_rotation_bounds = get_with_default(params, "object_rotation_bounds", {
            'block': (-np.pi / 4, np.pi / 4),  # radians
            'mug': (-np.pi / 4, np.pi / 4)  # radians
        })

        # only needed for urdf's
        self._object_base_scaling: dict = get_with_default(params, "object_base_scaling",
                                                           {'block': None, 'mug': 10. * 0.7 / 0.76})

        self._fixed_object_pose = get_with_default(params, "fixed_object_pose", False)

    def _init_setup(self):
        super(BlockEnv3D, self)._init_setup()
        p.setPhysicsEngineParameter(numSolverIterations=40, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=40, physicsClientId=self.id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.id)
        self.env_step = 0
        self.success_flag = False

        if "mug" in self._object_spec:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.obs_history = d()
        # each element is a tuple
        self.object_vis_col_shapes = d()

        self._ee_pointer_cyl = None
        self._ee_pointer_pt = None

    def _load_robot(self, presets: d = d()):
        self.robot = self.robot_model(self.id, self.robot_params.leaf_copy())
        self.robotId = self.robot.robot_id
        self.robotEndEffectorIndex = self.robot.end_effector_index

        base_ee_frame = self.robot.get_end_effector_frame()  # ee
        base_grip_pos = self.robot.get_gripper_tip_pos()  # center of two grippers
        base_grip_frame = CoordinateFrame(world_frame_3D, base_ee_frame.rot.inv(), base_grip_pos)
        self.tip_in_ee_frame = world_frame_3D.apply_a_to_b(base_ee_frame, base_grip_frame)
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

    def _load_assets(self, presets: d = d()):
        """ loading various assets into scene """

        """ TABLE & CABINET """
        self.load_surfaces()

        """ OBJECTS for the scene """
        if not self._reinit_objects_on_reset:
            # load once
            self._load_asset_objects(presets)

    def load_surfaces(self):
        table_path = os.path.join(self.resources_dir, 'urdf/table/table.urdf')
        self.table_id = p.loadURDF(table_path, [0.42, -0.1, self.table_z], [0, 0, np.pi * 0.32, 1],
                                   globalScaling=0.6, physicsClientId=self.id)
        texture_path = os.path.join(self.resources_dir, 'textures/wood.png')
        self.table_textid = p.loadTexture(texture_path, physicsClientId=self.id)
        p.changeVisualShape(self.table_id, -1, textureUniqueId=self.table_textid, physicsClientId=self.id)

        self.table_aabb = self.get_aabb(self.table_id)

        # # cabinet
        cabinet_texture = os.path.join(self.resources_dir, 'textures/wood.png')
        self.cabinet_textid = p.loadTexture(cabinet_texture, physicsClientId=self.id)

        def create(halfExtents, location, texUid=self.cabinet_textid):
            colcubeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=self.id)
            visplaneId = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=self.id)
            block = p.createMultiBody(0.0, colcubeId, visplaneId, location, physicsClientId=self.id)
            p.changeVisualShape(block, -1, textureUniqueId=texUid, physicsClientId=self.id)
            return block

        self._create_cabinet_fn = create

        # edges
        table_lens = self.table_aabb[1, :2] - self.table_aabb[0, :2]
        table_lens[1] = table_lens[1] / 2  # half of the length for y-axis, based on robot reachability
        table_center = self.table_aabb[1, :2] - table_lens / 2
        table_center = np.append(table_center, self.table_aabb[1, 2])
        table_center[:2] += self.table_xy_offset
        h = 0.08

        w1 = create(halfExtents=[table_lens[0] / 2, 0.01, h],
                    location=table_center + np.array([0, table_lens[1] / 2, h / 2]))
        w2 = create(halfExtents=[table_lens[0] / 2, 0.01, h],
                    location=table_center + np.array([0, -table_lens[1] / 2, h / 2]))
        w3 = create(halfExtents=[0.01, table_lens[1] / 2, h],
                    location=table_center + np.array([table_lens[0] / 2, 0, h / 2]))
        w4 = create(halfExtents=[0.01, table_lens[1] / 2, h],
                    location=table_center + np.array([-table_lens[0] / 2, 0, h / 2]))

        self.cabinet_obj_ids = [w1, w2, w3, w4]

        self.surface_bounds = table_lens  # 2D
        self.surface_center = table_center  # 3D
        self.free_surface_center = self.surface_center.copy()

        # print(self.table_aabb, table_center, table_lens)

        # # TableTop
        # width = 0.35
        # base = np.array([0.65, -width + 0.4, 0.5])
        # orn = R.from_euler("z", np.pi/2)
        #
        # # create(halfExtents=[width, 0.28, 0.005], location=base + np.array([0, 0.25, -0.03]))
        # # Cabinet back
        # create(halfExtents=np.abs(orn.apply([width, 0.01, 0.235])), location=base + orn.apply(np.array([0., 0.52, -0.00])))
        # # Cabinet top
        # width = 0.37
        # create(halfExtents=np.abs(orn.apply([width, 0.065, 0.005])), location=base + orn.apply(np.array([0., 0.45, 0.24])))
        # # Cabinet sides
        # width = 0.03
        # create(halfExtents=np.abs(orn.apply([width, 0.065, 0.235])), location=base + orn.apply(np.array([-0.34, 0.45, -0.00])))
        # create(halfExtents=np.abs(orn.apply([width, 0.065, 0.235])), location=base + orn.apply(np.array([0.34, 0.45, -0.00])))

    def _load_asset_objects(self, presets: d = d()):
        obj_presets = presets < ["objects"]  # only consider first object

        for i, on in enumerate(self._object_spec):
            self.objects.append(self.init_obj(d(object_cls=self._object_sources[on], object_name=on),
                                              presets=obj_presets.leaf_apply(lambda arr: arr[i])))

        # self.object = self.init_obj(d(object_cls='bottle_b1', object_scaling=self.scaling))
        # for name, source in self.object_sources.items():
        #     self.init_obj(d(object_cls=, object_scaling=self._object_base_scaling[name]))
        #
        # p.changeVisualShape(self.object.id, -1, rgbaColor=[1., 0., 0., 1], physicsClientId=self.id)
        #
        # self.table_x_offsets = np.asarray([0.075 * self.object.scaling, -0.075 * self.object.scaling])
        # self.table_y_offsets = np.asarray([0.075 * self.object.scaling + 0.25, -0.075 * self.object.scaling])
        #
        # # registering
        # self.objects.append(self.object)

    def _load_dynamics(self, presets: d = d()):
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

        for cab_obj_id in self.cabinet_obj_ids:
            p.changeDynamics(cab_obj_id, -1,
                             lateralFriction=self.table_friction_ceof,
                             rollingFriction=self.table_friction_ceof,
                             spinningFriction=self.table_friction_ceof,
                             contactStiffness=1.0,
                             contactDamping=0.9,
                             physicsClientId=self.id)
            # turn off collisions
            for l1 in range(self.get_num_links(self.robot.robot_id)):
                p.setCollisionFilterPair(self.robot.robot_id, cab_obj_id, l1, -1, 0, physicsClientId=self.id)

        self.robot.set_gripper_max_force(0.05)  # not very strong

        # self.robot.set_arm_max_force(1000.0)
        arm_damp = [0.1, 0.05, 0.04, 0.01, 0.01, 0.01, 0.01]
        self.robot.set_joint_damping(arm_damp + [0.01] * 6)

        for obj in self.objects:
            p.changeDynamics(obj >> "id", -1, mass=obj >> "mass",
                             lateralFriction=self.obj_friction_ceof,
                             # rollingFriction=self.obj_other_friction_coef,
                             # spinningFriction=self.obj_other_friction_coef,
                             # linearDamping=self.obj_linear_damping,
                             # angularDamping=self.obj_angular_damping,
                             # contactStiffness=self.obj_contact_stiffness,
                             # contactDamping=self.obj_contact_damping,
                             physicsClientId=self.id)

    def set_state(self, obs):
        pass

    def init_obj(self, obj_params: d, presets: d = d()):
        object_cls, object_name = obj_params.get_keys_required(['object_cls', 'object_name'])
        # object_scale, = obj_params.get_keys_optional(['object_scaling'], [None])

        obj = d(
            cls=object_cls,
            name=object_name,
            file=None,
            position=None,
            orientation=None,
            scaling=None,
            mass=0.,
            id=None,
        )

        if object_cls in ["box", "mug"]:
            # predefined geometry
            if self._fixed_object_pose:
                pos = np.zeros(2)  # center
            else:
                pos = np.random.uniform(self._object_start_bounds[object_name][0],
                                        self._object_start_bounds[object_name][1])
            # pos = [0.7, -0.7]
            pos = pos * self.surface_bounds / 2

            size = np.random.uniform(self._object_shape_bounds[object_name][0],
                                     self._object_shape_bounds[object_name][1])
            mass = np.random.uniform(self._object_mass_bounds[object_name][0], self._object_mass_bounds[object_name][1])

            if self._fixed_object_pose:
                yaw_rot = 0.  # no rotation
            else:
                yaw_rot = np.random.uniform(self._object_rotation_bounds[object_name][0],
                                            self._object_rotation_bounds[object_name][1])

            obj.scaling = get_with_default(presets, "objects/size", np.broadcast_to(size, (3,)))
            obj.position = get_with_default(presets, "objects/position",
                                            np.append(pos, obj.scaling[2] / 2) + self.surface_center).reshape((3,))
            if presets.has_leaf_key("objects/orientation"):
                obj.orientation = np.asarray(presets >> "objects/orientation").reshape((4,))
            elif presets.has_leaf_key("objects/orientation_eul"):
                obj.orientation = np.asarray(
                    p.getQuaternionFromEuler(list(presets >> "objects/orientation_eul"))).reshape((4,))
            else:
                obj.orientation = R.from_rotvec([0., 0., yaw_rot]).as_quat()
            obj.mass = get_with_default(presets, "objects/mass", mass)

            if object_cls == "box":
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj.scaling / 2, physicsClientId=self.id)
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=obj.scaling / 2, physicsClientId=self.id)
                obj.id = p.createMultiBody(obj.mass, col_shape, vis_shape, obj.position, obj.orientation,
                                           physicsClientId=self.id)
                # p.changeVisualShape(block, -1, textureUniqueId=texUid, physicsClientId=self.id)
            elif object_cls == "mug":
                assert obj.scaling[0] == obj.scaling[1] == obj.scaling[2], "Cannot reshape mug like that"
                # TODO scaling properly
                obj.id = p.loadURDF("objects/mug.urdf", obj.position, obj.orientation,
                                    globalScaling=obj.scaling[0] * self._object_base_scaling[object_name],
                                    physicsClientId=self.id)
        else:
            raise NotImplementedError

        # if object_cls == 'bottle':
        #     obj.file = os.path.join(self.resources_dir, "urdf/objmodels/urdfs/bottle1.urdf")
        #     obj.position = [0.4, -0.15, 0.42]
        #     obj.orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        #     obj.scaling = 1.4 if object_scale is None else object_scale
        #     obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
        #                         baseOrientation=obj.orientation,
        #                         globalScaling=obj.scaling, physicsClientId=self.id)
        #
        # if object_cls == 'cup':
        #     obj.file = os.path.join(self.resources_dir, "urdf/objmodels/urdfs/cup.urdf")
        #     obj.position = [0.45, -0.18, 0.34]
        #     obj.orientation = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
        #     obj.scaling = 0.11 if object_scale is None else object_scale
        #     obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
        #                         baseOrientation=obj.orientation,
        #                         globalScaling=obj.scaling, physicsClientId=self.id)
        #
        # if object_cls == 'nut':
        #     obj.file = os.path.join(self.resources_dir, "urdf/objmodels/nut.urdf")
        #     obj.position = [0.4, -0.15, 0.34]
        #     obj.scaling = 2 if object_scale is None else object_scale
        #     obj.orientation = p.getQuaternionFromEuler([np.pi / 2, -np.pi / 2, 0])
        #     obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
        #                         baseOrientation=obj.orientation,
        #                         globalScaling=obj.scaling, physicsClientId=self.id)
        #     p.changeVisualShape(obj.id, -1, rgbaColor=[0.3, 0.3, 0.9, 1], physicsClientId=self.id)
        #
        # if object_cls == 'bottle_b1':
        #     obj.file = os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf")
        #     obj.position = [0.4, -0.15, 0.42]
        #     obj.orientation = p.getQuaternionFromEuler([0, 0, 0])
        #     obj.scaling = 1.4 if object_scale is None else object_scale
        #     obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
        #                         baseOrientation=obj.orientation,
        #                         globalScaling=obj.scaling, physicsClientId=self.id)
        #
        # if object_cls == 'drawer':
        #     obj.file = os.path.join(self.resources_dir, "urdf/obj_libs/drawers/d4/d4.urdf")
        #     obj.position = [0.38, 0.0, 0.35]
        #     obj.orientation = p.getQuaternionFromEuler([np.pi / 2.0, 0, np.pi])
        #     obj.scaling = 1.0 if object_scale is None else object_scale
        #     obj.id = p.loadURDF(fileName=obj.file, basePosition=obj.position,
        #                         baseOrientation=obj.orientation, useFixedBase=1,
        #                         globalScaling=obj.scaling, physicsClientId=self.id)

        # the hard-coded frame for the object to start
        obj.base_frame = CoordinateFrame(world_frame_3D, R.from_quat(obj.orientation).inv(), np.array(obj.position))

        return obj

    def is_obj_upright(self, obj_idx: int):
        obj: d = self.objects[obj_idx]
        if (obj >> "cls") != "mug":
            return True

        _, orn = p.getBasePositionAndOrientation(obj >> "id", physicsClientId=self.id)
        axis, angle = p.getAxisAngleFromQuaternion(orn)
        xy_mag = np.linalg.norm(axis[:2])
        return xy_mag < 0.2 or angle < 0.1

    def is_obj_contacting_walls(self, obj_id, dir=None):
        # if dir is not None:
        #     dir = dir[:2]
        for cab_id in self.cabinet_obj_ids:
            cpts = p.getContactPoints(obj_id, cab_id, physicsClientId=self.id)
            if len(cpts) > 0:
                if dir is None:
                    return True
                else:
                    min_dist_idx = np.argmin([c[8] for c in cpts])
                    # print(np.asarray(cpts[min_dist_idx][7]))
                    contact_normal = np.asarray(cpts[min_dist_idx][7])  # [:2]
                    # if dir and contact_normal are
                    cos_theta = np.dot(contact_normal, dir) / (np.linalg.norm(contact_normal) * np.linalg.norm(dir))
                    # if the angle between cabinet -> object is within 45 on either side of movement dir, we are not in contact.
                    if np.abs(np.arccos(cos_theta)) > 90 * np.pi / 180:
                        return True
        return False

    def is_robot_contacting_table(self):
        cpts = p.getContactPoints(self.robotId, self.table_id, physicsClientId=self.id)
        return len(cpts) > 0

    def _get_obs(self, ret_images=False, seg_flag=False, **kwargs):
        assert not seg_flag or ret_images

        with timeit("env/get_obs/robot"):
            obs = d()
            obs.combine(self.robot.get_obs())

            # print(obs.gripper_pos)

            # reward terms
            # obs.success = np.asarray(self.get_success(seg=seg_flag))[None, None]
            # obs.table_collision = np.asarray(self.is_outside_bounds())[None, None]
            # obs.task_collision = np.asarray(self.taskColliDet())[None, None]
            obs.robot_collision = np.asarray(self.robot.is_colliding())[None, None]

        if ret_images:
            # get images
            with timeit("env/get_obs/image"):
                img_info = p.getCameraImage(width=self.img_width,
                                            height=self.img_height,
                                            viewMatrix=self.view_matrix,
                                            projectionMatrix=self.proj_matrix,
                                            shadow=True,
                                            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if seg_flag else p.ER_NO_SEGMENTATION_MASK,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
                obs.image = img_info[2][None, :, :, :3][..., ::-1]  # rgb image

            if seg_flag:
                obs.seg = img_info[4][None]  # seg map

        with timeit("env/get_obs/objects"):
            # obs.leaf_apply(lambda arr: arr.shape).pprint()
            obs.objects = d.leaf_combine_and_apply(
                [self._get_object_obs(obj).leaf_apply(lambda arr: arr[:, None]) for obj in self.objects],
                lambda vs: np.concatenate(vs, axis=1))

        return obs.leaf_filter_keys(
            self.env_spec.observation_names + self.env_spec.output_observation_names + self.env_spec.param_names + self.env_spec.final_names)

    def _step_simulation(self):
        super(BlockEnv3D, self)._step_simulation()
        self.robot.update_state_in_between_step(self.time_step)

        if self._draw_ee_pointer:
            length, rad = 0.4, 0.001
            gripper_tip = self.robot.get_gripper_tip_pos().reshape(3)
            ee_pos = self.robot.get_end_effector_pos().reshape(3)
            ee_ori = self.robot.get_end_effector_orn().reshape(4)
            start_pos = ee_pos + R.from_quat(ee_ori).apply(np.array([0., 0., length / 2]))

            if self._ee_pointer_cyl is None:
                _, self._ee_pointer_cyl = draw_cyl_gui(start_pos, ee_ori, radius=0.003, length=0.4,
                                                       color=[1., 0., 0., 1.], client=self.id)
                _, self._ee_pointer_pt = draw_point_gui(gripper_tip, color=[1., 0., 0., 1.], client=self.id,
                                                        radius=0.01)
            else:
                p.resetBasePositionAndOrientation(self._ee_pointer_cyl, start_pos, ee_ori, physicsClientId=self.id)
                p.resetBasePositionAndOrientation(self._ee_pointer_pt, gripper_tip, [0, 0, 0, 1.],
                                                  physicsClientId=self.id)

    def _after_step_simulation(self):
        super(BlockEnv3D, self)._after_step_simulation()
        self.robot.update_state_after_step(self.dt)

    def clear_gui_elements(self):
        super(BlockEnv3D, self).clear_gui_elements()
        if self._ee_pointer_cyl is not None:
            p.removeBody(self._ee_pointer_cyl, physicsClientId=self.id)
            p.removeBody(self._ee_pointer_pt, physicsClientId=self.id)
            self._ee_pointer_cyl = None
            self._ee_pointer_pt = None

    def reset_robot(self, presets: d = d()):
        ee_frame = None
        joints = None
        if presets.has_leaf_key("joint_positions"):
            joints = presets >> "joint_positions"
        elif presets.has_leaf_key("ee_position") and presets.has_leaf_key("ee_orientation_eul"):
            ee_pos = presets >> "ee_position"
            ee_orn = presets >> "ee_orientation_eul"
            ee_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", ee_orn).inv(), ee_pos)
        elif self._do_random_ee_position:
            low, high = self._object_start_bounds[self._object_spec[0]]
            table_max = np.append(high * self.surface_bounds / 2, 0) + self.surface_center
            table_min = np.append(low * self.surface_bounds / 2, 0) + self.surface_center
            ee_pos = np.random.uniform(table_min, table_max)
            ee_pos[2] += 0.35  # z above table
            # ee_pos = np.asarray([0.6, 0.04657779, 0.6389998])
            ee_orn = np.array([np.pi, 0, -np.pi / 2])  #
            ee_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", ee_orn).inv(), ee_pos)

        self.robot.reset(joints=joints, ee_frame=ee_frame)
        # self.init_grasp()
        self.set_initial_joint_positions()  # recording where we start

    def set_initial_joint_positions(self, q=None):
        if q is None:
            q = self.robot.get_joint_values()[self.robot.controlled_arm_joints]
        self._initial_joint_positions[:] = q

    def cleanup(self):
        if self._reinit_objects_on_reset:
            for o in self.objects:
                if o.id is not None:
                    p.removeBody(o.id, physicsClientId=self.id)

            self.objects.clear()  # none of the objects are valid

    def pre_reset(self, presets: d = d()):
        # occasionally we need to actually reset and reload assets to clear some pybullet caches
        self.num_resets += 1
        if is_next_cycle(self.num_resets, self.reset_full_every_n_resets):
            logger.warn("Reloading environment from scratch!")
            p.resetSimulation(physicsClientId=self.id)
            self.objects.clear()
            self.load()

    def reset_obj(self, obj: d, frame: CoordinateFrame = None, presets: d = d()):
        object_cls, obj_id = obj.get_keys_required(['cls', 'id'])

        # default is reset the obj to its base frame
        if frame is None:
            frame = obj.base_frame

        # TODO presets

        p.resetBasePositionAndOrientation(obj_id, frame.pos, frame.orn, physicsClientId=self.id)
        p.resetBaseVelocity(obj_id, np.zeros(3), np.zeros(3), physicsClientId=self.id)

    def reset_dynamics(self, presets: d = d()):
        object_mass = get_with_default(presets, "objects/mass", np.array([0.1] * len(self.objects))).reshape(-1)
        for obj, m in zip(self.objects, object_mass):
            p.changeDynamics(obj.id, -1, mass=m, physicsClientId=self.id)

    def reset_assets(self, presets: d = d()):
        super(BlockEnv3D, self).reset_assets(presets)

        if self._reinit_objects_on_reset:
            self._load_asset_objects(presets)
        else:
            for i, obj in enumerate(self.objects):
                self.reset_obj(obj, presets=(presets < ["objects"]).leaf_apply(lambda arr: arr[i]))
        #
        # if presets.has_leaf_key('object_position'):
        #     pos = np.asarray(presets >> 'object_position')
        # else:
        #     # base_frame will be set again for our object
        #     pos = np.asarray([0.3637 + 0.06, -0.05, 0.34])
        #     if self.randomize_object_start_location:
        #         robot_base_frame = self.robot.get_link_frame(0)
        #         xy_min, xy_max = self.table_aabb
        #         xy_min = xy_min.copy()
        #         xy_max = xy_max.copy()
        #         xy_min[0] += self.table_x_offsets[0]
        #         xy_min[1] += self.table_y_offsets[0]
        #         xy_max[0] += self.table_x_offsets[1]
        #         xy_max[1] += self.table_y_offsets[1]
        #         while True:
        #             pos[:2] = np.random.uniform(xy_min, xy_max)[
        #                       :2]
        #             err = np.linalg.norm(pos[:2] - robot_base_frame.pos[:2])
        #             if err > 0.3:
        #                 # print(err)
        #                 break
        #
        # if presets.has_leaf_key('object_rotation_angle'):
        #     rot = float(presets >> 'object_rotation_angle')
        # else:
        #     rot = 0 if self.rotation_noise == 0. else np.random.normal(0, self.rotation_noise)
        #
        # orn = R.from_rotvec([0, 0, rot])

        # self.set_obj_pose(self.object, pos, orn.as_quat().tolist(),  # [0, 0, -0.1494381, 0.9887711]
        #                   assign_base_frame=True)
        #
        # if self.start_in_grasp:
        #     self.init_grasp(self.object, presets)

    # def init_motion(self):
    #     self.data_q = np.load(os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
    #     self.data_gripper = np.load(self.configs_dir + '/init/gripper.npy')
    #     self.robot.setJointValue(self.data_q[0], gripper=self.data_gripper[0])

    def init_grasp(self, object: d, presets: d):
        # open, down facing

        # go above object
        above = np.asarray(object.position).copy()
        above[2] += 0.18
        ee_orn = R.from_quat(self.robot.get_end_effector_orn()).inv()
        above_frame = CoordinateFrame(world_frame_3D, ee_orn, above)
        rp = self.robot.rp
        for i in range(2):  # N chances for IK solver
            q_desired = self.robot.compute_frame_ik(self.robotEndEffectorIndex, above_frame, rest_pose=rp)
            # keep robot at q_desired
            self.robot.set_joint_values(q_desired, 0)
            self.robot.joint_position_control(self.robot.controlled_arm_joints, q_desired, gripper_pos=0)
            rp = q_desired
        # now step with gripper only
        for i in range(5):
            self.robot.gripper_control(80., step_simulation=True)  # start to close the gripper

        self.set_initial_joint_positions()  # sets to current location

    def _get_object_obs(self, object: d):
        id = object >> 'id'
        size = object >> "scaling"
        aabb_min, aabb_max = p.getAABB(id, -1, physicsClientId=self.id)
        contact_pts = p.getContactPoints(self.robotId, id, physicsClientId=self.id)
        contact = len(contact_pts) > 0
        contact_force = sum(cp[9] for cp in contact_pts)
        pos, orn = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
        vel, avel = p.getBaseVelocity(id, physicsClientId=self.id)
        return d(position=np.asarray(pos),
                 orientation=np.asarray(orn),
                 orientation_eul=np.asarray(p.getEulerFromQuaternion(orn)),
                 velocity=np.asarray(vel),
                 angular_velocity=np.asarray(avel),
                 size=np.asarray(size),
                 contact=np.asarray(contact),
                 contact_force=np.asarray(contact_force),
                 aabb=np.concatenate([aabb_min, aabb_max])).leaf_apply(lambda arr: arr[None])

    def get_initial_joint_positions(self):
        return self._initial_joint_positions

    def is_in_bounds(self, pos=None):
        table_max = np.append(0.95 * self.surface_bounds / 2, 0) + self.surface_center
        table_min = - np.append(0.95 * self.surface_bounds / 2, 0) + self.surface_center
        if pos is None:
            pos = self.robot.get_gripper_tip_pos()
        assert len(pos) == 3

        # z above table and gripper tip is within table min / max
        return np.all((pos >= table_min)[:2] & (pos <= table_max)[:2]) and pos[2] > table_min[2]

    def is_in_soft_bounds(self, pos=None):
        low, high = self._object_start_bounds[self._object_spec[0]]
        table_max = np.append(high * self.surface_bounds / 2, 0) + self.surface_center
        table_min = np.append(low * self.surface_bounds / 2, 0) + self.surface_center
        # print(pos, table_min, table_max)
        if pos is None:
            pos = self.robot.get_gripper_tip_pos()
        assert len(pos) == 3

        # z above table and gripper tip is within table min / max
        return np.all((pos >= table_min)[:2] & (pos <= table_max)[:2]) and pos[2] > table_min[2]

    def _control(self, action, **kwargs):
        # last element is always the "grip" command
        action = action.copy()
        grip = action[-1]

        # position control
        if self.robot_controller_mode == RCM.x_pid:
            assert len(action) == 7, action.shape
            # interpret action as pos + orn(eul) (+ grip)
            pos = action[:3]
            orn_eul = action[3:6]
            if self._clip_ee_ori:
                orn_eul = clip_ee_orientation_conical(orn_eul, ee_axis=np.array([0, 0, 1.]),
                                                      world_axis=np.array([0, 0, -1.]), max_theta=np.pi / 4)
            orn = R.from_euler("xyz", orn_eul)

            target_frame = CoordinateFrame(world_frame_3D, orn.inv(), pos)
            self.robot.os_position_control(self.robotEndEffectorIndex, target_frame, rest_pose=self.robot.rp,
                                           grip_pos=grip)

        elif self.robot_controller_mode == RCM.xdot_pid:
            assert len(action) == 7, action.shape
            # TODO grip here
            # interpret action as lin_vel + ang_vel(eul) (+ grip)
            lin_vel, ang_vel = action[:3], action[3:6]
            assert not self._clip_ee_ori, "not implemented"
            self.robot.os_velocity_control(self.robotEndEffectorIndex, lin_vel, angular_velocity=ang_vel)

        elif self.robot_controller_mode == RCM.xddot_pid:
            assert len(action) == 7, action.shape
            # interpret action as pos + orn(eul) + forces (+ grip)
            if self._clip_ee_ori:
                action[3:6] = clip_ee_orientation_conical(action[3:6], ee_axis=np.array([0, 0, 1.]),
                                                          world_axis=np.array([0, 0, -1.]), max_theta=np.pi / 4)
            pose = action[:6]

            # forces come in at the wrist f/t sensor, convert to world frame
            ee_link_frame = self.robot.get_end_effector_frame()
            posture_q_inps = d(desired=np.concatenate(
                [self.robot.rp, [0] * self.robot.num_controlled_gripper_joints]),
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

        elif self.robot_controller_mode in [RCM.xddot_with_force_pid, RCM.direct_xddot_with_force_pid,
                                            RCM.xddot_with_zero_force_pid]:
            if self._clip_ee_ori:
                action[3:6] = clip_ee_orientation_conical(action[3:6], ee_axis=np.array([0, 0, 1.]),
                                                          world_axis=np.array([0, 0, -1.]), max_theta=np.pi / 4)
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
                [self.robot.rp, [0] * self.robot.num_controlled_gripper_joints]),
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

    def get_safenet(self):
        # default safenet is determined by the table boundaries.
        low, high = self.table_aabb[0], self.table_aabb[1]
        return ((*low[:2], high[2] + 0.02), (*high[:2], high[2] + 0.3))


def drawProcess(queue):
    matplotlib.use('TkAgg')

    # print('here')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    xdata, ydata, ydata_g = [], [], []
    lns, lns_g = [], []
    for i in range(6):
        ln, = axes[i // 3, i % 3].plot([], [], label='raw')
        ln2, = axes[i // 3, i % 3].plot([], [], label='comp')
        if i // 3 == 0:
            title = 'force_'
        else:
            title = 'torque_'
        title += "%s" % (['x', 'y', 'z'][i % 3])
        # cpts = p.getContactPoints(bodyA=robotId, linkIndexA=18)  # 18, 19 are the finger pads
        axes[i // 3, i % 3].set_title(title)
        axes[i // 3, i % 3].legend()
        lns.append(ln)
        lns_g.append(ln2)
        ydata.append([])
        ydata_g.append([])

    def init():
        return lns + lns_g

    def update(time_frame):

        all_arr = [queue.get()]
        xdata.append(0 if len(xdata) == 0 else xdata[-1] + 1)
        while not queue.empty():
            all_arr.append(queue.get())
            xdata.append(xdata[-1] + 1)

        arr = np.stack(all_arr)
        for i in range(2):
            for j in range(3):
                y = arr[:, i * 3 + j].tolist()
                y_g = arr[:, i * 3 + j].tolist()
                ylist = ydata[i * 3 + j]
                ylist_g = ydata_g[i * 3 + j]
                ylist.extend(y)
                ylist_g.extend(y_g)
                # keep things small
                # if len(xdata) > 100:
                #     ylist.pop(0)
                #     ylist_g.pop(0)
                #     xdata.pop(0)
                # print(xdata[-10:], ylist[-10:])
                axes[i, j].set_xlim(xdata[-1] - 100, xdata[-1])
                axes[i, j].set_ylim(min(ylist[-100:] + ylist_g[-100:]) - 1, max(ylist[-100:] + ylist_g[-100:]) + 1)
                lns[i * 3 + j].set_data(xdata, ylist)
                lns_g[i * 3 + j].set_data(xdata, ylist_g)

        return lns + lns_g

    print('here2')
    ani = FuncAnimation(fig, update, interval=10,
                        init_func=init, blit=False)

    print('here3')
    plt.show()


def get_block3d_example_spec_params(NB=1, img_height=256, img_width=256, img_channels=3, no_names=False):
    prms = d(names_shapes_limits_dtypes=[
        ("image", (img_height, img_width, img_channels), (0, 255), np.uint8),
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('ee_orientation_eul', (3,), (-np.inf, np.inf), np.float32),
        ('ee_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('ee_angular_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('finger_left_contact', (1,), (False, True), np.bool),
        ('finger_right_contact', (1,), (False, True), np.bool),
        ('joint_positions', (13,), (-np.inf, np.inf), np.float32),
        ('gripper_pos', (1,), (-np.inf, np.inf), np.float32),
        ('gripper_tip_pos', (3,), (-np.inf, np.inf), np.float32),
        ('objects/position', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/orientation_eul', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/velocity', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/angular_velocity', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/size', (NB, 3), (0, np.inf), np.float32),
        ('objects/contact', (NB,), (False, True), np.bool),
        ('objects/aabb', (NB, 6), (0, np.inf), np.float32),
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),

        # TODO in a pre-defined box of allowed motion.
        ('action', (7,), (-np.inf, np.inf), np.float32),
        ('target/ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('target/ee_orientation_eul', (3,), (-2 * np.pi, 2 * np.pi), np.float32),
        ('target/gripper_pos', (1,), (0, 255.), np.float32),

        ("policy_type", (1,), (0, 255), np.uint8),
        ("policy_name", (1,), (0, 1), np.object),
        ("policy_switch", (1,), (False, True), np.bool),  # marks the beginning of a policy

    ], observation_names=[
        "wrist_ft", "ee_position", "ee_orientation_eul", "ee_velocity", "ee_angular_velocity",
        "joint_positions", "gripper_pos", "gripper_tip_pos", "finger_left_contact", "finger_right_contact",
        "objects/position", "objects/orientation_eul", "objects/velocity", "objects/angular_velocity", "objects/aabb",
        "objects/contact"
    ],
        param_names=["objects/size"],
        final_names=[],
        action_names=["action", "target/ee_position", "target/ee_orientation_eul", "target/gripper_pos", "policy_type",
                      "policy_name", "policy_switch"],
        output_observation_names=[]
    )
    if no_names:
        prms.action_names.remove('policy_name')
    return prms


def get_block3d_example_params(NB=1, render=False,
                               BLOCK_LOW=(0.025, 0.025, 0.025), BLOCK_HIGH=(0.055, 0.055, 0.055),
                               BLOCK_MASS_RANGE=(0.05, 0.15), BLOCK_ROTATION_BOUNDS=(-np.pi / 4, np.pi / 4),
                               MUG_SCALES=(0.05, 0.09), MUG_MASSES=(0.01, 0.05), MUG_ROTS=(-np.pi, np.pi),
                               better_view=False, use_mug=False):
    # 10Hz
    params = d()
    params.skip_n_frames_every_step = 5
    params.time_step = 0.02
    params.num_blocks = NB
    params.render = render
    params.compute_images = False
    params.debug = False
    params.combine(os_torque_control_panda_cfg)

    # robot start is randomized
    params.do_random_ee_position = True

    params.object_shape_bounds = {'block': (np.broadcast_to(BLOCK_LOW, (3,)), np.broadcast_to(BLOCK_HIGH, (3,))),
                                  'mug': (np.array([MUG_SCALES[0]]), np.array([MUG_SCALES[1]]))}
    params.object_mass_bounds = {'block': tuple(BLOCK_MASS_RANGE), 'mug': tuple(MUG_MASSES)}
    params.object_rotation_bounds = {'block': tuple(BLOCK_ROTATION_BOUNDS), 'mug': tuple(MUG_ROTS)}

    if use_mug:
        # no block, only mug
        params.object_spec = ['mug']

    if better_view:
        params.debug_cam_dist = 0.25
        params.debug_cam_p = -45
        params.debug_cam_y = 0
        params.debug_cam_target_pos = [0.4, 0, 0.45]

    return params


# teleop code as a test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mug', action='store_true')
    args = parser.parse_args()

    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 10000

    params = get_block3d_example_params(use_mug=args.use_mug, render=True)

    params.debug_cam_dist = 0.35
    params.debug_cam_p = -45
    params.debug_cam_y = 0
    params.debug_cam_target_pos = [0.4, 0, 0.45]

    env_spec_params = get_block3d_example_spec_params()
    env = BlockEnv3D(params, ParamEnvSpec(env_spec_params))

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
        ord(';'): np.array([0.04, 0, 0]),
        ord('\''): np.array([-0.04, 0, 0]),
    }

    observation, _ = env.reset(presets)
    # Get the position and orientation of the end effector
    target_position, target_orientation = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
    target_rpt_orientation = convert_quat_to_rpt(target_orientation)  # rpt rotation about default

    this_object = (observation >> "objects").leaf_apply(lambda arr: arr[0, 0])

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
        # queue.put(np.concatenate([observation.contact_force[0], np.zeros(3)]))
        # print(observation['joint_positions'])

        # aabb = (observation >> "objects/aabb").reshape(2, 3)
        # print(aabb[1] - aabb[0])
