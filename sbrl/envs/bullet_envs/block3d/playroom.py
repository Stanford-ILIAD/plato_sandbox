#!/usr/bin/env python3
import argparse
## plotting forces
import multiprocessing as mp
import os
import sys

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D, get_block3d_example_spec_params, \
    get_block3d_example_params
from sbrl.envs.bullet_envs.utils_env import RobotControllerMode
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers.pid_controller import ControlType
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.python_utils import AttrDict as d, get_with_default
from sbrl.utils.transform_utils import euler2quat, quat_difference

RCM = RobotControllerMode
CT = ControlType


class DrawerPlayroomEnv3D(BlockEnv3D):
    def _init_params_to_attrs(self, params: d):
        # proportional to the surface bounds (table)
        params.object_start_bounds = get_with_default(params, "object_start_bounds", {
            'block': (np.array([-0.6, -0.2]), np.array([0., 0.8]))
            # % [-1, 1] is the range for each, object initializes inside this percentage of w/2
        })

        params.object_mass_bounds = get_with_default(params, "object_mass_bounds", {
            'block': (0.005, 0.015),  # kg
            'mug': (0.005, 0.015),  # kg
        })

        # annoying
        params.reset_full_every_n_resets = get_with_default(params, "reset_full_every_n_resets", 1,
                                                            map_fn=int)

        params.object_friction_coef = 1000.
        params.object_other_friction_coef = 0.1
        params.object_linear_damping = 0.04
        params.object_angular_damping = 0.04
        params.object_contact_stiffness = 10000
        params.object_contact_damping = 200

        super(DrawerPlayroomEnv3D, self)._init_params_to_attrs(params)
        self.playroom_resources_dir = os.path.join(self.asset_directory, 'playroom')
        assert os.path.exists(self.playroom_resources_dir)

        # initialize slider at random positions. otherwise starts closed.
        self.random_init_drawer = get_with_default(params, "random_init_drawer", False)
        self.random_init_cabinet = get_with_default(params, "random_init_cabinet", False)

        # takes precedence over above^
        self.random_init_snap_drawer = get_with_default(params, "random_init_snap_drawer", False)
        self.random_init_snap_cabinet = get_with_default(params, "random_init_snap_cabinet", False)

        self.object_start_in_dcab = get_with_default(params, "object_start_in_dcab", False)

        self._use_short_cabinet = get_with_default(params, "use_short_cabinet", False)

        self._use_buttons = get_with_default(params, "use_buttons", False)

        self.drawer_max_out = 0.  # meters
        self.drawer_max_in = 0.194  # meters

        self.cabinet_max_open = 0.  # radians
        self.cabinet_max_closed = np.pi / 2 - 0.1  # radians
        if self._use_short_cabinet:
            self.cabinet_max_closed = np.pi / 2 - 0.2  # radians, less since robot might hit top of table.

        self.drawer_id = None
        self.cab_door_id = None

        self._table_offset_z = -0.1  # m in z

        self.door_id = None
        self.buttons = []
        self._button_constraints = []

    def drawer_pos_normalize(self, pos, inverse=False):
        # normalized: 0 is closed, 1 is open.
        if inverse:
            return (1 - pos) * (self.drawer_max_in - self.drawer_max_out) + self.drawer_max_out
        else:
            return 1 - (pos - self.drawer_max_out) / (self.drawer_max_in - self.drawer_max_out)

    def is_in_cabinet(self, pos=None):
        table_max = np.append(0.95 * self.cab_lens / 2, 0) + self.cab_center
        table_min = - np.append(0.95 * self.cab_lens / 2, 0) + self.cab_center
        if pos is None:
            pos = self.robot.get_gripper_tip_pos()
        assert len(pos) == 3

        # z above table and gripper tip is within table min / max
        return np.all((pos >= table_min)[:2] & (pos <= table_max)[:2]) and pos[2] > table_min[2]

    def is_in_drawer(self, pos=None):
        daabb = self.get_aabb(self.drawer_id)
        if pos is None:
            pos = self.robot.get_gripper_tip_pos()
        assert len(pos) == 3
        return np.all((pos >= daabb[0]) & (pos <= daabb[1]))

    def is_button_pressed(self, i):
        assert self._use_buttons
        b = self.buttons[i]
        bz = p.getBasePositionAndOrientation(b, physicsClientId=self.id)[0][2] - self._button_positions[i][2]
        return bz < -0.01

    def generate_free_space_point(self, obj_idx=0):
        low, high = self._object_start_bounds[self._object_spec[obj_idx]]
        lower = np.append(0.5 * low * self.surface_bounds, 0.025) + self.surface_center
        higher = np.append(0.5 * high * self.surface_bounds, 0.035) + self.surface_center
        return np.random.uniform(lower, higher)

    def generate_cabinet_space_point(self):
        c_center = self.cab_center.copy()
        return c_center + np.random.uniform([-0.04, 0.04, 0.0], [-0.02, 0.045, 0.01])

    def generate_drawer_space_point(self):
        daabb = self.get_aabb(self.drawer_id)
        d_center = np.average(daabb, axis=0)
        d_center[:2] += np.random.uniform([-0.04, 0.05], [0.04, 0.09])
        d_center[2] = daabb[1, 2]
        return d_center

    def load_surfaces(self):
        # super(PlayroomEnv3D, self).load_surfaces()
        # self.playroom_objects = p.loadMJCF(os.path.join(self.playroom_resources_dir, "playroom.xml"), flags=p.URDF_USE_IMPLICIT_CYLINDER)
        self.table_id = p.loadURDF(os.path.join(self.playroom_resources_dir, "Playground_StudyTable.urdf"),
                                   useFixedBase=True,
                                   globalScaling=0.6, basePosition=[0.3, -0.08, self._table_offset_z],
                                   baseOrientation=[0., 0., 1., 0.],
                                   physicsClientId=self.id)
        self.cabinet_obj_ids = []

        self.table_texture = p.loadTexture(
            os.path.join(self.playroom_resources_dir, "textures/Surface_wood_chipboard.jpg.png"))
        p.changeVisualShape(self.table_id, -1, textureUniqueId=self.table_texture, physicsClientId=self.id)

        self.table_aabb = self.get_aabb(self.table_id)
        y_center = 0.5 * (self.table_aabb[1, 1] + self.table_aabb[0, 1])
        y_width = 0.5 * (self.table_aabb[1, 1] - self.table_aabb[0, 1])
        self.table_aabb[0, 1] = y_center - y_width * 0.5
        self.table_aabb[1, 1] = y_center + y_width * 0.5  #

        # edges
        table_lens = self.table_aabb[1, :2] - self.table_aabb[0, :2]
        table_lens[1] = table_lens[1] / 2  # half of the length for y-axis, based on the playroom table size.
        table_center = 0.5 * (self.table_aabb[1, :2] + self.table_aabb[0, :2])
        table_center[1] += table_lens[1] / 2

        table_center = np.append(table_center, 0.655 * self.table_aabb[1, 2] + 0.345 * self.table_aabb[0, 2])

        # p.addUserDebugLine(table_center + np.array([table_lens[0] / 2, table_lens[1] / 2, 0]), table_center + np.array([table_lens[0] / 2, -table_lens[1] / 2, 0]))
        # p.addUserDebugLine(table_center + np.array([table_lens[0] / 2, -table_lens[1] / 2, 0]), table_center + np.array([-table_lens[0] / 2, -table_lens[1] / 2, 0]))
        # p.addUserDebugLine(table_center + np.array([-table_lens[0] / 2, -table_lens[1] / 2, 0]), table_center + np.array([-table_lens[0] / 2, table_lens[1] / 2, 0]))
        # p.addUserDebugLine(table_center + np.array([-table_lens[0] / 2, table_lens[1] / 2, 0]), table_center + np.array([table_lens[0] / 2, table_lens[1] / 2, 0]))

        self.surface_bounds = table_lens  # 2D
        self.surface_center = table_center  # 3D

        # free space center
        low, high = self._object_start_bounds[self._object_spec[0]]
        self.free_surface_center = self.surface_center + np.append(
            0.25 * (low * self.surface_bounds + high * self.surface_bounds), 0)

        self._tether_cstr = None

    def _load_asset_objects(self, presets: d = d()):
        super(DrawerPlayroomEnv3D, self)._load_asset_objects(presets)

        self.drawer_file = os.path.join(self.playroom_resources_dir, "Playground_Drawer.urdf")
        self.cab_door_file = os.path.join(self.playroom_resources_dir, "cabinet_door_short.urdf") \
            if self._use_short_cabinet else os.path.join(self.playroom_resources_dir, "cabinet_door.urdf")

        self.drawer_id = p.loadURDF(self.drawer_file,
                                    globalScaling=0.58, basePosition=[0.3, -0.2, 0.11 + self._table_offset_z],
                                    baseOrientation=[0., 0., 1., 0.],
                                    physicsClientId=self.id)

        self.cab_door_id = p.loadURDF(self.cab_door_file, globalScaling=0.8,
                                      basePosition=[0.62, -0.13, 0.43 + self._table_offset_z],
                                      baseOrientation=euler2quat(np.array([0., 0., np.pi])).tolist(),
                                      physicsClientId=self.id)
        # cab_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.1, 0.1])
        # self.cab_door_id = p.createMultiBody(0.0, -1, -1, [0.3,0.05,0.6], [0,0,0,1],
        #                                        linkMasses=[0.00005], linkCollisionShapeIndices=[cab_col], linkVisualShapeIndices=[-1], linkPositions=[[0,-0.,0]], linkOrientations=[[0,0,0,1]],
        #                                        linkInertialFramePositions=[[0,0,0]], linkInertialFrameOrientations=[[0,0,0,1]], linkParentIndices=[0], linkJointTypes=[p.JOINT_REVOLUTE], linkJointAxis=[[0,0,1]])

        for j in range(p.getNumJoints(self.drawer_id, physicsClientId=self.id)):
            p.changeVisualShape(self.drawer_id, j, textureUniqueId=self.table_texture, physicsClientId=self.id)

        self.cab_center = np.array([0.48, -0.17, self.surface_center[2]])
        self.cab_lens = np.array([0.33, 0.11])
        # p.addUserDebugLine(cab_center + np.array([cab_lens[0] / 2, cab_lens[1] / 2, 0]), cab_center + np.array([cab_lens[0] / 2, -cab_lens[1] / 2, 0]))
        # p.addUserDebugLine(cab_center + np.array([cab_lens[0] / 2, -cab_lens[1] / 2, 0]), cab_center + np.array([-cab_lens[0] / 2, -cab_lens[1] / 2, 0]))
        # p.addUserDebugLine(cab_center + np.array([-cab_lens[0] / 2, -cab_lens[1] / 2, 0]), cab_center + np.array([-cab_lens[0] / 2, cab_lens[1] / 2, 0]))
        # p.addUserDebugLine(cab_center + np.array([-cab_lens[0] / 2, cab_lens[1] / 2, 0]), cab_center + np.array([cab_lens[0] / 2, cab_lens[1] / 2, 0]))

        # p.changeDynamics(self.drawer_id, 1,
        #                  mass=0.005,
        #                  lateralFriction=1.,
        #                  # restitution=0.001,
        #                  # rollingFriction=self.table_friction_ceof,
        #                  # spinningFriction=self.table_friction_ceof,
        #                  # contactStiffness=10000.,
        #                  # contactDamping=200,
        #                  physicsClientId=self.id)
        # turn off collisions
        for l1 in range(self.get_num_links(self.cab_door_id)):
            for j1 in range(self.get_num_links(self.table_id)):
                p.setCollisionFilterPair(self.cab_door_id, self.table_id, l1, j1, 0, physicsClientId=self.id)

            for j1 in range(self.get_num_links(self.drawer_id)):
                p.setCollisionFilterPair(self.cab_door_id, self.drawer_id, l1, j1, 0, physicsClientId=self.id)

        if self._use_buttons:
            import pybullet_planning as pp
            color = [pp.BLUE, pp.GREEN, pp.YELLOW]
            self.buttons = [pp.create_cylinder(0.02, 0.01, mass=0.01, color=color[i]) for i in range(3)]
            button_offset = np.array([0., 0., 0.02])
            lateral_offset = np.array([0.09, 0., 0.])
            self._button_positions = [self.cab_center + button_offset, self.cab_center + button_offset + lateral_offset, self.cab_center + button_offset - lateral_offset]

            for i in range(3):
                p.resetBasePositionAndOrientation(self.buttons[i], self._button_positions[i],
                                                  [0, 0, 0, 1], physicsClientId=self.id)
                cid = p.createConstraint(self.buttons[i], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self._button_positions[i], physicsClientId=self.id)
                p.changeConstraint(cid, maxForce=2.0, physicsClientId=self.id)
                self._button_constraints.append(cid)
                for j1 in range(self.get_num_links(self.table_id)):
                    p.setCollisionFilterPair(self.buttons[i], self.table_id, -1, j1, 1, physicsClientId=self.id)

    def reset_assets(self, presets: d = d()):
        super(DrawerPlayroomEnv3D, self).reset_assets(presets)

        depth = self.drawer_max_out  # default is start open.
        if self.random_init_snap_drawer:
            depth = self.drawer_max_out if np.random.random() > 0.5 else self.drawer_max_in
        elif self.random_init_drawer:
            depth = np.random.uniform(self.drawer_max_out, self.drawer_max_in)

        depth = get_with_default(presets, "drawer/joint_position", np.array([depth])).item()

        p.resetJointState(self.drawer_id, 0, depth, 0, physicsClientId=self.id)
        p.enableJointForceTorqueSensor(self.drawer_id, 0, 1, physicsClientId=self.id)

        angle = self.cabinet_max_open  # default is start open.
        if self.random_init_snap_cabinet:
            angle = self.cabinet_max_open if np.random.random() > 0.5 else self.cabinet_max_closed
        elif self.random_init_cabinet:
            angle = np.random.uniform(self.cabinet_max_open, self.cabinet_max_closed)

        angle = get_with_default(presets, "cabinet/joint_position", np.array([angle])).item()

        p.resetJointState(self.cab_door_id, 0, angle, 0, physicsClientId=self.id)
        p.enableJointForceTorqueSensor(self.cab_door_id, 0, 1, physicsClientId=self.id)

        # TODO make these not hard coded
        self._const_handle_size = np.array([0.01, 0.04, 0.04])  # xyz on loading (y and z are diameters)
        if self._use_short_cabinet:
            self._const_cabinet_size = np.array([0.02, 0.3, 0.15])  # xyz on loading (y is the radial length)
        else:
            self._const_cabinet_size = np.array([0.02, 0.3, 0.2])  # xyz on loading (y is the radial length)

        if self._tether_cstr is not None:
            p.removeConstraint(self._tether_cstr)
            self._tether_cstr = None
        self._tether_contact_count = 0

        if self.object_start_in_dcab:
            # object preset will be in the drawer or cabinet, randomly selected.
            if not presets.has_leaf_key("objects/position"):
                assert self._reinit_objects_on_reset, "Must reinit objects to do this."
                presets = presets.leaf_copy()  # otherwise we are potentially editing the default..
                presets.objects.position = np.zeros((self.num_blocks, 3))
                for i in range(self.num_blocks):
                    offset = np.array([0., 0., 0.08])
                    in_drawer = np.random.rand() < 0.5
                    # dpos does not need the offset.
                    presets.objects.position[i] = self.generate_drawer_space_point() if in_drawer \
                        else self.generate_cabinet_space_point() + offset

                # clear just the objects
                super(DrawerPlayroomEnv3D, self).cleanup()
                # reinit just the objects again
                super(DrawerPlayroomEnv3D, self)._load_asset_objects(presets)

    def cleanup(self):
        super(DrawerPlayroomEnv3D, self).cleanup()
        if self.drawer_id is not None:
            p.removeBody(self.drawer_id, physicsClientId=self.id)
            self.drawer_id = None
        if self.cab_door_id is not None:
            p.removeBody(self.cab_door_id, physicsClientId=self.id)
            self.cab_door_id = None
        for i in range(len(self.buttons)):
            p.removeConstraint(self._button_constraints[i])
            p.removeBody(self.buttons[i], physicsClientId=self.id)

        self.buttons = []
        self._button_constraints = []

    def _control(self, action, **kwargs):

        # disable motors on revolute joints
        mode = p.VELOCITY_CONTROL
        for i in range(p.getNumJoints(self.cab_door_id)):
            p.setJointMotorControl2(self.cab_door_id, i, mode,
                                    force=0.005, physicsClientId=self.id)

        super(DrawerPlayroomEnv3D, self)._control(action)

        opos, oori = p.getBasePositionAndOrientation(self.objects[0] >> "id", physicsClientId=self.id)[:2]
        if self.is_in_drawer(opos):
            # p.applyExternalForce(self.objects[0] >> "id", -1, [0, 0, -0.05], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.id)
            cpts = p.getContactPoints(self.objects[0] >> "id", self.drawer_id)
            cpts_rob = p.getContactPoints(self.objects[0] >> "id", self.robotId)
            if len(cpts_rob) > 0:
                if self._tether_cstr is not None:
                    p.removeConstraint(self._tether_cstr)
                    self._tether_cstr = None
                self._tether_contact_count = 0
            elif len(cpts) > 0:
                self._tether_contact_count += 1

            if self._tether_contact_count > 4 * self.skip_n_frames_every_step and len(cpts_rob) == 0:
                # tether
                if self._tether_cstr is None:
                    pp, po = p.getLinkState(self.drawer_id, 1, physicsClientId=self.id)[:2]  # parent
                    cp, co = np.array(opos), np.array(oori)  # child

                    daabb = self.get_aabb(self.drawer_id)
                    center = ((daabb[0] + daabb[1]) * 0.5)[:2]
                    hl = ((daabb[1] - daabb[0]) * 0.5)[:2]
                    cp[:2] = np.clip(cp[:2], center - 0.9 * hl, center + 0.9 * hl)
                    cp[1] = max(cp[1], self.table_aabb[1, 1] + 0.02)

                    gfp = (cp - np.array(pp))
                    pfo = quat_difference(co, po).tolist()

                    pfp = R.from_quat(po).apply(gfp, inverse=True).tolist()
                    # pfp = [0, -0.05, 0]

                    self._tether_cstr = p.createConstraint(self.drawer_id, 0, self.objects[0].id, -1, p.JOINT_FIXED,
                                                           [0, 0, 1], pfp, [0, 0, 0], pfo)
                    # p.changeConstraint(self._tether_cstr, maxForce=1.)

            if self._use_buttons:
                for i in range(len(self.buttons)):
                    if self.is_button_pressed(i):
                        dz = p.getBasePositionAndOrientation(self.buttons[i], physicsClientId=self.id)[0][2] - \
                             self._button_positions[i][2]
                        p.applyExternalForce(self.buttons[i], -1, [0, 0, -500. * dz], [0,0,0], p.LINK_FRAME, physicsClientId=self.id)
            # f = np.zeros(3)
            # for pt in cpts:
            #     f += np.array(pt[11]) * pt[10]
            # logger.debug(f"FRIX: {f}")

    def _get_obs(self, **kwargs):
        obs = super(DrawerPlayroomEnv3D, self)._get_obs(**kwargs)

        dj_info = p.getJointState(self.drawer_id, 0, physicsClientId=self.id)

        aabb_min, aabb_max = p.getAABB(self.drawer_id, 1, physicsClientId=self.id)

        # contact with the robot
        contact_pts = p.getContactPoints(self.robotId, self.drawer_id, physicsClientId=self.id)
        contact = len(contact_pts) > 0
        contact_force = sum(cp[9] for cp in contact_pts)

        obs.drawer = d(
            joint_position=np.array([dj_info[0]])[None],
            # 0 = in all the way, 1 = out all the way
            joint_position_normalized=np.array([self.drawer_pos_normalize(dj_info[0])])[None],
            joint_velocity=np.array([dj_info[1]])[None],
            joint_ft=np.asarray(dj_info[2])[None],
            contact=np.asarray([contact])[None],
            contact_force=np.asarray([contact_force])[None],
            aabb=np.concatenate([aabb_min, aabb_max])[None],
        )
        obs.drawer.closed = obs.drawer.joint_position_normalized < 0.1  # closed is 10% of the max range (somewhat arbitrary)

        # cabinet
        cj_info = p.getJointState(self.cab_door_id, 0, physicsClientId=self.id)

        # cabinet handle
        chl_info = p.getLinkState(self.cab_door_id, 2, physicsClientId=self.id)

        aabb_min, aabb_max = p.getAABB(self.cab_door_id, 0, physicsClientId=self.id)

        haabb_min, haabb_max = p.getAABB(self.cab_door_id, 2, physicsClientId=self.id)

        # contact with the robot
        contact_pts = p.getContactPoints(self.robotId, self.cab_door_id, physicsClientId=self.id)
        contact = len(contact_pts) > 0
        contact_force = sum(cp[9] for cp in contact_pts)

        obs.cabinet = d(
            joint_position=np.array([cj_info[0]])[None],
            # 0 = closed all the way, 1 = open all the way
            joint_position_normalized=np.array([1 - cj_info[0] / (np.pi / 2)])[None],
            joint_velocity=np.array([cj_info[1]])[None],
            joint_ft=np.asarray(cj_info[2])[None],
            contact=np.asarray([contact])[None],
            contact_force=np.asarray([contact_force])[None],
            aabb=np.concatenate([aabb_min, aabb_max])[None],
            size=self._const_cabinet_size[None],

            handle_position=np.array(chl_info[0])[None],
            handle_orientation_eul=np.array(p.getEulerFromQuaternion(chl_info[1]))[None],
            handle_aabb=np.concatenate([haabb_min, haabb_max])[None],
            handle_size=self._const_handle_size[None],
        )

        obs.cabinet.closed = obs.cabinet.joint_position_normalized < 0.1  # closed is 10% of the max range (somewhat arbitrary)

        if self._use_buttons:
            all_contact_pts = [p.getContactPoints(self.robotId, b, physicsClientId=self.id) for b in self.buttons]
            contact = [len(cps) > 0 for cps in all_contact_pts]

            cbpos = [p.getBasePositionAndOrientation(b, physicsClientId=self.id)[0] for b in self.buttons]
            bz = [cbpos[i][2] - self._button_positions[i][2] for i in range(len(self.buttons))]
            pressed = [z < -0.01 for z in bz]
            obs.buttons = d(
                position=np.array(cbpos)[None],  # (3, 3)
                joint_position=np.array(bz)[None],  # (3,)
                closed=np.array(pressed)[None],  # (3, )
                contact=np.asarray(contact)[None],  # (3, )
            )
            # obs.buttons.pprint()

        return obs


def get_playroom3d_example_spec_params(use_buttons=False, **kwargs):
    base_prms = get_block3d_example_spec_params(**kwargs)
    return base_prms & d(
        names_shapes_limits_dtypes=base_prms.names_shapes_limits_dtypes + [
            ("drawer/joint_position", (1,), (-np.inf, np.inf), np.float32),  #
            ("drawer/joint_position_normalized", (1,), (-np.inf, np.inf), np.float32),  #
            ("drawer/joint_velocity", (1,), (-np.inf, np.inf), np.float32),  #
            ("drawer/joint_ft", (6,), (-np.inf, np.inf), np.float32),  #
            ("drawer/aabb", (6,), (-np.inf, np.inf), np.float32),  #
            ("drawer/contact_force", (1,), (-np.inf, np.inf), np.float32),  #
            ("drawer/contact", (1,), (False, True), np.bool),  #
            ("drawer/closed", (1,), (False, True), np.bool),  #
            ("cabinet/joint_position", (1,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/joint_position_normalized", (1,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/joint_velocity", (1,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/joint_ft", (6,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/aabb", (6,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/size", (3,), (0, np.inf), np.float32),  #
            ("cabinet/contact_force", (1,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/contact", (1,), (False, True), np.bool),  #
            ("cabinet/closed", (1,), (False, True), np.bool),  #
            ("cabinet/handle_position", (3,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/handle_orientation_eul", (3,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/handle_aabb", (6,), (-np.inf, np.inf), np.float32),  #
            ("cabinet/handle_size", (3,), (0, np.inf), np.float32),  #
            ("buttons/position", (3, 3), (-np.inf, np.inf), np.float32),  #
            ("buttons/joint_position", (3,), (-np.inf, np.inf), np.float32),  #
            ("buttons/closed", (3,), (False, True), np.bool),  #
            ("buttons/contact", (3,), (False, True), np.bool),  #
        ],
        observation_names=base_prms.observation_names + [
            "drawer/joint_position", "drawer/joint_position_normalized", "drawer/joint_velocity", "drawer/joint_ft",
            "drawer/closed", "drawer/aabb", "drawer/contact", "drawer/contact_force",
            "cabinet/joint_position", "cabinet/joint_position_normalized", "cabinet/joint_velocity", "cabinet/joint_ft",
            "cabinet/closed", "cabinet/aabb", "cabinet/contact", "cabinet/contact_force",
            "cabinet/handle_position", "cabinet/handle_orientation_eul", "cabinet/handle_aabb",
        ] + (['buttons/position', 'buttons/joint_position', 'buttons/closed', 'buttons/contact'] if use_buttons else []),
        param_names=base_prms.param_names + ["cabinet/size", "cabinet/handle_size"]
    )


def get_playroom3d_example_params(NB=1, use_buttons=False, better_view=False, **kwargs):
    # 10Hz
    params = get_block3d_example_params(**kwargs)

    # drawer snap
    params.random_init_snap_drawer = True
    params.random_init_snap_cabinet = True

    # buttons in the cabinets
    params.use_buttons = use_buttons

    if better_view:
        params.debug_cam_dist = 0.5
        params.debug_cam_p = -30
        params.debug_cam_y = 120
        params.debug_cam_target_pos = [0.4, 0, 0.45]

    return params


# teleop code as a test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_buttons', action='store_true')
    args = parser.parse_args()

    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 5000
    params = get_playroom3d_example_params(render=True, use_buttons=args.use_buttons)

    params.clip_ee_ori = True

    # params.random_init_snap_drawer = False
    # params.random_init_snap_cabinet = False

    env_spec_params = get_playroom3d_example_spec_params()
    env = DrawerPlayroomEnv3D(params, ParamEnvSpec(env_spec_params))

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
        ord('='): np.array([0, 0, 0.02]),
        ord('-'): np.array([0, 0, -0.02]),
        ord('['): np.array([0, 0.02, 0]),
        ord(']'): np.array([0, -0.02, 0]),
        ord(';'): np.array([0.04, 0, 0]),
        ord('\''): np.array([-0.04, 0, 0]),
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
        target_position = target_position * 0.75 + curr_pos * 0.25
        # target_rpt_orientation = target_rpt_orientation * 0.9 + curr_rpt * 0.1
        target_orientation, target_orientation_eul = convert_rpt(*target_rpt_orientation)

        # target end effector state
        # targ_frame = CoordinateFrame(world_frame_3D, R.from_quat(orientation).inv(), np.asarray(position))
        act = np.concatenate([np.asarray(target_position), np.asarray(target_orientation_eul), [grip_state * 255.]])

        observation, _, done = env.step(act)
