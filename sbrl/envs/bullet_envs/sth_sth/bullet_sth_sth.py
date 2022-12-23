"""
TODO all of this

Base environment for something 2 something envs
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from sbrl.envs.bullet_envs.bullet import RobotBulletEnv
from scipy.spatial.transform import Rotation as R

from sbrl.envs.env_spec import EnvSpec
from sbrl.experiments import logger
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import AttrDict, timeit, get_with_default, get_required
from sbrl.utils.script_utils import is_next_cycle


def get_example_params() -> AttrDict:
    return AttrDict(
        # TODO
    )


class BulletSthSth(RobotBulletEnv):

    def _init_params_to_attrs(self, params: AttrDict):
        super(BulletSthSth, self)._init_params_to_attrs(params)

        # this is the task id for sth-sth < 174
        self.task_id = get_required(params, "task_id")
        assert 0 <= self.task_id < 174

        # current state
        self.curr_state = AttrDict()
        # each item is a list of array(T, feature_size) for that state
        self.state_history = AttrDict()

        # determines action space
        self.control_mode = get_with_default(params, "control_mode", 'joint', map_fn=str).lower()
        assert self.control_mode in ['joint', 'ee']

        self.robot_params = get_required(params, "robot_params")
        self.ee_type = get_with_default(self.robot_params, "ee_type", "robotiq")

        # # controller
        # self.robot_control_mode = params.get("robot_control_mode", p.VELOCITY_CONTROL)  # todo implement this
        # self.is_ee_control = params.get("is_ee_control", True)  # todo implement this
        # self.robot_action_length = 7 if self.robot_control_mode == p.POSITION_CONTROL else 7
        # self.robot_joint_position_gains = params.get("robot_joint_position_gains", [0.005] * self.robot_action_length)
        # self.robot_joint_position_forces = params.get("robot_joint_position_forces", [50.] * self.robot_action_length)
        # self.robot_joint_velocity_scale = params.get("robot_joint_velocity_scale", 4.)

        self.fov = 60
        self.near = 0.005
        self.far = 0.1
        self.yaw, self.pitch, self.roll = 60, 0, 0

        self.num_resets = 0
        self.reset_full_every_n_resets = params.get("reset_full_every_n_resets", 20)  # memory leaks otherwise :(

        # table / robot base positions / orientations
        self.include_table = params.get("include_table", True)  # TODO

        self.table_base_position = params.get("table_base_position", [0.35, -0.9, 0])
        self.table_base_orientation_eul = params.get("table_base_orientation_eul", [0, 0, 0])

        self.robot_base_position = params.get("robot_base_position",
                                              [-0.35, -0.8, 0.7])  # [0.33, -1.2, 0.75]) # [-0.35, -0.9, 0.75])
        self.robot_base_orientation_eul = params.get("robot_base_orientation_eul", [0, 0, 0])
        self.robot_joint_default_positions = params.get("robot_joint_default_positions",
                                                        [0, -np.pi / 4, 0, -3 * np.pi / 4, np.pi / 2, np.pi / 2,
                                                         -np.pi / 4])

        self.robot_base_frame = CoordinateFrame(world_frame_3D,
                                                R.from_euler("xyz", self.robot_base_orientation_eul).inv(),
                                                np.asarray(self.robot_base_position))


        self.objects = []
        # # object urdfs to load in the scene
        # self.objects = list(get_required(params, "objects"))
        # self.num_objects = len(self.objects)
        #
        # # reset the objects in the scene
        # self.reset_objects_fn = get_required(params, "reset_objects_fn")
        # assert callable(self.reset_objects_fn), type(self.reset_objects_fn)

        # takes Env + ... -> bool related to task completion
        self.success_metric = get_with_default(params, "success_metric", BulletSthSth._get_success)
        assert callable(self.success_metric), type(self.success_metric)

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        self.im1 = self.ax[0, 0].imshow(np.zeros((self.img_width, self.img_height, 4)))
        self.im2 = self.ax[0, 1].imshow(np.zeros((self.img_width, self.img_height, 4)))
        self.im1_d = self.ax[1, 0].imshow(np.zeros((self.img_width, self.img_height, 4)), cmap='gray', vmin=0,
                                          vmax=1)
        self.im2_d = self.ax[1, 1].imshow(np.zeros((self.img_width, self.img_height, 4)), cmap='gray', vmin=0,
                                          vmax=1)

    def _load_robot(self):
        self.robot, self.robot_lower_limits, self.robot_upper_limits, self.robot_arm_indices = self.init_panda()
        self.robot_lower_limits = self.robot_lower_limits[self.robot_arm_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_arm_indices]

    def _load_assets(self):
        """
        Loads all assets into sim.
        """
        # fixed reference point for mouth
        self.sphere_vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 1, 0, 1],
                                              physicsClientId=self.id)
        self.sphere_coll = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.005, physicsClientId=self.id)
        self.sphere_vis_red = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0, 0, 1],
                                                  physicsClientId=self.id)

        """ 1. table """
        self.table = p.loadURDF(os.path.join(self.asset_directory, 'table', 'table_tall.urdf'),
                                basePosition=self.table_base_position,
                                baseOrientation=p.getQuaternionFromEuler(self.table_base_orientation_eul),
                                physicsClientId=self.id)

        """ 2. scene objects """
        self._load_objects()

        """ 3. fork """
        ee_initial_link_state = p.getLinkState(self.robot, 8, computeForwardKinematics=True,
                                               physicsClientId=self.id)

        # (4:6) is link, (0:2) is center of mass too bc pybullet is weird
        self.ee_com2link, self.ee_com_in_link = CoordinateFrame.relative_frame_a_to_b(
            R.from_quat(ee_initial_link_state[1]), np.asarray(ee_initial_link_state[0]),
            R.from_quat(ee_initial_link_state[5]), np.asarray(ee_initial_link_state[4]))
        self.base_com_relative_to_link_frame = CoordinateFrame(world_frame_3D, self.ee_com2link.inv(),
                                                               self.ee_com_in_link)

    def _load_objects(self):
        pass

    def _set_gripper(self, pos):
        if self.robotmo

    def _set_object_in_gripper(self, id, object_frame_rel_ee: CoordinateFrame):


    def _load_dynamics(self):
        self._mouth_constraint = p.createConstraint(self.mouth, -1,
                                                    -1, -1, p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=self.mouth_base_position,
                                                    parentFrameOrientation=[0, 0, 0, 1],
                                                    childFrameOrientation=p.getQuaternionFromEuler(
                                                        self.mouth_base_orientation_eul),
                                                    physicsClientId=self.id)
        p.changeConstraint(self._mouth_constraint, maxForce=500, physicsClientId=self.id)

        # Create constraint that keeps the tool in the gripper
        self._fork_in_gripper_constraint = p.createConstraint(self.robot, 8, self.drop_fork, -1, p.JOINT_FIXED,
                                                              [0, 0, 0], parentFramePosition=self.tool_pos_in_ee_com,
                                                              childFramePosition=[0, 0, 0],
                                                              parentFrameOrientation=p.getQuaternionFromEuler(
                                                                  self.tool_orient_in_ee_com),
                                                              physicsClientId=self.id)

        # disable collisions between fork and robot
        for j in list(range(p.getNumJoints(self.robot, physicsClientId=self.id))) + [-1]:
            for tj in list(range(p.getNumJoints(self.drop_fork, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(self.robot, self.drop_fork, j, tj, False, physicsClientId=self.id)

        # Disable collisions between the ref pts and robot
        for tj in list(range(p.getNumJoints(self.closest_pt_body, physicsClientId=self.id))) + [-1]:
            for ti in list(range(p.getNumJoints(self.robot, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(self.robot, self.closest_pt_body, ti, tj, False, physicsClientId=self.id)
            for ti in list(range(p.getNumJoints(self.drop_fork, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(self.drop_fork, self.closest_pt_body, ti, tj, False, physicsClientId=self.id)

    # this may be overriden
    def _load_mouth(self):
        # self.mouth = p.loadURDF(os.path.join(self.asset_directory, 'mouth', 'hole.urdf'), useFixedBase=False,
        #                         basePosition=self.mouth_base_position,
        #                         baseOrientation=[0, 0, 0, 1],
        #                         physicsClientId=self.id)

        self.mouth_center_position = self.mouth_base_position + np.array(
            [0, -0.02, -0.005])  # unique to the mouth file we are loading and the scale

        mouth_shape = [0.035, 0.035, 0.06]
        self.mouthVisualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                      fileName=os.path.join(self.asset_directory, 'mouth', 'hole2.stl'),
                                                      rgbaColor=[1, 1, 1, 1],
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0, 0, 0],
                                                      visualFrameOrientation=p.getQuaternionFromEuler(
                                                          [-np.pi / 2, 0, 0]),
                                                      meshScale=np.array(mouth_shape)[np.array([0, 2, 1])],
                                                      physicsClientId=self.id)
        self.mouthCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                            fileName=os.path.join(self.asset_directory, 'mouth',
                                                                                  'hole2_vhacd.obj'),
                                                            collisionFramePosition=[0, 0, 0],
                                                            meshScale=mouth_shape,
                                                            physicsClientId=self.id)
        self.mouth = p.createMultiBody(baseMass=0.01,
                                       baseInertialFramePosition=[0, 0, 0],
                                       baseCollisionShapeIndex=self.mouthCollisionShapeId,
                                       baseVisualShapeIndex=self.mouthVisualShapeId,
                                       basePosition=self.mouth_base_position,
                                       physicsClientId=self.id)

        p.changeVisualShape(self.mouth, -1,
                            textureUniqueId=p.loadTexture(os.path.join(self.asset_directory, 'mouth', "mouth_tex.png")))

        # self.mouth_base_vis = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
        #                                           baseVisualShapeIndex=self.sphere_vis,
        #                                           basePosition=self.mouth_base_position,
        #                                           useMaximalCoordinates=False, physicsClientId=self.id)

        # raycast helpers (for debugging)
        self.batch_ray_offsets = [[0.08, 0, 0], [-0.08, 0, 0], [0, 0, 0.05], [0, 0, -0.05]]
        self.ray_markers = [
            p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.sphere_vis,
                              basePosition=self.mouth_base_position, useMaximalCoordinates=False,
                              physicsClientId=self.id)
            for _ in range(len(self.batch_ray_offsets))]
        self.ray_markers_end = [
            p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.sphere_vis,
                              basePosition=self.mouth_base_position, useMaximalCoordinates=False,
                              physicsClientId=self.id)
            for _ in range(len(self.batch_ray_offsets))]

    def init_panda(self):
        # Enable self collisions to prevent the arm from going through the torso
        robot = p.loadURDF(os.path.join(self.asset_directory, 'panda', 'panda_model.urdf'), useFixedBase=True,
                           basePosition=self.robot_base_position,
                           baseOrientation=p.getQuaternionFromEuler(self.robot_base_orientation_eul),
                           flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)

        robot_arm_joint_indices = []
        for i in range(p.getNumJoints(robot, physicsClientId=self.id)):
            joint_info = p.getJointInfo(robot, i, physicsClientId=self.id)
            jtype = joint_info[2]
            jname = joint_info[1].decode('UTF-8')
            if jtype is p.JOINT_REVOLUTE or jtype is p.JOINT_PRISMATIC:
                robot_arm_joint_indices.append(i)
                # p.resetJointState(robot, i, positions[len(robot_arm_joint_indices)-1], physicsClientId=self.id)
                logger.debug("ADDING JOINT %s, %d" % (jname, i))
        robot_arm_joint_indices = robot_arm_joint_indices[:7]  # last joints are arm

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_arm_joint_indices

    def reset_joint_positions(self, q, **kwargs):
        for joint, value in zip(self.robot_arm_indices, q):
            p.resetJointState(self.robot, joint, value, physicsClientId=self.id)

    def reset_gripper_positions(self, pos, vel=0):
        nj = p.getNumJoints(self.robot, physicsClientId=self.id)
        p.resetJointState(self.robot, nj - 1, targetValue=pos, targetVelocity=vel, physicsClientId=self.id)
        p.resetJointState(self.robot, nj - 2, targetValue=pos, targetVelocity=vel, physicsClientId=self.id)

    def get_world_pose_relative_to_robot_base(self, pose):
        return CoordinateFrame.pose_a_view_in_b(pose, world_frame_3D, self.robot_base_frame)

    def get_robot_base_pose_relative_to_world(self, pose):
        return CoordinateFrame.pose_a_view_in_b(pose, self.robot_base_frame, world_frame_3D)

    def _get_success(self, *args, **kwargs):
        return False

    def get_object(self, i):
        return self.objects[i]

    def get_joint_indices(self):
        return self.robot_arm_indices

    def get_joint_limits(self):
        return self.robot_lower_limits, self.robot_upper_limits

    def get_initial_joint_positions(self):
        return self.robot_start_joint_positions.copy()

    # @profile
    def _get_obs(self, forces=None, ret_images=False):
        robot_right_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_arm_indices,
                                                    physicsClientId=self.id)
        robot_right_joint_positions = np.array([x[0] for x in robot_right_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        ee_pos, ee_orient = p.getLinkState(self.robot, 8, physicsClientId=self.id)[
                            4:6]  # we want offset from link frames
        # print(ee_pos, fork_pos, food_pos)

        # TODO all this is outdated, use coordinate frames
        _, ee_inverse = p.invertTransform(position=[0, 0, 0], orientation=ee_orient)
        offset = np.array(food_pos) - np.array(ee_pos)
        offset = p.rotateVector(ee_inverse, offset)

        _, ff_inverse = p.invertTransform(position=[0, 0, 0], orientation=fork_orient)
        ff_offset = np.array(food_pos) - np.array(fork_pos)
        food_in_fork_offset = p.rotateVector(ff_inverse, ff_offset)

        # print("fifo: ", food_in_fork_offset)
        # print(self.food_true_scale)

        velocity, angular_velocity = p.getBaseVelocity(self.drop_fork, physicsClientId=self.id)

        # get forces if not precomputed
        if forces is None:
            forces = self.get_total_force()  # robot, fork, food

        # RAY TRACE
        mouth_pos, mouth_orientation = p.getBasePositionAndOrientation(self.mouth, physicsClientId=self.id)
        difference = p.getDifferenceQuaternion(
            quaternionStart=p.getQuaternionFromEuler(self.mouth_base_orientation_eul),
            quaternionEnd=mouth_orientation)
        if self.debug:
            logger.debug("difference: %s" % str(difference))
        new_offsets = []
        rayEnds = []
        rayStarts = []
        for i in range(len(self.batch_ray_offsets)):
            new_offsets.append(p.rotateVector(difference, self.batch_ray_offsets[i]))
            # START OUTSIDE OF MOUTH
            rayEnds.append(np.array(food_pos))
            rayStarts.append(rayEnds[-1] + np.array(new_offsets[-1]))

        # output = p.rayTestBatch(rayStarts, rayEnds, physicsClientId=self.id)
        #
        #     if output[i][0] != -1 and output[i][0] != self.mouth:
        #         # cast again
        #         rayStarts[i] = np.array(output[i][3]) + 0.05 * np.array(
        #             self.batch_ray_offsets[i])  # start at the old intersection + 5% collision margin

        output2 = p.rayTestBatch(rayStarts, rayEnds, physicsClientId=self.id)
        # for i in range(len(self.ray_markers)):
        #     p.resetBasePositionAndOrientation(self.ray_markers_end[i], rayStarts[i], [0,0,0,1], physicsClientId=self.id)
        #     p.resetBasePositionAndOrientation(self.ray_markers[i], output2[i][3], [0,0,0,1], physicsClientId=self.id)

        insidemouth = all([output2[i][0] == self.mouth for i in range(len(output2))])
        margin = np.array([0] * len(output2))
        if insidemouth:
            margin = np.array([1. - output2[i][2] for i in
                               range(len(output2))])  # margin will be 1 when we are maximally far in this direction

        if self.debug:
            logger.debug(
                "In mouth: %s (%s) margin: %s" % (insidemouth, [output2[i][0] for i in range(len(output2))], margin))
            logger.debug("Forces: %s" % forces)

        cpts = p.getClosestPoints(bodyA=self.closest_pt_body, bodyB=self.food_item, distance=100,
                                  physicsClientId=self.id)
        i = np.argmin([cpt[8] for cpt in cpts])
        pClose, pFoodItemClose = cpts[i][5], cpts[i][6]
        distClose = cpts[i][8]

        cptsFar = p.getClosestPoints(bodyA=self.farthest_pt_body, bodyB=self.food_item, distance=100,
                                     physicsClientId=self.id)
        i = np.argmin([cpt[8] for cpt in cptsFar])
        pFar, pFoodItemFar = cptsFar[i][5], cptsFar[i][6]
        distFar = cptsFar[i][8]

        if self.debug:
            if hasattr(self, "debugLine1"):
                self.debugLine1 = p.addUserDebugLine(lineFromXYZ=pClose,
                                                     lineToXYZ=pFoodItemClose,
                                                     lineColorRGB=[1, 0, 0],
                                                     replaceItemUniqueId=self.debugLine1)

                self.debugLine2 = p.addUserDebugLine(lineFromXYZ=pFar,
                                                     lineToXYZ=pFoodItemFar,
                                                     lineColorRGB=[1, 0, 1],
                                                     replaceItemUniqueId=self.debugLine2)
            else:

                self.debugLine1 = p.addUserDebugLine(lineFromXYZ=pClose,
                                                     lineToXYZ=pFoodItemClose,
                                                     lineColorRGB=[1, 0, 0])

                self.debugLine2 = p.addUserDebugLine(lineFromXYZ=pFar,
                                                     lineToXYZ=pFoodItemFar,
                                                     lineColorRGB=[1, 0, 1])

        d = {
            "joint_positions": robot_right_joint_positions,  # joint positions
            "fork_position": fork_pos,  # 3D drop fork position (link frame)
            "fork_orientation": fork_orient,  # drop fork orientation (link frame)
            "food_position": food_pos,  # 3D food position (link frame)
            "food_position_on_fork": food_in_fork_offset,  # 3D food position (relative to fork frame)
            "food_orientation": food_orient,  # food orientation (absolute)
            "food_orientation_on_fork": self.food_orient_quat,  # food orientation (relative to fork)
            "food_size": self.food_true_size,  # food size scale
            "food_type": [self.food_type],  # type of food
            "food_closest_point": pFoodItemClose,  # the farthest back point on the food item (i.e. the min extent)
            "food_farthest_point": pFoodItemFar,  # the farthest back point on the food item (i.e. the min extent)
            "mouth_position": mouth_pos,  # position of mouth link frame
            "mouth_orientation": mouth_orientation,  # orientation of mouth (absolute)
            "mouth_orientation_delta": difference,  # mouth orientation change from start (difference quat)
            "mouth_forces": forces,  # forces exerted on mouth from each non mouth object [robot, fork, food item]
            "is_inside_mouth": [int(insidemouth)],  # is food obscured by mouth in all axes
            "mouth_margin": margin,  # margin to mouth walls from object CENTER along raycast directions.
            "ee_to_food_offset": offset,  # in ee coordinate frame, offset from end effector to mouth (link frames)
            "ee_position": ee_pos,  # position of the end effector link frame
            "ee_orientation": ee_orient,  # orientation of the end effector link frame
            "ee_velocity": velocity,
            "ee_angular_velocity": angular_velocity,
        }
        if ret_images:
            d["depth1"] = self.depth_opengl1
            d["depth2"] = self.depth_opengl2

        return AttrDict.from_dict(d).leaf_apply(lambda arr: np.asarray(arr)[None])

    # assumes drop_fork and food_item have been created already
    # NOTE: if you override this, set child offset and parent offset and food orient quat/eul to be consistent!
    def gen_food_constraint(self):
        food2tool, food_in_tool = CoordinateFrame.transform_from_a_to_b(self._base_food_frame, self._base_tool_frame)

        # Create constraint that keeps the food item in the tool
        self._food_constraint = p.createConstraint(self.drop_fork, -1,
                                                   self.food_item, -1, p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                   parentFramePosition=food_in_tool,
                                                   childFramePosition=[0, 0, 0],
                                                   parentFrameOrientation=food2tool.as_quat(),
                                                   childFrameOrientation=[0, 0, 0, 1],
                                                   physicsClientId=self.id)
        p.changeConstraint(self._food_constraint, maxForce=np.inf, physicsClientId=self.id)

    def remove_food_constraint(self):
        if hasattr(self, "_food_constraint"):
            p.changeConstraint(self._food_constraint, maxForce=1, physicsClientId=self.id)

    def set_external_forces(self, action):
        pass

    # override safe
    def cleanup(self):
        # Remove all things that will be created
        if hasattr(self, "_food_constraint"):
            p.removeConstraint(self._food_constraint, physicsClientId=self.id)
            del self._food_constraint
        if hasattr(self, "food_item"):
            # p.removeCollisionShape(self.food_collision, physicsClientId=self.id)
            p.removeBody(self.food_item, physicsClientId=self.id)
            del self.food_item
            # del self.food_collision

    # CALL THIS AT THE BEGINNING if you OVERRIDE
    def pre_reset(self, presets: AttrDict = AttrDict()):
        # occasionally we need to actually reset and reload assets to clear some pybullet caches
        self.num_resets += 1
        if is_next_cycle(self.num_resets, self.reset_full_every_n_resets):
            logger.warn("Reloading environment from scratch!")
            p.resetSimulation(physicsClientId=self.id)
            self.load()

    def reset_robot(self,
                    presets: AttrDict = AttrDict()):  # ret_images=False, food_type=None, food_size=None, food_orient_eul=None, mouth_orient_eul=None):
        """
        :param presets:
            robot_joint_positions
        :return:
        """

        joint_pos = get_with_default(presets, "robot_joint_positions", self.robot_joint_default_positions)
        # Reset all robot joints
        for rj in range(len(joint_pos)):
            p.resetJointState(self.robot, jointIndex=rj, targetValue=joint_pos[rj],
                              targetVelocity=0, physicsClientId=self.id)

        # these are the new "initial"
        self.robot_start_joint_positions = np.asarray(joint_pos).copy()
        self.robot_start_ee_position, self.robot_start_ee_orientation = p.getLinkState(self.robot, 8,
                                                                                       physicsClientId=self.id)[4:6]
        self.robot_start_ee_position = np.asarray(self.robot_start_ee_position)
        self.robot_start_ee_orientation = np.asarray(self.robot_start_ee_orientation)

    def reset_assets(self, presets: AttrDict = AttrDict()):
        """
        :param presets:
            food_type
            food_true_size
            food_orientation_on_fork
            food_position_in_food_frame (right now we interpret this as a slide in y direction (i.e. in child frame)
            mouth_position (defaults to some fixed dist from mouth)
            mouth_orientation_eul

        :return:
        """

        # HANDLE PRESETS
        self.food_type = get_with_default(presets, "food_type", np.random.randint(0, len(self.foods), dtype=int))
        assert 0 <= self.food_type < len(self.foods)

        if self.food_scale_3D:
            random_food_scale_factor = np.random.uniform(self.foods_scale_range[self.food_type][0],
                                                         self.foods_scale_range[self.food_type][1], (3,))
        else:
            random_food_scale_factor = np.random.uniform(self.foods_scale_range[self.food_type][0],
                                                         self.foods_scale_range[self.food_type][1])

        if presets.has_leaf_key("food_true_size"):
            self.food_scale = np.divide(presets.food_true_size, self.food_base_true_size[self.food_type])
        elif presets.has_leaf_key("food_scale"):
            self.food_scale = presets.food_scale
        else:
            self.food_scale = random_food_scale_factor

        self.food_orient_quat = get_with_default(presets, "food_orientation_on_fork",
                                                 np.asarray(p.getQuaternionFromEuler(np.random.rand(3) * 2 * np.pi)))

        default_food_pos_on_food = np.zeros(3, dtype=float)
        if self.food_offset_on_fork_randomize:  # TODO implement for non carrots
            default_food_pos_on_food[1] = ((np.random.rand(1)[0] * 2.) - 1.) * self.food_base_true_size[self.food_type][
                1] / 4.

        self.food_pos_on_food = get_with_default(presets, "food_position_in_food_frame", default_food_pos_on_food)

        # TODO p.getLinkState(self.robot, 8, )[4:6]
        new_mouth_base_position = self.robot_start_ee_position + np.array([0., 0.5, 0.])

        self.init_mouth_position = get_with_default(presets, "mouth_position", new_mouth_base_position)
        self.init_mouth_orientation_eul = get_with_default(presets, "mouth_orientation_eul",
                                                           self.mouth_base_orientation_eul)

        # Reset table position
        p.resetBasePositionAndOrientation(self.table,
                                          posObj=self.table_base_position,
                                          ornObj=p.getQuaternionFromEuler(self.table_base_orientation_eul),
                                          physicsClientId=self.id)

        # reset mouth position
        p.resetBasePositionAndOrientation(self.mouth,
                                          posObj=self.init_mouth_position,
                                          ornObj=p.getQuaternionFromEuler(self.init_mouth_orientation_eul),
                                          physicsClientId=self.id)

        # fixed ref pt close
        p.resetBasePositionAndOrientation(self.closest_pt_body,
                                          posObj=self.init_mouth_position + np.array([0., -1.0, 0.]),
                                          ornObj=[0, 0, 0, 1],
                                          physicsClientId=self.id)

        # fixed ref pt far away
        p.resetBasePositionAndOrientation(self.farthest_pt_body,
                                          posObj=self.init_mouth_position + np.array([0., 1.0, 0.]),
                                          ornObj=[0, 0, 0, 1],
                                          physicsClientId=self.id)

        # Reset all mouth joints
        for rj in range(p.getNumJoints(self.mouth, physicsClientId=self.id)):
            p.resetJointState(self.mouth, jointIndex=rj, targetValue=0,
                              targetVelocity=0, physicsClientId=self.id)

        # reset drop fork & gripper
        self.reset_gripper_positions(0.0)
        self.base_ee_pos, self.base_ee_orient = p.getLinkState(self.robot, 8, computeForwardKinematics=True,
                                                               physicsClientId=self.id)[4:6]  # link frames
        self._base_ee_frame = CoordinateFrame(world_frame_3D, R.from_quat(self.base_ee_orient).inv(),
                                              np.asarray(self.base_ee_pos))
        self._base_tool_frame = CoordinateFrame(self._base_ee_frame, R.from_quat(self.tool_orient_in_ee).inv(),
                                                self.tool_pos_in_ee)

        t2w, t_in_w = self._base_tool_frame.get_transform_to_global()

        p.resetBasePositionAndOrientation(self.drop_fork,
                                          posObj=t_in_w,
                                          ornObj=t2w.as_quat(),
                                          physicsClientId=self.id)

        """ FOOD """

        # self.food_orient_quat = p.getQuaternionFromAxisAngle([1., 0., 0.], np.pi/4, physicsClientId=self.id)
        # -z axis forward (towards face)
        # y axis down
        # x axis left (right side of face)

        assert np.all(self.foods_scale_range[self.food_type][
                          0] <= self.food_scale), "Food scale was outside of range: %s" % self.food_scale
        assert np.all(self.food_scale <= self.foods_scale_range[self.food_type][
            1]), "Food scale was outside of range: %s" % self.food_scale

        mesh_scale = np.array([self.food_base_scale[self.food_type]] * 3) * self.food_scale

        logger.debug("LOADING food file: %s with scale %s" % (self.foods[self.food_type], str(self.food_scale)))

        food_visual = p.createVisualShape(shapeType=p.GEOM_MESH,
                                          fileName=os.path.join(self.asset_directory, 'food_items',
                                                                self.foods[self.food_type] + ".obj"),
                                          rgbaColor=[1.0, 1.0, 1.0, 1.0],
                                          meshScale=mesh_scale, physicsClientId=self.id)

        self.food_collision = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                     fileName=os.path.join(self.asset_directory, 'food_items',
                                                                           self.foods[self.food_type] + "_vhacd.obj"),
                                                     meshScale=mesh_scale, physicsClientId=self.id)
        self.food_item = p.createMultiBody(baseMass=0.012, baseCollisionShapeIndex=self.food_collision,
                                           baseVisualShapeIndex=food_visual,
                                           basePosition=np.array(self.init_mouth_position) + np.array([0, -0.1, 0]),
                                           useMaximalCoordinates=False,
                                           physicsClientId=self.id)

        # bounding box = scale for the object
        low, high = p.getAABB(self.food_item, -1, physicsClientId=self.id)
        self.food_true_size = np.abs(np.array(high) - np.array(low))

        self.food_multiplier = np.array(self.food_true_size) / np.asarray(mesh_scale)

        logger.debug("Food scale: %s" % self.food_scale)
        logger.debug("Food true size: %s" % self.food_true_size)
        logger.debug("Food multiplier: %s" % self.food_multiplier)

        # defined relative to SIM FORK frame, where is fork tip
        self.parent_offset = np.array([0, 0, -0.025 * 2])  # fixed (where the food is in the fork frame)
        self.child_offset = self.food_scale * self.food_offsets[
            self.food_type]  # [-0.004, -0.0065]  # TODO from bounding box info

        self.child_offset += self.food_pos_on_food  # in child frame, be careful with this

        logger.debug("Child offset: %s" % str(self.child_offset))

        # COORD FRAME of food relative to EE (this is the same frame as ee)
        self._base_tool_tip_frame_sim = CoordinateFrame(self._base_tool_frame, R.identity(), self.parent_offset)
        self._base_tool_tip_frame = CoordinateFrame(self._base_tool_tip_frame_sim, R.from_quat(self.tool_orient_in_ee),
                                                    np.zeros(3))  # EE frame orientation, tip position
        child_offset_in_food_frame = -R.from_quat(self.food_orient_quat).apply(self.child_offset)
        self._base_food_frame = CoordinateFrame(self._base_tool_tip_frame, R.from_quat(self.food_orient_quat).inv(),
                                                child_offset_in_food_frame)

        # import open3d
        # from sbrl.utils.o3d_utils import draw_frame
        # open3d.visualization.draw_geometries([draw_frame(f, (i+1) *0.02) for i,f in
        #                                       enumerate([world_frame_3D, self._base_ee_frame, self._base_tool_frame, self._base_tool_tip_frame_sim,
        #                                                  self._base_tool_tip_frame, self._base_food_frame])])

        f2w, f_in_w = self._base_food_frame.get_transform_to_global()

        p.resetBasePositionAndOrientation(self.food_item,
                                          posObj=f_in_w,
                                          ornObj=f2w.as_quat(),
                                          physicsClientId=self.id)

        self.texture = p.loadTexture(os.path.join(self.asset_directory, 'food_items', self.foods[self.food_type] + ".png"))
        p.changeVisualShape(self.food_item, -1,
                            textureUniqueId=self.texture)

    def reset_dynamics(self, presets: AttrDict = AttrDict()):
        # this gives the food item some oomf lol
        p.changeDynamics(self.drop_fork, -1, lateralFriction=0.2, localInertiaDiagonal=[0.001, 0.001, 0.001],
                         physicsClientId=self.id)
        p.changeDynamics(self.food_item, -1, lateralFriction=0.5, localInertiaDiagonal=[0.001, 0.001, 0.001],
                         physicsClientId=self.id)

        # Disable collisions between the tool and food item
        for ti in list(range(p.getNumJoints(self.drop_fork, physicsClientId=self.id))) + [-1]:
            for tj in list(range(p.getNumJoints(self.food_item, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(self.drop_fork, self.food_item, ti, tj, False, physicsClientId=self.id)

        p.changeConstraint(self._mouth_constraint, jointChildPivot=self.init_mouth_position,
                           jointChildFrameOrientation=p.getQuaternionFromEuler(self.init_mouth_orientation_eul),
                           physicsClientId=self.id)

        # Create constraint that keeps the food item in the tool
        self.gen_food_constraint()

    def _reset_images(self, presets: AttrDict = AttrDict()):
        food_pos, food_orient = p.getBasePositionAndOrientation(self.food_item)

        self.viewMat1 = p.computeViewMatrixFromYawPitchRoll(food_pos, self.camdist, self.yaw, self.pitch, self.roll,
                                                            2,
                                                            self.id)
        self.viewMat2 = p.computeViewMatrixFromYawPitchRoll(food_pos, self.camdist, -self.yaw, self.pitch,
                                                            self.roll,
                                                            2,
                                                            self.id)
        self.projMat = p.computeProjectionMatrixFOV(self.fov, self.img_width / self.img_height, self.near, self.far)
        images1 = p.getCameraImage(self.img_width,
                                   self.img_height,
                                   self.viewMat1,
                                   self.projMat,
                                   shadow=True,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                   physicsClientId=self.id)
        images2 = p.getCameraImage(self.img_width,
                                   self.img_height,
                                   self.viewMat2,
                                   self.projMat,
                                   shadow=True,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                   physicsClientId=self.id)
        self.rgb_opengl1 = np.reshape(images1[2], (self.img_width, self.img_height, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images1[3], [self.img_width, self.img_height])
        self.depth_opengl1 = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_opengl)

        self.rgb_opengl2 = np.reshape(images2[2], (self.img_width, self.img_height, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images2[3], [self.img_width, self.img_height])
        self.depth_opengl2 = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_opengl)

        self.im1.set_data(self.rgb_opengl1)
        self.im2.set_data(self.rgb_opengl2)
        self.im1_d.set_data(self.depth_opengl1)
        self.im2_d.set_data(self.depth_opengl2)
        print(np.min(self.depth_opengl1), np.max(self.depth_opengl1))
        self.fig.canvas.draw()

    #
    # def _reset_robot(self, joint_position):
    #     self.state = {}
    #     self.jacobian = {}
    #     self.desired = {}
    #     for idx in range(len(joint_position)):
    #         p.resetJointState(self.robot, idx, joint_position[idx])
    #     self._read_state()
    #     self._read_jacobian()
    #     self.desired['joint_position'] = self.state['joint_position']
    #     self.desired['ee_position'] = self.state['ee_position']
    #     self.desired['ee_quaternion'] = self.state['ee_quaternion']
    #
    # # copied from panda-env repo
    # def _read_jacobian(self):
    #     linear_jacobian, angular_jacobian = p.calculateJacobian(self.robot, 11, [0, 0, 0],
    #                                                             list(self.state['joint_position']), [0] * 9, [0] * 9)
    #     linear_jacobian = np.asarray(linear_jacobian)[:, :7]
    #     angular_jacobian = np.asarray(angular_jacobian)[:, :7]
    #     full_jacobian = np.zeros((6, 7))
    #     full_jacobian[0:3, :] = linear_jacobian
    #     full_jacobian[3:6, :] = angular_jacobian
    #     self.jacobian['full_jacobian'] = full_jacobian
    #     self.jacobian['linear_jacobian'] = linear_jacobian
    #     self.jacobian['angular_jacobian'] = angular_jacobian

    def _read_state(self):
        joint_position = [0] * 9
        joint_velocity = [0] * 9
        joint_torque = [0] * 9
        joint_states = p.getJointStates(self.robot, jointIndices=range(9), physicsClientId=self.id)
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
            joint_velocity[idx] = joint_states[idx][1]
            joint_torque[idx] = joint_states[idx][3]
        ee_states = p.getLinkState(self.robot, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.robot, linkIndexA=10)
        self.state['joint_position'] = np.asarray(joint_position)
        self.state['joint_velocity'] = np.asarray(joint_velocity)
        self.state['joint_torque'] = np.asarray(joint_torque)
        self.state['ee_position'] = np.asarray(ee_position)
        self.state['ee_quaternion'] = np.asarray(ee_quaternion)
        self.state['ee_euler'] = np.asarray(p.getEulerFromQuaternion(ee_quaternion))
        self.state['gripper_contact'] = len(gripper_contact) > 0

    # to get to position and quaternion (end effector)
    def _control(self, action, **kwargs):
        assert (action.shape[0] == 7 and self.is_ee_control) or (
                    action.shape[0] == len(self.robot_arm_indices) and not self.is_ee_control)
        # mode 0 is end effector control, 1 is joint control
        if self.is_ee_control:
            target_ee_position = action[:3]
            target_ee_quaternion = action[3:7]
            q_dot = np.asarray(p.calculateInverseKinematics(self.robot, 8, target_ee_position, target_ee_quaternion)) - \
                    self.state['joint_position']
        else:
            target_joint_position = action
            q_dot = np.asarray(target_joint_position - self.state['joint_position'])

        q_dot = q_dot[self.robot_arm_indices]

        if self.robot_control_mode == p.VELOCITY_CONTROL:
            q_dot *= self.robot_joint_velocity_scale
            p.setJointMotorControlArray(self.robot, self.robot_arm_indices, p.VELOCITY_CONTROL,
                                        forces=self.robot_joint_position_forces, targetVelocities=q_dot)
        elif self.robot_control_mode == p.POSITION_CONTROL:
            p.setJointMotorControlArray(self.robot, self.robot_arm_indices, p.POSITION_CONTROL,
                                        targetPositions=self.state["joint_position"][self.robot_arm_indices] + q_dot,
                                        positionGains=np.array(self.robot_joint_position_gains),
                                        forces=self.robot_joint_position_forces,
                                        physicsClientId=self.id)
        else:
            raise NotImplementedError

        p.setJointMotorControlArray(self.robot, [9, 10], p.POSITION_CONTROL, targetPositions=[0, 0])

    def get_motor_joint_states(self, robot):
        num_joints = p.getNumJoints(robot, physicsClientId=self.id)
        joint_states = p.getJointStates(robot, range(num_joints), physicsClientId=self.id)
        joint_infos = [p.getJointInfo(robot, i, physicsClientId=self.id) for i in range(num_joints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def enforce_joint_limits(self, body):
        # Enforce joint limits
        joint_states = p.getJointStates(body, jointIndices=list(range(p.getNumJoints(body, physicsClientId=self.id))),
                                        physicsClientId=self.id)
        joint_positions = np.array([x[0] for x in joint_states])
        lower_limits = []
        upper_limits = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_pos = joint_positions[j]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
            # print(joint_name, joint_pos, lower_limit, upper_limit)
            if joint_pos < lower_limit:
                p.resetJointState(body, jointIndex=j, targetValue=lower_limit, targetVelocity=0,
                                  physicsClientId=self.id)
            elif joint_pos > upper_limit:
                p.resetJointState(body, jointIndex=j, targetValue=upper_limit, targetVelocity=0,
                                  physicsClientId=self.id)
        lower_limits = np.array(lower_limits)
        upper_limits = np.array(upper_limits)
        return lower_limits, upper_limits

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    def get_rpt(self, roll, phi, theta):
        rot1 = R.from_quat(p.getQuaternionFromAxisAngle([0, 0, 1], roll))
        rot2 = R.from_quat(p.getQuaternionFromAxisAngle([1, 0, 0], -np.pi / 2 + phi))
        rot3 = R.from_quat(p.getQuaternionFromAxisAngle([0, 0, 1], theta))
        chained = rot3 * rot2 * rot1
        quat = chained.as_quat()
        # self.quaternion = np.array(angle_axis_apply_to_quat([ax, ay, az], w, base_quaternion))
        eul = np.array(p.getEulerFromQuaternion(quat))
        return quat, eul


# teleop code as a test
if __name__ == '__main__':
    max_steps = 5000

    params = get_example_params()
    params.render = True
    params.compute_images = False
    timeit.reset()
    timeit.start("episode")
    params.debug = False
    params.food_scale_3D = True
    params.mouth_base_orientation_eul = [np.pi / 2, np.pi / 2 + np.pi / 8, -np.pi / 2]
    env = BulletBiteTransfer(params,
                             EnvSpec(AttrDict(names_shapes_limits_dtypes=[])))  # gym.make('BiteTransferPanda-v0')

    presets = AttrDict()
    presets.food_type = BulletBiteTransfer.VALID_FOODS.index("carrot")
    presets.food_scale = 1.0
    presets.food_orientation_eul_on_fork = np.array([0, 0, 0])
    presets.food_position_in_food_frame = np.array([0, 0, 0])
    # presets.mouth_position = np.array([0.5, 1.2, 0.5])
    # presets.mouth_orientation_eul = np.array([0.5, 1.2, 0.5])

    observation = env.reset()
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

    # Get the position and orientation of the end effector
    position, orientation = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2]
    rpt_orientation = [0, 0, 0]  # rpt rotation about default

    i = 0
    while True:
        i += 1
        keys = p.getKeyboardEvents()
        if i >= max_steps or ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            i = 0
            print("Resetting from keyboard (after %d iters)!" % i)
            observation = env.reset()
            timeit.stop("episode")
            logger.debug(timeit)
            timeit.reset()
            timeit.start("episode")
            position, orientation = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2]

        for key, action in keys_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                position += action

        for key, action in keys_orient_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                # orientation = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(orientation) + action) % (2 * np.pi))
                rpt_orientation = (rpt_orientation + action) % (2 * np.pi)
                orientation, _ = env.get_rpt(*rpt_orientation)

        # IK to get new joint positions (angles) for the robot
        target_joint_positions = p.calculateInverseKinematics(env.robot, 8, position, orientation)
        target_joint_positions = target_joint_positions[:7]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        # print(position, p.getEulerFromQuaternion(orientation))
        # logger.debug("rpt: %s, quat: %s)" % (rpt_orientation, orientation))
        # print(joint_positions)

        # Set joint action to be the error between current and target joint positions
        # joint_action = (target_joint_positions - joint_positions) * 10
        # observation = env.step(joint_action)

        observation = env.step(np.append(np.asarray(position), np.asarray(orientation)), )
