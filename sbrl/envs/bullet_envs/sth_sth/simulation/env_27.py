#!/usr/bin/env python3

## plotting forces
import multiprocessing as mp
import sys
from multiprocessing import Process

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.sth_sth.simulation.env import BulletSth
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers.robot_config import os_torque_control_panda_cfg
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.python_utils import AttrDict as d, get_with_default


class BulletSth27(BulletSth):
    def _init_params_to_attrs(self, params: d):
        super(BulletSth27, self)._init_params_to_attrs(params)
        assert self.task_id == 27
        self.obj_friction_ceof = get_with_default(params, "object_friction_coef", 2.0)  # 200.
        self.obj_linear_damping = get_with_default(params, "object_linear_damping", 1.0)
        self.obj_angular_damping = get_with_default(params, "object_angular_damping", 1.0)
        self.obj_contact_stiffness = get_with_default(params, "object_contact_stiffness", 100.0)  # 1.
        self.obj_contact_damping = get_with_default(params, "object_contact_damping", 0.9)

        self.randomize_object_start_location = get_with_default(params, "randomize_object_start_location", True)
        self.start_in_grasp = get_with_default(params, "start_in_grasp", True)  # means call init_grasp
        # generated [-e, e]
        self.randomize_grasp_start_location_noise = get_with_default(params, "randomize_grasp_start_location_noise", np.zeros(3),
                                                                     map_fn=lambda arr: np.broadcast_to(arr, (3,)).copy())
        self.randomize_grasp_start_location_noise[self.randomize_grasp_start_location_noise == 0] = 1e-20
        self.scaling = get_with_default(params, "scaling", None)  # 1.4 is the default (see BulletSth)
        self.mass_bounds = get_with_default(params, "mass_bounds", None)  # 1.4 is the default (see BulletSth)

        if self.mass_bounds is not None:
            logger.info(f"Mass bounds are set to -> {self.mass_bounds}")

        # nonzero means randomize the object orientation
        self.rotation_noise = get_with_default(params, "rotation_noise", 0.)  # radian std for the shape

        if self.rotation_noise > 0:
            logger.info(f"Nonzero rotation noise -> {self.rotation_noise}")

        if np.any(self.randomize_grasp_start_location_noise > 0):
            logger.info(f"Nonzero grasp start location noise -> {self.randomize_grasp_start_location_noise}")

    def _init_setup(self):
        super(BulletSth27, self)._init_setup()

    def _load_asset_objects(self):
        self.object = self.init_obj(d(object_cls='bottle_b1', object_scaling=self.scaling))
        p.changeVisualShape(self.object.id, -1, rgbaColor=[1., 0., 0., 1], physicsClientId=self.id)

        self.table_x_offsets = np.asarray([0.075 * self.object.scaling, -0.075 * self.object.scaling])
        self.table_y_offsets = np.asarray([0.075 * self.object.scaling + 0.25, -0.075 * self.object.scaling])

        # registering
        self.objects.append(self.object)

    def _load_dynamics(self):
        super(BulletSth27, self)._load_dynamics()

        self.robot.set_gripper_max_force(2.0)  # not very strong
        # self.robot.set_arm_max_force(1000.0)
        arm_damp = [0.1, 0.05, 0.04, 0.01, 0.01, 0.01, 0.01]
        self.robot.set_joint_damping(arm_damp + [0.01] * 6)
        p.changeDynamics(self.object.id, -1, mass=0.1,
                         lateralFriction=self.obj_friction_ceof,
                         rollingFriction=self.obj_friction_ceof,
                         spinningFriction=self.obj_friction_ceof,
                         linearDamping=self.obj_linear_damping,
                         angularDamping=self.obj_angular_damping,
                         contactStiffness=self.obj_contact_stiffness,
                         contactDamping=self.obj_contact_damping,
                         physicsClientId=self.id)

    def reset_dynamics(self, presets: d = d()):
        super(BulletSth27, self).reset_dynamics(presets)
        m = 0.1 if self.mass_bounds is None else np.random.uniform(*self.mass_bounds)
        mass, = presets.get_keys_optional(['mass'], [m])
        p.changeDynamics(self.object.id, -1, mass=mass, physicsClientId=self.id)

    # def init_obj(self):
    #     self.obj_id = self.p.loadURDF(os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf"))
    #     self.p.

    def reset_assets(self, presets: d = d()):
        super(BulletSth27, self).reset_assets(presets)

        if presets.has_leaf_key('object_position'):
            pos = np.asarray(presets >> 'object_position')
        else:
            # base_frame will be set again for our object
            pos = np.asarray([0.3637 + 0.06, -0.05, 0.34])
            if self.randomize_object_start_location:
                robot_base_frame = self.robot.get_link_frame(0)
                xy_min, xy_max = self.table_aabb
                xy_min = xy_min.copy()
                xy_max = xy_max.copy()
                xy_min[0] += self.table_x_offsets[0]
                xy_min[1] += self.table_y_offsets[0]
                xy_max[0] += self.table_x_offsets[1]
                xy_max[1] += self.table_y_offsets[1]
                while True:
                    pos[:2] = np.random.uniform(xy_min, xy_max)[
                              :2]
                    err = np.linalg.norm(pos[:2] - robot_base_frame.pos[:2])
                    if err > 0.3:
                        # print(err)
                        break

        if presets.has_leaf_key('object_rotation_angle'):
            rot = float(presets >> 'object_rotation_angle')
        else:
            rot = 0 if self.rotation_noise == 0. else np.random.normal(0, self.rotation_noise)

        orn = R.from_rotvec([0, 0, rot])

        self.set_obj_pose(self.object, pos, orn.as_quat().tolist(),  # [0, 0, -0.1494381, 0.9887711]
                          assign_base_frame=True)

        if self.start_in_grasp:
            self.init_grasp(self.object, presets)

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
        obj_obs = super(BulletSth27, self)._get_object_obs(object)
        obj_obs.is_gripping = np.array([self.is_gripping(object)])[None]
        return obj_obs

    def get_success(self, seg=None):
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        if pos1[2] - self.object.position[2] > 0.15:  # position field holds the initial position
            return True
        else:
            return False


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


# teleop code as a test
if __name__ == '__main__':

    # import sharedmem as shm
    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 5000
    params = d()
    params.task_id = 27
    params.render = True
    params.compute_images = False
    params.debug = False
    params.skip_n_frames_every_step = 2
    params.time_step = 0.025  # 20Hz
    params.combine(os_torque_control_panda_cfg)
    env = BulletSth27(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
        ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('success', (1,), (False, True), np.bool),
    ], observation_names=['wrist_ft'], output_observation_names=['success'])))  # gym.make('BiteTransferPanda-v0')

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

    grip_state = 0
    done = False

    i = 0

    # shared_ft = shm.empty((6,), dtype=np.float32)
    # shared_ft[:] = 0

    queue = mp.Queue()

    proc = Process(target=drawProcess, args=(queue,), daemon=True)
    proc.start()

    # vel = np.array([0., -0.01, 0.])
    # for i in range(30):
    #     observation, _, done = env.step(np.concatenate([target_position + i * vel, target_orientation, np.array([0])]))

    while True:
        i += 1
        keys = p.getKeyboardEvents(physicsClientId=env.id)
        if done or i >= max_steps or ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            logger.debug("Resetting (after %d iters)! done = %s, success = %s" % (i, done, observation.success.reshape(-1)[-1]))
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
