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

from sbrl.envs.bullet_envs.sth_sth import BulletSth27
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.policies.controllers import os_torque_control_panda_cfg
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.math_utils import convert_rpt, convert_quat_to_rpt
from sbrl.utils.python_utils import AttrDict as d


class BulletSthGrabLift27(BulletSth27):
    def _init_params_to_attrs(self, params: d):
        params.robot_params.start_pos = [0.3, 0.5, 0.]
        params.robot_params.collision_detection_links = [2,3,4]
        params.scaling = params.scaling if params.has_leaf_key("scaling") else 1.2
        super(BulletSthGrabLift27, self)._init_params_to_attrs(params)

    def _load_asset_objects(self):
        super(BulletSthGrabLift27, self)._load_asset_objects()
        # offset in x dir necessary
        self.table_x_offsets = np.asarray([0.075 * self.object.scaling + 0.075, -0.1 - 0.075 * self.object.scaling])
        self.table_y_offsets = np.asarray([0.075 * self.object.scaling + 0.35, -0.15 + -0.075 * self.object.scaling])

    def init_grasp(self, object: d, presets: d):
        # open, down facing

        # go "behind" object
        behind = np.asarray(object.position).copy()
        behind[0] -= 0.25  # x distance
        behind[2] += 0.05  # z distance

        if presets.has_leaf_key("grasp_offset"):
            grasp_offset = np.asarray(presets >> "grasp_offset")
            # x y z offset for grasp start
            assert len(grasp_offset) == 3 and len(grasp_offset.shape) == 1
        else:
            grasp_offset = np.random.uniform(-self.randomize_grasp_start_location_noise, self.randomize_grasp_start_location_noise)

        behind += grasp_offset

        ee_orn = R.from_quat(self.robot.get_end_effector_orn())
        ee_orn = R.from_rotvec([0, -np.pi / 2, 0]) * R.from_rotvec([0, 0, np.pi / 2]) * ee_orn  # right facing
        behind_frame = CoordinateFrame(world_frame_3D, ee_orn.inv(), behind)
        rp = self.robot.rp
        for i in range(2):  # N chances for IK solver
            q_desired = self.robot.compute_frame_ik(self.robotEndEffectorIndex, behind_frame, rest_pose=rp)
            # keep robot at q_desired
            self.robot.set_joint_values(q_desired, 0)
            self.robot.joint_position_control(self.robot.controlled_arm_joints, q_desired, gripper_pos=0)
            rp = q_desired

        # now step with gripper only
        for i in range(2):
            self.robot.gripper_control(0., step_simulation=True)  # keep the gripper open

        self.set_initial_joint_positions()  # sets to current location

    def get_success(self, seg=None):
        is_gripping = self.is_gripping(self.object)
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        if is_gripping and pos1[2] - self.object.position[2] > 0.05:  # in contact and lifted up a bit
            return True
        else:
            return False


class BulletSthGrabPull(BulletSthGrabLift27):
    def get_success(self, seg=None):
        is_gripping = self.is_gripping(self.object)
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        delta = pos1[:2] - self.object.position[:2]
        # in contact and pulled only back
        if is_gripping and delta[0] < -0.1 and abs(delta[1]) < 0.04:
            return True
        else:
            return False


class BulletSthGrabPush(BulletSthGrabLift27):
    def get_success(self, seg=None):
        is_gripping = self.is_gripping(self.object)
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        delta = pos1[:2] - self.object.position[:2]
        # in contact and pulled only right (relative to ee direction)
        if is_gripping and delta[0] > 0.1 and abs(delta[1]) < 0.04:
            return True
        else:
            return False


class BulletSthGrabLeft(BulletSthGrabLift27):
    def get_success(self, seg=None):
        is_gripping = self.is_gripping(self.object)
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        delta = pos1[:2] - self.object.position[:2]
        # in contact and pulled only left (relative to ee direction)
        if is_gripping and abs(delta[0]) < 0.04 and delta[1] > 0.05:
            return True
        else:
            return False


class BulletSthGrabRight(BulletSthGrabLift27):
    def get_success(self, seg=None):
        is_gripping = self.is_gripping(self.object)
        pos1 = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)[0]
        pos1 = np.asarray(pos1)
        delta = pos1[:2] - self.object.position[:2]
        # in contact and pulled only right (relative to ee direction)
        if is_gripping and abs(delta[0]) < 0.04 and delta[1] < -0.1:
            return True
        else:
            return False


class BulletSthPushForward(BulletSthGrabLift27):
    def is_tipped(self, object: d):
        pos1, orn1 = p.getBasePositionAndOrientation(object.id, physicsClientId=self.id)
        rot = R.from_quat(np.asarray(orn1))
        obj_z_in_world = rot.apply([0, 0, 1])
        # 75+ degrees from z axis
        return np.arccos(np.abs(obj_z_in_world[2])) / np.pi * 180.0 > 75.0

    def is_in_contact(self, object: d):
        return len(p.getContactPoints(bodyA=self.robotId, bodyB=object.id)) > 0

    def is_moved(self, object: d):
        pos, orn = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)
        pos = np.asarray(pos)
        delta = pos - object.position
        # not moved up, moved 5cm back
        return abs(delta[2]) < 0.02 and abs(delta[1]) < 0.03 and delta[0] > 0.1

    def get_success(self, seg=None):
        # is_gripping = self.is_gripping(self.object)
        is_tipped = self.is_tipped(self.object)
        is_in_contact = self.is_in_contact(self.object)
        is_moved = self.is_moved(self.object)
        # print("GRIP: %s | TIP: %s | CONTACT: %s | MOVE: %s" % (False, is_tipped, is_in_contact, is_moved))

        # in contact and pulled only right (relative to ee direction)
        if is_in_contact and is_moved and not is_tipped:
            return True
        else:
            return False


class BulletSthPushLeft(BulletSthPushForward):
    def is_moved(self, object: d):
        pos, orn = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)
        pos = np.asarray(pos)
        delta = pos - object.position
        # not moved up/forward, moved 5cm in left (+y)
        return abs(delta[2]) < 0.02 and abs(delta[0]) < 0.05 and delta[1] > 0.1


class BulletSthPushRight(BulletSthPushForward):
    def is_moved(self, object: d):
        pos, orn = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.id)
        pos = np.asarray(pos)
        delta = pos - object.position
        # not moved up/forward, moved 5cm in left (+y)
        return abs(delta[2]) < 0.02 and abs(delta[0]) < 0.05 and delta[1] < -0.1


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
    env = BulletSthGrabLeft(params, ParamEnvSpec(d(names_shapes_limits_dtypes=[
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
