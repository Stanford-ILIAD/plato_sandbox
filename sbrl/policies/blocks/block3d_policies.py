# this is for block_env_3d and its derivatives.
from random import random
from typing import List

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D, get_block3d_example_params, \
    get_block3d_example_spec_params
from sbrl.envs.bullet_envs.block3d.platform_block_env_3d import PlatformBlockEnv3D
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.blocks.stack_block2d_success_metrics import circular_difference_fn
from sbrl.policies.policy import Policy
from sbrl.utils.control_utils import orientation_error, batch_orientation_eul_add
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D, np_project_points_onto_plane
from sbrl.utils.python_utils import AttrDict as d, get_with_default
from sbrl.utils.torch_utils import to_numpy
from sbrl.utils.transform_utils import euler2mat


class Block3DPrimitive(Policy):

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, block_idx=0, rel_frame: CoordinateFrame = world_frame_3D,
                     tolerance=0.005, ori_tolerance=0.05, **kwargs):
        assert self._env.num_blocks > block_idx, [block_idx, self._env.num_blocks]
        # assert np.sum(np.abs(offset)) > np.min(self._env.block_size) and len(offset) == 2, offset
        self._num_blocks = self._env.num_blocks
        # self._block_size = self._env.block_size
        self._block_idx = block_idx
        if self._block_idx < 0:
            assert -2 <= self._block_idx  # [-1 for drawer, -2 for cabinet]
            self._object_id = [self._env.cab_door_id, self._env.drawer_id][self._block_idx]
        else:
            self._object_id = self._env.objects[block_idx].id
        self._target_frame = rel_frame
        self._tolerance = tolerance
        self._ori_tolerance = ori_tolerance

    def _init_params_to_attrs(self, params):
        self._max_pos_vel = get_with_default(params, "max_pos_vel", 0.75, map_fn=np.asarray)  # m/s per axis
        self._max_ori_vel = get_with_default(params, "max_ori_vel", 10.0, map_fn=np.asarray)  # rad/s per axis
        self._smooth_vel_coef = get_with_default(params, "smooth_vel_coef", 0.8, map_fn=float)  # smooths controller out
        self._vel_noise = get_with_default(params, "vel_noise", 0.01, map_fn=float)  # smooths controller out
        self._use_intermediate_targets = get_with_default(params, "use_intermediate_targets",
                                                          False)  # smooths controller out

    def _init_setup(self):
        if self._env is not None:
            assert isinstance(self._env, BlockEnv3D), type(self._env)

    def is_obj_contacting_walls(self, obj_id, dir=None):
        # if dir is not None:
        #     dir = dir[:2]
        for cab_id in self._env.cabinet_obj_ids:
            cpts = p.getContactPoints(obj_id, cab_id, physicsClientId=self._env.id)
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

    @property
    def curr_name(self) -> str:
        # returns a string identifier for the policy, rather than ints.
        raise NotImplementedError


class Reach3DPrimitive(Block3DPrimitive):

    def reset_policy(self, kp=(30., 30., 5.), ko=30.0, kg=20.0, timeout=2000, ori_on_block=True, tip_servo=True,
                     target_gripper=0, smooth_noise=0., noise_period=5, min_noise_dist=0.075,
                     random_slow_prob=0, random_slow_duration_bounds=(5, 15), random_slowness_bounds=(0, 0.5),
                     stop_prob=0.33,
                     # half to one second slow/stop, with likelihood at each step (should be low)
                     mid_frames: List[CoordinateFrame] = (), **kwargs):
        super(Reach3DPrimitive, self).reset_policy(**kwargs)
        self._kp = np.array(kp)
        self._ko = np.array(ko)
        self._kg = np.array(kg)
        self._timeout = timeout

        self._target_gripper = target_gripper
        self._curr_target_frame = None

        self._mid_frames = list(mid_frames)
        self._num_mid_stages = len(self._mid_frames)
        self._curr_mid_stage = 0

        self._ori_on_block = ori_on_block

        self._latest_desired_frame = world_frame_3D
        self._latest_action_frame = world_frame_3D
        self._latest_desired_gripper = 0

        self._tip_in_ee_frame = self._env.tip_in_ee_frame

        self.num_steps = 0

        self.tip_servo = tip_servo
        self.smooth_noise = smooth_noise
        self._noise_obj = None
        self._noise_steps = 0
        self._noise2global = None
        self._curr_noisy_goal = np.zeros(3)

        self._min_noise_dist = min_noise_dist
        self._noise_period = noise_period
        self._latest_spatial_desired_frame = None

        # temporal
        self._random_slow_prob = random_slow_prob  # at each step, likelihood of entering a slow period.
        self._random_slow_duration_bounds = tuple(
            random_slow_duration_bounds)  # (low,high) how long to be slow for before returning to plan
        self._random_slowness_bounds = tuple(random_slowness_bounds)  # (low, high) what percentage of max speed to go
        self._stop_prob = stop_prob  # in a given slow interval, upweight the probability of stopping (slowness = 0) completely.

        self._curr_slowness = None  # 3-element AttrDict (curr_step, slowness, max_step)

    @staticmethod
    def clip_norm(arr, norm):
        return arr * min(norm / (np.linalg.norm(arr) + 1e-11), 1)

    @staticmethod
    def compute_delta(pos, ori, gripper, des_pos, des_ori, des_gripper):
        # state is 7 elements
        delta = des_pos - pos
        curr_mat = euler2mat(ori)
        des_mat = euler2mat(des_ori)
        ori_delta = orientation_error(des_mat, curr_mat)
        grab_pos_delta = des_gripper - gripper

        return delta, ori_delta, grab_pos_delta

    def update_noisy_goal(self, pos, ori, desired_tip_frame):

        delta = desired_tip_frame.pos - pos
        delta_mag = np.linalg.norm(delta)

        target_tip_frame = desired_tip_frame

        # spatial
        if self.smooth_noise > 0:
            if self._latest_spatial_desired_frame is None:
                self._latest_spatial_desired_frame = desired_tip_frame

            # smoothing logic when outside min_noise_dist from goal
            if delta_mag >= self._min_noise_dist:
                if self._noise_steps % self._noise_period == 0:
                    intermediate = pos + delta
                    perp_noise = delta_mag * self.smooth_noise  # some fraction of the distance to the goal is the noise we add
                    tangent_vec = np_project_points_onto_plane(np.random.rand(3)[None], delta / delta_mag, np.zeros(3))[
                        0]
                    # logger.debug(f"{tangent_vec}, {perp_noise}, {delta.dot(tangent_vec)}")
                    tangent_vec /= np.linalg.norm(tangent_vec)
                    perp_noise = np.random.uniform(0.1 * perp_noise, perp_noise)
                    self._curr_noisy_goal = intermediate + perp_noise * tangent_vec

                desired_pos = self._latest_spatial_desired_frame.pos * 0.5 + self._curr_noisy_goal * 0.5
                desired_tip_frame = CoordinateFrame(world_frame_3D, desired_tip_frame.rot.inv(), desired_pos)
                self._noise_steps += 1

            else:
                desired_pos = self._latest_spatial_desired_frame.pos * 0.5 + desired_tip_frame.pos * 0.5
                desired_tip_frame = CoordinateFrame(world_frame_3D, desired_tip_frame.rot.inv(), desired_pos)

            self._latest_spatial_desired_frame = desired_tip_frame  # copy before any other mutations

        # temporal, done after other things.
        if self._random_slow_prob > 0:

            if self._curr_slowness is None and random() < self._random_slow_prob:
                # new window
                # logger.debug(f"Starting new slow...{self._stage}")
                self._curr_slowness = d(
                    step=0,
                    slowness=0. if random() < self._stop_prob else np.random.uniform(*self._random_slowness_bounds),
                    max_step=np.random.randint(*self._random_slow_duration_bounds),
                    initial_pos=pos.copy(),
                    initial_ori=ori.copy(),
                )

            if self._curr_slowness is not None:
                step, slowness, max_step = self._curr_slowness.get_keys_required(['step', 'slowness', 'max_step'])
                step += 1
                self._curr_slowness.step = step
                self.num_steps -= 1  # this doesn't count as a true step
                if hasattr(self, "_curr_step"):
                    self._curr_step -= 1  # likewise
                if step >= max_step:
                    self._curr_slowness = None  # clear
                    return desired_tip_frame, target_tip_frame

                # compute closer delta (by slowness factor) for action frame
                if slowness < 1e-11:
                    pos_new = self._curr_slowness >> "initial_pos"  # stabilize
                    # oriq_new = euler2quat(ori)
                else:
                    pos_new = slowness * desired_tip_frame.pos + (1 - slowness) * pos
                    # oriq_new = quat_slerp(euler2quat(ori), desired_tip_frame.orn, slowness)

                # print(self._stage, pos, desired_tip_frame.pos, pos_new, step, max_step)

                # todo fix the rotation stuff
                desired_tip_frame = CoordinateFrame(world_frame_3D, desired_tip_frame.rot.inv(), pos_new)

        return desired_tip_frame, target_tip_frame

    def get_action(self, model, observation, goal, **kwargs):
        # get the pos, vel, and block pos's
        # print(observation.leaf_filter(lambda k, v: k != "image"))
        get_keys = ['ee_position', 'gripper_pos', 'ee_orientation_eul', 'objects/position', 'objects/orientation_eul']
        pos, gripper_pos, ori, blocks_pos, block_ori = (observation > get_keys).leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(get_keys)

        if self._block_idx == -2:  # cabinet
            h_pos, h_ori = observation.get_keys_required(['cabinet/handle_position', 'cabinet/handle_orientation_eul'])
            h_pos = to_numpy(h_pos[0, 0], check=True)
            h_ori = to_numpy(h_ori[0, 0], check=True)
            block_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", h_ori).inv(),
                                          h_pos)
        elif self._block_idx == -1:
            raise NotImplementedError
        else:
            block_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", block_ori[self._block_idx]).inv(),
                                          blocks_pos[self._block_idx])

        # progress forward if we have reached the last target frame.
        if self.has_reached(pos, ori) and self._curr_mid_stage < self._num_mid_stages:
            self._curr_mid_stage += 1

        self._curr_target_frame = (self._mid_frames + [self._target_frame])[self._curr_mid_stage]

        if self._ori_on_block:
            # the target frame is the transformation on the block frame of reference.
            desired_tip_frame = block_frame.apply_a_to_b(world_frame_3D, self._curr_target_frame)
        else:
            # target frame orientation is global.
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._curr_target_frame.rot.inv(),
                                                self._curr_target_frame.pos + block_frame.pos)

        if self.tip_servo:
            desired_tip_frame = desired_tip_frame.apply_a_to_b(self._tip_in_ee_frame, world_frame_3D)

        # logger.debug(f"pos: {desired_tip_frame.pos} | ori: {desired_tip_frame.rot.as_euler('xyz')}")
        # desired_tip_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", np.asarray([-3.14159265,  0.,         -1.6])).inv(), np.asarray([ 0.63370089, -0.00613581,  0.45066847]))

        # these are generated as mid points on the current path when not too close to the goal (min_noise_distance)

        desired_tip_frame, target_tip_frame = self.update_noisy_goal(pos, ori, desired_tip_frame)

        delta = desired_tip_frame.pos - pos  # pos error
        curr = R.from_euler("xyz", ori)
        ori_delta = orientation_error(desired_tip_frame.rot.as_matrix(), curr.as_matrix())  # orientation error
        grab_pos_delta = self._target_gripper - gripper_pos
        # pos_v = np.clip(self._kp * delta, -self._max_pos_vel * self._env.dt, self._max_pos_vel * self._env.dt)
        # ori_v = np.clip(self._kp * delta, -self._max_pos_vel * self._env.dt, self._max_pos_vel * self._env.dt)

        # v = self._smooth_vel_coef * v + (1 - self._smooth_vel_coef) * vel

        dx_pos = Reach3DPrimitive.clip_norm(self._kp * delta, self._max_pos_vel) * self._env.dt
        dx_ori = Reach3DPrimitive.clip_norm(self._ko * ori_delta, self._max_ori_vel) * self._env.dt
        dx_grip = Reach3DPrimitive.clip_norm(self._kg * grab_pos_delta, 150.) * self._env.dt

        # dx = np.concatenate([dx_pos, dx_ori, dx_grip])
        # # print(dx)
        #
        # dx[:3] += np.random.randn(3) * self._vel_noise

        # v = np.append(v, 0)  # grab
        # print(blocks_pos[self._block_idx], pos, "ACT:", v)
        self.num_steps += 1
        self._latest_desired_frame = target_tip_frame  # this is where we really want to go
        self._latest_action_frame = desired_tip_frame  # this is where we command to go
        self._latest_desired_gripper = self._target_gripper

        setpoint_pose = desired_tip_frame.as_pose(world_frame_3D)
        setpoint_grab = np.array([self._target_gripper])

        # the desired waypoint
        out = d(action=(np.concatenate(
            [pos + dx_pos, batch_orientation_eul_add(ori, dx_ori), gripper_pos + dx_grip]))[None])
        out.policy_type = np.array([4])[None]
        out.policy_name = np.array([self.curr_name])[None]
        out['target/ee_position'] = setpoint_pose[None, :3]
        out['target/ee_orientation_eul'] = setpoint_pose[None, 3:]
        out['target/gripper_pos'] = setpoint_grab[None]

        return out

    def has_reached(self, pos, ori):
        # has reached uses the TRUE target desired frame.
        return np.linalg.norm(
            self._latest_desired_frame.pos - pos) < self._tolerance and \
               np.linalg.norm(
                   orientation_error(self._latest_desired_frame.rot.as_matrix(), R.from_euler("xyz", ori).as_matrix())
               ) < self._ori_tolerance

    def is_terminated(self, model, observation, goal, timeout=None, **kwargs) -> bool:
        if timeout is None:
            timeout = self._timeout
        pos, gripper_pos, ori = observation.leaf_apply(lambda arr: to_numpy(arr[0], check=True)) \
            .get_keys_required(['ee_position', 'gripper_pos', 'ee_orientation_eul'])

        if "drawer" not in observation.keys():
            obj_pos = (observation >> "objects/position")[0].copy()
            if self._env._object_spec[0] == "mug":
                obj_pos[..., 2] += 0.02
            block_in_bounds = all(self._env.is_in_bounds(o) for o in obj_pos)
        else:
            block_in_bounds = True

        return not block_in_bounds or self.num_steps >= timeout or (
                self._curr_mid_stage == self._num_mid_stages and self.has_reached(pos, ori))

    @property
    def curr_name(self) -> str:
        return 'reach'


class ReachMoveRetreatPrimitive(Reach3DPrimitive):
    def reset_policy(self, move_steps=20, wait_steps=5, retreat_steps=0, retreat_direction=None, retreat_velocity=0.15,
                     uniform_velocity=False,
                     retreat_first=False,
                     stop_at_wall=True, num_motion_stages=1, reach_steps=None, **kwargs):
        super(ReachMoveRetreatPrimitive, self).reset_policy(**kwargs)
        self._move_steps = move_steps
        self._wait_steps = wait_steps
        self._reach_steps = self._timeout / 3 if reach_steps is None else reach_steps
        self._retreat_velocity = retreat_velocity
        self._uniform_velocity = uniform_velocity  # true changes how actions are computed to the online way
        self._curr_step = 0
        self._retreat_steps = retreat_steps
        self._retreat_direction = retreat_direction
        if self._retreat_steps > 0:
            self._retreat_direction = self._retreat_direction / np.linalg.norm(retreat_direction)
        self._stage = 0
        self._num_motion_stages = num_motion_stages
        self._stop_at_wall = stop_at_wall
        self._retreat_first = retreat_first

        if self._retreat_first:
            # start at the last stage
            assert self._retreat_steps > 0
            self._stage = self._num_motion_stages + 2

        self._base_ori = None  # will be set once reach is done
        self._base_pos = None  # will be set once reach is done
        self._base_grip = None  # will be set once reach is done
        self._targ_pos = None
        self._targ_ori = None
        self._targ_grip = None
        self._motion_dir = None

    def start_motion(self):
        # should set self._targ_pos for the motion phase (middle)
        # can optionally set self._motion_dir
        self._targ_pos = self._base_pos.copy()
        self._targ_ori = self._base_ori.copy()
        self._targ_grip = self._base_grip.copy()

    def motion_loop(self, observation) -> bool:
        # might update self._base_* and self._targ_pos
        #  or (self._stop_at_wall and self.is_obj_contacting_walls(self._object_id, dir=self._motion_dir))
        return True

    def end_motion(self):
        self._targ_pos = self._base_pos.copy()
        self._targ_ori = self._base_ori.copy()
        self._targ_grip = self._base_grip.copy()

    def get_action(self, model, observation, goal, **kwargs):
        observation = observation.leaf_apply(lambda arr: to_numpy(arr, check=True))
        setpoint_position = setpoint_ori = setpoint_grab = None
        ac = d()
        if self._stage == 0:
            # first reach
            ac = super(ReachMoveRetreatPrimitive, self).get_action(model, observation, goal, **kwargs)
            setpoint_position = (ac >> 'target/ee_position').copy()
            setpoint_ori = (ac >> 'target/ee_orientation_eul').copy()
            setpoint_grab = (ac >> 'target/gripper_pos').copy()
            # logger.debug(self.num_steps)
            if super(ReachMoveRetreatPrimitive, self).is_terminated(model,
                                                                    observation.leaf_apply(lambda arr: arr[:, 0]),
                                                                    goal.leaf_apply(lambda arr: arr[:, 0]),
                                                                    timeout=self._reach_steps, **kwargs):
                # logger.debug(f"REACH TERMINATED: {self.num_steps} / {self._reach_steps}")
                self._stage += 1
                self._curr_step = 0
                self._base_ori = setpoint_ori[0].copy()
                self._base_pos = setpoint_position[0].copy()
                self._base_grip = setpoint_grab[0].copy()
                # logger.debug(
                #     f"REACH -> EE ORN: {np.rad2deg(observation.ee_orientation_eul[0, 0])}, setpoint: {np.rad2deg(self._base_ori)}")
                self.start_motion()
        elif 0 < self._stage <= self._num_motion_stages:
            self._curr_step += 1
            done = self.motion_loop(observation)

            if done or self._curr_step > self._move_steps:
                self.end_motion()
                self._curr_step = 0
                self._stage += 1

            self.num_steps += 1
        elif self._stage == self._num_motion_stages + 1:
            # wait after
            self._curr_step += 1
            if self._curr_step > self._wait_steps:
                self._curr_step = 0
                self._stage += 2 if self._retreat_first else 1
                self._curr_slowness = None  # reset before retreat
            self.num_steps += 1
        else:
            # retreat
            if self._retreat_steps > 0:
                pos = (observation >> "ee_position")[0, 0]
                ori = (observation >> "ee_orientation_eul")[0, 0]
                if self._curr_step == 0:
                    self._noise_steps = 0  # in case noise is being used in retreat
                    self._last_spatial_desired_frame = None
                    if self._base_pos is None:
                        self._base_pos = (observation >> "ee_position")[0, 0].copy()
                        self._base_ori = (observation >> "ee_orientation_eul")[0, 0].copy()
                        self._base_grip = (observation >> "gripper_pos")[0, 0].copy()
                        self._targ_ori = self._base_ori.copy()
                        self._targ_grip = self._base_grip.copy()

                    if self._uniform_velocity:
                        self._retreat_targ_pos = self._base_pos + self._retreat_velocity * self._retreat_direction * self._env.dt  # steps used as timeout
                    else:
                        self._retreat_targ_pos = self._base_pos + self._retreat_velocity * self._retreat_direction * self._retreat_steps * self._env.dt

                    self._targ_pos = self._retreat_targ_pos.copy()

                # noisy retreat
                self._curr_noisy_goal = self._targ_pos.copy()
                desired_tip_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", self._targ_ori).inv(),
                                                    self._retreat_targ_pos.copy())
                desired_tip_frame, target_tip_frame = self.update_noisy_goal(pos, ori, desired_tip_frame)
                self._targ_pos = desired_tip_frame.pos.copy()
                if self._curr_slowness is None:
                    self._targ_pos[2] = self._retreat_targ_pos[2]  # z is important for retreat
                # print(self._targ_pos, pos, original_targ_pos)
                # targ ori?

                # logger.debug(f"RETREAT: {self._retreat_direction}, {original_targ_pos - pos}, {self._targ_pos - pos}")

                if self._uniform_velocity:
                    reached = np.linalg.norm(self._retreat_targ_pos - pos) <= self._tolerance
                    dx = Reach3DPrimitive.clip_norm(self._kp * (self._targ_pos - pos), self._max_pos_vel) * self._env.dt
                    dx += pos - self._base_pos  # rebase around current pos rather than base.
                else:
                    reached = False
                    dx = + self._retreat_velocity * self._retreat_direction.copy() * self._env.dt

                # much larger lateral noise on retreat
                # if not self._env.is_in_bounds():  # TODO
                #     dx = -dx  # bang bang control kinda
                # else:
                #     v = np.where(np.abs(v) < 0.1, 10 * self._vel_noise * np.random.randn(2), v)

                self._base_pos += dx
                self._curr_step += 1
                if self._curr_step > self._retreat_steps or reached:
                    self._curr_step = 0
                    self._targ_pos = self._retreat_targ_pos.copy()  # in case it changed
                    if self._retreat_first:
                        self._stage = 0
                    else:
                        self._stage += 1
            else:
                logger.warn("ReachMoveRetreat Policy is done but action requested...")

            self.num_steps += 1

        if setpoint_position is None:
            setpoint_position = self._targ_pos.copy()[None]
        if setpoint_ori is None:
            setpoint_ori = self._targ_ori.copy()[None]
        if setpoint_grab is None:
            setpoint_grab = self._targ_grip.copy()[None]

        if ac.is_empty():
            ac = d(action=np.concatenate([self._base_pos, self._base_ori, self._base_grip])[None])
        ac.policy_type = np.array([self.policy_type])[None]
        ac.policy_name = np.array([self.curr_name])[None]
        ac['target/ee_position'] = setpoint_position.copy()
        ac['target/ee_orientation_eul'] = setpoint_ori.copy()
        ac['target/gripper_pos'] = setpoint_grab.copy()

        return ac

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self._stage == 2 + self._num_motion_stages + int(
            self._retreat_steps > 0)

    def motion_dir_suffix(self) -> str:
        v = self._motion_dir
        if v is not None:
            theta = np.arctan2(v[1], v[0])
            if np.pi / 4 >= theta > -np.pi / 4:
                motion_dir = "_right"
            elif 3 * np.pi / 4 >= theta > np.pi / 4:
                motion_dir = "_forward"
            elif theta > 3 * np.pi / 4 or theta <= -3 * np.pi / 4:
                motion_dir = "_left"
            else:
                motion_dir = "_backward"
        else:
            motion_dir = ""
        return motion_dir

    @property
    def curr_name(self) -> str:
        return f'reach_move{self.motion_dir_suffix()}'

    @property
    def policy_type(self):
        return 6

class PushPrimitive(ReachMoveRetreatPrimitive):
    def reset_policy(self, push_steps=20, push_velocity=0.1, retreat_velocity=None, **kwargs):
        if retreat_velocity is None:
            retreat_velocity = push_velocity
        super(PushPrimitive, self).reset_policy(move_steps=push_steps, retreat_velocity=retreat_velocity, **kwargs)
        self._push_steps = push_steps
        self._push_direction = - self._target_frame.pos[:2] / np.linalg.norm(
            self._target_frame.pos[:2])  # push in the direction of the offset.
        self._push_direction = np.append(self._push_direction, 0)  # no z pushing
        self._push_velocity = push_velocity

        # this tells us the name.
        self._motion_dir = self._push_direction.copy()

    def start_motion(self):
        super(PushPrimitive, self).start_motion()
        # orn and grip do not change.
        if self._use_intermediate_targets:
            self._targ_pos = self._base_pos.copy()
        else:
            if self._uniform_velocity:
                self._targ_pos = self._base_pos + self._push_velocity * self._push_direction * self._env.dt
            else:
                self._targ_pos = self._base_pos + self._push_velocity * self._push_direction * self._env.dt * self._push_steps

    def motion_loop(self, observation) -> bool:
        if self._uniform_velocity:
            pos = (observation >> "ee_position")[0, 0]
            reached = np.linalg.norm(self._targ_pos - pos) <= self._tolerance
            dx = Reach3DPrimitive.clip_norm(self._kp * (self._targ_pos - pos), self._max_pos_vel) * self._env.dt
            dx += pos - self._base_pos  # rebase around current pos rather than base.
        else:
            reached = False
            dx = self._env.dt * self._push_velocity * self._push_direction.copy()  # move towards block

        self._base_pos += dx
        if self._use_intermediate_targets:
            self._targ_pos += dx

        # only stop early on contact
        return reached or (self._stop_at_wall and self.is_obj_contacting_walls(self._object_id, dir=self._motion_dir))

    @property
    def curr_name(self) -> str:
        return f'push{self.motion_dir_suffix()}'

    @property
    def policy_type(self):
        return 0


class PullPrimitive(ReachMoveRetreatPrimitive):
    def reset_policy(self, pull_steps=20, pull_velocity=0.1, pull_direction=np.zeros(2), pull_points=(),
                     retreat_velocity=None, grasp_delta=0, grasp_steps=20, wait_steps=5, pull_z=0, pull_tolerance_scale=1., xy_tolerance=False, prefix="", **kwargs):
        assert 'target_gripper' in kwargs.keys(), "Target gripper must be passed in"
        if retreat_velocity is None:
            retreat_velocity = pull_velocity
        super(PullPrimitive, self).reset_policy(move_steps=np.inf, retreat_velocity=retreat_velocity,
                                                num_motion_stages=3 + len(pull_points), wait_steps=0, **kwargs)
        self._pull_steps = pull_steps
        self._pull_velocity = pull_velocity
        self._retreat_velocity = retreat_velocity
        self._pull_direction = np.array(pull_direction)  # which way to pull (x,y)
        self._pull_direction = self._pull_direction / np.linalg.norm(self._pull_direction)
        self._pull_direction = np.append(self._pull_direction.copy(), pull_z)
        self._grasp_delta = grasp_delta  # grasping "delta" on top of reach target_gripper, default none
        self._grasp_steps = grasp_steps  # num steps to initiate the grasp
        self._grasp_open_steps = wait_steps  # num steps to end the grasp
        self._pull_points = list(pull_points)  # intermediate waypoints
        self._pull_tolerance_scale = pull_tolerance_scale
        self._xy_tolerance = xy_tolerance  # if true, will only consider xy tolerance for pulling waypoints

        if self._uniform_velocity:
            self._all_pull_points = self._pull_points + [self._pull_direction * self._pull_velocity * self._env.dt]
        else:
            assert len(self._pull_points) == 0, "not implemented"


        self._prefix = prefix

        self._start_grip = None
        self._mid_grip = None  # will be set once reach is done
        self._begin_pull_pos = None
        self._curr_direction = None

        self._motion_dir = self._pull_direction

    def start_motion(self):
        super(PullPrimitive, self).start_motion()
        self._start_grip = self._base_grip.copy()
        self._targ_grip = self._base_grip + self._grasp_delta

    def motion_loop(self, observation):
        is_grasped = ((observation >> "finger_left_contact") & (observation >> "finger_right_contact"))[0, 0, 0]
        if self._stage == 1:
            # initiate grasp (targ_grip)
            self._base_grip += (self._start_grip + self._grasp_delta) / self._grasp_steps
            self._base_grip = np.clip(self._base_grip, 0, 255)
            if self._curr_step > self._grasp_steps or is_grasped:
                self._curr_step = 0
                self._stage += 1
                # update targ pos

                if self._uniform_velocity:
                    self._targ_pos = self._base_pos + self._all_pull_points[0]
                    self._begin_pull_pos = self._base_pos.copy()  # everything is relative to this
                    self._curr_direction = self._all_pull_points[0] / np.linalg.norm(self._all_pull_points[0])
                else:
                    self._curr_direction = self._pull_direction
                    if self._use_intermediate_targets:
                        self._targ_pos = self._base_pos.copy()
                    else:
                        self._targ_pos = self._base_pos + self._pull_direction * self._pull_velocity * self._env.dt * self._pull_steps
                self._base_grip += 1  # margin for gripper after contact
                self._targ_grip = self._base_grip.copy()
                self._mid_grip = self._base_grip.copy()
                # print(self._targ_grip)

        elif 3 + len(self._pull_points) > self._stage >= 2:
            # pull in direction
            if self._uniform_velocity:
                pos = (observation >> "ee_position")[0, 0]
                delta = self._targ_pos - pos
                if self._xy_tolerance:
                    delta = delta[:2]
                reached = np.linalg.norm(delta) <= self._pull_tolerance_scale * self._tolerance
                dx = Reach3DPrimitive.clip_norm(self._kp * (self._targ_pos - pos), self._max_pos_vel) * self._env.dt
                dx += pos - self._base_pos  # rebase around current pos rather than base.
            else:
                reached = False
                dx = self._env.dt * self._pull_velocity * self._pull_direction.copy()  # move towards block

            self._base_pos += dx
            if self._use_intermediate_targets:
                self._targ_pos += dx

            if self._curr_step > self._pull_steps or reached or (
                    self._stop_at_wall and self.is_obj_contacting_walls(self._object_id, dir=self._curr_direction)):
                self._curr_step = 0
                if self._stage == 2 + len(self._all_pull_points) - 1:  # last stage
                    self._targ_pos = self._base_pos.copy()  # stop
                else:
                    self._targ_pos = self._begin_pull_pos + self._all_pull_points[1 + self._stage - 2]  # next stage
                    self._curr_direction = self._targ_pos - self._base_pos
                    self._curr_direction /= np.linalg.norm(self._curr_direction)
                self._targ_grip[:] = 0
                self._mid_grip = self._base_grip.copy()

                self._stage += 1

        elif self._stage == 3 + len(self._pull_points):
            # wait and self._pull_steps
            if self._curr_step > self._grasp_open_steps:
                # done
                self._targ_grip = self._base_grip.copy()
                return True
            else:
                self._base_grip += (0 - self._mid_grip) / self._grasp_open_steps

        # termination
        return False

    @property
    def curr_name(self) -> str:
        return f'{self._prefix}pull{self.motion_dir_suffix()}'

    @property
    def policy_type(self):
        return 1


class TopRotatePrimitive(ReachMoveRetreatPrimitive):
    def reset_policy(self, rotation_steps=10, rotation_velocity=0.5,
                     retreat_velocity=0.2, grasp_delta=0, grasp_steps=20, wait_steps=5, radius=np.zeros(2),
                     sweep_arc=False, sweep_tolerance_scale=1., stop_sweep_at_table=False, ptype=2, prefix="",
                     **kwargs):
        assert 'target_gripper' in kwargs.keys(), "Target gripper must be passed in"
        super(TopRotatePrimitive, self).reset_policy(move_steps=np.inf, retreat_velocity=retreat_velocity,
                                                     num_motion_stages=3, wait_steps=0, **kwargs)
        self._rotation_steps = rotation_steps
        self._rotation_velocity = rotation_velocity
        self._retreat_velocity = retreat_velocity
        self._grasp_delta = grasp_delta  # grasping "delta" on top of reach target_gripper, default none
        self._grasp_steps = grasp_steps  # num steps to initiate the grasp
        self._grasp_open_steps = wait_steps  # num steps to end the grasp

        self._radius = np.concatenate(
            [np.asarray(radius), [0]]).copy()  # ray from gripper to center of rotation circle.

        self._start_grip = None
        self._mid_grip = None  # will be set once reach is done

        self._sweep_arc = sweep_arc
        self._sweep_tolerance_scale = sweep_tolerance_scale

        self._stop_sweep_at_table = stop_sweep_at_table

        self._ptype = ptype
        self._prefix = prefix

    def start_motion(self):
        super(TopRotatePrimitive, self).start_motion()
        self._start_grip = self._base_grip.copy()
        self._targ_grip = self._base_grip + self._grasp_delta
        # logger.debug(self.curr_name)

    def motion_loop(self, observation):
        is_grasped = ((observation >> "finger_left_contact") & (observation >> "finger_right_contact"))[0, 0, 0]
        if self._stage == 1:
            # initiate grasp (targ_grip)
            self._base_grip += (self._start_grip + self._grasp_delta) / self._grasp_steps
            if self._curr_step > self._grasp_steps or is_grasped:
                self._curr_step = 0
                self._stage += 1
                # update targ pos

                self._targ_pos = self._base_pos.copy()
                if self._use_intermediate_targets:
                    self._targ_ori = self._base_ori.copy()
                    assert np.linalg.norm(self._radius) < 1e-4, "not implemented"
                else:
                    steps = 1 if self._uniform_velocity else self._rotation_steps
                    theta = -self._rotation_velocity * steps * self._env.dt
                    # theta = 0
                    if self._sweep_arc:
                        # will make sure position and orientation change together in an arc
                        self._sweep_theta = 0.  # keeps track of how far we are in arc
                        self._sweep_theta_max = theta
                        self._sweep_rot_base = R.from_euler("xyz", self._base_ori)
                        self._sweep_pos_base = self._targ_pos.copy()
                    else:
                        new_radius = R.from_euler("z", theta).apply(self._radius, inverse=True)

                        self._targ_ori = (R.from_euler(
                            "xyz", self._base_ori) * R.from_euler("z", theta)).as_euler("xyz")
                        self._targ_pos += new_radius - self._radius

                        # logger.debug(f" {self._radius}, {new_radius}")

                self._base_grip += 1  # margin for gripper after contact
                self._targ_grip = self._base_grip.copy()
                self._mid_grip = self._base_grip.copy()
                # print(self._targ_grip)

        elif self._stage == 2:
            # rotation in direction
            ori = (observation >> "ee_orientation_eul")[0, 0]
            pos = (observation >> "ee_position")[0, 0]

            if self._uniform_velocity:
                if self._sweep_arc:
                    self._sweep_theta = np.sign(self._sweep_theta_max) * min(
                        abs(self._sweep_theta) + 0.5 * self._env.dt, abs(self._sweep_theta_max))
                    rot_sweep_theta = R.from_euler("z", self._sweep_theta)
                    self._targ_ori = (self._sweep_rot_base * rot_sweep_theta).as_euler("xyz")
                    small_new_radius = rot_sweep_theta.apply(self._radius, inverse=True)
                    self._targ_pos = self._sweep_pos_base + small_new_radius - self._radius

                diff = orientation_error(euler2mat(self._targ_ori), euler2mat(ori))

                pos_diff = self._targ_pos - pos
                # print(diff, pos_diff)
                # logger.debug(diff)
                dori = Reach3DPrimitive.clip_norm(self._ko * diff, self._max_ori_vel) * self._env.dt
                dpos = Reach3DPrimitive.clip_norm(self._kp * pos_diff, self._max_pos_vel) * self._env.dt

                self._base_ori = batch_orientation_eul_add(dori, ori)
                self._base_pos = pos + dpos

                if not self._sweep_arc:
                    reached = np.linalg.norm(
                        diff) <= self._ori_tolerance  # and np.linalg.norm(pos_diff) <= self._tolerance
                elif abs(self._sweep_theta) >= abs(self._sweep_theta_max):
                    reached = np.linalg.norm(diff) <= self._sweep_tolerance_scale * self._ori_tolerance
                    if self._stop_sweep_at_table:
                        reached = reached or self._env.is_robot_contacting_table()  # stops sweeping if we hit the table
                else:
                    reached = False
            else:
                reached = False
                self._base_ori = (R.from_euler("z", -self._rotation_velocity * self._env.dt) * R.from_euler("xyz",
                                                                                                            self._base_ori)).as_euler(
                    "xyz")

            if self._use_intermediate_targets:
                self._targ_ori = self._base_ori.copy()

            if self._curr_step > self._rotation_steps or reached or (
                    self._stop_at_wall and self.is_obj_contacting_walls(self._object_id)):

                if self._curr_step < 5:
                    logger.debug(
                        f"Rotation ended super early!!! Might be a bug {reached} {self._rotation_velocity} {self._rotation_steps} {self._curr_step}")
                self._curr_step = 0
                self._targ_pos = self._base_pos = pos.copy()  # self._base_pos.copy()
                self._targ_ori = self._base_ori = ori.copy()  # self._base_ori.copy()
                self._targ_grip[:] = 0

                # logger.debug(f"reached orientation after: {np.rad2deg(self._base_ori)}")
                self._mid_grip = self._base_grip.copy()
                self._stage += 1

        elif self._stage == 3:
            # wait and self._pull_steps
            if self._curr_step > self._grasp_open_steps:
                # done
                self._targ_grip = self._base_grip.copy()
                return True
            else:
                self._base_grip += (0 - self._mid_grip) / self._grasp_open_steps

        # termination
        return False

    @property
    def curr_name(self) -> str:
        suffix = "right" if self._rotation_velocity > 0 else "left"
        return f'{self._prefix}top_rot_{suffix}'

    @property
    def policy_type(self):
        return self._ptype


class LiftPrimitive(ReachMoveRetreatPrimitive):
    def reset_policy(self, lift_steps=20, lift_velocity=0.1, lift_direction=np.zeros(3),
                     retreat_velocity=None, down_steps=0, down_velocity=1., down_direction=np.zeros(3), grasp_delta=0,
                     grasp_steps=20, wait_steps=5, prefix="", **kwargs):
        assert 'target_gripper' in kwargs.keys(), "Target gripper must be passed in"
        if retreat_velocity is None:
            retreat_velocity = lift_velocity
        super(LiftPrimitive, self).reset_policy(move_steps=np.inf, retreat_velocity=retreat_velocity,
                                                num_motion_stages=4, wait_steps=0, **kwargs)
        self._lift_steps = lift_steps
        self._down_steps = down_steps
        self._lift_velocity = lift_velocity
        self._down_velocity = down_velocity
        self._retreat_velocity = retreat_velocity
        self._lift_direction = np.array(lift_direction)  # which way to pull (x,y,z)
        self._down_direction = np.array(down_direction)  # which way to pull (x,y,z)
        assert self._lift_direction[2] > 0, "must pull up at least a bit"
        assert self._down_direction[2] <= 0, "must drop down"
        self._lift_direction = self._lift_direction / np.linalg.norm(self._lift_direction)
        self._down_direction = self._down_direction / np.linalg.norm(self._down_direction)

        # self._lift_direction = np.append(self._lift_direction, 0)
        self._grasp_delta = grasp_delta  # grasping "delta" on top of reach target_gripper, default none
        self._grasp_steps = grasp_steps  # num steps to initiate the grasp
        self._grasp_open_steps = wait_steps  # num steps to end the grasp

        self._prefix = prefix

        self._start_grip = None
        self._mid_grip = None  # will be set once reach is done
        self._motion_dir = self._lift_direction + self._down_direction * np.array([1, 1, 0])

    def start_motion(self):
        super(LiftPrimitive, self).start_motion()
        self._start_grip = self._base_grip.copy()
        self._targ_grip = self._base_grip + self._grasp_delta

    def motion_loop(self, observation):
        is_grasped = ((observation >> "finger_left_contact") & (observation >> "finger_right_contact"))[0, 0, 0]
        if self._stage == 1:
            # initiate grasp (targ_grip)
            self._base_grip += (self._start_grip + self._grasp_delta) / self._grasp_steps
            if self._curr_step > self._grasp_steps or is_grasped:
                self._curr_step = 0
                self._stage += 1
                # update targ pos

                if self._use_intermediate_targets:
                    self._targ_pos = self._base_pos.copy()
                else:
                    steps = 1 if self._uniform_velocity else self._lift_steps
                    self._targ_pos = self._base_pos + self._lift_direction * self._lift_velocity * self._env.dt * steps
                self._base_grip += 1  # margin for gripper after contact
                self._targ_grip = self._base_grip.copy()
                self._mid_grip = self._base_grip.copy()
                # print(self._targ_grip)

        elif self._stage == 2:
            # lift in direction

            if self._uniform_velocity:
                pos = (observation >> "ee_position")[0, 0]
                reached = np.linalg.norm(self._targ_pos - pos) <= 5 * self._tolerance
                dx = Reach3DPrimitive.clip_norm(self._kp * (self._targ_pos - pos), self._max_pos_vel) * self._env.dt
                dx += pos - self._base_pos  # rebase around current pos rather than base.
            else:
                reached = False
                dx = self._env.dt * self._lift_velocity * self._lift_direction.copy()  # move towards block

            self._base_pos += dx
            if self._use_intermediate_targets:
                self._targ_pos += dx

            if self._curr_step > self._lift_steps or reached:
                # or (self._stop_at_wall and self.is_obj_contacting_walls(self._object_id, dir=self._lift_direction)):
                self._curr_step = 0
                self._stage += 1
                steps = 1 if self._uniform_velocity else self._lift_steps
                self._targ_pos = self._base_pos + self._down_direction * self._down_velocity * self._env.dt * steps


        elif self._stage == 3:
            # down in direction

            if self._uniform_velocity:
                pos = (observation >> "ee_position")[0, 0]
                reached = np.linalg.norm(self._targ_pos - pos) <= 5 * self._tolerance
                dx = Reach3DPrimitive.clip_norm(self._kp * (self._targ_pos - pos), self._max_pos_vel) * self._env.dt
                dx += pos - self._base_pos  # rebase around current pos rather than base.
            else:
                reached = False
                dx = self._env.dt * self._down_velocity * self._down_direction.copy()  # move towards block

            self._base_pos += dx
            if self._use_intermediate_targets:
                self._targ_pos += dx

            if self._curr_step > self._down_steps or reached or (
                    self._stop_at_wall and self.is_obj_contacting_walls(self._object_id, dir=self._down_direction)):
                self._curr_step = 0
                self._targ_pos = self._base_pos.copy()
                self._targ_grip[:] = 0
                self._mid_grip = self._base_grip.copy()
                self._stage += 1

        elif self._stage == 4:
            # wait and self._pull_steps
            if self._curr_step > self._grasp_open_steps:
                # done
                self._targ_grip = self._base_grip.copy()
                return True
            else:
                self._base_grip += (0 - self._mid_grip) / self._grasp_open_steps

        # termination
        return False

    @property
    def curr_name(self) -> str:
        return f'{self._prefix}lift{self.motion_dir_suffix()}'

    @property
    def policy_type(self):
        return 3


def get_side_retreat_xy_dir(delta, epsilon=np.deg2rad(15)):
    # when you are on the side of a block, this will pick a random retreat direction in [-90+eps, 90-eps] relative to the delta ray
    base_retreat_dir = delta.copy()
    base_retreat_dir[2] = 0
    theta = np.random.uniform(-np.pi / 2 + epsilon, np.pi / 2 - epsilon)
    return R.from_euler("z", theta).apply(base_retreat_dir), theta


## DEFAULT initializers

def push_policy_params_fn(obs, goal, env=None, favor_center=True, push_steps=20, retreat_steps=20, retreat_velocity=0.2,
                          retreat_first=False, retreat_xy=False, push_velocity=0.1, with_platform=False,
                          uniform_velocity=False,
                          axis=None, soft_bound=False, pitch=0, smooth_noise=0, random_slow_prob=0., stop_prob=0.33):
    # rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, -np.pi / 2]),
    #                             np.array([0.05, 0, 0.17]))  # should push -x
    aabb = (obs >> "objects/aabb")[0, 0].reshape(2, 3)
    obj_pos = (obs >> "objects/position")[0, 0]

    if axis is None:
        if with_platform and isinstance(env, PlatformBlockEnv3D) and obj_pos[2] > env.platform_z:
            if obj_pos[0] > env.surface_center[0] + env.surface_bounds[0] / 2 - env._platform_extent or \
                    obj_pos[0] < env.surface_center[0] - env.surface_bounds[0] / 2 + env._platform_extent:
                which = np.random.choice([0, 1], p=[0.1, 0.9])
            else:
                which = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY
    delta = aabb[1, which] - aabb[0, which]
    grip_target = max(aabb[1, 2] - aabb[0, 2] - 0.01, 0.025)
    rot_yaw = -np.pi / 2 if which else 0

    # range computation (based on ray from start -> desired), to account for gripper rotation limits.
    ray = obj_pos[:2] - env.robot.start_pos[:2]
    # should be in range (90, 270)
    theta = np.arctan2(ray[1], ray[0]) % (2 * np.pi)

    # bound
    rot_yaw = np.where(rot_yaw > theta - np.pi / 2, rot_yaw - np.pi, rot_yaw)
    rot_yaw = np.where(rot_yaw < np.pi / 2 - theta, rot_yaw + np.pi, rot_yaw)

    # print(x_bb)
    offset = np.array([0, 0, grip_target])

    bpos = (obs >> "objects/position")[0, 0]

    if favor_center:
        assert env is not None
        block_ray = (bpos - env.free_surface_center)[:2]
        ps = [0.25, 0.75] if block_ray[which] > 0 else [0.75, 0.25]
        sign = np.random.choice([-1, 1], p=ps)
    else:
        sign = np.random.choice([-1, 1])

    offset[which] = delta * 1.3 * sign
    if retreat_xy:
        retreat_dir, retreat_theta = get_side_retreat_xy_dir(offset)
    else:
        retreat_dir = offset.copy()
    retreat_dir[2] = np.random.uniform(0.02, 0.07) if retreat_xy else np.random.uniform(0.06, 0.1)
    # print(which)
    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]) * R.from_euler("x", -pitch),
                                offset)  # should push -x

    if soft_bound:
        low, high = env._object_start_bounds[env._object_spec[0]]
        sb = env.surface_bounds.copy()
        upper = env.surface_center.copy()[:2]
        lower = env.surface_center.copy()[:2]

        upper += sb[:2] * high * 0.5
        lower += sb[:2] * low * 0.5

        motion_delta = - push_steps * push_velocity * env.dt * offset[:2] / np.linalg.norm(offset[:2])
        clipped_delta = np.clip(bpos[:2] + motion_delta, lower, upper) - bpos[:2]
        ratio = np.min(
            (np.abs(clipped_delta) + 1e-5) / (np.abs(motion_delta) + 1e-5))  # scale down the whole vec by this
        new_motion_delta = motion_delta * ratio

        push_steps = int(np.linalg.norm(new_motion_delta) / (push_velocity * env.dt))  # steps = dist / vel

    if retreat_first:
        p1 = obj_pos + offset  # where we are going
        p1[2] -= offset[2]
        p2 = (obs >> "gripper_tip_pos")[0]
        p2[2] -= offset[2]
        line = p2 - p1
        line2 = np.array([line[1], -line[0], 0])
        line2 /= np.linalg.norm(line2)
        # 1 cm margin on either side of line
        p1_left = p1 - 0.01 * line2
        p2_left = p1_left + line
        p1_right = p1 + 0.01 * line2
        p2_right = p1_right + line
        outs = p.rayTestBatch(rayFromPositions=[list(p1), list(p1_right), list(p1_left)],
                              rayToPositions=[list(p2), list(p2_right), list(p2_left)], physicsClientId=env.id)
        # p.addUserDebugLine(lineFromXYZ=list(p1), lineToXYZ=list(p2), lineColorRGB=[255, 0, 0], lineWidth=6, lifeTime=10.)
        # p.addUserDebugLine(lineFromXYZ=list(p1_left), lineToXYZ=list(p2_left), lineColorRGB=[0, 0, 255], lineWidth=6, lifeTime=10.)
        # p.addUserDebugLine(lineFromXYZ=list(p1_right), lineToXYZ=list(p2_right), lineColorRGB=[0, 0, 255], lineWidth=6, lifeTime=10.)
        # print(out)
        # print(len(out) > 0 and (out[0][0] in [env.objects[0].id] + env.cabinet_obj_ids, out))

        # if we don't retreat, will we hit the object?
        retreat_first = any(out[0] in [env.objects[0].id] + env.cabinet_obj_ids for out in outs)
        if not retreat_first:
            # don't retreat if we are in the clear
            retreat_steps = 0
        else:
            # if we need to retreat, move away from the block
            off = p2 - obj_pos
            off[2] = 0
            # clip norm of offset to be at least 1 cm, to incur some xy motion
            ret_dir, ret_theta = get_side_retreat_xy_dir(off / np.linalg.norm(off) * max(np.linalg.norm(off), 0.01))
            retreat_dir[:2] = ret_dir[:2]  # move away from object a bit if we are retreating

    kp = (30., 30., 5.)
    if uniform_velocity:
        push_velocity = push_velocity * push_steps
        push_steps = 40 * int(push_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(
        next_obs=obs, next_goal=goal, rel_frame=rel_frame, target_gripper=210, ori_on_block=False, kp=kp,
        retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
        push_steps=push_steps, retreat_first=retreat_first, push_velocity=push_velocity,
        uniform_velocity=uniform_velocity, smooth_noise=smooth_noise, random_slow_prob=random_slow_prob, stop_prob=stop_prob,
    )


def pull_policy_params_fn(obs, goal, env=None, favor_center=True, pull_steps=20, retreat_steps=20, retreat_velocity=0.2,
                          retreat_first=False, retreat_xy=False, pull_velocity=0.1, uniform_velocity=False,
                          axis=None):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    size = (obs >> "objects/size")[0, 0]

    idxs, new_coordinate_frame, best_axes, gripper_yaws = get_gripper_yaws(pos, orn_eul, size, env)

    if axis is None:
        which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY
    rot_yaw = -gripper_yaws[which]  # pick one of the gripping angles
    # rot_yaw = 0
    axis_norm = np.linalg.norm(best_axes[:, which])

    grip_width = max(255 * (1 - axis_norm / 0.08), 0) - 60  # overestimate of max grip width
    final_grip_width = 255.  # close until contact is noticed.

    # print((obs >> "objects/orientation_eul")[0, 0, 2], gripper_yaws, size, best_axes[:, 0], np.linalg.norm(best_axes[:, 0]))

    # current block height, width minus 2cm, for the z grip target
    grip_target_z = max(np.linalg.norm(new_coordinate_frame[:, idxs[2]]) - 0.03, 0.025)
    offset = np.array([0, 0, grip_target_z])

    thetas = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    if favor_center:
        assert env is not None
        bpos = (obs >> "objects/position")[0, 0]
        block_ray = (env.surface_center - bpos)[:2]
        block_ray_theta = np.arctan2(block_ray[1], block_ray[0])
        p = (np.pi - np.abs(circular_difference_fn(thetas, block_ray_theta))) + 5e-2  # base probability
        p = p / p.sum()
    else:
        p = np.ones(4)
    # random pull direction along cardinal xy axes
    random_theta = np.random.choice(thetas, p=p)
    pull_dir = np.array([np.cos(random_theta), np.sin(random_theta)])

    retreat_dir = np.array([0, 0, 1.])
    assert not retreat_xy, "not implemented"

    kp = (30., 30., 5.)
    if uniform_velocity:
        pull_velocity = pull_velocity * pull_steps
        pull_steps = 40 * int(pull_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                                offset)  # no offset
    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, pull_direction=pull_dir, pull_steps=pull_steps, pull_velocity=pull_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20)  # 2 seconds or grasped, whichever first


def lift_platform_policy_params_fn(obs, goal, env=None, favor_center=True, retreat_steps=12, retreat_velocity=0.2,
                                   retreat_first=False, retreat_xy=False, lift_velocity=0.1, uniform_velocity=False,
                                   sample_directions=False, axis=None, smooth_noise=0):
    assert isinstance(env, PlatformBlockEnv3D)
    assert not retreat_first, "not implemented"
    assert not retreat_xy, "not implemented"
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    size = (obs >> "objects/size")[0, 0]
    aabb = (obs >> "objects/aabb")[0, 0]

    idxs, new_coordinate_frame, best_axes, gripper_yaws = get_gripper_yaws(pos, orn_eul, size, env)

    if axis is None:
        which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY
    rot_yaw = -gripper_yaws[which]  # pick one of the gripping angles
    # rot_yaw = 0
    axis_norm = np.linalg.norm(best_axes[:, which])

    grip_width = 0.  # max(255 * (1 - axis_norm / 0.08), 0) - 60  # overestimate of max grip width
    final_grip_width = 255.  # close until contact is noticed.

    # print((obs >> "objects/orientation_eul")[0, 0, 2], gripper_yaws, size, best_axes[:, 0], np.linalg.norm(best_axes[:, 0]))

    # current block height, width minus 2cm, for the z grip target
    grip_target_z = max(np.linalg.norm(new_coordinate_frame[:, idxs[2]]) - 0.03, 0.025)
    offset = np.array([0, 0, grip_target_z])

    # thetas = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    # if favor_center:
    #     assert env is not None
    #     bpos = (obs >> "objects/position")[0, 0]
    #     block_ray = (env.surface_center - bpos)[:2]
    #     block_ray_theta = np.arctan2(block_ray[1], block_ray[0])
    #     p = (np.pi - np.abs(circular_difference_fn(thetas, block_ray_theta))) + 5e-2  # base probability
    #     p = p / p.sum()
    # else:
    #     p = np.ones(4)
    # random pull direction along cardinal xy axes
    # random_theta = np.random.choice(thetas, p=p)
    # pull_dir = np.array([np.cos(random_theta), np.sin(random_theta)])

    # lift up and to closest wall.

    # table_z = env.surface_center[2]
    if sample_directions:
        closest_pt, close_idx, distances, points = env.get_nearest_platform(
            (obs >> "objects").leaf_apply(lambda arr: arr[:, 0]), return_all=True)
        beta = 1.
        table_lens = np.repeat(env.surface_bounds, 2)
        soft = np.exp(-beta * distances / table_lens)
        p = soft / soft.sum()
        # update
        close_idx = np.random.choice(4, p=p)
        closest_pt = points[close_idx]
    else:
        closest_pt, _, _, _ = env.get_nearest_platform((obs >> "objects").leaf_apply(lambda arr: arr[:, 0]))

    # p.addUserDebugLine(lineFromXYZ=list(pos), lineToXYZ=list(closest_pt), lineColorRGB=[255,0,0], lineWidth=3,)
    # p.addUserDebugLine(lineFromXYZ=list(pos)[:2] + [env.platform_z], lineToXYZ=list(closest_pt)[:2] + [env.platform_z], lineColorRGB=[255,0,0], lineWidth=3,)

    obj_height = (aabb[5] - aabb[2])
    if pos[2] > env.platform_z:
        # lift to the table center if we are already on a platform.
        closest_pt = env.surface_center + obj_height / 2
        extent = np.random.uniform(-0.3, 0.3, 2)
        closest_pt[:2] += extent * env.surface_bounds / 2  # randomize xy a good amount

    closest_pt[2] += np.random.uniform(0.005, 0.015)  # margin for object above table

    # lifts to mid point, then comes down, extra amount is random (0.03 -> 0.08)
    lift_point = (closest_pt + pos) / 2
    lift_point[2] = env.platform_z + obj_height / 2 + np.random.uniform(0.05, 0.1)  # random elevation

    if uniform_velocity:
        lift_point[2] += 0.1  # since we need to clear the object
        closest_pt[2] += 0.025

    down_velocity = lift_velocity
    lift_dir = lift_point - pos
    down_dir = closest_pt - lift_point
    lift_steps = int(np.ceil(np.linalg.norm(lift_dir) / (lift_velocity * env.dt)))
    down_steps = int(np.ceil(np.linalg.norm(down_dir) / (down_velocity * env.dt)))

    # directional retreat, inwards
    retreat_dir = pos - closest_pt
    retreat_dir, _ = get_side_retreat_xy_dir(retreat_dir)
    retreat_dir = retreat_dir / np.linalg.norm(retreat_dir)
    retreat_dir[2] = 1.  # retreat up too
    if uniform_velocity:
        retreat_dir[2] = 2.  # retreat up more

    kp = (30., 30., 5.)
    if uniform_velocity:
        lift_velocity = lift_velocity * lift_steps
        lift_steps = 40 * int(lift_steps > 0)
        down_velocity = down_velocity * down_steps
        down_steps = 40 * int(down_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                                offset)  # no offset
    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, lift_direction=lift_dir, lift_steps=lift_steps, lift_velocity=lift_velocity,
             down_direction=down_dir, down_steps=down_steps, down_velocity=down_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20,
             smooth_noise=smooth_noise)  # TODO  # 2 seconds or grasped, whichever first


def get_gripper_yaws(obj_pos, obj_eul, obj_size, env, get_thetas=False):
    obj_rot = R.from_euler("xyz", obj_eul)
    # rotate size=
    vectors = np.diag(obj_size)
    new_coordinate_frame = obj_rot.apply(vectors)
    idxs = np.argsort(np.abs(new_coordinate_frame[2, :]))  # sort by z contribution (last row) in global frame
    best_axes = new_coordinate_frame[:, idxs[:2]]  # smallest z axes are the grasping axes
    # y / x
    gripper_yaws = np.arctan2(best_axes[1, :], best_axes[0, :])
    # gripper_yaws = np.where(gripper_yaws > np.pi / 4, gripper_yaws - np.pi,
    #                         gripper_yaws)  # so we don't predict a bad yaw ever.

    # range computation (based on ray from start -> desired), to account for gripper rotation limits.
    ray = obj_pos[:2] - env.robot.start_pos[:2]
    # should be in range (90, 270)
    theta = np.arctan2(ray[1], ray[0]) % (2 * np.pi)


    gripper_yaws = np.where(gripper_yaws > theta - np.pi / 2, gripper_yaws - np.pi, gripper_yaws)
    gripper_yaws = np.where(gripper_yaws < np.pi / 2 - theta, gripper_yaws + np.pi, gripper_yaws)

    if get_thetas:
        # positive_theta_max, negative_theta_max
        return idxs, new_coordinate_frame, best_axes, gripper_yaws, (theta - np.pi / 2, np.pi / 2 - theta)
    return idxs, new_coordinate_frame, best_axes, gripper_yaws

# def get_gripper_yaws(obj_pos, obj_eul, obj_size, env, get_thetas=False):
#     obj_rot = R.from_euler("xyz", obj_eul)
#     # rotate size=
#     vectors = np.diag(obj_size)
#     new_coordinate_frame = obj_rot.apply(vectors)
#     idxs = np.argsort(np.abs(new_coordinate_frame[2, :]))  # sort by z contribution (last row) in global frame
#     best_axes = new_coordinate_frame[:, idxs[:2]]  # smallest z axes are the grasping axes
#     # y / x
#     gripper_yaws = np.arctan2(best_axes[1, :], best_axes[0, :])
#     # gripper_yaws = np.where(gripper_yaws > np.pi / 4, gripper_yaws - np.pi,
#     #                         gripper_yaws)  # so we don't predict a bad yaw ever.
#
#     # range computation (based on ray from start -> desired), to account for gripper rotation limits.
#     ray = obj_pos[:2] - env.robot.start_pos[:2]
#     # should be in range (180, 360)
#     theta = np.arctan2(ray[1], ray[0]) % (2 * np.pi)
#
#     # gripper_yaws = np.where(gripper_yaws > theta - np.pi / 2, gripper_yaws - np.pi, gripper_yaws)
#     # gripper_yaws = np.where(gripper_yaws < np.pi / 2 - theta, gripper_yaws + np.pi, gripper_yaws)
#     gripper_yaws = np.where(gripper_yaws < theta - 2 * np.pi, gripper_yaws + np.pi, gripper_yaws)
#     gripper_yaws = np.where(gripper_yaws > theta - np.pi, gripper_yaws - np.pi, gripper_yaws)
#
#     if get_thetas:
#         # positive_theta_max, negative_theta_max
#         return idxs, new_coordinate_frame, best_axes, gripper_yaws, (theta - 2 * np.pi, theta - np.pi)  # np.pi / 2 - theta
#     return idxs, new_coordinate_frame, best_axes, gripper_yaws

def push_directional_policy_params_fn(obs, goal, env=None, favor_center=True, push_steps=20, retreat_steps=10,
                                      retreat_first=False, retreat_xy=False, with_platform=False,
                                      uniform_velocity=False,
                                      push_velocity=0.1, retreat_velocity=0.2, push_theta_noise=0.25, axis=None,
                                      smooth_noise=0):
    # rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, -np.pi / 2]),
    #                             np.array([0.05, 0, 0.17]))  # should push -x

    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    pos = (obs >> "objects/position")[0, 0]
    aabb = (obs >> "objects/aabb")[0, 0].reshape(2, 3)
    size = (obs >> "objects/size")[0, 0]

    _, _, _, gripper_yaws = get_gripper_yaws(pos, orn_eul, size, env)

    # targpos = (obs >> "objects/position")[0, 0].copy()
    # targpos[2] += 0.2
    #
    # gripper_yaws = convert_to_feasible_gripper_yaws(gripper_yaws, targpos, env)

    if axis is None:
        if with_platform and isinstance(env, PlatformBlockEnv3D) and pos[2] > env.platform_z:
            if pos[0] > env.surface_center[0] + env.surface_bounds[0] / 2 - env._platform_extent or \
                    pos[0] < env.surface_center[0] - env.surface_bounds[0] / 2 + env._platform_extent:
                which = np.random.choice([0, 1], p=[0.25, 0.75])
            else:
                which = np.random.choice([0, 1], p=[0.75, 0.25])
        else:
            which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY

    # delta = aabb[1, which] - aabb[0, which]
    delta = size[which] / 2 + np.random.uniform(0.02, 0.03)  # distance from gripper to object center. 3cm margin.
    grip_target_z = max(aabb[1, 2] - aabb[0, 2] - 0.01, 0.025)  # z
    # grip_target_z = 0.1  # TODO remove

    grip_yaw = -gripper_yaws[which]  # pick one of the rotation angles.
    rot_yaw = grip_yaw + np.random.uniform(-push_theta_noise, push_theta_noise)  # noisy theta, in radians
    # print(x_bb)
    offset = np.array([0, 0, grip_target_z])

    if favor_center:
        assert env is not None
        bpos = (obs >> "objects/position")[0, 0]
        block_ray = (bpos - env.surface_center)[:2]
        p = [0.25, 0.75] if block_ray[which] > 0 else [0.75, 0.25]
        sign = np.random.choice([-1, 1], p=p)
    else:
        sign = np.random.choice([-1, 1])

    # set x,y offset based on rotation yaw.
    offset[0] = delta * sign * np.cos(rot_yaw)
    offset[1] = delta * sign * np.sin(rot_yaw)

    # grip_yaw = np.random.choice([-np.pi/2, np.pi/4])
    # grip_yaw = np.random.choice([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    # logger.debug(np.rad2deg(grip_yaw))

    kp = (30., 30., 7.5)
    if uniform_velocity:
        push_velocity = push_velocity * push_steps
        push_steps = 40 * int(push_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    if retreat_xy:
        retreat_dir, retreat_theta = get_side_retreat_xy_dir(offset)
    else:
        retreat_dir = offset.copy()
    retreat_dir[2] = np.random.uniform(0.02, 0.07) if retreat_xy else np.random.uniform(0.05, 0.1)
    # print(which)
    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, grip_yaw]),
                                offset)  # should push -x
    return d(
        next_obs=obs, next_goal=goal, rel_frame=rel_frame, target_gripper=210, ori_on_block=False, kp=kp,
        retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first, push_steps=push_steps,
        push_velocity=push_velocity, uniform_velocity=uniform_velocity,
        retreat_velocity=retreat_velocity, smooth_noise=smooth_noise,  # TODO
    )


def pull_directional_policy_params_fn(obs, goal, env=None, favor_center=True, pull_steps=20, retreat_steps=10,
                                      retreat_first=False, retreat_xy=False, retreat_velocity=0.2,
                                      uniform_velocity=False,
                                      pull_velocity=0.1, pull_theta_noise=0.25, axis=None):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    # aabb = (obs >> "objects/aabb")[0, 0].reshape(2, 3)
    size = (obs >> "objects/size")[0, 0]

    idxs, new_coordinate_frame, best_axes, gripper_yaws = get_gripper_yaws(pos, orn_eul, size, env)

    if axis is None:
        which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY
    rot_yaw = -gripper_yaws[which]  # pick one of the gripping angles
    # print(rot_yaw)
    # rot_yaw = 0
    axis_norm = np.linalg.norm(best_axes[:, which])

    grip_width = 1.  # large overestimate
    final_grip_width = 255.  # close until contact is noticed.

    # print((obs >> "objects/orientation_eul")[0, 0, 2], gripper_yaws, size, best_axes[:, 0], np.linalg.norm(best_axes[:, 0]))

    # current block height, width minus 2cm, for the z grip target
    grip_target_z = max(np.linalg.norm(new_coordinate_frame[:, idxs[2]]) - 0.03, 0.025)
    offset = np.array([0, 0, grip_target_z])

    thetas = rot_yaw + np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    if favor_center:
        assert env is not None
        bpos = (obs >> "objects/position")[0, 0]
        block_ray = (env.surface_center - bpos)[:2]
        block_ray_theta = np.arctan2(block_ray[1], block_ray[0])
        p = (np.pi - np.abs(circular_difference_fn(thetas, block_ray_theta))) + 5e-2  # base probability
        p = p / p.sum()
    else:
        p = np.ones(4)

    # random pull direction along cardinal xy axes
    random_theta = np.random.choice(thetas, p=p) + np.random.uniform(-pull_theta_noise, pull_theta_noise)
    pull_dir = np.array([np.cos(random_theta), np.sin(random_theta)])

    retreat_dir = np.array([0, 0, 1.])
    assert not retreat_xy, "not implemented"

    kp = (30., 30., 7.5)
    if uniform_velocity:
        pull_velocity = pull_velocity * pull_steps
        pull_steps = 40 * int(pull_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                                offset)  # no offset
    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, pull_direction=pull_dir, pull_steps=pull_steps, pull_velocity=pull_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first,
             ori_on_block=False, retreat_velocity=retreat_velocity, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20)  # 2 seconds or grasped, whichever first


def top_rot_directional_policy_params_fn(obs, goal, env=None, rotation_steps=10, rotation_velocity=0.5,
                                         retreat_velocity=0.2,
                                         retreat_steps=10, retreat_first=False, retreat_xy=False, stop_at_wall=False,
                                         uniform_velocity=False, axis=None, smooth_noise=0):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    # aabb = (obs >> "objects/aabb")[0, 0].reshape(2, 3)
    size = (obs >> "objects/size")[0, 0]

    idxs, new_coordinate_frame, best_axes, gripper_yaws, thetas = get_gripper_yaws(pos, orn_eul, size, env,
                                                                                   get_thetas=True)

    if axis is None:
        which = np.random.randint(0, 2)
    else:
        which = axis  # predefined x or y ONLY
    rot_yaw = -gripper_yaws[which]  # pick one of the gripping angles
    # print(rot_yaw)
    # rot_yaw = 0
    # axis_norm = np.linalg.norm(best_axes[:, which])

    grip_width = 1.  # large overestimate
    final_grip_width = 255.  # close until contact is noticed.

    # print((obs >> "objects/orientation_eul")[0, 0, 2], gripper_yaws, size, best_axes[:, 0], np.linalg.norm(best_axes[:, 0]))

    # current block height, width minus 2cm, for the z grip target
    grip_target_z = max(np.linalg.norm(new_coordinate_frame[:, idxs[2]]) - 0.03, 0.02)
    offset = np.array([0, 0, grip_target_z])

    # thetas = rot_yaw + np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    #
    # p = np.ones(2)
    #
    # # random pull direction along relative xy axes
    # random_theta = np.random.choice(thetas, p=p) + np.random.uniform(-pull_theta_noise, pull_theta_noise)
    # pull_dir = np.array([np.cos(random_theta), np.sin(random_theta)])

    delta_obj = (obs >> "gripper_tip_pos")[0] - pos
    if retreat_xy and np.linalg.norm(delta_obj) > 0.01:
        retreat_dir, _ = get_side_retreat_xy_dir(delta_obj)
        retreat_dir[2] = 1.
    else:
        retreat_dir = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 1.])

    if retreat_xy:
        retreat_dir[2] = np.random.uniform(0.1, 0.2)

    # after rotation, we need to limit
    new_grip_yaw = gripper_yaws[which] + rotation_steps * env.dt * rotation_velocity
    # logger.debug(f"grip_yaw before: {np.rad2deg(new_grip_yaw)}. starting from {np.rad2deg(gripper_yaws[which])}")
    # if thetas[1] >= thetas[0]:
    #     new_grip_yaw = np.clip(new_grip_yaw, thetas[0], thetas[1])
    # else:
    #     new_grip_yaw = np.clip(new_grip_yaw, thetas[1], thetas[0])
    mid = thetas[0] + np.pi/2
    new_grip_yaw = np.clip(new_grip_yaw, mid - 2*np.pi, mid - np.pi)

    # logger.debug(f"grip_yaw after: {np.rad2deg(new_grip_yaw)}")

    # rotation amount is limited by the max and min gripper rotations at the given rotation.
    rotation_steps = int(np.floor((new_grip_yaw - gripper_yaws[which]) / (env.dt * rotation_velocity)))
    # print(f"{rotation_steps} steps, from {np.rad2deg(gripper_yaws[which])} -> {np.rad2deg(new_grip_yaw)})")
    # print(f" -- thetas: ({thetas[0]}, {thetas[1]})")

    kp = (30., 30., 5.)
    if uniform_velocity:
        rotation_velocity = rotation_velocity * rotation_steps
        rotation_steps = 25 * int(rotation_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    current_rot_yaw = (obs >> "ee_orientation_eul")[0, 2]  # z rotation
    current_rot_yaw = (current_rot_yaw + np.pi) % (2 * np.pi) - np.pi

    rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                                offset)  # no offset
    mid_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, current_rot_yaw]),
                                offset + np.array([0, 0, np.random.uniform(0.1, 0.15)]))
    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp, uniform_velocity=uniform_velocity,
             target_gripper=grip_width, rotation_steps=rotation_steps, rotation_velocity=rotation_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first,
             ori_on_block=False, retreat_velocity=retreat_velocity, mid_frames=[mid_frame],
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, stop_at_wall=stop_at_wall,
             smooth_noise=smooth_noise)  # 2 seconds or grasped, whichever first


if __name__ == "__main__":
    from sbrl.envs.param_spec import ParamEnvSpec

    # use_meta = True
    # rotate = True
    # no_push_pull = True

    env_spec_params = get_block3d_example_spec_params()
    env_params = get_block3d_example_params()
    env_params.render = True
    # env_params.block_size = (30, 30)

    env_params.debug_cam_dist = 0.35
    env_params.debug_cam_p = -45
    env_params.debug_cam_y = 0
    env_params.debug_cam_target_pos = [0.4, 0, 0.45]

    uniform_vel = True
    max_pos_vel = 0.4 if uniform_vel else 0.75
    # extra_dc = d()
    # if uniform_vel:
    #     extra_dc.kp = (30., 30., 10.)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    block_env = PlatformBlockEnv3D(env_params, env_spec)
    # block_env = BlockEnv3D(env_params, env_spec)

    # env presets
    # presets = d()
    presets = d(objects=d(position=np.array([0.4, 0.1, 0.35])[None], orientation_eul=np.array([0., 0., 0.])[None],
                          size=np.array([0.032, 0.043, 0.03])[None]))

    model = Model(d(ignore_inputs=True), env_spec, None)

    # policy = PushPrimitive(d(vel_noise=0), env_spec, env=block_env)
    # policy = PullPrimitive(d(vel_noise=0), env_spec, env=block_env)
    policy = LiftPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy_params_fn = push_policy_params_fn
    # policy_params_fn = pull_policy_params_fn
    policy_params_fn = lambda *args, **kwargs: lift_platform_policy_params_fn(*args, **kwargs,
                                                                              uniform_velocity=uniform_vel)

    # target frame
    # default is facing the block
    obs, goal = block_env.user_input_reset(1, presets=presets)  # trolling with a fake UI
    policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())

    iters = 10
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    i = 0
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))
        obs, goal, done = block_env.step(act)
        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
            i += 1
