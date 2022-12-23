import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from sbrl.envs.bullet_envs.block3d.block_env_3d import get_block3d_example_params, get_block3d_example_spec_params, \
    BlockEnv3D
from sbrl.envs.bullet_envs.block3d.platform_block_env_3d import PlatformBlockEnv3D
from sbrl.envs.bullet_envs.block3d.playroom import DrawerPlayroomEnv3D, get_playroom3d_example_params, \
    get_playroom3d_example_spec_params
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.blocks.block3d_policies import Reach3DPrimitive, PullPrimitive, TopRotatePrimitive, \
    get_side_retreat_xy_dir
from sbrl.policies.blocks.stack_block2d_success_metrics import circular_difference_fn
from sbrl.policies.policy import Policy
from sbrl.utils.control_utils import orientation_error
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import get_with_default, AttrDict as d
from sbrl.utils.torch_utils import to_numpy
from sbrl.utils.transform_utils import euler2mat


class DrawerMovePrimitive(Policy):

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, rel_frame: CoordinateFrame = world_frame_3D, mid_offset=np.array([0., 0., 0.1]), move_delta=0.1,
                     retreat_delta=np.array([0, 0.1, 0]),
                     tolerance=0.005, ori_tolerance=0.05, kp=(30., 30., 10.), ko=40.0, kg=10.0,
                     timeout=2000, stage_timeouts=np.array([80, 50, 80, 50]), tip_servo=True,
                     target_gripper=250, open_gripper=50, **kwargs):

        self._object_id = self._env.drawer_id
        # tip position relative to the drawer grasp point.
        self._target_frame = rel_frame
        self._mid_offset = np.asarray(mid_offset)
        self._move_delta = move_delta  # relative amount of motion relative to start.
        self._retreat_delta = np.asarray(retreat_delta)
        self._tolerance = tolerance
        self._ori_tolerance = ori_tolerance

        self._kp = np.array(kp)
        self._ko = np.array(ko)
        self._kg = np.array(kg)
        self._timeout = timeout
        self._stage_timeouts = np.asarray(stage_timeouts).astype(dtype=np.int)
        assert len(self._stage_timeouts) == 7

        self._target_gripper = float(target_gripper)
        self._open_gripper = float(open_gripper)

        self._latest_desired_frame = world_frame_3D
        self._latest_desired_gripper = 0

        self._tip_in_ee_frame = self._env.tip_in_ee_frame

        self.num_steps = 0

        self.tip_servo = tip_servo

        # counters
        self.stage = 0
        self.num_steps = 0  # total
        self._curr_step = 0  # within stage

        self._target_grasp_point_offset = None

    def _init_params_to_attrs(self, params):
        self._max_pos_vel = get_with_default(params, "max_pos_vel", 0.75, map_fn=np.asarray)  # m/s per axis
        self._max_ori_vel = get_with_default(params, "max_ori_vel", 10.0, map_fn=np.asarray)  # rad/s per axis
        self._smooth_vel_coef = get_with_default(params, "smooth_vel_coef", 0.8, map_fn=float)  # smooths controller out
        self._vel_noise = get_with_default(params, "vel_noise", 0.01, map_fn=float)  # smooths controller out
        self._use_intermediate_targets = get_with_default(params, "use_intermediate_targets",
                                                          False)  # smooths controller out

    def _init_setup(self):
        if self._env is not None:
            assert isinstance(self._env, DrawerPlayroomEnv3D), type(self._env)

        self.stage = 0
        self.num_steps = 0
        self._curr_step = 0

    def reached(self, curr_pos, curr_ori, des_pos, des_ori):
        dpos = np.linalg.norm(des_pos - curr_pos)
        dori = np.linalg.norm(orientation_error(euler2mat(des_ori), euler2mat(curr_ori)))
        # print(self.stage, dpos, dori)
        return dpos < self._tolerance and dori < self._ori_tolerance

    def get_action(self, model, observation, goal, **kwargs):
        pos, gripper_pos, gripper_tip_pos, ori, drawer_posn, drawer_aabb = observation.leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(
            ['ee_position', 'gripper_pos', 'gripper_tip_pos', 'ee_orientation_eul', 'drawer/joint_position_normalized', 'drawer/aabb'])

        desired_tip_frame = None
        desired_gripper = None

        curr_pos = CoordinateFrame.point_from_a_to_b(pos, world_frame_3D, self._tip_in_ee_frame) if self.tip_servo else pos
        curr_ori = ori

        daabb = drawer_aabb.reshape(2, 3)
        # dlens = daabb[1, :2] - daabb[0, :2]
        # center = (daabb[1] + daabb[0]) / 2
        # p.addUserDebugLine(center + np.array([dlens[0] / 2, dlens[1] / 2, 0]), center + np.array([dlens[0] / 2, -dlens[1] / 2, 0]), lineWidth=100., lifeTime=self._env.dt)
        # p.addUserDebugLine(center + np.array([dlens[0] / 2, -dlens[1] / 2, 0]), center + np.array([-dlens[0] / 2, -dlens[1] / 2, 0]), lineWidth=100., lifeTime=self._env.dt)
        # p.addUserDebugLine(center + np.array([-dlens[0] / 2, -dlens[1] / 2, 0]), center + np.array([-dlens[0] / 2, dlens[1] / 2, 0]), lineWidth=100., lifeTime=self._env.dt)
        # p.addUserDebugLine(center + np.array([-dlens[0] / 2, dlens[1] / 2, 0]), center + np.array([dlens[0] / 2, dlens[1] / 2, 0]), lineWidth=100., lifeTime=self._env.dt)

        if self.stage < 2 or self._target_grasp_point_offset is None:
            y_max = np.max(daabb[:, 1])
            x_mid = (daabb[0, 0] + daabb[1, 0]) / 2
            z_mid = (daabb[0, 2] + daabb[1, 2]) / 2
            self._target_grasp_point_offset = np.array([x_mid, y_max, z_mid])

        ## REACH UP ##
        if self.stage == 0:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_frame.pos + self._target_grasp_point_offset + self._mid_offset)
            desired_gripper = self._open_gripper

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")) \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

        ## REACH DOWN ##
        if self.stage == 1:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_frame.pos + self._target_grasp_point_offset)
            desired_gripper = self._open_gripper

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")) \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

        ## GRASP handle ##
        elif self.stage == 2:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_frame.pos + self._target_grasp_point_offset)

            desired_gripper = self._target_gripper  # closed amount

            self._curr_step += 1
            is_grasped = ((observation >> "finger_left_contact") & (observation >> "finger_right_contact"))[0, 0, 0]
            if abs(gripper_pos[0] - desired_gripper) < 10 \
                    or is_grasped\
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1
                self._target_gripper = gripper_pos[0]  # stop here

        ## MOVE IN DIRECTION ##
        elif self.stage == 3:
            # new target in y-axis (move_delta)
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_frame.pos + self._target_grasp_point_offset + np.array([0, self._move_delta, 0]))

            desired_gripper = self._target_gripper  # closed amount

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")) \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

                self._targ_pos_after = gripper_tip_pos + np.array([0, 0, 0.03])

        ## UNGRASP ##
        elif self.stage == 4:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(), self._targ_pos_after)

            desired_gripper = self._open_gripper
            self._curr_step += 1
            if abs(gripper_pos[0] - desired_gripper) < 10 \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

        ## RETREAT UP  ##
        elif self.stage == 5:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._targ_pos_after + self._mid_offset)
            desired_gripper = self._open_gripper

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")) \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

        ## RETREAT OUT ##
        elif self.stage == 6:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_frame.pos + self._target_grasp_point_offset + np.array([0, self._move_delta, 0])
                                                + self._mid_offset + self._retreat_delta)
            desired_gripper = self._open_gripper

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")) \
                    or self._curr_step >= self._stage_timeouts[self.stage]:
                self._curr_step = 0
                self.stage += 1

        if self.tip_servo:
            desired_tip_frame = desired_tip_frame.apply_a_to_b(self._tip_in_ee_frame, world_frame_3D)

        # p.addUserDebugLine(list(pos), list(desired_tip_frame.pos), lifeTime=self._env.dt)
        # servo to the correct position
        delta = desired_tip_frame.pos - pos
        curr = R.from_euler("xyz", ori)
        ori_delta = orientation_error(desired_tip_frame.rot.as_matrix(), curr.as_matrix())
        grab_pos_delta = desired_gripper - gripper_pos

        dx_pos = Reach3DPrimitive.clip_norm(self._kp * delta, self._max_pos_vel) * self._env.dt
        dx_ori = Reach3DPrimitive.clip_norm(self._ko * ori_delta, self._max_ori_vel) * self._env.dt
        dx_grip = Reach3DPrimitive.clip_norm(self._kg * grab_pos_delta, 150.) * self._env.dt

        self.num_steps += 1
        setpoint_pose = desired_tip_frame.as_pose(world_frame_3D)
        setpoint_grab = np.array([desired_gripper])

        # the desired waypoint
        out = d(action=(np.concatenate(
            [pos + dx_pos, (R.from_euler("xyz", dx_ori) * curr).as_euler("xyz"), gripper_pos + dx_grip]))[None])
        out.policy_type = np.array([self.policy_type])[None]
        out.policy_name = np.array([self.curr_name])[None]
        out['target/ee_position'] = setpoint_pose[None, :3]
        out['target/ee_orientation_eul'] = setpoint_pose[None, 3:]
        out['target/gripper_pos'] = setpoint_grab[None]
        return out

    @property
    def curr_name(self) -> str:
        motion_dir = "open" if self._move_delta > 0 else "close"
        return f'drawer_{motion_dir}'

    @property
    def policy_type(self):
        return 5

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self.stage == 7


class ButtonPressPrimitive(Policy):

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, button_id=0, rel_frame: CoordinateFrame = world_frame_3D, mid_offset=np.array([0., 0., 0.1]),
                     button_offset=np.array([0, 0.04, 0.0]),
                     retreat_delta=np.array([0, 0.1, 0]), wait_press_steps=15,
                     tolerance=0.005, ori_tolerance=0.05, kp=(30., 30., 10.), ko=40.0, kg=10.0,
                     timeout=2000, tip_servo=True, target_gripper=250, **kwargs):

        self._object_id = button_id
        # tip position relative to the drawer grasp point.
        self._target_frame = rel_frame
        self._mid_offset = np.asarray(mid_offset)
        self._button_offset = np.asarray(button_offset)
        self._retreat_delta = np.asarray(retreat_delta)
        self._tolerance = tolerance
        self._ori_tolerance = ori_tolerance

        self._kp = np.array(kp)
        self._ko = np.array(ko)
        self._kg = np.array(kg)
        self._timeout = timeout

        self._wait_press_steps = wait_press_steps

        self._target_gripper = float(target_gripper)

        self._latest_desired_frame = world_frame_3D
        self._latest_desired_gripper = 0

        self._tip_in_ee_frame = self._env.tip_in_ee_frame

        self.num_steps = 0

        self.tip_servo = tip_servo

        # counters
        self.stage = 0
        self.num_steps = 0  # total
        self._curr_step = 0  # within stage

        self._target_button_pos = None

    def _init_params_to_attrs(self, params):
        self._max_pos_vel = get_with_default(params, "max_pos_vel", 0.75, map_fn=np.asarray)  # m/s per axis
        self._max_ori_vel = get_with_default(params, "max_ori_vel", 10.0, map_fn=np.asarray)  # rad/s per axis
        self._smooth_vel_coef = get_with_default(params, "smooth_vel_coef", 0.8, map_fn=float)  # smooths controller out
        self._vel_noise = get_with_default(params, "vel_noise", 0.01, map_fn=float)  # smooths controller out
        self._use_intermediate_targets = get_with_default(params, "use_intermediate_targets",
                                                          False)  # smooths controller out

    def _init_setup(self):
        if self._env is not None:
            assert isinstance(self._env, DrawerPlayroomEnv3D), type(self._env)

        self.stage = 0
        self.num_steps = 0
        self._curr_step = 0

    def reached(self, curr_pos, curr_ori, des_pos, des_ori):
        dpos = np.linalg.norm(des_pos - curr_pos)
        dori = np.linalg.norm(orientation_error(euler2mat(des_ori), euler2mat(curr_ori)))
        # print(self.stage, dpos, dori)

        return dpos < self._tolerance and dori < self._ori_tolerance

    def get_action(self, model, observation, goal, **kwargs):
        pos, gripper_pos, gripper_tip_pos, ori, drawer_posn, drawer_aabb, button_pos = observation.leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)) \
            .get_keys_required(
            ['ee_position', 'gripper_pos', 'gripper_tip_pos', 'ee_orientation_eul', 'drawer/joint_position_normalized', 'drawer/aabb', 'buttons/position'])

        desired_tip_frame = None
        desired_gripper = None

        ee_frame = CoordinateFrame.from_pose(np.concatenate([pos, ori]), world_frame_3D)
        tip_frame = CoordinateFrame(ee_frame, self._tip_in_ee_frame.rot.inv(), self._tip_in_ee_frame.pos)
        curr_pos = tip_frame.pos if self.tip_servo else pos
        curr_ori = tip_frame.rot.as_euler("xyz") if self.tip_servo else ori
        button_pos = button_pos[self._object_id].reshape(3)

        desired_gripper = self._target_gripper

        if self._target_button_pos is None:
            self._target_button_pos = button_pos.copy()

        ## Orient and tilt ##
        if self.stage == 0:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(), self._target_button_pos + self._button_offset + self._target_frame.pos + self._mid_offset)

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")):
                self._curr_step = 0
                self.stage += 1

        ## ABOVE THE BLOCK ##
        elif self.stage == 1:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_button_pos + self._button_offset + self._target_frame.pos)

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")):
                self._curr_step = 0
                self.stage += 1

        ## PUSH DOWN ##
        elif self.stage == 2:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_button_pos + self._button_offset)


            self._curr_step += 1
            if (button_pos - self._target_button_pos)[2] < -0.01:
                self._curr_step = 0
                self.stage += 1

        ## Wait ##
        elif self.stage == 3:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_button_pos + self._button_offset)
            self._curr_step += 1
            if self._curr_step > self._wait_press_steps:
                self._curr_step = 0
                self.stage += 1

        ## BACK UP ##
        elif self.stage == 4:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_button_pos + self._button_offset + self._target_frame.pos)

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")):
                self._curr_step = 0
                self.stage += 1

        ## RETREAT UP AND OUT ##
        elif self.stage == 5:
            desired_tip_frame = CoordinateFrame(world_frame_3D, self._target_frame.rot.inv(),
                                                self._target_button_pos + self._button_offset + self._target_frame.pos + self._retreat_delta)

            self._curr_step += 1
            if self.reached(curr_pos, curr_ori, desired_tip_frame.pos, desired_tip_frame.rot.as_euler("xyz")):
                self._curr_step = 0
                self.stage += 1

        if self.tip_servo:
            desired_tip_frame = desired_tip_frame.apply_a_to_b(self._tip_in_ee_frame, world_frame_3D)

        # p.addUserDebugLine(list(pos), list(desired_tip_frame.pos), lifeTime=self._env.dt)
        # servo to the correct position
        delta = desired_tip_frame.pos - pos
        curr = R.from_euler("xyz", ori)
        ori_delta = orientation_error(desired_tip_frame.rot.as_matrix(), curr.as_matrix())
        grab_pos_delta = desired_gripper - gripper_pos

        dx_pos = Reach3DPrimitive.clip_norm(self._kp * delta, self._max_pos_vel) * self._env.dt
        dx_ori = Reach3DPrimitive.clip_norm(self._ko * ori_delta, self._max_ori_vel) * self._env.dt
        dx_grip = Reach3DPrimitive.clip_norm(self._kg * grab_pos_delta, 150.) * self._env.dt

        self.num_steps += 1
        setpoint_pose = desired_tip_frame.as_pose(world_frame_3D)
        setpoint_grab = np.array([desired_gripper])

        # the desired waypoint
        out = d(action=(np.concatenate(
            [pos + dx_pos, (R.from_euler("xyz", dx_ori) * curr).as_euler("xyz"), gripper_pos + dx_grip]))[None])
        out.policy_type = np.array([self.policy_type])[None]
        out.policy_name = np.array([self.curr_name])[None]
        out['target/ee_position'] = setpoint_pose[None, :3]
        out['target/ee_orientation_eul'] = setpoint_pose[None, 3:]
        out['target/gripper_pos'] = setpoint_grab[None]
        return out

    @property
    def curr_name(self) -> str:
        return f'button_press_{self._object_id}'

    @property
    def policy_type(self):
        return 7

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self.stage == 6


def drawer_move_params_fn(obs, goal, env=None, offset=np.array([0, -0.01, -0.004]), move_delta_bounds=(0.1, 0.8), dir_probs=(0.75, 0.25),
                          open_gripper=160, target_gripper=210, x_noise_bounds=(-0.015, 0.015), do_partial=True, **kwargs):
    # rel_frame: CoordinateFrame = world_frame_3D, move_delta=0.1, retreat_delta=np.array([0, 0.1, 0]),
    #                 tolerance=0.005, ori_tolerance=0.05, kp=(30., 30., 5.), ko=30.0, kg=20.0,
    #                 timeout=2000, stage_timouts=np.array([80, 50, 80, 50]), tip_servo=True,
    #                 target_gripper=20,

    # tip_pos = (obs >> "gripper_tip_pos")[0, 0]
    drawer_pos = (obs >> "drawer/joint_position")[0, 0]
    drawer_posn = (obs >> "drawer/joint_position_normalized")[0, 0]
    ori = (obs >> "ee_orientation_eul")[0]  # ee and gripper are same rot

    x_noise = np.random.uniform(*x_noise_bounds)
    offset = offset + np.array([x_noise, 0, 0])

    # rotate gripper to 90 deg then rotate end effector to be tilted up.
    offset_rel_frame = CoordinateFrame(world_frame_3D, (R.from_euler("xyz", [-np.pi, 0., 0])).inv(), offset)

    # picks the direction with the most room. clip

    if do_partial:
        move_delta_norm = np.random.uniform(*move_delta_bounds)
        direction = np.random.choice([1, -1], p=dir_probs) if drawer_posn < 0.5 else \
            np.random.choice([-1, 1], p=dir_probs)
        new_drawer_pos = np.clip(drawer_posn + (direction * move_delta_norm), 0.05, 0.95)
    else:
        new_drawer_pos = 0.9 if drawer_posn < 0.5 else 0.08  # no partial motion

    # rescale
    move_delta = env.drawer_pos_normalize(new_drawer_pos, inverse=True) - drawer_pos

    # print(drawer_posn, new_drawer_pos)  # where are we tryna go

    # mid point of the reaching and retreating trajectories
    mid_offset = np.random.uniform(np.array([-0.01, 0, 0.08]), np.array([0.01, 0, 0.18]))

    # where to retreat to (rel to final point + mid_offset)
    retreat_delta = np.random.uniform(np.array([-0.15, -0.1, 0.05]), np.array([0.15, .1, 0.1]))

    return d(
        rel_frame=offset_rel_frame,
        move_delta=-move_delta,  # y axis is flipped for joint
        mid_offset=mid_offset,
        retreat_delta=retreat_delta,
        target_gripper=target_gripper,
        open_gripper=open_gripper,
        #               servo up,   down,       grasp,      move,       ungrasp,    retreat_up  retreat_bck
        stage_timeouts=[4 / env.dt, 4 / env.dt, 1.2 / env.dt, np.random.uniform(3.5, 4) / env.dt, 1 / env.dt, 3 / env.dt, 4 / env.dt],  # in steps
        timeout=int(30 / env.dt),
    )


def button_press_params_fn(obs, goal, env=None, button_id=None, offset=np.array([0, 0.0, 0.04]), target_gripper=210, **kwargs):

    if button_id is None:
        button_id = np.random.randint(0, 3)  # 3 buttons

    pitch = -np.pi/3.5
    yaw = [0, np.pi/5, -np.pi/7][button_id]  # middle, left, right
    offset = offset.copy()
    offset[0] += - int(button_id == 1) * 0.02  # if 1, on the left, so bring it in
    rel_frame = CoordinateFrame(world_frame_3D, (R.from_euler("z", yaw) * R.from_euler("x", pitch) * R.from_euler("xyz", [-np.pi, 0, -np.pi/2])).inv(), offset)

    # mid point of reaching motion
    mid_offset = np.random.uniform(np.array([-0.01, 0.08, 0.03]), np.array([0.01, 0.13, 0.05]))

    # where to retreat to (rel to final point)
    retreat_delta = np.random.uniform(np.array([-0.08, 0.1, 0.1]), np.array([-0.05, .2, 0.11]))

    #2-4 second wait
    wps = np.random.randint(20, 40)

    return d(
        button_id=button_id,
        rel_frame=rel_frame,
        wait_press_steps=wps,
        mid_offset=mid_offset,
        retreat_delta=retreat_delta,
        target_gripper=target_gripper,
        timeout=200,
    )


# def top_rot_directional_policy_params_fn(obs, goal, env=None, rotation_steps=10, rotation_velocity=0.5, retreat_velocity=0.2,
#                                          retreat_steps=10, retreat_first=False, retreat_xy=False,
#                                          uniform_velocity=False, axis=None, smooth_noise=0):
#     pos = (obs >> "objects/position")[0, 0]
#     orn_eul = (obs >> "objects/orientation_eul")[0, 0]
#     # aabb = (obs >> "objects/aabb")[0, 0].reshape(2, 3)
#     size = (obs >> "objects/size")[0, 0]
#
#     idxs, new_coordinate_frame, best_axes, gripper_yaws, thetas = get_gripper_yaws(pos, orn_eul, size, env,
#                                                                                    get_thetas=True)
#
#     if axis is None:
#         which = np.random.randint(0, 2)
#     else:
#         which = axis  # predefined x or y ONLY
#     rot_yaw = -gripper_yaws[which]  # pick one of the gripping angles
#     # print(rot_yaw)
#     # rot_yaw = 0
#     # axis_norm = np.linalg.norm(best_axes[:, which])
#
#     grip_width = 1.  # large overestimate
#     final_grip_width = 255.  # close until contact is noticed.
#
#     # print((obs >> "objects/orientation_eul")[0, 0, 2], gripper_yaws, size, best_axes[:, 0], np.linalg.norm(best_axes[:, 0]))
#
#     # current block height, width minus 2cm, for the z grip target
#     grip_target_z = max(np.linalg.norm(new_coordinate_frame[:, idxs[2]]) - 0.03, 0.02)
#     offset = np.array([0, 0, grip_target_z])
#
#     # thetas = rot_yaw + np.array([0, np.pi / 2, np.pi, -np.pi / 2])
#     #
#     # p = np.ones(2)
#     #
#     # # random pull direction along relative xy axes
#     # random_theta = np.random.choice(thetas, p=p) + np.random.uniform(-pull_theta_noise, pull_theta_noise)
#     # pull_dir = np.array([np.cos(random_theta), np.sin(random_theta)])
#
#     delta_obj = (obs >> "gripper_tip_pos")[0] - pos
#     if retreat_xy and np.linalg.norm(delta_obj) > 0.01:
#         retreat_dir, _ = get_side_retreat_xy_dir(delta_obj)
#         retreat_dir[2] = 1.
#     else:
#         retreat_dir = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 1.])
#
#     if retreat_xy:
#         retreat_dir[2] = np.random.uniform(0.1, 0.2)
#
#     # after rotation, we need to limit
#     new_grip_yaw = gripper_yaws[which] + rotation_steps * env.dt * rotation_velocity
#     # logger.debug(f"grip_yaw before: {np.rad2deg(new_grip_yaw)}. starting from {np.rad2deg(gripper_yaws[which])}")
#     if thetas[1] >= thetas[0]:
#         new_grip_yaw = np.clip(new_grip_yaw, thetas[0], thetas[1])
#     else:
#         new_grip_yaw = np.clip(new_grip_yaw, thetas[1], thetas[0])
#     # logger.debug(f"grip_yaw after: {np.rad2deg(new_grip_yaw)}")
#
#     # rotation amount is limited by the max and min gripper rotations at the given rotation.
#     rotation_steps = int(np.floor((new_grip_yaw - gripper_yaws[which]) / (env.dt * rotation_velocity)))
#
#     kp = (30., 30., 5.)
#     if uniform_velocity:
#         rotation_velocity = rotation_velocity * rotation_steps
#         rotation_steps = 40 * int(rotation_steps > 0)
#         retreat_velocity = retreat_velocity * retreat_steps
#         retreat_steps = 40 * int(retreat_steps > 0)
#         kp = (30., 30., 10.)
#
#     rel_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
#                                 offset)  # no offset
#     return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp, uniform_velocity=uniform_velocity,
#              target_gripper=grip_width, rotation_steps=rotation_steps, rotation_velocity=rotation_velocity,
#              retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first,
#              ori_on_block=False, retreat_velocity=retreat_velocity,
#              grasp_delta=final_grip_width - grip_width, grasp_steps=20, smooth_noise=smooth_noise)  # 2 seconds or grasped, whichever first


def get_gripper_yaw_for_mug(obj_pos, obj_eul, obj_size, env, idx=1, get_thetas=False):
    obj_rot = R.from_euler("xyz", obj_eul)
    # rotate size=
    new_axes = obj_rot.apply(np.eye(3))
    # atan(y / x) for the new y axis
    gripper_yaws = np.arctan2(new_axes[1, idx:idx+1], new_axes[0, idx:idx+1])
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
        return new_axes, gripper_yaws, (theta - np.pi / 2, np.pi / 2 - theta)
    return new_axes, gripper_yaws


def get_grasp_mug_target(mug_pos, mug_ori, mug_size, env, ret_all=False, diverse=False):

    axis, angle = p.getAxisAngleFromQuaternion(p.getQuaternionFromEuler(mug_ori))
    xy_mag = np.linalg.norm(axis[:2])
    assert xy_mag < 0.2 or angle < 0.1, "mug cannot be fallen over."

    # if axis[2] < 0:
    #     angle = -angle
    #     axis = -axis

    if ret_all:
        new_axes, gripper_yaws, thetas = get_gripper_yaw_for_mug(mug_pos, mug_ori, mug_size, env, get_thetas=True)
    else:
        new_axes, gripper_yaws = get_gripper_yaw_for_mug(mug_pos, mug_ori, mug_size, env)

    rot_yaw = -gripper_yaws[0]
    handle_dist_scale = (np.random.uniform(0.6, 0.65) if diverse else 0.6) * mug_size[1]
    offset = handle_dist_scale * np.linalg.inv(new_axes)[:, 1]  #best_axes[:, 0]  # y axis * y size / 2
    offset[2] = mug_size[2] / 2 + 0.02
    if diverse:
        offset[2] += np.random.uniform(0., 0.015)  # z-axis variability.

    target_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                    offset)

    if ret_all:
        return target_frame, rot_yaw, thetas
    return target_frame


def get_grasp_cabinet_handle_target(handle_pos, handle_ori, handle_size, env, safe=False, diverse=False, ret_all=False):

    # if axis[2] < 0:
    #     angle = -angle
    #     axis = -axis

    if ret_all:
        new_axes, gripper_yaws, thetas = get_gripper_yaw_for_mug(handle_pos, handle_ori, handle_size, env, idx=0, get_thetas=True)
    else:
        new_axes, gripper_yaws = get_gripper_yaw_for_mug(handle_pos, handle_ori, handle_size, env, idx=0)

    rot_yaw = -gripper_yaws[0]-np.pi
    offset = np.zeros(3)  # 0.6 * mug_size[1] * np.linalg.inv(new_axes)[:, 1]  #best_axes[:, 0]  # y axis * y size / 2
    offset[2] = handle_size[2] / 2
    if safe:
        # offset from front of handle
        vec = np.array([np.random.uniform(-0.01, 0.), 0., 0.]) if diverse else np.array([-0.01, 0., 0.])
        offset[:2] += R.from_euler("xyz", handle_ori).apply(vec)[:2]

    if diverse:
        # offset vertical for grasp
        offset[2] += np.random.uniform(-0.25 * handle_size[2], 0.1 * handle_size[2])

    target_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]),
                    offset)

    if ret_all:
        return target_frame, rot_yaw, thetas
    return target_frame


def cabinet_rot_directional_policy_params_fn(obs, goal, env=None, rotation_steps=10, rotation_velocity=0.5, retreat_velocity=0.2,
                                         retreat_steps=10, retreat_first=False, retreat_xy=False, do_partial=True,
                                         uniform_velocity=False, axis=None, smooth_noise=0, safe_open=False, diverse=False, direction=None):
    pos = (obs >> "cabinet/handle_position")[0]
    # aabb = (obs >> "objects/aabb")[0, idx].reshape(2, 3)
    size = (obs >> "cabinet/handle_size")[0]
    cab_size = (obs >> "cabinet/size")[0]
    cab_angle = (obs >> "cabinet/joint_position")[0, 0]
    orn_eul = np.array([0, 0, cab_angle])

    min_tolerance = np.pi/8

    # -1 means close, +1 means open
    if direction is None:
        weight_to_dir = 0.1 if do_partial else 0.25  # either 60 - 40 or 75 - 25
        direction = 2 * int(cab_angle > np.pi/4) - 1  # if closed (pi/2), we should open it (+1)
        new_min_tolerance = [abs(env.cabinet_max_open), abs(np.pi/2 - env.cabinet_max_closed)][int((direction + 1)/2)]  # tolerance is chosen based on open/closed
        min_tolerance = max(min_tolerance, new_min_tolerance)
        if cab_angle > min_tolerance and cab_angle < np.pi/2 - min_tolerance:  # some margin to make sure moves actually happen.
            # with probability 0.75, go with the direction that makes sense
            direction = [-1, 1][np.random.choice(2, p=[0.5 - weight_to_dir * direction, 0.5 + weight_to_dir * direction])]

    if direction == -1:
        cabinet_radius = R.from_euler("z", cab_angle).apply(np.array([0.02, cab_size[1] - 0.06, 0]))
    else:
        cabinet_radius = R.from_euler("z", cab_angle).apply(np.array([0.02, cab_size[1] - 0.03, 0]))

    rel_frame, rot_yaw, thetas = get_grasp_cabinet_handle_target(pos, orn_eul, size, env, safe=safe_open, diverse=diverse, ret_all=True)

    # rot_yaw = rot_yaw + np.pi

    grip_width = 1.  # overestimate of max grip width
    final_grip_width = 255.  # close until contact is noticed.

    targ_pos = rel_frame.pos
    grip_pos = (obs >> "gripper_tip_pos")[0] - pos
    mid_pt = 0.75 * targ_pos + 0.25 * grip_pos  # interpolate closer to targpos for middle pt
    mid_pt[2] += cab_size[2] * 0.65  # margin for clearing the top of the cabinet
    mid_pt_halfway = 0.5 * targ_pos + 0.5 * grip_pos
    mid_pt_halfway[2] += cab_size[2] * 0.65

    dir_xy = (mid_pt - mid_pt_halfway)[:2]
    if safe_open and cab_angle < np.pi/4:
        dir_xy /= np.linalg.norm(dir_xy)
        mid_pt[:2] += 0.04 * dir_xy  # go past by 4 cm

    if diverse:
        # make motion diverse for first waypoint
        ortho_dir_xy = np.array([-dir_xy[1], dir_xy[0]])
        mid_pt_halfway[:2] += np.random.uniform(-0.04, 0.04) * ortho_dir_xy

    mid_frame_halfway = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, -np.pi / 2]), mid_pt_halfway)  # relative to object COM
    mid_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, -np.pi / 2]), mid_pt)  # relative to object COM
    if safe_open:
        mid_frame = CoordinateFrame(world_frame_3D, R.from_euler("xyz", [-np.pi, 0, rot_yaw]), mid_pt)  # relative to object COM
    # rel_frame = CoordinateFrame(world_frame_3D, rel_frame.rot.inv(), rel_frame.pos + np.array([0, 0, cab_size[2] * 0.6]))  # relative to object COM

    mid_frames = []
    if cab_angle < np.pi/3:  # only go up if the door is kinda open
        mid_frames = [mid_frame_halfway, mid_frame]

    assert not retreat_xy
    if direction == -1:
        retreat_dir = np.array([np.random.uniform(-0.1, 0.3), np.random.uniform(0.4, 0.6), np.random.uniform(0., 0.2)])
    else:
        retreat_dir = np.array([np.random.uniform(0.1, 0.2), np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.4)])

    # after rotation, we need to limit
    target_cab_angle = -direction * np.pi/4 + np.pi/4  # if 1, open the door (yaw = 0)
    if do_partial:
        target_cab_angle = np.clip(cab_angle - direction * np.random.uniform(2 * min_tolerance, np.pi/2), 0, np.pi/2)
    target_cab_angle = min(target_cab_angle, env.cabinet_max_closed)
    target_cab_angle = max(target_cab_angle, env.cabinet_max_open)

    rotation_steps = int(abs(target_cab_angle - cab_angle) / abs(env.dt * rotation_velocity))
    rotation_velocity = rotation_velocity * np.sign(target_cab_angle - cab_angle)
    # new_grip_yaw = -rot_yaw + rotation_steps * env.dt * rotation_velocity * np.sign(target_cab_angle - cab_angle)
    # logger.debug(f"grip_yaw before: {np.rad2deg(new_grip_yaw)}. starting from {np.rad2deg(gripper_yaws[which])}")
    # if thetas[1] >= thetas[0]:
    #     new_grip_yaw = np.clip(new_grip_yaw, thetas[0], thetas[1])
    # else:
    #     new_grip_yaw = np.clip(new_grip_yaw, thetas[1], thetas[0])
    # logger.debug(f"grip_yaw after: {np.rad2deg(new_grip_yaw)}")

    # rotation amount is limited by the max and min gripper rotations at the given rotation.
    # rotation_steps = int(np.floor((new_grip_yaw - (-rot_yaw)) / (env.dt * rotation_velocity)))

    kp = (30., 30., 5.)
    if uniform_velocity:
        rotation_velocity = rotation_velocity * rotation_steps
        rotation_steps = 60 * int(rotation_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 20 * int(retreat_steps > 0)  # already pretty high up, limit the distance.
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, mid_frames=mid_frames, kp=kp, uniform_velocity=uniform_velocity,
             target_gripper=grip_width, rotation_steps=rotation_steps, rotation_velocity=rotation_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first,
             ori_on_block=False, retreat_velocity=retreat_velocity, radius=cabinet_radius[:2], sweep_arc=True,
             sweep_tolerance_scale=2.5 if safe_open else 1., stop_sweep_at_table=safe_open, block_idx=-2,  # -2 means servo with cabinet
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, smooth_noise=smooth_noise,  # 2 seconds or grasped, whichever first
             prefix="cabinet_", ptype=4)  # cabinet open


# moves along 4 cardinal directions in table plane
def mug_grasp_move_policy_params_fn(obs, goal, env=None, favor_center=True, pull_steps=20, retreat_steps=20, retreat_velocity=0.2,
                          retreat_first=False, retreat_xy=False, pull_velocity=0.1, uniform_velocity=False,
                          axis=None, mug_idx=0, diverse_grasp=False):
    pos = (obs >> "objects/position")[0, mug_idx]
    orn_eul = (obs >> "objects/orientation_eul")[0, mug_idx]
    size = (obs >> "objects/size")[0, mug_idx]

    rel_frame = get_grasp_mug_target(pos, orn_eul, size, env, diverse=diverse_grasp)

    grip_width = np.random.uniform(100, 200)  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.

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

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, pull_direction=pull_dir, pull_z=0.02, pull_steps=pull_steps, pull_velocity=pull_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20)  # 2 seconds or grasped, whichever first


def put_block_into_cabinet_or_drawer_policy_params_fn(obs, goal, env=None, pull_velocity=0.1, pull_steps=20, retreat_steps=20, retreat_velocity=0.2, drawer=False, retreat_first=False, use_cab_waypoints=False, xy_tolerance=False, uniform_velocity=False):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    size = (obs >> "objects/size")[0, 0]

    pitch = 0 if drawer else -np.pi/4

    rel_frame = CoordinateFrame(world_frame_3D, (R.from_euler("x", pitch) * R.from_euler("xyz", [-np.pi, 0, -np.pi/2])).inv(), np.array([0, 0.01, 0.015]))

    grip_width = 0.  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.
    pull_z = 0.05

    pull_pts = []
    if drawer:
        prefix = "to_drawer_"
        d_center = np.average(env.get_aabb(env.drawer_id), axis=0)
        pull_dir = (d_center + np.random.uniform([-0.04, 0.05, -0.01], [0.04, 0.09, 0.01]) - pos)[:2]
    else:
        prefix = "to_cabinet_"
        pull_dir = (env.cab_center + np.random.uniform([-0.04, 0.02, 0.0], [-0.02, 0.025, 0.001]) - pos)[:2]
        if use_cab_waypoints:
            # noisy midpt
            mid_pt = np.append(pull_dir * np.array([1, 0]) + np.random.uniform([-0.04, 0.], [0.04, 0.03]), pull_z)
            pull_pts = [mid_pt]  # only move in x direction first.
    pull_steps = int(np.linalg.norm(pull_dir) / (pull_velocity * env.dt))
    pull_dir = pull_dir / np.linalg.norm(pull_dir)

    retreat_dir = np.array([0, 1., 1.])
    if drawer:
        retreat_dir[1] = 0.1

    kp = (30., 30., 5.)
    if uniform_velocity:
        pull_velocity = pull_velocity * pull_steps
        pull_steps = 40 * int(pull_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 30 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, pull_direction=pull_dir, pull_z=pull_z, pull_steps=pull_steps,
             pull_velocity=pull_velocity, pull_points=pull_pts,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity, xy_tolerance=xy_tolerance,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, prefix=prefix)  # 2 seconds or grasped, whichever first


def move_cabinet_block_to_freespace_policy_params_fn(obs, goal, env=None, pull_velocity=0.1, pull_steps=20, retreat_steps=20, retreat_velocity=0.2, retreat_first=False, use_cab_waypoints=False, uniform_velocity=False, xy_tolerance=False,):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    size = (obs >> "objects/size")[0, 0]

    # move from drawer or cabinet
    pitch = -np.pi/4
    pull_z = 0.02

    rel_frame = CoordinateFrame(world_frame_3D, (R.from_euler("x", pitch) * R.from_euler("xyz", [-np.pi, 0, -np.pi/2])).inv(), np.array([0, 0.01, 0.015]))

    grip_width = 0.  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.

    pull_dir = (env.free_surface_center - pos)[:2]

    pull_pts = []
    if use_cab_waypoints:
        # noisy midpt
        mid_pt = np.append(pull_dir * np.array([0, 1]) + np.random.uniform([-0.04, -0.03], [0.04, 0.0]), pull_z)
        pull_pts = [mid_pt]  # only move in y direction first.

    pull_steps = int(np.linalg.norm(pull_dir) / (pull_velocity * env.dt))
    pull_dir = pull_dir / np.linalg.norm(pull_dir)
    retreat_dir = np.array([0, 1., 1.])

    kp = (30., 30., 5.)
    if uniform_velocity:
        pull_velocity = pull_velocity * pull_steps
        pull_steps = 40 * int(pull_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, pull_direction=pull_dir, pull_z=pull_z, pull_steps=pull_steps,
             pull_velocity=pull_velocity, pull_points=pull_pts,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity, xy_tolerance=xy_tolerance,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, prefix="from_cabinet_")  # 2 seconds or grasped, whichever first


def move_drawer_block_to_freespace_policy_params_fn(obs, goal, env=None, lift_velocity=0.1, lift_steps=20, retreat_steps=20, retreat_velocity=0.2, retreat_first=False, uniform_velocity=False, smooth_noise=0):
    pos = (obs >> "objects/position")[0, 0]
    orn_eul = (obs >> "objects/orientation_eul")[0, 0]
    size = (obs >> "objects/size")[0, 0]

    pitch = 0

    rel_frame = CoordinateFrame(world_frame_3D, (R.from_euler("x", pitch) * R.from_euler("xyz", [-np.pi, 0, -np.pi/2])).inv(), np.array([0, 0.01, 0.015]))

    grip_width = 0.  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.

    down_point = env.generate_free_space_point()
    down_point[2] += 0.01  # gripper margin
    # lift above, somewhat noisily
    lift_point = pos + np.random.uniform([-0.01, 0.01, 0.15], [0.01, 0.03, 0.2])

    down_velocity = lift_velocity
    lift_dir = lift_point - pos
    down_dir = down_point - lift_point
    lift_steps = int(np.ceil(np.linalg.norm(lift_dir) / (lift_velocity * env.dt)))
    down_steps = int(np.ceil(np.linalg.norm(down_dir) / (down_velocity * env.dt)))

    retreat_dir = np.random.uniform([-0.3, 0.5, 0.7], [0.3, 1.0, 1])

    kp = (30., 30., 5.)
    if uniform_velocity:
        lift_velocity = lift_velocity * lift_steps
        lift_steps = 40 * int(lift_steps > 0)
        down_velocity = down_velocity * down_steps
        down_steps = 40 * int(down_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 20 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, lift_direction=lift_dir, lift_steps=lift_steps, lift_velocity=lift_velocity,
             down_direction=down_dir, down_steps=down_steps, down_velocity=down_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, smooth_noise=smooth_noise, prefix="from_drawer_")  # TODO  # 2 seconds or grasped, whichever first


def mug_rot_directional_policy_params_fn(obs, goal, env=None, rotation_steps=10, rotation_velocity=0.5, retreat_velocity=0.2,
                                         retreat_steps=10, retreat_first=False, retreat_xy=False, stop_at_wall=True,
                                         uniform_velocity=False, axis=None, smooth_noise=0, idx=0, diverse_grasp=False):
    pos = (obs >> "objects/position")[0, idx]
    orn_eul = (obs >> "objects/orientation_eul")[0, idx]
    # aabb = (obs >> "objects/aabb")[0, idx].reshape(2, 3)
    size = (obs >> "objects/size")[0, idx]

    rel_frame, rot_yaw, thetas = get_grasp_mug_target(pos, orn_eul, size, env, ret_all=True, diverse=diverse_grasp)

    grip_width = 1.  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.

    delta_obj = (obs >> "gripper_tip_pos")[0] - pos
    assert not retreat_xy
    retreat_dir = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 1.])

    # after rotation, we need to limit
    new_grip_yaw = -rot_yaw + rotation_steps * env.dt * rotation_velocity
    # logger.debug(f"grip_yaw before: {np.rad2deg(new_grip_yaw)}. starting from {np.rad2deg(gripper_yaws[which])}")
    if thetas[1] >= thetas[0]:
        new_grip_yaw = np.clip(new_grip_yaw, thetas[0], thetas[1])
    else:
        new_grip_yaw = np.clip(new_grip_yaw, thetas[1], thetas[0])
    # logger.debug(f"grip_yaw after: {np.rad2deg(new_grip_yaw)}")

    # rotation amount is limited by the max and min gripper rotations at the given rotation.
    rotation_steps = int(np.floor((new_grip_yaw - (-rot_yaw)) / (env.dt * rotation_velocity)))

    kp = (30., 30., 5.)
    if uniform_velocity:
        rotation_velocity = rotation_velocity * rotation_steps
        rotation_steps = 40 * int(rotation_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp, uniform_velocity=uniform_velocity,
             target_gripper=grip_width, rotation_steps=rotation_steps, rotation_velocity=rotation_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_first=retreat_first,
             ori_on_block=False, retreat_velocity=retreat_velocity, radius=rel_frame.pos[:2], stop_at_wall=stop_at_wall,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, smooth_noise=smooth_noise)  # 2 seconds or grasped, whichever first


def mug_lift_platform_policy_params_fn(obs, goal, env=None, favor_center=True, retreat_steps=12, retreat_velocity=0.2,
                                   retreat_first=False, retreat_xy=False, lift_velocity=0.1, uniform_velocity=False,
                                   sample_directions=False, axis=None, smooth_noise=0, idx=0, diverse_grasp=False):
    assert isinstance(env, PlatformBlockEnv3D)
    assert not retreat_first, "not implemented"
    assert not retreat_xy, "not implemented"
    pos = (obs >> "objects/position")[0, idx]
    orn_eul = (obs >> "objects/orientation_eul")[0, idx]
    size = (obs >> "objects/size")[0, idx]
    aabb = (obs >> "objects/aabb")[0, idx]

    rel_frame, rot_yaw, thetas = get_grasp_mug_target(pos, orn_eul, size, env, ret_all=True, diverse=diverse_grasp)

    grip_width = 0. #max(255 * (1 - axis_norm / 0.08), 0) - 60  # overestimate of max grip width
    final_grip_width = 250.  # close until contact is noticed.

    # lift up and to closest wall.

    # table_z = env.surface_center[2]
    if sample_directions:
        closest_pt, close_idx, distances, points = env.get_nearest_platform((obs >> "objects").leaf_apply(lambda arr: arr[:, idx]), return_all=True, margin=0.2 * size[2])
        beta = 1.
        table_lens = np.repeat(env.surface_bounds, 2)
        soft = np.exp(-beta * distances / table_lens)
        p = soft / soft.sum()
        # update
        close_idx = np.random.choice(4, p=p)
        closest_pt = points[close_idx]
    else:
        closest_pt, _, _, _ = env.get_nearest_platform((obs >> "objects").leaf_apply(lambda arr: arr[:, idx]), margin=0.2 * size[2])

    # p.addUserDebugLine(lineFromXYZ=list(pos), lineToXYZ=list(closest_pt), lineColorRGB=[255,0,0], lineWidth=3,)
    # p.addUserDebugLine(lineFromXYZ=list(pos)[:2] + [env.platform_z], lineToXYZ=list(closest_pt)[:2] + [env.platform_z], lineColorRGB=[255,0,0], lineWidth=3,)

    obj_height = (aabb[5] - aabb[2])
    if pos[2] > env.platform_z:
        # lift to the table center if we are already on a platform.
        closest_pt = env.surface_center.copy()
        # closest_pt[2] += obj_height / 2
        extent = np.random.uniform(-0.3, 0.3, 2)
        closest_pt[:2] += extent * env.surface_bounds / 2  # randomize xy a good amount

    # closest_pt[2] += np.random.uniform(0.005, 0.01)  # margin for object above table

    closest_pt[2] += np.random.uniform(0.01, 0.02)  # margin for object above table

    # transform to gripper frame (new), since mug is grasped at an offset
    # closest_pt[:2] += 1 * rel_frame.pos[:2]

    extra_z = np.random.uniform(0.08, 0.13)

    # lifts to mid point, then comes down, extra amount is random (0.03 -> 0.08)
    lift_point = (closest_pt + pos) / 2
    if np.linalg.norm(closest_pt[:2] - pos[:2]) < 0.05:
        lift_point[:2] = pos[:2]
    lift_point[2] = env.platform_z + obj_height / 2 + extra_z  # random elevation


    # if uniform_velocity:
    #     lift_point[2] += 0.1  # since we need to clear the platform
    #     # closest_pt[2] += 0.025

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
        retreat_dir[2] = 3.  # retreat up more

    kp = (30., 30., 5.)
    if uniform_velocity:
        lift_velocity = lift_velocity * lift_steps
        lift_steps = 40 * int(lift_steps > 0)
        down_velocity = down_velocity * down_steps
        down_steps = 40 * int(down_steps > 0)
        retreat_velocity = retreat_velocity * retreat_steps
        retreat_steps = 40 * int(retreat_steps > 0)
        kp = (30., 30., 10.)

    return d(next_obs=obs, next_goal=goal, rel_frame=rel_frame, kp=kp,
             target_gripper=grip_width, lift_direction=lift_dir, lift_steps=lift_steps, lift_velocity=lift_velocity,
             down_direction=down_dir, down_steps=down_steps, down_velocity=down_velocity,
             retreat_direction=retreat_dir, retreat_steps=retreat_steps, retreat_velocity=retreat_velocity,
             retreat_first=retreat_first, ori_on_block=False, uniform_velocity=uniform_velocity,
             grasp_delta=final_grip_width - grip_width, grasp_steps=20, smooth_noise=smooth_noise)  # TODO  # 2 seconds or grasped, whichever first


def test_drawer():
    from sbrl.envs.param_spec import ParamEnvSpec

    # use_meta = True
    # rotate = True
    # no_push_pull = True

    env_spec_params = get_playroom3d_example_spec_params()
    env_params = get_playroom3d_example_params()
    env_params.render = True
    # env_params.block_size = (30, 30)

    # env_params.debug_cam_dist = 0.35
    # env_params.debug_cam_p = -45
    # env_params.debug_cam_y = 0
    # env_params.debug_cam_target_pos = [0.4, 0, 0.45]

    max_pos_vel = 0.4
    # extra_dc = d()
    # if uniform_vel:
    #     extra_dc.kp = (30., 30., 10.)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    block_env = DrawerPlayroomEnv3D(env_params, env_spec)

    # env presets
    presets = d()
    # presets = d(objects=d(position=np.array([0.4, 0.1, 0.35])[None], orientation_eul=np.array([0., 0., 0.])[None], size=np.array([0.032, 0.043, 0.03])[None]))

    model = Model(d(ignore_inputs=True), env_spec, None)

    # policy = PushPrimitive(d(vel_noise=0), env_spec, env=block_env)
    # policy = PullPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy = DrawerMovePrimitive(d(vel_noise=0,  max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy = TopRotatePrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    policy = PullPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy_params_fn = push_policy_params_fn
    # policy_params_fn = pull_policy_params_fn
    # policy_params_fn = lambda *args, **kwargs: drawer_move_params_fn(*args, uniform_velocity=True, **kwargs)
    # policy_params_fn = lambda *args, **kwargs: cabinet_rot_directional_policy_params_fn(*args, uniform_velocity=True, **kwargs)
    policy_params_fn = lambda *args, **kwargs: put_block_into_cabinet_or_drawer_policy_params_fn(*args, uniform_velocity=True, drawer=True, **kwargs)

    # target frame
    # default is facing the block
    obs, goal = block_env.user_input_reset(1, presets=presets)  # trolling with a fake UI
    policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
    logger.debug(f"Start: {obs >> 'drawer/joint_position_normalized'}")
    logger.debug(f"Start: {obs >> 'cabinet/joint_position_normalized'}")

    iters = 10
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    i = 0
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))
        obs, goal, done = block_env.step(act)
        # logger.debug(f"-> contact: {obs >> 'drawer/contact'}")

        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug(f"End: {obs >> 'drawer/joint_position_normalized'}")
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
            logger.debug(f"Start: {obs >> 'drawer/joint_position_normalized'}")
            logger.debug(f"Start: {obs >> 'cabinet/joint_position_normalized'}")
            i += 1


def test_mug():
    from sbrl.envs.param_spec import ParamEnvSpec

    # use_meta = True
    # rotate = True
    # no_push_pull = True

    platform = True

    env_spec_params = get_block3d_example_spec_params()
    env_params = get_block3d_example_params()
    env_params.render = True
    # env_params.block_size = (30, 30)
    env_params.object_spec = ['mug']

    # env_params.debug_cam_dist = 0.35
    # env_params.debug_cam_p = -45
    # env_params.debug_cam_y = 0
    # env_params.debug_cam_target_pos = [0.4, 0, 0.45]

    max_pos_vel = 0.4  # uniform vel
    # extra_dc = d()
    # if uniform_vel:
    #     extra_dc.kp = (30., 30., 10.)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    if platform:
        block_env = PlatformBlockEnv3D(env_params, env_spec)
    else:
        block_env = BlockEnv3D(env_params, env_spec)

    # env presets
    presets = d()
    # presets = d(objects=d(position=np.array([0.4, 0.1, 0.35])[None], orientation_eul=np.array([0., 0., 0.])[None], size=np.array([0.032, 0.043, 0.03])[None]))

    model = Model(d(ignore_inputs=True), env_spec, None)

    # policy = PushPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy = PullPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    policy = TopRotatePrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy = LiftPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)

    # policy = DrawerMovePrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel), env_spec, env=block_env)
    # policy_params_fn = push_policy_params_fn
    # policy_params_fn = pull_policy_params_fn
    # policy_params_fn = lambda *args, **kwargs: mug_grasp_move_policy_params_fn(*args, uniform_velocity=True, **kwargs)
    policy_params_fn = lambda *args, **kwargs: mug_rot_directional_policy_params_fn(*args, uniform_velocity=True, **kwargs)
    # policy_params_fn = lambda *args, **kwargs: mug_lift_platform_policy_params_fn(*args, uniform_velocity=True, **kwargs)

    # target frame
    # default is facing the block
    obs, goal = block_env.user_input_reset(1, presets=presets)  # trolling with a fake UI
    policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
    logger.debug(f"Start: {obs >> 'objects/position'}")

    iters = 10
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    i = 0
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))
        obs, goal, done = block_env.step(act)
        # logger.debug(f"-> contact: {obs >> 'drawer/contact'}")

        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug(f"End: {obs >> 'objects/position'}")
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
            logger.debug(f"Start: {obs >> 'objects/position'}")
            i += 1


def test_buttons():
    from sbrl.envs.param_spec import ParamEnvSpec

    # use_meta = True
    # rotate = True
    # no_push_pull = True

    env_spec_params = get_playroom3d_example_spec_params()
    env_params = get_playroom3d_example_params()
    env_params.render = True
    env_params.random_init_snap_cabinet = False
    env_params.use_buttons = True
    # env_params.block_size = (30, 30)

    # env_params.debug_cam_dist = 0.35
    # env_params.debug_cam_p = -45
    # env_params.debug_cam_y = 0
    # env_params.debug_cam_target_pos = [0.4, 0, 0.45]

    max_pos_vel = 0.4
    max_ori_vel = 5
    # extra_dc = d()
    # if uniform_vel:
    #     extra_dc.kp = (30., 30., 10.)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    block_env = DrawerPlayroomEnv3D(env_params, env_spec)

    # env presets
    presets = d()
    # presets = d(objects=d(position=np.array([0.4, 0.1, 0.35])[None], orientation_eul=np.array([0., 0., 0.])[None], size=np.array([0.032, 0.043, 0.03])[None]))

    model = Model(d(ignore_inputs=True), env_spec, None)

    policy = ButtonPressPrimitive(d(vel_noise=0, max_pos_vel=max_pos_vel, max_ori_vel=max_ori_vel), env_spec, env=block_env)
    # policy_params_fn = push_policy_params_fn
    # policy_params_fn = pull_policy_params_fn
    # policy_params_fn = lambda *args, **kwargs: drawer_move_params_fn(*args, uniform_velocity=True, **kwargs)
    # policy_params_fn = lambda *args, **kwargs: cabinet_rot_directional_policy_params_fn(*args, uniform_velocity=True, **kwargs)
    policy_params_fn = lambda *args, **kwargs: button_press_params_fn(*args, uniform_velocity=True, **kwargs)

    # target frame
    # default is facing the block
    obs, goal = block_env.user_input_reset(1, presets=presets)  # trolling with a fake UI
    policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
    logger.debug("Policy terminated, resetting")

    iters = 10
    i = 0
    pressed = []
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))
        obs, goal, done = block_env.step(act)
        pressed.append((obs >> "buttons/closed")[0, policy._object_id])

        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug(f"End: {obs >> 'buttons/position'}, Was it pressed? {np.any(pressed)}")
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(**policy_params_fn(obs, goal, env=block_env).as_dict())
            logger.debug(f"New policy: button {policy._object_id}")
            i += 1


if __name__ == '__main__':
    ## ONE OF THESE
    # test_drawer()
    # test_mug()
    test_buttons()
