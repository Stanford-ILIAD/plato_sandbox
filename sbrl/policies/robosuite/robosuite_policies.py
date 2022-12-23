import numpy as np
from scipy.spatial.transform import Rotation

from sbrl.envs.robosuite.robosuite_env import RobosuiteEnv, get_ordered_objects_from_arr
from sbrl.policies.policy import Policy
from sbrl.utils import transform_utils as T
from sbrl.utils.np_utils import clip_norm, clip_scale
from sbrl.utils.python_utils import get_with_default, AttrDict
from sbrl.utils.torch_utils import to_numpy
from sbrl.utils.trajectory_utils import Waypoint


class WaypointPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._max_pos_vel = get_with_default(params, "max_pos_vel", 0.4, map_fn=np.asarray)  # m/s per axis
        self._max_ori_vel = get_with_default(params, "max_ori_vel", 5.0, map_fn=np.asarray)  # rad/s per axis

    def _init_setup(self):
        assert self._env is not None
        assert isinstance(self._env, RobosuiteEnv), type(self._env)

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, pose_waypoints=None, ptype=0, tolerance=0.005, ori_tolerance=0.05, **kwargs):
        assert len(pose_waypoints) > 0, pose_waypoints
        for wp in pose_waypoints:
            assert isinstance(wp, Waypoint), type(wp)

        self._pose_waypoints = pose_waypoints
        self._curr_idx = 0
        self._curr_step = 0
        self._ptype = ptype

        self._tolerance = tolerance
        self._ori_tolerance = ori_tolerance
        self._done = False

    def gripper_q_to_scaled(self, q):
        robot = self._env.rs_env.robots[0]
        gripper = robot.gripper
        actuator_idxs = [robot.sim.model.actuator_name2id(actuator) for actuator in gripper.actuators]
        # rescale normalized gripper action to control ranges
        ctrl_range = robot.sim.model.actuator_ctrlrange[actuator_idxs]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        gripper_scaled = (q - bias) / weight  # -1 -> 1
        # these should be the same, so avg...
        return gripper_scaled.mean()

    def get_action(self, model, observation, goal, **kwargs):
        # todo multirobot support
        keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_eul', 'robot0_gripper_qpos', 'object']
        # index out batch and horizon
        pos, quat, ori, grq, obj = (observation > keys).leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)).get_keys_required(keys)
        gr = self.gripper_q_to_scaled(grq)

        obj_d, num_objects = get_ordered_objects_from_arr(self._env, obj)
        object_poses = np.concatenate([obj_d >> "objects/position", obj_d >> "objects/orientation_eul"], axis=-1)
        assert len(object_poses) == num_objects

        parent = None if self._curr_idx == 0 else self._pose_waypoints[self._curr_idx - 1]

        wp = self._pose_waypoints[self._curr_idx]
        if wp.cf is not None:
            reached = self.reached(pos, ori, gr, wp) if wp.check_reach else False
            if reached or self._curr_step > wp.timeout:
                if self._curr_idx < len(self._pose_waypoints) - 1:
                    self._curr_idx += 1
                    self._curr_step = 0
                else:
                    self._done = True

        wp = self._pose_waypoints[self._curr_idx]
        wp_pose, wp_grip = wp.update(parent, [pos], object_poses, gr)

        # compute the action
        dpos = wp_pose[:3] - pos
        target_q = T.euler2quat_ext(wp_pose[3:])
        curr_q = T.euler2quat_ext(ori)

        mpv, mov = wp.max_pos_ori_vel
        if mpv is None:
            mpv = self._max_pos_vel
        if mov is None:
            mov = self._max_ori_vel

        dpos = clip_norm(dpos, mpv * 0.05)

        # ori clip
        q_angle = T.quat_angle(target_q, curr_q)
        abs_q_angle_clipped = min(abs(q_angle), mov * 0.05)
        goal_q = T.quat_slerp(curr_q, target_q, abs(abs_q_angle_clipped / q_angle))
        # goal_eul = T.quat2euler_ext(goal_q)
        dori = T.quat2euler_ext(T.quat_difference(goal_q, curr_q))
        # dori = np.zeros(3)  #orientation_error(T.euler2mat(wp_pose[3:]), T.euler2mat(ori))

        goal_gr = wp_grip

        # clipping based on the control range
        _, omax = self._env.get_control_range()
        dpos = clip_scale(dpos, np.abs(omax[:3]))
        dori = clip_scale(dori, np.abs(omax[3:6]))

        self._curr_step += 1

        return AttrDict(
            target=AttrDict(
                position=wp_pose[:3],
                orientation_eul=wp_pose[3:],
                gripper=np.array([wp_grip]),
            ),
            action=self._env.unscale_action(np.concatenate([dpos, dori, [goal_gr]])),
            policy_name=np.array([self.curr_name]),
            policy_type=np.array([self.policy_type]),
        ).leaf_apply(lambda arr: arr[None])

    @property
    def curr_name(self) -> str:
        # returns a string identifier for the policy, rather than ints.
        return self._env.name

    @property
    def policy_type(self) -> int:
        # returns a string identifier for the policy, rather than ints.
        return self._ptype

    def reached(self, pos, ori, gr, wp):
        # has reached uses the TRUE target desired frame.
        # print(T.quat_angle(wp.cf.rot.as_quat(), T.euler2quat_ext(ori)))
        return np.linalg.norm(
            wp.cf.pos - pos) < self._tolerance and \
               abs(T.quat_angle(wp.cf.rot.as_quat(), T.euler2quat_ext(ori))) < self._ori_tolerance

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done


def get_min_yaw(yaw):
    allowed_yaws = np.array([abs(yaw), abs(abs(yaw) - np.pi), abs(yaw) + np.pi])
    sgn = np.array([1, np.sign(abs(yaw) - np.pi), 1])
    min_idx = np.argmin(allowed_yaws)
    return allowed_yaws[min_idx] * sgn[min_idx] * np.sign(yaw)


def get_nut_assembly_square_policy_params(obs, goal, env=None, random_motion=True):
    keys = ['robot0_eef_pos', 'robot0_eef_eul', 'robot0_gripper_qpos', 'object']
    # index out batch and horizon
    pos, _, grq, obj = (obs > keys).leaf_apply(lambda arr: to_numpy(arr[0], check=True)).get_keys_required(keys)
    od, no = get_ordered_objects_from_arr(env, obj)

    base_ori = np.array([-np.pi, 0, 0])
    base_offset = np.array([0.06, 0., 0.])
    obj_q = (od >> "objects/orientation")[0]
    offset = Rotation.from_quat(obj_q).apply(base_offset)

    obj_yaw = T.quat2euler_ext(obj_q)[2]
    yaw = (obj_yaw + np.pi) % (2 * np.pi) - np.pi
    # minimum yaw
    yaw = get_min_yaw(yaw)

    desired_obj_yaws = np.array([np.pi / 2, 3 * np.pi / 2])  # np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    desired_yaws = np.array([get_min_yaw(y) for y in desired_obj_yaws])
    delta = (desired_yaws - yaw) % (2 * np.pi)  # put delta in 0->360
    delta = np.minimum(delta, 2 * np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    which_idx = np.argmin(delta)
    desired_yaw = desired_yaws[which_idx]  # the one that requires the least rotation

    # delta = (desired_obj_yaws - obj_yaw) % (2 * np.pi)  # put delta in 0->360
    # delta = np.minimum(delta, 2*np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    # desired_obj_yaw = desired_obj_yaws[np.argmin(delta)]  # the one that requires the least rotation
    # desired_yaw = (desired_obj_yaw + np.pi) % (2 * np.pi) - np.pi
    # desired_yaw = get_min_yaw(desired_yaw)

    # print(np.rad2deg(obj_yaw), np.rad2deg(yaw))
    ori = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -yaw)).as_euler("xyz")
    ori_goal = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -desired_yaw)).as_euler("xyz")

    # open
    above = Waypoint(np.concatenate([offset + np.array([0., 0., 0.05]), ori]), -1, timeout=20 * 6,
                     relative_to_parent=False,
                     relative_to_object=0,
                     relative_ori=False)

    down = Waypoint(np.concatenate([offset, ori]), -1, timeout=20 * 1.5,
                    relative_to_parent=False,
                    relative_to_object=0,
                    relative_ori=False)

    grasp = Waypoint(np.concatenate([offset, ori]), 1, timeout=20 * 1,
                     relative_to_parent=False,
                     relative_to_object=0,
                     relative_ori=False, check_reach=False)

    up_pos = np.array([0., 0., 0.15])
    if random_motion:
        up_pos[:2] += np.random.uniform(-0.04, 0.04, 2)
    up_rot = Waypoint(np.concatenate([up_pos, ori_goal]), 1, timeout=20 * 3,
                      relative_to_parent=True,
                      relative_to_object=0,
                      relative_ori=False, )

    peg_pos = np.array(env.rs_env.sim.data.body_xpos[env.rs_env.peg1_body_id])
    obj_offset = Rotation.from_euler('z', desired_yaw - yaw + obj_yaw).apply(base_offset)

    above_peg_z = np.random.uniform(0.14, 0.17) if random_motion else 0.15
    down_peg_z = np.random.uniform(0.06, 0.1) if random_motion else 0.08

    above_peg = Waypoint(np.concatenate([peg_pos + obj_offset + np.array([0., 0., above_peg_z]), ori_goal]), 1,
                         timeout=20 * 8,
                         relative_to_parent=False, )

    on_peg = Waypoint(np.concatenate([np.array([0., 0., down_peg_z - above_peg_z]), ori_goal]), 1, timeout=20 * 2,
                      relative_to_parent=True, relative_to_robot=0, relative_ori=False)

    end_open = Waypoint(np.concatenate([np.array([0., 0., 0.]), ori_goal]), -1, timeout=20 * 1,
                        relative_to_parent=True, relative_to_robot=0, relative_ori=False, check_reach=False)

    return AttrDict(pose_waypoints=[above, down, grasp, up_rot, above_peg, on_peg, end_open])


def get_tool_hang_policy_params(obs, goal, env=None, random_motion=True):
    keys = ['robot0_eef_pos', 'robot0_eef_eul', 'robot0_gripper_qpos', 'object']
    # index out batch and horizon
    pos, _, grq, obj = (obs > keys).leaf_apply(lambda arr: to_numpy(arr[0], check=True)).get_keys_required(keys)

    # object order is [base, frame, tool]
    od, no = get_ordered_objects_from_arr(env, obj)
    which_obj = 1

    base_ori = np.array([-np.pi, 0, 0])
    base_offset = np.array([0.0, -0.05, 0.])
    obj_q = (od >> "objects/orientation")[which_obj]
    obj_yaw = T.quat2euler_ext(obj_q)[2]

    base2_offset = np.array([0.0, 0., 0.])
    obj2_q = (od >> "objects/orientation")[2]
    obj2_yaw = T.quat2euler_ext(obj2_q)[2]

    offset = Rotation.from_euler('z', obj_yaw).apply(base_offset)
    yaw = (obj_yaw + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    # minimum yaw
    yaw = get_min_yaw(yaw)

    offset2 = Rotation.from_euler('z', obj2_yaw).apply(base2_offset)
    yaw2 = (obj2_yaw + np.pi) % (2 * np.pi) - np.pi
    # minimum yaw
    yaw2 = get_min_yaw(yaw2)

    # desired_obj_yaws = np.array([np.pi / 2, 3 * np.pi / 2])  # np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    # desired_yaws = np.array([get_min_yaw(y) for y in desired_obj_yaws])
    # delta = (desired_yaws - yaw) % (2 * np.pi)  # put delta in 0->360
    # delta = np.minimum(delta, 2 * np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    # which_idx = np.argmin(delta)
    # desired_yaw = desired_yaws[which_idx]  # the one that requires the least rotation

    # delta = (desired_obj_yaws - obj_yaw) % (2 * np.pi)  # put delta in 0->360
    # delta = np.minimum(delta, 2*np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    # desired_obj_yaw = desired_obj_yaws[np.argmin(delta)]  # the one that requires the least rotation
    # desired_yaw = (desired_obj_yaw + np.pi) % (2 * np.pi) - np.pi
    # desired_yaw = get_min_yaw(desired_yaw)

    # print(np.rad2deg(obj_yaw), np.rad2deg(yaw))
    ori = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -yaw)).as_euler("xyz")
    # rotate y in end effector frame
    ori_goal = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("y", -np.pi / 2)).as_euler("xyz")

    ori2 = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -yaw2)).as_euler("xyz")
    ori2_goal = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -np.pi / 2)).as_euler("xyz")

    # open
    above_delta = np.array([0., 0., 0.1])
    if random_motion:
        above_delta[:2] += np.random.uniform(-0.01, 0.01, 2)
        above_delta[2] += np.random.uniform(-0.02, 0.04)
    above = Waypoint(np.concatenate([offset + above_delta, ori]), -1, timeout=20 * 6,
                     relative_to_parent=False,
                     relative_to_object=which_obj,
                     relative_ori=False)

    down = Waypoint(np.concatenate([offset, ori]), -1, timeout=20 * 1.5,
                    relative_to_parent=False,
                    relative_to_object=which_obj,
                    relative_ori=False)

    grasp = Waypoint(np.concatenate([offset, ori]), 1, timeout=20 * 1,
                     relative_to_parent=False,
                     relative_to_object=which_obj,
                     relative_ori=False, check_reach=False)
    #
    up_pos = np.array([0., 0., 0.2])
    if random_motion:
        up_pos[:2] += np.random.uniform(-0.03, 0.03, 2)
    up = Waypoint(np.concatenate([up_pos, ori]), 1, timeout=20 * 3,
                  relative_to_parent=True,
                  relative_to_object=which_obj,
                  relative_ori=False, )

    zup = 0.4
    dzdw = -0.2
    if random_motion:
        zup += np.random.uniform(-0.01, 0.02)
        dzdw += np.random.uniform(-0.03, 0.01)
    up_rot_dpos = np.array(([0.0, 0.045, zup]))
    up_rot = Waypoint(np.concatenate([up_rot_dpos, ori_goal]), 1, timeout=20 * 7,
                      relative_to_parent=False,
                      relative_to_object=0,
                      relative_ori=False)

    peg_insert_delta = np.array([0., 0., dzdw])
    peg_in = Waypoint(np.concatenate([peg_insert_delta, ori_goal]), 1, timeout=20 * 2,
                      relative_to_parent=True,
                      relative_to_object=0,
                      relative_ori=False, max_pos_vel=0.4)

    ungrasp = Waypoint(np.concatenate([np.zeros(3), ori_goal]), -1, timeout=20 * 0.5,
                       relative_to_parent=True,
                       relative_to_object=0,
                       relative_ori=False, check_reach=False)

    out_a = Waypoint(np.concatenate([np.array([0.05, 0., 0.2]), base_ori]), -1, timeout=20 * 2,
                     relative_to_parent=True,
                     relative_to_object=0,
                     relative_ori=False)

    out_b = Waypoint(np.concatenate([np.array([0.13, 0., -0.15]), base_ori]), -1, timeout=20 * 1.5,
                     relative_to_parent=True,
                     relative_to_object=0,
                     relative_ori=False)

    dz_above = np.random.uniform(0.05, 0.15) if random_motion else 0.1
    out2 = Waypoint(np.concatenate([offset2 + np.array([0., 0., dz_above]), ori2]), -1, timeout=20 * 7,
                    relative_to_parent=False,
                    relative_to_object=2,
                    relative_ori=False)

    down2 = Waypoint(np.concatenate([offset2, ori2]), -1, timeout=20 * 1.,
                     relative_to_parent=False,
                     relative_to_object=2,
                     relative_ori=False)

    grasp2 = Waypoint(np.concatenate([offset2, ori2]), 1, timeout=20 * 1,
                      relative_to_parent=False,
                      relative_to_object=2,
                      relative_ori=False, check_reach=False)

    zup = 0.05
    if random_motion:
        zup += np.random.uniform(0, 0.03)
    up_rot_2 = Waypoint(np.concatenate([np.array([0., -0.25, zup]), ori2_goal]), 1, timeout=20 * 3,
                        relative_to_parent=False,
                        relative_to_object=which_obj,
                        relative_ori=False, )

    above_deposit = Waypoint(np.concatenate([np.array([0., -0.181, zup]), ori2_goal]), 1, timeout=20 * 2,
                             relative_to_parent=False,
                             relative_to_object=which_obj,
                             relative_ori=False, )

    down_deposit = Waypoint(np.concatenate([np.array([0., 0., -zup]), ori2_goal]), 1, timeout=20 * 2,
                            relative_to_parent=True,
                            relative_to_object=which_obj,
                            relative_ori=False, max_pos_vel=0.4)

    down_deposit2 = Waypoint(np.concatenate([np.array([0., 0., 0]), ori2_goal]), -1, timeout=20 * 0.5,
                             relative_to_parent=True,
                             relative_to_object=which_obj,
                             relative_ori=False, check_reach=False)

    retreat_up = Waypoint(np.concatenate([np.array([0., 0., 0.05]), ori2_goal]), -1, timeout=20 * 1,
                          relative_to_parent=True,
                          relative_to_object=which_obj,
                          relative_ori=False)

    start_close = Waypoint(np.concatenate([np.array([0., 0., 0.0]), ori2_goal]), 1, timeout=20 * 1,
                          relative_to_parent=True,
                          relative_to_object=which_obj,
                          relative_ori=False, check_reach=False)

    down_deposit_push = Waypoint(np.concatenate([np.array([0., 0., -zup - 0.1]), ori2_goal]), 1, timeout=20 * 2,
                                 relative_to_parent=True,
                                 relative_to_object=which_obj,
                                 relative_ori=False, )

    wait = Waypoint(np.concatenate([np.array([0., 0., 0]), ori2_goal]), 1, timeout=20 * 1,
                    relative_to_parent=True,
                    relative_to_object=which_obj,
                    relative_ori=False, check_reach=False)

    return AttrDict(
        pose_waypoints=[above, down, grasp, up, up_rot, peg_in, ungrasp, out_a, out_b, out2, down2, grasp2, up_rot_2,
                        above_deposit, down_deposit, down_deposit2, retreat_up, start_close, down_deposit_push, wait])
