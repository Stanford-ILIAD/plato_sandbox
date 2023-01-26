import numpy as np
import torch

from configs.exp_lfp.utils import Block2DLearningUtils
from sbrl.experiments import logger
from sbrl.utils import config_utils, transform_utils as T
from sbrl.utils.np_utils import clip_norm, clip_scale
from sbrl.utils.python_utils import AttrDict as d, is_array
from sbrl.utils.torch_utils import to_numpy, to_torch
from sbrl.utils.transform_utils import get_normalized_quat


class Robot3DLearningUtils(Block2DLearningUtils):
    # replace this if needed, this is the default.
    state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_eul']
    wp_keys = ['target/position', 'target/orientation', 'target/orientation_eul']
    ac_keys = ['action']
    ac_shapes = [(7,)]
    default_delta_wp_high = np.array([0.02] * 3 + [0.0] * 3)
    default_delta_wp_low = -default_delta_wp_high

    def __init__(self, fast_dynamics=False, clip_l1=False, no_ori=False, use_target_gripper=False):
        self.no_ori = no_ori
        self.use_target_gripper = use_target_gripper
        if self.no_ori:
            self.state_keys = ['robot0_eef_pos']
            self.wp_keys = ['target/position']
            self.ac_shapes = [(4,)]
            self.default_delta_wp_high = self.default_delta_wp_high[:3]
            self.default_delta_wp_low = self.default_delta_wp_low[:3]

        if use_target_gripper and 'target/gripper' not in self.wp_keys:
            self.wp_keys.append('target/gripper')
            self.default_delta_wp_low = np.append(self.default_delta_wp_low, -1)
            self.default_delta_wp_high = np.append(self.default_delta_wp_high, 1)

        self.fast_dynamics = fast_dynamics
        self.clip_l1 = clip_l1
        super(Robot3DLearningUtils, self).__init__()

    def get_wp_dynamics_fn(self, max_pos_vel=0.4, max_ori_vel=5.0, DT=0.05, TOL=0.0075, ORI_TOL=0.075,
                           true_max_pos_vel=1., true_max_ori_vel=10., fast_dynamics=None, scale_action=True,
                           clip_l1=None):

        # defaults
        fast_dynamics = self.fast_dynamics if fast_dynamics is None else fast_dynamics
        clip_l1 = self.clip_l1 if clip_l1 is None else clip_l1

        """ (robot pose, mode0_action) -> (next robot pose, mode1_action, reached) """
        max_ac_dpos = true_max_pos_vel * DT
        max_ac_dori = true_max_ori_vel * DT

        if fast_dynamics:
            max_pos_vel = true_max_pos_vel
            max_ori_vel = true_max_ori_vel

        def wp_dyn(robot_pose, targ_pose):
            torch_device = None if isinstance(robot_pose, np.ndarray) else robot_pose.device
            if torch_device is not None:
                robot_pose = to_numpy(robot_pose).copy()
                targ_pose = to_numpy(targ_pose).copy()

            front_shape = list(robot_pose.shape[:-1])
            robot_pose = robot_pose.reshape(-1, robot_pose.shape[-1])
            targ_pose = targ_pose.reshape(-1, targ_pose.shape[-1])

            # targ_pose can be either
            # - 6 dim (pos, ori_eul)
            # - 10 dim (pos, ori, ori_eul)
            pos = robot_pose[..., :3]
            dpos = (targ_pose[..., :3] - pos)
            goal_gr = np.zeros_like(dpos[..., :1])

            # TODO torch implementation of these

            if isinstance(dpos, np.ndarray):

                dpos = clip_norm(dpos, max_pos_vel * DT, axis=-1)  # see robosuite_policies.py

                target_q = get_normalized_quat(targ_pose)
                curr_q = get_normalized_quat(robot_pose)

                q_angle = T.quat_angle(target_q, curr_q).astype(targ_pose.dtype)
                abs_q_angle_clipped = np.minimum(np.abs(q_angle), max_ori_vel * DT)
                scale = abs_q_angle_clipped.copy()
                scale[np.abs(q_angle) > 0] /= np.abs(q_angle[np.abs(q_angle) > 0])
                goal_q = T.batch_quat_slerp(curr_q, target_q, scale)
                # goal_eul = T.quat2euler_ext(goal_q)

                dori_q = T.quat_difference(goal_q, curr_q)
                dori = T.fast_quat2euler_ext(dori_q)

                dpos = clip_scale(dpos, max_ac_dpos)
                dori = clip_scale(dori, max_ac_dori)
                dori_q = T.fast_euler2quat_ext(dori)

                next_pos = pos + dpos
                next_ori_q = T.quat_multiply(curr_q, dori_q)
                next_ori = T.fast_quat2euler_ext(next_ori_q)
                next_state = np.concatenate([next_pos, next_ori_q, next_ori], axis=-1)

                if scale_action:
                    # actions are scaled to (-1 -> 1)
                    unscaled_dpos = dpos / max_ac_dpos
                    unscaled_dori = dori / max_ac_dori
                else:
                    # actions are scaled to (-1 -> 1)
                    unscaled_dpos = dpos
                    unscaled_dori = dori

                # hopefully gripper will be replaced later.. hard to compute now.
                mode1_action = np.concatenate([unscaled_dpos, unscaled_dori, goal_gr], axis=-1)
                reached = (np.linalg.norm(dpos, axis=-1) < TOL) & (np.abs(q_angle) < ORI_TOL)
            else:
                raise NotImplementedError

            if torch_device is not None:
                mode1_action = to_torch(mode1_action, device=torch_device)
                next_state = to_torch(next_state, device=torch_device)
                reached = to_torch(reached, device=torch_device)

            # reshape back
            mode1_action = mode1_action.reshape(front_shape + [mode1_action.shape[-1]])
            next_state = next_state.reshape(front_shape + [next_state.shape[-1]])
            return mode1_action, next_state, reached

        def wp_dyn_noori(robot_pose, targ_pose):
            torch_device = None if isinstance(robot_pose, np.ndarray) else robot_pose.device
            if torch_device is not None:
                robot_pose = to_numpy(robot_pose).copy()
                targ_pose = to_numpy(targ_pose).copy()

            front_shape = list(robot_pose.shape[:-1])
            robot_pose = robot_pose.reshape(-1, robot_pose.shape[-1])
            targ_pose = targ_pose.reshape(-1, targ_pose.shape[-1])

            # targ_pose must be 3 dim (pos)
            pos = robot_pose[..., :3]
            dpos = (targ_pose[..., :3] - pos)
            goal_gr = np.zeros_like(dpos[..., :1])
            dpos = clip_norm(dpos, max_pos_vel * DT, axis=-1)  # see robosuite_policies.py
            assert robot_pose.shape[-1] == 3, robot_pose.shape
            next_state = pos + dpos

            if scale_action:
                # actions are scaled to (-1 -> 1)
                unscaled_dpos = dpos / max_ac_dpos
            else:
                # actions are not scaled to (-1 -> 1)
                unscaled_dpos = dpos

            # hopefully gripper will be replaced later.. hard to compute now.
            mode1_action = np.concatenate([unscaled_dpos, goal_gr], axis=-1)
            reached = (np.linalg.norm(dpos, axis=-1) < TOL)

            if torch_device is not None:
                mode1_action = to_torch(mode1_action, device=torch_device)
                next_state = to_torch(next_state, device=torch_device)
                reached = to_torch(reached, device=torch_device)

            # reshape back
            mode1_action = mode1_action.reshape(front_shape + [mode1_action.shape[-1]])
            next_state = next_state.reshape(front_shape + [next_state.shape[-1]])
            return mode1_action, next_state, reached

        return wp_dyn_noori if self.no_ori else wp_dyn

    def get_combine_waypoint_fn(self, delta_wp_low=None, delta_wp_high=None):
        delta_wp_low = self.default_delta_wp_low if delta_wp_low is None else delta_wp_low
        delta_wp_high = self.default_delta_wp_high if delta_wp_high is None else delta_wp_high

        def combine_waypoint_fn(wp, wp_delta):
            nonlocal delta_wp_low, delta_wp_high
            if isinstance(wp, torch.Tensor) and not isinstance(delta_wp_low, torch.Tensor):
                delta_wp_low, delta_wp_high = torch.tensor(delta_wp_low, device=wp.device), torch.tensor(delta_wp_high,
                                                                                                         device=wp.device)

            # [-1 -> 1] -> [delta_low, delta_high]
            wp = wp.clone() if isinstance(wp, torch.Tensor) else wp.copy()
            assert wp.shape[-1] in [6, 10], wp.shape
            true_wp_delta = delta_wp_low + (wp_delta[..., :6] + 1) * (delta_wp_high - delta_wp_low) / 2
            wp[..., :3] += true_wp_delta[..., :3]
            # # for now, simple orientation addition in euler space, copy over to quaternion
            wp[..., -3:] = (wp[..., -3:] + true_wp_delta[..., 3:6] + np.pi) % (
                        2 * np.pi) - np.pi  # bring the waypoint value in -pi -> pi.
            if wp.shape[-1] == 10:
                wp[..., 3:-3] = T.euler2quat_ext(wp[..., -3:])
            return wp

        return combine_waypoint_fn

    # interactive setting
    def get_waypoint_correction_fn(self):
        # user input?
        def wp_correction(trainer, env, env_memory):
            raise NotImplementedError

    def parse_orientations(self, unnorm_out, targ_prefix):
        if f'{targ_prefix}orientation' in unnorm_out.list_leaf_keys():
            targ_quat = to_numpy(unnorm_out >> f"{targ_prefix}orientation", check=True)[:, 0]
            targ_quat = targ_quat / (
                        np.linalg.norm(targ_quat, axis=-1, keepdims=True) + 1e-8)  # make sure quaternion is normalized
            targ_eul = T.fast_quat2euler_ext(targ_quat)
        else:
            targ_eul = to_numpy(unnorm_out >> f"{targ_prefix}orientation_eul", check=True)[:, 0]
            targ_quat = T.fast_euler2quat_ext(targ_eul)
        return targ_quat, targ_eul

    def default_online_action_postproc_fn(self, model, obs, out, policy_out_names,
                                          max_vel=None, max_orn_vel=None, max_gripper_vel=None, memory: d = d(),
                                          policy_out_norm_names=None, use_setpoint=False,
                                          clip_targ_orn=False, free_orientation=True, vel_act=True,
                                          pos_key="robot0_eef_pos", quat_key="robot0_eef_quat",
                                          eul_key="robot0_eef_eul", targ_prefix="target/", wp_dyn=None, **kwargs):
        if policy_out_norm_names is None:
            policy_out_norm_names = policy_out_names
        if wp_dyn is None:
            wp_dyn = self.wp_dynamics_fn

        unnorm_out = model.normalize_by_statistics(out, policy_out_norm_names, inverse=True) > policy_out_names
        # import ipdb; ipdb.set_trace()
        targ_eul, targ_quat = None, None

        if vel_act:
            ac = to_numpy(unnorm_out >> "action", check=True)
            assert ac.shape[-1] == (4 if self.no_ori else 7), ac.shape
        else:
            if unnorm_out.has_leaf_key('target/gripper'):
                base_gripper = to_numpy(unnorm_out >> 'target/gripper', check=True)
            else:
                base_action = to_numpy(unnorm_out >> "action", check=True)  # needs this for gripper
                base_gripper = base_action[..., -1:]
            # front_shape = base_action.shape[:-1]
            pos = to_numpy(obs >> pos_key, check=True)[:, 0]
            if quat_key in obs.leaf_keys():
                quat = to_numpy(obs >> quat_key, check=True)[:, 0]
                quat = quat / (
                            np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)  # make sure quaternion is normalized
                eul = T.fast_quat2euler_ext(quat)
            else:
                eul = to_numpy(obs >> pos_key, check=True)[:, 0]
                quat = T.fast_euler2quat_ext(eul)

            targ_pos = to_numpy(unnorm_out >> f"{targ_prefix}position", check=True)[:, 0]
            if self.no_ori:
                curr_pose = pos
                targ_pose = targ_pos
            else:
                targ_quat, targ_eul = self.parse_orientations(unnorm_out, targ_prefix)
                curr_pose = np.concatenate([pos, quat, eul], axis=-1)
                targ_pose = np.concatenate([targ_pos, targ_quat, targ_eul], axis=-1)

            # dynamics of waypoint -> action
            ac, _, _ = wp_dyn(curr_pose, targ_pose)

            ac = ac[:, None]  # re-expand to match horizon
            ac[..., -1:] = base_gripper  # fill in gripper action from base action.
            # print(np.linalg.norm(base_action[..., 3:6]), np.linalg.norm(ac[..., 3:6]))

        out.combine(unnorm_out.leaf_apply(lambda arr: arr))
        out.action = to_torch(ac, device="cpu").to(device=model.device, dtype=torch.float32)

        # add in orientation keys if missing
        if not self.no_ori and unnorm_out.has_leaf_key(f'{targ_prefix}orientation') or unnorm_out.has_leaf_keys(
                f'{targ_prefix}orientation_eul'):
            if targ_quat is None or targ_eul is None:
                # extract orientations
                targ_quat, targ_eul = self.parse_orientations(unnorm_out, targ_prefix)

            out[f'{targ_prefix}orientation'] = to_torch(targ_quat[:, None], device=model.device)
            out[f'{targ_prefix}orientation_eul'] = to_torch(targ_eul[:, None], device=model.device)

        out.leaf_modify(lambda arr: (arr[:, 0] if is_array(arr) else arr))
        return out

    def get_default_lmp_names_and_sizes(self, common_info, plan_name, plan_size, INCLUDE_GOAL_PROPRIO, SINGLE_GRAB,
                                        DO_GRAB_NORM,
                                        ENCODE_ACTIONS=True, INCLUDE_BLOCK_SIZES=False, ENCODE_OBJECTS=True,
                                        PRIOR_OBJECTS_ONLY=False, POLICY_GOALS=True, VEL_ACT=True, DO_ORN_NORM=True,
                                        USE_DRAWER=False, NO_OBJECTS=False, REAL_INPUTS=False, EXCLUDE_VEL=False,
                                        OBS_USE_QUAT=False, TARG_USE_QUAT=False):
        nsld = common_info >> "names_shapes_limits_dtypes"

        # not in nlsd, be careful, NOTE : active constraints will not force the grab to happen...
        # assert VEL_ACT
        assert not SINGLE_GRAB
        assert DO_ORN_NORM
        assert not INCLUDE_BLOCK_SIZES
        assert not USE_DRAWER
        assert not REAL_INPUTS, "not implemented"

        assert not NO_OBJECTS or not ENCODE_OBJECTS, "cannot encode objects if no objects is true."

        if VEL_ACT:
            policy_out_names = ['action']
        else:
            if self.no_ori:
                policy_out_names = ['target/position']
            else:
                policy_out_names = ['target/position',
                                    'target/orientation' if TARG_USE_QUAT else 'target/orientation_eul']
            if self.use_target_gripper:
                policy_out_names.append('target/gripper')

        policy_out_norm_names = policy_out_names

        policy_out_size = config_utils.nsld_get_dims_for_keys(nsld, policy_out_names)

        proprio_names = ["robot0_eef_pos", f"robot0_eef_{'quat' if OBS_USE_QUAT else 'eul'}", "robot0_eef_vel_ang",
                         "robot0_eef_vel_lin", "robot0_gripper_qpos"]
        if self.no_ori:
            # leave out orientation names
            proprio_names = proprio_names[:1] + proprio_names[-2:]

        if EXCLUDE_VEL:
            proprio_names = [n for n in proprio_names if "_vel" not in n]
        proprio_size = config_utils.nsld_get_dims_for_keys(nsld, proprio_names)

        obs_enc_names = ['object']
        obs_enc_size = config_utils.nsld_get_dims_for_keys(nsld, obs_enc_names)
        optional_obs_enc_names = [] if NO_OBJECTS else obs_enc_names
        optional_obs_enc_size = config_utils.nsld_get_dims_for_keys(nsld, optional_obs_enc_names)

        # states and actions
        TRAJECTORY_NAMES = proprio_names
        if ENCODE_OBJECTS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + optional_obs_enc_names
        if ENCODE_ACTIONS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + policy_out_names

        NORMALIZATION_NAMES = list(set(list(TRAJECTORY_NAMES) + optional_obs_enc_names))
        if not DO_GRAB_NORM:
            NORMALIZATION_NAMES.remove('robot0_gripper_qpos')
        SAVE_NORMALIZATION_NAMES = proprio_names + optional_obs_enc_names + policy_out_names
        TRAJECTORY_SIZE = config_utils.nsld_get_dims_for_keys(nsld, TRAJECTORY_NAMES)

        # just states
        POLICY_NAMES = proprio_names + optional_obs_enc_names + [plan_name]
        POLICY_IN_SIZE = (1 + int(INCLUDE_GOAL_PROPRIO)) * proprio_size + 2 * optional_obs_enc_size + plan_size

        # states without plan, duplicated (start and goal)
        PRIOR_NAMES = proprio_names
        PRIOR_IN_SIZE = (1 + int(INCLUDE_GOAL_PROPRIO)) * proprio_size
        if ENCODE_OBJECTS:
            if PRIOR_OBJECTS_ONLY:
                assert not INCLUDE_GOAL_PROPRIO, "cannot include goal proprio if prior takes only object states"
                # prior only encodes objects,
                PRIOR_NAMES = []
                PRIOR_IN_SIZE = 0
            PRIOR_NAMES = PRIOR_NAMES + optional_obs_enc_names
            PRIOR_IN_SIZE += 2 * optional_obs_enc_size

        # just external states
        if INCLUDE_GOAL_PROPRIO:
            PRIOR_GOAL_STATE_NAMES = optional_obs_enc_names + proprio_names if ENCODE_OBJECTS else proprio_names
            POLICY_GOAL_STATE_NAMES = optional_obs_enc_names + proprio_names  # if ENCODE_OBJECTS else proprio_names
        else:
            if not ENCODE_OBJECTS:
                logger.warn('Objects not being encoded, but will be used for goal_state.')
            PRIOR_GOAL_STATE_NAMES = optional_obs_enc_names
            POLICY_GOAL_STATE_NAMES = optional_obs_enc_names

        if not POLICY_GOALS:
            POLICY_GOAL_STATE_NAMES = []
            POLICY_IN_SIZE = proprio_size + optional_obs_enc_size + plan_size

        PRIOR_GOAL_IN_SIZE = config_utils.nsld_get_dims_for_keys(nsld, PRIOR_GOAL_STATE_NAMES)

        return d(
            TRAJECTORY_NAMES=TRAJECTORY_NAMES,  # posterior
            TRAJECTORY_SIZE=TRAJECTORY_SIZE,
            POLICY_NAMES=POLICY_NAMES,  # policy
            POLICY_IN_SIZE=POLICY_IN_SIZE,
            PRIOR_NAMES=PRIOR_NAMES,  # prior
            PRIOR_IN_SIZE=PRIOR_IN_SIZE,
            PRIOR_GOAL_IN_SIZE=PRIOR_GOAL_IN_SIZE,
            PRIOR_GOAL_STATE_NAMES=PRIOR_GOAL_STATE_NAMES,
            POLICY_GOAL_STATE_NAMES=POLICY_GOAL_STATE_NAMES,
            NORMALIZATION_NAMES=NORMALIZATION_NAMES,
            SAVE_NORMALIZATION_NAMES=SAVE_NORMALIZATION_NAMES,
            policy_out_names=policy_out_names,
            policy_out_norm_names=policy_out_norm_names,
            policy_out_size=policy_out_size,
            proprio_names=proprio_names,
            proprio_size=proprio_size,
            obs_enc_names=obs_enc_names,
            obs_enc_size=obs_enc_size,
        )
