import numpy as np
import torch

from configs.exp_lfp.utils import Block2DLearningUtils
from sbrl.utils import config_utils
from sbrl.utils.control_utils import orientation_error, batch_orientation_eul_add
from sbrl.utils.np_utils import clip_norm
from sbrl.utils.python_utils import AttrDict as d
from sbrl.utils.torch_utils import to_numpy, to_torch
from sbrl.utils.transform_utils import euler2mat, fast_quat2euler_ext
from sbrl.policies.blocks.block3d_success_metrics import map_pname_to_success as success_metric_fns


class Block3DLearningUtils(Block2DLearningUtils):
    ONLINE_KP = (30., 30., 5.)
    ONLINE_FAST_KP = (30., 30., 10.)
    state_keys = ['ee_position', 'ee_orientation_eul', 'gripper_pos']
    wp_keys = ['target/ee_position', 'target/ee_orientation_eul', 'target/gripper_pos']
    ac_keys = ['action']
    contact_names = ['objects/contact', 'drawer/contact', 'cabinet/contact', 'buttons/contact']
    ac_shapes = [(7,)]
    default_delta_wp_low = -np.array([0.02] * 3 + [0.0] * 3)
    default_delta_wp_high = np.array([0.02] * 3 + [0.0] * 3)
    ac_gripper_dims = 1

    def __init__(self):

        # default. TODO do this better, dataset needs it
        self.parse_interaction_from_episode_fn = self.get_parse_interaction_from_episode_fn(max_contact_len=80,
                                                                                            max_interaction_len=150)

        self.parse_interaction_bounds_from_episode_fn = self.get_parse_interaction_from_episode_fn(max_contact_len=100,
                                                                                                   max_interaction_len=0,
                                                                                                   bounds=True)

        # TODO
        self.wp_dynamics_fn = None

    def get_success_metric_fns(self):
        return success_metric_fns

    def default_online_action_postproc_fn(self, model, obs, out, policy_out_names,
                                             Kv_P=(30., 30., 5.), Kv_O=(30., 30., 30.), Kv_G=20.,
                                             max_vel=0.4, max_orn_vel=10.0, max_gripper_vel=150., relative=False,
                                             vel_act=False, memory: d = d(), policy_out_norm_names=None,
                                             use_setpoint=False,
                                             clip_targ_orn=False, free_orientation=False, **kwargs):
        if policy_out_norm_names is None:
            policy_out_norm_names = policy_out_names

        unnorm_out = model.normalize_by_statistics(out, policy_out_norm_names, inverse=True) > policy_out_names

        original_obs = obs.leaf_copy()
        obs = (obs > ['ee_position', 'ee_orientation_eul', "gripper_pos"]).leaf_apply(
            lambda arr: to_numpy(arr, check=True)) \
            .leaf_apply(lambda arr: arr[:, :1])

        if not use_setpoint or not memory.has_leaf_key("controller/ee_position"):
            # much more stable target (use_setpoint = True)
            memory.controller.ee_position = obs.ee_position.copy()
            memory.controller.ee_orientation_eul = obs.ee_orientation_eul.copy()
            memory.controller.gripper_pos = obs.gripper_pos.copy()

        DT = 0.1

        if vel_act:
            ac = to_numpy(unnorm_out >> "action", check=True)
            assert ac.shape[-1] == 7
        else:
            targ_pos = to_numpy(unnorm_out >> "target/ee_position", check=True)
            # accepts either target quat or euler
            if 'target/ee_orientation_eul' in unnorm_out.leaf_keys():
                targ_orn = to_numpy(unnorm_out >> "target/ee_orientation_eul", check=True)
            else:
                targ_orn_q = to_numpy(unnorm_out >> "target/ee_orientation", check=True)
                targ_orn = fast_quat2euler_ext(targ_orn_q)
            targ_grip = to_numpy(unnorm_out >> "target/gripper_pos", check=True)
            if relative:
                raise NotImplementedError
            Kv_P = np.asarray(Kv_P) if isinstance(targ_pos, np.ndarray) else torch.tensor(Kv_P, device=targ_pos.device)
            Kv_O = np.asarray(Kv_O) if isinstance(targ_orn, np.ndarray) else torch.tensor(Kv_O, device=targ_orn.device)
            Kv_G = np.asarray(Kv_G) if isinstance(targ_grip, np.ndarray) else torch.tensor(Kv_G,
                                                                                           device=targ_grip.device)
            # yaw only if not in drawer env
            if "drawer" not in original_obs.keys() and not free_orientation:
                targ_orn[..., 0] = -np.pi
                targ_orn[..., 1] = 0.
            # targ_orn[..., 2] = np.where(targ_orn[..., 2] < -np.pi/2, targ_orn[..., 2] + np.pi, targ_orn[..., 2])
            ac_vel = Kv_P * (targ_pos - memory.controller.ee_position)
            ac_orn_vel = Kv_O * orientation_error(euler2mat(targ_orn.reshape(3)),
                                                  euler2mat(memory.controller.ee_orientation_eul.reshape(3)))
            # B x H x N,   max over N blocks
            ac_grab = Kv_G * (targ_grip - memory.controller.gripper_pos)

            ac_dx = clip_norm(ac_vel, max_vel) * DT
            ac_orn_dx = clip_norm(ac_orn_vel, max_orn_vel) * DT
            ac_grab_dx = clip_norm(ac_grab, max_gripper_vel) * DT

            clip_next_gripper = np.clip(memory.controller.gripper_pos + ac_grab_dx, 0,
                                        250)  # max gripper is 250, not 255

            # print(obs.ee_orientation_eul, ac_orn_vel, batch_orientation_eul_add(obs.ee_orientation_eul, ac_orn_vel * DT))
            # print(ac_grab.shape, ac_vel.shape, norm_out.leaf_apply(lambda arr: arr.shape))
            ac = np.concatenate([memory.controller.ee_position + ac_dx,
                                 batch_orientation_eul_add(
                                     ac_orn_dx.reshape(memory.controller.ee_orientation_eul.shape),
                                     memory.controller.ee_orientation_eul),
                                 clip_next_gripper],
                                axis=-1)

        out.combine(unnorm_out.leaf_apply(lambda arr: arr[:, 0]))
        out.action = to_torch(ac[:, 0], device="cpu").to(device=model.device, dtype=torch.float32)

        # save the setpoint for next time
        memory.controller.ee_position = ac[..., :3].copy()
        memory.controller.ee_orientation_eul = ac[..., 3:6].copy()
        memory.controller.gripper_pos = ac[..., 6:].copy()
        return out

    def get_default_lmp_names_and_sizes(self, common_info, plan_name, plan_size, INCLUDE_GOAL_PROPRIO, SINGLE_GRAB,
                                           DO_GRAB_NORM,
                                           ENCODE_ACTIONS=True, INCLUDE_BLOCK_SIZES=False, ENCODE_OBJECTS=True,
                                           PRIOR_OBJECTS_ONLY=False, POLICY_GOALS=True, VEL_ACT=False, DO_ORN_NORM=True,
                                           USE_DRAWER=False, NO_OBJECTS=False, REAL_INPUTS=False, EXCLUDE_VEL=False,
                                           OBS_USE_QUAT=False, TARG_USE_QUAT=False):
        nsld = common_info >> "names_shapes_limits_dtypes"

        # if in the observation_names, add it
        USE_BUTTONS = "buttons/position" in (common_info >> "observation_names")

        # not in nlsd, be careful, NOTE : active constraints will not force the grab to happen...
        if VEL_ACT:
            assert not SINGLE_GRAB
            assert DO_GRAB_NORM
            policy_out_names = ['action']
        else:
            if SINGLE_GRAB:
                raise NotImplementedError

            policy_out_names = ['target/ee_position', 'target/ee_orientation_eul', 'target/gripper_pos']
            if TARG_USE_QUAT:
                policy_out_names[1] = 'target/ee_orientation'

        if not DO_ORN_NORM:
            policy_out_norm_names = ['target/ee_position', 'target/gripper_pos']
        else:
            policy_out_norm_names = None

        policy_out_size = config_utils.nsld_get_dims_for_keys(nsld, policy_out_names)

        proprio_names = ['ee_position', 'ee_velocity', 'ee_orientation_eul', 'ee_angular_velocity',
                         'gripper_pos', 'finger_left_contact',
                         'finger_right_contact']  # proprioceptive cues, active constraints = force on each block
        if REAL_INPUTS:
            proprio_names = ['gripper_tip_pos', 'ee_orientation_eul', 'gripper_pos']
        if EXCLUDE_VEL:
            proprio_names = [n for n in proprio_names if "velocit" not in n]
        proprio_size = config_utils.nsld_get_dims_for_keys(nsld, proprio_names)

        obs_enc_names = ['objects/position', 'objects/velocity', 'objects/orientation_eul',
                         'objects/angular_velocity']
        if REAL_INPUTS:
            obs_enc_names = ['objects/position', 'objects/orientation_eul']

        if INCLUDE_BLOCK_SIZES and not NO_OBJECTS:
            obs_enc_names.append('objects/size')
        if USE_DRAWER:
            if NO_OBJECTS:
                obs_enc_names = []  # do not include object state info. in LMP
            obs_enc_names.extend(['drawer/joint_position_normalized', 'drawer/joint_velocity'])
            obs_enc_names.extend(['cabinet/joint_position_normalized', 'cabinet/joint_velocity'])
            if USE_BUTTONS:
                obs_enc_names.extend(['buttons/closed'])

        if EXCLUDE_VEL:
            obs_enc_names = [n for n in obs_enc_names if "velocit" not in n]

        # vis_enc_out_names = ['visual_encoding']  # just a stacking IRL
        obs_enc_size = config_utils.nsld_get_dims_for_keys(nsld, obs_enc_names)
        # visual_feature_size = obs_enc_size  # concat'd obs, no model here

        # states and actions
        TRAJECTORY_NAMES = proprio_names
        if ENCODE_OBJECTS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + obs_enc_names
        if ENCODE_ACTIONS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + policy_out_names

        NORMALIZATION_NAMES = list(set(list(TRAJECTORY_NAMES) + obs_enc_names))
        SAVE_NORMALIZATION_NAMES = proprio_names + obs_enc_names + policy_out_names
        if not VEL_ACT and ENCODE_ACTIONS and not DO_GRAB_NORM:
            NORMALIZATION_NAMES.remove('target/gripper_pos')  # do not normalize grabs
            NORMALIZATION_NAMES.remove('gripper_pos')  # do not normalize grabs

        TRAJECTORY_SIZE = config_utils.nsld_get_dims_for_keys(nsld, TRAJECTORY_NAMES)

        # just states
        POLICY_NAMES = proprio_names + obs_enc_names + [plan_name]
        POLICY_IN_SIZE = (1 + int(INCLUDE_GOAL_PROPRIO)) * proprio_size + 2 * obs_enc_size + plan_size

        # states without plan, duplicated (start and goal)
        PRIOR_NAMES = proprio_names
        PRIOR_IN_SIZE = (1 + int(INCLUDE_GOAL_PROPRIO)) * proprio_size
        if ENCODE_OBJECTS:
            if PRIOR_OBJECTS_ONLY:
                assert not INCLUDE_GOAL_PROPRIO, "cannot include goal proprio if prior takes only object states"
                # prior only encodes objects,
                PRIOR_NAMES = []
                PRIOR_IN_SIZE = 0
            PRIOR_NAMES = PRIOR_NAMES + obs_enc_names
            PRIOR_IN_SIZE += 2 * obs_enc_size

        # just external states
        if INCLUDE_GOAL_PROPRIO:
            PRIOR_GOAL_STATE_NAMES = obs_enc_names + proprio_names if ENCODE_OBJECTS else proprio_names
            POLICY_GOAL_STATE_NAMES = obs_enc_names + proprio_names  # if ENCODE_OBJECTS else proprio_names
        else:
            assert ENCODE_OBJECTS, "including goal proprio and not encoding objects!!"
            PRIOR_GOAL_STATE_NAMES = obs_enc_names
            POLICY_GOAL_STATE_NAMES = obs_enc_names

        if not POLICY_GOALS:
            POLICY_GOAL_STATE_NAMES = []
            POLICY_IN_SIZE = proprio_size + obs_enc_size + plan_size

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

    def get_contact_start_ends(self, inputs, current_horizon=None, ep_lengths=None):
        # import ipdb; ipdb.set_trace()
        assert current_horizon is not None
        if "objects" in inputs.keys():
            bc = (inputs >> "objects/contact")
        else:
            bc = None

        if "drawer" in inputs.keys():
            bc = (bc | (inputs >> "drawer/contact")) if bc is not None else (inputs >> "drawer/contact")
        if "cabinet" in inputs.keys():
            bc = (bc | (inputs >> "cabinet/contact")) if bc is not None else (inputs >> "cabinet/contact")

        if isinstance(bc, np.ndarray):
            in_block_contact = np.any(bc, axis=-1)  # (B x H x NB) -> (B x H)
            assert bc.dtype == np.bool  # making sure these weren't normalized first.
            first_contact = np.argmax(in_block_contact, axis=1)
            og_last_contact = np.argmax(np.flip(in_block_contact, axis=1), axis=1)
            last_contact = (in_block_contact.shape[1] - 1) - og_last_contact
            # last_contact = in_contact, [1]).argmax(dim=1)
        else:
            in_block_contact = torch.any(bc, dim=-1)  # (B x H x NB) -> (B x H)
            assert bc.dtype == torch.bool  # making sure these weren't normalized first.
            _, first_contact = torch.max(in_block_contact, dim=1)
            _, og_last_contact = torch.max(torch.flip(in_block_contact, dims=[1]), dim=1)
            last_contact = (in_block_contact.shape[1] - 1) - og_last_contact
            # last_contact = in_contact, [1]).argmax(dim=1)

        if ep_lengths is not None:
            if (first_contact >= ep_lengths).any():
                import ipdb;
                ipdb.set_trace()
            if (last_contact >= ep_lengths).any():
                import ipdb;
                ipdb.set_trace()
            assert (first_contact < ep_lengths).all(), [in_block_contact, first_contact,
                                                        ep_lengths]  # checking for improper data
            assert (last_contact < ep_lengths).all(), [in_block_contact, last_contact,
                                                       ep_lengths]  # checking for improper data

        return first_contact, last_contact, in_block_contact

    def default_get_contact_fn(self, inputs):
        if "drawer" in inputs.keys():
            if "objects" in inputs.keys():
                if "buttons" in inputs.keys():
                    # DRAWER + OBJECTS + BUTTONS
                    res = np.any(inputs >> "objects/contact", axis=-1, keepdims=True) | (inputs >> "drawer/contact") | (
                            inputs >> "cabinet/contact") | np.any(inputs >> "buttons/contact", axis=-1, keepdims=True)
                else:
                    # DRAWER + OBJECTS
                    res = np.any(inputs >> "objects/contact", axis=-1, keepdims=True) | (inputs >> "drawer/contact") | (
                            inputs >> "cabinet/contact")
            else:
                # DRAWER
                res = (inputs >> "drawer/contact") | (inputs >> "cabinet/contact")
        else:
            # OBJECTS
            res = np.any(inputs >> "objects/contact", axis=-1, keepdims=True)

        return res.reshape(-1)

    @staticmethod
    def closest_block_pos(ins):
        raise NotImplementedError('3D environment does not presuppose blocks')
