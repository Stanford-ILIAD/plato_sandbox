import numpy as np
import torch

from sbrl.utils import config_utils
from sbrl.utils.config_utils import Utils
from sbrl.utils.dist_utils import get_dist_first_horizon
from sbrl.utils.loss_utils import mae_err_fn, get_default_mae_action_loss_fn
from sbrl.utils.python_utils import AttrDict as d
from sbrl.utils.torch_utils import combine_after_dim, concatenate, split_dim, unconcatenate, \
    unsqueeze_then_gather, combine_then_concatenate, view_unconcatenate, \
    unsqueeze_n
from sbrl.policies.blocks.stack_block2d_success_metrics import map_pname_to_success as success_metric_fns


class Block2DLearningUtils(Utils):
    """
    All the utilities for block 2D env. learning, including dynamics, etc.
    """

    # constants
    ONLINE_KP = (1.5, 5.)
    ONLINE_FAST_KP = (1.5, 5.)
    state_keys = ['position', 'grab_binary']
    wp_keys = ['target/position', 'target/grab_binary']
    ac_keys = ['action']
    contact_names = ['block_contact', 'grab_binary']
    contact_conds = [lambda x: x, lambda x: x > 0]
    ac_shapes = [(3,)]
    default_delta_wp_low = np.array([-30, -30])
    default_delta_wp_high = np.array([30, 30])
    ac_gripper_dims = 1

    # dynamics lfp utils:
    def __init__(self):

        # default. TODO do this better, dataset needs it
        self.parse_interaction_from_episode_fn = self.get_parse_interaction_from_episode_fn()
        self.parse_interaction_bounds_from_episode_fn = self.get_parse_interaction_from_episode_fn(bounds=True)

        # replace this if needed, this is the default.
        self.wp_dynamics_fn = self.get_wp_dynamics_fn()

    def get_success_metric_fns(self):
        return success_metric_fns

    # (robot pose, mode0_action) -> (next robot pose, mode1_action, reached)
    @staticmethod
    def get_wp_dynamics_fn(max_vel=100, DT=0.1, TOL=10):
        def wp_dyn(robot_state, targ_state):
            pos = robot_state[..., :2]
            vel = (targ_state[..., :2] - pos) / DT  # delta / dt
            vel = np.clip(vel, -max_vel, max_vel) if isinstance(vel, np.ndarray) else torch.clip(vel, -max_vel, max_vel)
            next_pos = pos + vel * DT
            if isinstance(vel, np.ndarray):
                next_state = np.concatenate([next_pos, targ_state[..., 2:]], axis=-1)
                grab_single = np.any(targ_state[..., 2:], axis=-1)[..., None]
                mode1_action = np.concatenate([vel, grab_single], axis=-1)
                reached = np.linalg.norm(next_pos - pos, axis=-1) < TOL
            else:
                next_state = torch.cat([next_pos, targ_state[..., 2:]], dim=-1)
                grab_single = torch.any(targ_state[..., 2:], dim=-1)[..., None]
                mode1_action = torch.cat([vel, grab_single], dim=-1)
                reached = torch.norm(next_pos - pos, dim=-1) < TOL

            return mode1_action, next_state, reached

        return wp_dyn

    # see mode_key_da_config, for example if states change, compute the new actions from the waypoint.
    def label_action_from_wp_fn(self, inputs, outputs, mask=None, memory=None, **kwargs):
        # must be (B x H x ...). the final shape might include some optional keys, hence checking before combining.
        wp = combine_then_concatenate(inputs, [s for s in self.wp_keys if s in inputs.leaf_keys()], dim=2)
        old_ac = combine_then_concatenate(inputs, self.ac_keys, dim=2)
        partial_state = combine_then_concatenate(inputs, [s for s in self.state_keys if s in inputs.leaf_keys()], dim=2)
        # inputs.leaf_apply(lambda arr: arr.dtype).pprint()
        new_ac, next_state, reached = self.wp_dynamics_fn(partial_state, wp)

        if mask is not None:
            mask = unsqueeze_n(mask, len(new_ac.shape) - len(mask.shape), dim=-1)
            mask = torch.broadcast_to(mask, new_ac.shape)
            new_ac = torch.where(mask, new_ac, old_ac)
            if self.ac_gripper_dims > 0:
                new_ac[..., -self.ac_gripper_dims:] = old_ac[..., -self.ac_gripper_dims:]

        new_ac_dc = view_unconcatenate(new_ac, self.ac_keys, dict(zip(self.ac_keys, self.ac_shapes)))
        inputs.combine(new_ac_dc)
        return inputs, outputs

    def get_combine_waypoint_fn(self, delta_wp_low=None, delta_wp_high=None):
        delta_wp_low = self.default_delta_wp_low if delta_wp_low is None else delta_wp_low
        delta_wp_high = self.default_delta_wp_high if delta_wp_high is None else delta_wp_high

        def combine_waypoint_fn(wp, wp_delta):
            nonlocal delta_wp_low, delta_wp_high
            if isinstance(wp, torch.Tensor) and not isinstance(delta_wp_low, torch.Tensor):
                delta_wp_low, delta_wp_high = torch.tensor(delta_wp_low, device=wp.device), torch.tensor(delta_wp_high,
                                                                                                         device=wp.device)

            # [-1 -> 1] -> [delta_low, delta_high]
            true_wp_delta = delta_wp_low + (wp_delta[..., :2] + 1) * (delta_wp_high - delta_wp_low) / 2
            wp[..., :2] += true_wp_delta
            return wp

        return combine_waypoint_fn

    def default_online_action_postproc_fn(self, model, obs, out, policy_out_names, GRAB_THRESH=0.25, Kv_P=(1.5, 5.),
                                          skip_grab=False, max_vel=150, relative=False, vel_act=False,
                                          policy_out_norm_names=None, get_first_action=True, **kwargs):
        if policy_out_norm_names is None:
            policy_out_norm_names = policy_out_names
        unnorm_out = model.normalize_by_statistics(out, policy_out_norm_names, inverse=True) > policy_out_names

        if vel_act:
            ac = unnorm_out >> "action"
            assert ac.shape[-1] == 3
            ac_vel = ac[..., :2]
            ac_grab = ac[..., 2:]
        else:
            targ_pos = unnorm_out >> "target/position"
            if relative:
                targ_pos = targ_pos + self.closest_block_pos(
                    obs.leaf_apply(lambda arr: arr[:, 0]))  # make sure relative is also True in loss!
                unnorm_out.target.position = targ_pos
            Kv_P = np.asarray(Kv_P) if isinstance(targ_pos, np.ndarray) else torch.tensor(Kv_P, device=targ_pos.device)
            ac_vel = Kv_P * (unnorm_out.target.position - obs.position[:, 0])
            # B x H x N,   max over N blocks
            ac_grab = \
                torch.max(unnorm_out >> "target/grab_binary" if skip_grab else unnorm_out >> "target/grab_binary",
                          dim=-1,
                          keepdim=True)[0]

        ac_vel = torch.clip(ac_vel, -max_vel, max_vel)
        ac_grab = (ac_grab > GRAB_THRESH)

        # print(ac_grab.shape, ac_vel.shape, norm_out.leaf_apply(lambda arr: arr.shape))
        ac = torch.cat([ac_vel, ac_grab.to(dtype=torch.float32)], dim=-1).to(dtype=torch.float32)
        if get_first_action:
            unnorm_out = unnorm_out.leaf_apply(get_dist_first_horizon)
            ac = ac[:, 0]
        out.combine(unnorm_out)
        out.action = ac
        return out

    @staticmethod
    def get_default_policy_postproc_fn(nsld, policy_out_names, raw_out_name="policy_raw",
                                       use_policy_dist=False, use_policy_dist_mean=False,
                                       do_orn_norm=False, relative=False):
        names_to_shapes = {tup[0]: list(tup[1]) for tup in nsld if tup[0] in policy_out_names}
        names_to_sizes = {k: int(np.prod(v)) for k, v in names_to_shapes.items()}
        ordered_sizes = [names_to_sizes[k] for k in policy_out_names]
        policy_out_size = sum(names_to_sizes.values())

        if do_orn_norm:
            # very specific ordering required. meant for 3D envs
            assert policy_out_names[1] in ["target/ee_orientation_eul", "target/orientation_eul"], policy_out_names

        def robot_policy_postproc_fn(inputs: d, model_outputs: d):
            new_outs = model_outputs.leaf_copy()
            raw = model_outputs >> raw_out_name  # B x H x
            # gets the actions
            if relative:
                first = (inputs > policy_out_names).leaf_apply(lambda arr: arr[:, :1])
                cat_first = concatenate(first.leaf_apply(lambda arr: combine_after_dim(arr, 2)), policy_out_names,
                                        dim=-1)
                # each action is a delta, and they are baselined by the first value, which should be present here.
                raw = raw.cumsum(dim=1) + cat_first

            assert raw.shape[-1] == policy_out_size, raw.shape
            chunks = torch.split_with_sizes(raw, ordered_sizes, dim=-1)
            for chunk, key in zip(chunks, policy_out_names):
                new_outs[key] = split_dim(chunk, -1, names_to_shapes[key])
                if do_orn_norm and key in ["target/ee_orientation_eul", "target/orientation_eul"]:
                    new_outs[key] = torch.tanh(new_outs[key])
                    # outputs won't be normalized, but we want the output range to be euler angles (after tanh)
                    new_outs[key] = torch.stack([
                        new_outs[key][..., 0] * np.pi,
                        new_outs[key][..., 1] * np.pi / 2,
                        new_outs[key][..., 2] * np.pi,
                    ], dim=-1)
            return new_outs

        def robot_policy_dist_postproc_fn(inputs: d, model_outputs: d):
            # gets the actions
            new_outs = model_outputs.leaf_copy()
            dist = model_outputs >> raw_out_name  # dist
            if use_policy_dist_mean:
                sample = dist.mean  # mean
            else:
                sample = dist.rsample()  # sample
            new_outs.combine(unconcatenate(sample, policy_out_names, names_to_shapes))
            return new_outs

        return robot_policy_postproc_fn if not use_policy_dist else robot_policy_dist_postproc_fn

    def get_policy_postproc_fn(self, common_info, policy_out_names, use_policy_dist=False, do_orn_norm=False):
        return self.get_default_policy_postproc_fn(common_info >> "names_shapes_limits_dtypes", policy_out_names,
                                                   do_orn_norm=do_orn_norm,
                                                   use_policy_dist=use_policy_dist)

    def get_action_loss_fn(self, policy_out_names, single_grab, do_grab_norm, err_fn=mae_err_fn, use_outs=False,
                           relative=False,
                           mask_name=None, vel_act=False, policy_out_norm_names=None):
        if single_grab:
            raise NotImplementedError
        else:
            action_loss_fn = get_default_mae_action_loss_fn(policy_out_names, max_grab=1 if not do_grab_norm else None,
                                                            err_fn=err_fn, use_outs=use_outs, relative=relative,
                                                            mask_name=mask_name, vel_act=vel_act,
                                                            policy_out_norm_names=policy_out_norm_names)
        return action_loss_fn

    def get_default_lmp_names_and_sizes(self, common_info, plan_name, plan_size, INCLUDE_GOAL_PROPRIO, SINGLE_GRAB,
                                        DO_GRAB_NORM,
                                        ENCODE_ACTIONS=True, INCLUDE_BLOCK_SIZES=False, ENCODE_OBJECTS=True,
                                        PRIOR_OBJECTS_ONLY=False, POLICY_GOALS=True, VEL_ACT=False, USE_DRAWER=False,
                                        NO_OBJECTS=False, REAL_INPUTS=False, EXCLUDE_VEL=False, OBS_USE_QUAT=False,
                                        TARG_USE_QUAT=False):
        nsld = common_info >> "names_shapes_limits_dtypes"
        assert not USE_DRAWER
        assert not NO_OBJECTS

        # not in nlsd, be careful, NOTE : active constraints will not force the grab to happen...
        if VEL_ACT:
            assert not SINGLE_GRAB
            assert DO_GRAB_NORM
            policy_out_names = ['action']
        else:
            if SINGLE_GRAB:
                policy_out_names = ['target/position', 'target/grab_binary_single']
            else:
                policy_out_names = ['target/position', 'target/grab_binary']

        policy_out_size = config_utils.nsld_get_dims_for_keys(nsld, policy_out_names)

        proprio_names = ['position', 'velocity',
                         'grab_binary']  # proprioceptive cues, active constraints = force on each block
        proprio_size = config_utils.nsld_get_dims_for_keys(nsld, proprio_names)

        obs_enc_names = ['block_positions', 'block_velocities', 'block_angles',
                         'block_angular_velocities']  # 'active_constraints', 'maze']
        if INCLUDE_BLOCK_SIZES:
            obs_enc_names.append('block_sizes')
        # vis_enc_out_names = ['visual_encoding']  # just a stacking IRL
        obs_enc_size = config_utils.nsld_get_dims_for_keys(nsld, obs_enc_names)
        # visual_feature_size = obs_enc_size  # concat'd obs, no model here

        # states and actions
        TRAJECTORY_NAMES = proprio_names
        if ENCODE_OBJECTS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + obs_enc_names
        if ENCODE_ACTIONS:
            TRAJECTORY_NAMES = TRAJECTORY_NAMES + policy_out_names
            if not VEL_ACT and SINGLE_GRAB:
                # input still gets binary grab across objects
                TRAJECTORY_NAMES.remove('target/grab_binary_single')
                TRAJECTORY_NAMES.append('target/grab_binary')

        NORMALIZATION_NAMES = list(set(list(TRAJECTORY_NAMES) + obs_enc_names))
        SAVE_NORMALIZATION_NAMES = proprio_names + obs_enc_names + policy_out_names
        if not VEL_ACT and ENCODE_ACTIONS and not DO_GRAB_NORM:
            NORMALIZATION_NAMES.remove('target/grab_binary')  # do not normalize grabs
            NORMALIZATION_NAMES.remove('grab_binary')  # do not normalize grabs

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
            policy_out_size=policy_out_size,
            proprio_names=proprio_names,
            proprio_size=proprio_size,
            obs_enc_names=obs_enc_names,
            obs_enc_size=obs_enc_size,
            # waypoint names
            waypoint_names=self.wp_keys,
            action_names=self.ac_keys,
        )

    def default_get_contact_fn(self, inputs):
        c0 = self.contact_conds[0](inputs >> self.contact_names[0])

        if isinstance(c0, torch.Tensor):
            contact = torch.any(c0, dim=-1).view(-1)
            for i in range(1, len(self.contact_names)):
                next_c = torch.any(self.contact_conds[i](inputs >> self.contact_names[i]), dim=-1).view(-1)
                contact = torch.logical_or(contact, next_c)
        else:
            contact = np.any(c0, axis=-1).reshape(-1)
            for i in range(1, len(self.contact_names)):
                next_c = np.any(self.contact_conds[i](inputs >> self.contact_names[i]), axis=-1).reshape(-1)
                contact = np.logical_or(contact, next_c)
        return contact

    def get_parse_interaction_from_episode_fn(self, window=3, min_contact_len=1, max_contact_len=80, max_interaction_len=0,
                                              get_contact_fn=None, bounds=False):
        if get_contact_fn is None:
            get_contact_fn = self.default_get_contact_fn

        W = 4 if bounds else 2

        def parse_interaction_from_episode_fn(inputs, outputs):
            # first mask by contact
            contact = get_contact_fn(inputs)

            if len(contact) <= min_contact_len:
                # logger.warn("Skipping episode -- short episode!")
                return np.zeros((0, W)), 0

            # first pass: smooth
            mask = np.zeros(contact.shape, dtype=bool)
            for i in range(len(mask)):
                start = max(i - window, 0)
                end = min(i + window, len(mask))
                mask[i] = np.max(contact[start:end])
            # second pass: chunk
            # print(mask.shape)
            mask_pre = np.concatenate([[False], mask[:-1]])
            # first idx with smoothed contact
            contact_begins = np.logical_and(~mask_pre, mask).nonzero()[0]
            # first idx with no smoothed contact
            contact_ends = np.logical_and(mask_pre, ~mask).nonzero()[0]

            n_skip = 0
            to_keep = None
            if len(contact_begins) > 0:
                # cannot end more than beginnings.
                assert len(contact_ends) in [len(contact_begins) - 1,
                                             len(contact_begins)], "Too many/few contact endings"
                # no ending, create an artificial ending if we end in contact
                if len(contact_ends) == len(contact_begins) - 1:
                    contact_ends = np.append(contact_ends, len(mask))

                c_lens = contact_ends - contact_begins
                to_keep = c_lens >= min_contact_len
                if max_contact_len > 0:
                    to_keep = np.logical_and(to_keep, c_lens <= max_contact_len)

            # lists of start then end of each contact. (N, 2)
            # segments = np.stack([contact_begins, contact_ends], axis=1).reshape(-1, 2)
            if len(contact_begins) == 0:
                # logger.warn("Skipping episode -- no contact interactions!")
                return np.zeros((0, W)), 0
            else:
                # an interaction goes from the last end to the next start

                # NOTE : ends are the starting time (include), starts are the first steps with contact (exclude)
                prev_ends = np.concatenate([[0], contact_ends])[:-1]
                next_starts = np.concatenate([contact_begins, [len(mask)]])[1:]
                cstarts = contact_begins
                cends = contact_ends

                if max_interaction_len > 0 and to_keep is not None:
                    to_keep = np.logical_and(to_keep, next_starts - prev_ends + 1 <= max_interaction_len)

                if to_keep is not None:
                    n_skip = len(prev_ends)
                    prev_ends = prev_ends[to_keep]
                    n_skip = n_skip - len(prev_ends)  # how much was removed
                    next_starts = next_starts[to_keep]
                    if bounds:
                        cstarts = cstarts[to_keep]
                        cends = cends[to_keep]

                if bounds:
                    # N x 4
                    return np.stack([prev_ends, cstarts, cends, next_starts], axis=-1), n_skip
                else:
                    # N x 2, from last end, to next starts.
                    return np.stack([prev_ends, next_starts], axis=-1), n_skip

        return parse_interaction_from_episode_fn

    def get_contact_start_ends(self, inputs, current_horizon=None, ep_lengths=None):
        assert current_horizon is not None
        in_contact = self.default_get_contact_fn(inputs)
        if isinstance(in_contact, np.ndarray):
            # NOTE: these will be normalized, we can assume grab <= 0 means no grab.
            assert in_contact.dtype == np.bool
            first_contact = np.argmax(in_contact, axis=1)
            og_last_contact = np.argmax(np.flip(in_contact, axis=1), axis=1)
            last_contact = (in_contact.shape[1] - 1) - og_last_contact
            # last_contact = in_contact, [1]).argmax(dim=1)
        else:
            # NOTE: these will be normalized, we can assume grab <= 0 means no grab.
            assert in_contact.dtype == torch.bool
            _, first_contact = torch.max(in_contact, dim=1)
            _, og_last_contact = torch.max(torch.flip(in_contact, dims=[1]), dim=1)
            last_contact = (in_contact.shape[1] - 1) - og_last_contact
            # last_contact = in_contact, [1]).argmax(dim=1)

        if ep_lengths is not None:
            if (last_contact >= ep_lengths).any():
                import ipdb;
                ipdb.set_trace()
            assert (first_contact < ep_lengths).all()  # checking for improper data
            assert (last_contact < ep_lengths).all(), [last_contact[last_contact >= ep_lengths],
                                                       ep_lengths[
                                                           last_contact >= ep_lengths]]  # checking for improper data

        return first_contact, last_contact, in_contact

    @staticmethod
    def closest_block_pos(ins):
        bpos = ins >> "block_positions"
        epos = (ins >> "position")[..., None, :]
        # (B, H, N, 2) -> (B, H)
        _, min_block_idxs = torch.min((epos - bpos).abs().sum(-1), dim=-1)
        return unsqueeze_then_gather(bpos, min_block_idxs, dim=-2)  # B, H
