from enum import Enum
from typing import Tuple, List

import numpy as np
import torch

from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_with_default, get_required, AttrDict as d, is_array
from sbrl.utils.torch_utils import concatenate, to_torch, to_numpy, combine_after_last_dim


class TrajectoryType(Enum):
    pose_grab=0
    pose_grab_delta=1
    pose_delta_grab_abs=2
TT = TrajectoryType


map_default_mode_names = {
    TT.pose_delta_grab_abs : ['ee_position', 'ee_orientation_eul', 'gripper_pos'],
    TT.pose_grab : ['ee_position', 'ee_orientation_eul', 'gripper_pos'],
    TT.pose_grab_delta : ['ee_position', 'ee_orientation_eul', 'gripper_pos'],
}


class SthTrajectoryPolicy(Policy):
    def _init_params_to_attrs(self, params):
        self.trajectory_mode: TT = get_with_default(params, "trajectory_mode", TT.pose_delta_grab_abs)
        self.trajectory_names: List = get_with_default(params, "trajectory_names", map_default_mode_names[self.trajectory_mode])

        self.cat_name = get_with_default(params, "cat_name", None)
        # size of output
        self.total_steps = int(get_required(params, "total_steps"))
        # active length, may not be used (e.g. variable length)
        self.steps = int(get_required(params, "steps"))
        assert self.steps <= self.total_steps, "Steps of motion should be less than total steps"

    def _init_setup(self):
        self.reset_obs = d()
        self.x0 = None

    def warm_start(self, model, observation, goal):
        pass

    def parse(self, concat_arr, x0=None):
        if self.cat_name is None:
            x_dict = self._env_spec.parse_from_concatenated_flat(concat_arr, self.trajectory_names)  # T x dim
            out = d(x=x_dict)
            if x0 is not None:
                out.x0 = self._env_spec.parse_from_concatenated_flat(x0, self.trajectory_names)
        else:
            dc = {self.cat_name: concat_arr}
            if x0 is not None:
                dc['x0'] = x0
            out = d.from_dict(dc)
        return out

    def convert_trajectory(self, trajectory, in_mode: TT, out_mode: TT) -> d:
        if in_mode == out_mode:
            return trajectory

        assert self.cat_name is not None, "this is required for now"
        if in_mode == TT.pose_delta_grab_abs:
            if out_mode == TT.pose_grab:
                grab = (trajectory >> self.cat_name)[..., -1:]
                delta_from_start = np.cumsum((trajectory >> self.cat_name)[..., :6], axis=-2)
                start = (trajectory >> "x0")[..., None, :6]
                # [start + delta | grab]
                trajectory[self.cat_name] = np.concatenate([start + delta_from_start, grab], axis=-1)
                return trajectory
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def reset_policy(self, next_obs=None, next_goal=None, **kwargs):
        self.reset_obs = next_obs.leaf_apply(lambda arr: to_numpy(arr, check=True))
        # flatten, then concat
        flat = combine_after_last_dim(self.reset_obs.leaf_filter_keys(self.trajectory_names))
        self.x0 = concatenate(flat, self.trajectory_names, dim=-1)

    def get_trajectory(self, model, observation, goal, **kwargs) -> Tuple[d, TT]:
        return d(), TT.pose_delta_grab_abs

    def get_action(self, model, observation, goal, **kwargs):
        # NP input
        np_obs = observation.leaf_apply(lambda arr: to_numpy(arr, check=True) if is_array(arr) else arr)
        np_goal = goal.leaf_apply(lambda arr: to_numpy(arr, check=True) if is_array(arr) else arr)
        trajectory, mode = self.get_trajectory(model, np_obs, np_goal, **kwargs)
        converted_trajectory = self.convert_trajectory(trajectory, mode, self.trajectory_mode)
        converted_trajectory = self._env_spec.map_to_types(converted_trajectory, skip_keys=True)
        if self.trajectory_mode != TT.pose_grab:
            converted_trajectory.leaf_modify(lambda arr: to_torch(arr, device=model.device))
            self._env_spec.clip(converted_trajectory, converted_trajectory.leaf_key_intersection(self._env_spec.all_spec_names))
        return converted_trajectory


class GrabItemTrajectory(SthTrajectoryPolicy):

    def _init_params_to_attrs(self, params):
        super(GrabItemTrajectory, self)._init_params_to_attrs(params)
        assert self.steps > 0
        self.grip_strength_low = get_with_default(params, "grip_strength_low", 88)
        self.grip_strength_high = get_with_default(params, "grip_strength_high", 95)
        self.forward_steps_low = get_with_default(params, "forward_steps_low", 0.3)
        self.forward_steps_high = get_with_default(params, "forward_steps_high", 0.4)
        self.direction = get_with_default(params, "direction", np.array([0, 0, 1]), map_fn=np.asarray)

    def reset_policy(self, next_obs=None, next_goal=None, object_i=0, **kwargs):
        super(GrabItemTrajectory, self).reset_policy(next_obs, next_obs, **kwargs)
        # self.ee_0 = next_obs >> "ee_position"
        self.grip_tip_0 = next_obs >> "gripper_tip_position"
        self.target_pos = next_obs >> ("object_%d/position" % object_i)
        self.target_pos[..., 0] -= 0.01

        self.grip_strength = np.random.randint(self.grip_strength_low, self.grip_strength_high)
        self.forward_steps = int(self.steps * np.random.uniform(self.forward_steps_low, self.forward_steps_high))
        self.wait_steps = 6
        self.after_steps = self.steps - self.forward_steps - self.wait_steps

        self.lift_magnitude = 0.12  # TODO
        assert self.forward_steps > 0 and self.after_steps > 0

    def get_trajectory(self, model, observation, goal, **kwargs) -> Tuple[d, TT]:
        delta = (self.target_pos - self.grip_tip_0)
        sh = list(delta.shape[:-1]) + [7]
        delta_forward = np.zeros(sh)
        delta_forward[..., :3] = delta
        delta_after = np.zeros(sh)
        delta_after[..., :3] += self.lift_magnitude * self.direction  # move this amount after

        all_forward = [delta_forward / self.forward_steps] * self.forward_steps
        all_wait = [np.zeros_like(delta_forward)] * self.wait_steps
        all_after = [delta_after / self.after_steps] * self.after_steps
        all_last = [np.zeros_like(delta_forward)] * (self.total_steps - self.steps)

        # ... T x flat_dim
        concat_arr = np.stack(all_forward + all_wait + all_after + all_last, axis=-2)
        # grip dim
        concat_arr[..., :self.forward_steps, -1] = 0
        concat_arr[..., self.forward_steps:self.forward_steps+self.wait_steps, -1] = (np.arange(self.wait_steps) + 1) * (self.grip_strength / self.wait_steps)
        concat_arr[..., self.forward_steps+self.wait_steps:, -1] = self.grip_strength


        return self.parse(concat_arr, self.x0), TT.pose_delta_grab_abs


class PushItemTrajectory(SthTrajectoryPolicy):

    def _init_params_to_attrs(self, params):
        super(PushItemTrajectory, self)._init_params_to_attrs(params)
        assert self.steps > 0
        self.forward_steps_low = get_with_default(params, "forward_steps_low", 0.4)
        self.forward_steps_high = get_with_default(params, "forward_steps_high", 0.5)
        self.direction = get_with_default(params, "direction", np.array([1., 0]), map_fn=np.asarray)
        assert self.direction[0] >= 0., "Cannot push \"backwards\" yet"

    def reset_policy(self, next_obs=None, next_goal=None, object_i=0, **kwargs):
        super(PushItemTrajectory, self).reset_policy(next_obs, next_obs, **kwargs)
        # self.ee_0 = next_obs >> "ee_position"
        self.grip_tip_0 = next_obs >> "gripper_tip_position"
        self.target_pos = next_obs >> ("object_%d/position" % object_i)
        self.target_pos[..., 0] -= 0.01
        self.target_pos[..., 2] -= 0.005
        # tip should be 4cm offset from the target position (opp direction of motion)
        self.target_pos[..., :2] += -0.09 * self.direction

        self.forward_steps = int(self.steps * np.random.uniform(self.forward_steps_low, self.forward_steps_high))
        self.after_steps = self.steps - self.forward_steps

        self.push_distance = 0.12 + 0.09
        assert self.forward_steps > 0 and self.after_steps > 0

    def get_trajectory(self, model, observation, goal, **kwargs) -> Tuple[d, TT]:
        delta = (self.target_pos - self.grip_tip_0)
        sh = list(delta.shape[:-1]) + [7]
        delta_forward = np.zeros(sh)
        delta_forward[..., :3] = delta
        delta_after = np.zeros(sh)
        delta_after[..., :2] += self.push_distance * self.direction  # move this amount after

        all_forward = [delta_forward / self.forward_steps] * self.forward_steps
        all_after = [delta_after / self.after_steps] * self.after_steps
        all_last = [np.zeros_like(delta_forward)] * (self.total_steps - self.steps)

        # ... T x flat_dim
        concat_arr = np.stack(all_forward + all_after + all_last, axis=-2)
        # grip dim (gripper should be closed
        concat_arr[..., -1] = 210

        return self.parse(concat_arr, self.x0), TT.pose_delta_grab_abs


class LiftItemTrajectory(SthTrajectoryPolicy):

    def _init_params_to_attrs(self, params):
        super(LiftItemTrajectory, self)._init_params_to_attrs(params)
        assert self.steps > 0
        self.grip_strength_low = get_with_default(params, "grip_strength_low", 120)
        self.grip_strength_high = get_with_default(params, "grip_strength_high", 150)
        self.up_steps_low = get_with_default(params, "up_steps_low", 0.2)  # relative to total steps
        self.up_steps_high = get_with_default(params, "up_steps_high", 0.5)
        self.up_height_low = get_with_default(params, "up_steps_low", 0.1)  # m
        self.up_height_high = get_with_default(params, "up_steps_high", 0.2)

    def reset_policy(self, next_obs=None, next_goal=None, object_i=0, **kwargs):
        super(LiftItemTrajectory, self).reset_policy(next_obs, next_obs, **kwargs)
        # self.ee_0 = next_obs >> "ee_position"

        self.grip_strength = np.random.randint(self.grip_strength_low, self.grip_strength_high)
        self.up_steps = int(np.random.uniform(self.up_steps_low, self.up_steps_high) * self.total_steps)
        self.up_height = np.random.uniform(self.up_height_low, self.up_height_high)

        assert self.up_steps > 0 and self.up_height > 0, [self.up_steps, self.up_height]

    def get_trajectory(self, model, observation, goal, **kwargs) -> Tuple[d, TT]:
        delta_up = np.zeros_like(self.x0)
        delta_up[..., 2] += self.up_height  # go up this amount

        all_up = [delta_up / self.up_steps] * self.up_steps
        all_last = [np.zeros_like(delta_up)] * (self.total_steps - self.up_steps)

        # ... T x flat_dim
        concat_arr = np.stack(all_up + all_last, axis=-2)
        # grip dim
        concat_arr[..., -1] = self.grip_strength

        return self.parse(concat_arr, self.x0), TT.pose_delta_grab_abs


def interpolate_linear(start, end, schedule):
    # schedule is how much to progress at each state, normalized to sum to 1 per dimension
    if isinstance(schedule, int):
        schedule = np.arange(schedule)[:, None]

    # .. x T x dim
    schedule = schedule / schedule.sum(0)[None]

    all = []
    time = torch.zeros_like(schedule[0])
    for si in range(schedule.shape[0]):
        time += schedule[si]
        now = time * (end - start) + start
        all.append(now)
    return all
