from random import random

import numpy as np

from configs.exp_lfp.utils3d import clip_norm
from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import AttrDict, get_with_default
from sbrl.utils.torch_utils import to_numpy


class Waypoint2D:
    """
    Flexible specification of absolute / relative 2D waypoints.
    """

    def __init__(self, pos, gripper, timeout=np.inf,
                 relative_to_parent=True,
                 relative_to_robot=-1,
                 relative_to_object=-1,
                 relative_gripper=False,
                 check_reach=True,
                 max_vel=None, ):
        # if False, will use the last state, not the last waypoint of the given robot/object
        self.relative_to_parent = relative_to_parent
        # ... -1 = robot0, 0 = None, 1 = object0 ... ignored if relative_to_parent == True
        self.relative_to_robot = relative_to_robot
        self.relative_to_object = relative_to_object

        assert relative_to_robot < 0 or relative_to_object < 0, "Cannot be relative to both..."
        if relative_to_parent:
            assert relative_to_robot >= 0 or relative_to_object >= 0, "If relative to parent, must specify either ego or object"

        self._base_pos = pos.copy()
        self._base_gripper = gripper
        self.curr_pos = None

        self.timeout = timeout
        self.check_reach = check_reach  # will check reached, otherwise wait til timeout

        self.max_vel = max_vel  # policy can choose to use this to limit vel

    def update(self, parent, robot_pos, object_pos, gripper):
        # get the source
        if self.relative_to_robot < 0 and self.relative_to_object < 0:
            # don't keep creating things if not relative to anything.
            self.curr_pos = self._base_pos.copy() if self.curr_pos is None else self.curr_pos
        else:
            if self.relative_to_parent:
                rel_source_pos = parent.pos  # last cf,gripper of the parent
            else:
                rel_source_pos = robot_pos[self.relative_to_robot] if self.relative_to_robot >= 0 else object_pos[
                    self.relative_to_object]

            self.curr_pos = self._base_pos + rel_source_pos

        return self.pos, self._base_gripper

    @property
    def pos(self):
        return self.curr_pos.copy()

    @property
    def gripper(self):
        return self._base_gripper


class Waypoint2DPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._max_vel = get_with_default(params, "max_vel", 150., map_fn=float)
        self._clip_norm = get_with_default(params, "clip_norm", True)  # else will clip by action space only.

    def _init_setup(self):
        assert self._env is not None
        assert isinstance(self._env, BlockEnv2D), type(self._env)

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, pos_waypoints=None, ptype=0, tolerance=10, name="", smooth_noise=0., noise_period=5,
                     min_noise_dist=0.075, random_slow_prob=0, random_slow_duration_bounds=(5, 15),
                     random_slowness_bounds=(0, 0.25), stop_prob=0.2, **kwargs):
        assert len(pos_waypoints) > 0, pos_waypoints
        for wp in pos_waypoints:
            assert isinstance(wp, Waypoint2D), type(wp)

        self._pos_waypoints = pos_waypoints
        self._curr_idx = 0
        self._curr_step = 0
        self._ptype = ptype
        self._name = name

        self._tolerance = int(tolerance)
        self._done = False

        # NOISE
        self.smooth_noise = smooth_noise
        self._noise_obj = None
        self._noise_steps = 0
        self._curr_noisy_goal = np.zeros(3)

        self._min_noise_dist = min_noise_dist
        self._noise_period = noise_period
        self._latest_spatial_desired_pos = None

        # temporal
        self._random_slow_prob = random_slow_prob  # at each step, likelihood of entering a slow period.
        self._random_slow_duration_bounds = tuple(
            random_slow_duration_bounds)  # (low,high) how long to be slow for before returning to plan
        self._random_slowness_bounds = tuple(random_slowness_bounds)  # (low, high) what percentage of max speed to go
        self._stop_prob = stop_prob  # in a given slow interval, upweight the probability of stopping (slowness = 0) completely.

        self._curr_slowness = None  # 3-element AttrDict (curr_step, slowness, max_step)
        self.num_steps = 0

    def get_action(self, model, observation, goal, **kwargs):
        keys = ['position', 'grab_binary']
        # index out batch and horizon
        pos, gr = (observation > keys).leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)).get_keys_required(keys)

        parent = None if self._curr_idx == 0 else self._pos_waypoints[self._curr_idx - 1]

        wp = self._pos_waypoints[self._curr_idx]
        if wp.curr_pos is not None:
            reached = self.reached(pos, gr, wp) if wp.check_reach else False
            if reached or self._curr_step > wp.timeout:
                if self._curr_idx < len(self._pos_waypoints) - 1:
                    self._curr_idx += 1
                    self._curr_step = 0
                else:
                    self._done = True

        wp = self._pos_waypoints[self._curr_idx]
        wp_pos, wp_grip = wp.update(parent, [pos], np.zeros((0, 2)), gr)

        wp_pos = self.update_noisy_goal(pos, wp_pos)

        # compute the action
        dpos = wp_pos - pos

        mv = wp.max_vel
        if mv is None:
            mv = self._max_vel

        if self._clip_norm:
            dpos = clip_norm(dpos, mv * self._env.dt)

        goal_gr = wp_grip

        self._curr_step += 1

        out = AttrDict(
            target=AttrDict(
                position=wp_pos,
                grab_binary=np.array([wp_grip]),
            ),
            action=np.concatenate([dpos / self._env.dt, [goal_gr]]),
            policy_name=np.array([self.curr_name]),
            policy_type=np.array([self.policy_type]),
        ).leaf_apply(lambda arr: arr[None])

        if not self._clip_norm:
            # modifies in place
            self._env_spec.clip(out, out.list_leaf_keys(), object_safe=True)

        return out

    def update_noisy_goal(self, pos, desired_pos):

        delta = desired_pos - pos
        delta_mag = np.linalg.norm(delta)

        # spatial
        if self.smooth_noise > 0:
            if self._latest_spatial_desired_pos is None:
                self._latest_spatial_desired_pos = desired_pos.copy()

            # smoothing logic when outside min_noise_dist from goal
            if delta_mag >= self._min_noise_dist:
                if self._noise_steps % self._noise_period == 0:
                    intermediate = pos + delta
                    perp_noise = delta_mag * self.smooth_noise  # some fraction of the distance to the goal is the noise we add
                    tangent_vec = np.array([delta[1], -delta[0]])  # perpendicular
                    # logger.debug(f"{tangent_vec}, {perp_noise}, {delta.dot(tangent_vec)}")
                    tangent_vec /= np.linalg.norm(tangent_vec)
                    perp_noise = np.random.uniform(0.1 * perp_noise, perp_noise)  # random amount of noise
                    self._curr_noisy_goal = intermediate + perp_noise * tangent_vec

                desired_pos = self._latest_spatial_desired_pos * 0.5 + self._curr_noisy_goal * 0.5
                self._noise_steps += 1

            else:
                # exp avg
                desired_pos = self._latest_spatial_desired_pos * 0.5 + desired_pos * 0.5

            self._latest_spatial_desired_pos = desired_pos.copy()  # copy before any other mutations

        # temporal, done after other things.
        if self._random_slow_prob > 0:

            if self._curr_slowness is None and random() < self._random_slow_prob:
                # new window
                # logger.debug(f"Starting new slow...{self._stage}")
                self._curr_slowness = AttrDict(
                    step=0,
                    slowness=0. if random() < self._stop_prob else np.random.uniform(*self._random_slowness_bounds),
                    max_step=np.random.randint(*self._random_slow_duration_bounds),
                    initial_pos=pos.copy()
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
                    return desired_pos

                # compute closer delta (by slowness factor) for action frame
                if slowness < 1e-11:
                    pos_new = self._curr_slowness >> "initial_pos"  # stabilize
                    # oriq_new = euler2quat(ori)
                else:
                    pos_new = slowness * desired_pos + (1 - slowness) * pos
                    # oriq_new = quat_slerp(euler2quat(ori), desired_tip_frame.orn, slowness)

                # print(self._stage, pos, desired_tip_frame.pos, pos_new, step, max_step)

                # todo fix the rotation stuff
                desired_pos = pos_new.copy()

        return desired_pos

    @property
    def curr_name(self) -> str:
        # returns a string identifier for the policy, rather than ints.
        return self._name

    @property
    def policy_type(self) -> int:
        # returns a string identifier for the policy, rather than ints.
        return self._ptype

    def reached(self, pos, gr, wp):
        return np.linalg.norm(wp.pos - pos) < self._tolerance

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done


def get_basic_nav_policy_params(obs, goal, env=None, random_motion=True, **kwargs):
    assert env is not None
    # basic nav stuff
    bn_center_pos = env.get_bottleneck_center_pos()
    bn_idxs = env.curr_bottleneck_indices
    horizontal = env.horizontal
    up_offset = np.array([0, 40]) if horizontal else np.array([-40, 0])
    goal_pos = to_numpy(obs >> "goal_position", check=True).reshape(2)

    name = "basic_bottleneck"
    for i in bn_idxs:
        name += f"_{i}"

    # + 25% tangent direction noise
    noise = (lambda x: x) if not random_motion else (lambda x: x + np.flip(up_offset) * np.random.uniform(-0.25, 0.25))
    wps = []
    for i, bn in enumerate(bn_center_pos):
        above = Waypoint2D(noise(bn + up_offset), False, timeout=int(6 / env.dt),
                           relative_to_parent=False, )
        through = Waypoint2D(noise(bn - up_offset), False, timeout=int(3 / env.dt),
                             relative_to_parent=False, )

        wps.extend([above, through])

    # reach goal
    wps.append(Waypoint2D(goal_pos, False, timeout=int(6 / env.dt),
                          relative_to_parent=False, ))
    # wait for a sec
    wps.append(Waypoint2D(goal_pos, False, timeout=int(1 / env.dt),
                          relative_to_parent=False, check_reach=False))

    return AttrDict(pos_waypoints=wps, name=name, **kwargs)
