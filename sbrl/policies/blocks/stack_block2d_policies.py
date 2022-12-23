import numpy as np
import pymunk

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d
## ROTATE RIGHT / LEFT
from sbrl.utils.torch_utils import to_numpy


class TipBlockPrimitive(PushPrimitive):
    def reset_policy(self, bbox=None, **kwargs):
        super(TipBlockPrimitive, self).reset_policy(**kwargs)
        # left right top bottom
        assert bbox is not None and len(bbox) == 4
        assert np.abs(self._target_offset[0]) > 0, "x offset must be nonzero (tip from side)"
        mid_height = (bbox[3] + bbox[2]) / 2
        starting_offset_max = 0.5 * np.max(self._env.ego_block_size)
        assert bbox[2] + starting_offset_max >= self._target_offset[1] + mid_height >= bbox[
            2], f"y offset must be in the range of the top: {bbox}, {self._target_offset}, {mid_height}"
        assert bbox[1] - bbox[0] < bbox[2] - bbox[3], "height must be greater than width to tip over"
        assert self._push_steps > 5, "must push for enough time"
        self._push_direction[1] = 0.  # no y movement

    def get_action(self, model, observation, goal, **kwargs):
        action = super(TipBlockPrimitive, self).get_action(model, observation, goal, **kwargs)
        action.policy_type = np.array([2])[None]
        action.policy_name = np.array([self.curr_name])[None]
        return action

    @property
    def curr_name(self) -> str:
        return 'tip'


class RotateBlockPrimitive(PullPrimitive):
    def reset_policy(self, bbox=None, rotate_left=False, down_distance_frac=0.5,  down_speed_scale=1., **kwargs):
        self._up_distance = 90.
        if 'speed_scale' not in kwargs.keys():
            kwargs['pull_steps'] = 1.
        vel_up = kwargs['speed_scale'] * (kwargs['offset'][1] + 1e-11)
        # (dist / vel) / dt = time / dt = steps
        kwargs['pull_steps'] = int((self._up_distance / vel_up) / self._env.dt)
        super(RotateBlockPrimitive, self).reset_policy(**kwargs)
        assert bbox is not None and len(bbox) == 4
        # assert bbox[1] - bbox[0] > bbox[2] - bbox[3], "width must be greater than height to rotate over"
        self._width = bbox[1] - bbox[0]
        self._pos_at_top = None
        self._rotate_left = rotate_left
        self._vel_down = 150. * down_speed_scale
        self._down_steps = int(
            (down_distance_frac * self._up_distance / self._vel_down) / self._env.dt)  # e.g., 50% of the elevation
        assert np.abs(self._target_offset[1]) > 0, "y offset must be nonzero (rotate from top)"

    def get_action(self, model, observation, goal, **kwargs):
        if self._stage == 2:
            # no longer waiting, disconnect and rapidly move to right / left
            curr_pos = (observation >> "position")[0, 0]
            if self._pos_at_top is None:
                self._pos_at_top = curr_pos.copy()

            sgn = -1 if self._rotate_left else 1
            if sgn * (curr_pos[0] - self._pos_at_top[0]) >= self._width / 2:
                self._stage += 1
                self._curr_step = 0
                v = np.array([0, 0])
            else:
                v = np.array([sgn * self._vel_down, 0])  # 150 per sec (fast)
            setpoint_position = self._pos_at_top + np.array([0, sgn * self._width / 2.])
            setpoint_grab = np.zeros((1, self._num_blocks))  # detach
            v = np.append(v, 0)  # without grabbing
            setpoint_position = setpoint_position[None]
            ac = d(action=v[None])
            self.num_steps += 1
        elif self._stage == 3:
            # move down fast
            v = np.zeros(3)
            setpoint_position = observation.position[0, 0].copy()
            if self._curr_step < self._down_steps:
                v = -np.array([0, self._vel_down])  # move down FAST to finish rot
                setpoint_position += v * self._env.dt * (self._down_steps - self._curr_step)
                v = np.append(v, 0)
                self._curr_step += 1
                # with out grabbing
            elif self._curr_step < self._down_steps + 10:
                self._curr_step += 1
            else:
                self._curr_step = 0
                self._stage += 1
            setpoint_position = setpoint_position[None]
            setpoint_grab = np.zeros((1, self._num_blocks))
            ac = d(action=v[None])
            self.num_steps += 1
        else:
            ac = super(RotateBlockPrimitive, self).get_action(model, observation, goal, **kwargs)
            setpoint_position = ac >> "target/position"
            setpoint_grab = ac >> "target/grab_binary"

        ac.policy_type = np.array([3])[None]
        ac.policy_name = np.array([self.curr_name])[None]
        ac['target/position'] = setpoint_position.copy()
        ac['target/grab_binary'] = setpoint_grab.copy()
        return ac

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self._stage >= 4

    @property
    def curr_name(self) -> str:
        direction = "left" if self._rotate_left else "right"
        return f'rotate_{direction}'


class SideRotateBlockPrimitive(PullPrimitive):
    def reset_policy(self, bbox=None, randomize_offset=True, down_speed_scale=1., up_margin=60., **kwargs):
        assert bbox is not None and len(bbox) == 4

        self._width = np.abs(bbox[1] - bbox[0])
        self._height = np.abs(bbox[2] - bbox[3])
        self._up_distance = max(self._width, self._height) + up_margin  # raise with margin

        if 'speed_scale' not in kwargs.keys():
            kwargs['pull_steps'] = 1.

        vel_up = kwargs['speed_scale'] * 75
        # (dist / vel) / dt = time / dt = steps
        kwargs['pull_steps'] = int((self._up_distance / vel_up) / self._env.dt)

        super(SideRotateBlockPrimitive, self).reset_policy(**kwargs)
        assert np.abs(self._target_offset[0]) > 0, "x offset must be nonzero (rotate from side)"
        x_offset_min, x_offset_max = self.compute_x_offset_bounds(self._width, self._height)
        # self._block_size = (nobs >> "block_sizes")[0, self._block_idx]

        if randomize_offset:
            self._target_offset[0] = np.sign(self._target_offset[0]) * np.random.uniform(x_offset_min, x_offset_max)
        else:
            # just clip
            self._target_offset[0] = np.sign(self._target_offset[0]) * np.clip(np.abs(self._target_offset[0]),
                                                                               x_offset_min, x_offset_max)
        self._target_offset[1] = 0  # required.

        assert np.sign(self._target_offset[0]) == np.sign(
            self._pull_tangent), "must be pulling in the correct tangent dir"

        # assert bbox[1] - bbox[0] > bbox[2] - bbox[3], "width must be greater than height to rotate over"
        self._rotate_left = self._target_offset[0] < 0  # left side of the block
        self._vel_down = 80. * down_speed_scale
        self._down_steps = int((0.5 * self._up_distance / self._vel_down) / self._env.dt)  # e.g., 50% of the elevation

    def compute_x_offset_bounds(self, width, height):
        # enough to contact at tether between 45 degrees but contact at all.
        x_offset_min = (width + np.sqrt(2) * np.max(self._env.ego_block_size)) / 2
        x_offset_max = np.linalg.norm([0.5 * (width + np.max(self._env.ego_block_size)),
                                       0.5 * (height + np.max(self._env.ego_block_size))])
        if x_offset_min < x_offset_max - 4:
            x_offset_min += 2
            x_offset_max -= 2  # buffer if we can
        return x_offset_min, x_offset_max

    def get_action(self, model, observation, goal, **kwargs):

        if self._stage == 2:
            # move down gradually to stabilize object (no longer grasping)
            v = np.zeros(3)
            setpoint_position = observation.position[0, 0].copy()
            if self._curr_step < self._down_steps:
                v = -np.array([0, self._vel_down])  # move down FAST to finish rot
                setpoint_position += v * self._env.dt * (self._down_steps - self._curr_step)
                v = np.append(v, 0)
                self._curr_step += 1
                # with out grabbing
            elif self._curr_step < self._down_steps + 5:  # short wait after push down
                self._curr_step += 1
            else:
                self._curr_step = 0
                self._stage += 1
            setpoint_position = setpoint_position[None]
            setpoint_grab = np.zeros((1, self._num_blocks))
            ac = d(action=v[None])
            self.num_steps += 1
        else:
            if self._stage == 0:
                curr_bbox = observation.block_bounding_boxes[0, 0, self._block_idx]
                w, h = curr_bbox[1] - curr_bbox[0], curr_bbox[2] - curr_bbox[3]
                x_offset_min, x_offset_max = self.compute_x_offset_bounds(w, h)

                # just clip the random one to make sure
                self._target_offset[0] = np.sign(self._target_offset[0]) * np.clip(np.abs(self._target_offset[0]),
                                                                                   x_offset_min, x_offset_max)
            ac = super(SideRotateBlockPrimitive, self).get_action(model, observation, goal, **kwargs)
            setpoint_position = ac >> "target/position"
            setpoint_grab = ac >> "target/grab_binary"

        ac.policy_type = np.array([4])[None]
        ac.policy_name = np.array([self.curr_name])[None]
        ac['target/position'] = setpoint_position.copy()
        ac['target/grab_binary'] = setpoint_grab.copy()
        return ac

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self.num_steps >= self._timeout or self._stage >= 3 + int(self._retreat_steps > 0)

    @property
    def curr_name(self) -> str:
        direction = "left" if self._rotate_left else "right"
        return f'{direction}_side_rotate'


### other helpers
def get_valid_paths_from_bbox(env, obs: d, margin=None):
    p, block_ps, bbox, block_bbox = obs.leaf_apply(lambda arr: arr[0]).get_keys_required(
        ['position', 'block_positions', 'bounding_box', 'block_bounding_boxes'])
    all_block_bboxes = [
        pymunk.BB(left=block_bbox[i, 0], right=block_bbox[i, 1], top=block_bbox[i, 2], bottom=block_bbox[i, 3])
        for i in range(len(block_ps))]
    ego_bbox = pymunk.BB(left=bbox[0], right=bbox[1], top=bbox[2], bottom=bbox[3])

    if margin is None:
        ewm = 0.5 * np.abs(ego_bbox.left - ego_bbox.right)
        ehm = 0.5 * np.abs(ego_bbox.top - ego_bbox.bottom)
    else:
        ewm = margin
        ehm = margin

    all_paths = []
    for i in range(len(block_ps)):
        bb = all_block_bboxes[i]
        width = np.abs(bb.left - bb.right)
        height = np.abs(bb.top - bb.bottom)
        targets = np.array([
            bb.center() + [ewm + width / 2, 0],
            bb.center() + [-ewm - width / 2, 0],
            bb.center() + [0, ehm + height / 2],
            bb.center() + [0, -ehm - height / 2],
        ])
        coll_free = np.asarray([not bb.intersects_segment(a=ego_bbox.center(), b=tuple(targets[i])) for i in range(4)])
        in_bounds = np.all(np.logical_and(targets >= 0, targets <= env.grid_size), axis=-1)
        all_paths.append(targets[np.logical_and(coll_free, in_bounds)])
    return all_paths


def get_push_pull_lift_memory_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                    min_retreat, max_retreat, random_side=True):
    def memory_policy_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        # sides we can get to in a straight path
        valid_targets = get_valid_paths_from_bbox(env, obs.leaf_apply(lambda arr: arr[:, 0]))
        all_valid = np.concatenate(valid_targets, axis=0)
        if len(all_valid) == 0:
            return None, d()
        # pick a side that is close
        p, block_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions'])
        p = to_numpy(p, check=True)
        block_ps = to_numpy(block_ps, check=True)

        dists = np.linalg.norm(p[None] - all_valid, axis=-1)

        # PICK THE PATH TO TAKE
        if random_side:
            path_idx = np.random.choice(len(dists))
        else:
            path_idx = np.argmin(dists)

        vt = all_valid[path_idx]
        # block idx is one minus the first to be greater
        block_idx = np.argmax(np.cumsum([len(vt) for vt in valid_targets]) > path_idx) - 1
        assert block_idx < len(block_ps)
        block_bbox = (obs >> "block_bounding_boxes")[0, 0, block_idx]
        bbox = (obs >> "bounding_box")[0, 0]

        # if vt[1] > block_bbox[2]:  # TOP
        #     print("pull")
        #     vt[0] += np.random.randn() * env.block_size[0] / 8  # x axis noise
        # else:
        #     print("push")
        #     vt[1] += np.random.randn() * env.block_size[1] / 8  # y axis noise

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        allowed_policies = [0, 1]
        if vt[1] > block_bbox[2]:  # above top
            # print("push action removed")
            allowed_policies.remove(0)  # only pulling allowed from above

        if memory.reset_count < memory.max_iters:
            offset = 1.2 * (vt - block_ps[block_idx])
            # for each primitive (the direction of travel)
            movement_dirs = np.stack([-offset, offset])[allowed_policies]
            # against the left wall
            moving_against_the_wall = np.array([False, False])[allowed_policies]
            if block_bbox[0] <= 5 or bbox[0] <= 0:
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] < 0)
            # against the right wall
            if env.grid_size[0] - block_bbox[1] <= 5 or env.grid_size[0] - bbox[1] <= 0:
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] > 0)

            # policies not against the right wall or against the left wall
            allowed_policies = np.array(allowed_policies)[~moving_against_the_wall]
            if len(allowed_policies) == 0:
                # end the execution if we get trapped
                policy_idx = None
                pps = d()
            else:

                # print(allowed_policies, moving_against_the_wall)
                policy_idx = np.random.choice(allowed_policies)
                # upward component is randomized for retreat
                retreat_dir = offset + np.array([0., np.random.uniform(0, abs(offset[0]))])
                pps = d(
                    block_idx=block_idx,
                    offset=offset,
                    kp_vel=[1.5, 5.],
                    speed_scale=np.random.uniform(1, 1.5),
                    retreat_steps=np.random.randint(min_retreat, max_retreat),
                    retreat_direction=retreat_dir,
                    timeout=70,
                )
                if policy_idx == 0:
                    pps.push_steps = np.random.randint(5, 15)
                else:
                    # ptan = policy_idx - 2
                    policy_idx = 1
                    pps.pull_steps = np.random.randint(5, 15)
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0

                memory.reset_count += 1
        else:
            # done with max iters
            policy_idx = None
            pps = d()
        return policy_idx, pps

    return memory_policy_params_fn


def get_push_pull_lift_rotate_memory_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                           min_retreat, max_retreat, random_side=True,
                                                           randomize_offset=False, oversample_rot=False,
                                                           prefer_idxs=()):
    def memory_policy_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        # sides we can get to in a straight path
        valid_targets = get_valid_paths_from_bbox(env, obs.leaf_apply(lambda arr: arr[:, 0]))
        all_valid = np.concatenate(valid_targets, axis=0)

        if len(all_valid) == 0:
            return None, d()

        # pick a side that is close
        p, block_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions'])
        p = to_numpy(p, check=True)
        block_ps = to_numpy(block_ps, check=True)

        dists = np.linalg.norm(p[None] - all_valid, axis=-1)

        # PICK THE PATH TO TAKE
        if random_side:
            path_idx = np.random.choice(len(dists))
        else:
            path_idx = np.argmin(dists)

        vt = all_valid[path_idx]
        # block idx is one minus the first to be greater
        block_idx = np.argmax(np.cumsum([len(v) for v in valid_targets]) > path_idx) - 1
        assert block_idx < len(block_ps)
        block_bbox = (obs >> "block_bounding_boxes")[0, 0, block_idx]
        bbox = (obs >> "bounding_box")[0, 0]

        # if vt[1] > block_bbox[2]:  # TOP
        #     print("pull")
        #     vt[0] += np.random.randn() * env.block_size[0] / 8  # x axis noise
        # else:
        #     print("push")
        #     vt[1] += np.random.randn() * env.block_size[1] / 8  # y axis noise

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        allowed_policies = [0, 1, 2, 3]
        above_top = vt[1] > block_bbox[2]
        on_ground = block_bbox[3] < 20  # bottom is almost on the ground
        # 50% taller than wide is the only way we can tip
        tippable = 1.5 * (block_bbox[1] - block_bbox[0]) < block_bbox[2] - block_bbox[3]

        # push
        if above_top:  # if the side chosen is above the top
            allowed_policies.remove(0)  # only pulling allowed from above

        # tip from side
        if not on_ground or above_top or not tippable:
            allowed_policies.remove(2)

        # rotate from ground, starting above
        if not on_ground or not above_top:
            allowed_policies.remove(3)

        # print(above_top, on_ground, tippable, allowed_policies)
        if memory.reset_count < memory.max_iters:
            scale_factor = 1.2
            base_offset = (vt - block_ps[block_idx])
            if randomize_offset:
                scale_factor = np.random.uniform(1.2, 1.3)
            offset = scale_factor * base_offset
            # for each primitive (the direction of travel)
            movement_dirs = np.stack([-offset, offset, -offset * np.array([1, 0]), offset])[allowed_policies]
            # against the left wall
            moving_against_the_wall = np.zeros(4).astype(bool)[allowed_policies]
            grounded = bbox[3] <= block_bbox[2] and block_bbox[3] <= 20
            # if block is too far left, or we are wedged between on the left
            if block_bbox[0] <= 5 or \
                    (bbox[0] <= 5 and block_bbox[1] <= 15 and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] < 0)
            # against the right wall, or we are wedged
            if env.grid_size[0] - block_bbox[1] <= 5 or \
                    (env.grid_size[0] - 5 <= bbox[1] and env.grid_size[0] - 15 <= block_bbox[0] and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] > 0)

            # policies not against the right wall or against the left wall
            allowed_policies = np.array(allowed_policies)[~moving_against_the_wall]
            # print("before select", allowed_policies)
            # print(allowed_policies, prefer_idxs)
            if len(allowed_policies) == 0 or (
                    len(prefer_idxs) > 0 and len(set(allowed_policies).intersection(prefer_idxs)) == 0):
                # end the execution if we get trapped
                policy_idx = None
                pps = d()
            else:
                # print(allowed_policies, moving_against_the_wall)
                p = np.ones(len(allowed_policies))
                if oversample_rot and 2 in allowed_policies:
                    # print(np.argmax(allowed_policies == 2))
                    p[np.argmax(allowed_policies == 2)] = 8.  # 8x likely to sample tipping
                if oversample_rot and 3 in allowed_policies:
                    # print(np.argmax(allowed_policies == 3))
                    p[np.argmax(allowed_policies == 3)] = 2.  # twice as likely to sample rotate

                for idx in set(allowed_policies).intersection(prefer_idxs):
                    p[np.argmax(allowed_policies == idx)] = 100.  # effectively guarantee these idxs.

                p = p / p.sum()
                policy_idx = np.random.choice(allowed_policies, p=p)
                # upward component is randomized for retreat
                retreat_dir = offset + np.array([0., np.random.uniform(0, abs(offset[0]))])
                pps = d(
                    block_idx=block_idx,
                    offset=offset,
                    kp_vel=[1.5, 5.],
                    speed_scale=np.random.uniform(1, 1.5),
                    retreat_steps=np.random.randint(min_retreat, max_retreat),
                    retreat_direction=retreat_dir,
                    timeout=100,
                )
                if policy_idx == 0:
                    # push
                    pps.push_steps = np.random.randint(5, 15)
                elif policy_idx == 1:
                    # pull
                    pps.pull_steps = np.random.randint(5, 15)
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0
                    if min_retreat == 0 and max_retreat == 1:  # if no retreat, do not wait.
                        pps.wait_steps = 0
                elif policy_idx == 2:
                    # tipping
                    # get rid of the offset component of the retreat dir
                    new_base_offset = np.array([base_offset[0], (block_bbox[2] - block_bbox[3]) / 2.])
                    if randomize_offset:
                        scale_factor = np.random.uniform(1.05, 1.3)
                    pps.offset = new_base_offset * scale_factor
                    pps.retreat_direction[1] -= offset[1]
                    pps.push_steps = np.random.randint(10, 15)
                    pps.bbox = block_bbox
                elif policy_idx == 3:
                    # rotating
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    pps.down_distance_frac = np.random.uniform(0.5, 0.75)
                    pps.bbox = block_bbox
                    pps.rotate_left = np.random.choice([True, False])
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0
                else:
                    raise NotImplementedError

                memory.reset_count += 1
        else:
            # done with max iters
            policy_idx = None
            pps = d()
        return policy_idx, pps

    return memory_policy_params_fn


def get_push_pull_lift_tip_srot_rot_memory_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                 min_retreat, max_retreat, random_side=True, randomize_offset=False, undersample_lift=False,
                                                 oversample_tip=False, no_lift_rot=False, no_push=False, no_pull=False, no_tip=False, prefer_idxs=()):
    def memory_policy_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        # sides we can get to in a straight path
        valid_targets = get_valid_paths_from_bbox(env, obs.leaf_apply(lambda arr: arr[:, 0]))

        # pick a side that is close
        p, block_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions'])
        p = to_numpy(p, check=True)
        block_ps = to_numpy(block_ps, check=True)

        # if no_lift_rot:
        #     new_valid_targets = []
        #     for b_idx, vt in enumerate(valid_targets):
        #         bbox = (obs >> "block_bounding_boxes")[0, 0, b_idx]
        #         x_center = (bbox[0] + bbox[1]) / 2
        #         new_valid_targets.append(vt[np.abs(vt[:, 0] - x_center) > 3])
        #     valid_targets = new_valid_targets

        all_valid = np.concatenate(valid_targets, axis=0)

        if len(all_valid) == 0:
            return None, d()

        dists = np.linalg.norm(p[None] - all_valid, axis=-1)

        # PICK THE PATH TO TAKE
        p = []
        if undersample_lift:
            for i in range(len(valid_targets)):
                offset = (valid_targets[i] - block_ps[i:i+1])
                # within (80, 100) degrees
                p.append(np.where(np.abs(np.pi/2 - np.arctan2(offset[:, 1], offset[:, 0])) < np.pi/18, 0.2, 1.))
            p = np.concatenate(p)
            p = p / p.sum()

        if random_side:
            if len(p) == 0:
                p = None
            path_idx = np.random.choice(len(dists), p=p)
        else:
            path_idx = np.argmin(dists)

        vt = all_valid[path_idx]
        # block idx is one minus the first to be greater
        block_idx = np.argmax(np.cumsum([len(v) for v in valid_targets]) > path_idx) - 1
        assert block_idx < len(block_ps)
        block_bbox = (obs >> "block_bounding_boxes")[0, 0, block_idx]
        bbox = (obs >> "bounding_box")[0, 0]

        # if vt[1] > block_bbox[2]:  # TOP
        #     print("pull")
        #     vt[0] += np.random.randn() * env.block_size[0] / 8  # x axis noise
        # else:
        #     print("push")
        #     vt[1] += np.random.randn() * env.block_size[1] / 8  # y axis noise

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        allowed_policies = [0, 1, 2, 3, 4] if not no_lift_rot else [0, 1, 2, 4]
        if no_tip:
            allowed_policies.remove(2)
        if no_pull:
            allowed_policies.remove(1)
        if no_push:
            allowed_policies.remove(0)

        above_top = vt[1] > block_bbox[2]
        below_bottom = vt[1] < block_bbox[3]
        on_ground = block_bbox[3] < 20  # bottom is almost on the ground
        # 30% taller than wide is the only way we can tip
        tippable = 1.3 * (block_bbox[1] - block_bbox[0]) < block_bbox[2] - block_bbox[3]

        # push
        if not no_push and above_top:  # if the side chosen is above the top
            allowed_policies.remove(0)  # only pulling allowed from above

        # tip from side
        if not no_tip and (not on_ground or above_top or not tippable):
            allowed_policies.remove(2)

        # rotate from ground, starting above
        if not no_lift_rot and (not on_ground or not above_top):
            allowed_policies.remove(3)

        # rotate from ground, starting on the side
        if not on_ground or above_top:
            allowed_policies.remove(4)

        # print(above_top, on_ground, tippable, allowed_policies)
        if memory.reset_count < memory.max_iters:
            scale_factor = 1.2
            base_offset = (vt - block_ps[block_idx])
            if randomize_offset:
                scale_factor = np.random.uniform(1.2, 1.3)
            offset = scale_factor * base_offset
            # for each primitive (the direction of travel)
            movement_dirs = np.stack([-offset, offset, -offset * np.array([1, 0]), offset, np.array([0, 1])])[
                allowed_policies]
            # against the left wall
            moving_against_the_wall = np.zeros(5).astype(bool)[allowed_policies]
            grounded = bbox[3] <= block_bbox[2] and block_bbox[3] <= 20
            # if block is too far left, or we are wedged between on the left
            if block_bbox[0] <= 5 or \
                    (bbox[0] <= 5 and block_bbox[1] <= 15 and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] < 0)
            # against the right wall, or we are wedged
            if env.grid_size[0] - block_bbox[1] <= 5 or \
                    (env.grid_size[0] - 5 <= bbox[1] and env.grid_size[0] - 15 <= block_bbox[0] and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] > 0)

            # policies not against the right wall or against the left wall
            allowed_policies = np.array(allowed_policies)[~moving_against_the_wall]
            # print("before select", allowed_policies)
            # print(allowed_policies, prefer_idxs)
            if len(allowed_policies) == 0 or (
                    len(prefer_idxs) > 0 and len(set(allowed_policies).intersection(prefer_idxs)) == 0):
                # end the execution if we get trapped
                policy_idx = None
                pps = d()
            else:
                # print(allowed_policies, moving_against_the_wall)
                p = np.ones(len(allowed_policies))
                if oversample_tip and 2 in allowed_policies:
                    # print(np.argmax(allowed_policies == 2))
                    p[np.argmax(allowed_policies == 2)] = 8.  # 8x likely to sample tipping

                for idx in set(allowed_policies).intersection(prefer_idxs):
                    p[np.argmax(allowed_policies == idx)] = 100.  # effectively guarantee these idxs.

                p = p / p.sum()
                policy_idx = np.random.choice(allowed_policies, p=p)
                # upward component is randomized for retreat
                retreat_dir = offset + np.array([0., np.random.uniform(0, abs(offset[0]))])
                pps = d(
                    block_idx=block_idx,
                    offset=offset,
                    kp_vel=[1.5, 5.],
                    speed_scale=np.random.uniform(1, 1.5),
                    retreat_steps=np.random.randint(min_retreat, max_retreat),
                    retreat_direction=retreat_dir,
                    timeout=80,
                )

                if policy_idx == 0:
                    # push
                    pps.push_steps = np.random.randint(5, 15)
                elif policy_idx == 1:
                    # pull
                    pps.pull_steps = np.random.randint(5, 15)
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0
                    if min_retreat == 0 and max_retreat == 1:  # if no retreat, do not wait.
                        pps.wait_steps = 0
                    if offset[1] > 5:
                        retreat_dir[0] += 2 * np.random.uniform(-np.max(np.abs(retreat_dir)), np.max(np.abs(retreat_dir)))
                elif policy_idx == 2:
                    # tipping
                    # get rid of the offset component of the retreat dir
                    # offset is half of the y block ( push from the top )
                    new_base_offset = np.array([base_offset[0], (block_bbox[2] - block_bbox[3]) / 2.])
                    if randomize_offset:
                        scale_factor = np.random.uniform(1.05, 1.3)
                    pps.offset = new_base_offset * scale_factor
                    pps.retreat_direction[1] -= offset[1]
                    pps.push_steps = np.random.randint(10, 15)
                    pps.bbox = block_bbox
                elif policy_idx == 3:
                    if no_lift_rot:
                        raise NotImplementedError
                    # rotating
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    pps.down_distance_frac = np.random.uniform(0.5, 0.75)
                    pps.down_speed_scale = np.random.uniform(0.8, 1.)
                    pps.bbox = block_bbox
                    pps.rotate_left = np.random.choice([True, False])
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0
                elif policy_idx == 4:
                    # rotating from the side
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = 0.04  # needs to be quite accurate
                    pps.randomize_offset = True  # this means offset might not be used exactly.
                    pps.bbox = block_bbox
                    pps.up_margin = np.random.uniform(40, 60)  # margin to move above max(block height, width)
                    # direction to pull (0 is back) is relative to which side we are on
                    pps.pull_tangent = np.sign(offset[0])
                else:
                    raise NotImplementedError

                memory.reset_count += 1
        else:
            # done with max iters
            policy_idx = None
            pps = d()
        return policy_idx, pps

    return memory_policy_params_fn


def get_rotate_only_memory_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                 min_retreat, max_retreat, random_side=True, randomize_offset=False,
                                                 oversample_tip=False, no_lift_rot=False, prefer_idxs=()):
    def memory_policy_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):
        # sides we can get to in a straight path
        valid_targets = get_valid_paths_from_bbox(env, obs.leaf_apply(lambda arr: arr[:, 0]))

        # pick a side that is close
        p, block_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions'])
        p = to_numpy(p, check=True)
        block_ps = to_numpy(block_ps, check=True)

        if no_lift_rot:
            new_valid_targets = []
            for b_idx, vt in enumerate(valid_targets):
                bbox = (obs >> "block_bounding_boxes")[0, 0, b_idx]
                x_center = (bbox[0] + bbox[1]) / 2
                new_valid_targets.append(vt[np.abs(vt[:, 0] - x_center) > 3])
            valid_targets = new_valid_targets

        all_valid = np.concatenate(valid_targets, axis=0)

        if len(all_valid) == 0:
            return None, d()

        dists = np.linalg.norm(p[None] - all_valid, axis=-1)

        # PICK THE PATH TO TAKE
        if random_side:
            path_idx = np.random.choice(len(dists))
        else:
            path_idx = np.argmin(dists)

        vt = all_valid[path_idx]
        # block idx is one minus the first to be greater
        block_idx = np.argmax(np.cumsum([len(v) for v in valid_targets]) > path_idx) - 1
        assert block_idx < len(block_ps)
        block_bbox = (obs >> "block_bounding_boxes")[0, 0, block_idx]
        bbox = (obs >> "bounding_box")[0, 0]

        # if vt[1] > block_bbox[2]:  # TOP
        #     print("pull")
        #     vt[0] += np.random.randn() * env.block_size[0] / 8  # x axis noise
        # else:
        #     print("push")
        #     vt[1] += np.random.randn() * env.block_size[1] / 8  # y axis noise

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        allowed_policies = [2, 3, 4] if not no_lift_rot else [2, 4]
        above_top = vt[1] > block_bbox[2]
        below_bottom = vt[1] < block_bbox[3]
        on_ground = block_bbox[3] < 20  # bottom is almost on the ground
        # 30% taller than wide is the only way we can tip
        tippable = 1.3 * (block_bbox[1] - block_bbox[0]) < block_bbox[2] - block_bbox[3]

        # tip from side
        if not on_ground or above_top or not tippable:
            allowed_policies.remove(2)

        # rotate from ground, starting above
        if not no_lift_rot and (not on_ground or not above_top):
            allowed_policies.remove(3)

        # rotate from ground, starting on the side
        if not on_ground or above_top:
            allowed_policies.remove(4)

        # print(above_top, on_ground, tippable, allowed_policies)
        if memory.reset_count < memory.max_iters:
            scale_factor = 1.2
            base_offset = (vt - block_ps[block_idx])
            if randomize_offset:
                scale_factor = np.random.uniform(1.2, 1.3)
            offset = scale_factor * base_offset
            # for each primitive (the direction of travel)
            movement_dirs = np.stack([-offset, offset, -offset * np.array([1, 0]), offset, np.array([0, 1])])[
                allowed_policies]
            # against the left wall
            moving_against_the_wall = np.zeros(5).astype(bool)[allowed_policies]
            grounded = bbox[3] <= block_bbox[2] and block_bbox[3] <= 20
            # if block is too far left, or we are wedged between on the left
            if block_bbox[0] <= 5 or \
                    (bbox[0] <= 5 and block_bbox[1] <= 15 and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] < 0)
            # against the right wall, or we are wedged
            if env.grid_size[0] - block_bbox[1] <= 5 or \
                    (env.grid_size[0] - 5 <= bbox[1] and env.grid_size[0] - 15 <= block_bbox[0] and grounded):
                moving_against_the_wall = np.logical_or(moving_against_the_wall, movement_dirs[:, 0] > 0)

            # policies not against the right wall or against the left wall
            allowed_policies = np.array(allowed_policies)[~moving_against_the_wall]
            # print("before select", allowed_policies)
            # print(allowed_policies, prefer_idxs)
            if len(allowed_policies) == 0 or (
                    len(prefer_idxs) > 0 and len(set(allowed_policies).intersection(prefer_idxs)) == 0):
                # end the execution if we get trapped
                policy_idx = None
                pps = d()
            else:
                # print(allowed_policies, moving_against_the_wall)
                p = np.ones(len(allowed_policies))
                if oversample_tip and 2 in allowed_policies:
                    # print(np.argmax(allowed_policies == 2))
                    p[np.argmax(allowed_policies == 2)] = 8.  # 8x likely to sample tipping, since it occurs less frequently

                for idx in set(allowed_policies).intersection(prefer_idxs):
                    p[np.argmax(allowed_policies == idx)] = 100.  # effectively guarantee these idxs.

                p = p / p.sum()
                policy_idx = np.random.choice(allowed_policies, p=p)
                # upward component is randomized for retreat
                retreat_dir = offset + np.array([0., np.random.uniform(0, abs(offset[0]))])
                pps = d(
                    block_idx=block_idx,
                    offset=offset,
                    kp_vel=[1.5, 5.],
                    speed_scale=np.random.uniform(1, 1.5),
                    retreat_steps=np.random.randint(min_retreat, max_retreat),
                    retreat_direction=retreat_dir,
                    timeout=100,
                )
                if policy_idx == 2:
                    # tipping
                    # get rid of the offset component of the retreat dir
                    # offset is half of the y block ( push from the top )
                    new_base_offset = np.array([base_offset[0], (block_bbox[2] - block_bbox[3]) / 2.])
                    if randomize_offset:
                        scale_factor = np.random.uniform(1.05, 1.3)
                    pps.offset = new_base_offset * scale_factor
                    pps.retreat_direction[1] -= offset[1]
                    pps.push_steps = np.random.randint(10, 15)
                    pps.bbox = block_bbox
                elif policy_idx == 3:
                    if no_lift_rot:
                        raise NotImplementedError
                    # rotating
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = np.random.uniform(0.03, 0.1)
                    pps.down_distance_frac = np.random.uniform(0.5, 0.75)
                    pps.down_speed_scale = np.random.uniform(0.8, 1.)
                    pps.bbox = block_bbox
                    pps.rotate_left = np.random.choice([True, False])
                    # direction to pull (0 is back)
                    pps.pull_tangent = 0
                elif policy_idx == 4:
                    # rotating from the side
                    pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
                    pps.grab_binary = True  # 0/1 actions, simpler
                    pps.tolerance = 0.04  # needs to be quite accurate
                    pps.randomize_offset = True  # this means offset might not be used exactly.
                    pps.bbox = block_bbox
                    pps.up_margin = np.random.uniform(40, 60)  # margin to move above max(block height, width)
                    # direction to pull (0 is back) is relative to which side we are on
                    pps.pull_tangent = np.sign(offset[0])
                else:
                    raise NotImplementedError

                memory.reset_count += 1
        else:
            # done with max iters
            policy_idx = None
            pps = d()
        return policy_idx, pps

    return memory_policy_params_fn


if __name__ == "__main__":
    from sbrl.envs.block2d.block_env_2d import get_block2d_example_params
    from sbrl.envs.block2d.stack_block_env_2d import StackBlockEnv2D, get_stack_block2d_example_params
    from sbrl.envs.param_spec import ParamEnvSpec

    use_meta = True
    rotate = True
    no_push_pull = True

    _, env_spec_params = get_block2d_example_params()
    env_params = get_stack_block2d_example_params()
    # env_params.block_size = (30, 30)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    block_env = StackBlockEnv2D(env_params, env_spec)
    # env presets
    presets = d()

    model = Model(d(ignore_inputs=True), env_spec, None)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    obs, goal = block_env.user_input_reset(1)  # trolling with a fake UI

    if use_meta:
        if rotate and not no_push_pull:
            get_next_policy_params = get_push_pull_lift_rotate_memory_meta_policy_params_fn(3, 6, 5, 10,
                                                                                            random_side=True,
                                                                                            randomize_offset=True)
            all_policies = [d(cls=PushPrimitive, params=d()), d(cls=PullPrimitive, params=d()),
                            d(cls=TipBlockPrimitive, params=d()), d(cls=RotateBlockPrimitive, params=d())]
        elif rotate and no_push_pull:
            get_next_policy_params = get_rotate_only_memory_meta_policy_params_fn(3, 6, 5, 10, random_side=True,
                                                                                  randomize_offset=True,
                                                                                  oversample_tip=False)
            all_policies = [d(cls=PushPrimitive, params=d()), d(cls=PullPrimitive, params=d()),
                            d(cls=TipBlockPrimitive, params=d()), d(cls=RotateBlockPrimitive, params=d()),
                            d(cls=SideRotateBlockPrimitive, params=d())]
        else:
            get_next_policy_params = get_push_pull_lift_memory_meta_policy_params_fn(3, 6, 5, 10, random_side=True)
            all_policies = [d(cls=PushPrimitive, params=d()), d(cls=PullPrimitive, params=d())]
        policy = MetaPolicy(d(all_policies=all_policies,
                              next_param_fn=get_next_policy_params), env_spec, env=block_env)

        policy.reset_policy(next_obs=obs, next_goal=goal)

    else:
        if rotate:
            ## ROTATE
            CLS = RotateBlockPrimitive


            def get_next_policy_params(obs, goal):
                valid_targets = get_valid_paths_from_bbox(block_env, obs)
                valid = np.concatenate(valid_targets)
                # get the ones where y offset is positive
                valid = valid[valid[:, 1] - (obs >> 'block_positions')[0, 0, 1] > 5]
                if len(valid) == 0:
                    return None
                offset = valid[np.random.randint(len(valid))] - (obs >> 'block_positions')[0, 0]
                bbox = (obs >> "block_bounding_boxes")[0, 0]  # l, r, t, b
                # # bbox = (bbox[1] - bbox[0], bbox[2] - bbox[3])
                # bheight = bbox[2] - bbox[3]
                # offset = 1.2 * offset + np.array([0, np.random.uniform(bheight / 2, bheight / 1.5)])
                return d(bbox=bbox, rotate_left=np.random.choice([True, False]),
                         down_distance_frac=np.random.uniform(0.5, 0.75), block_idx=0,
                         offset=1.1 * offset, speed_scale=np.random.uniform(1., 1.5), timeout=100)
        else:
            ## TIP
            CLS = TipBlockPrimitive


            def get_next_policy_params(obs, goal):
                valid_targets = get_valid_paths_from_bbox(block_env, obs)
                valid = np.concatenate(valid_targets)
                # get the ones where x dimension is nonzero
                valid = valid[np.abs(valid[:, 0] - (obs >> 'block_positions')[0, 0, 0]) > 5]
                if len(valid) == 0:
                    return None
                offset = valid[np.random.randint(len(valid))] - (obs >> 'block_positions')[0, 0]
                bbox = (obs >> "block_bounding_boxes")[0, 0]  # l, r, t, b
                # bbox = (bbox[1] - bbox[0], bbox[2] - bbox[3])
                bheight = bbox[2] - bbox[3]
                offset = 1.2 * offset + np.array([0, np.random.uniform(bheight / 2, bheight / 1.5)])
                return d(bbox=bbox, block_idx=0, offset=offset, speed_scale=2., timeout=100)

        policy = CLS(d(), env_spec, env=block_env)
        pparams = get_next_policy_params(obs, goal)
        while pparams is None:
            obs, goal = block_env.reset(presets)
            pparams = get_next_policy_params(obs, goal)
        policy.reset_policy(**(pparams & d(next_obs=obs, next_goal=goal)).as_dict())

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

            if use_meta:
                policy.reset_policy(next_obs=obs, next_goal=goal)
            else:
                pparams = get_next_policy_params(obs, goal)
                while pparams is None:
                    obs, goal = block_env.reset(presets)
                    pparams = get_next_policy_params(obs, goal)
                policy.reset_policy(**(pparams & d(next_obs=obs, next_goal=goal)).as_dict())

            i += 1
