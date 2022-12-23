import numpy as np

from sbrl.utils.python_utils import AttrDict

circular_difference_fn = lambda a, b: (a - b + np.pi) % (2 * np.pi) - np.pi

def get_l2_then_mean(pred, true, key, diff_fn=lambda a, b: a - b, keepdims=False):
    return get_l2(pred, true, key, diff_fn, keepdims=keepdims).mean(-1)

def get_l2(pred, true, key, diff_fn=lambda a, b: a - b, keepdims=True):
    return ((diff_fn((pred >> key), (true >> key)) ** 2).sum(-1, keepdims=keepdims) ** 0.5)

def move_object_to_position_and_angle_reward(trajectory_obs, done_idxs, goal_obs, l2_tol=10., l2_angle_tol=np.deg2rad(10.), pos_weight=0.75):
    # traj: (sum(Hi) x ...).  done_idxs: (L), goal_obs: (L ...), for H_1 ... H_L
    # final_obs = trajectory_obs.leaf_apply(lambda arr: arr[done_idxs])

    ep_lens = np.diff([0] + list(done_idxs + 1))
    goal_obs_rep = goal_obs.leaf_copy()
    goal_obs_rep.block_positions = np.repeat(goal_obs >> "block_positions", ep_lens, axis=0)
    goal_obs_rep.block_angles = np.repeat(goal_obs >> "block_angles", ep_lens, axis=0)

    # mean over blocks, and norm over each (x,y)
    l2_all_dist = get_l2_then_mean(trajectory_obs, goal_obs_rep, 'block_positions')
    l2_all_angle = get_l2_then_mean(trajectory_obs, goal_obs_rep, 'block_angles', diff_fn=circular_difference_fn, keepdims=True)
    # print((l2_all_angle < l2_angle_tol).sum())
    rew = pos_weight * (l2_all_dist < l2_tol).astype(float) + (1-pos_weight) * (l2_all_angle < l2_angle_tol).astype(float)

    rew_by_eps = np.split(rew, (done_idxs + 1)[:-1])
    best = np.asarray([ep_rew.max() for ep_rew in rew_by_eps])
    initial = np.asarray([ep_rew[0] for ep_rew in rew_by_eps])
    final = np.asarray([ep_rew[-1] for ep_rew in rew_by_eps])
    # (L,)
    return AttrDict(
        best=best,
        initial=initial,
        final=final,
    )


def get_move_object_to_position_and_angle_reward(l2_tol=15., l2_angle_tol=np.deg2rad(15.), pos_weight=0.75):
    return lambda tobs, didx, gobs: move_object_to_position_and_angle_reward(tobs, didx, gobs, l2_tol=l2_tol, l2_angle_tol=l2_angle_tol, pos_weight=pos_weight)


map_pname_to_success = {
    # mostly position
    'pull_back_move_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'pull_back_move_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'pull_back_move_up': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    # mostly angle
    'rotate_left': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'rotate_right': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'left_side_rotate': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'right_side_rotate': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'tip': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
}
