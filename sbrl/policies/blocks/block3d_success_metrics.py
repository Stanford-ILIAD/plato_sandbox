import numpy as np
from scipy.spatial.transform import Rotation

from sbrl.policies.blocks.stack_block2d_success_metrics import get_l2_then_mean
from sbrl.utils.geometry_utils import np_quat_angle
from sbrl.utils.python_utils import AttrDict


def move_object_to_position_and_angle_reward(trajectory_obs, done_idxs, goal_obs, l2_tol=0.025, l2_angle_tol=np.deg2rad(10.), pos_weight=0.75):
    # traj: (sum(Hi) x ...).  done_idxs: (L), goal_obs: (L ...), for H_1 ... H_L
    # final_obs = trajectory_obs.leaf_apply(lambda arr: arr[done_idxs])

    ep_lens = np.diff([0] + list(done_idxs + 1))
    goal_obs_rep = goal_obs.leaf_copy()
    goal_obs_rep.objects.position = np.repeat(goal_obs >> "objects/position", ep_lens, axis=0)
    goal_obs_rep.objects.orientation_eul = np.repeat(goal_obs >> "objects/orientation_eul", ep_lens, axis=0)

    # mean over blocks, and norm over each (x,y)
    l2_all_dist = get_l2_then_mean(trajectory_obs, goal_obs_rep, 'objects/position')
    # quat_distance = get_l2_then_mean(trajectory_obs, goal_obs_rep, 'objects/orientation_eul', )
    first_quat = Rotation.from_euler('xyz', (trajectory_obs >> "objects/orientation_eul").reshape(-1, 3)).as_quat()
    goal_quat = Rotation.from_euler('xyz', (goal_obs_rep >> "objects/orientation_eul").reshape(-1, 3)).as_quat()
    l2_all_angle = np_quat_angle(first_quat, goal_quat)
    # l2_all_angle = get_l2_then_mean(trajectory_obs, goal_obs_rep, 'objects/orientation_eul', diff_fn=batch_orientation_eul_error)
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

def move_slider_to_position_reward(trajectory_obs, done_idxs, goal_obs, l2_tol=0.02, name="drawer"):
    # traj: (sum(Hi) x ...).  done_idxs: (L), goal_obs: (L ...), for H_1 ... H_L
    # final_obs = trajectory_obs.leaf_apply(lambda arr: arr[done_idxs])

    ep_lens = np.diff([0] + list(done_idxs + 1))
    goal_obs_rep = goal_obs.leaf_copy()
    goal_obs_rep[f"{name}/joint_position"] = np.repeat(goal_obs >> f"{name}/joint_position", ep_lens, axis=0)

    # mean over blocks, and norm over each (x,)
    l2_all_dist = get_l2_then_mean(trajectory_obs, goal_obs_rep, f'{name}/joint_position', keepdims=True)
    # print((l2_all_angle < l2_angle_tol).sum())
    rew = (l2_all_dist < l2_tol).astype(float)

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

# 3CM, 15 degrees
def get_move_object_to_position_and_angle_reward(l2_tol=0.025, l2_angle_tol=np.deg2rad(15.), pos_weight=0.75):
    return lambda tobs, didx, gobs: move_object_to_position_and_angle_reward(tobs, didx, gobs, l2_tol=l2_tol, l2_angle_tol=l2_angle_tol, pos_weight=pos_weight)

def get_button_press_success(button_idx):
    def button_press_success(trajectory_obs, done_idxs, goal_obs):
        bc = (trajectory_obs >> "buttons/closed")
        rew = bc.astype(float).reshape(bc.shape[0], 3)[:, button_idx]
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
    return button_press_success

map_pname_to_success = {
    # mostly position
    'pull_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'pull_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'pull_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'pull_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'push_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'top_rot_left': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'top_rot_right': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    'lift_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'lift_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'lift_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'lift_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'drawer_open': move_slider_to_position_reward,
    'drawer_close': move_slider_to_position_reward,
    'cabinet_top_rot_left': lambda *args, **kwargs: move_slider_to_position_reward(*args, l2_tol=np.deg2rad(10.), name="cabinet", **kwargs),
    'cabinet_top_rot_right': lambda *args, **kwargs: move_slider_to_position_reward(*args, l2_tol=np.deg2rad(10.), name="cabinet", **kwargs),
    'to_cabinet_pull_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'to_cabinet_pull_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'to_cabinet_pull_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'to_cabinet_pull_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75),
    'from_cabinet_pull_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),  # more tolerance for this one
    'from_cabinet_pull_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'from_cabinet_pull_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'from_cabinet_pull_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'to_drawer_pull_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),  # more tolerance for this one
    'to_drawer_pull_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'to_drawer_pull_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'to_drawer_pull_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'from_drawer_lift_left': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),  # more tolerance for this one
    'from_drawer_lift_right': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'from_drawer_lift_forward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),
    'from_drawer_lift_backward': get_move_object_to_position_and_angle_reward(pos_weight=0.75, l2_tol=0.05),

    'button_press_0': get_button_press_success(0),
    'button_press_1': get_button_press_success(1),
    'button_press_2': get_button_press_success(2),

    # # mostly angle
    # 'rotate_left': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    # 'rotate_right': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    # 'left_side_rotate': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    # 'right_side_rotate': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
    # 'tip': get_move_object_to_position_and_angle_reward(pos_weight=0.25),
}

