import time

import numba
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from sbrl.utils import transform_utils as TU


@numba.jit(nopython=True, cache=True)
def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    # J M^-1 J^T
    lambda_full_inv = np.dot(
        np.dot(J_full, mass_matrix_inv),
        J_full.transpose())

    # (J M^-1 J^T)^-1
    # lambda_full = np.linalg.inv(lambda_full_inv)

    # Jx M^-1 Jx^T
    lambda_pos_inv = np.dot(
        np.dot(J_pos, mass_matrix_inv),
        J_pos.transpose())

    # Jr M^-1 Jr^T
    lambda_ori_inv = np.dot(
        np.dot(J_ori, mass_matrix_inv),
        J_ori.transpose())

    svd_u, svd_s, svd_v = np.linalg.svd(lambda_full_inv)
    singularity_threshold = 0.05
    svd_s_inv = np.array([0 if x < singularity_threshold else 1. / x for x in svd_s])
    lambda_full = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

    # take the inverse, but zero out elements in cases of a singularity
    svd_u, svd_s, svd_v = np.linalg.svd(lambda_pos_inv)
    singularity_threshold = 0.002
    svd_s_inv = np.array([0 if x < singularity_threshold else 1. / x for x in svd_s])
    lambda_pos = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

    svd_u, svd_s, svd_v = np.linalg.svd(lambda_ori_inv)
    svd_s_inv = np.array([0 if x < singularity_threshold else 1. / x for x in svd_s])
    lambda_ori = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

    # nullspace
    Jbar = np.dot(mass_matrix_inv, J_full.transpose()).dot(lambda_full)
    nullspace_matrix = np.eye(J_full.shape[-1], J_full.shape[-1]) - np.dot(Jbar, J_full)

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


@numba.jit(nopython=True, cache=True)
def nullspace_torques(mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel, joint_kp=10):
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
     subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
     controller objective may attempt to be maintained using these nullspace torques.
    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
     positions @initial_joint with zero velocity using proportinal gain @joint_kp
    Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
     timestep
    """

    # kv calculated below corresponds to critical damping
    joint_kv = np.sqrt(joint_kp) * 2

    # calculate desired torques based on gains and error
    pose_torques = np.dot(mass_matrix, (joint_kp * (
            initial_joint - joint_pos) - joint_kv * joint_vel))

    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = np.dot(nullspace_matrix.transpose(), pose_torques)
    return nullspace_torques


@numba.jit(nopython=True, cache=True)
def cross_product(vec1, vec2):
    mat = np.array(([0, -vec1[2], vec1[1]],
                    [vec1[2], 0, -vec1[0]],
                    [-vec1[1], vec1[0], 0]))
    return np.dot(mat, vec2)


def orientation_error(desired, current):
    """
    Optimized function to determine orientation error from matrices
    """

    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    return 0.5 * (cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3))


# def orientation_error(desired, current):
#     q_ed = TU.mat2quat(current.T @ desired)
#     return current @ q_ed[:3]  # vector part of


def batch_orientation_error(desired, current):
    """
    Optimized function to determine orientation error from matrices
    """

    rc1 = current[..., 0:3, 0]
    rc2 = current[..., 0:3, 1]
    rc3 = current[..., 0:3, 2]
    rd1 = desired[..., 0:3, 0]
    rd2 = desired[..., 0:3, 1]
    rd3 = desired[..., 0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


def batch_orientation_eul_error(desired, current):
    # print(desired.shape, current.shape)
    desired_rot = R.from_euler("xyz", desired.reshape(-1, 3)).as_matrix()
    current_rot = R.from_euler("xyz", current.reshape(-1, 3)).as_matrix()
    return batch_orientation_error(desired_rot, current_rot).reshape(desired.shape)


def batch_orientation_eul_add(a, b):
    ar = R.from_euler("xyz", a.reshape(-1, 3))
    br = R.from_euler("xyz", b.reshape(-1, 3))
    return (br * ar).as_euler("xyz").reshape(a.shape)


def torch_batch_orientation_error(desired, current):
    """
    Optimized function to determine orientation error from matrices
    """

    rc1 = current[..., 0:3, 0]
    rc2 = current[..., 0:3, 1]
    rc3 = current[..., 0:3, 2]
    rd1 = desired[..., 0:3, 0]
    rd2 = desired[..., 0:3, 1]
    rd3 = desired[..., 0:3, 2]

    error = 0.5 * (torch.cross(rc1, rd1) + torch.cross(rc2, rd2) + torch.cross(rc3, rd3))
    return error


def pose_diff_fn(a: np.ndarray, b: np.ndarray):
    pos_diff = a[:3] - b[:3]
    orn_diff = orientation_error(R.from_euler("xyz", a[3:]).as_matrix(), R.from_euler("xyz", b[3:]).as_matrix())
    return np.concatenate([pos_diff, orn_diff])


def batch_pose_diff_fn(a: np.ndarray, b: np.ndarray):
    pos_diff = a[..., :3] - b[..., :3]
    orn_diff = batch_orientation_error(R.from_euler("xyz", a[..., 3:]).as_matrix(), R.from_euler("xyz", b[..., 3:]).as_matrix())
    return np.concatenate([pos_diff, orn_diff], axis=-1)


def get_dead_band_difference_fn(band_low, band_high, diff_fn=lambda a, b: a - b, smooth=False):
    assert np.all(band_low <= band_high)

    def smooth_fn(a, b):
        out = diff_fn(a, b)
        return np.where(out > band_high, out - band_high, 0) + np.where(out < band_low, out - band_low, 0)

    def fn(a, b):
        out = diff_fn(a, b)
        return np.where((band_low <= out) & (out <= band_high), 0., out)

    return smooth_fn if smooth else fn


# @numba.jit(nopython=True, cache=True)
def set_goal_position(delta,
                      current_position,
                      position_limit=None,
                      set_pos=None):
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @delta and @current_position must be specified if a relative goal is requested, else @set_pos must be
    specified to define a global goal position
    """
    n = len(current_position)
    if set_pos is not None:
        goal_position = set_pos
    else:
        goal_position = current_position + delta

    if position_limit is not None:
        if position_limit.shape != (2, n):
            raise ValueError("Position limit should be shaped (2,{}) "
                             "but is instead: {}".format(n, position_limit.shape))

        # Clip goal position
        goal_position = np.clip(goal_position, position_limit[0], position_limit[1])

    return goal_position


# @numba.jit(nopython=True, cache=True)
def set_goal_orientation(delta,
                         current_orientation,
                         orientation_limit=None,
                         set_ori=None,
                         axis_angle=True):
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation
    If @axis_angle is set to True, then this assumes the input in axis angle form, that is,
        a scaled axis angle 3-array [ax, ay, az]
    """
    # directly set orientation
    if set_ori is not None:
        goal_orientation = set_ori

    # otherwise use delta to set goal orientation
    else:
        if axis_angle:
            # convert axis-angle value to rotation matrix
            quat_error = R.from_rotvec(delta).as_quat()
            rotation_mat_error = R.from_quat(quat_error).as_matrix()
        else:
            # convert euler value to rotation matrix
            rotation_mat_error = R.from_euler("xyz", -delta).as_matrix()
        goal_orientation = np.dot(rotation_mat_error.T, current_orientation)

    # check for orientation limits
    if np.array(orientation_limit).any():
        if orientation_limit.shape != (2, 3):
            raise ValueError("Orientation limit should be shaped (2,3) "
                             "but is instead: {}".format(orientation_limit.shape))

        # Convert to euler angles for clipping
        euler = R.from_matrix(goal_orientation).as_euler("xyz")

        # Clip euler angles according to specified limits
        limited = False
        for idx in range(3):
            if orientation_limit[0][idx] < orientation_limit[1][idx]:  # Normal angle sector meaning
                if orientation_limit[0][idx] < euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
            else:  # Inverted angle sector meaning
                if (orientation_limit[0][idx] < euler[idx]
                        or euler[idx] < orientation_limit[1][idx]):
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
        if limited:
            goal_orientation = R.from_euler("xyz", np.array([euler[1], euler[0], euler[2]])).as_matrix()
    return goal_orientation


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


if __name__ == '__main__':
    c = [0, 0, 0]
    d = [-np.pi, 0, -np.pi/2]

    print(orientation_error(TU.euler2mat(d), TU.euler2mat(c)))
    print(TU.quat2euler_ext(TU.quat_difference(TU.euler2quat_ext(d), TU.euler2quat_ext(c))))
