import math

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

# returns a quaternion
from sbrl.utils.loss_utils import write_avg_per_last_dim
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import concatenate, combine_after_dim


def round_to_n(x, n=1):
    # round scalar to n significant figure(s)
    return round(x, -int(math.floor(math.log10(abs(x))) + (n-1)))

def angle_axis_apply_to_quat(axis, angle, quaternion):
    quat2 = p.getQuaternionFromAxisAngle(axis, angle)
    position, orientation = p.multiplyTransforms(positionA=[0, 0, 0], orientationA=quaternion, positionB=[0, 0, 0], orientationB=quat2)
    return orientation


def matrix_to_quat(m):
    qw = np.sqrt(1 + np.diag(m)) / 2.
    return np.array([
        (m[2, 1] - m[1, 2]) / (4 * qw),
        (m[0, 2] - m[2, 0]) / (4 * qw),
        (m[1, 0] - m[0, 1]) / (4 * qw),
        qw
    ])


# wrap around 2pi
def get_absolute_angle_difference(angles1, angles2):
    if isinstance(angles1, np.ndarray):
        # return np.absolute((np.absolute(angles1 - angles2) + np.pi) % (2 * np.pi) - np.pi)
        angles1 = angles1 % (2 * np.pi)
        angles2 = angles2 % (2 * np.pi)
        greater = np.where(angles1 > angles2, angles1, angles2)
        smaller = np.where(angles1 <= angles2, angles1, angles2)
        # print(greater, smaller)

        between = greater - smaller  # within bounds of [0, 2pi]
        wrapped = 2*np.pi - greater + smaller  # wrap around 2pi
        return np.where(between < wrapped, between, wrapped)  # get the smaller (positive) angle
    else:
        raise NotImplementedError
        # return (((angles1 - angles2).abs() + np.pi) % (2 * np.pi) - np.pi).abs()

def get_quaternion_angle_difference(quat1, quat2):
    return np.arccos(np.asarray(quat1).dot(np.asarray(quat2)) / (np.linalg.norm(quat1) * np.linalg.norm(quat2)))

def get_mse(diff, axis=-1, sqrt=True):  # L2
    return (diff ** 2).mean(axis) ** 0.5 if sqrt else (diff ** 2).mean(axis)

def get_mae(diff, axis=-1):  # L1
    if isinstance(diff, np.ndarray):
        diff = np.absolute(diff)
    else:
        diff = diff.abs()
    return diff.mean(axis)

def slerp_quaternions(q1, q2, t):
    # half angle: cos(th/2) = w1*w2 + x1*x2 + y1*y2 + z1*z2
    theta = 2. * np.arccos((q1 * q1).sum())
    return (q1 * np.sin((1.-t) * theta) + q2 * np.sin(t * theta))  /  np.sin(theta)

def angular_vel_from(q1, q2):
    pass

def convert_rpt(roll, phi, theta, bias=np.array([0, -np.pi/2, 0])):
    # returns quat and eul
    rot1 = R.from_rotvec([0, 0, bias[0] + roll])
    rot2 = R.from_rotvec([bias[1] + phi, 0, 0])
    rot3 = R.from_rotvec([0, 0, bias[2] + theta])
    chained = rot3 * rot2 * rot1
    return chained.as_quat(), chained.as_euler("xyz")

def convert_eul_to_rpt(eul, bias=np.array([0, -np.pi/2, 0])):
    # returns quat and eul
    chained = R.from_euler("xyz", eul)
    rpt = chained.as_euler("zxz") - bias
    return rpt

def concat_then_err(err_fn, pred_dc: AttrDict, true_dc: AttrDict, combine_dim, names=None,
                    i=0, writer=None, writer_prefix=""):
    if names is not None:
        pred_dc = pred_dc > names
    else:
        names = pred_dc.list_leaf_keys()
    true_dc = true_dc > names

    cat_pred = concatenate(pred_dc.leaf_apply(lambda arr: combine_after_dim(arr, combine_dim)), names, -1)
    cat_true = concatenate(true_dc.leaf_apply(lambda arr: combine_after_dim(arr, combine_dim)), names, -1)

    # error per element.
    err = err_fn(cat_true, cat_pred)

    if writer is not None:
        write_avg_per_last_dim(err, i=i, writer=writer, writer_prefix=writer_prefix)

    # mean across last dim
    return err.mean(-1)


# BETTER, use this
def convert_rpt_to_quat_eul(rpt, bias=np.array([0, -np.pi/2, 0])):
    rot = R.from_euler("zxz", rpt + bias)
    return rot.as_quat(), rot.as_euler("xyz")


def convert_quat_to_rpt(quat, bias=np.array([0, -np.pi/2, 0])):
    # returns r,p,t
    chained = R.from_quat(quat)
    rpt = chained.as_euler("zxz") - bias
    return rpt


def limit_norm(vector: np.ndarray, limit, axis=-1):
    norms = np.linalg.norm(vector, axis=axis, keepdims=True)
    scaling = limit / (norms + 1e-11)

    return vector * np.minimum(scaling, 1.)


def line_circle_intersection(p1, p2, r, circle_center=np.zeros(2)):
    # line from p1 -> p2 intersections w/ circle at either 0, 1, or 2 points

    # recenter around circle center
    p1 = p1 - circle_center
    p2 = p2 - circle_center

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dr2 = dx**2 + dy**2
    D = p1[0]*p2[1] - p1[1]*p2[0]

    incidence = r ** 2 * dr2 - D ** 2
    if incidence < 0:
        return []
    elif incidence == 0:
        x = D * dy / dr2
        y = -D * dx / dr2
        return [np.array([x,y]) + circle_center]
    else:
        sgny = -1 if dy < 0 else 1
        x1 = (D * dy + sgny * dx * np.sqrt(incidence)) / dr2
        x2 = (D * dy - sgny * dx * np.sqrt(incidence)) / dr2
        y1 = (-D * dx + abs(dy) * np.sqrt(incidence)) / dr2
        y2 = (-D * dx - abs(dy) * np.sqrt(incidence)) / dr2
        return [np.array([x1, y1]) + circle_center,
                np.array([x2, y2]) + circle_center]


if __name__ == '__main__':
    quat = np.array([0.707, 0, 0.707, 0.])
    rpt2 = convert_quat_to_rpt(quat)
    quat2, eul = convert_rpt(*rpt2)

    rpt = np.array([0.1, 0.5, -1.2])
    quat3, eul3 = convert_rpt(*rpt)
    rpt3 = convert_quat_to_rpt(quat3)
    print(quat, quat2)
    print(rpt, rpt3)

    print(get_quaternion_angle_difference(quat,quat2))
    print(get_quaternion_angle_difference(convert_rpt(*rpt)[0],convert_rpt(*rpt3)[0]))
    # print(get_absolute_angle_difference(rpt, rpt2))

    quat, eul = convert_rpt(0,0,0, bias=np.array([-np.pi/2, -np.pi/2, 0]))

    print(quat, " vs:", [0.5,0.5,0.5,-0.5])

    ulim = limit_norm(np.random.uniform(0, 100., (1000,3)), 10.)
    assert np.all(np.linalg.norm(ulim, axis=-1) <= 10.)
