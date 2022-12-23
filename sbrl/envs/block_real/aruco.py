import time

import cv2
# define names of each possible ArUco tag OpenCV supports
import numpy as np
from scipy.spatial.transform import Rotation

from sbrl.experiments import logger
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def get_aruco_settings(aruco_type):
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    return arucoDict, arucoParams


def get_tag_pose(camera_sensor, K, D, tag_id, tag_size, aruco_dict, aruco_params, n_frames=1, dt=0.1):

    all_rmat=[]
    all_tvec=[]
    for n in range(n_frames):
        obs = camera_sensor.read_state()
        # Convert images to numpy arrays
        color_image = obs >> "rgb"
        depth_image = obs >> "depth"

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


        (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_params)
        cv2.aruco.drawDetectedMarkers(color_image, corners)  # Draw A square around the markers

        assert tag_id in ids.flatten(), f"Missing reference tag: {tag_id}"

        idx = list(ids.flatten()).index(tag_id)

        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[idx], tag_size, K, D)
        rvec, _ = cv2.Rodrigues(rvec.reshape(3))  # to matrix form
        tvec = tvec.reshape(3)
        all_rmat.append(rvec)
        all_tvec.append(tvec)
        time.sleep(dt)


    if n_frames > 1:
        tvec = np.average(all_tvec, axis=0)
        rvec = Rotation.from_matrix(np.stack(all_rmat, axis=0)).mean().as_matrix()
        tvec_std = np.std(all_tvec, axis=0)
        logger.debug(f"Translation std: {tvec_std}")
    else:
        tvec = all_tvec[0]
        rvec = all_rmat[0]

    cv2.drawFrameAxes(color_image, K, D, rvec, tvec, tag_size / 2)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    return rvec, tvec, images, color_image, depth_colormap


def get_multi_tag_poses(camera_sensor, K, D, tag_ids, tag_size, aruco_dict, aruco_params):

    obs = camera_sensor.read_state()
    # Convert images to numpy arrays
    color_image = obs >> "rgb"
    depth_image = obs >> "depth"

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_params)

    filtered_corners = []
    all_frames = dict()  # tag in camera frame
    if len(corners) > 0:
        for i, idx in enumerate(ids.flatten()):
            if idx in tag_ids:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], tag_size, K, D)
                cv2.drawFrameAxes(color_image, K, D, rvec, tvec, tag_size / 2)
                filtered_corners.append(corners[i])
                all_frames[idx] = CoordinateFrame(world_frame_3D, Rotation.from_matrix(cv2.Rodrigues(rvec.reshape(3))[0]).inv(), tvec.reshape(3))

        cv2.aruco.drawDetectedMarkers(color_image, filtered_corners)  # Draw A square around the markers

    return all_frames, color_image, depth_colormap
