"""
"""
from argparse import ArgumentParser

from sbrl.utils.python_utils import AttrDict

# declares this group's parser, and defines any sub groups we need

REFERENCE_MARKER_SIZE = 0.11  # reference
# REFERENCE_MARKER_SIZE = 0.036  # reference
OBJECT_MARKER_SIZE = 0.036
BLOCK_SIZE = 0.05
referenceID = 0
blockSideIDs = list(range(1, 7))
ARUCO_TYPE = "DICT_4X4_50"


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('-e', '--extrinsics', type=str, default='extrinsics.npz',
                        help="Calibration output (or input if exists)")
    parser.add_argument('--imgs', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-f', '--force_reload', action='store_true',
                        help="Forces calibration to reload")
    parser.add_argument('--single_cam', type=int, default=None, help="specify the index 1-3 of the single camera to load (default does all)")
    return parser


def process_params(group_name, common_params):
    from sbrl.envs.block_real.multi_cam_block_sensor import MultiCameraBlockSensor
    from sbrl.envs.sensor.camera import RSDepthCamera, RSSeries
    prms = common_params >> group_name

    def get_camera_sr(device_num):
        device_num = int(device_num)
        if device_num == 1:
            return "619205002599"
        elif device_num == 2:
            return "620201001527"
        elif device_num == 3:
            return "617205003295"

    def get_cam(num):
        return AttrDict(
            cls=RSDepthCamera,
            params=AttrDict(
                series=RSSeries.sr300,
                serial=get_camera_sr(num),
                do_depth=False,
            )
        )

    if prms >> "single_cam" is not None:
        camId = prms >> "single_cam"
        all_cams = AttrDict.from_dict({f'camera{camId}': get_cam(camId)})
    else:
        all_cams = AttrDict(
            camera1=get_cam(1),
            camera2=get_cam(2),
            camera3=get_cam(3),
        )

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=MultiCameraBlockSensor,
        params=AttrDict(
            camera_sensors=all_cams,
            visualize=prms >> "visualize",
            extrinsics=prms >> "extrinsics",
            reload_extrinsics=prms >> "force_reload",
            aruco_type=ARUCO_TYPE,
            get_images=prms >> "imgs",  # puts them in obs

            reference_tag_size=REFERENCE_MARKER_SIZE,
            reference_tag=referenceID,
            block_side_tag_size=OBJECT_MARKER_SIZE,
            block_size=BLOCK_SIZE,
            block_side_tags=blockSideIDs,
        )
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
