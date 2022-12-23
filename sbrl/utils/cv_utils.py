"""
ROS-safe import
"""

import sys
DIST = "kinetic"
ROS_PYPATH = "/opt/ros/%s/lib/python2.7/dist-packages" % DIST


def pre_cv2_import() -> bool:
    if ROS_PYPATH in sys.path:
        sys.path.remove(ROS_PYPATH)
        return True
    return False


def post_cv2_import(add):
    if add:
        sys.path.append(ROS_PYPATH)


res = pre_cv2_import()
import cv2
post_cv2_import(res)

# now you can import cv2 from this util file!
version = cv2.version