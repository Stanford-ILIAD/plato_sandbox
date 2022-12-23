import enum
import json
import threading
import time

import numpy as np

from sbrl.envs.sensor.sensors import Sensor
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_with_default, get_required


class USBCamera(Sensor):
    def _init_params_to_attrs(self, params: AttrDict):
        self.cv_cam_id = params.get("cv_cam_id", 0)
        self.W = params.get("img_width", 640)
        self.H = params.get("img_height", 480)
        assert self.cv_cam_id != -1, "USB camera cannot be created with cam_id = -1"

    def _init_setup(self, **kwargs):
        import cv2
        logger.debug("Opening USB camera id = %d..." % self.cv_cam_id)
        self.vcap = cv2.VideoCapture(self.cv_cam_id)
        self.vcap.set(3, self.W)
        self.vcap.set(4, self.H)
        # self.vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no queue
        # self.q = queue.Queue()
        # t = threading.Thread(target=self._reader)
        # t.daemon = True
        # t.start()
        while not self.vcap.isOpened():
            pass
        logger.debug("USB camera opened.")

    # # Read frames as soon as they are available, keeping only most recent one!
    # def _reader(self):
    #     while True:
    #         ret, frame = self.vcap.read()
    #         if not ret:
    #             break
    #         if not self.q.empty():
    #             try:
    #                 self.q.get_nowait()   # discard previous (unprocessed) frame
    #             except queue.Empty:
    #                 pass
    #         self.q.put(frame)

    def read_state(self, **kwargs) -> AttrDict:
        # self.frame = self.q.get()
        ret, self.frame = self.vcap.read()
        return AttrDict(frame=self.frame)

    def close(self):
        self.vcap.release()


# REALSENSE STUFF

class RSSeries(enum.Enum):
    d400 = 0
    sr300 = 1


class RSDepthCamera(Sensor):
    def _init_params_to_attrs(self, params: AttrDict):
        self.series = get_with_default(params, "series", RSSeries.d400)
        if self.series == RSSeries.sr300:
            self.config_json = None
            self.jsonObj = None
            self.json_string = None

            self.serial = get_required(params, "serial")
            self.visual_preset = get_with_default(params, "visual_preset", 9)

            self.W = get_with_default(params, "width", 640)
            self.H = get_with_default(params, "height", 480)
            self.fps = get_with_default(params, "fps", 30)
            self.zunits = get_with_default(params, "zunits", 1000)
        else:
            self.config_json = params.config_json
            self.jsonObj = json.load(open(self.config_json))
            self.json_string = str(self.jsonObj).replace("'", '\"')

            self.serial = None

            self.W = int(self.jsonObj['stream-width'])
            self.H = int(self.jsonObj['stream-height'])
            self.fps = int(self.jsonObj['stream-fps'])

            self.zunits = int(self.jsonObj['param-zunits'])  # units per meter

        self.do_decimate = params.get("do_decimate", True)
        self.do_spatial = params.get("do_spatial", True)
        self.do_temporal = params.get("do_temporal", True)
        self.do_hole_filling = params.get("do_hole_filling", False)
        self.do_disparity = params.get("do_disparity", False)
        self.do_colorize = params.get("do_colorize", False)

        self.do_depth = params.get("do_depth", True)

        self.rgb_img = np.empty((self.H, self.W, 3), dtype=np.uint8)

        if self.do_decimate:
            self.dH = self.H // 2
            self.dW = self.W // 2
        else:
            self.dH = self.H
            self.dW = self.W

        if self.do_colorize:
            self.depth_img = np.empty((self.dH, self.dW, 3), dtype=np.uint8)
        else:
            self.depth_img = np.empty((self.dH, self.dW), dtype=np.uint16)

        self.lock = threading.Lock()

    def get_depth_for_pixel(self, pix, image=None):
        if image is None:
            image = self.depth_img
        val = image[
            pix[1], pix[0]]  # pix is (x,y) corresponding to (width, height), true img is stored (height, width) order
        return float(val / self.zunits)  # in meters

    def get_intrinsics(self):
        import pyrealsense2 as rs
        profile = self.pipeline.get_active_profile()
        cprofile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        cintrinsics = cprofile.get_intrinsics()
        return np.array([[cintrinsics.fx, 0, cintrinsics.ppx], [0, cintrinsics.fy, cintrinsics.ppy],
                         [0, 0, 1.]]), cintrinsics.width, cintrinsics.height

    def _init_setup(self, **kwargs):
        import pyrealsense2 as rs
        logger.debug("Opening RealSense Depth camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.series == RSSeries.sr300:
            logger.debug(f"--> SR300: {self.serial}")
            self.config.enable_device(self.serial)

        if self.do_depth:
            self.config.enable_stream(rs.stream.depth, self.W,
                                      self.H, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.W,
                                  self.H, rs.format.bgr8, self.fps)
        self.cfg = self.pipeline.start(self.config)
        self.dev = self.cfg.get_device()

        if self.series != RSSeries.sr300:
            self.advnc_mode = rs.rs400_advanced_mode(self.dev)
            self.advnc_mode.load_json(self.json_string)
        elif self.do_depth:
            depth_sensor = self.cfg.get_device().first_depth_sensor()
            # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            depth_sensor.set_option(rs.option.visual_preset, self.visual_preset)

        self.align_to = rs.stream.color  # align everything to color frame for easy postproc
        self.align = rs.align(self.align_to)

        # filters
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.colorizer = rs.colorizer()

        self.has_frames = False
        self.is_open = True
        self._read_thread = threading.Thread(target=self._reader, daemon=True)
        self._read_thread.start()

        logger.debug("RS camera waiting for frames.")
        while not self.has_frames:
            time.sleep(0.01)

        logger.debug("RS camera opened.")

    # Read frames as soon as they are available, keeping only most recent one!
    def _reader(self):
        try:
            while self.is_open:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                self.has_frames = True
                if self.do_depth:
                    depth = aligned_frames.get_depth_frame()
                    if not depth: continue
                color_frame = aligned_frames.get_color_frame()

                if self.do_depth:
                    if self.do_decimate:
                        depth = self.decimation.process(depth)
                    depth = self.depth_to_disparity.process(depth)
                    if self.do_spatial:
                        depth = self.spatial.process(depth)
                    if self.do_temporal:
                        depth = self.temporal.process(depth)
                    if not self.do_disparity:
                        depth = self.disparity_to_depth.process(depth)
                    if self.do_hole_filling:
                        depth = self.hole_filling.process(depth)
                    if self.do_colorize:
                        depth = self.colorizer.colorize(depth)

                    np_image_depth = np.asanyarray(depth.get_data())

                np_image_rgb = np.asanyarray(color_frame.get_data())

                self.lock.acquire()
                if self.do_depth:
                    self.depth_img[:] = np_image_depth
                self.rgb_img[:] = np_image_rgb
                self.lock.release()
        finally:
            self.pipeline.stop()
            self.has_frames = False
            self.is_open = False

    def read_state(self, **kwargs) -> AttrDict:
        # self.frame = self.q.get()
        self.lock.acquire()
        out = AttrDict(active=np.array([self.has_frames and self.is_open]), rgb=self.rgb_img.copy(), depth=self.depth_img.copy())
        self.lock.release()
        return out

    def has_received_frames(self):
        return self.has_frames

    def close(self):
        self.is_open = False
        if hasattr(self, "_read_thread"):
            self._read_thread.join()
