"""
multi-camera block environment. Single-object,
"""
import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from sbrl.envs.block_real.aruco import get_tag_pose, get_aruco_settings, get_multi_tag_poses
from sbrl.envs.sensor.sensors import Sensor
from sbrl.experiments import logger
from sbrl.utils import o3d_utils as o3u
from sbrl.utils.geometry_utils import CoordinateFrame, world_frame_3D
from sbrl.utils.python_utils import get_with_default, get_required, get_or_instantiate_cls, AttrDict


class MultiCameraBlockSensor(Sensor):

    def _init_params_to_attrs(self, params):
        # dict/AttrDict of AttrDicts
        self._camera_sensors = get_required(params, "camera_sensors")
        self._num_cameras = len(self._camera_sensors.keys())
        assert self._num_cameras > 0, "Must have camera in params!"
        # instantiate all the cameras, e.g. RSDepthCamera
        for key in list(self._camera_sensors.keys()):
            self._camera_sensors[key] = get_or_instantiate_cls(self._camera_sensors, key, Sensor)

        self._camera_order = sorted(list(self._camera_sensors.keys()))

        # will fill in with calibrate.
        self._origin_in_camera_frames = None
        self._camera_K = None

        self._visualize = get_with_default(params, "visualize", True)
        self._get_images = get_with_default(params, "get_images", False)

        self._extrinsics = get_with_default(params, "extrinsics", None)
        self._reload_extrinsics = get_with_default(params, "reload_extrinsics", False)
        self._reference_tag = get_with_default(params, "reference_tag", 0)
        self._reference_tag_size = get_with_default(params, "reference_tag_size", 0.04)
        self._aruco_type = get_with_default(params, "aruco_type", "DICT_4X4_50")
        self._aruco_dict, self._aruco_params = get_aruco_settings(self._aruco_type)

        # in "order" TODO
        self._block_side_tags = get_with_default(params, "block_side_tags", list(range(1, 7)))
        self._block_side_tag_size = get_with_default(params, "block_side_tag_size", 0.04)
        self._block_size = get_with_default(params, "block_size", 0.04)

        # LAYOUT: 1 is on +y axis (+z up), 2 is on +z axis (-y up), 3,4 mirror 1,2, respectively
        #         5 is on -x axis, (+z up), and 6 mirrors this.
        default_center2tag = [np.array([[-1, 0, 0],
                                        [0, 0, 1],
                                        [0, 1, 0]]),
                              np.array([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]]),
                              np.array([[1, 0, 0],
                                        [0, 0, 1],
                                        [0, -1, 0]]),
                              np.array([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]]),
                              np.array([[0, -1, 0],
                                        [0, 0, 0],
                                        [-1, 0, 1]]),
                              np.array([[0, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 0]]),
                              ]
        default_center_rel_tag = [CoordinateFrame(world_frame_3D, Rotation.from_matrix(c2t).inv(),
                                                  np.array([0., 0., -self._block_size / 2.]))
                                  for c2t in default_center2tag]
        self._object_center_rel_tag_frames = get_with_default(params, "object_center_rel_tag_frames",
                                                              default_center_rel_tag)
        self._last_object_pos = np.zeros(3)
        self._last_object_rot = Rotation.identity()
        self._origin_rel_world = world_frame_3D

    def _init_setup(self, **kwargs):
        # open all the cameras
        for key in self._camera_sensors.keys():
            self._camera_sensors[key].open()

        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            self.origin_geometry = o3u.draw_frame(world_frame_3D, size=0.04)
            self.object_geometry = o3u.draw_frame(world_frame_3D, size=0.04)
            self.table_geom = o3d.geometry.TriangleMesh.create_box(width=0.30, height=0.40, depth=0.001, create_uv_map=True)
            self.table_geom.translate(np.array([-0.15, -0.2, -0.001]))
            self.vis.add_geometry(self.origin_geometry)
            self.vis.add_geometry(self.object_geometry)
            self.vis.add_geometry(self.table_geom)

        self._camera_K = self._camera_sensors.leaf_apply(lambda s: s.get_intrinsics()[0])
        # Will fill in self._camera_coordinate_frames fully.
        self.calibrate()

        if self._visualize:
            for o_in_c in self._origin_in_camera_frames:
                self.vis.add_geometry(o3u.draw_frame(o_in_c.inv(), size=0.01))

    def read_state(self, **kwargs) -> AttrDict:
        # get object position in each camera
        # camera_obs = self._camera_sensors.leaf_apply(lambda sensor: sensor.read_state())
        computed_center_frames = []
        for i, key in enumerate(self._camera_order):
            # cimg = camera_obs >> f"{key}/rgb"
            true_o_rel_c = self._origin_in_camera_frames[i]
            all_t_rel_c, _, _ = get_multi_tag_poses(self._camera_sensors[key], self._camera_K[key], np.zeros(5),
                                                    self._block_side_tags, self._block_side_tag_size,
                                                    self._aruco_dict, self._aruco_params)
            for k in all_t_rel_c.keys():
                # now what is the transform for the id (which side is it)
                side_idx = self._block_side_tags.index(k)
                object_center_rel_tag = self._object_center_rel_tag_frames[side_idx]
                center_rel_c = all_t_rel_c[k].apply_a_to_b(world_frame_3D, object_center_rel_tag)
                # what is object frame relative to true origin?
                center_rel_o = center_rel_c.view_from_frame(true_o_rel_c)
                center_rel_world = CoordinateFrame(self._origin_rel_world, center_rel_o.rot.inv(), center_rel_o.pos)
                computed_center_frames.append(center_rel_world)

        if len(computed_center_frames) == 0:
            logger.warn("Object not in any frame!!")
            mean_pos = self._last_object_pos
            mean_rot = self._last_object_rot
        else:
            mean_pos = np.average([f.pos for f in computed_center_frames], axis=0)
            std_pos = np.std([f.pos for f in computed_center_frames], axis=0)
            mean_rot = Rotation.from_matrix(
                np.stack([f.rot.as_matrix() for f in computed_center_frames], axis=0)).mean()
            # logger.debug(f"[nf={len(computed_center_frames)}]: Object pos: {mean_pos}, std: {std_pos}. rotation = {mean_rot.as_euler('xyz')}")

            if self._visualize:
                # self.object_geometry.rotate(mean_rot.as_matrix())
                self.vis.remove_geometry(self.object_geometry)
                self.object_geometry = o3u.draw_frame(CoordinateFrame(world_frame_3D, mean_rot.inv(), mean_pos), size=0.04)
                self.vis.add_geometry(self.object_geometry)

                # vc.set_lookat(np.array([0, 0, 0.]))
                # vc.set_up(np.array([0, 0, 1.]))
                # self.vis.reset_view_point(True)
                # vc.set_zoom(0.5)

                # self.vis.update_geometry(self.object_geometry)
                self.vis.poll_events()
                self.vis.update_renderer()
                # vc = self.vis.get_view_control()
                # vc.rotate(10., 0.)

                # self.object_geometry.rotate(mean_rot.inv().as_matrix())

        # update latest
        self._last_object_pos = mean_pos
        self._last_object_rot = mean_rot

        obs = AttrDict(
            objects=AttrDict(
                position=mean_pos.copy(),
                orientation_eul=mean_rot.as_euler("xyz"),
                orientation=mean_rot.as_quat(),
                size=np.array([self._block_size] * 3),  # assumes cube
            ).leaf_apply(lambda arr: arr[None])  # 1 block. broadcast to fit this format
        )

        if self._get_images:
            for key in self._camera_order:
                obs[key] = self._camera_sensors[key].read_state() > ['rgb']
            obs['image'] = obs[self._camera_order[0]] >> 'rgb'

        return obs

    def reset(self, origin=world_frame_3D, **kwargs):
        self._camera_sensors.leaf_apply(lambda sensor: sensor.reset(**kwargs))
        self._origin_rel_world = origin
        self._last_object_pos = origin.pos
        self._last_object_rot = origin.rot

        if self._visualize:
            if hasattr(self, "vis"):
                self.vis.destroy_window()
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            # frame adjusted visualization
            self.origin_geometry = o3u.draw_frame(self._origin_rel_world, size=0.04)
            self.object_geometry = o3u.draw_frame(self._origin_rel_world, size=0.04)
            self.table_geom = o3d.geometry.TriangleMesh.create_box(width=0.30, height=0.40, depth=0.001, create_uv_map=True)
            self.table_geom.translate(CoordinateFrame.point_from_a_to_b(np.array([-0.15, -0.2, -0.001]), self._origin_rel_world, world_frame_3D))
            self.vis.add_geometry(self.origin_geometry)
            self.vis.add_geometry(self.object_geometry)
            self.vis.add_geometry(self.table_geom)

            for o_in_c in self._origin_in_camera_frames:
                c_in_o = o_in_c.inv()
                self.vis.add_geometry(o3u.draw_frame(CoordinateFrame(self._origin_rel_world, c_in_o.rot.inv(), c_in_o.pos), size=0.01))

    def close(self):
        self._camera_sensors.leaf_apply(lambda sensor: sensor.close())

    def calibrate(self):
        if not self._reload_extrinsics and self._extrinsics is not None and os.path.exists(self._extrinsics):
            extr = AttrDict.from_dict(np.load(self._extrinsics, allow_pickle=True))
            origin_in_camera_frames = []
            for key in self._camera_order:
                cm = extr >> key
                o2c = cm >> "origin_to_camera"
                o_in_c = cm >> "origin_in_camera"
                # where is the origin frame, from the camera_frame
                origin_in_camera_frames.append(CoordinateFrame(world_frame_3D, Rotation.from_matrix(o2c).inv(), o_in_c))
            logger.info("Camera frames loaded from file! Calibration done.")
        else:
            origin_in_camera_frames = []

            # for key in self._camera_order:
            #     cv2.namedWindow(key, cv2.WINDOW_AUTOSIZE)
            #     cv2.imshow(key, self._camera_sensors[key].read_state() >> "rgb")
            # cv2.waitKey(0)

            for key in self._camera_order:
                logger.info(f"Calibrating camera: {key}...")
                o2c, o_in_c, images, _, _ = get_tag_pose(self._camera_sensors[key], self._camera_K[key], np.zeros(5),
                                                         self._reference_tag, self._reference_tag_size,
                                                         self._aruco_dict, self._aruco_params, n_frames=10)
                # AR is not identical to global frame, this accounts for that
                o_rel_c = CoordinateFrame(world_frame_3D, Rotation.from_matrix(o2c).inv(), o_in_c)
                # true_o_rel_c = CoordinateFrame(o_rel_c, Rotation.from_euler("z", -np.pi / 2), np.zeros(3))
                origin_in_camera_frames.append(o_rel_c)

                # cv2.namedWindow(key, cv2.WINDOW_AUTOSIZE)
                # cv2.imshow(key, images)

            # cv2.waitKey(0)

            logger.info("Done calibrating.")

            if self._extrinsics is not None:
                logger.info(f"Saving extrinsics to -> {self._extrinsics}")
                to_save = AttrDict()
                for i, key in enumerate(self._camera_order):
                    to_save[key] = AttrDict(origin_to_camera=origin_in_camera_frames[i].rot.as_matrix(),
                                            origin_in_camera=origin_in_camera_frames[i].pos)
                np.savez(self._extrinsics, **to_save.as_dict())

        self._origin_in_camera_frames = origin_in_camera_frames
