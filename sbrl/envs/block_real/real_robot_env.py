import numpy as np
from scipy.spatial.transform import Rotation as R

import sbrl.utils.transform_utils as T
from sbrl.envs.env import Env
from sbrl.envs.sensor.sensors import Sensor
from sbrl.experiments import logger
from sbrl.utils.geometry_utils import world_frame_3D, CoordinateFrame, clip_ee_orientation_conical
from sbrl.utils.python_utils import AttrDict, get_or_instantiate_cls, get_required, get_with_default
from sbrl.utils.torch_utils import to_numpy


class RealRobotEnv(Env):
    def __init__(self, params, env_spec):
        super(RealRobotEnv, self).__init__(params, env_spec)

        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        # achilles client
        from achilles.robots.robot_interface import RobotInterface
        self.robot_interface = get_or_instantiate_cls(params, "robot_interface", RobotInterface)

        self.dt = get_required(params, "dt")  # in seconds
        self.action_name = get_with_default(params, "action_name", "action")  # will be pose [pos | euler | grip]
        self.tip_pos_in_ee = get_with_default(params, "tip_pos_in_ee",
                                              np.array([0, 0, 0.1]))  # will be pose [pos | euler | grip]

        self.robot_rel_world = get_with_default(params, "robot_rel_world",
                                                world_frame_3D)  # will be the base coordinate frame, relative to world_frame_3D

        self.reset_q = get_with_default(params, "reset_q", None)  # joint angles
        self._alpha = get_with_default(params, "smoothing", 1.)  # smoothing (default 1. = no smoothing)

        # object interface, for reading from cameras, for example.
        if (params << "object_sensor") is not None:
            self.object_sensor = get_or_instantiate_cls(params, "object_sensor", Sensor)
        else:
            self.object_sensor = None

        self._object_sensor_cf = get_with_default(params, "object_sensor_frame", None)
        self._last_grip = np.inf

    def _init_setup(self):
        """ Initialize robot through achilles """
        self.robot_interface.run_setup()
        # self.robot_interface.home_gripper()
        logger.info(f"[{1}] q-start {self.robot_interface.q.round(4)}")

        self.rate = self.robot_interface.ros_node.create_rate(1 / self.dt)

        if self.object_sensor is not None:
            self.object_sensor.open()

        if self._object_sensor_cf is None:
            # calibrate object sensor position.
            self.robot_interface.change_controller("Floating", AttrDict())
            # init_ee_pos = self.robot_interface.ee_position.copy()
            logger.debug(f"Move gripper tip to the table center.")  # Current ee pose: {init_ee_pos}")
            input("Press enter to continue..")
            # this is really the tip
            tip_pos = self.robot_interface.ee_position
            # ee_rot = R.from_quat(self.robot_interface.ee_orientation)
            # ee_frame = CoordinateFrame(self.robot_rel_world, ee_rot.inv(), ee_pos)
            # tip_frame = CoordinateFrame(ee_frame, R.identity(), self.tip_pos_in_ee)
            # tip_pos = ee_pos
            logger.debug(f"Final Tip pose: {tip_pos}")

            # relative to robot position, but viewed in world frame
            self._object_sensor_cf = CoordinateFrame(world_frame_3D, R.identity(), tip_pos)

        if self.object_sensor is not None:
            self.object_sensor.reset(origin=self._object_sensor_cf)  # set the coordinate frame

    def step(self, action):
        if action is not None:
            act = action >> self.action_name
            act = to_numpy(act, check=True).reshape((7,)).copy()
            # pos = act[:3]
            # quat = T.euler2quat_ext(act[3:6])

            desired_ee_frame = CoordinateFrame.from_pose(act[:6], world_frame_3D)
            desired_ee_frame = desired_ee_frame.view_from_frame(self.robot_rel_world)

            desired_tip_frame = CoordinateFrame(desired_ee_frame, R.identity(), self.tip_pos_in_ee) #+ np.array([0., 0., 0.02]))

            desired_tip_pose = self._clip_safe_action(desired_tip_frame.as_pose(world_frame_3D))
            desired_tip_frame = CoordinateFrame.from_pose(desired_tip_pose, world_frame_3D)

            pos = desired_tip_frame.pos
            quat = desired_tip_frame.rot.as_quat()

            logger.debug(f"RAW: {act[:3]} - ACTION: {pos} - CURRENT {self.robot_interface.ee_position} - LAST {self._last_desired_tip_pose[:3]}")
            pos = self._alpha * pos + (1 - self._alpha) * self._last_desired_tip_pose[:3]
            self._last_desired_tip_pose = desired_tip_frame.as_pose(world_frame_3D)

            # starts as 0 -> 255, we want as max_width -> 0 TODO property on robot int
            grip = self.robot_interface.gripper_max_width * (1 - (act[6] / 255))

            # send msg ( of where tip should be )
            self.robot_interface.set_ee_pose(pos.tolist(), quat.tolist())
            # # set gripper TODO does this spawn too many threads?
            if abs(grip - self.robot_interface.gripper_width) > 0.05 and abs(grip - self._last_grip) > 0.05:
                self.robot_interface.clear_gripper_actions()
                self.robot_interface.set_gripper_to_value(grip, sync=False)
                self._last_grip = grip
            print(grip, self.robot_interface.gripper_width)
            # thresh = self.robot_interface.gripper_max_width / 2.
            # if grip > thresh and self._last_grip <= thresh:
            #     self.robot_interface.open_gripper(sync=False)
            # elif grip <= thresh and self._last_grip > thresh:
            #     self.robot_interface.close_gripper(sync=False)

            self._last_grip = grip

        self.rate.sleep()
        return self._get_obs(), AttrDict(), np.array([False])

    def _clip_safe_action(self, pose):
        # clips orientation to be in a cone around -z axis
        pos = pose[:3]
        orn_eul = pose[3:6]
        orn_eul = clip_ee_orientation_conical(orn_eul, ee_axis=np.array([0, 0, 1.]),
                                              world_axis=np.array([0, 0, -1.]), max_theta=np.pi / 4)
        # clips position to be in safe box around object marker.
        object_center = self._object_sensor_cf.pos
        bounds = np.array([0.5, 0.5, 0.3])
        cage_center = object_center + np.array([0, 0, bounds[2] / 2.])  # up in the air + tolerance for gripper size
        min_pos = cage_center - bounds / 2
        max_pos = cage_center + bounds / 2
        pos = np.clip(pos, min_pos, max_pos)

        return np.concatenate([pos, orn_eul])

    def reset(self, presets: AttrDict = AttrDict()):
        # blocking
        #self.robot_interface.open_gripper(sync=False)
        self.robot_interface.reset(q=self.reset_q)
        self.robot_interface.change_controller("CartesianPosture",
                                               AttrDict(posture=self.robot_interface.neutral_joint_angles,
                                                        max_dpose=np.array([80., 1000.]),#))
                                                        kp=np.array([1200., 1200., 1200., 50., 50., 50.]),
                                                        kv=np.array([65, 65, 65, 10., 10., 10.])))  #

        self._last_desired_tip_pose = self.robot_interface.ee_pose.copy()

        if self.object_sensor is not None:
            self.object_sensor.reset(origin=self._object_sensor_cf)

        return self._get_obs(), AttrDict()

    def _get_obs(self, **kwargs):
        if self.object_sensor is not None:
            env_obs = self.object_sensor.read_state()
        else:
            env_obs = AttrDict()
        # env_obs.pprint()
        # TODO
        pos = self.robot_interface.ee_position
        ori = self.robot_interface.ee_orientation
        ori_eul = T.quat2euler_ext(ori)
        true_tip_frame = CoordinateFrame.from_pose(np.concatenate([pos, ori_eul]), self.robot_rel_world)
        tip_frame = CoordinateFrame(true_tip_frame, R.identity(), -np.array([0., 0., 0.0]))
        ee_frame = CoordinateFrame(tip_frame, R.identity(), -self.tip_pos_in_ee)

        obs = env_obs & AttrDict(
            ee_position=ee_frame.pos,
            ee_orientation=ee_frame.orn,
            ee_orientation_eul=ee_frame.rot.as_euler("xyz"),
            joint_positions=self.robot_interface.q,
            joint_velocities=self.robot_interface.dq,
            gripper_tip_pos=tip_frame.pos,
            gripper_pos=np.array([self.sim_to_real_gripper(self.robot_interface.gripper_width, inverse=True)]),
            # angle (0 -> 255)
            gripper_pos_raw=np.array([self.robot_interface.gripper_width]),
        )

        return obs.leaf_apply(lambda arr: arr[None])

    def sim_to_real_gripper(self, grip, inverse=False):
        # print(grip, self.robot_interface.gripper_max_width)
        if inverse:
            return (1 - grip / self.robot_interface.gripper_max_width) * 255.
        else:
            return (1 - grip / 255) * self.robot_interface.gripper_max_width

    def get_safenet(self):
        low, high = np.array([-0.25, -0.5, 0.]), np.array([0.25, -0.1, 0.5])
        return low + self.robot_rel_world.pos, high + self.robot_rel_world.pos

    def __del__(self):
        self.robot_interface.terminate_ctrl(0, None)


def get_block_real_example_spec_params(NB=1, img_height=640, img_width=480, img_channels=3, no_names=False, num_cams=3):
    prms = AttrDict(names_shapes_limits_dtypes=[
        ("image", (img_height, img_width, img_channels), (0, 255), np.uint8),
        *[(f"camera{i}/rgb", (img_height, img_width, img_channels), (0, 255), np.uint8) for i in range(num_cams)],
        ("image", (img_height, img_width, img_channels), (0, 255), np.uint8),
        # ('wrist_ft', (6,), (-np.inf, np.inf), np.float32),
        ('ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('ee_orientation_eul', (3,), (-np.inf, np.inf), np.float32),
        # ('ee_velocity', (3,), (-np.inf, np.inf), np.float32),
        # ('ee_angular_velocity', (3,), (-np.inf, np.inf), np.float32),
        ('joint_positions', (7,), (-np.inf, np.inf), np.float32),
        ('joint_velocities', (7,), (-np.inf, np.inf), np.float32),
        ('gripper_pos', (1,), (-np.inf, np.inf), np.float32),
        ('gripper_pos_raw', (1,), (-np.inf, np.inf), np.float32),
        ('gripper_tip_pos', (3,), (-np.inf, np.inf), np.float32),
        ('objects/position', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/orientation_eul', (NB, 3), (-np.inf, np.inf), np.float32),
        ('objects/orientation', (NB, 4), (-np.inf, np.inf), np.float32),
        ('objects/size', (NB, 3), (0, np.inf), np.float32),

        # TODO in a pre-defined box of allowed motion.
        ('action', (7,), (-np.inf, np.inf), np.float32),
        ('target/ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('target/ee_orientation_eul', (3,), (-2 * np.pi, 2 * np.pi), np.float32),
        ('target/gripper_pos', (1,), (0, 255.), np.float32),

        ("policy_type", (1,), (0, 255), np.uint8),
        ("policy_name", (1,), (0, 1), np.object),
        ("policy_switch", (1,), (False, True), np.bool),  # marks the beginning of a policy

    ], observation_names=[
        "ee_position", "ee_orientation_eul", "ee_orientation",
        "joint_positions", "gripper_pos", "gripper_pos_raw", "gripper_tip_pos",
        "objects/position", "objects/orientation_eul", "objects/orientation",
    ],
        param_names=["objects/size"],
        final_names=[],
        action_names=["action", "target/ee_position", "target/ee_orientation_eul", "target/gripper_pos", "policy_type",
                      "policy_name", "policy_switch"],
        output_observation_names=[]
    )
    if no_names:
        prms.action_names.remove('policy_name')
    return prms
