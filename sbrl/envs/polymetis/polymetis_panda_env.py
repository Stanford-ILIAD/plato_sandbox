"""
polymetis_panda_env.py

Core abstraction over the physical Franka Panda Robot hardware, sensors, and internal robot state. Follows a standard
OpenAI Gym-like API.

Credit: Sidd Karamcheti
"""
import time
from typing import Callable, Dict, List

import numpy as np
import torch

# from oncorr.robot.perception import RealSense
from sbrl.envs.env import Env
from sbrl.envs.interfaces import VRInterface
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.sensor.sensors import Sensor
from sbrl.experiments import logger
from sbrl.utils.control_utils import Rate
from sbrl.utils.input_utils import KeyInput as KI, ProcessUserInput, UserInput, wait_for_keydown_from_set
from sbrl.utils.python_utils import get_or_instantiate_cls, get_required, get_with_default, AttrDict as d
from sbrl.utils.torch_utils import to_numpy
from sbrl.utils.transform_utils import fast_euler2quat_ext as euler2quat, fast_quat2euler_ext as quat2euler, \
    quat_multiply, add_euler

# fmt: off
HOMES = {
    # ILIAD Lab & Libfranka Default
    "default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],
}

# Libfranka Constants
#   > Ref: Gripper constants from: https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 60, 0.08570, 0.01

# Joint Controller gains for recording demonstrations -- we want a compliant robot, so setting all gains to ~0.
REC_KQ_GAINS, REC_KQD_GAINS = [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]
REC_KX_GAINS, REC_KXD_GAINS = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]

# Hardcoded Low/High Joint Thresholds for the Franka Emika Panda Arm
LOW_JOINTS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH_JOINTS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# fmt: on


class PolymetisPandaEnv(Env, VRInterface):
    def __init__(self, params, env_spec):
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param params
            home: Default home position (specified in joint space - 7-DoF for Pandas)
            hz: Default policy control Hz; somewhere between 20-60 is a good range.
            do_kinesthetic: Whether or not to initialize joint controller with zero PD gains for kinesthetic demos
            use_gripper: Whether or not to initialize a gripper in addition to the robot "base"
            delta_pivot: When `delta=True` in env.step, compute actions based on < ground-truth | expected > pose.
            camera: Whether or not to log camera observations (RGB)
        :param env_spec

        """
        super(PolymetisPandaEnv, self).__init__(params, env_spec)

        self.home = get_required(params, "home")
        self.hz = get_required(params, "hz")
        self.dt = 1. / self.hz
        self.do_kinesthetic = get_with_default(params, "do_kinesthetic", False)
        self.use_gripper = get_with_default(params, "use_gripper", True)
        self.delta_pivot = get_with_default(params, "delta_pivot", "ground-truth")
        self.franka_ip = get_with_default(params, "franka_ip", "172.16.0.1")

        self.action_space = get_with_default(params, "action_space", "ee-euler-delta")

        self.safenet = get_with_default(params, "safenet", np.array([[-np.inf, -np.inf, 0.], [np.inf, np.inf, np.inf]]))

        self.gripper, self.kq, self.kqd, self.kx, self.kxd = None, None, None, None, None


        self.camera = get_with_default(params, "camera", None)
        self.img_name = get_with_default(params, "img_name", "frame")
        if self.camera:
            logger.info("Loading camera sensor...")
            self.camera = get_or_instantiate_cls(self.camera, None, Sensor)
            self.camera.open()
        
        # Pose & Robot State Trackers
        self.current_joint_pose, self.current_ee_pose, self.current_gripper_state, self.current_ee_rot = None, None, None, None
        self.initial_ee_pose, self.initial_gripper_state, self.gripper_open, self.gripper_act = None, None, True, None

        # Expected/Desired Poses (for PD Controller Deltas)
        self.expected_q, self.expected_ee_quat, self.expected_ee_euler = None, None, None
        self.desired_pose, self.use_desired_pose = {"pos": None, "ori": None}, True
        #
        # # Initialize Robot and Cartesian Impedance Controller
        # #   => Cartesian Impedance uses `HybridJointImpedanceController` so we can send `joint` or `end-effector` poses!
        # self.reset()
        # initialize the interface
        self.rate = Rate(self.hz)

    def start_controller(self) -> None:
        import torchcontrol as toco
        """Start a HybridJointImpedanceController with all 4 of the desired gains; Polymetis defaults don't set both."""
        torch_policy = toco.policies.HybridJointImpedanceControl(
            joint_pos_current=self.robot.get_joint_positions(),
            Kq=self.robot.Kq_default if self.kq is None else self.kq,
            Kqd=self.robot.Kqd_default if self.kqd is None else self.kqd,
            Kx=self.robot.Kx_default if self.kx is None else self.kx,
            Kxd=self.robot.Kxd_default if self.kxd is None else self.kxd,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )
        self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def set_controller(self) -> None:
        # Special handling *only* applicable for "kinesthetic teaching"
        if self.do_kinesthetic:
            self.kq, self.kqd, self.kx, self.kxd = REC_KQ_GAINS, REC_KQD_GAINS, REC_KX_GAINS, REC_KXD_GAINS
        else:
            self.kq, self.kqd, self.kx, self.kxd = None, None, None, None

        # Start a *Cartesian Impedance Controller* with the desired gains...
        #   Note: P/D values of "None" default to HybridJointImpedance PD defaults from Polymetis
        #         |-> These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
        self.start_controller()

    def set_kinesthetic_and_reset(self, do_kinesthetic: bool):
        self.do_kinesthetic = do_kinesthetic
        return self.reset()

    def set_kinesthetic_without_reset(self, do_kinesthetic: bool) -> d:
        self.do_kinesthetic = do_kinesthetic

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = self.current_ee_pose[3:]
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Set Robot Motion Controller (e.g., joint or cartesian impedance...)
        self.set_controller()

        # Set `expected` and `desired_pose` if necessary...
        self.expected_q, self.expected_ee_quat = self.current_joint_pose.copy(), self.current_ee_pose.copy()
        self.expected_ee_euler = np.concatenate([self.expected_ee_quat[:3], quat2euler(self.expected_ee_quat[3:])])
        if self.use_desired_pose:
            self.desired_pose = {"pos": self.current_ee_pose[:3], "ori": self.current_ee_rot}

        # Return initial observation
        return self.get_obs()

    def _reset_initialize(self, presets):
        from polymetis import GripperInterface, RobotInterface
        self.robot = RobotInterface(ip_address=self.franka_ip, enforce_version=False)
        if self.use_gripper:
            self.gripper = GripperInterface(ip_address=self.franka_ip)
        
        # WAIT a bit
        time.sleep(1.0)
        
        # Initialize Robot Interface and Reset to Home
        self.robot.set_home_pose(torch.Tensor(self.home))
        self.robot.go_home()

        # camera reset
        if self.camera:
            self.camera.reset()

        # Set Robot Motion Controller (e.g., joint or cartesian impedance...)
        self.set_controller()

        # Initialize Gripper Interface & Open Gripper on each `robot_setup()`
        if self.use_gripper:
            self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            # print(self.gripper.get_state().width)
            # Set Gripper State...
            self.gripper_open, self.gripper_act = True, np.array(0.0)
            self.initial_gripper_state = self.current_gripper_state = {
                "width": self.gripper.get_state().width,
                "max_width": GRIPPER_MAX_WIDTH,
                "gripper_open": self.gripper_open,
                "gripper_action": self.gripper_act,
            }

    def _reset_set_controller_and_gripper(self):

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = self.current_ee_pose[3:]
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Set `expected` and `desired_pose` if necessary...
        self.expected_q, self.expected_ee_quat = self.current_joint_pose.copy(), self.current_ee_pose.copy()
        self.expected_ee_euler = np.concatenate([self.expected_ee_quat[:3], quat2euler(self.expected_ee_quat[3:])])
        if self.use_desired_pose:
            self.desired_pose = {"pos": self.current_ee_pose[:3], "ori": self.current_ee_rot}


    def reset(self, presets: d = d()):
        self._reset_initialize(presets)
        self._reset_set_controller_and_gripper()
        
        # Return initial observation
        return self.get_obs(), d()

    def user_input_reset(self, user_input: UserInput, reset_action_fn=None, presets: d = d()):
        self.user_input = user_input

        self._reset_initialize(presets)

        if isinstance(reset_action_fn, Callable):
            reset_action_fn()

        self.populate_display_fn("Press ('y') to continue once scene is reset.")
        _ = wait_for_keydown_from_set(user_input, [KI('y', KI.ON.down)], do_async=False)
        
        self._reset_set_controller_and_gripper()
        
        self.populate_display_fn("Running...")
        return self.get_obs(), d()

    def compute_reward(self, obs):
        # override this in sub-classes
        obs.reward = np.array([0.])
        return obs

    def get_done(self):
        return np.array([False])

    def get_obs(self) -> d:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = new_ee_pose[3:]

        obs = d(
            q=new_joint_pose,
            qdot=self.robot.get_joint_velocities().numpy(),
            ee_pose=new_ee_pose,
            ee_position=new_ee_pose[:3],
            ee_orientation=new_ee_rot,
            ee_orientation_eul=quat2euler(new_ee_rot),
        )
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = new_joint_pose, new_ee_pose, new_ee_rot

        if self.use_gripper:
            new_gripper_state = self.gripper.get_state()
            obs = obs & d(
                gripper_width=np.array([new_gripper_state.width]),
                gripper_pos=np.array([1 - new_gripper_state.width / GRIPPER_MAX_WIDTH]),  # 1 for closed
                gripper_max_width=np.array([GRIPPER_MAX_WIDTH]),
                gripper_open=np.array([self.gripper_open]),
                gripper_action=self.gripper_act[None],
            )
            self.current_gripper_state = {
                "width": self.gripper.get_state().width,
                "max_width": GRIPPER_MAX_WIDTH,
                "gripper_open": self.gripper_open,
                "gripper_action": self.gripper_act,
            }

        # Get camera observation (if enabled)
        if self.camera:
            obs["image"] = self.camera.read_state() >> self.img_name

        obs = self.compute_reward(obs)

        return obs.leaf_apply(lambda arr: arr[None])

    def step(self, act: d):
        action = to_numpy(act >> 'action', check=True).reshape(-1)
        open_gripper = action[-1] <= 0.5  # 0 for open, 1 for closed.
        """Run an environment step, where `delta` specifies if we are sending absolute poses or finite differences."""
        if action is not None:
            if self.action_space == "joint":
                self.robot.update_desired_joint_positions(torch.from_numpy(action).float())
                self.expected_ee_euler = np.concatenate([self.current_ee_pose[:3], quat2euler(self.current_ee_pose[3:])])

            elif self.action_space == "joint-delta":
                if self.delta_pivot == "ground-truth":
                    next_q = self.current_joint_pose + action
                elif self.delta_pivot == "expected":
                    next_q = self.expected_q = self.expected_q + action
                else:
                    raise ValueError(f"Delta Pivot `{self.delta_pivot}` not supported!")

                # Act!
                self.robot.update_desired_joint_positions(torch.from_numpy(next_q).float())
                self.expected_ee_euler = np.concatenate([self.current_ee_pose[:3], quat2euler(self.current_ee_pose[3:])])

            elif self.action_space == "ee-euler":
                # Compute quaternion from euler...
                desired_pos, desired_quat = action[:3], euler2quat(action[3:6])

                # Send to controller =>> Both position & orientation control!
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(desired_pos).float(),
                    orientation=torch.from_numpy(desired_quat).float(),
                )
                self.expected_q = self.current_joint_pose

            elif self.action_space == "ee-euler-delta":
                if self.delta_pivot == "ground-truth":
                    next_pos = self.current_ee_pose[:3] + action[:3]
                    next_quat = quat_multiply(euler2quat(action[3:6]), self.current_ee_rot)
                elif self.delta_pivot == "expected":
                    next_pos = self.expected_ee_euler[:3] = self.expected_ee_euler[:3] + action[:3]
                    self.expected_ee_euler[3:] = add_euler(action[3:6], self.expected_ee_euler[3:])
                    next_quat = euler2quat(self.expected_ee_euler[3:])
                else:
                    raise ValueError(f"Delta Pivot `{self.delta_pivot}` not supported!")

                # Send to controller =>> Both position & orientation control!
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(next_pos).float(),
                    orientation=torch.from_numpy(next_quat).float(),
                )
                self.expected_q = self.current_joint_pose

            else:
                raise NotImplementedError(f"Support for Action Space `{self.action_space}` not yet implemented!")

        # Discrete Grasping (Open/Close)
        if open_gripper is not None and (self.gripper_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), d(), self.get_done()

    def plan_trajectory_to_ee_pose(
        self, position: torch.Tensor, orientation: torch.Tensor, op_space_interp: bool = True
    ) -> List[Dict]:
        """
        Simple function leveraging `torchcontrol` to generate a `min-jerk` trajectory of joint states to traverse to
        move from the current robot position to the specified end-effector pose.
            => Ref => Polymetis' default `robot_interface.py`

        :param position: 3-DoF target end-effector position.
        :param orientation: 3-DoF target end-effector orientation.
        :param op_space_interp: Whether to interpolate in operational space/Cartesian space vs. joint space, to get
                                smooth movement in end-effector space!

        :return List[Dict] where each element corresponds to the joint positions / velocities / accelerations for the
                length of the planned trajectory (synced to `self.hz`).
        """
        #assert len(orientation) == 4, "Orientation must be specified as quaternion!"
        joint_pos_current = self.robot.get_joint_positions()

        # Get desired position, orientation --> also compute joint space target via IK
        ee_pos_desired, ee_quat_desired = torch.Tensor(position), torch.Tensor(euler2quat(orientation))
        print(ee_pos_desired)
        print(ee_quat_desired)
        joint_pos_desired, success = self.robot.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_current
        )

        # Fail (crash) if no successful plans found --> this shouldn't ever happen!
        if not success:
            raise ValueError("No valid plans found for moving to desired position/orientation!")

        # Leverage built-in `adaptive_time_to_go` computation for figuring out trajectory length...
        #   => For short distances, adaptive_time_to_go returns short lengths... min duration should be 2s!
        time_to_go = self.robot._adaptive_time_to_go(joint_pos_desired - joint_pos_current)
        time_to_go = max(time_to_go, 2.0)

        # Generate Waypoints...
        if op_space_interp:
            import torchcontrol as toco
            from torchcontrol.transform import Rotation as R
            from torchcontrol.transform import Transformation as T
            # Compute operational space trajectory...
            ee_pose_desired = T.from_rot_xyz(rotation=R.from_quat(ee_quat_desired), translation=ee_pos_desired)
            return toco.planning.generate_cartesian_target_joint_min_jerk(
                joint_pos_start=joint_pos_current,
                ee_pose_goal=ee_pose_desired,
                time_to_go=time_to_go,
                hz=self.hz,
                robot_model=self.robot.robot_model,
                home_pose=self.robot.home_pose,
            )
        else:
            raise NotImplementedError("Joint space interpolation is not yet implemented!")

    @property
    def ee_position(self) -> np.ndarray:
        """Return current EE position --> 3D x/y/z."""
        return self.current_ee_pose[:3] if not self.use_desired_pose else self.desired_pose["pos"]

    @property
    def ee_orientation(self) -> np.ndarray:
        """Return current EE orientation --> quaternion [i, j, k, w]."""
        return self.current_ee_rot if not self.use_desired_pose else self.desired_pose["ori"]

    @property
    def ground_truth_ee_pose(self) -> np.ndarray:
        return np.concatenate([ee.numpy() for ee in self.robot.get_ee_pose()])

    @property
    def ground_truth_joint_state(self) -> np.ndarray:
        return self.robot.get_joint_positions().numpy()

    def vr_action_scale(
        self,
        pos_velocity: np.ndarray,
        ori_velocity: np.ndarray,
        max_pos_velocity: float = 1.0,
        max_ori_velocity: float = 1.0,
    ):
        # Clip Values to Max Norm
        pos_velocity_norm, ori_velocity_norm = np.linalg.norm(pos_velocity), np.linalg.norm(ori_velocity)
        if pos_velocity_norm > max_pos_velocity:
            pos_velocity = pos_velocity * (max_pos_velocity / pos_velocity_norm)
        elif ori_velocity_norm > max_ori_velocity:
            ori_velocity = ori_velocity * (max_ori_velocity / ori_velocity_norm)

        # Scale Delta by Hz... (only for impedance control?)
        return pos_velocity / self.hz, ori_velocity / self.hz

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented for Physical PolymetisPandaEnv...")

    def set_gripper_width(self, width):
        self.gripper.goto(width=width, speed=0.05, force=0.1)

    def open_gripper(self):
        self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
        self.gripper_open = True
        return self.get_obs()

    def close_gripper(self):
        self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
        self.gripper_open = False
        return self.get_obs()

    def close(self) -> None:
        # Terminate Policy
        self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot, self.gripper = None, None
        time.sleep(1)

    """ VR STUFF """
    def get_safenet(self):
        # default safe bounds of the robot are inf
        return self.safenet

    def change_view(self, **kwargs):
        logger.warn("Changing the view is not implemented on real robot!")

    """ UI RESET helpers """

    def populate_display_fn(self, *args, **kwargs):
        if self.user_input is not None and self.user_input.has_display():
            if isinstance(self.user_input, ProcessUserInput):
                self.user_input.call_cmd("populate_display", *args, **kwargs)
                # this only works with pygame displays as o f rn
            else:
                self.user_input.get_display().populate_display(*args, **kwargs)


# helper
def get_polymetis_panda_example_spec_params(img_height=256, img_width=256, img_channels=3, minimal=False, no_names=False, n_objects=0, action_space="ee-euler", use_gripper=True, use_imgs=False):
    if action_space in ['joint', 'joint-delta']:
        AC_DIM = 8
    elif action_space in ['ee-euler', 'ee-euler-delta']:
        AC_DIM = 7
    else:
        raise NotImplementedError(action_space)

    prms = d(names_shapes_limits_dtypes=[
        ("image", (img_height, img_width, img_channels), (0, 255), np.uint8),

        ("ee_pose", (7,), (-np.inf, np.inf), np.float32),
        ("ee_position", (3,), (-np.inf, np.inf), np.float32),
        ("ee_orientation_eul", (3,), (-np.pi, np.pi), np.float32),
        ("ee_orientation", (4,), (-1., 1.), np.float32),
        ("gripper_pos", (1,), (0., 1.), np.float32),
        ("gripper_width", (1,), (0., np.inf), np.float32),
        ("max_width", (1,), (0., np.inf), np.float32),
        ("gripper_open", (1,), (False, True), bool),

        ("qdot", (7,), (-np.inf, np.inf), np.float32),
        ("q", (7,), (-np.inf, np.inf), np.float32),

        # TODO put these in
        ("objects/position", (n_objects, 3), (-np.inf, np.inf), np.float32),
        ("objects/orientation", (n_objects, 4), (-np.inf, np.inf), np.float32),
        ("objects/orientation_eul", (n_objects, 3), (-np.inf, np.inf), np.float32),

        ('action', (AC_DIM,), (-1, 1.), np.float32),
        ('reward', (1,), (-np.inf, np.inf), np.float32),

        ("click_state", (1,), (False, True), np.bool),
        ("mode", (1,), (0, 255), np.uint8),
        ("real", (1,), (False, True), bool),

        ("policy_type", (1,), (0, 255), np.uint8),
        ("policy_name", (1,), (0, 1), object),
        ("policy_switch", (1,), (False, True), bool),  # marks the beginning of a policy

        # target
        ('target/ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('target/ee_orientation', (4,), (-1, 1.), np.float32),
        ('target/ee_orientation_eul', (3,), (-np.pi, np.pi), np.float32),

        # delta wp
        ('delta_waypoint', (6,), (-1., 1.), np.float32),

        # raw actions
        ('raw/action', (AC_DIM,), (-1, 1.), np.float32),
        ('raw/target/ee_position', (3,), (-np.inf, np.inf), np.float32),
        ('raw/target/ee_orientation', (4,), (-1, 1.), np.float32),
        ('raw/target/ee_orientation_eul', (3,), (-np.pi, np.pi), np.float32),

    ], observation_names=['ee_position', 'ee_orientation', 'ee_orientation_eul', 'q', 'qdot'] + (['gripper_width', 'gripper_pos', 'gripper_open'] if use_gripper else []),
        param_names=[],
        final_names=[],
        action_names=["action", "policy_type", "policy_name", "policy_switch"],
        output_observation_names=["reward"]
    )

    if n_objects > 0:
        prms.observation_names.extend(['objects/position', 'objects/orientation_eul'])

    if no_names:
        prms.action_names.remove('policy_name')

    if use_imgs:
        prms.observation_names.append('image')

    if minimal:
        raise NotImplementedError

    return prms


if __name__ == '__main__':
    ac_space = 'ee-euler-delta'
    HZ = 10
    env_params = d(
        home=HOMES['default'],
        hz=HZ,
        use_gripper=False,
        franka_ip="172.16.0.1",
    )
    env_spec = ParamEnvSpec(get_polymetis_panda_example_spec_params(action_space=ac_space))
    env = PolymetisPandaEnv(env_params, env_spec)

    logger.debug("Resetting environment...")
    env.reset()

    logger.debug("Stepping...")
    for i in range(10 * HZ):
        obs, goal, done = env.step(d(action=np.zeros(7)))
        if done:
            logger.warn("Done early!")
            break
        if i % HZ == 0:
            logger.debug(f"[{i}] EE POSE = {(obs >> 'ee_pose').reshape(-1)}")

    logger.debug("Done")
