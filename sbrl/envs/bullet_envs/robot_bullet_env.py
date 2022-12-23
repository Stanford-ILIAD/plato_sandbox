"""
Implements a single robot + scene abstraction as a gym-like Env, in line with the root Env class.

Scenes consist of a robot and a set of objects.
initialization is split into the following:

_init_bullet_world
_init_figure

loading is split into:

_load_robot
_load_assets
_load_dynamics

reset is split into

pre_reset
reset_robot
reset_assets
reset_dynamics
_reset_images

"""
import os
import time

import numpy as np
import pybullet as p

from sbrl.envs.block2d import teleop_functions
from sbrl.envs.env import Env
from sbrl.envs.env_spec import EnvSpec
from sbrl.envs.interfaces import VRInterface
from sbrl.experiments import logger
from sbrl.utils.input_utils import UserInput
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit
from sbrl.utils.torch_utils import to_numpy


class RobotBulletEnv(Env, VRInterface):
    def __init__(self, params: AttrDict, env_spec: EnvSpec):
        super(RobotBulletEnv, self).__init__(params, env_spec)
        self.asset_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../assets')
        logger.debug("Assets in: %s" % self.asset_directory)
        assert os.path.exists(self.asset_directory)

        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params: AttrDict):
        self.camdist = 0.05

        self.debug_cam_dist = get_with_default(params, "debug_cam_dist", 1.3)
        self.debug_cam_p = get_with_default(params, "debug_cam_p", -45)
        self.debug_cam_y = get_with_default(params, "debug_cam_y", 40)
        self.debug_cam_target_pos = get_with_default(params, "debug_cam_target_pos", [-0.2, 0, 0.75])
        self.gui_width, self.gui_height = get_with_default(params, "gui_width", 1920), get_with_default(params, "gui_height", 1080)
        self.img_width, self.img_height = params << "img_width", params << "img_height"

        self.time_step = get_with_default(params, "time_step", 0.02)  # stepSimulation dt
        self.skip_n_frames_every_step = get_with_default(params, "skip_n_frames_every_step", 5)  # 10Hz default
        self.dt = self.time_step * self.skip_n_frames_every_step
        self._render = get_with_default(params, "render", False)  # no GUI default
        self._use_gravity = get_with_default(params, "use_gravity", True)  # no gravity default
        self._control_inner_step = get_with_default(params, "control_inner_step", True)  # where to call _control
        self._max_steps = get_with_default(params, "max_steps", np.inf, map_fn=int)  # how many steps to run before quitting

        self.compute_images = get_with_default(params, "compute_images", True)  # returning images or not, False should speed things up

        # allows GL rendering,
        self.non_gui_mode = get_with_default(params, "non_gui_mode",
                                             p.GUI if self.compute_images and "DISPLAY" in os.environ.keys() else p.DIRECT)


        self.env_reward_fn = get_with_default(params, "env_reward_fn", None)  # computes reward, optional
        self._env_reward_requires_env = get_with_default(params, "env_reward_requires_env", False)  # pass in env

        # if self.compute_images and self._render:
        #     self._init_figure()

        self.debug = get_with_default(params, "debug", False)

        self._teleop_fn = get_with_default(params, "teleop_fn", teleop_functions.bullet_keys_teleop_fn)

    def _init_setup(self):
        self.id = None  # THIS SHOULD BE SET
        self._curr_obs = AttrDict()

        self._init_bullet_world()

        self.load()

        self.setup_timing()  # todo add to resetr

    def _init_bullet_world(self):
        if self._render:
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 '
                                               '--background_color_blue=1.0 --width=%d --height=%d' % (self.gui_width,
                                                                                                       self.gui_height))
            logger.warn('Render physics server ID: %d' % self.id)

            # if self.compute_images:
            #     print("Showing plot")
            #     plt.show(block=False)
            #     self.fig.canvas.draw()
        else:
            self.id = p.connect(self.non_gui_mode)
            logger.warn('Physics server ID: %d' % self.id)

        p.resetSimulation(physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

        # Disable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        p.setTimeStep(self.time_step, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=200, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)

    def _init_figure(self):
        raise NotImplementedError

    """ LOADERS """
    def load(self):
        """ 0. plane """
        p.loadURDF(os.path.join(self.asset_directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # load in robots
        self._load_robot()

        # load in assets
        self._load_assets()

        # load in dynamics (constraints, etc)
        self._load_dynamics()

    def _load_robot(self, presets: AttrDict = AttrDict()):
        raise NotImplementedError

    # Load all models off screen and then move them into place
    def _load_assets(self, presets: AttrDict = AttrDict()):
        pass

    # do stuff here like collision handling
    def _load_dynamics(self, presets: AttrDict = AttrDict()):
        pass

    """ GETTERS """
    def get_id(self):
        assert self.id is not None, "Id must be set!"
        return self.id

    def _get_obs(self, **kwargs):
        raise NotImplementedError

    def _get_goal(self, **kwargs):
        return AttrDict()

    def _get_reward(self, curr_obs, next_obs, goal, action, done):
        if self.env_reward_fn is None:
            return np.zeros((1, 1))
        if self._env_reward_requires_env:
            return self.env_reward_fn(curr_obs, goal, action, next_obs=next_obs, done=done, env=self)
        else:
            return self.env_reward_fn(curr_obs, goal, action, next_obs=next_obs, done=done)

    def _get_images(self, **kwargs):
        raise NotImplementedError

    def get_joint_indices(self):
        raise NotImplementedError

    def get_gripper_joint_indices(self):
        raise NotImplementedError

    def get_initial_joint_positions(self):
        raise NotImplementedError

    def get_joint_limits(self):
        raise NotImplementedError

    """ SETTERS """
    def reset_joint_positions(self, q, **kwargs):
        pass

    def reset_gripper_positions(self, g, **kwargs):
        pass

    def _control(self, action, **kwargs):
        pass

    def set_external_forces(self, action):
        pass

    def update_targets(self):
        pass

    """ TIMING """
    def slow_time(self, record=False):
        if record and self.last_sim_time is None:
            self.last_sim_time = time.time()
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)

        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    """ ENV """

    def _read_state(self):
        pass

    def _register_obs(self, obs: AttrDict, done: np.ndarray):
        self._curr_obs = obs.copy()

    def _get_done(self, obs: AttrDict = None):
        return np.array([self.iteration >= self._max_steps])

    def step(self, action, ret_images=True, skip_render=False, **kwargs):
        with timeit("env_step/total"):
            self.iteration += 1
            if self.last_sim_time is None:
                self.last_sim_time = time.time()

            with timeit("env_step/read_state"):
                self._read_state()

            # action = np.clip(action, a_min=self.robot_action_low, a_max=self.robot_action_high)  # TODO
            # action_robot = action * self.robot_action_multiplier
            if isinstance(action, AttrDict):
                act_np = to_numpy(action.action[0], check=True).copy()  # assumes batched (1,..)
            elif isinstance(action, np.ndarray):
                act_np = action.copy()  # assumes not batched
            else:
                pass

            # sets motor control for each joint
            if not self._control_inner_step:
                with timeit("env_step/control"):
                    self._control(act_np)

            # Update robot position
            for _ in range(self.skip_n_frames_every_step):
                if self._control_inner_step:
                    with timeit("env_step/control"):
                        self._control(act_np)

                with timeit("env_step/step"):
                    self.set_external_forces(act_np)  # each stepSim clears external forces
                    self._step_simulation()
                    self.update_targets()

                if self._render and not skip_render:
                    # Slow down time so that the simulation matches real time
                    with timeit("env_step/slow_time"):
                        self.slow_time()

            self._after_step_simulation()
            # self.record_video_frame()

            with timeit("env_step/post_step"):
                next_obs, next_goal, done = self._step_suffix(action, ret_images=ret_images, **kwargs)
        return next_obs, next_goal, done

    def _step_suffix(self, action, ret_images=True, **kwargs):
        next_obs = self._get_obs(ret_images=ret_images and self.compute_images)
        done = self._get_done(obs=next_obs)
        if not next_obs.has_leaf_key("reward"):
            next_obs.reward = self._get_reward(self._curr_obs, next_obs, AttrDict(), action, done)
        self._register_obs(next_obs, done)
        return next_obs, AttrDict(), done

    def _step_simulation(self):
        p.stepSimulation(physicsClientId=self.id)

    def _after_step_simulation(self):
        pass

    def clear_gui_elements(self):
        pass

    def cleanup(self):
        pass  # this is where you delete objects between resets

    # override safe
    def pre_reset(self, presets: AttrDict = AttrDict()):
        pass

    def reset_robot(self, presets: AttrDict = AttrDict()):
        raise NotImplementedError

    def reset_assets(self, presets: AttrDict = AttrDict()):
        pass

    def reset_dynamics(self, presets: AttrDict = AttrDict()):
        pass

    def _reset_images(self, presets: AttrDict = AttrDict()):
        pass

    # DO NOT OVERRIDE
    # @profile
    def reset(self,
              presets: AttrDict = AttrDict()):  # ret_images=False, food_type=None, food_size=None, food_orient_eul=None, mouth_orient_eul=None):

        self.setup_timing()

        # if is_next_cycle(self.num_resets, self.reset_full_every_n):  # TODO
        #     logger.warn("Resetting sim fully")
        #     p.resetSimulation(physicsClientId=self.id)
        #     self._load_assets()
        # process = psutil.Process(os.getpid())
        # before = process.memory_info().rss
        # logger.debug("Clearing old things: %s" % process.memory_info().rss)
        # cleaning up old objects (normal)
        self.cleanup()

        if self._render:
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # any initial setup actions
        self.pre_reset(presets)

        self.clear_gui_elements()

        self.reset_robot(presets)

        self.reset_assets(presets)

        self.reset_dynamics(presets)

        if self._render:
            p.resetDebugVisualizerCamera(cameraDistance=self.debug_cam_dist, cameraYaw=self.debug_cam_y, cameraPitch=self.debug_cam_p,
                                         cameraTargetPosition=self.debug_cam_target_pos, physicsClientId=self.id)

        if self._use_gravity:
            p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        else:
            p.setGravity(0, 0, 0, physicsClientId=self.id)

        # initialize everything
        for i in range(10):
            p.stepSimulation()

        # here is where we compute the camera images
        if self.compute_images:
            self._reset_images(presets)

        # process = psutil.Process(os.getpid())
        # after = process.memory_info().rss
        # logger.debug("-> After resetting things: %s | delta = %s" % (after, after - before))
        obs = self._get_obs(ret_images=presets.get("ret_images", True) and self.compute_images)
        if not obs.has_leaf_key("reward"):
            obs.reward = np.zeros((1, 1))
        done = self._get_done(obs=obs)
        # first obs
        self._register_obs(obs, done)
        return obs, self._get_goal()

    def get_link_info(self, object_id):
        numJoint = p.getNumJoints(object_id)
        LinkList = ['base']
        for jointIndex in range(numJoint):
            jointInfo = p.getJointInfo(object_id, jointIndex)
            # print("jointINFO", jointInfo)
            link_name = jointInfo[12]
            if link_name not in LinkList:
                LinkList.append(link_name)
        return LinkList

    def get_num_links(self, object_id):
        return len(self.get_link_info(object_id))

    def get_aabb(self, object_id):
        num_links = self.get_num_links(object_id)
        aabb_list = []
        # get all link bounding boxes, pick the max on each dim
        for link_id in range(-1, num_links-1):
            aabb_list.append(p.getAABB(object_id, link_id, physicsClientId=self.id))
        aabb_array = np.array(aabb_list)
        aabb_obj_min = np.min(aabb_array[:, 0, :], axis=0)
        aabb_obj_max = np.max(aabb_array[:, 1, :], axis=0)
        aabb_obj = np.array([aabb_obj_min, aabb_obj_max])
        return aabb_obj

    def get_default_teleop_model_forward_fn(self, user_input: UserInput):  # TODO add this to parents
        return self._teleop_fn(self, user_input)

    """ VR interface """
    def get_safenet(self):
        return np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])

    def change_view(self, delta_dist=0, delta_yaw=0, delta_pitch=0, delta_target=np.zeros(3)):
        self.debug_cam_dist += delta_dist
        self.debug_cam_y += delta_yaw
        self.debug_cam_p += delta_pitch
        self.debug_cam_target_pos = (np.asarray(self.debug_cam_target_pos) + delta_target).tolist()

        p.resetDebugVisualizerCamera(cameraDistance=self.debug_cam_dist, cameraYaw=self.debug_cam_y,
                                     cameraPitch=self.debug_cam_p,
                                     cameraTargetPosition=self.debug_cam_target_pos, physicsClientId=self.id)
