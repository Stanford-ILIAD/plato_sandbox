import gym
import numpy as np
from gym import envs

from sbrl.envs.env import Env
from sbrl.envs.env_spec import EnvSpec
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import to_numpy

"""
This is where actions actually get executed on the robot, and observations are received.
"""

SUPPORTED = {env_spec.id: env_spec
             for env_spec in list(envs.registry.all())
             }


class GymEnv(Env):
    def __init__(self, params, env_spec: EnvSpec):
        super().__init__(params, env_spec)

        # PARAMS
        self._render = params.render
        self._env_type = params.env_type

        # LOCAL VARIABLES
        self._env = gym.make(self._env_type)

        self._obs = np.zeros(self._env.observation_space.shape)
        self._reward = 0
        self._done = False

    def _init_parameters(self, params):
        pass

    # this will be overriden
    def step(self, action, **kwargs):
        # batch input
        base_action = to_numpy(action.action[0], check=True)
        self._obs, self._reward, self._done, info = self._env.step(base_action, )
        if self._render:
            self._env.render()
        return self.get_obs(), self.get_goal(), np.array([self._done])

    def reset(self, presets: AttrDict = AttrDict()):
        self._obs = self._env.reset()
        self._reward = 0
        return self.get_obs(), self.get_goal()

    def get_obs(self):
        return self._env_spec.map_to_types(
            AttrDict(
                obs=self._obs.copy()[None],
                reward=np.array([[self._reward]])
            )
        )

    def get_goal(self):
        return self._env_spec.map_to_types(
            AttrDict(
            )
        )
