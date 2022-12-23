import torch

from sbrl.envs.env import Env
from sbrl.envs.env_spec import EnvSpec
from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict


class Reward(object):

    def __init__(self, params: AttrDict, env_spec: EnvSpec, file_manager=None):
        self._params = params
        self._env_spec = env_spec
        self._file_manager = file_manager
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        pass

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def get_reward(self, env: Env, model: Model, observation, goal, action, next_observation, next_goal, env_memory, done, policy_done, goal_policy_done=None,  **kwargs):
        """
        :param env
        :param model:
        :param observation:
        :param goal:
        :param action:
        :param next_observation:
        :param next_goal:
        :param env_memory:
        :param done:
        :param policy_done:
        :param goal_policy_done:
        """
        shape = list(self._env_spec.get_front_size(observation)) + [1]
        return torch.zeros(shape, device=model.device)

    # this optional func is called when the environment resets (only when stepping env)
    def reset_reward(self, **kwargs):
        pass

    @property
    def params(self):
        return self._params.leaf_copy()
