from sbrl.envs.env import Env
from sbrl.envs.env_spec import EnvSpec
from sbrl.utils.python_utils import AttrDict


class Reward(object):
    def __init__(self, params: AttrDict, env_spec: EnvSpec, env: Env = None):
        self._params = params
        self._env_spec = env_spec
        self._env = env

        # this might be used in init_params_to_attrs
        params["env_spec"] = env_spec
        params["env"] = env
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        pass

    def _init_setup(self):
        pass

    def reset_reward(self):
        pass

    def get_reward(self, env, model, obs, goal, action, next_obs, next_goal,
                   env_memory, policy_done, goal_policy_done, done, env_step):
        # should return an AttrDict with at least the key 'reward' - should be a np.array of shape (1, 1)
        raise NotImplementedError
