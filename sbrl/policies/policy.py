"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
import numpy as np

from sbrl.envs.env import Env
from sbrl.envs.env_spec import EnvSpec
from sbrl.utils.python_utils import AttrDict, get_with_default


class Policy(object):

    def __init__(self, params: AttrDict, env_spec: EnvSpec, file_manager=None, env: Env = None, is_goal=False, input_handle=None):
        self._params = params
        self._env_spec = env_spec
        self._env = env
        self._file_manager = file_manager
        self._is_goal = is_goal
        self._out_names = self._env_spec.goal_names if self._is_goal else self._env_spec.action_names
        # additionally updates model outputs (inputs, model_outputs) -> new_model_outputs
        self._postproc_fn = get_with_default(params, "postproc_fn", lambda model, obs, goal, act: act)
        self._input_handle = input_handle
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        pass

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        return AttrDict()

    def get_random_action(self, model, observation, goal, **kwargs):
        """
        Gets a random action. default implementation is to return random actions of (B x H x ..)

        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        # first get the shape of the output (how many random action vectors)
        o = observation.leaf_copy()
        o.combine(goal)
        front_size = np.asarray(self._env_spec.get_front_size(o))
        batch_size = int(front_size[0])  # just the first dimension

        # then sample according to uniform distribution (default behavior, feel free to override based on policy type)
        random_uniform = self._env_spec.get_uniform(self._out_names, batch_size, torch_device=model.device)
        # AttrDict of B x ...
        ret = self._postproc_fn(model, observation, goal, random_uniform)
        return ret

    def _set_fn(self, name, func, ftype):
        if func is None:
            return

        def _internal_setter(fn: ftype):
            self.__setattr__(name, fn)
        _internal_setter(func)

    # this optional func is called when the environment resets (only when stepping env)
    def reset_policy(self, **kwargs):
        pass

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return False

    def is_initiated(self, model, observation, goal, **kwargs) -> bool:
        return True

    @property
    def params(self):
        return self._params.leaf_copy()

    @property
    def file_manager(self):
        return self._file_manager

    # policies can be duplicated on initialization, this handles that
    def duplicate(self):
        return self.__class__(self.params, self._env_spec, self.file_manager, env=self._env, is_goal=self._is_goal)
