from .env_spec import EnvSpec
from ..utils.python_utils import get_with_default

"""
Fully parameterizable spec
"""

def strlist(obj):
    obj = list(obj)
    for s in obj:
        assert isinstance(s, str), s
    return obj

class ParamEnvSpec(EnvSpec):
    def _init_params_to_attrs(self, params):
        self._out_obs_names = strlist(params.output_observation_names)
        self._obs_names = strlist(params.observation_names)
        self._action_names = strlist(params.action_names)
        self._goal_names = strlist(get_with_default(params, "goal_names", []))
        self._output_goal_names = strlist(get_with_default(params, "output_goal_names", []))
        self._param_names = strlist(get_with_default(params, "param_names", []))
        self._final_names = strlist(get_with_default(params, "final_names", []))

    @property
    def output_observation_names(self):
        """
        Returns:
            list(str)
        """
        return self._out_obs_names

    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        return self._obs_names

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return self._goal_names

    @property
    def output_goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return self._output_goal_names

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        return self._action_names

    @property
    def param_names(self):
        return self._param_names

    @property
    def final_names(self):
        return self._final_names
