import numpy as np

from sbrl.envs.spec import Spec
from sbrl.utils.python_utils import get_with_default


class EnvSpec(Spec):

    def __init__(self, params):
        self._done_key = get_with_default(params, "done_key", "done", map_fn=str)
        names_shapes_limits_dtypes = list(params.names_shapes_limits_dtypes)
        names_shapes_limits_dtypes += [(self._done_key, (), (False, True), np.bool),
                                       ('rollout_timestep', (), (0, 1e100), np.int)]
        params.names_shapes_limits_dtypes = names_shapes_limits_dtypes
        super().__init__(params)

    def _init_params_to_attrs(self, params):
        pass

    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def output_observation_names(self):
        return self.observation_names

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return []

    @property
    def output_goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return []

    @property
    def param_names(self):
        """
        Params are returned in the FIRST returned observation (i.e. in reset)
        Returns:
            list(str)
        """
        return []

    @property
    def final_names(self):
        """
        Final names are returned in the LAST observation (i.e. when done == True)
        Returns:
            list(str)
        """
        return []

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def names(self):
        """
        Returns:
            list(str)
        """
        return self.observation_names + self.goal_names + self.action_names + self.output_observation_names \
               + self.output_goal_names

    @property
    def all_names(self):
        """
        Returns:
            list(str)
        """
        return self.names + self.param_names + self.final_names
