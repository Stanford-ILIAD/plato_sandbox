"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_with_default


class RandomPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._action_names = get_with_default(params, "action_names", self._out_names)
        assert set(self._action_names).issubset(self._env_spec.all_names)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        B = observation.get_one().shape[0]
        return self._env_spec.get_uniform(self._action_names, B, torch_device=model.device)
