"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import AttrDict, get_with_default
from sbrl.utils.torch_utils import to_torch


class MemoryPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._policy_model_forward_fn = get_with_default(params, "policy_model_forward_fn",
                                                         lambda model, obs, goal, memory, **kwargs: model.forward(
                                                             obs.leaf_apply(lambda arr: arr[:, 0]), **kwargs))
        self._is_terminated_fn = get_with_default(params, "is_terminated_fn", lambda model, obs, goal, memory, **kwargs: False)
        self._provide_policy_env_in_forward = get_with_default(params, "provide_policy_env_in_forward", False)

    def warm_start(self, model, observation, goal):
        pass

    def _init_setup(self):
        self.memory = AttrDict()

    def reset_policy(self, **kwargs):
        self.memory.clear()

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)
        note: kwargs are for model forward fn

        :return action: AttrDict (B x ...)
        """

        # make sure types are correct for each array
        observation = self._env_spec.map_to_types(observation, skip_keys=True)
        goal = self._env_spec.map_to_types(goal, skip_keys=True)
        # convert to torch tensors
        observation = observation.leaf_apply(func=lambda arr: to_torch(arr, device=model.device, check=True))
        goal = goal.leaf_apply(func=lambda arr: to_torch(arr, device=model.device, check=True))
        if self._provide_policy_env_in_forward:
            kwargs['policy'] = self
            kwargs['env'] = self._env

        action_output = self._policy_model_forward_fn(model, observation, goal, self.memory, **kwargs)
        intersection = [key for key in action_output.keys() if key in self._env_spec.action_names]
        self._env_spec.clip(action_output, intersection, object_safe=True)
        return action_output

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._is_terminated_fn(model, observation, goal, self.memory, **kwargs)
