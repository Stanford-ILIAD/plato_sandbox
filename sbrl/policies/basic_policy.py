"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
from sbrl.experiments import logger
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_with_default
from sbrl.utils.torch_utils import to_torch


class BasicPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # self.sample_action = params.sample_action
        # takes in model, obs, goal  (all will have shape (B, H, ...) -> action (B, ...)
        self._policy_model_forward_fn = get_with_default(params, "policy_model_forward_fn",
                                                   lambda model, obs, goal, **kwargs: model.forward(
                                                       (obs & goal).leaf_apply(lambda arr: arr[:, 0]), ))
        self._provide_policy_env_in_forward = get_with_default(params, "provide_policy_env_in_forward", False)
        self._timeout = get_with_default(params, "timeout", 0)

        if self._timeout > 0:
            logger.debug(f"{'Goal ' if self._is_goal else ''}Policy with timeout = {self._timeout}")

    def _init_setup(self):
        self._step_counter = 0

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, **kwargs):
        self._step_counter = 0

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
        action_output = self._policy_model_forward_fn(model, observation, goal, **kwargs)
        action_output = self._postproc_fn(model, observation, goal, action_output)
        intersection = [key for key in action_output.keys() if key in self._out_names]
        self._env_spec.clip(action_output, intersection, object_safe=True)
        self._step_counter += 1
        return action_output

    def get_random_action(self, model, observation, goal, **kwargs):
        self._step_counter += 1
        return super(BasicPolicy, self).get_random_action(model, observation, goal, **kwargs)

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        # if env_memory is passed in (see GoalTrainer), this will return True if we have timed out (and timeout > 0).
        return 0 < self._timeout <= self._step_counter
