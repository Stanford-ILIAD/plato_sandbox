"""
This is a policy that integrates feedback from an observation with "commanded" actions from another policy

1. using some underlying policy, compute (model, obs, goal) -> action
2. based on particular reactive controller, modifies action with knowledge in the obs & goal
3. returns new action

An example of this can be found in force_reactive_policy.py
"""
from sbrl.policies.policy import Policy


class ReactivePolicy(Policy):

    def _init_params_to_attrs(self, params):
        # self.sample_action = params.sample_action
        # takes in model, obs, goal  (all will have shape (B, H, ...) -> action (B, ...)

        self._base_policy = params.base_policy.cls(params.base_policy.params, self._env_spec, self._file_manager)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def modify_action_from_obs(self, action, model, observation, goal, **kwargs):
        return action

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)
        note: kwargs are for model forward fn

        :return action: AttrDict (B x ...)
        """

        base_action = self._base_policy.get_action(model, observation, goal, **kwargs)

        action_output = self.modify_action_from_obs(base_action, model, observation, goal, **kwargs)

        action_output.base_action = base_action  # setting nested AttrDict in case ppl want to know the base policy

        intersection = [key for key in action_output.keys() if key in self._env_spec.action_names]
        self._env_spec.clip(action_output, intersection)  # clips only these keys (action output can have more)
        return action_output
