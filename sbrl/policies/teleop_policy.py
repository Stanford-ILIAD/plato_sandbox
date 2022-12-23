"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""

from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.python_utils import get_with_default


class TeleopPolicy(BasicPolicy):
    def _init_params_to_attrs(self, params):
        assert self._input_handle is not None, "input handle must be specified for teleop policy!"

        self.get_teleop_model_forward_fn = params << "get_teleop_model_forward_fn"
        if self.get_teleop_model_forward_fn is None:
            # fill in with env's model forward fn.
            self._get_teleop_kwargs = get_with_default(params, "get_teleop_kwargs", {}, map_fn=dict)
            # overrides
            assert hasattr(self._env, "get_default_teleop_model_forward_fn"), "Need to implement \"get_default_teleop_model_forward_fn\" in order to run teleop policy"
            params.policy_model_forward_fn = self._env.get_default_teleop_model_forward_fn(self._input_handle, **self._get_teleop_kwargs)
        else:
            params.policy_model_forward_fn = self.get_teleop_model_forward_fn(self._env, self._input_handle)
        super(TeleopPolicy, self)._init_params_to_attrs(params)


# class FunctionalPolicy(BasicPolicy):
#
#     def _init_params_to_attrs(self, params):
#         super(FunctionalPolicy, self)._init_params_to_attrs(params)
#         self._is_terminated_fn = get_required(params, "is_terminated_fn")
#         self._reset_policy_fn = get_with_default(params, "reset_policy_fn", lambda **kwargs: None)
#         self._context = AttrDict()
#
#     def is_terminated(self, model, observation, goal, **kwargs) -> bool:
#         return self._is_terminated_fn(self._context, model, observation, goal, env=self._env, **kwargs)
#
#     def reset_policy(self, **kwargs):
#         self._reset_policy_fn(**kwargs)
