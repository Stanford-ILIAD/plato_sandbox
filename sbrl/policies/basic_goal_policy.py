from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.python_utils import get_required


class BasicGoalPolicy(BasicPolicy):
    def _init_params_to_attrs(self, params):
        super(BasicGoalPolicy, self)._init_params_to_attrs(params)
        self.goal_ids = get_required(params, "goal_ids")
        self._goal_termination_fn = get_required(params, "goal_termination_fn")
        # denotes the types of goals (ex a push goal, a rotate goal)
        self._goal_prefix = params >> "goal_prefix"

    def is_terminated(self, model, observation, goal, **kwargs):
        term = self._goal_termination_fn(model, observation, goal, self._goal_prefix, **kwargs)
        return (term >> "goal_success").item()
