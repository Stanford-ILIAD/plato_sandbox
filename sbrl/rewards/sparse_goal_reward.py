import numpy as np

from sbrl.rewards.reward import Reward
from sbrl.utils.python_utils import get_with_default, AttrDict


class SparseGoalReward(Reward):
    def _init_params_to_attrs(self, params):
        self._success_fn = params >> "success_fn"
        self._goal_prefix = params >> "goal_prefix"
        self._success_rew = get_with_default(params, "success_rew", 1.0)
        self._failure_rew = get_with_default(params, "failure_rew", 0.0)
        self._additional_reward_names = get_with_default(params, "additional_reward_names", None)

    def get_reward(self, env, model, obs, goal, action, next_obs, next_goal,
                   env_memory, policy_done, goal_policy_done, done, env_step):
        success_d = self._success_fn(model, next_obs, goal, self._goal_prefix)
        success_bool = (success_d >> "goal_success").item()
        if success_bool:
            rew = np.array([[self._success_rew]])
        else:
            rew = np.array([[self._failure_rew]])
        ret = AttrDict(reward=rew)

        if self._additional_reward_names is not None:
            for name in self._additional_reward_names:
                ret[name] = success_d >> name
        return ret
