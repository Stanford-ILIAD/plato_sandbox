import numpy as np

from sbrl.rewards.reward import Reward
from sbrl.utils.python_utils import get_required, get_with_default, AttrDict
from sbrl.utils.torch_utils import torch_to_numpy


class SparseGoalBCReward(Reward):
    def _init_params_to_attrs(self, params):
        self._sparse_reward = params.sparse_reward.cls(
            params.sparse_reward.params, params["env_spec"], params["env"]
        )
        self._bc_model = params.bc_model.cls(
            params.bc_model.params, params["env_spec"], None
        )
        if params.bc_model_file:
            self._bc_model.restore_from_file(params.bc_model_file)

        self._bc_policy = params.bc_policy.cls(
            params.bc_policy.params, params["env_spec"], None, params["env"]
        )
        self._bc_pol_prepare_inputs_fn = get_required(params, "bc_pol_prepare_inputs_fn")
        self._bc_reward_fn = get_required(params, "bc_reward_fn")
        self._bc_weight_annealer = params >> "bc_weight_annealer"
        self._additional_reward_names = get_with_default(params, "additional_reward_names", None)
        self._bc_reward_only = get_required(params, "bc_reward_only")

    def get_reward(self, env, model, obs, goal, action, next_obs, next_goal,
                   env_memory, policy_done, goal_policy_done, done, env_step):
        sparse_reward_d = self._sparse_reward.get_reward(env, model, obs, goal, action, next_obs, next_goal,
                                                          env_memory, policy_done, goal_policy_done, done, env_step)
        sparse_reward = sparse_reward_d >> "reward"
        bc_reward = self.get_bc_reward(env, obs, goal, action, model)

        bc_weight = 1.0 if (self._bc_weight_annealer is None or self._bc_reward_only) else self._bc_weight_annealer.get_val(env_step)
        reward = bc_reward if self._bc_reward_only else bc_weight * bc_reward + sparse_reward
        ret = AttrDict(
            bc_reward=bc_reward,
            sparse_reward=sparse_reward,
            reward=reward,
            bc_weight=np.array([[bc_weight]])
        )
        if self._additional_reward_names is not None:
            for name in self._additional_reward_names:
                ret[name] = sparse_reward_d >> name

        return ret

    def get_bc_reward(self, env, obs, goal, action, model):
        bc_pol_obs = self._bc_pol_prepare_inputs_fn(env, self._bc_model, self._bc_policy, obs, goal)
        bc_pol_out = self._bc_policy.get_action(self._bc_model, bc_pol_obs, goal).leaf_apply(torch_to_numpy)
        bc_reward = self._bc_reward_fn(action, bc_pol_out, model, self._bc_model)
        return bc_reward

    def reset_reward(self):
        self._bc_policy.reset_policy()
