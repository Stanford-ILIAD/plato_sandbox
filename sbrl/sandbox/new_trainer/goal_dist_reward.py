import numpy as np

from sbrl.envs.env import Env
from sbrl.models.model import Model
from sbrl.sandbox.new_trainer.reward import Reward
from sbrl.utils.python_utils import get_with_default
from sbrl.utils.torch_utils import combine_after_dim, concatenate, to_numpy


class GoalDistReward(Reward):

    def _init_params_to_attrs(self, params):
        self._distance_fn = get_with_default(params, "distance_fn", lambda a, b: np.abs(a - b).mean(-1, keepdims=True))
        self._obs_names = get_with_default(params, "obs_names", self._env_spec.observation_names)
        self._goal_names = get_with_default(params, "goal_names", [f"goal/{o}" for o in self._obs_names])
        self._normalization_weights = get_with_default(params, "normalization_weights", None)
        if self._normalization_weights is not None:
            self._normalization_weights = np.array(self._normalization_weights)
        assert set(self._goal_names).issubset(self._env_spec.goal_names), f"Goal names {self._goal_names} is not a subset of spec names: {self._env_spec.goal_names}"

    def warm_start(self, model, observation, goal):
        pass

    def get_reward(self, env: Env, model: Model, observation, goal, action, next_observation, next_goal, env_memory, done, policy_done, goal_policy_done=None,  **kwargs):
        relevant_next = next_observation > self._obs_names
        relevant_goal = goal > self._goal_names

        next_obs_flat = to_numpy(concatenate(relevant_next.leaf_apply(lambda arr: combine_after_dim(arr, 1)), self._obs_names, dim=-1), check=True)
        goal_obs_flat = to_numpy(concatenate(relevant_goal.leaf_apply(lambda arr: combine_after_dim(arr, 1)), self._goal_names, dim=-1), check=True)

        if self._normalization_weights is not None:
            next_obs_flat = next_obs_flat / self._normalization_weights
            goal_obs_flat = goal_obs_flat / self._normalization_weights
        # rew = - || goal - next || ... shape = (B, 1)
        return -self._distance_fn(next_obs_flat, goal_obs_flat)
