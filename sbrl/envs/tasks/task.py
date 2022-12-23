"""
Tasks operate as an interface for an agent that doesn't directly interact with the environment.
It follows a similar interface as an environment, with additional termination conditions and reward functions.
should work with batches.

Also Provides clean torch abstraction to a numpy env.
"""
import torch

from sbrl.envs.env import Env
from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict as d, get_with_default
from sbrl.utils.torch_utils import to_torch


class Task(torch.nn.Module):
    def __init__(self, params: d, env: Env = None):
        super(Task, self).__init__()
        # sub module of the task.
        self._env = env
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params: d):
        self._step_generate_goal = get_with_default(params, "step_generate_goal", True)
        self._step_override_reward = get_with_default(params, "step_override_reward", False)

    def bind(self, env: Env):
        self._env = env

    def is_terminated(self, model: Model, obs: d, goal: d, **kwargs):
        """
        Is the task finished, independently of the env.

        :param model: Model, optionally call this to come up with goals
        :param obs: AttrDict (B x D)
        :param goal: AttrDict (B x D)
        :return: (B,)
        """

        return torch.zeros(obs.get_one().shape[0], dtype=torch.bool, device=model.device)

    def generate_goal(self, model: Model, obs: d, env_goal: d, **kwargs):
        """
        This is where you would use the model to generate a goal for the policy

        :param model: Model, optionally call this to come up with goals
        :param obs: AttrDict (B x D), observation to use to generate the goal
        :return:
        """
        return env_goal

    def get_reward(self, model: Model, obs: d, goal: d, done, obs_history: d = None, **kwargs):
        """
        Compute rewards, for example on each step

        :param model: Model
        :param obs: AttrDict (B x D)
        :param goal: AttrDict (B x D)
        :param done: (B,)
        :param obs_history: (B x H x D) TODO
        :return: rew: (B, H) or (B, 1)
        """
        return torch.zeros(*obs.get_one().shape[:2], dtype=torch.bool, device=model.device)

    def step(self, action: d, task_model: Model, env=None, obs_history: d = None, **kwargs):
        """
        Convenient env wrapper, which calls step and process_step.
        :param action:
        :param task_model:
        :param obs_history: (B x H x D), None is allowed
        :param kwargs:
        :return:
        """
        # take an action
        if env is None:
            env = self.env
        next_obs, next_goal, done = env.step(action)
        return self.process_step(task_model, next_obs, next_goal, done, obs_history=obs_history, **kwargs)

    def process_step(self, task_model: Model, next_obs: d, next_goal: d, done, obs_history: d = None, **kwargs):
        """
        Process the outputs of step here, to generate new goals & new rewards, and evaluate termination conditions.

        :param task_model: Model specific to task (e.g. reward model)
        :param next_obs:
        :param next_goal:
        :param done:
        :param obs_history: (B x H x D), None is allowed
        :param kwargs:
        :return: (obs, goal, done)
        """
        gg_kwargs = kwargs['generate_goal'] if 'generate_goal' in kwargs.keys() else {}
        term_kwargs = kwargs['is_terminated'] if 'is_terminated' in kwargs.keys() else {}
        rew_kwargs = kwargs['get_reward'] if 'get_reward' in kwargs.keys() else {}

        # torch conversion
        next_obs = next_obs.leaf_apply(lambda arr: to_torch(arr, device=task_model.device, check=True))
        next_goal = next_goal.leaf_apply(lambda arr: to_torch(arr, device=task_model.device, check=True))
        done = to_torch(done, device=task_model.device, check=True)

        # generate goals
        if self._step_generate_goal:
            next_goal = self.generate_goal(task_model, next_obs, next_goal, **gg_kwargs)

        # check if done
        done = torch.logical_or(done, self.is_terminated(task_model, next_obs, next_goal, **term_kwargs))

        # env already outputs a reward, add ours to this, and
        rew = self.get_reward(task_model, next_obs, next_goal, done, obs_history=obs_history, **rew_kwargs)
        if self._step_override_reward or 'reward' not in next_obs.leaf_keys():
            next_obs.reward = rew
        else:
            # if not overriding, we add to the existing reward
            next_obs.reward += rew

        return next_obs, next_goal, done

    def reset_task(self, task_model: Model, obs: d, env_goal: d, **kwargs):
        # computes the true goal. should be called after env.reset.
        return obs, self.generate_goal(task_model, obs, env_goal, **kwargs)

    @property
    def env(self):
        return self._env
