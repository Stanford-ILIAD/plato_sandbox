from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import to_torch


class PPOModel(Model):
    """PPO algorithm."""

    # sampling
    @staticmethod
    def get_default_actor_postproc_fn(params):
        mu_min, mu_max = list(params.MEAN_BOUNDS)
        mu_min = to_torch(mu_min)
        mu_max = to_torch(mu_max)

        def fn(inputs: AttrDict, model_outputs: AttrDict) -> AttrDict:
            mu = model_outputs.action_dist
            mmn = mu_min.to(mu.device)
            mmx = mu_max.to(mu.device)

            scaled_mu = mmn + 0.5 * (mmx - mmn) * (1 + mu)
            return AttrDict(
                action_dist=scaled_mu
            )
        return fn

    @staticmethod
    def get_default_policy_model_forward_fn(params):
        def forward_fn(ppo_model: PPOModel, observation: AttrDict, goal: AttrDict):
            # horizon = 1, extract the first horizon element
            outputs = ppo_model.forward(observation.leaf_apply(lambda arr: arr[:, 0]), sample=params.SAMPLE_ACTION)
            return outputs
        return forward_fn

    def _init_params_to_attrs(self, params):
        keys = list(params.leaf_keys())
        assert 'env_reward_fn' in keys, "PPO requires an environment reward function"
        self.set_fn("_env_reward_fn", params.env_reward_fn, Callable[[AttrDict, AttrDict], torch.Tensor])

        self.ppo_clip = params.ppo_clip
        self.ppo_value_coef = params.ppo_value_coefficient
        self.ppo_entropy_coef = params.ppo_entropy_coefficient

        # actor critic
        self.actor = params.actor.cls(params.actor.params, self._env_spec, self._dataset_train).to(self.device)
        self.critic = params.critic.cls(params.critic.params, self._env_spec, self._dataset_train).to(self.device)

        self.action_log_std = nn.Parameter(torch.zeros((1, self.env_spec.dim(self.env_spec.action_names))).to(self.device), requires_grad=True)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, sample=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn
        :param sample: (bool) sample from action_dist or return mean

        :return model_outputs: (AttrDict)  (B x ...)
        """
        output = self.actor.forward(inputs, )
        log_std = self.action_log_std.expand_as(output.action_dist)
        output.action_dist = torch.distributions.Normal(loc=output.action_dist, scale=log_std.exp())

        critic_output = self.critic.forward(inputs, )
        output.combine(critic_output)
        if sample:
            output.action = output.action_dist.sample()
        else:
            output.action = output.action_dist.mean
        output.action_log_prob = output.action_dist.log_prob(output.action).sum(-1, keepdim=True)
        # output should contain:
        #   - action_dist
        #   - action_log_prob
        #   - action
        #   - value
        return output

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        inputs = inputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1
        outputs = outputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1
        model_outputs = self.forward(inputs)

        # P(old_action | new_policy)
        action_log_prob = model_outputs.action_dist.log_prob(inputs.action).sum(-1, keepdim=True)
        action_entropy = model_outputs.action_dist.entropy().sum(-1).mean()
        value = model_outputs.value

        old_action_log_prob = outputs.action_log_prob
        old_advantage = outputs.advantage
        future_value = outputs.future_value

        assert action_log_prob.shape == value.shape

        ratio = (action_log_prob - old_action_log_prob).exp()

        surr1 = ratio * old_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * old_advantage

        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(future_value, value)

        loss = action_loss + self.ppo_value_coef * value_loss - self.ppo_entropy_coef * action_entropy

        if writer is not None:
            writer.add_scalar(writer_prefix + "loss", loss.item(), i)
            writer.add_scalar(writer_prefix + "action_loss", action_loss.item(), i)
            writer.add_scalar(writer_prefix + "value_loss", value_loss.item(), i)
            writer.add_scalar(writer_prefix + "action_entropy_loss", action_entropy.item(), i)
            writer.add_scalar(writer_prefix + "advantage", outputs.advantage.mean().item(), i)
            writer.add_scalar(writer_prefix + "future_value", outputs.future_value.mean().item(), i)
            writer.add_scalar(writer_prefix + "action_log_std_avg", self.action_log_std.mean().item(), i)

        return loss, action_loss, value_loss, action_entropy

    def load_statistics(self, dd=None):
        self.actor.load_statistics(dd)
        self.critic.load_statistics(dd)
