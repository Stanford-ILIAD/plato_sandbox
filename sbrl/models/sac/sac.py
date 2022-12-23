from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.models.sac.critic import REDQCritic
from sbrl.utils.dist_utils import SquashedNormal
from sbrl.utils.file_utils import prepend_to_base_name
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit, get_required
from sbrl.utils.torch_utils import to_torch, get_zeroth_horizon


class SACModel(Model):
    """SAC algorithm."""

    def _init_params_to_attrs(self, params):
        self.discount = get_required(params, "discount")
        self.learnable_temperature = bool(get_required(params, "learnable_temperature"))
        self.init_temperature = get_required(params, "init_temperature")

        self.critic = params.critic.cls(params.critic.params, self._env_spec, self._dataset_train).to(self.device)
        self.critic_target = params.critic.cls(params.critic.params, self._env_spec, self._dataset_train).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ensemble of q functions supported here and in critic_loss.
        default_ensemble_size = self.critic.N if isinstance(self.critic, REDQCritic) else 2
        self.ensemble_size = get_with_default(params, "ensemble_size", default_ensemble_size)

        self.actor = params.actor.cls(params.actor.params, self._env_spec, self._dataset_train).to(self.device)
        self._actor_min_q = get_with_default(params, "actor_min_q", True)
        self.action_name = get_with_default(params, "action_name", "action", map_fn=str)

        self.inputs = get_with_default(params, "model_inputs", self.critic.inputs)

        # assert self.action_name in self.env_spec.action_names and len(self.env_spec.action_names) >= 1
        low, high = self.env_spec.limits([self.action_name])
        low = torch.from_numpy(low[0]).to(self.device)
        high = torch.from_numpy(high[0]).to(self.device)

        self.lp_dims = [-(l + 1) for l in reversed(range(len(low.shape)))]  # [... -2 -1]
        logger.debug("SAC: log_prob reduction over dims: %s" % self.lp_dims)
        # scales the squashed normal distribution
        self._action_loc = 0.5 * (low + high)
        self._action_scale = 0.5 * (high - low)
        assert self._action_scale.count_nonzero() == self._action_scale.numel()

        # outputting actions correctly
        self._action_rescale = lambda x: self._action_loc + x * self._action_scale
        self._action_rescale_inv = lambda x: (x - self._action_loc) / self._action_scale

        self._recurrent_critic = get_with_default(params, "recurrent_critic", False)
        self._alpha_q_over_horizon = get_with_default(params, "alpha_q_over_horizon", 0.)
        assert self._alpha_q_over_horizon == 0. or self._recurrent_critic, "Recurrent critic required to enable q averaging"

        if self._alpha_q_over_horizon > 0:
            logger.debug("SAC nonzero exp averaging for Q: %f" % self._alpha_q_over_horizon)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(self.init_temperature)).to(self.device),
                                      requires_grad=self.learnable_temperature)
        # set target entropy to -|A|
        self.target_entropy = -self.env_spec.dim(self.env_spec.action_names)

        keys = list(params.leaf_keys())
        if 'env_reward_fn' in keys:
            logger.debug("SAC using an environment reward function")
            self.set_fn("_env_reward_fn", params.env_reward_fn, Callable[[AttrDict, AttrDict], torch.Tensor])
        # if 'sac_advance_obs_fn' in keys:
        #     logger.debug("SAC is using the config's specified sac advance obs fn")
        #     self.set_fn("_sac_advance_obs_fn", params.sac_advance_obs_fn, Callable[[AttrDict, AttrDict, AttrDict], AttrDict])

        # TODO rewind and advance obs
        # if 'sac_advance_obs_fn' in keys and 'sac_rewind_obs_fn' in keys:
        #     logger.debug("SAC is using the config's specified sac advance/rewind obs fn")
        #     # inputs, outputs, next_action
        #     self.set_fn("_sac_advance_obs_fn", params.sac_advance_obs_fn, Callable[[AttrDict, AttrDict, AttrDict], AttrDict])
        #     # inputs, outputs,
        #     self.set_fn("_sac_rewind_obs_fn", params.sac_rewind_obs_fn, Callable[[AttrDict, AttrDict, AttrDict], AttrDict])

        self._use_nested_next = get_with_default(params, "use_nested_next", True)

        # this convention needs to be followed for SAC to work properly TODO fix this eventually
        # this is because we need to compute the next inputs easily
        for key in list(set(self.env_spec.observation_names).intersection(self.inputs)):
            # keys we don't auto forward, we must have
            if self._use_nested_next:
                assert f"next/{key}" in self.env_spec.output_observation_names, f"next/{key} must be present in output names (EnvSpec)"
            else:
                assert prepend_to_base_name(key, "next_") in self.env_spec.output_observation_names, f"{prepend_to_base_name(key, 'next_')} must be present in output names (EnvSpec)"

        self.train()
        self.critic_target.train()

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    # def train(self, training=True):
    #     self.training = training
    #     self.actor.train(training)
    #     self.critic.train(training)
    #
    # def eval(self):
    #     super().eval()
    #     self.train(training=False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # @abstract.overrides
    def forward(self, inputs, sample=False, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)

        :return model_outputs: (AttrDict)  (B x ...)
        """
        # will contain action_dist, where action dist scales -1 to 1
        actor_out = self.actor.forward(inputs, **kwargs)
        act_dist = actor_out << (self.action_name + "_dist")
        if act_dist is not None:
            if sample:
                actor_out[self.action_name] = act_dist.sample()
            else:
                actor_out[self.action_name] = act_dist.mean
            if torch.any(actor_out[self.action_name].abs() > 1):
                logger.warn("SAC: actor output is outside of bounds...")
            actor_out[self.action_name] = self._action_rescale(actor_out[self.action_name])
        return self._postproc_fn(inputs, actor_out)

    def load_statistics(self, dd=None):
        assert dd is None, "Non-none not implemented"
        logger.debug("-> Loading statistics for SAC model's dataset")
        self.actor.load_statistics()
        self.critic.load_statistics()
        self.critic_target.load_statistics()

    # don't call this and then do backprop!! graphs are not properly retained for some reason.
    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, meta: AttrDict = AttrDict(), **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """

        # this will get populated once actor_loss_promise is called, if we want to optimize.
        critic_loss = None
        actor_loss = None
        alpha_loss = None
        loss = None

        def critic_loss_promise():
            nonlocal critic_loss
            critic_loss = self.critic_loss(inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix,
                             training=training, **kwargs)
            return critic_loss

        def actor_loss_promise():
            assert critic_loss is not None
            nonlocal alpha_loss
            nonlocal actor_loss
            nonlocal loss
            actor_loss, alpha_loss = self.actor_and_alpha_loss(inputs, i=i, writer=writer, writer_prefix=writer_prefix,

                                                               training=training)
            if alpha_loss is not None:
                loss = critic_loss + alpha_loss + actor_loss
            else:
                loss = critic_loss + actor_loss
            return actor_loss

        if training:
            # should be used with the sac optimizer
            return AttrDict(
                critic_loss=critic_loss_promise,
                actor_loss=actor_loss_promise,
                alpha_loss=lambda: alpha_loss,
                loss=lambda: loss
            )
        else:
            critic_loss_promise()
            actor_loss_promise()
            return AttrDict(
                critic_loss=critic_loss,
                actor_loss=actor_loss,
                alpha_loss=alpha_loss,
                loss=critic_loss + actor_loss + alpha_loss,
            )

    def critic_loss(self, inputs, outputs, not_done=None, reward=None, i=0, writer=None, writer_prefix="",
                    training=False, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param not_done: (torch.Tensor) (B x H) will extract from outputs if this is None
        :param reward: (torch.Tensor) (B x H x 1) will extract from outputs if this is None
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        # inputs = inputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1
        # outputs = outputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1

        if not_done is None:
            not_done = outputs.done.logical_not()  # must be in outputs
        # else:
        #     not_done = not_done[:, 0]
        not_done = not_done[..., None]  # expand for broadcasting to (B x 1)
        H = not_done.shape[1]

        if reward is None:
            if "reward" not in outputs.leaf_keys():
                reward = self._env_reward_fn(inputs, outputs)  # torch tensor output
            else:
                reward = outputs.reward
        # else:
        #     reward = reward[:, 0]

        # getting next inputs from current inputs, outputs TODO function for this
        next_inputs = AttrDict()
        for key in self.inputs:
            next_key = f"next/{key}" if self._use_nested_next else prepend_to_base_name(key, "next_")
            # find it in inputs
            if next_key in outputs.leaf_keys():
                next_inputs[key] = outputs[next_key]
            elif key in inputs.leaf_keys():
                # TODO this could lead to bugs where the outputs are not specified
                next_inputs[key] = inputs[key]  # just copy it over if it's not there
            else:
                raise NotImplementedError

        # # copy over param keys & goal keys
        # for key in self.env_spec.param_names + self.env_spec.goal_names:
        #     assert key in inputs.leaf_keys(), "param key %s is missing from inputs"
        #     next_inputs[key] = inputs[key]

        if self._recurrent_critic:
            for key in next_inputs.leaf_keys():
                # first inputs get prepended (new size is H + 1)
                next_inputs[key] = torch.cat([inputs[key][:, :1], next_inputs[key]], dim=1)

        # now we have (inputs, outputs, next_inputs, not_done, reward)

        """ JQ computation: Q(s',a') & pi(a'|s') """

        with timeit("critic_loss/target_v"):
            next_actor_output = self.actor.forward(next_inputs, )
            dist = next_actor_output[self.action_name + "_dist"]
            next_action_unscaled = dist.sample()  # -1 -> 1
            log_prob = dist.log_prob(next_action_unscaled).sum(self.lp_dims)[..., None]  # pi(a'|s')
            next_inputs[self.action_name] = self._action_rescale(next_action_unscaled)  # add a' to inputs

            targ_out = self.critic_target(next_inputs)  # Q_i(s',a')
            target_v = torch.minimum(targ_out >> "q1",
                                 targ_out >> "q2") - self.alpha.detach() * log_prob

            if self._recurrent_critic:
                # ignore q value from first time step, since this was appended on for consistency
                target_v = target_v[:, 1:]

            if not target_v.isfinite().all():
                logger.warn("Encountered infinite / NaN logprob")
                import ipdb;
                ipdb.set_trace()

            future_v = not_done * target_v

            if self._alpha_q_over_horizon > 0:
                # first, Q_h = V_h+1
                target_q = future_v.clone()
                # q's are exponentially averaged (in a DP fashion)  # TODO copy to SAC
                for h in reversed(range(H)):
                    # balance out estimated value V (from network) with DP computed Q:
                    #   Q_h = r_h + gamma * ( (1-a) * V_h+1, a * Q_h+1)
                    if h < H - 1:
                        target_q[:, h] *= (1 - self._alpha_q_over_horizon)
                        target_q[:, h] += self._alpha_q_over_horizon * target_q[:, h + 1]  # DP step
                    target_q[:, h] *= self.discount
                    target_q[:, h] += reward[:, h]

            else:
                target_q = reward + self.discount * future_v
                # target_q = reward + (not_done * self.discount * target_v)

            target_q = target_q.detach()

        with timeit("critic_loss/critic_forward_and_mse"):
            # get current Q estimates (in non shuffled order, ensemble supported).
            curr_out = self.critic.forward(inputs, q_idxs=np.arange(self.ensemble_size), **kwargs)

            if not curr_out.q1.isfinite().all() or not curr_out.q2.isfinite().all():
                logger.warn("Encountered infinite / NaN logprob")
                import ipdb;
                ipdb.set_trace()

            # all critics should be mapped.
            critic_loss = 0.
            for j in range(self.ensemble_size):
                critic_loss += F.mse_loss(curr_out >> f"q{j + 1}", target_q)

        with timeit("writer"):
            if writer:
                writer.add_scalar(writer_prefix + 'critic_loss', critic_loss.item(), i)
                writer.add_scalar(writer_prefix + 'mean_reward', reward.mean().item(), i)
                writer.add_scalar(writer_prefix + 'max_reward', reward.max().item(), i)
                writer.add_scalar(writer_prefix + 'mean_target_q', target_q.mean().item(), i)
                writer.add_scalar(writer_prefix + 'max_target_q', target_q.max().item(), i)

        return critic_loss

    def actor_and_alpha_loss(self, inputs, i=0, writer=None, writer_prefix="", training=False, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        # inputs = inputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1

        with timeit("actor_loss/forward"):
            actor_output = self.actor.forward(inputs, **kwargs)
            dist = actor_output[self.action_name + "_dist"]
            action_unscaled = dist.rsample()  # -1 to 1
            log_prob = dist.log_prob(action_unscaled).sum(self.lp_dims)[..., None]  # pi(a_tilde|s)
            if not log_prob.isfinite().all():
                logger.warn("Encountered infinite / NaN logprob")
                import ipdb;
                ipdb.set_trace()
            new_inputs = inputs.leaf_copy()
            new_inputs[self.action_name] = self._action_rescale(action_unscaled)  # scale properly

        with timeit("actor_loss/critic_forward"):
            critic_out = self.critic.forward(new_inputs, q_idxs=np.arange(self.ensemble_size))

        if self._actor_min_q:
            # min q
            actor_q = critic_out >> 'q1'
            for j in range(1, self.ensemble_size):
                actor_q = torch.minimum(actor_q, critic_out >> f'q{j+1}')
        else:
            # mean q
            actor_q = torch.stack([critic_out >> f'q{j+1}' for j in range(self.ensemble_size)], dim=-1).mean(-1)
        actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()

        with timeit("writer"):
            if writer:
                writer.add_scalar(writer_prefix + 'actor_loss', actor_loss.item(), i)
                writer.add_scalar(writer_prefix + 'actor_target_entropy', self.target_entropy.item(), i)
                writer.add_scalar(writer_prefix + 'actor_entropy', -log_prob.mean().item(), i)
                writer.add_scalar(writer_prefix + 'actor_q', actor_q.mean().item(), i)

        alpha_loss = None
        if self.learnable_temperature:
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()

            with timeit("writer"):
                if writer:
                    writer.add_scalar(writer_prefix + 'alpha_loss', alpha_loss.item(), i)
                    writer.add_scalar(writer_prefix + 'alpha_value', self.alpha.item(), i)

        return actor_loss, alpha_loss

    @staticmethod
    def get_default_actor_postproc_fn(params):
        log_std_min, log_std_max = list(params.LOG_STD_BOUNDS)
        device = params.get("DEVICE", "cpu")
        action_name = params.get("action_name", "action")
        log_std_min = to_torch(log_std_min, device=device)
        log_std_max = to_torch(log_std_max, device=device)

        # tanh_mu_bound = params.get("ACTION_SCALE_TANH", 1.)
        # tanh_mu_bound = to_torch(tanh_mu_bound, device=device)

        def forward_fn(inputs: AttrDict, model_outputs: AttrDict) -> AttrDict:
            mu, log_std = model_outputs[action_name + "_dist"].chunk(2, dim=-1)

            # constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
            dist = SquashedNormal(mu, std)  # THIS DISTRIBUTION must be -1 to 1 "scaled," SAC handles scaling
            return AttrDict({action_name + "_dist": dist})

        return forward_fn

    @staticmethod
    def get_default_policy_model_forward_fn(params):
        def forward_fn(sac_model: SACModel, observation: AttrDict, goal: AttrDict, *args, **kwargs):
            if "greedy_action" in kwargs and kwargs["greedy_action"]:
                sample = False
            else:
                sample = params >> "SAMPLE_ACTION"
            outputs = sac_model.forward(observation & goal, sample=sample, **kwargs)
            # horizon = 1, extract the first horizon element
            return outputs.leaf_apply(get_zeroth_horizon)

        return forward_fn

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[SAC]")
        for p in self.parameters(recurse=False):
            print_fn(prefix + "[SAC] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

        if hasattr(self.actor, "print_parameters"):
            print_fn(prefix + "\t[Actor]")
            self.actor.print_parameters(prefix=prefix + "\t\t", print_fn=print_fn)
        else:
            for p in self.actor.parameters():
                print_fn(prefix + "\t\t[Actor] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

        if hasattr(self.critic, "print_parameters"):
            print_fn(prefix + "\t[Critic]")
            self.critic.print_parameters(prefix=prefix + "\t\t", print_fn=print_fn)
        else:
            for p in self.critic.parameters():
                print_fn(prefix + "\t\t[Critic] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

        if hasattr(self.critic_target, "print_parameters"):
            print_fn(prefix + "\t[CriticTarget]")
            self.critic_target.print_parameters(prefix=prefix + "\t\t", print_fn=print_fn)
        else:
            for p in self.critic_target.parameters():
                print_fn(
                    prefix + "\t\t[CriticTarget] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))
