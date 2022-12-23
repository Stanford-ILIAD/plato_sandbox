from typing import Callable

import torch
import torch.nn.functional as F

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.file_utils import prepend_to_base_name
from sbrl.utils.python_utils import AttrDict, get_with_default


class DDPGActorCriticModel(Model):
    """DDPG algorithm."""

    def _init_params_to_attrs(self, params):
        self.discount = params.discount

        # critic & critic target
        self.critic = params.critic.cls(params.critic.params, self._env_spec, self._dataset_train).to(self.device)
        self.critic_target = params.critic.cls(params.critic.params, self._env_spec, self._dataset_train).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # actor and actor target
        self.actor = params.actor.cls(params.actor.params, self._env_spec, self._dataset_train).to(self.device)
        self.actor_target = params.actor.cls(params.actor.params, self._env_spec, self._dataset_train).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.action_name = get_with_default(params, "action_name", "action", map_fn=str)
        self.reward_name = get_with_default(params, "reward_name", "reward", map_fn=str)
        # this turns off future Q estimation, and turns Q() into a reward predictor
        self.single_step_q = get_with_default(params, "single_step_q", False)

        assert self.action_name in self.env_spec.action_names and len(self.env_spec.action_names) >= 1
        low, high = self.env_spec.limits([self.action_name])
        low = torch.from_numpy(low[0]).to(self.device)
        high = torch.from_numpy(high[0]).to(self.device)
        # scales the squashed normal distribution
        self._action_loc = 0.5 * (low + high)
        self._action_scale = 0.5 * (high - low)
        assert self._action_scale.count_nonzero() == self._action_scale.numel()

        # should take [actor output] and compute an additional loss to add to -Q(a), differentiable in a
        self.actor_additional_reward_fn = get_with_default(params, "actor_additional_reward_fn", None)
        self.critic_additional_reward_fn = get_with_default(params, "critic_additional_reward_fn", None)

        # outputting actions correctly
        self._action_rescale = lambda x: self._action_loc + x * self._action_scale
        self._action_rescale_inv = lambda x: (x - self._action_loc) / self._action_scale

        self._action_noise = get_with_default(params, "action_noise", default=0.1)  # relative to -1 to 1
        if self._action_noise > 0:
            logger.debug("DDPG action noise: %f" % self._action_noise)

        self._recurrent_critic = get_with_default(params, "recurrent_critic", False)
        self._alpha_q_over_horizon = get_with_default(params, "alpha_q_over_horizon", 0.)
        assert self._alpha_q_over_horizon == 0. or self._recurrent_critic, "Recurrent critic required to enable q averaging"

        if self._alpha_q_over_horizon > 0:
            logger.debug("DDPG nonzero exp averaging for Q: %f" % self._alpha_q_over_horizon)

        keys = list(params.leaf_keys())
        if 'env_reward_fn' in keys:
            logger.debug("DDPG using an environment reward function")
            self.set_fn("_env_reward_fn", params.env_reward_fn, Callable[[AttrDict, AttrDict], torch.Tensor])

        if not self.single_step_q:
            # this is because we need to compute the next inputs easily
            for key in self.env_spec.observation_names:
                # keys we don't auto forward, we must have
                assert prepend_to_base_name(key, "next_") in self.env_spec.output_observation_names, "next_{} must be present in output names (EnvSpec)"

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        super().eval()
        self.train(training=False)

    # @abstract.overrides
    def forward(self, inputs, sample=False, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param sample: bool, if true will add noise to the actions

        :return model_outputs: (AttrDict)  (B x ...)
        """
        # will contain action, deterministic, scales -1 to 1
        actor_out = self.actor.forward(inputs, **kwargs)
        act = actor_out << self.action_name
        if act is not None:
            if torch.any(act.abs() > 1):
                logger.warn("Actor model output is outside of bounds...")
            if sample:
                act = torch.clip(act + self._action_noise * torch.randn_like(act), -1, 1)
            actor_out[self.action_name] = self._action_rescale(act)
        return actor_out

    def load_statistics(self, dd=None):
        assert dd is None, "Non-none not implemented"
        logger.debug("-> Loading statistics for AC model's dataset")
        self.actor.load_statistics()
        self.actor_target.load_statistics()
        self.critic.load_statistics()
        self.critic_target.load_statistics()

    def load_target_state_dicts(self):
        # copies parameters to the targets (e.g. on init)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    # don't call this and then do backprop!! graphs are not properly retained for some reason.
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
        raise NotImplementedError

    def critic_loss(self, inputs, outputs, not_done=None, reward=None, i=0, writer=None, writer_prefix="", training=False):
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
        not_done = not_done[..., None]  # expand for broadcasting to (B x H x 1)
        H = not_done.shape[1]

        if reward is None:
            if self.reward_name not in outputs.leaf_keys():
                reward = self._env_reward_fn(inputs, outputs)  # torch tensor output
            else:
                reward = outputs[self.reward_name]
        # else:
        #     reward = reward[:, 0]

        if self.critic_additional_reward_fn is not None:
            add_reward = self.critic_additional_reward_fn(inputs, outputs)
            assert list(add_reward.shape) == list(reward.shape)
            reward = reward + add_reward

        if self.single_step_q:
            target_q = reward
        else:
            # getting next inputs from current inputs, outputs TODO function for this
            next_inputs = AttrDict()
            for key in self.env_spec.observation_names:
                next_key = prepend_to_base_name(key, "next_")
                # find it in inputs
                if next_key in outputs.leaf_keys():
                    next_inputs[key] = outputs[next_key]
                elif key in inputs.leaf_keys():
                    # TODO this could lead to bugs where the outputs are not specified
                    next_inputs[key] = inputs[key]  # just copy it over if it's not there
                else:
                    raise NotImplementedError

            # recurrent critics are estimated this way
            if self._recurrent_critic:
                for key in next_inputs.leaf_keys():
                    # first inputs get prepended (new size is H + 1)
                    next_inputs[key] = torch.cat([inputs[key][:, :1], next_inputs[key]], dim=1)

            # copy over param keys
            for key in self.env_spec.param_names:
                assert key in inputs.leaf_keys(), "param key %s is missing from inputs"
                next_inputs[key] = inputs[key]

            # now we have (inputs, outputs, next_inputs, not_done, reward)

            """ JQ computation: Q(s',a') & a* using pi' """

            next_actor_target_output = self.actor_target.forward(next_inputs)
            next_inputs[self.action_name] = self._action_rescale(next_actor_target_output[self.action_name])  # add a' to inputs
            targ_out = self.critic_target(next_inputs)  # Q_t(s',a')
            target_v = torch.min(targ_out.q1, targ_out.q2)

            if self._recurrent_critic:
                # ignore q value from first time step, since this was appended on for consistency
                target_v = target_v[:, 1:]

            if not target_v.isfinite().all():
                logger.warn("Encountered infinite / NaN logprob")
                import ipdb; ipdb.set_trace()

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

        target_q = target_q.detach()

        # get current Q estimates
        curr_out = self.critic.forward(inputs)

        if not curr_out.q1.isfinite().all() or not curr_out.q2.isfinite().all():
            logger.warn("Encountered infinite / NaN logprob")
            import ipdb; ipdb.set_trace()

        critic_loss = F.mse_loss(curr_out.q1, target_q) + F.mse_loss(
            curr_out.q2, target_q)

        if writer:
            writer.add_scalar(writer_prefix + 'critic_loss', critic_loss.item(), i)
            writer.add_scalar(writer_prefix + 'mean_reward', reward.mean().item(), i)
            writer.add_scalar(writer_prefix + 'max_reward', reward.max().item(), i)
            writer.add_scalar(writer_prefix + 'mean_target_q', target_q.mean().item(), i)
            writer.add_scalar(writer_prefix + 'max_target_q', target_q.max().item(), i)
            if self.critic_additional_reward_fn is not None:
                writer.add_scalar(writer_prefix + 'mean_add_reward', add_reward.mean().item(), i)
                writer.add_scalar(writer_prefix + 'max_add_reward', add_reward.max().item(), i)

        return critic_loss

    def actor_loss(self, inputs, i=0, writer=None, writer_prefix="", training=False):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        # inputs = inputs.leaf_apply(lambda arr: arr[:, 0])  # horizon = 1

        actor_output = self.actor.forward(inputs)
        if not actor_output[self.action_name].isfinite().all():
            logger.warn("Encountered infinite / NaN logprob")
            import ipdb; ipdb.set_trace()
        new_inputs = inputs.copy()
        new_inputs[self.action_name] = self._action_rescale(actor_output[self.action_name])  # scale properly

        critic_out = self.critic.forward(new_inputs)

        actor_q = torch.min(critic_out.q1, critic_out.q2)
        # NOTE additional loss will affect the gradients / stability, don't be dumb
        if self.actor_additional_reward_fn is not None:
            add_reward = self.actor_additional_reward_fn(new_inputs, actor_output)
            if writer:
                writer.add_scalar(writer_prefix + 'actor_additional_loss', -add_reward.mean().item(), i)
            actor_q = actor_q + add_reward
        actor_loss = (-actor_q).mean()

        if writer:
            writer.add_scalar(writer_prefix + 'actor_loss', actor_loss.item(), i)

        return actor_loss

    @staticmethod
    def get_default_policy_model_forward_fn(params):
        sample_action = get_with_default(params, "ACTOR_SAMPLE_ACTION", False)
        def forward_fn(ac_model: DDPGActorCriticModel, observation: AttrDict, goal: AttrDict, *args, **kwargs):
            # horizon = 1, extract the first horizon element
            outputs = ac_model.forward(observation.leaf_apply(lambda arr: arr[:, 0]), sample=sample_action, **kwargs)
            return outputs
        return forward_fn

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[DDPG]")
        for p in self.parameters(recurse=False):
            print_fn(prefix + "[DDPG] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

        if hasattr(self.actor, "print_parameters"):
            print_fn(prefix + "\t[Actor]")
            self.actor.print_parameters(prefix=prefix + "\t\t", print_fn=print_fn)
        else:
            for p in self.actor.parameters():
                print_fn(prefix + "\t\t[Actor] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

        if hasattr(self.actor_target, "print_parameters"):
            print_fn(prefix + "\t[ActorTarget]")
            self.actor_target.print_parameters(prefix=prefix + "\t\t", print_fn=print_fn)
        else:
            for p in self.actor_target.parameters():
                print_fn(prefix + "\t\t[ActorTarget] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

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
                print_fn(prefix + "\t\t[CriticTarget] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))
