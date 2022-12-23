"""
The trainer for SAC models, uses 2 optimizers
"""

import numpy as np
import torch
from torch import nn

from sbrl.experiments import logger
from sbrl.trainers.trainer import Trainer
from sbrl.utils.python_utils import timeit
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import soft_update_params


class TrainerAC(Trainer):
    # @abstract.overrides
    def _init_parameters(self, params):
        self._critic_tau = params.critic_tau
        self._actor_tau = params.actor_tau
        self._actor_update_every_n_steps = int(params.actor_update_every_n_train_steps)
        self._actor_target_update_every_n_steps = int(params.actor_target_update_every_n_steps)
        self._critic_target_update_every_n_steps = int(params.critic_target_update_every_n_train_steps)

    # @abstract.overrides
    def _init_optimizers(self, params):
        assert (hasattr(self._model, "actor"))
        assert (hasattr(self._model, "actor_target"))
        assert (hasattr(self._model, "critic"))
        assert (hasattr(self._model, "critic_target"))
        self._actor = self._model.actor
        self._critic = self._model.critic
        self._actor_target = self._model.actor_target
        self._critic_target = self._model.critic_target

        self._actor_optimizer = params.actor_optimizer(self._actor.parameters())
        self._critic_optimizer = params.critic_optimizer(self._critic.parameters())

    # @abstract.overrides
    def _train_step(self):

        if len(self._dataset_train) == 0:
            logger.warn("Skipping training step since dataset is empty.")
            return

        # (B x H x ...)
        with timeit('train/get_batch'):
            with torch.no_grad():
                res = self._dataset_train.get_batch(torch_device=self._model.device)
                inputs, outputs = res[:2]
                meta = res[2] if len(res) == 3 else AttrDict()

        with timeit('train/data_augmentation'):
            if self._train_do_data_augmentation and self.data_augmentation is not None:
                inputs, outputs = self.data_augmentation.forward()

        self._model.train()

        if not isinstance(self._current_train_loss, np.ndarray):
            self._current_train_loss = np.array([np.inf, np.inf, np.inf])

        sw = None
        if is_next_cycle(self._current_train_step, self._write_to_tensorboard_every_n):
            sw = self._summary_writer

        with timeit('train/critic_loss'):
            critic_loss = self._model.critic_loss(inputs, outputs, i=self._current_step,
                                                  writer=sw,
                                                  writer_prefix="train/", training=True, meta=meta)

        with timeit('train/critic_backprop'):
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
            self._critic_optimizer.step()
            self._current_train_loss[0] = critic_loss.item()

        if is_next_cycle(self._current_train_step, self._actor_update_every_n_steps):
            with timeit('train/actor_loss'):
                actor_loss = self._model.actor_loss(inputs, i=self._current_step,
                                                    writer=sw,
                                                    writer_prefix="train/", training=True, meta=meta)

            with timeit('train/actor_backprop'):
                # actor loss is not none with some frequency defined above
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                if self._max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
                self._actor_optimizer.step()
                self._current_train_loss[1] = actor_loss.item()

        # soft update critic
        if is_next_cycle(self._current_train_step, self._critic_target_update_every_n_steps):
            soft_update_params(self._critic, self._critic_target,
                               self._critic_tau)

        # soft update actor
        if is_next_cycle(self._current_train_step, self._actor_target_update_every_n_steps):
            soft_update_params(self._actor, self._actor_target,
                               self._actor_tau)

        # TODO schedulers
        # with timeit("train/scheduler"):
        #     if self._base_scheduler is not None:
        #         self._base_scheduler.step()

        self.additional_train_step(inputs, outputs, sw)

        with timeit("writer"):
            if self._summary_writer is not None:
                if is_next_cycle(self._current_train_step, self._write_to_tensorboard_every_n):
                    self._summary_writer.add_scalar("train_step", self._current_train_step, self._current_step)
                    for i, pg in enumerate(self._actor_optimizer.param_groups):
                        self._summary_writer.add_scalar("train/actor_learning_rate_pg_%d" % i, pg['lr'], self._current_step)
                    for i, pg in enumerate(self._critic_optimizer.param_groups):
                        self._summary_writer.add_scalar("train/critic_learning_rate_pg_%d" % i, pg['lr'], self._current_step)

        self._current_train_step += 1

    def additional_train_step(self, inputs, outputs, writer):
        pass
