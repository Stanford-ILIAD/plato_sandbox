"""
The trainer for SAC models, uses 2 optimizers
"""

import numpy as np
from torch import nn

from sbrl.experiments import logger
from sbrl.models.sac.sac import SACModel
from sbrl.trainers.trainer import Trainer
from sbrl.utils.python_utils import timeit, AttrDict
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import soft_update_params


class TrainerSAC(Trainer):
    # @abstract.overrides
    def _init_parameters(self, params):
        self._critic_tau = params.critic_tau
        self._actor_update_every_n_steps = params.actor_update_every_n_train_steps
        self._critic_target_update_every_n_steps = params.critic_target_update_every_n_train_steps

        self._sac_model = self._model

    # @abstract.overrides
    def _init_optimizers(self, params):
        assert (hasattr(self._sac_model, "actor"))
        assert (hasattr(self._sac_model, "critic"))
        assert (hasattr(self._sac_model, "critic_target"))
        assert (hasattr(self._sac_model, "log_alpha"))

        assert isinstance(self._sac_model, SACModel)

        self._actor = self._sac_model.actor
        self._critic = self._sac_model.critic
        self._critic_target = self._sac_model.critic_target
        self._log_alpha = self._sac_model.log_alpha

        self._actor_optimizer = params.actor_optimizer(self._actor.parameters())
        self._critic_optimizer = params.critic_optimizer(self._critic.parameters())
        self._log_alpha_optimizer = params.log_alpha_optimizer([self._log_alpha])

    def _additional_train_step(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True):
        # any unhandled actions
        pass

    # @abstract.overrides
    def _train_step(self):
        if len(self._dataset_train) == 0:
            logger.warn("Skipping training step since dataset is empty.")
            return
        with timeit('base'):
            if not isinstance(self._current_train_loss, np.ndarray):
                self._current_train_loss = np.array([np.inf, np.inf, np.inf])

            sw = None  # denotes skip writing
            if is_next_cycle(self._current_train_step, self._write_to_tensorboard_every_n):
                sw = self._summary_writer

            with timeit('get_batch'):
                res = self._dataset_train.get_batch(torch_device=self._model.device)
                inputs, outputs = res[:2]
                meta = res[2] if len(res) == 3 else AttrDict()
            self._model.train()
            with timeit('critic_loss'):
                critic_loss = self._sac_model.critic_loss(inputs, outputs, i=self._current_step,
                                                          writer=sw,
                                                          writer_prefix="train/", training=True, meta=meta)

            with timeit('critic_optimizer'):
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                if self._max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
                self._critic_optimizer.step()
                self._current_train_loss[0] = critic_loss.item()

            if is_next_cycle(self._current_train_step, self._actor_update_every_n_steps):
                with timeit('actor_alpha_loss'):
                    actor_loss, alpha_loss = self._sac_model.actor_and_alpha_loss(inputs, i=self._current_step,
                                                                                  writer=sw,
                                                                                  writer_prefix="train/", training=True, meta=meta)
                with timeit('actor_optimizer'):
                    # actor loss is not none with some frequency defined above
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if sw is not None:
                        sw.add_scalar("train/actor_grad_norm", param_grad_sum(self._actor.parameters()),
                                      self._current_step)
                    if self._max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
                    self._actor_optimizer.step()
                    self._current_train_loss[1] = actor_loss.item()

                    # checking for learnable temperature
                    if alpha_loss is not None:
                        with timeit('alpha_optimizer'):
                            self._log_alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            self._log_alpha_optimizer.step()
                            self._current_train_loss[2] = alpha_loss.item()

            if is_next_cycle(self._current_train_step, self._critic_target_update_every_n_steps):
                with timeit('soft_update'):
                    soft_update_params(self._critic, self._critic_target,
                                       self._critic_tau)

            self._additional_train_step(inputs, outputs, i=self._current_step, writer=sw, writer_prefix="train/", training=True)

            self._current_train_step += 1


def param_grad_sum(params):
    norm = 0.
    for p in params:
        if p.grad is not None:
            norm += (p.grad ** 2).mean()
    return norm
