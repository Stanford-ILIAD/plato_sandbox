from typing import Tuple, Callable

from sbrl.experiments import logger
from sbrl.models.sac.sac import SACModel
from sbrl.sandbox.new_trainer.optimizer import MultiOptimizer
from sbrl.utils.python_utils import AttrDict as d, timeit, get_required
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import soft_update_params


class SACOptimizer(MultiOptimizer):

    def _init_params_to_attrs(self, params: d):
        super(SACOptimizer, self)._init_params_to_attrs(params)

        self._actor_update_every_n_steps = get_required(params, "actor_update_every_n_steps")
        self._critic_target_update_every_n_steps = get_required(params, "critic_target_update_every_n_steps")
        self._critic_tau = get_required(params, "critic_tau")

        logger.debug(f"SACOptimizer: c_tau = {self._critic_tau}, ct_freq = {self._critic_target_update_every_n_steps}, actor_freq = {self._actor_update_every_n_steps}")

    def _init_setup(self):
        super(SACOptimizer, self)._init_setup()
        assert isinstance(self._model, SACModel)
        assert self._num_optimizers == 3
        self._critic_optimizer, self._actor_optimizer, self._alpha_optimizer = self._optimizers

        self._actor = self._model.actor
        self._critic = self._model.critic
        self._critic_target = self._model.critic_target
        self._log_alpha = self._model.log_alpha

    def step(self, loss, inputs, outputs, dataset_idx, meta: d = d(), i=0, ti=0, writer=None, writer_prefix="", **kwargs):
        assert isinstance(loss, Tuple) or isinstance(loss, d)

        if isinstance(loss, Tuple):
            assert len(loss) == 3, ["two losses required to optimize"]
            critic_loss, actor_loss, alpha_loss = loss
        else:
            critic_loss = loss >> "critic_loss"
            actor_loss = loss >> "actor_loss"
            alpha_loss = loss >> "alpha_loss"

        # CRITIC
        with timeit('critic_optimizer'):
            if isinstance(critic_loss, Callable):
                with timeit('critic_loss'):
                    loss.critic_loss = critic_loss = critic_loss()

            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_grad_norm(self._critic)
            self._critic_optimizer.step()

        # ACTOR & ALPHA
        if is_next_cycle(ti, self._actor_update_every_n_steps):
            with timeit('actor_optimizer'):
                if isinstance(actor_loss, Callable):
                    with timeit('actor_loss'):
                        loss.actor_loss = actor_loss = actor_loss()
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                self._clip_grad_norm(self._actor)
                self._actor_optimizer.step()

            if alpha_loss is not None:
                with timeit('alpha_optimizer'):
                    if isinstance(alpha_loss, Callable):
                        with timeit('alpha_loss'):
                            loss.alpha_loss = alpha_loss = alpha_loss()
                    self._alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self._alpha_optimizer.step()

        if is_next_cycle(ti, self._critic_target_update_every_n_steps):
            with timeit('critic_target_soft_update'):
                soft_update_params(self._critic, self._critic_target,
                                   self._critic_tau)
