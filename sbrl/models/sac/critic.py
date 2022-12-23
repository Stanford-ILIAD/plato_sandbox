import numpy as np

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict, get_with_default


class DoubleQCritic(Model):
    """Critic network, employes double Q-learning."""

    def _init_params_to_attrs(self, params):
        # obs_dim + action_dim -> 1
        self.Q1 = params.q1.cls(params.q1.params, self._env_spec, self._dataset_train).to(self.device)
        self.Q2 = params.q1.cls(params.q1.params, self._env_spec, self._dataset_train).to(self.device)

        self.inputs = self.Q1.inputs
        self.output_name = params.get("output_name", "qval")

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=False, postproc=False, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        q1 = self.Q1(inputs)
        q2 = self.Q2(inputs)

        return AttrDict(q1=q1.qval, q2=q2.qval)

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """

        q1 = self.Q1(inputs)
        q2 = self.Q2(inputs)

        loss_q1 = self._loss_fn(inputs, outputs, q1)
        loss_q2 = self._loss_fn(inputs, outputs, q2)

        if writer is not None:
            writer.add_scalar(writer_prefix + "q1_mean", q1[self.output_name].mean().item(), i)
            writer.add_scalar(writer_prefix + "q2_mean", q2[self.output_name].mean().item(), i)
            writer.add_scalar(writer_prefix + "q1_loss", loss_q1.item(), i)
            writer.add_scalar(writer_prefix + "q2_loss", loss_q2.item(), i)

        return loss_q1 + loss_q2

    def load_statistics(self, dd=None):
        logger.debug("-> Loading statistics for DoubleQCritic model's dataset")
        self.Q1.load_statistics(dd)
        self.Q2.load_statistics(dd)


class REDQCritic(Model):
    """Critic network, employes REDQ style Q-learning, using ensemble size N with per-step min targets M"""

    def _init_params_to_attrs(self, params):
        # obs_dim + action_dim -> 1
        self.M = get_with_default(params, "num_targets", 2)
        self.N = get_with_default(params, "num_ensemble", 10)
        assert self.N >= self.M, f"ensemble size = {self.N} but using num targets = {self.M}"

        # instantiate N identical Q's
        for i in range(1, self.N + 1):
            setattr(self, f"Q{i}", params.q1.cls(params.q1.params, self._env_spec, self._dataset_train).to(self.device))

        self.inputs = self.Q1.inputs
        self.output_name = params.get("output_name", "qval")

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=False, postproc=False, q_idxs=None, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        if q_idxs is None:
            q_idxs = np.random.choice(self.N, self.M, replace=False)
        assert max(q_idxs) <= self.N - 1 and min(q_idxs) >= 0

        out = AttrDict()
        for i, idx in enumerate(q_idxs):
            out[f"q{i+1}"] = getattr(self, f"Q{idx+1}")(inputs).qval

        return out

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        q_idxs = np.random.choice(self.N, self.M, replace=False)
        loss = 0
        for j, idx in enumerate(q_idxs):
            qj = getattr(self, f"Q{idx+1}")(inputs).qval
            loss_qj = self._loss_fn(inputs, outputs, qj)
            loss += loss_qj

            if writer is not None:
                writer.add_scalar(writer_prefix + f"q{j+1}_mean", (qj >> self.output_name).mean().item(), i)
                writer.add_scalar(writer_prefix + f"q{j+1}_loss", loss_qj.item(), i)

        return loss

    def load_statistics(self, dd=None):
        logger.debug("-> Loading statistics for REDQCritic model's dataset")

        for i in range(1, self.N + 1):
            getattr(self, f"Q{i}").load_statistics(dd)
