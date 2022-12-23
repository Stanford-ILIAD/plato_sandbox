"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
from typing import Callable

from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict, get_required


class FunctionModel(Model):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self.inputs = params << "model_inputs"
        self.outputs = params << "model_outputs"

        self.set_fn("_forward_fn", get_required(params, "forward_fn"), Callable[[FunctionModel, AttrDict], AttrDict])

    # @abstract.overrides
    def _init_setup(self):
        pass

    # @abstract.overrides
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        inputs = inputs.copy()
        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs)

        if preproc:
            inputs = self._preproc_fn(inputs)

        out = self._forward_fn(self, inputs, **kwargs)

        return self._postproc_fn(inputs, out) if postproc else out

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        assert hasattr(self, "_loss_fn"), "Basic Model needs a user defined loss function"
        model_outputs = self.forward(inputs, training=training, **kwargs)
        loss = self._loss_fn(inputs, outputs, model_outputs)
        if writer:
            writer.add_scalar(writer_prefix + "loss", loss.mean().item(), i)
        return loss
