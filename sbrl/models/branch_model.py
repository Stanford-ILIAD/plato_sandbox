"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""

import torch

from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict, get_required, get_with_default
from sbrl.utils.torch_utils import concatenate


# N input chains, merge to 1 output tensor
class BranchModel(Model):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self.inputs_list = get_required(params, "model_inputs_list")  # list of lists of names per input branch
        self.input_dims = get_required(params, "model_input_dims")  # concatenation dimension
        self.output = str(get_required(params, "model_output"))

        for i in range(len(params.input_networks)):
            setattr(self, "input_network_%d" % i, params.input_networks[i].to_module_list(as_sequential=True).to(self.device))

        assert len(self.inputs_list) == len(params.input_networks) == len(self.input_dims)

        self.N = len(self.inputs_list)

        self.concat_dim = get_with_default(params, "concat_dim", -1)
        self.concat_dtype = get_with_default(params, "concat_dtype", torch.float32)
        self.output_net = params.output_network.to_module_list(as_sequential=True).to(self.device)

    # @abstract.overrides
    def _init_setup(self):
        pass

    # @abstract.overrides
    def warm_start(self, model, observation, goal):
        pass

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True,
                ret_partial_outputs=False, use_partial_inputs=(), **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn
        :param ret_partial_outputs: (bool) return the outputs from each model branch, along with model output
        :param use_partial_inputs: (tuple) feed the inputs in to bypass individual branches either () or full tuple (None to ignore element)

        :return model_outputs: (AttrDict)  (B x ...)
        """

        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        if preproc:
            inputs = self._preproc_fn(inputs)

        assert len(use_partial_inputs) == 0 or len(use_partial_inputs) == self.N

        # for each network, pass in inputs (or partial inputs)
        outs = []
        for i in range(self.N):
            if len(use_partial_inputs) > 0 and use_partial_inputs[i] is not None:
                out = use_partial_inputs[i]
            else:
                obs = concatenate(inputs, self.inputs_list[i], dim=int(self.input_dims[i]))
                out = getattr(self, "input_network_%d" % i)(obs)
            outs.append(out)

        # concatenate all outputs into one big vector
        obs = torch.cat(outs, dim=self.concat_dim)
        out = self.output_net(obs)
        out = AttrDict({self.output: out})

        out_proc = self._postproc_fn(inputs, out) if postproc else out
        return (out_proc, outs) if ret_partial_outputs else out_proc

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
        assert hasattr(self, "_loss_fn"), "Branch Model needs a user defined loss function"
        model_outputs = self.forward(inputs, training=training, **kwargs)
        loss = self._loss_fn(inputs, outputs, model_outputs)
        if writer:
            writer.add_scalar(writer_prefix + "loss", loss.mean().item(), i)
        return loss
