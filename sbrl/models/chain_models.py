"""


Models are applied on AttrDicts, while Layers/Networks are applied on tensors (Models call Layers)
"""

from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict, get_required, get_or_instantiate_cls


class ChainModels(Model):

    def _init_params_to_attrs(self, params):
        super(ChainModels, self)._init_params_to_attrs(params)
        # for parent
        if not params.has_leaf_key('normalization_inputs'):
            params.normalization_inputs = []

        chain = list(get_required(params, "chain"))
        self.chains = []
        self.num_models = len(chain)
        for i, c in enumerate(chain):
            model = get_or_instantiate_cls(AttrDict(key=c), "key", Model, constructor=lambda cls, prms: cls(prms, self.env_spec, self._dataset_train)).to(self._device)
            self.chains.append(model)
            setattr(self, "model_%d" % i, model)  # for torch

    def _init_setup(self):
        pass

    def get_model(self, i) -> Model:
        return self.chains[i]

    def warm_start(self, model, observation, goal):
        for i in range(self.num_models):
            self.get_model(i).warm_start(model, observation, goal)

    def forward(self, inputs, chain_start=0, chain_end=-1, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn
        :param chain_start: int < num_models, the first model to use in the chain
        :param chain_end: int < num_models, the last model to use in the chain

        :return model_outputs: (AttrDict)  (B x ...)
        """
        curr = inputs
        for i in range(chain_start % self.num_models, (chain_end % self.num_models) + 1):
            curr = self.get_model(i)(curr, **kwargs)
        return curr

    def loss(self, inputs, outputs, chain_start=0, chain_end=-1, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param chain_end:
        :param chain_start:

        :return loss: (torch.Tensor)  (1,)
        """
        partial_outs = self.forward(inputs, chain_start=chain_start, chain_end=chain_end - 1, **kwargs)
        return self.get_model(chain_end).loss(partial_outs, outputs, **kwargs)
