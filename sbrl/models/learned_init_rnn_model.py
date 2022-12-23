from typing import Tuple, List

from sbrl.models.rnn_model import RnnModel
from sbrl.utils.python_utils import get_with_default, timeit
from sbrl.utils.torch_utils import concatenate, combine_after_dim


class LearnedInitRnnModel(RnnModel):
    def _init_params_to_attrs(self, params):
        super(LearnedInitRnnModel, self)._init_params_to_attrs(params)

        self.init_net = (params >> "init_network").to_module_list().to(self.device)
        self.init_inputs = get_with_default(params, "init_inputs", self.inputs)

        assert self.parallel_model is None, "Parallel model might not work with preproc(), see forward()"

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, rnn_hidden_init=None, **kwargs):
        """
        Runs self.net and self.rnn_net, not necessarily in that order

        :param inputs: (AttrDict)  (B x SEQ LEN x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn
        :param rnn_hidden_init: (B x layers x hidden) the initial rnn state, must be same shape as the output of self.net(inputs) will be
                IF THIS IS NONE, WE WILL USE `self.init_net` TO GENERATE IT.
        :return model_outputs: (AttrDict)  [rnn_output_name]: (B x SEQ_LEN x num_directions*hidden_size)
                                           [hidden_name]: (num_layers*num_directions x B x hidden_size)
                               for batch_first=True, for example
        """

        inputs = inputs.leaf_copy()
        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        if preproc:
            inputs = self._preproc_fn(inputs)

        if rnn_hidden_init is None:
            with timeit("rnn/learned_init"):
                # in this case, get the first state in the sequence.
                init_inputs = (inputs > self.init_inputs).leaf_apply(lambda arr: combine_after_dim(arr[:, :1], 2))
                init_obs = concatenate(init_inputs, self.init_inputs, dim=self.concat_dim)
                init_obs = init_obs.view(init_obs.shape[0], -1)
                rnn_hidden_init = self.init_net(init_obs)
                if self.tuple_hidden:
                    assert isinstance(rnn_hidden_init, Tuple) or isinstance(rnn_hidden_init, List), f"Init net must output a tuple, but is {type(rnn_hidden_init)}"
                    assert len(rnn_hidden_init) > 1, len(rnn_hidden_init)

        # skip preproc and normalization, since we did it already
        # TODO debug this for parallel_model. preproc needs to not happen...
        return super(LearnedInitRnnModel, self).forward(inputs, training=training, preproc=False, postproc=postproc, rnn_hidden_init=rnn_hidden_init, do_normalize=False, **kwargs)
