import torch

from sbrl.experiments import logger
from sbrl.models.basic_model import BasicModel
from sbrl.utils.python_utils import get_with_default, timeit
from sbrl.utils.torch_utils import combine_after_dim, unsqueeze_n, broadcast_dims, to_torch


class EnergyBasedModel(BasicModel):

    def _init_params_to_attrs(self, params):
        # override default to ensure training computes the negatives
        super(EnergyBasedModel, self)._init_params_to_attrs(params)

        self.negatives_per_samples = get_with_default(params, "negatives_per_samples", 256)

        # output size becomes |action|, each element corresponding to p(x_{<=i}), TODO not implemented
        self.autoregressive = get_with_default(params, "autoregressive", False)

        self.use_uniform_negatives = get_with_default(params, "use_uniform_negatives", False)

        # will only sample negatives for these
        self.negative_inputs = get_with_default(params, "negative_model_inputs", self.inputs)

        assert set(self.negative_inputs).issubset(self.inputs), [self.negative_inputs, self.inputs]
        self.fixed_inputs = list(set(self.inputs).difference(self.negative_inputs))

        # if the input distribution (including spec) is already normalized
        self.inputs_pre_normalized = get_with_default(params, "inputs_pre_normalized", False)

        if self.inputs_pre_normalized and self.use_uniform_negatives:
            logger.warn("Inputs are supposedly pre-normalized, but negatives will be uniformly sampled from env spec.. "
                        "normalization is ignored.")

        # either not normalizing (all inputs are already normalized), or all inputs must provide normalizing stats,
        assert self.inputs_pre_normalized or len(
            set(self.save_normalization_inputs).symmetric_difference(self.inputs)) == 0

    def _init_setup(self):
        super(EnergyBasedModel, self)._init_setup()
        # for de-normalizing sample
        self._flat_mean = None
        self._flat_std = None
        # for uniform sampling
        self._neg_min = None
        self._neg_max = None

        if 'compute_negatives' not in self._loss_forward_kwargs:
            self._loss_forward_kwargs['compute_negatives'] = True

        if self.use_uniform_negatives:
            self._neg_min, self._neg_max = self.env_spec.limits(self.negative_inputs, flat=True)
            self._neg_max = to_torch(self._neg_max, device=self.device)
            self._neg_min = to_torch(self._neg_min, device=self.device)

        self._negative_input_size = self.env_spec.dim(self.negative_inputs)

    def load_statistics(self, dd=None):
        dd = super(EnergyBasedModel, self).load_statistics(dd=dd)

        if not self.inputs_pre_normalized:
            mean = []
            std = []
            for n in self.negative_inputs:
                mean.append(combine_after_dim(self.torch_means[n], 0))
                std.append(combine_after_dim(self.torch_stds[n], 0))

            self._flat_mean = torch.cat(mean, dim=0)
            self._flat_std = torch.cat(std, dim=0)

        return dd

    def forward(self, inputs, training=False, preproc=True, postproc=True, compute_negatives=False, **kwargs):
        """

        :param inputs:
        :param training:
        :param preproc:
        :param postproc:
        :param compute_negatives: if True, will pass in negatives for each input
        :param kwargs:
        :return:
        """

        with timeit("ebm/forward/positive"):
            outs = super(EnergyBasedModel, self).forward(inputs, training=training, preproc=preproc, postproc=postproc,
                                                         **kwargs)

        if compute_negatives:
            with timeit("ebm/forward/negative"):
                front_size = self.env_spec.get_front_size(inputs)
                shape = list(front_size) + [self.negatives_per_samples, self._negative_input_size]

                if self.use_uniform_negatives:
                    eps = torch.rand(shape, device=self.device, dtype=self.concat_dtype)
                    eps = eps * (self._neg_max - self._neg_min) + self._neg_min
                else:
                    eps = torch.randn(shape, device=self.device, dtype=self.concat_dtype)

                    if not self.inputs_pre_normalized:
                        flat_mean, flat_std = unsqueeze_n(self._flat_mean, len(shape) - 1), unsqueeze_n(self._flat_std,
                                                                                                        len(shape) - 1)
                        eps = flat_std * eps + flat_mean

                negative_inputs = self.env_spec.parse_view_from_concatenated_flat(eps, self.negative_inputs)
                # copy fixed inputs over, and broadcast to same shape.
                for name in self.fixed_inputs:
                    negative_inputs[name] = broadcast_dims(inputs[name].unsqueeze(len(front_size)),
                                                           [len(front_size)], [self.negatives_per_samples])

                outs['negative'] = super(EnergyBasedModel, self).forward(negative_inputs, training=training,
                                                                         preproc=preproc, postproc=postproc, **kwargs)

        return outs
