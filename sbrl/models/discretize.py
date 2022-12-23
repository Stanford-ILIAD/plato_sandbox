import scipy.cluster.vq
import torch
import torch.nn.functional as F

from sbrl.experiments import logger
from sbrl.models.layers.linear_vq_vae import vq_forward
from sbrl.models.model import Model
from sbrl.utils.python_utils import get_with_default, get_required, AttrDict
from sbrl.utils.torch_utils import combine_after_dim, concatenate, to_torch


class Discretize(Model):
    def forward(self, inputs, inverse=False, training=False, preproc=True, postproc=True, **kwargs):
        """

        if inverse:
            turns discrete and/or discrete_idx, residual -> continuous
        else:
            turns continous inputs -> discrete, discrete_idx, residual

        :param inputs:
        :param inverse:
        :param training:
        :param preproc:
        :param postproc:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class KMeansDiscretize(Discretize):

    def _init_params_to_attrs(self, params):
        super(KMeansDiscretize, self)._init_params_to_attrs(params)
        self.discrete_input_names = get_required(params, "discrete_inputs")  # which input names to discretize

        # discrete vector length
        self.vec_dim = get_required(params, "vec_dim")

        self.centroid_name = get_with_default(params, "centroid_name", "centroid")
        self.idxs_name = get_with_default(params, "idxs_name", "idxs")
        self.residual_name = get_with_default(params, "residual_name", "residual")

        # how many clusters to generate in k-means
        self.num_clusters = get_required(params, "num_clusters")

        logger.debug(f"KMeansDiscretize using vector_size={self.vec_dim} with num_clusters={self.num_clusters}")

        # codebook, will be initialized in load_statistics
        embed = torch.empty((self.vec_dim, self.num_clusters), device=self.device)
        self.register_buffer('embed', embed)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def load_statistics(self, dd=None):
        dd = super(KMeansDiscretize, self).load_statistics(dd=dd)

        # ----- K MEANS -----

        logger.debug(f"Computing K-means clusters (#={self.num_clusters})")
        # gets the datadict
        datadict = self._dataset_train.get_datadict() > self.discrete_input_names
        # |Data| x |A| TODO normalize first?
        flat_data = concatenate(datadict.leaf_apply(lambda arr: combine_after_dim(arr, 1)),
                                self.discrete_input_names, dim=-1)

        # will be k x |A|
        codebook, distortion = scipy.cluster.vq.kmeans(flat_data, self.num_clusters)
        self.embed[:] = to_torch(codebook.T, device=self.device)

        logger.debug(f"K-means done. final distortion = {distortion})")

        return dd

    def forward(self, inputs, inverse=False, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param input: AttrDict (B x H...)
        :return: AttrDict with relevant inputs mapped to cluster and residual
        """

        if inverse:
            # DISCRETE -> CONTINUOUS
            centroid = inputs << self.centroid_name
            if centroid is None:
                embed_idxs = inputs >> self.idxs_name
                centroid = F.embedding(embed_idxs, self.embed.transpose(0, 1))

            res = inputs >> self.residual_name
            res_flat = concatenate(res.leaf_apply(lambda arr: combine_after_dim(arr, 2)),
                                   self.discrete_input_names, dim=-1)
            cont = centroid + res_flat

            return inputs & self._env_spec.parse_view_from_concatenated_flat(cont, self.discrete_input_names)
        else:
            # CONTINOUS -> DISCRETE
            inputs_c = inputs > self.discrete_input_names
            flat_in = concatenate(inputs_c.leaf_apply(lambda arr: combine_after_dim(arr, 2)),
                                  self.discrete_input_names, dim=-1)

            out = vq_forward(self.embed, flat_in, return_dict=True) > ['quantize', 'embed_idxs']
            residual = flat_in - out.quantize

            quantized_dict = self._env_spec.parse_view_from_concatenated_flat(out.quantize, self.discrete_input_names)
            residual_dict = self._env_spec.parse_view_from_concatenated_flat(residual, self.discrete_input_names)

            return inputs & AttrDict.from_dict({
                self.centroid_name: quantized_dict,
                self.residual_name: residual_dict,
                self.idxs_name: out.embed_idxs,
            })

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, ret_dict=False, meta=AttrDict(),
             **kwargs):
        # no loss for this module
        return torch.tensor([0], device=self.device)
