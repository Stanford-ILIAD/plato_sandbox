"""
VQ portion of VQ-VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import split_dim, combine_dims


class VectorQuantize(nn.Module):
    # decay=0.8, eps=1e-5,
    def __init__(self, e_dim, n_embed, return_dict=False, vec_dim=None):
        super().__init__()

        self.e_dim = e_dim  # embedding size (each code)
        self.vec_dim = vec_dim  # last dimension of input size. input will be split by e_dim to map into different codes if not None
        assert self.vec_dim is None or self.vec_dim % self.e_dim == 0, "Embeddings must be evenly divisible in the vec dim."
        self.n_embed = n_embed  # codebook size
        self.default_return_dict = return_dict

        if self.vec_dim is not None:
            logger.debug(f"VectorQuantize using vector_size={self.vec_dim} with embed_size={self.e_dim}")

        # codebook, stored in the layer.
        embed = torch.empty((e_dim, n_embed)).uniform_(-1/n_embed, 1/n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, return_dict=None, **kwargs):
        """
        This implements linear "snapping" for discrete VAE's. can be added as a layer.

        :param input:
        :param return_dict: overrides default if specified, returns AttrDict if true, otherwise tuple
        :return:
        """
        if return_dict is None:
            return_dict = self.default_return_dict
        if self.vec_dim is not None:
            input = split_dim(input, dim=-1, new_shape=(self.vec_dim // self.e_dim, self.e_dim))
        out = vq_forward(self.embed, input, return_dict=return_dict)
        if self.vec_dim is not None:
            if return_dict:
                (out > ['quantize', 'input', 'quantize_input_grad']).leaf_apply(lambda arr: combine_dims(arr, -2, 2))
            else:
                out = list(out)
                for i in range(3):
                    out[i] = combine_dims(out[i], -2, 2)
                out = tuple(out)
        return out


def vq_unsupervised_losses(vq_out: AttrDict, dims=-1, beta=0.25, **kwargs):
    """

    :param vq_out: quantize: The embedding vectors substituted
                   input: The continuous vector
                   quantize_input_grad: The embedding vector, with gradients only through the model
                   embed_idxs: indices of embeddings
    :param kwargs:
    :param dims: mean reduction dimensions
    :return:
    """

    # quantized z, original z (encoder), z passed into decoder (quant + grad), idxs in codebook
    z_q, z_enc, z_dec, embed_idxs = vq_out.get_keys_required(['quantize', 'input', 'quantize_input_grad', 'embed_idxs'])

    embed_update_loss = (z_enc.detach() - z_q).pow(2).mean(dims)
    enc_update_loss = (z_enc - z_q.detach()).pow(2).mean(dims)

    return AttrDict(
        loss=embed_update_loss + beta * enc_update_loss,
        embed_loss=embed_update_loss,
        enc_loss=enc_update_loss,
    )


def vq_forward(embed, input, return_dict=True):
    """
    Model-less version of VQ forward, for any shape.

    :param embed: codebook (embed_dim, num_codes)
    :param input: (..., embed_dim)
    :param return_dict: whether to return as tuples or dict.
    :return:
    """

    e_dim, n_embed = embed.shape

    dtype = input.dtype
    flatten = input.view(-1, e_dim)
    # ||in - emb||^2 = ||in||^2 - 2 * in.dot(emb) +||emb||^2
    dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
    )
    # min over embedding dists for (N, emb)
    _, embed_idxs = (-dist).max(1)
    embed_onehot = F.one_hot(embed_idxs, n_embed).type(dtype)
    embed_idxs = embed_idxs.view(*input.shape[:-1])

    # the quantized vectors for (..., e_dim), this will have gradients only to embedding vectors.
    quantize = F.embedding(embed_idxs, embed.transpose(0, 1))

    # if self.training:
    #     ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
    #     embed_sum = flatten.transpose(0, 1) @ embed_onehot
    #     ema_inplace(self.embed_avg, embed_sum, self.decay)
    #     cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
    #     embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
    #     self.embed.data.copy_(embed_normalized)

    # loss = F.mse_loss(quantize.detach(), input) * self.commitment

    # this will have gradients only to the inputs
    quantize_input_grad = input + (quantize - input).detach()

    if return_dict:
        return AttrDict(quantize=quantize, input=input, quantize_input_grad=quantize_input_grad, embed_idxs=embed_idxs, embed=embed)
    else:
        return quantize, input, quantize_input_grad, embed_idxs, embed
