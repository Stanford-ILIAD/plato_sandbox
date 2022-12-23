"""
For distributions in torch
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as D
from torch.nn import functional as F

from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_with_default, get_required, get_or_instantiate_cls, is_array
from sbrl.utils.torch_utils import split_dim, torch_clip, unsqueeze_then_gather


def detach_normal(dist):
    assert isinstance(dist, D.Normal) or (isinstance(dist, D.Independent) and isinstance(dist.base_dist, D.Normal))
    event_dim = 0 if isinstance(dist, D.Normal) else dist.reinterpreted_batch_ndims
    bd = dist if isinstance(dist, D.Normal) else dist.base_dist
    new_loc = bd.loc.detach()
    new_scale = bd.scale.detach()
    new_normal = D.Normal(new_loc, new_scale)
    if event_dim > 0:
        new_normal = D.Independent(new_normal, event_dim)
    return new_normal


class DistributionCap(nn.Module):
    """
    Creates a distribution at the "output" of a network. This can be used as a sequential layer through layer params
    """

    def __init__(self, params: AttrDict):
        super().__init__()
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.out_name = get_with_default(params, "out_name", None)
        self.sample_out_name = get_with_default(params, "sample_out_name", None)
        self.use_argmax = get_with_default(params, "use_argmax", False)  # for sampling
        self.event_dim = get_with_default(params, "event_dim", 0)  # default distribution over last element

    def forward(self, x, **kwargs):
        return x

    def get_argmax(self, out_dist):
        raise NotImplementedError(str(__class__))

    def get_return(self, out_dist):
        # helper for dictionary based return
        if self.out_name is not None:
            out_dc = AttrDict.from_dict({self.out_name: out_dist})
            if self.sample_out_name is not None:
                if self.use_argmax:
                    sample = self.get_argmax(out_dist)
                else:
                    sample = out_dist.rsample() if out_dist.has_rsample else out_dist.sample()
                out_dc[self.sample_out_name] = sample
            return out_dc
        else:
            return out_dist


class MultiDistributionCap(nn.Module):
    """
    packages DistributionCaps at each specified name. (AttrDict packaging) -> TODO where does this apply?
    """

    def __init__(self, params: AttrDict):
        super().__init__()
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.names, self.distributions = params.get_keys_required(['names', 'distributions'])
        self.output_names = get_with_default(params, "output_names", [n + "_dist" for n in self.names])

        assert len(self.names) == len(self.distributions), "Must pass in same number of distributions as inputs"

        self._map = {}
        # create or get handed each DistributionCap
        for name, dist in zip(self.names, self.distributions):
            if isinstance(dist, AttrDict):
                dist = dist.cls(dist.params)
            assert isinstance(dist, DistributionCap)
            self._map[name] = dist

    def forward(self, inputs: AttrDict, **kwargs):
        """
        :return: each input name, with mapped keys, as a distribution
        """
        inputs = inputs.copy()
        # make sure all inputs present
        assert inputs.has_leaf_keys(self.names), [inputs.list_leaf_keys(), self.names]
        # make sure not overriding
        assert not any(inputs.has_leaf_key(o) for o in self.output_names), [inputs.list_leaf_keys(), self.output_names]

        for oname, name in zip(self.output_names, self.names):
            inputs[oname] = self._map[name].forward(inputs[name])

        return inputs


####################################### SPECIFICS #############################################


class GaussianDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(GaussianDistributionCap, self)._init_params_to_attrs(params)
        self.use_log_sig = get_with_default(params, "use_log_sig", True)
        self.use_tanh_mean = get_with_default(params, "use_tanh_mean", False)
        self.clamp_with_tanh = get_with_default(params, "clamp_with_tanh", False)
        self.sig_min = get_with_default(params, "sig_min", 1e-6)
        self.sig_max = get_with_default(params, "sig_max", 1e6)

        # inferred
        self.log_sig_min = np.log(self.sig_min)
        self.log_sig_max = np.log(self.sig_max)

    def get_distribution(self, raw_mu, raw_sig):
        # subclasses might override, example gaussian
        if self.use_log_sig:
            if self.clamp_with_tanh:
                raw_sig = self.log_sig_min + 0.5 * (self.log_sig_max - self.log_sig_min) * (torch.tanh(raw_sig) + 1)
                raw_sig = raw_sig.exp()
            else:
                # continuous log_sigma, exponentiated (less stable than softplus)
                raw_sig = raw_sig.clamp(min=self.log_sig_min, max=self.log_sig_max).exp()
        else:
            # continuous sigma, converted to R+
            raw_sig = F.softplus(raw_sig).clamp(min=self.sig_min, max=self.sig_max)
        if self.use_tanh_mean:
            raw_mu = torch.tanh(raw_mu)
        dist = D.Normal(loc=raw_mu, scale=raw_sig)
        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)

        return dist

    def forward(self, x, **kwargs):
        m, s = torch.chunk(x, 2, -1)
        return self.get_return(self.get_distribution(m, s))

    def get_argmax(self, out_dist):
        return out_dist.mean


class SquashedGaussianDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(SquashedGaussianDistributionCap, self)._init_params_to_attrs(params)
        self.use_log_sig = get_with_default(params, "use_log_sig", True)
        self.clamp_with_tanh = get_with_default(params, "clamp_with_tanh", True)
        self.sig_min = get_with_default(params, "sig_min", 1e-6)
        self.sig_max = get_with_default(params, "sig_max", 1e6)

        # inferred
        self.log_sig_min = np.log(self.sig_min)
        self.log_sig_max = np.log(self.sig_max)

    def get_distribution(self, raw_mu, raw_sig):
        # subclasses might override, example gaussian
        if self.use_log_sig:
            if self.clamp_with_tanh:
                raw_sig = self.log_sig_min + 0.5 * (self.log_sig_max - self.log_sig_min) * (torch.tanh(raw_sig) + 1)
                raw_sig = raw_sig.exp()
            else:
                # continuous log_sigma, exponentiated (less stable than softplus)
                raw_sig = raw_sig.clamp(min=self.log_sig_min, max=self.log_sig_max).exp()
        else:
            # continuous sigma, converted to R+
            raw_sig = F.softplus(raw_sig).clamp(min=self.sig_min, max=self.sig_max)
        dist = SquashedNormal(loc=raw_mu, scale=raw_sig)
        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)
        return dist

    def forward(self, x, **kwargs):
        m, s = torch.chunk(x, 2, -1)
        return self.get_return(self.get_distribution(m, s))

    def get_argmax(self, out_dist):
        return out_dist.mean


class CategoricalDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(CategoricalDistributionCap, self)._init_params_to_attrs(params)
        self.num_bins = int(get_required(params, "num_bins"))

    def forward(self, x, **kwargs):
        # x should be (..., num_bins)
        assert x.shape[-1] == self.num_bins

        dist = D.Categorical(logits=x)

        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)

        return self.get_return(dist)

    def get_argmax(self, out_dist):
        if self.event_dim:
            out_dist = out_dist.base_dist
        return torch.argmax(out_dist.probs, dim=-1)


class MixedDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(MixedDistributionCap, self)._init_params_to_attrs(params)
        self.num_mix = int(get_required(params, "num_mix"))
        assert self.num_mix > 0
        self.split_dim = get_with_default(params, "split_dim", -1, map_fn=int)
        self.chunk_dim = get_with_default(params, "chunk_dim", -1, map_fn=int)
        self.is_categorical = get_with_default(params, "is_categorical", True)
        self.all_mix = []
        self.base_dist: DistributionCap = get_or_instantiate_cls(params, "base_dist", DistributionCap)

        # goes from (input tensor, output dist, Cat tensor) -> Distribution or
        self.combine_dists_fn = get_with_default(params, "combine_dists_fn", lambda ls_in, ls_out,
                                                                                    cat: MixedDistributionCap.default_combine_to_mixture_fn)

    def forward_i(self, x, i):
        return self.all_mix[i].forward(x)

    def forward(self, x, **kwargs):
        # each gets equal inputs
        if isinstance(x, List):
            assert len(x) == 2 and self.is_categorical, len(x)
            x, cat = x
        elif self.is_categorical:
            # TODO handle chunk dim here...
            assert self.chunk_dim % len(x.shape) == len(x.shape) - 1, "Not implemented cat w/ chunk_dim neq -1"
            cat = x[..., :self.num_mix]
            x = x[..., self.num_mix:]
        else:
            cat = torch.zeros(list(x.shape[:-1]) + [self.num_mix], device=x.device)

        cat = torch.distributions.Categorical(logits=cat)

        # split dim
        sh = list(x.shape)
        split_dim = self.split_dim % len(sh)
        new_shape = sh[:split_dim] + [self.num_mix, x.shape[-1] // self.num_mix] + sh[split_dim + 1:]
        x = x.view(new_shape)

        # forward
        out = self.base_dist.forward(x)
        out_dist = self.combine_dists_fn(x, out, cat)
        return self.get_return(out_dist)

    def get_argmax(self, out_dist):
        # maximum likelihood sample.
        mean = out_dist.component_distribution.mean  # .. x k x D
        _, max_idxs = torch.max(out_dist.mixture_distribution.probs, dim=-1)  # ..

        return unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D

    @staticmethod
    def default_combine_to_mixture_fn(ins, out_dist, cat):
        return D.MixtureSameFamily(mixture_distribution=cat, component_distribution=out_dist)


####################################### DISTRIBUTIONS #############################################


# stable tanh
class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


# this is to avoid annoying numerical logprob instabilities TODO necessary?
class AffineTransformNoLogDet(D.AffineTransform):
    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        if self.event_dim:
            shape = shape[:-self.event_dim]
        return torch.zeros(shape, dtype=x.dtype, device=x.device)


# normalizes the loc (real domain) to be between -max and max using Tanh (bounded codomain)
class SquashedNormal(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, out_low=np.array(-1.), out_high=np.array(1.), event_dim=0):
        self.loc = loc
        self.scale = scale

        self.base_dist = D.Normal(loc, scale)
        self.event_dim = event_dim
        # reinterpret last N as part of dist
        if self.event_dim > 0:
            self.base_dist = D.Independent(self.base_dist, event_dim)

        self.bound_low = torch.tensor(out_low, dtype=self.base_dist.mean.dtype, device=self.base_dist.mean.device)
        self.bound_high = torch.tensor(out_high, dtype=self.base_dist.mean.dtype, device=self.base_dist.mean.device)

        mid = (self.bound_low + self.bound_high) / 2.
        range = (self.bound_low - self.bound_high) / 2.

        transforms = [TanhTransform()]  # , D.AffineTransform(mid, range, event_dim)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    # def rsample(self, sample_shape=torch.Size()):
    #     out = super(SquashedNormal, self).rsample(sample_shape)
    #     return SquashedNormal.smooth_clamp(out)
    #
    # def sample(self, sample_shape=torch.Size()):
    #     out = super(SquashedNormal, self).sample(sample_shape)
    #     return SquashedNormal.smooth_clamp(out)
    #
    # @staticmethod
    # def smooth_clamp(out, beta=30):
    #
    #     # don't allow samples to be exactly 1. or -1. (for gradient stability through logprob, for example)
    #     clamp_upper = 1. - F.softplus(1 - out, beta=beta)
    #     clamp_lower = F.softplus(out + 1, beta=beta) - 1.
    #
    #     gez = (out > 0).float()
    #
    #     # double ended softplus to "clamp" smoothly
    #     return gez * clamp_upper + (1 - gez) * clamp_lower


class BestCategorical(D.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super(BestCategorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)



class SoftmaxMixtureSameFamily(D.MixtureSameFamily):
    """
    Same as mixture of same family but component dist is
    """
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None, temperature=1.):
        super(SoftmaxMixtureSameFamily, self).__init__(mixture_distribution, component_distribution,
                                                       validate_args=validate_args)
        # high temperature means more of a hard-max, only for log prob TODO extend this
        self._temp = temperature
        # logger.debug(f"Softmax init w/ log_prob temperature: {temperature}")

    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        # k best per batch item, TODO
        if self._temp > 0:
            log_best_prob = torch.log_softmax((1. / self._temp) * log_prob_x, dim=-1).detach()
        else:
            log_best_prob = 0
        # categorical weights TODO normalize properly?
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_best_prob + log_mix_prob, dim=-1)  # [S, B]

    def rsample(self, sample_shape=torch.Size(), sample_all=False):
        # rsampling a mixture doesn't work that well...
        if self.component_distribution.has_rsample:
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.rsample(sample_shape)
            if sample_all:
                return comp_samples

            # mixture samples [n, B, k]
            if isinstance(self.mixture_distribution, D.Categorical):
                mix_sample = F.gumbel_softmax(self.mixture_distribution.logits)
            else:
                logger.warn("Not backpropping sampling mixture!")
                mix_sample = self.mixture_distribution.sample(sample_shape)  # if not categorical..
                mix_sample = F.one_hot(mix_sample, num_classes=comp_samples.shape[-2])
            mix_shape = mix_sample.shape

            for i in range(len(mix_shape), len(comp_samples.shape)):
                mix_sample = mix_sample.unsqueeze(-1)

            # Gather along the k dimension
            # mix_sample_r = mix_sample.reshape(
            #     mix_shape + torch.Size([1] * (len(es) + 1)))
            # mix_sample_r = mix_sample_r.repeat(
            #     torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            # samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            # return samples.squeeze(gather_dim)

            # mixed sample via gumbel
            samples = (comp_samples * mix_sample).sum(gather_dim)
            return samples
        else:
            raise NotImplementedError

    def rsample_each(self, sample_shape=torch.Size()):
        return self.component_distribution.rsample(sample_shape)

    @property
    def mean(self):
        # maximum likelihood sample.
        mean = self.component_distribution.mean  # .. x k x D
        _, max_idxs = torch.max(self.mixture_distribution.probs, dim=-1)  # ..

        return unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D


# upper bound on the KL divergence between gaussian p and mixture q
def upper_kl_normal_softmaxmixnormal(p, q, temp=None, min_weight=0):
    # p will have (... x N), q will be (..., M x N)
    qc = q.component_distribution
    qm = q.mixture_distribution
    if isinstance(qc, D.Independent):
        qc = qc.base_dist
    assert isinstance(qc, D.Normal)
    all_var_ratio = (p.scale[..., None, :] / qc.scale).pow(2)
    all_t1 = ((p.loc[..., None, :] - qc.loc) / qc.scale).pow(2)
    all_kl = 0.5 * (all_var_ratio + all_t1 - 1 - all_var_ratio.log())
    # all_kl = []
    # for j in range(qc.batch_shape[-2]):
    #     loc = qc.loc[..., j, :]
    #     scale = qc.scale[..., j, :]
    #     assert list(loc.shape) == list(p.loc.shape)
    #     var_ratio = (p.scale / scale).pow(2)
    #     t1 = ((p.loc - loc) / scale).pow(2)
    #     kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    #     all_kl.append(kl)
    # mix of the kl (... x M x N)
    # all_kl = torch.stack(all_kl, dim=-2)
    each_kl_avg = all_kl.mean(-1)  # KL mean over rightmost dimensions used to estimate the "best" distribution (of M)
    if temp is not None or q._temp > 0:
        temp = q._temp if temp is None else temp

    if temp is not None and temp < np.inf:
        # weight each KL as a probability.
        each_kl_avg_normalized = each_kl_avg / each_kl_avg.sum(-1, keepdim=True)
        log_softmin_each_kl = torch.log_softmax(- 1 / temp * each_kl_avg_normalized.detach().log(), dim=-1)  # soft min on KL
    else:
        log_softmin_each_kl = 0

    # (..., M)
    log_alphas = qm.logits + log_softmin_each_kl

    if temp == np.inf:
        # pick the best kl, match component dist to that. then match mixture to the idx of the best kl.
        best_kl, best_kl_idxs = each_kl_avg.min(-1)
        if min_weight > 0:
            mask = F.one_hot(best_kl_idxs, num_classes=each_kl_avg.shape[-1]).to(dtype=best_kl.dtype)
            mask = torch.where(mask < min_weight, min_weight, mask)
            best_kl = (each_kl_avg * mask).sum(-1)
        # (B, ) logit loss
        xe_logits = F.cross_entropy(qm.logits, best_kl_idxs, reduction='none')
        # best_all_kl = unsqueeze_then_gather(all_kl, best_kl_idxs, dim=len(best_kl_idxs))
        # (B, ) min dist loss
        return best_kl + xe_logits

    # normalize to sum to 1
    log_alphas = log_alphas - torch.logsumexp(log_alphas, -1, keepdim=True)
    return (log_alphas.exp().unsqueeze(-1) * all_kl).sum(-2)


def upper_kl_normal_softmaxmix_indnormal(p, q, temp=None):
    # p will have (... x N), q will be (..., M x N)
    assert isinstance(p.base_dist, D.Normal)
    if temp is None or temp < np.inf:
        return upper_kl_normal_softmaxmixnormal(p.base_dist, q, temp=temp).sum(-1)
    else:
        return upper_kl_normal_softmaxmixnormal(p.base_dist, q, temp=temp)  # return is (B,)

#
# # upper bound on the KL divergence between gaussian p and mixture q, pick the best KL.
# def upper_kl_normal_max_mixnormal(p, q):
#     # p will have (... x N), q will be (..., M x N)
#     qc = q.component_distribution
#     qm = q.mixture_distribution
#     if isinstance(qc, D.Independent):
#         qc = qc.base_dist
#     assert isinstance(qc, D.Normal)
#     all_kl = []
#     for j in range(qc.batch_shape[-2]):
#         loc = qc.loc[..., j, :]
#         scale = qc.scale[..., j, :]
#         assert list(loc.shape) == list(p.loc.shape)
#         var_ratio = (p.scale / scale).pow(2)
#         t1 = ((p.loc - loc) / scale).pow(2)
#         kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
#         all_kl.append(kl)
#     # mix of the kl (... x M x N)
#     all_kl = torch.stack(all_kl, dim=-2)
#     each_kl_avg = all_kl.mean(-1)  # KL mean over rightmost dimensions used to estimate the "best" distribution (of M)
#     if q._temp > 0:
#         log_softmin_each_kl = torch.log_softmax(- 1 / q._temp * each_kl_avg, dim=-1)  # soft min on KL
#     else:
#         log_softmin_each_kl = 0
#     log_alphas = qm.logits + log_softmin_each_kl
#
#     # normalize to sum to 1
#     log_alphas = log_alphas - torch.logsumexp(log_alphas, -1, keepdim=True)
#     return (log_alphas.exp().unsqueeze(-1) * all_kl).sum(-2)


# TODO this is broken :( must fix later

def get_squashed_normal_dist_from_mu_sigma_tensor_fn(out_low, out_high, event_dim=0):
    def squashed_normal_dist_from_mu_sigma_tensor_fn(key, raw, mu, log_std):
        return SquashedNormal(loc=mu, scale=F.softplus(log_std), out_low=out_low, out_high=out_high,
                              event_dim=event_dim)

    return squashed_normal_dist_from_mu_sigma_tensor_fn


# TODO this yields NaN's, somewhat predictably

def get_sgmm_postproc_fn(num_mixtures, names_in, names_out, act_lows, act_highs,
                         log_std_min=-5., log_std_max=2., logit_min=-4., logit_max=4.):
    assert not isinstance(names_in, str) and not isinstance(names_out, str)
    assert isinstance(num_mixtures, int) and num_mixtures > 0, num_mixtures
    zipped = list(zip(names_in, names_out, act_lows, act_highs))

    log_std_max = torch.tensor(log_std_max, dtype=torch.float32)
    log_std_min = torch.tensor(log_std_min, dtype=torch.float32)

    logit_max = torch.tensor(logit_max, dtype=torch.float32)
    logit_min = torch.tensor(logit_min, dtype=torch.float32)

    def sgmm_postproc_fn(inputs, model_output):
        result = AttrDict()
        for nin, nout, act_low, act_high in zipped:
            assert model_output.has_leaf_key(nin), nin
            ldim = (model_output[nin].shape[-1] - num_mixtures) // 2  # first num_mixtures are the weights (categorical)
            logits, mu, log_std = torch.split(model_output[nin], [num_mixtures, ldim, ldim], -1)
            mu = split_dim(mu, -1, (num_mixtures, ldim // num_mixtures))
            log_std = split_dim(log_std, -1, (num_mixtures, ldim // num_mixtures))

            mix = D.Categorical(logits=torch.clamp(logits, logit_min, logit_max))  # (..., num_mix)
            comp = SquashedNormal(mu, F.softplus(torch_clip(log_std, log_std_min.to(log_std.device),
                                                            log_std_max.to(log_std.device))),
                                  act_low, act_high, event_dim=1)  # (..., num_mix, dim)
            gmm = D.MixtureSameFamily(mix, comp)
            result[nout] = gmm
        return result

    return sgmm_postproc_fn


def get_dist_first_horizon(arr):
    if is_array(arr):
        return arr[:, 0]
    elif isinstance(arr, torch.distributions.Distribution):
        if isinstance(arr, D.Normal):
            return D.Normal(arr.loc[:, 0], arr.scale[:, 0])
        elif isinstance(arr, D.Categorical):
            return D.Categorical(arr.probs[:, 0])
        elif isinstance(arr, D.Independent):
            return D.Independent(get_dist_first_horizon(arr.base_dist), arr.reinterpreted_batch_ndims)
        elif isinstance(arr, SoftmaxMixtureSameFamily):
            return SoftmaxMixtureSameFamily(get_dist_first_horizon(arr.mixture_distribution), get_dist_first_horizon(arr.component_distribution), temperature=arr._temp)
        else:
             raise NotImplementedError

if __name__ == '__main__':
    DEVICE = "cpu"

    md = CategoricalDistributionCap(AttrDict(num_bins=15))

    test_inputs = torch.zeros((1, 15), dtype=torch.float32, device=DEVICE)
    test_outputs = torch.zeros((1,), dtype=torch.int, device=DEVICE)

    test_inputs[:, 3] = 100.
    test_outputs[0] = 3

    outs = md.forward(test_inputs)

    assert (outs.log_prob(test_outputs) == 0).item()

    test_inputs = torch.zeros((1, 20), dtype=torch.float32, device=DEVICE)
    test_inputs[:, 10:] = -1e10
    test_outputs = torch.zeros((1, 10), dtype=torch.float32, device=DEVICE)

    mdg = GaussianDistributionCap(AttrDict(use_log_sig=False, event_dim=1))

    outs = mdg(test_inputs)

    assert (outs.log_prob(test_outputs).item() > 100)
