import math
import sys
from numbers import Number
from typing import List, Optional, Mapping, Union, Callable

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as RU

from sbrl.utils.input_utils import get_str_from
from sbrl.utils.python_utils import AttrDict, is_array

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

reduce_map_fn = {
    'sum': lambda t: t.sum(),
    'mean': lambda t: t.mean(),
    'max': lambda t: t.max(),
    'min': lambda t: t.min(),
}

torch_to_numpy_dtype_dict = {val: key for key, val in numpy_to_torch_dtype_dict.items()}


torch_mappable = lambda dt: dt in numpy_to_torch_dtype_dict.keys() or dt in torch_to_numpy_dtype_dict.keys() \
    or (isinstance(dt, np.dtype) and dt.type in numpy_to_torch_dtype_dict.keys())


def dc_torch_mappable(dc):
    return dc.leaf_filter(lambda k, v: torch_mappable(v.dtype))


def get_zeroth_horizon(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x[:, 0]
    else:
        return x


def add_horizon_dim(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x[:, None]
    else:
        return x


def unsqueeze_n(x, n, dim=0):
    for _ in range(n):
        x = x.unsqueeze(dim)
    return x


def cat_any(vs, *args, dim=0, **kwargs):
    if isinstance(vs[0], np.ndarray):
        return np.concatenate(vs, *args, axis=dim, **kwargs)
    else:
        return torch.cat(vs, *args, dim=dim, **kwargs)


def concatenate(input_dict: AttrDict, names, dim=0, default_shape=None):
    # print(names, list(input_dict.leaf_keys()))
    all = []
    for n in names:
        all.append(input_dict >> n)
    if len(all) > 0:
        if type(all[0]) is torch.Tensor:
            return torch.cat(all, dim=dim)
        else:
            return np.concatenate(all, axis=dim)
    elif default_shape is not None:
        return np.empty(list(default_shape) + [0], dtype=np.float32)
    else:
        raise Exception(
            "No elements to concatenate for names: %s from keys: %s" % (names, list(input_dict.leaf_keys())))


def concat_apply_split(input_dict: AttrDict, names, func, combine_dim=-1):
    relevant = input_dict > names
    names_to_shapes = relevant.leaf_apply(lambda arr: list(arr.shape))
    arr = concatenate(relevant.leaf_apply(lambda arr: combine_after_dim(arr, combine_dim)), names, dim=combine_dim)
    new_arr = func(arr)
    return unconcatenate(new_arr, names, names_to_shapes)


def combine_then_concatenate(input_dict, names, dim=-1):
    # will combine after dim, then concatenate along that dim.
    relevant = input_dict > names
    return concatenate(relevant.leaf_apply(lambda arr: combine_after_dim(arr, dim)), names, dim=dim)


def unconcatenate(x, names, names_to_shapes, dtypes=None, outs=None, copy=False):
    """

    :param x: inputs flat combined array
    :param names:
    :param names_to_shapes:
    :param dtypes: if specified, outs will by copied to the given dtypes. Only works if outs is None or copy = False
    :param outs: if copy = True, writes to outs from x
                if copy = False, fills outs (AttrDict) but does not copy to existing out arrays.
    :param copy:
    :return:
    """

    assert not (dtypes is not None and outs is not None and copy), "Cannot specify dtypes AND outs when copy=True"

    d = AttrDict()  # where we write to initially
    prepend_shape = list(x.shape[:-1])
    shapes = [list(names_to_shapes[n]) for n in names]
    flat_shapes = [int(np.prod(sh)) for sh in shapes]
    # with timeit("split"):
    if isinstance(x, np.ndarray):
        chunked = np.split(x, np.cumsum(flat_shapes)[:-1], -1)
    else:
        chunked = torch.split(x, flat_shapes, -1)
    # with timeit("copyto"):
    for i, n in enumerate(names):
        # with timeit(f"copy_to_{n}"):
        d[n] = chunked[i].reshape(prepend_shape + shapes[i])

        if dtypes is not None:
            d[n] = d[n].astype(dtypes[n], copy=copy) if isinstance(x, np.ndarray) else d[n].to(dtype=dtypes[n])

        # mem copy if outs is specified, or just transfer.
        if copy and outs is not None:
            dst = outs >> n
            src = d >> n
            # print(n, dst.shape, dst.dtype)
            np.copyto(dst, src, casting='unsafe')
            d[n] = outs[n]  # outs will be correct.
        elif outs is not None:
            outs[n] = d[n]

        # # type cast
        # if dtypes is not None:
        #     if outs is not None and isinstance(x, np.ndarray):
        #         dst = outs >> n
        #         src = d >> n
        #         # print(dst.shape, src.shape)
        #         # print(dst.dtype, src.dtype)
        #         if src.dtype != dst.dtype or copy:
        #             np.copyto(dst, src, casting='unsafe')
        #             d[n] = dst
        #     else:
        #         d[n] = d[n].astype(dtypes[n], copy=copy) if isinstance(x, np.ndarray) else d[n].to(dtype=dtypes[n])

    return d


def view_unconcatenate(x, names, names_to_shapes):
    """
    Gets views of the big x array for each name, sliced and "viewed" or reshaped (memory efficient)

    :param x: inputs flat combined array
    :param names:
    :param names_to_shapes:
    :return:
    """

    d = AttrDict()  # where we write to initially
    prepend_shape = list(x.shape[:-1])

    idx_start = 0

    for name in names:
        shape = list(names_to_shapes[name])
        flat_shape = int(np.prod(shape))
        d[name] = x[..., idx_start:idx_start + flat_shape]
        if isinstance(d[name], np.ndarray):
            d[name] = d[name].reshape(prepend_shape + shape)
        else:
            d[name] = d[name].view(prepend_shape + shape)

        idx_start += flat_shape  # increment for next key

    assert idx_start == x.shape[-1], [idx_start, x.shape[-1], names]

    return d


def get_indices_for_flat(indices_names, names, names_to_shapes):
    """
    Gets the indices that would get "indices_names" from a concatenated flat array with "names".

    Names should be in order, but "indices_names" might not be! we go with the order in names.
        therefore the mapping from indices -> out_names will be permuted (not in indices_names order)

    :param x: inputs flat combined array
    :param indices_names:
    :param names:
    :param names_to_shapes:
    :return:
    """
    idx_count = 0
    indices = []
    for name in names:
        shape = list(names_to_shapes[name])
        flat_shape = int(np.prod(shape))
        if name in indices_names:
            indices.extend(range(idx_count, idx_count + flat_shape))
        idx_count += flat_shape

    return np.asarray(indices)


# Concat-able AttrDict. Frozen key map after init. Values can be changed
class CAttrDict(AttrDict):

    def __init__(self, names, after_dim=-1):
        super(CAttrDict, self).__init__()
        self.__dict__['_fixed_names'] = list(names)
        self.__dict__['_after_dim'] = after_dim  # after which dim to combine and concatenate

        for n in names:
            self[n] = None
        self.freeze()  # now it cannot be changed and the order is fixed

        self.__dict__['_curr_dim'] = self._after_dim
        self.__dict__['_concat_arr'] = None

    def __setitem__(self, key, value):
        # Only top node will be a CAttrDict.
        assert key in self.__dict__["_fixed_names"], f"Cannot add new key {key}"

        if isinstance(key, str) and '/' in key:
            key_split = key.split('/')
            curr_key = key_split[0]
            next_key = '/'.join(key_split[1:])
            if not self.has_key(curr_key):
                new_d = AttrDict()
                new_d[next_key] = value
                super(AttrDict, self).__setitem__(curr_key, new_d)
            else:
                self[curr_key][next_key] = value
        else:
            super(AttrDict, self).__setitem__(key, value)

        self.__dict__["_concat_arr"] = None  # invalidates concatenation cache

    @staticmethod
    def from_dynamic(input_dict: AttrDict, order=None, concat_arr=None, after_dim=-1):
        order = order or input_dict.list_leaf_keys()
        d = CAttrDict(order)
        for n in order:
            d[n] = input_dict >> n  # must be present
            assert is_array(d[n]), [n, type(d[n])]

        # initialize concatenation state
        d.__dict__["_concat_arr"] = concat_arr
        d.__dict__["_curr_dim"] = after_dim

        return d

    def concat(self, dim=None):
        if self.__dict__["_concat_arr"] is None or (dim is not None and dim != self.__dict__["_curr_dim"]):
            dim = dim if dim is not None else self.__dict__["_after_dim"]
            self.__dict__["_concat_arr"] = combine_then_concatenate(self, self.__dict__["_fixed_names"], dim=dim)

        return self.__dict__["_concat_arr"]


def to_torch(numpy_in, device="cuda", check=False):
    if check and isinstance(numpy_in, torch.Tensor):
        return numpy_in.to(device)
    if check and not isinstance(numpy_in, np.ndarray):
        return numpy_in
    else:
        return torch.from_numpy(numpy_in).to(device)


def torch_clip(torch_in, low, high):
    clip_low = torch.where(torch_in >= low, torch_in, low)
    return torch.where(clip_low <= high, clip_low, high)

def torch_clip_norm(arr, norm, dim=None):
    scale = norm / (torch.norm(arr, dim=dim) + 1e-11)
    return arr * torch.minimum(scale, torch.tensor([1], dtype=scale.dtype))

def to_numpy(torch_in, check=False):
    if check and isinstance(torch_in, np.ndarray):
        return torch_in
    return torch_in.detach().cpu().numpy()


def torch_to_numpy(torch_in):
    if not isinstance(torch_in, torch.Tensor):
        return torch_in
    return torch_in.detach().cpu().numpy()


def detach_grad_with_check(torch_in):
    if not isinstance(torch_in, torch.Tensor):
        return torch_in
    else:
        return torch_in.detach()


def to_mean_scalar(inp):
    if is_array(inp):
        return inp.mean().item()
    else:
        assert isinstance(inp, Number), "not an array so must be a number! %s" % inp
        return inp


def split_dim(torch_in, dim, new_shape):
    sh = list(torch_in.shape)
    if dim < 0:
        dim = len(sh) + dim
    assert dim < len(sh)
    assert sh[dim] == np.prod(new_shape), [sh[dim], new_shape, dim]
    new_shape = sh[:dim] + list(new_shape) + sh[dim + 1:]
    return torch_in.view(new_shape)


def split_dim_np(np_in, axis, new_shape):
    sh = list(np_in.shape)
    if axis < 0:
        axis = len(sh) + axis
    assert axis < len(sh)
    assert sh[axis] == np.prod(new_shape)
    new_shape = sh[:axis] + list(new_shape) + sh[axis + 1:]
    return np_in.reshape(new_shape)


def unsqueeze_then_gather(arr, idxs, dim):
    # idxs is (N1 .. Nj)
    # arr  is (N1 .. Nj M Nj+1 ... Nn)
    dim = dim % len(arr.shape)
    # this will unsqueeze idxs to match (N1 .. Nj Nj+1 ... Nn), and then gather
    assert list(arr.shape[:dim]) == list(
        idxs.shape), f"Indices must have same pre shape as arr: {idxs.shape}, {arr.shape}, dim={dim}"
    idxs = split_dim(idxs[..., None], dim=-1, new_shape=[1] * (len(arr.shape) - dim))
    new_shape = list(arr.shape[:dim]) + [1] + list(arr.shape[dim + 1:])
    idxs = torch.broadcast_to(idxs, new_shape)
    gathered = torch.gather(arr, dim=dim, index=idxs)
    return gathered.squeeze(dim)


def combine_dims(torch_in, start_dim, num_dims=2):
    sh = list(torch_in.shape)
    if start_dim < 0:
        start_dim = len(sh) + start_dim
    assert start_dim < start_dim + num_dims <= len(sh)
    new_sh = sh[:start_dim] + [-1] + sh[start_dim + num_dims:]
    return torch_in.view(new_sh)


def combine_dims_np(np_in, start_axis, num_axes=2):
    sh = list(np_in.shape)
    if start_axis < 0:
        start_axis = len(sh) + start_axis
    assert start_axis < start_axis + num_axes <= len(sh)
    new_shape = sh[:start_axis] + [-1] + sh[start_axis + num_axes:]
    if np.prod(sh) > 0:
        return np_in.reshape(new_shape)
    else:
        comb_shape = np.prod(sh[start_axis:start_axis+num_axes])
        new_shape[start_axis] = int(comb_shape)
        return np.zeros_like(np_in, shape=new_shape)


def combine_after_dim(arr, start_dim, allow_no_dim=False):
    max = len(arr.shape)
    if start_dim < 0:
        start_dim = max + start_dim
    if start_dim == max - 1:
        # already combined to this level
        return arr
    elif allow_no_dim and start_dim == max:
        return arr[..., None]  # add on a final dim
    elif isinstance(arr, torch.Tensor):
        return combine_dims(arr, start_dim, max - start_dim)
    else:
        return combine_dims_np(arr, start_dim, max - start_dim)


def combine_after_last_dim(inputs: AttrDict):
    min_len = inputs.leaf_reduce(lambda red, val: min(red, len(val.shape) if is_array(val) else np.inf), seed=np.inf)
    if 0 < min_len < np.inf:
        return inputs.leaf_apply(lambda arr: combine_after_dim(arr, int(min_len) - 1))
    else:
        return inputs


def broadcast_dims_np(arr: np.ndarray, axes: List[int], new_shape: List[int]):
    assert len(axes) == len(new_shape), get_str_from([axes, new_shape])
    sh = list(arr.shape)
    axes = [idx_wrap(i, len(sh)) for i in axes]
    new_sh = sh.copy()
    for i, ax in enumerate(axes):
        new_sh[ax] = new_shape[i]
    return np.broadcast_to(arr, new_sh)


def broadcast_dims(arr: torch.Tensor, dims: List[int], new_shape: List[int]):
    assert len(dims) == len(new_shape), get_str_from([dims, new_shape])
    sh = [-1] * len(arr.shape)
    for i, ax in enumerate(dims):
        sh[ax] = new_shape[i]
    return arr.expand(sh)


# horizon
def expand_h(dc):
    return dc.leaf_apply(lambda arr: arr[:, None])


# batch
def expand_b(dc):
    return dc.leaf_apply(lambda arr: arr[None])


def pad_dims(arr: Union[np.ndarray, torch.Tensor], dims: List[int], new_dims: List[int], val=0., mode='constant',
             after=True, delta=False):
    assert len(dims) == len(new_dims) > 0, [len(dims), len(new_dims), dims, new_dims]
    if not delta:
        # pad space check
        assert all(arr.shape[dim] <= desired for dim, desired in zip(dims, new_dims))
        # subtract each entry to get delta
        new_dims = [desired - arr.shape[dim] for dim, desired in zip(dims, new_dims)]

    # ((before0, after(right)0), ... beforeN, afterN)
    pads = [[0, 0] for _ in range(len(arr.shape))]
    b = int(after)  # pad right if after = True
    for dim, desired in zip(dims, new_dims):
        pads[dim][b] = desired

    if isinstance(arr, torch.Tensor):
        tpad = []
        for p in pads[::-1]:
            tpad += p
        return F.pad(arr, tpad, mode=mode, value=val)
    else:
        return np.pad(arr, pads, mode=mode, constant_values=val)


def numel(arr: Union[np.ndarray, torch.Tensor]):
    if isinstance(arr, np.ndarray):
        return arr.size
    elif is_array(arr):
        return arr.numel()


def get_horizon_chunks(arr, horizon, start_idx, end_idx, dim, stack_dim=None, skip_horizon=1):
    # (start, end) is inclusive
    sh = list(arr.shape)
    dim = idx_wrap(dim, len(sh))
    assert 0 <= dim < len(sh), get_str_from(dc=AttrDict(dim=dim, sh=sh))
    start_idx = idx_wrap(start_idx, sh[dim])
    end_idx = idx_wrap(end_idx, sh[dim])
    assert sh[dim] >= end_idx + horizon >= start_idx + horizon >= horizon > 0, get_str_from(dc=AttrDict(
        dim=dim, sh=sh, horizon=horizon, start=start_idx, end=end_idx))

    # chunkify!
    if stack_dim is None:
        stack_dim = dim
    # print(start_idx, end_idx, horizon ,dim)
    all = []
    for i in range(start_idx, end_idx + 1, skip_horizon):
        slc = [slice(None)] * len(sh)
        slc[dim] = slice(i, i + horizon)
        sliced = arr[tuple(slc)]
        # print(sliced.shape)
        all.append(sliced)

    # new shape will be (sh[0], sh[1] ... S(stackdim) ... H(dim) ... sh[-2], sh[-1])
    if isinstance(arr, torch.Tensor):
        return torch.stack(all, dim=stack_dim)
    else:
        return np.stack(all, axis=stack_dim)


## packed sequence operations

def get_first_elements(packed_sequence: RU.PackedSequence):
    starts = packed_sequence.batch_sizes


## others

def idx_wrap(idx: int, length: int) -> int:
    return idx % length


# All torch tensors, same size, P(targ | mu, sigma)
def log_gaussian_prob(mu_obs, sigma_obs, targ_obs):
    assert mu_obs.shape == sigma_obs.shape == targ_obs.shape, "%s, %s, %s" % \
                                                              (mu_obs.shape, sigma_obs.shape, targ_obs.shape)
    # assume last dimension is N
    N = mu_obs.shape[-1]

    det = (sigma_obs ** 2).prod(dim=-1)  # should be (batch, num_models) or just (batch,)
    a = torch.log((2 * np.pi) ** N * det)
    b = ((targ_obs - mu_obs) ** 2 / sigma_obs ** 2).sum(-1)  # (batch, nm, 3) -> (batch, nm) or without num_models

    assert a.shape == b.shape

    # cov determinant term + scaled squared error
    return - 0.5 * (a + b).mean()  # mean over batch and num models


# All torch tensors
def kl_regularization(latent_mu, latent_log_sigma, mean_p=0., sigma_p=1.):
    var_q = (latent_log_sigma.exp()) ** 2
    var_p = sigma_p ** 2
    sigma_p = torch.tensor(sigma_p, device=latent_mu.device)
    kl = ((latent_mu - mean_p) ** 2 + var_q) / (2. * var_p) + (torch.log(sigma_p) - latent_log_sigma) - 0.5
    return kl.mean()


# history is N x hist_len x dim, obs is N x dim
# prepends obs to history along second to last dimension
def advance_history(history, obs):
    # print(history.shape)
    if history.shape[-1] == 0:
        return history

    longer = torch.cat([obs.unsqueeze(-2), history], dim=-2)
    return longer[:, :-1]  # delete the last element


def disable_gradients(model: torch.nn.Module):
    ls_of_prev = []
    for param in model.parameters():
        ls_of_prev.append(param.requires_grad)
        param.requires_grad = False

    return ls_of_prev


def enable_gradients(model: torch.nn.Module, which_params=None):
    if which_params is None:
        which_params = [True] * len(list(model.parameters()))

    for param, prev_value in zip(model.parameters(), which_params):
        param.requires_grad = prev_value


class torch_disable_grad:
    """
    Context manager that will set all parameters of a Module to requires_grad=False.
    This disables logging gradients for this module while << still enabling gradient tracking >>
    """
    def __init__(self, model: torch.nn.Module, eval_mode=True):
        self._model = model
        self._ls_of_prev = None
        self._eval = eval_mode
        self._pre_mode = None

    def __enter__(self):
        self._ls_of_prev = disable_gradients(self._model)
        if self._eval:
            self._pre_mode = self._model.training
            self._model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._ls_of_prev is not None, "bug"
        enable_gradients(self._model, self._ls_of_prev)
        if self._eval:
            assert self._pre_mode is not None, "bug"
            self._model.train(self._pre_mode)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def get_residual_postproc_fn(input_keys, output_keys, final_keys=None):
    if final_keys is None:
        final_keys = output_keys
    assert len(input_keys) == len(output_keys) == len(final_keys), [input_keys, output_keys, final_keys]

    def postproc_fn(inputs, model_out):
        base = inputs.node_leaf_filter_keys_required(input_keys)
        delta = model_out.node_leaf_filter_keys_required(output_keys)
        final = AttrDict()
        for ik, ok, fk in zip(input_keys, output_keys, final_keys):
            final[fk] = base[ik] + delta[ok]
        return model_out & final

    return postproc_fn


def get_normal_dist_from_mu_sigma_tensor_fn(event_dim=0, log_std_min=-4., log_std_max=4.):
    def normal_dist_from_mu_sigma_tensor_fn(key, raw, mu, log_std):
        dist = D.Normal(loc=mu, scale=torch.clamp(log_std, min=log_std_min, max=log_std_max).exp())
        if event_dim:
            dist = D.Independent(dist, event_dim)
        return dist

    return normal_dist_from_mu_sigma_tensor_fn


def get_gaussian_like_postproc_fn(names_in, names_out,
                                  dist_from_mu_sigma_tensor_fn=get_normal_dist_from_mu_sigma_tensor_fn()):
    assert not isinstance(names_in, str) and not isinstance(names_out, str)
    zipped = list(zip(names_in, names_out))

    def gaussian_postproc_fn(inputs, model_output):
        result = AttrDict()
        for nin, nout in zipped:
            assert model_output.has_leaf_key(nin), nin
            mu, log_std = torch.chunk(model_output[nin], 2, -1)
            result[nout] = dist_from_mu_sigma_tensor_fn(nin, model_output[nin], mu, log_std)
        return result

    return gaussian_postproc_fn


# TODO
class Logistic(D.TransformedDistribution):
    def __init__(self, loc, scale):
        self.base_dist = D.Uniform(torch.zeros_like(loc), torch.ones_like(scale))
        self.transforms = [D.SigmoidTransform().inv, D.AffineTransform(loc=loc, scale=scale)]
        super(Logistic, self).__init__(base_distribution=self.base_dist, transforms=self.transforms)


class DiscreteLogistic(D.Distribution):
    # Pixel CNN
    def __init__(self, loc, scale, quantized_dim, min=-1, max=1):
        """
        :param loc: element wise
        :param scale: element wise
        :param quantized_dim: number of bins per element
        :param min:
        :param max:
        """
        self.loc = loc
        self.scale = scale
        self.min = min
        self.max = max
        self.quantized_dim = quantized_dim
        self.half_range = 0.5 * (self.max - self.min)
        self.mid = 0.5 * (self.min + self.max)
        self.pm = 1. / (self.quantized_dim - 1)
        super(DiscreteLogistic, self).__init__(batch_shape=loc.shape)

    has_rsample = False

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale

    def log_prob(self, value):
        """

        :param value: tensor of shape (d0, ... di,)
        :return: log probs of each element in this tensor, according the scale and range, (d0, ... di,)
        """
        x = (value - self.mid) / self.half_range  # scale to -1 -> 1
        centered_x = x - self.loc

        inv_stdv = 1. / self.scale
        plus_in = inv_stdv * (centered_x + self.pm)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - self.pm)
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - self.scale.log() - 2. * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - np.log((self.quantized_dim - 1) / 2))
        inner_cond = (x > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (x < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        return log_probs

    def cdf(self, value):
        x = (value - self.mid) / self.half_range  # scale to -1 -> 1
        centered_x = value - self.loc

        inv_stdv = 1. / self.scale
        plus_in = inv_stdv * (centered_x + self.pm)
        min_in = inv_stdv * (centered_x - self.pm)
        outer_cond = (x > 0.999).float()
        inner_cond = (x <= 0.999).float()

        outer_cond * (1 - torch.sigmoid(min_in)) + inner_cond * (torch.sigmoid(plus_in))

        return torch.sigmoid(plus_in)

    # we sample from the underlying logistic
    def sample(self, sample_shape=torch.Size()):
        u = self.loc.data.new(sample_shape).uniform_(1e-5, 1.0 - 1e-5)
        x = self.loc + self.scale * (torch.log(u) - torch.log(1. - u))

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        # NOTE samples are continuous here
        return x * self.half_range + self.mid


def get_dlm_postproc_fn(num_mixtures, names_in, names_out, discrete_min, discrete_max, num_bins,
                        log_std_min=-3., log_std_max=3.):
    assert not isinstance(names_in, str) and not isinstance(names_out, str)
    assert isinstance(num_mixtures, int) and num_mixtures > 0, num_mixtures
    zipped = list(zip(names_in, names_out))

    def dlm_postproc_fn(inputs, model_output):
        result = AttrDict()
        for nin, nout in zipped:
            assert model_output.has_leaf_key(nin), nin
            # ldim = (model_output[nin].shape[-1] - num_mixtures) // 2  # first num_mixtures are the weights (categorical)
            distrib_data = split_dim(model_output[nin], -1,
                                     [model_output[nin].shape[-1] // (3 * num_mixtures), 3 * num_mixtures])
            logit_probs, mu, log_std = torch.chunk(distrib_data, 3, -1)
            log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)

            mix = D.Categorical(logits=logit_probs)  # (..., num_mix)
            comp = DiscreteLogistic(mu, torch.exp(log_std), num_bins, discrete_min, discrete_max)  # (..., num_mix)
            dlm = D.MixtureSameFamily(mix, comp)
            result[nout] = dlm
        return result

    return dlm_postproc_fn


def get_dist_postproc_fn(names_in, names_out, dist_from_tensor_fn):
    # function dist_from_tensor_fn takes (key, model_outputs[key]) -> torch distribution
    assert not isinstance(names_in, str) and not isinstance(names_out, str)
    zipped = list(zip(names_in, names_out))

    def dist_postproc_fn(inputs, model_output):
        result = AttrDict()
        for nin, nout in zipped:
            assert model_output.has_leaf_key(nin), nin
            result[nout] = dist_from_tensor_fn(nin, model_output[nin])
        return result

    return dist_postproc_fn


#
# # logistic mixture model
# def get_lmm_quantized_bin_postproc_fn(num_mixtures, names_in, names_out):
#     assert not isinstance(names_in, str) and not isinstance(names_out, str)
#     assert isinstance(num_mixtures, int) and num_mixtures > 0, num_mixtures
#     zipped = list(zip(names_in, names_out))
#
#     def lmm_postproc_fn(inputs, model_output):
#         result = AttrDict()
#         for nin, nout in zipped:
#             assert model_output.has_leaf_key(nin), nin
#
#
#
#             result[nout] = gmm
#         return result
#     return gmm_postproc_fn


def get_image_flip_preproc_fn(names, contiguous=True, base_preproc_fn=lambda x: x):
    def preproc_fn(inputs: AttrDict):
        nin = base_preproc_fn(inputs).copy()
        for key in names:
            shape = list(nin[key].shape)
            offset = len(shape) - 3
            assert offset >= 0, shape
            # last 3 inputs are (H, W, C), need to be (C, H, W)
            nin[key] = nin[key].permute(list(range(offset)) + [offset + 2, offset, offset + 1])
            if contiguous:
                nin[key] = nin[key].contiguous()
        return nin

    return preproc_fn


def get_type_conversion_preproc_fn(names, types, base_preproc_fn=lambda x: x):
    nt = list(zip(names, types))

    def preproc_fn(inputs: AttrDict):
        nin = base_preproc_fn(inputs).copy()
        for key, type in nt:
            nin[key] = nin[key].to(dtype=type)
        return nin

    return preproc_fn


def get_normalize_fixed_bounds_fn(names, bounds, base_preproc_fn=lambda x: x):
    nt = list(zip(names, bounds))

    def preproc_fn(inputs: AttrDict):
        nin = base_preproc_fn(inputs).copy()
        for key, bounds in nt:
            nin[key] = (nin[key] - bounds[0]) / (bounds[1] - bounds[0])
        return nin

    return preproc_fn


def same_padding(H, W, Hout, Wout, kernel_size, stride, dilation=(1, 1)):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    #  ceil ( stride * (H - 1) - [ (H - 1) + dilation * (kernel_size - 1) ] )

    Ph = math.ceil(0.5 * (stride[0] * (Hout - 1) - (H - 1 - dilation[0] * (kernel_size[0] - 1))))
    Pw = math.ceil(0.5 * (stride[1] * (Wout - 1) - (W - 1 - dilation[1] * (kernel_size[1] - 1))))

    return (Ph, Pw)


def randint_between(t1, t2, fn=lambda x: x):
    t1, t2 = torch.broadcast_tensors(t1, t2)
    assert torch.all(t1 < t2)
    delta = (t2 - t1).to(dtype=torch.int)
    # [0, b-a)
    eps = delta * fn(torch.rand_like(t1, dtype=torch.float))
    # truncate
    eps.trunc_()
    return t1 + eps.to(dtype=torch.int)


def uniform(t1, t2):
    pass


class ExtractKeys(nn.Module):
    """ Extracts keys of given shapes from input flat array, or from a specific key in input dict."""
    def __init__(self, keys, shapes, from_key=None, postproc_fn=None):
        super(ExtractKeys, self).__init__()
        self.keys = keys
        self.shapes = shapes
        assert len(self.keys) == len(shapes)
        self.names_to_shapes = {k: s for k, s in zip(keys, shapes)}
        self.from_key = from_key
        self.postproc_fn = (lambda x: x) if postproc_fn is None else postproc_fn

    def forward(self, x):
        out = AttrDict()
        if self.from_key is not None:
            assert isinstance(x, AttrDict)
            out = x.leaf_copy()
            x = x >> self.from_key

        assert isinstance(x, torch.Tensor)
        return self.postproc_fn(out & view_unconcatenate(x, self.keys, self.names_to_shapes))


# N dimensional sample layer
class SamplingLinearLayer(nn.Linear):
    def __init__(self, sample_prob, in_features, bias=True):
        super(SamplingLinearLayer, self).__init__(in_features=in_features, out_features=2 * in_features, bias=bias)
        self.sample_prob = sample_prob

    def forward(self, x):
        output2x = super().forward(x).view(list(x.shape[:-1]) + [self.in_features, 2])
        idxs = torch.rand(output2x.shape[:-1]).unsqueeze(-1)  # (..., in_features, 1)
        idxs = (idxs > self.sample_prob).type(torch.int64)
        return torch.gather(output2x, -1, idxs), idxs

    # bernoulli
    def log_prob(self, idxs):
        return torch.where(idxs == 0, torch.ones_like(idxs) * self.sample_prob,
                           torch.ones_like(idxs) * (1 - self.sample_prob)).log()


class ResidualLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ResidualLinear, self).__init__(in_features, out_features, bias=bias)
        assert in_features == out_features, "Residual layers require equal num inputs and outputs"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super(ResidualLinear, self).forward(input)


# Linear layer which masks by "groups", with leftover features
class MaskLinear(nn.Linear):
    def __init__(self, num_chunks, in_features, out_features, in_masked_features=None, bias=True, residual=False):
        super(MaskLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        effective_in_features = in_features
        if in_masked_features is not None:
            # NOTE: will compute mask only for the first N features.
            assert 0 < in_masked_features <= in_features, "Must mask nonzero # features, less than/eq # total in features"
            effective_in_features = in_masked_features

        assert effective_in_features % num_chunks == 0, f"ef_in={effective_in_features} must be divisible by masking chunks {num_chunks}"
        assert out_features % num_chunks == 0, f"f_out={out_features} must be divisible by masking chunks {num_chunks}"

        # outf x inf
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self._residual = residual
        if residual:
            assert in_features == out_features, "Residual layers require equal num inputs and outputs"

        col_spacing = effective_in_features // num_chunks
        last_col = effective_in_features
        row_spacing = out_features // num_chunks
        for i in range(num_chunks):
            # outputs i*rs:(i+1)*rs don't see anything after (i+1)*cs
            # e.g., for in=6, out=4, num_ch=2, in_masked_f=3
            # mask = [[1. 1. 1. 0. 0., 0.,  1., 1., 1.]
            #         [1. 1. 1. 0. 0., 0.,  1., 1., 1.]
            #         [1. 1. 1. 1. 1., 1.,  1., 1., 1.]
            #         [1. 1. 1. 1. 1., 1.,  1., 1., 1.]]
            self.mask[i * row_spacing:(i + 1) * row_spacing, (i + 1) * col_spacing:last_col] = 0.

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # element-wise mask to ensure visibility
        out = F.linear(input, self.mask * self.weight, self.bias)
        if self._residual:
            return input + out
        else:
            return out


# reshapes an input to arbitrary output size
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def forward(self, x):
        return x.reshape([-1] + self.shape)


# reshapes an input to arbitrary output size
class SplitDim(nn.Module):
    def __init__(self, dim, new_shape):
        super(SplitDim, self).__init__()
        self.dim = dim
        self.new_shape = list(int(d) for d in new_shape)
        assert sum(int(d == -1) for d in self.new_shape) <= 1, "Only one dimension can be negative: %s" % self.new_shape
        self.idx, = np.nonzero([d == -1 for d in self.new_shape])

    def forward(self, x):
        new_sh = list(self.new_shape)
        if len(self.idx) > 0:
            new_sh[self.idx[0]] = 1
            new_sh[self.idx[0]] = int(x.shape[self.dim] / np.prod(new_sh))
        return split_dim(x, self.dim, new_sh)


# reshapes an input to arbitrary output size
class CombineDim(nn.Module):
    def __init__(self, dim, num_dims=2):
        super(CombineDim, self).__init__()
        self.dim = dim
        self.num_dims = num_dims

    def forward(self, x):
        return combine_dims(x, self.dim, self.num_dims)


# selects item from list (slice compatible), along dim
class ListSelect(nn.Module):
    def __init__(self, list_index, dim=0):
        super(ListSelect, self).__init__()
        self.list_index = list_index
        self.dim = dim

    def forward(self, x):
        if self.dim == 0:
            return x[self.list_index]
        else:
            slice_obj = [slice(None) for _ in range(len(x.shape))]
            slice_obj[self.dim] = self.list_index
            return x[tuple(slice_obj)]


# concats a list
class ListConcat(nn.Module):
    def __init__(self, dim=0):
        super(ListConcat, self).__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, dim=self.dim)


# splits an axis into a list
class ListFromDim(nn.Module):
    def __init__(self, dim, split_size_or_sections=1, check_length=None):
        super(ListFromDim, self).__init__()
        self.dim = dim
        self.split_size_or_sections = split_size_or_sections
        self.check_length = check_length

    def forward(self, x):
        x_ls = torch.split(x, self.split_size_or_sections, dim=self.dim)
        if self.check_length and len(x_ls) != self.check_length:
            raise ValueError(f"For shape: {x.shape}, splits: {self.split_size_or_sections}. Expected len={self.check_length} but len={len(x_ls)}!")
        if self.split_size_or_sections == 1:  # special case, all elements will flatten since each chunk is size=1
            x_ls = [arr.squeeze(self.dim) for arr in x_ls]
        return x_ls


# reshapes an input to arbitrary output size
class Permute(nn.Module):
    def __init__(self, order, order_includes_batch=False, contiguous=True):
        super(Permute, self).__init__()
        self.order = list(order)
        self.order_includes_batch = order_includes_batch

        self.contiguous = contiguous

        self.full_ord = self.order if self.order_includes_batch else [0] + self.order
        assert len(np.unique(self.full_ord)) == len(self.full_ord) and np.amax(self.full_ord) < len(
            self.full_ord), self.full_ord  # all idxs are unique and within range

    def forward(self, x):
        px = x.permute(self.full_ord)
        return px.contiguous() if self.contiguous else px


class Functional(nn.Module):
    def __init__(self, func):
        super(Functional, self).__init__()
        self.func = func
        assert isinstance(func, Callable), "Requires a callable function as input"

    def forward(self, x, **kwargs):
        return self.func(x, **kwargs)


class Assert(Functional):
    def forward(self, x, **kwargs):
        cond, err_msg = self.func(x, **kwargs)
        assert cond, err_msg
        return x


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']

        _, _, H, W = self.weight.size()

        self.register_buffer('mask', self.weight.data.clone())

        self.mask.fill_(1)

        is_mask_b = 0
        if mask_type == 'B':
            is_mask_b = 1

        # right segment nulled
        self.mask[:, :, H // 2, W // 2 + is_mask_b:] = 0
        # downward segment nulled
        self.mask[:, :, H // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(-1), (self.left_padding, 0, 0, 0)).squeeze(-1)

        return super(CausalConv1d, self).forward(x)


class ResMaskBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels, 1, **kwargs)
        )

    def forward(self, x):
        return self.block(x) + x


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class LinConv(nn.Linear):
    def __init__(self, permute_in, permute_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pin = permute_in
        self.pout = permute_out

    def forward(self, x):
        if self.pin:
            x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        if self.pout:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x) + x


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, batch_first=False, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] (flip first to if batch first)
            output: [sequence length, batch size, embed dim]
        Examples:
            # >>> output = pos_encoder(x)
        """

        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# coding: utf-8
# Code is adapted from:
# https://github.com/pclucas14/pixel-cnn-pp
# https://github.com/openai/pixel-cnn


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.

    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
                      torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                      (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.zeros(tensor.size() + (n,), dtype=torch.float32, device=tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0,
                                         clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = to_one_hot(argmax, nr_mix)
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x


# we can easily define discretized version of the gaussian loss, however,
# use continuous version as same as the https://clarinet-demo.github.io/
def mix_gaussian_loss(y_hat, y, log_scale_min=-7.0, reduce=True):
    """Mixture of continuous gaussian distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    C = y_hat.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y_hat.size(1) % 3 == 0
        nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters.
    if C == 2:
        # special case for C == 2, just for compatibility
        logit_probs = None
        means = y_hat[:, :, 0:1]
        log_scales = torch.clamp(y_hat[:, :, 1:2], min=log_scale_min)
    else:
        #  (B, T, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    dist = D.Normal(loc=0., scale=torch.exp(log_scales))
    # do we need to add a trick to avoid log(0)?
    log_probs = dist.log_prob(centered_y)

    if nr_mix > 1:
        log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        if nr_mix == 1:
            return -torch.sum(log_probs)
        else:
            return -torch.sum(log_sum_exp(log_probs))
    else:
        if nr_mix == 1:
            return -log_probs
        else:
            return -log_sum_exp(log_probs).unsqueeze(-1)


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)

    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]

    if nr_mix > 1:
        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, nr_mix)

        # Select means and log scales
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    else:
        if C == 2:
            means, log_scales = y[:, :, 0], y[:, :, 1]
        elif C == 3:
            means, log_scales = y[:, :, 1], y[:, :, 2]
        else:
            assert False, "shouldn't happen"

    scales = torch.exp(log_scales)
    dist = D.Normal(loc=means, scale=scales)
    x = dist.sample()

    x = torch.clamp(x, min=-1.0, max=1.0)
    return x


def bias_zero_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def zero_weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


################################################################################
def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


#################################################################################

class BranchedModules(nn.ModuleDict):
    def __init__(self, order=List[str], modules: Optional[Mapping[str, nn.Module]] = None, cat_dim=None,
                 split_sizes: List[int] = None, split_dim=None) -> None:
        super(BranchedModules, self).__init__(modules)
        assert modules is not None
        self._order = order
        self._cat_dim = cat_dim
        self._split_sizes = split_sizes
        self._split_dim = split_dim
        assert self._split_sizes is None or len(self._order) == len(self._split_sizes)
        assert self._split_sizes is None or all(s > 0 for s in self._split_sizes)
        assert self._split_sizes is None or split_dim is not None
        assert set(order) == set(modules.keys()), [order, modules.keys()]

    def forward(self, obs, ret_dict=False, **kwargs):
        all = AttrDict()
        all_ls = []

        if self._split_sizes is not None:
            obs_all = torch.split(obs, self._split_sizes, dim=self._split_dim)
        else:
            obs_all = [obs] * len(self._order)

        for k, o in zip(self._order, obs_all):
            all[k] = self[k](o, **kwargs)
            all_ls.append(all[k])

        if ret_dict:
            return all

        if self._cat_dim is not None:
            return torch.cat(all_ls, dim=self._cat_dim)

        return all_ls


class Ridge:
    def __init__(self, alpha=0, fit_intercept=True, ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones((X.shape[0], 1), device=X.device), X], dim=1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        lhs = X.T @ X
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0])
            self.w, _ = torch.lstsq(rhs, lhs + ridge)

    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None).to(self.w.device)
        if self.fit_intercept:
            X = torch.cat([torch.ones((X.shape[0], 1), device=X.device), X], dim=1)
        return X @ self.w

    @property
    def weight(self):
        return self.w


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets
    From: https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
    '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' \
            else torch.sum(loss) if self.reduction == 'sum' \
            else loss


if __name__ == '__main__':
    linear_1 = torch.nn.Linear(in_features=1, out_features=10)
    linear_2 = torch.nn.Linear(in_features=10, out_features=10)
    linear_1.train()
    linear_2.train()

    in_arr = torch.ones(1, dtype=torch.float32)
    optim = torch.optim.Adam(list(linear_1.parameters()) + list(linear_2.parameters()), lr=1e-3)

    l1_out = linear_1.forward(in_arr)
    with torch_disable_grad(linear_2):
        l2_out = linear_2.forward(l1_out)

    optim.zero_grad()
    loss = (l2_out + 2).abs().mean()
    loss.backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    optim.zero_grad()
    l1_out_2 = linear_1.forward(in_arr)
    l2_out_2 = linear_2.forward(l1_out_2)
    (l2_out_2 - 2).abs().mean().backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    # combined usage
    optim.zero_grad()
    l1_out_3 = linear_1.forward(in_arr)
    with torch_disable_grad(linear_2):
        l2_out_3_ng = linear_2.forward(l1_out_3)
    l2_out_3 = linear_2.forward(l1_out_3)

    ls = (l2_out_3_ng + 2).abs().mean() + (l2_out_3 - 2).abs().mean()
    ls.backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    sys.exit(0)

    ## funcs
    arr = torch.arange(5 * 3 * 6).view((5, 3, 6))
    out = pad_dims(arr, [0, 2], [2, 4], delta=True, after=True, val=-1)
    out_np = pad_dims(to_numpy(arr), [0, 2], [2, 4], delta=True, after=True, val=-1)
    assert list(out.shape) == list(out_np.shape) == [7, 3, 10], out
    assert torch.equal(out[:-2, :, :-4], arr)
    assert np.alltrue(out_np[:-2, :, :-4] == to_numpy(arr))

    t1 = torch.arange(5)
    t2 = 10 * torch.ones(5)
    for i in range(1000):
        r = randint_between(t1, t2)
        assert torch.all(r >= t1) and torch.all(r < t2), r

    ## DIST
    # loc = torch.zeros((5, 3))
    #
    # scale = 1 * torch.ones((5, 3))
    #
    # out_low = np.array([-1., -2., -10.])
    # out_high = np.array([1., 2., 10.])

    # dist = SquashedNormal(loc=loc, scale=scale, out_low=out_low, out_high=out_high, event_dim=1)
    # assert dist.has_rsample
    # print(dist.mean)
    #
    # sample = dist.sample()
    # sample = torch.zeros_like(loc, device=loc.device)

    # rescaled_sample = (to_numpy(sample) - ((out_high + out_low) / 2.)) / ((out_high - out_low) / 2.)
    #
    # proxy = D.transformed_distribution.TransformedDistribution(D.Independent(D.Normal(loc, scale), 1),
    #                                                            [D.TanhTransform()])
    #
    # prob = dist.log_prob(sample)
    # real_prob = proxy.log_prob(to_torch(rescaled_sample, device=sample.device)).to(dtype=torch.float32)
    #
    # print('Sample:', sample)
    # print('Rescaled Sample:', rescaled_sample)
    # print('Prob', prob)
    # print('Rescaled prob', real_prob)

    # assert torch.allclose(prob, real_prob)

    loc = 0.1 * torch.ones((1,))
    log_std = -2 * torch.ones((1,))

    # l = Logistic(loc, log_std.exp())
    dl = DiscreteLogistic(loc, log_std.exp(), 256)
    val = torch.linspace(-1., 1., 256)
    probs = dl.log_prob(val).exp()
    # probs_l = l.log_prob(val).exp()

    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots()
    #
    # axes.plot(val, probs, label="dl")
    # # axes.plot(val, probs_l, label="l")
    #
    # plt.show()

    # like a big batch of data
    big_arr = np.empty((1024 * 256, 200, 30), dtype=np.float32)

    sizes = [(2,)] * 8 + [(4,)] + [(5,)] * 2
    keys = [f"key_{i}" for i in range(len(sizes))]
    names_to_shapes = AttrDict.from_kvs(keys, sizes)
    dtypes = [np.float32, np.uint8, np.float16] * 3 + [np.int, np.int]
    names_to_dtypes = AttrDict.from_kvs(keys, dtypes)

    with timeit("unconcatenate_copy"):
        out = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=True)
    print(timeit)
    timeit.reset()
    with timeit("unconcatenate_no_copy"):
        out2 = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=False)
    print(timeit)
    timeit.reset()
    with timeit("unconcatenate_no_copy_out"):
        out3 = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=False, outs=out2)
    print(timeit)
    timeit.reset()
