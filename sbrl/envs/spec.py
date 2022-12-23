"""
Also the parent class of an EnvSpec, TODO move this somewhere else
"""

import numpy as np
import torch

from sbrl.utils.python_utils import AttrDict, get_required, is_array
# a wrapper for AttrDicts with numpy arrays or torch arrays to define a high level "specification"
from sbrl.utils.torch_utils import to_torch, torch_clip, numpy_to_torch_dtype_dict, unconcatenate, view_unconcatenate, \
    get_indices_for_flat


class Spec:
    def __init__(self, params: AttrDict):
        names_shapes_limits_dtypes = list(get_required(params, "names_shapes_limits_dtypes"))

        self._names_to_shapes = AttrDict()
        self._names_to_limits = AttrDict()
        self._names_to_dtypes = AttrDict()
        self._names = []
        for name, shape, limit, dtype in names_shapes_limits_dtypes:
            self._names_to_shapes[name] = shape
            self._names_to_limits[name] = limit
            self._names_to_dtypes[name] = dtype
            self._names.append(name)

        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params: AttrDict):
        pass

    def map_to_types(self, input_dict: AttrDict, skip_keys=False):
        def map_helper(key, arr_in):
            if skip_keys and key not in self._names_to_dtypes.keys():
                return arr_in
            else:
                assert key in self.all_spec_names, key
                if isinstance(arr_in, np.ndarray):
                    return arr_in.astype(self._names_to_dtypes[key])
                elif isinstance(arr_in, torch.Tensor):
                    return arr_in.to(dtype=numpy_to_torch_dtype_dict[self._names_to_dtypes[key]])
                else:
                    raise Exception("Non-array type should not be mapped with this function")

        input_dict = input_dict.leaf_copy()
        input_dict.leaf_kv_modify(map_helper)
        return input_dict

    def assert_has_names(self, names):
        for n in names:
            assert n in self.all_spec_names, "%s not in env spec names" % n

    def limits(self, names, as_dict=False, flat=False):
        lower, upper = [], []
        for name in names:
            shape = self.names_to_shapes[name]
            typ = self.names_to_dtypes[name]
            assert len(shape) >= 1
            l, u = self.names_to_limits[name]
            if flat:
                lower += [np.broadcast_to(l, shape).astype(typ).reshape(-1)]
                upper += [np.broadcast_to(u, shape).astype(typ).reshape(-1)]
            else:
                lower += [np.broadcast_to(l, shape).astype(typ)]
                upper += [np.broadcast_to(u, shape).astype(typ)]
        if as_dict:
            return AttrDict.from_kvs(names, list(zip(lower, upper)))

        if flat:
            return np.concatenate(lower), np.concatenate(upper)

        return np.array(lower), np.array(upper)

    def scale_to_unit_box(self, d, names):
        # careful of infinity / NaN!!
        lower, upper = self.limits(names)
        new_d = d.leaf_copy()
        for i, n in enumerate(names):
            new_d[n] = d[n] - (upper[i] + lower[i]) / 2.
            new_d[n] = d[n] / ((upper[i] - lower[i]) / 2.)
        return new_d

    def scale_from_unit_box(self, d, names):
        # careful of infinity / NaN!!
        lower, upper = self.limits(names)
        new_d = d.leaf_copy()
        for i, n in enumerate(names):
            new_d[n] = d[n] * ((upper[i] - lower[i]) / 2.) + (upper[i] + lower[i]) / 2.
        return new_d

    def reshape_from_flat(self, d: AttrDict, names):
        # mutates
        relevant = d.node_leaf_filter_keys_required(names)
        prepend_lengths = relevant.leaf_apply(lambda arr: list(arr.shape[:-1]))
        assert prepend_lengths.all_equal(), prepend_lengths.pprint(ret_string=True)
        prepend_shape = prepend_lengths.get_one()
        for n in names:
            d[n] = d[n].reshape(prepend_shape + list(self.names_to_shapes[n]))
        return d

    def parse_from_concatenated_flat(self, x, names, dtypes=None, outs=None, copy=False):
        # last dim is flat
        # assert isinstance(x, np.ndarray), "Torch not supported yet"
        return unconcatenate(x, names, self.names_to_shapes, dtypes=dtypes, outs=outs, copy=copy)

    def parse_view_from_concatenated_flat(self, x, names):
        # last dim is flat
        # assert isinstance(x, np.ndarray), "Torch not supported yet"
        return view_unconcatenate(x, names, self.names_to_shapes)

    def get_indices_for_flat(self, indices_names, names):
        return get_indices_for_flat(indices_names, names, self.names_to_shapes)

    def clip(self, d, names, object_safe=False):
        # mutates
        low, high = self.limits(names)
        for i in range(len(names)):
            name = names[i]
            if object_safe and self._names_to_dtypes[name] == np.object:
                continue
            if isinstance(d[name], torch.Tensor):
                l = to_torch(low[i], device=d[name].device)
                h = to_torch(high[i], device=d[name].device)
                d[name] = torch_clip(d[name], l, h)
            else:
                d[name] = np.clip(d[name], low[i], high[i])

    def dims(self, names):
        return np.array([np.prod(self.names_to_shapes[name]) for name in names])

    def dim(self, names):
        return np.sum(self.dims(names))

    def get_uniform(self, names, batch_size, torch_device=None):
        low, upp = self.limits(names)

        d = AttrDict()
        for i, name in enumerate(names):
            d[name] = np.random.uniform(low[i], upp[i], size=[batch_size] + list(low[i].shape))
            d[name] = d[name].astype(low[i].dtype)

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d

    def get_midpoint(self, names, batch_size, torch_device=None):
        low, upp = self.limits(names)

        d = AttrDict()
        for i, name in enumerate(names):
            d[name] = ((low[i] + upp[i]) / 2)[None].repeat(batch_size, axis=0)
            d[name] = d[name].astype(low[i].dtype)

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d

    def get_zeros(self, names, batch_size, torch_device=None):
        d = AttrDict()
        for name in names:
            assert name in self.all_spec_names, name
            d[name] = np.zeros([batch_size] + list(self.names_to_shapes[name]))
            d[name] = d[name].astype(self.names_to_dtypes[name])

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d

    def get_ones(self, names, batch_size, torch_device=None):
        out = self.get_zeros(names, batch_size, torch_device)
        for name in names:
            if out[name].dtype not in [np.bool, torch.bool]:
                out[name] += 1
        return out

    def get_empty(self, names, torch_device=None):
        # no elements
        return self.get_zeros(names, 0, torch_device=torch_device)

    def get_front_size(self, dc: AttrDict):
        int_names = list(set(dc.leaf_keys()).intersection(self.all_spec_names))

        b_shape = None
        # of the names present in dc, find the batch size (if all match), else return None
        for k in int_names:
            if is_array(dc[k]):
                # shape of the unbatched entry
                sh = self.names_to_shapes[k]
                # shape of the batched entry
                new_b_shape = dc[k].shape
                if len(sh) > 0:
                    # truncate by the amount that is "new"
                    assert len(sh) <= len(new_b_shape), [sh, new_b_shape]
                    new_b_shape = new_b_shape[:-len(sh)]
                # first key or batch mismatch
                if b_shape is None or b_shape == list(new_b_shape):
                    b_shape = list(new_b_shape)
                else:
                    # mismatch, return None
                    return None

        # this is None means we had no keys
        return b_shape

    @property
    def all_names(self):
        return list(self._names)

    @property
    def all_spec_names(self):
        return list(self._names)

    @property
    def names_to_shapes(self):
        """
        Knowing the dimensions is useful for building neural networks

        Returns:
            AttrDict
        """
        return self._names_to_shapes

    @property
    def names_to_limits(self):
        """
        Knowing the limits is useful for normalizing data

        Returns:
            AttrDict
        """
        return self._names_to_limits

    @property
    def names_to_dtypes(self):
        """
        Knowing the data type is useful for building neural networks and datasets

        Returns:
            AttrDict
        """
        return self._names_to_dtypes

