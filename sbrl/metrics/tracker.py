"""
Trackers can be used to estimate time-series metrics (computed over single elements)
"""
from numbers import Number

import torch

from sbrl.metrics.metric import Metric
from sbrl.utils.python_utils import AttrDict as d, get_with_default, get_required
from sbrl.utils.script_utils import is_next_cycle


class Tracker:
    """
    Maintains a single metric across time. Default behavior is to only keep the last computed metric
    """
    def __init__(self, params: d):
        self._params = params
        self._init_params_to_attrs(params)
        self.name = self._metric.name
        self._state = d()
        self._state_stale = d()
        self.reset_tracked_state()

    def duplicate(self):
        return self.__class__(self._params.leaf_copy())

    def _init_params_to_attrs(self, params: d):
        self._metric: Metric = get_required(params, "metric")
        self._time_agg_fn = get_with_default(params, "time_agg_fn", lambda key, a, b: b)

        self._compute_grouped = get_with_default(params, "compute_grouped", False)
        # will record all present at first compute if None
        self._tracked_names = get_with_default(params, "tracked_names", None)
        # should return (g, n_groups)
        self._metric_get_group_fn = get_with_default(params, "metric_get_group_fn", None)
        assert not self._compute_grouped or self._metric_get_group_fn is not None, "Must specify grouping function"
        assert isinstance(self._metric, Metric)

    def reset_tracked_state(self, value=0):
        self._state.leaf_modify(lambda val: value)
        self._state_stale.leaf_modify(lambda val: True)

    def update(self, metric_results, group=None):
        if self._state.is_empty():
            if self._tracked_names is None:
                self._tracked_names = metric_results.list_leaf_keys()

            keys = self._tracked_names
            if self._compute_grouped:
                keys = [k for k in metric_results.leaf_keys() if any(k.startswith(t) for t in self._tracked_names)]

            self._state = d.from_kvs(keys, [0] * len(keys))
            self._state_stale = self._state.leaf_apply(lambda val: True)

        to_agg = d()
        for t in self._tracked_names:
            found = False
            for key in metric_results.leaf_keys():
                if key.startswith(t):
                    to_agg[key] = metric_results[key]
                    found = True
            assert found, f"Missing key start {t}, for metrics: {metric_results.list_leaf_keys()}"

        if self._compute_grouped and group is not None:
            for key in to_agg.leaf_keys():
                # if key has group idx at the end, it has been updated.
                if any(f":{g}" in key for g in group):
                    self._state_stale[key] = False
        elif not self._compute_grouped:
            for key in to_agg.leaf_keys():
                self._state_stale[key] = False

        # to_agg = metric_results > self._tracked_names
        # overrides
        self._state = d.leaf_combine_and_apply([self._state, to_agg], lambda k, vs: self._time_agg_fn(k, vs[0], vs[1]),
                                               pass_in_key_to_func=True)

    def compute_and_update(self, inputs: d, outputs: d, model_outputs: d):
        if self._compute_grouped:
            g, n_groups = self._metric_get_group_fn(inputs, outputs, model_outputs)
            m = self._metric.compute_group_wise(inputs, outputs, model_outputs, g, n_groups, return_dict=True)
        else:
            g = None
            m = self._metric.compute(inputs, outputs, model_outputs, return_dict=True)

        # batch size should be maintained by the metric
        for key, arr in m.items():
            if isinstance(arr, torch.Tensor):
                m[key] = arr.item()

        m.leaf_modify(lambda arr: arr.item() if isinstance(arr, torch.Tensor) else arr)
        m.leaf_assert(lambda arr: isinstance(arr, Number))

        self.update(m, group=g)
        return self._state.leaf_copy()

    def get_time_series(self, names=None):
        if names is None:
            return self._state.leaf_apply(lambda val: [val])
        else:
            return self._state.node_leaf_filter_keys_required(names).leaf_apply(lambda val: [val])

    @property
    def tracked_names(self):
        if self._tracked_names is None:
            raise ValueError
        else:
            return list(self._tracked_names)


class BufferedTracker(Tracker):
    """
    Maintains a single metric across time. Computes over a specified buffer length
    """
    def _init_params_to_attrs(self, params: d):
        super(BufferedTracker, self)._init_params_to_attrs(params)
        self.buffer_len = int(get_required(params, "buffer_len"))
        # how often to update (0 means update only reset)
        self.buffer_freq = get_with_default(params, "buffer_freq", 1, map_fn=int)
        assert self.buffer_len > 0
        self._buffer = d()

        self._counter = 0

    def reset_tracked_state(self, value=0):
        # special case, update the buffer only on resets
        if self.buffer_freq == 0 and not self._state.is_empty():
            self.update_buffer()
        super(BufferedTracker, self).reset_tracked_state(value)
        self._counter = 0

    def update_buffer(self):
        if self._buffer.is_empty():
            self._buffer = self._state.leaf_apply(lambda val: [])

        # truncated add
        for key, is_stale in self._state_stale.leaf_items():
            if not is_stale:
                self._buffer[key] = (self._buffer[key] + [self._state[key]])[-self.buffer_len:]
        # self._buffer = d.leaf_combine_and_apply([self._buffer, self._state],
        #                                         lambda vs: (vs[0] + [vs[1]])[-self.buffer_len:])

    def update(self, metric_results, group=None):
        super(BufferedTracker, self).update(metric_results, group=group)
        if is_next_cycle(self._counter, self.buffer_freq):
            self.update_buffer()
        self._counter += 1

    def get_time_series(self, names=None):
        if names is None:
            return self._buffer.leaf_copy()
        else:
            return self._buffer.node_leaf_filter_keys_required(names)

    def has_data(self, names=None):
        all_data = (self._buffer > names).leaf_values()
        return any(len(data) > 0 for data in all_data)
