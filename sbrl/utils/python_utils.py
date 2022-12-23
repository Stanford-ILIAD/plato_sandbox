import atexit
import inspect
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from json import dumps
from typing import Tuple, List, Set, Any, Callable, Iterable, Optional, Dict

import numpy as np
import torch
from dotmap import DotMap

from sbrl.experiments import logger


class AttrDict(DotMap):

    def __getitem__(self, item):
        if isinstance(item, str) and '/' in item:
            item_split = item.split('/')
            curr_item = item_split[0]
            next_item = '/'.join(item_split[1:])
            return self[curr_item][next_item]
        else:
            return super(AttrDict, self).__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, str) and '/' in key:
            key_split = key.split('/')
            curr_key = key_split[0]
            next_key = '/'.join(key_split[1:])
            self[curr_key][next_key] = value
        else:
            super(AttrDict, self).__setitem__(key, value)

    def pprint(self, str_max_len=30, ret_string=False):
        str_self = self.leaf_apply(lambda x: str(x)[:str_max_len] + '...')
        if ret_string:
            return dumps(str_self.toDict(), indent=4, sort_keys=True)
        else:
            return super(AttrDict, str_self).pprint(pformat='json')

    def leaf_keys(self):
        def _get_leaf_keys(d, prefix=''):
            for key, value in d.items():
                new_prefix = prefix + '/' + key if len(prefix) > 0 else key
                if isinstance(value, AttrDict):
                    yield from _get_leaf_keys(value, prefix=new_prefix)
                else:
                    yield new_prefix

        yield from _get_leaf_keys(self)

    def node_leaf_keys(self):
        def _get_node_leaf_keys(d, prefix=''):
            for key, value in d.items():
                new_prefix = prefix + '/' + key if len(prefix) > 0 else key
                if isinstance(value, AttrDict):
                    yield new_prefix  # yield AttrDict mid level nodes as well
                    yield from _get_node_leaf_keys(value, prefix=new_prefix)
                else:
                    yield new_prefix

        yield from _get_node_leaf_keys(self)

    def list_leaf_keys(self) -> List[str]:
        # for printing the keys
        return list(self.leaf_keys())

    def list_node_leaf_keys(self):
        # for printing the keys
        return list(self.node_leaf_keys())

    def leaf_values(self):
        for key in self.leaf_keys():
            yield self[key]

    def node_leaf_values(self):
        for key in self.leaf_keys():
            yield self[key]

    def leaf_items(self):
        for key in self.leaf_keys():
            yield key, self[key]

    def node_leaf_items(self):
        for key in self.node_leaf_keys():
            yield key, self[key]

    def leaf_filter(self, func):
        d = AttrDict()
        for key, value in self.leaf_items():
            if func(key, value):
                d[key] = value
        return d

    def leaf_partition(self, cond):
        d_true = AttrDict()
        d_false = AttrDict()
        for key, value in self.leaf_items():
            if cond(key, value):
                d_true[key] = value
            else:
                d_false[key] = value
        return d_true, d_false

    def leaf_arrays(self):
        return self.leaf_filter(lambda k, v: is_array(v))

    def leaf_shapes(self):
        # mainly good for debugging tensor dicts.
        return self.leaf_arrays().leaf_apply(lambda arr: arr.shape)

    def node_leaf_filter(self, func, copy_nodes=False):
        d = AttrDict()
        for key, value in self.node_leaf_items():
            if func(key, value):
                d[key] = value
                if copy_nodes and isinstance(d[key], AttrDict):
                    d[key] = d[key].leaf_copy()
        return d

    def leaf_filter_keys(self, names):
        return self.leaf_filter(lambda key, value: key in names)

    def node_leaf_filter_keys(self, names):
        return self.node_leaf_filter(lambda key, value: key in names)

    def node_leaf_filter_keys_required(self, names, copy_nodes=False):
        """

        :param names: keys to get, can include nodes
        :param copy_nodes: if True, will recursively copy from a filtered key.
        :return:
        """
        out = AttrDict()
        for key in names:
            out[key] = self >> key
            if copy_nodes and isinstance(out[key], AttrDict):
                out[key] = out[key].leaf_copy()
        return out

    def leaf_assert(self, func):
        """
        Recursively asserts func on each value
        :param func (lambda): takes in one argument, outputs True/False
        """
        for value in self.leaf_values():
            assert func(value), [value, [key for key, item in self.leaf_items() if item is value]]

    def leaf_reduce(self, reduce_fn, seed=None):
        """
        sequentially reduces the given values for this dict, using reduce_fn
        Fixed order reduction should not be assumed.

        :param reduce_fn: [red, val_i] -> new_red
        :param seed: red0, if not present will use the first value to be popped
        :return:
        """
        vs = list(self.leaf_values())
        if seed is None:
            assert len(vs) > 0, len(vs)
            reduced_val = vs.pop()
        else:
            reduced_val = seed

        while len(vs) > 0:
            reduced_val = reduce_fn(reduced_val, vs.pop())
        return reduced_val

    def all_equal(self, equality_fn: Callable[[Any, Any], bool] = lambda a, b: a == b):
        v = list(self.leaf_values())
        if len(v) <= 1:
            return True
        return all(equality_fn(v[i], v[i+1]) for i in range(len(v) - 1))

    def leaf_modify(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in one argument and returns one object
        """
        for key, value in self.leaf_items():
            try:
                self[key] = func(value)
            except Exception as e:
                raise type(e)(key + ' : ' + str(e)).with_traceback(e.__traceback__)

    def leaf_kv_modify(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in two arguments and returns one object
        """
        for key, value in self.leaf_items():
            self[key] = func(key, value)

    def leaf_key_change(self, func):
        """
        Applies func to each key value (recursively), modifying in-place
        :param func (lambda): takes in two arguments and returns one object
        """
        d = AttrDict()
        for key, value in self.leaf_items():
            d[func(key, value)] = value
        return d

    def leaf_apply(self, func):
        """
        Applies func to each value (recursively) and returns a new AttrDict
        :param func (lambda): takes in one argument and returns one object
        :return AttrDict
        """
        d = AttrDict()
        for key, value in self.leaf_items():
            try:
                d[key] = func(value)
            except Exception as e:
                raise type(e)(key + ' : ' + str(e)).with_traceback(e.__traceback__)
        return d

    def leaf_kv_apply(self, func):
        d = AttrDict()
        for key, value in self.leaf_items():
            try:
                d[key] = func(key, value)
            except Exception as e:
                raise type(e)(key + ' : ' + str(e)).with_traceback(e.__traceback__)
        return d

    def leaf_call(self, func, pass_in_key_to_func=False):
        """
        Applies func to each value and ignores the func return
        :param func (lambda): takes in one argument, return unused
        """
        for key, value in self.leaf_items():
            func(key, value) if pass_in_key_to_func else func(value)

    def combine(self, d_other, ret=False):
        for k, v in d_other.leaf_items():
            self[k] = v

        if ret:
            return self

    def safe_combine(self, d_other, ret=False, warn_conflicting=False):
        others = set(d_other.leaf_keys())
        if not others.isdisjoint(self.leaf_keys()):
            if warn_conflicting:
                logger.warn(f"Combine found conflicts: {list(others.intersection(self.leaf_keys()))}")
            # keep keys in other dict that aren't conflicting
            d_other = d_other.leaf_filter_keys(list(others.difference(self.leaf_keys())))
        return self.combine(d_other, ret=ret)

    def freeze(self):
        frozen = AttrDict(self, _dynamic=False)
        self.__dict__.update(frozen.__dict__)
        return self

    def is_empty(self):
        return len(self.list_leaf_keys()) == 0

    def get_one(self):
        # raises StopIteration if is_empty()
        k, item = next(self.leaf_items())
        return item

    def has_leaf_key(self, key):
        return key in self.leaf_keys()

    def has_leaf_keys(self, keys):
        lk = set(self.leaf_keys())
        keys = set(keys)
        common = lk.intersection(keys)
        return len(common) == len(keys)

    def has_node_leaf_key(self, key):
        return key in self.node_leaf_keys()

    def has_node_leaf_keys(self, keys):
        k = set(self.node_leaf_keys())
        keys = set(keys)
        common = k.intersection(keys)
        return len(common) == len(keys)

    def leaf_key_intersection(self, ls: Set):
        return list(set(ls).intersection(self.leaf_keys()))

    def leaf_key_symmetric_difference(self, ls: Set):
        return list(set(ls).symmetric_difference(self.leaf_keys()))

    def leaf_key_difference(self, ls):
        return list(set(self.leaf_keys()).difference(ls))

    def leaf_key_missing(self, ls):
        return list(set(ls).difference(set(self.leaf_keys())))

    def node_leaf_key_overlap(self, ls: Set):
        return list(set(ls).intersection(self.node_leaf_keys()))

    def node_leaf_key_leftovers(self, ls):
        return list(set(self.node_leaf_keys()).difference(ls))

    def get_keys_required(self, keys) -> Tuple:
        assert self.has_node_leaf_keys(keys), list(set(keys).difference(self.node_leaf_keys()))
        return tuple(self[key] for key in keys)

    def get_keys_optional(self, keys, defaults):
        all_keys = list(self.node_leaf_keys())
        return tuple(self[keys[i]] if keys[i] in all_keys else defaults[i] for i in range(len(keys)))

    @staticmethod
    def leaf_combine_and_apply(ds, func, map_func=lambda x: x, match_keys=True, pass_in_key_to_func=False):
        # if match_keys false, default to the first dataset element's keys
        leaf_keys = tuple(sorted(ds[0].leaf_keys()))
        if match_keys:
            for d in ds[1:]:
                assert leaf_keys == tuple(sorted(d.leaf_keys())), "\n %s \n %s \n %s" % (leaf_keys, tuple(sorted(d.leaf_keys())), set(leaf_keys).symmetric_difference(d.leaf_keys()))

        d_combined = AttrDict()
        for k in leaf_keys:
            values = [map_func(d >> k) for d in ds]
            if pass_in_key_to_func:
                d_combined[k] = func(k, values)
            else:
                d_combined[k] = func(values)

        return d_combined

    @staticmethod
    def from_dict(d):
        d_attr = AttrDict()
        for k, v in d.items():
            d_attr[k] = v
        return d_attr

    def as_dict(self, out=None):
        if out is None:
            out = dict()
        for name in self.leaf_keys():
            out[name] = self[name]
        return out

    @staticmethod
    def from_kvs(keys: List[str], vals: List[Any]):
        assert len(keys) == len(vals)
        out = AttrDict()
        for k, v in zip(keys, vals):
            out[k] = v
        return out

    def leaf_copy(self):
        out = AttrDict()
        for k, v in self.leaf_items():
            out[k] = v
        return out

    # d >> key is short hand for getting a required key
    def __rshift__(self, name):
        assert self.has_node_leaf_key(name), ">>: missing key %s" % name
        return self[name]

    # d << key is short hand for getting an optional key with default None
    def __lshift__(self, name):
        if self.has_node_leaf_key(name):
            return self[name]
        return None

    # d > Iter: node_leaf_keys_required
    def __gt__(self, names: Iterable[str]):
        assert isinstance(names, Iterable) or names is None
        if names is None:
            return self.leaf_copy()
        return self.node_leaf_filter_keys_required(names)

    # d < Iter:  node_leaf_keys_optional
    def __lt__(self, names: Optional[Iterable[str]]):
        assert isinstance(names, Iterable)
        return self.node_leaf_filter_keys(names)

    # d1 & d2 is shorthand for combining dictionaries without modifying the original structs
    def __and__(self, other):
        out = self.leaf_copy()
        if other is None:
            return out
        return out.combine(other, ret=True)


class IterableAttrDict(AttrDict, Iterable):
    def __iter__(self):
        return


class TimeIt(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

        self._with_name_stack = []

    def __call__(self, name):
        self._with_name_stack.append(name)
        return self

    def __enter__(self):
        self.start(self._with_name_stack[-1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(self._with_name_stack.pop())

    def start(self, name):
        assert(name not in self.start_times), name
        self.start_times[name] = time.time()

    def stop(self, name):
        assert(name in self.start_times), name
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def reset(self):
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def as_np_dict(self, names=None):
        names = self.elapsed_times.keys() if names is None else names
        dc = AttrDict()
        for n in names:
            dc[n] = np.array([self.elapsed_times[n]])
        return dc

    def __str__(self):
        s = ''
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if 'total' not in self.elapsed_times:
                s += '{0}: {1: <10} {2:.1f}\n'.format(self.prefix, name, elapsed)
            else:
                assert(self.elapsed_times['total'] >= max(self.elapsed_times.values()))
                pct = 100. * elapsed / self.elapsed_times['total']
                s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, name, elapsed, pct)
        if 'total' in self.elapsed_times:
            times_summed = sum([t for k, t in self.elapsed_times.items() if k != 'total'])
            other_time = self.elapsed_times['total'] - times_summed
            assert(other_time >= 0)
            pct = 100. * other_time / self.elapsed_times['total']
            s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, 'other', other_time, pct)
        return s


class ThreadedTimeIt:

    def __init__(self, prefix=''):
        self.prefix = prefix
        self.timeit_by_thread: Dict[int, TimeIt] = dict()
        self._thread_count = 0

    def __call__(self, name):
        if threading.get_ident() not in self.timeit_by_thread.keys():
            # prefix only for later initialized threads.
            self.timeit_by_thread[threading.get_ident()] = TimeIt(prefix=self.prefix + ("/thread_" + str(threading.get_ident()) if self._thread_count > 0 else ""))
            self._thread_count += 1
        return self.timeit_by_thread[threading.get_ident()].__call__(name)

    def __enter__(self):
        return self.timeit_by_thread[threading.get_ident()].__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.timeit_by_thread[threading.get_ident()].__exit__(exc_type, exc_val, exc_tb)

    def start(self, name):
        return self.timeit_by_thread[threading.get_ident()].start(name)

    def stop(self, name):
        return self.timeit_by_thread[threading.get_ident()].stop(name)

    def elapsed(self, name):
        return self.timeit_by_thread[threading.get_ident()].elapsed(name)

    def reset(self):
        [self.timeit_by_thread[key].reset() for key in self.timeit_by_thread.keys()]

    def as_np_dict(self, names=None):
        dc = AttrDict()
        for key in sorted(self.timeit_by_thread.keys()):
            dc.combine(self.timeit_by_thread[key].as_np_dict(names=names))
        return dc

    def __str__(self):
        return "".join(self.timeit_by_thread[k].__str__() for k in sorted(self.timeit_by_thread.keys()))



timeit = ThreadedTimeIt()



def exit_on_ctrl_c():
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

@atexit.register
def cleanup_on_main():
    # I don't like that python doesn't do this automatically...
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.send_signal(signal.SIGHUP)

def pdb_on_exception(debugger="pdb", limit=100):
    """Install handler attach post-mortem pdb console on an exception."""
    pass

    def pdb_excepthook(exc_type, exc_val, exc_tb):
        traceback.print_tb(exc_tb, limit=limit)
        __import__(str(debugger).strip().lower()).post_mortem(exc_tb)

    sys.excepthook = pdb_excepthook


# call this for ipdb
ipdb_on_exception = lambda: pdb_on_exception("ipdb")


class NodeMetadata:
    def __init__(self, name, required, param_type=None):
        self.name = name
        self.required = required
        self.param_type = param_type


class ParameterizedObject(object):
    """
    Parameterized objects have a dict mapping supported keys to values

    This class just holds a list of each, and allows read / write access
    """

    def _init_parameterized_object(self):
        """
        Always call this before any get_..., when a class subclasses ParameterizedObject
        :return:
        """
        self._param_key_dict = AttrDict()

    @property
    def required_keys(self):
        return self._param_key_dict.leaf_filter(lambda k, v: v.required).list_leaf_keys()

    @property
    def optional_keys(self):
        return self._param_key_dict.leaf_filter(lambda k, v: not v.required).list_leaf_keys()

    @property
    def supported_keys(self):
        return self._param_key_dict.list_leaf_keys()

    def register_key(self, name, required: bool, param_type):
        assert isinstance(name, str), "Registered name must be string"
        assert not self._param_key_dict.has_node_leaf_key(name)
        if param_type != AttrDict:
            self._param_key_dict[name] = NodeMetadata(name, required, param_type)
        else:
            self._param_key_dict[name] = AttrDict()

    def assert_all_supported(self, params: AttrDict):
        leftovers = params.node_leaf_key_difference(self.supported_keys)
        assert len(leftovers) > 0, f"params have unsupported keys: {leftovers}"


# use these, since they interact with the parameterized object
def get_with_default(obj, attr, default, map_fn=None):

    if isinstance(obj, AttrDict):
        has = attr in obj.node_leaf_keys()
    else:
        has = hasattr(obj, attr)

    final = None
    if has:
        at = getattr(obj, attr)
        if at is not None and not (isinstance(at, AttrDict) and at.is_empty()):
            final = at if map_fn is None else map_fn(at)
    if final is None:
        final = default

    caller_frame = inspect.currentframe().f_back.f_locals
    if 'self' in caller_frame.keys() and isinstance(caller_frame['self'], ParameterizedObject):
        caller_frame['self'].register_key(attr, required=False, param_type=type(final))

    return final


def get_required(obj, attr):
    if isinstance(obj, AttrDict):
        assert attr in obj.node_leaf_keys(), "AttrDict missing key: %s" % attr
    else:
        assert hasattr(obj, attr), "Missing attr: %s" % attr

    out = getattr(obj, attr)
    assert out is not None and not (isinstance(out, AttrDict) and out.is_empty())

    caller_frame = inspect.currentframe().f_back.f_locals
    if 'self' in caller_frame.keys() and isinstance(caller_frame['self'], ParameterizedObject):
        caller_frame['self'].register_key(attr, required=True, param_type=type(out))

    return out


def get_from_ls(obj, attr, ls, default_idx=None, map_fn=None):

    if default_idx is not None:
        new_at = get_with_default(obj, attr, ls[default_idx], map_fn=map_fn)
    else:
        new_at = get_required(obj, attr)

    assert new_at in ls, "Specified %s, but not in supported list %s" % (new_at, ls)

    caller_frame = inspect.currentframe().f_back.f_locals
    if 'self' in caller_frame.keys() and isinstance(caller_frame['self'], ParameterizedObject):
        caller_frame['self'].register_key(attr, required=default_idx is None, param_type=type(new_at))

    return new_at


def get_cls_param_instance(params, cls_name, params_name, class_type, constructor=lambda cls, cls_params: cls(cls_params)):
    cls = get_required(params, cls_name)
    cls_params = params[params_name]

    assert isinstance(cls_params, AttrDict), cls_params

    obj = constructor(cls, cls_params)
    assert isinstance(obj, class_type), [type(obj), class_type]
    return obj


def get_or_instantiate_cls(params: AttrDict, attr_name: str, class_type, cls_name="cls", params_name="params", constructor=lambda cls, cls_params: cls(cls_params)):
    if attr_name is not None and len(attr_name) > 0:
        attr = get_required(params, attr_name)
    else:
        attr = params
    if isinstance(attr, AttrDict):
        return get_cls_param_instance(attr, cls_name, params_name, class_type, constructor=constructor)
    else:
        assert isinstance(attr, class_type)
        return attr


class dummy_context_mgr:
    """
    A dummy context manager - useful for having conditional scopes (such
    as @maybe_no_grad). Nothing happens in this scope.
    """
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_context(do, get_context, *args):
    """
    Optionally loads a context

    Args:
        do (bool): if True, the returned context will be get_context(), otherwise
            it will be a dummy context
    """
    return get_context(*args) if do else dummy_context_mgr()


if __name__ == '__main__':
    d = AttrDict(dict(
        a=dict(
            b=1,
            c=2
        )
    ))
    # print('start')
    # print(d['a/b'])
    # print(d['a/c'])
    # d['a/d'] = 3
    # print(d['a/d'])
    print(d.pprint())

    d1 = AttrDict(
        a=AttrDict(
            e=4
        )
    )
    d2 = d.copy()
    d.combine(d1)

    # mutates
    d.pprint()
    # does not mutate
    d2.pprint()
    out = d2.pprint(ret_string=True)
    print(out)

    ## other combining method

    # d1_dict = d1.as_dict()
    # d3 = AttrDict(dict(
    #     a=dict(
    #         b=1,
    #         c=2
    #     ),
    #     **d1_dict
    # ))
    #
    # d3.pprint()


def is_array(arr):
    return isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
