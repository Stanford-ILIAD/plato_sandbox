import inspect
import os
import pickle
import resource
import time
from multiprocessing.managers import SyncManager

import cloudpickle
import numpy as np
import sharedmem

from sbrl.utils.python_utils import AttrDict


class CloudPickleWrapper(object):
    """
    Sending data through process
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def NpArrayFrom(manager: SyncManager, initial):
    assert isinstance(initial, np.ndarray)
    assert len(initial.shape) == 1
    return manager.Array(initial.dtype.char, initial.copy().tolist())


def np_wrap_read(arr, dtype):
    return np.array(arr._callmethod("__getitem__", (slice(0, arr._callmethod("__len__")),)), dtype=dtype)


# proxy object
def np_wrap_write(arr, val):
    assert isinstance(val, np.ndarray)
    assert len(arr) == len(val), "Expected %s, got %s" % (len(arr), len(val))
    for i in range(len(val)):
        arr._callmethod("__setitem__", (i, val[i]))

def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.shared


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def elapsed_since(start):
    #return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed*1000,2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed/60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def profile(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        rss_before, vms_before, shared_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format("<" + func.__name__ + ">",
                    format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
        return result
    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args,**kwargs)

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)  # MiB
    return mem
    # return psutil.virtual_memory().used / float(2 ** 20)

def memory_usage_resource():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / float(2 ** 20)

# wrapper for global information
class GlobalState(object):
    def __init__(self, params: AttrDict):
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params: AttrDict):
        self.running = sharedmem.full((1,), 1, dtype=bool)
