import importlib.util
import itertools
import os
import sys


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders]))


def import_config(config_fname):
    assert config_fname.endswith('.py')
    spec = importlib.util.spec_from_file_location('config', config_fname)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params


def file_path_with_default_dir(fname, default_dir, expand_user=True, mkdirs=False):
    if expand_user:
        fname = os.path.expanduser(fname)
    dir = os.path.dirname(fname)
    if dir != '':
        assert os.path.exists(dir), "No such path: %s" % dir
        out_path = fname
    else:
        if mkdirs:
            os.makedirs(default_dir, exist_ok=True)
        out_path = os.path.join(default_dir, fname)
    return out_path


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def prepend_to_base_name(path: str, prefix: str):
    if "/" in path:
        bname = os.path.basename(path)
        dir = os.path.dirname(path)
        return os.path.join(dir, prefix + bname)
    else:
        return prefix + path


def postpend_to_base_name(path: str, postfix: str):
    if "/" in path:
        bname = os.path.basename(path)
        dir = os.path.dirname(path)
    else:
        bname = path
        dir = None

    ext_split = list(os.path.splitext(bname))
    ext_split[0] = ext_split[0] + postfix
    out = ''.join(ext_split)
    if dir is not None:
        out = os.path.join(dir, out)
    return out
