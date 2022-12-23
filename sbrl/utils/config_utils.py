"""
Extra utilities for the config module.
"""
import importlib.util
import os

import numpy as np

from sbrl.envs.env import Env
from sbrl.utils.file_utils import prepend_to_base_name, import_config
from sbrl.utils.python_utils import AttrDict as d


class ArgContext:
    """
    The set of arguments in a given context, managed by a ConfigLoader
    """
    def __init__(self, args=()):
        self._all_arguments = list(args)

    def register(self, arguments):
        self._all_arguments.extend(arguments)

    def get_args(self):
        return list(self._all_arguments)

# unique
_default_config_ctx = ArgContext()
_current_ctx = _default_config_ctx


class ConfigLoader(object):
    """
    Manages contextual "arguments" for nested parsers (e.g. if one parser only sees some of the arguments)
    """
    def __init__(self, args=()):
        self._last_ctx = None
        self._this_ctx = ArgContext(args)

    def __enter__(self):
        global _current_ctx
        self._last_ctx = _current_ctx
        _current_ctx = self._this_ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_ctx
        _current_ctx = self._last_ctx


def register_config_args(arguments):
    _current_ctx.register(arguments)

def get_config_args():
    return _current_ctx.get_args()


# parser
_parser = None
def set_current_parser(parser):
    global _parser
    _parser = parser

def get_current_parser():
    from argparse import ArgumentParser
    return _parser if _parser is not None else ArgumentParser()


class Utils:
    """ Utils base class, each module should experiment some options here """
    pass


def get_base_preproc_fn(names):
    def preproc_fn(inputs: d) -> d:
        new_inputs = inputs.copy()
        # collapse horizon
        for name in names:
            new_inputs[name] = inputs[name].view(-1, *inputs[name].shape[2:])
        return new_inputs

    return preproc_fn


def bool_cond_add_to_exp_name(name, params, args_or_arg2abbrev, abbrevs=None, sep="_"):
    # args_or_arg2abbrev can be either [arg1....] (in which case abbrevs must be specified)
    #                               or [(arg1, abb1)...]
    N = len(args_or_arg2abbrev)
    assert N > 0, "empty args passed to bool_cond_add_to_exp_name()"
    if isinstance(args_or_arg2abbrev[0], str):
        assert abbrevs is not None and len(abbrevs) == N, "Abbreviations must be specified externally!"
        arg2abbrev = list(zip(args_or_arg2abbrev, abbrevs))
    else:
        arg2abbrev = args_or_arg2abbrev
        assert abbrevs is None, "Cannot specify abbreviations!"

    for key, abbrev in arg2abbrev:
        # checks if key is True
        if params >> key:
            name += f"{sep}{abbrev}"

    return name

def get_path_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_config_params_with_args(config_path, args):
    # extra args are populated only with these
    with ConfigLoader(args):
        config_params = import_config(config_path)
    return config_params


def get_config_exp_name(config_path, args) -> str:
    config_params = get_config_params_with_args(config_path, args)
    return config_params >> "exp_name"


def get_config_exp_name_v2(config_path, args, group_args=None, gp=None, ordered_modules=('env_spec', 'env_train', 'model', 'policy')) -> str:
    if gp is None:
        from sbrl.experiments.grouped_parser import GroupedArgumentParser
        gp = GroupedArgumentParser()
    from sbrl.utils.script_utils import load_standard_ml_config
    exp_name, _, _ = load_standard_ml_config(config_path, args, gp, ordered_modules, grouped_parser_arguments=group_args, debug=False)
    return exp_name


def get_config_utils_module(config_file, utils_module_file="utils.py"):
    # gets the utils within a config folder, e.g., get_config_utils_module(__file__) from base_config.py
    return get_path_module("utils", os.path.join(os.path.dirname(config_file), utils_module_file))


##############

def prepend_next_process_env_step_output_fn(env: Env, obs: d, goal: d, next_obs: d, next_goal: d, policy_outputs: d,
                                       env_action: d, done: d) -> (
        d, d):
    # "next/{}"
    new_next_obs = next_obs.copy()
    for name in env.env_spec.observation_names:
        new_next_obs[f"next/{name}"] = next_obs[name]
    new_next_obs['reward'] = next_obs.reward
    return new_next_obs, next_goal


def base_level_prepend_next_process_env_step_output_fn(env: Env, obs: d, goal: d, next_obs: d, next_goal: d, policy_outputs: d,
                                       env_action: d, done: d) -> (
        d, d):
    # "next_{}"
    new_next_obs = next_obs.copy()
    for name in env.env_spec.observation_names:
        new_next_obs[prepend_to_base_name(name, "next_")] = next_obs[name]
    new_next_obs['reward'] = next_obs.reward
    return new_next_obs, next_goal


def default_process_env_step_output_fn(env: Env, obs: d, goal: d, next_obs: d, next_goal: d, policy_outputs: d,
                                       env_action: d, done: d) -> (
        d, d):
    return next_obs, next_goal


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_dims_for_keys(nsld, names):
    dims = 0

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            dims += np.prod(shape)

    return dims


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_lims_for_keys(nsld, names):
    lims = {}

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            lims[name] = limits

    return [lims[name] for name in names]


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_dtypes_for_keys(nsld, names):
    dtypes = []

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            dtypes.append(dtype)

    return dtypes


def nsld_get_names_to_shapes(nsld):
    nts = d()
    for n, s, _, _ in nsld:
        nts[n] = s
    return nts


def nsld_get_row(nsld, name):
    for row in nsld:
        if row[0] == name:
            return row
    raise ValueError(name)
