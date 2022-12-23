import os
from numbers import Number
from typing import Callable

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import add_loadable_if_not_present, GroupedArgumentParser
from sbrl.utils.config_utils import ConfigLoader
from sbrl.utils.file_utils import file_path_with_default_dir, import_config


def is_next_cycle(current, period):
    return period > 0 and current % period == 0


def listify(value_or_ls, desired_len):
    if isinstance(value_or_ls, Number):
        value_or_ls = [value_or_ls] * desired_len
    else:
        value_or_ls = list(value_or_ls)
    assert len(value_or_ls) == desired_len
    return value_or_ls


def eval_preamble(config_fname, unknown, grouped_parser, ordered_modules, model_file, debug=False):
    """
    Preamble for eval script, loads the file manager, and model fname
    """
    config_fname = os.path.abspath(config_fname)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                     debug=debug)

    file_manager = ExperimentFileManager(exp_name, is_continue=True)

    if model_file is not None:
        model_fname = file_path_with_default_dir(model_file, file_manager.models_dir)
        assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
        logger.debug("Using model: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))
        
    return exp_name, args, params, model_fname


def load_standard_ml_config(base_config_name, arguments, grouped_parser, ordered_modules=('env_spec', 'env_train'), debug=False, grouped_parser_arguments=None,
                            optional=None):
    """
    Reads a base config with the added loadable modules
    (1) declares module args
    (2) parses args from command line or file
    (3) processes args into each module, sequentially
    (4) instantiation (not part of this function) will create objects from the set of params returned here

    :param base_config_name:
    :param arguments: These are all the arguments for the base config file specifically.
    :param grouped_parser: The parser to add the loaders to, and to call parse_args on.
    :param ordered_modules:
    :param debug: prints arguments and final params
    :return: Tuple(exp_name: str, args: group_parser Namespace, params: processed AttrDict)
    """
    ordered_modules = list(ordered_modules)
    assert len(ordered_modules) > 0, "Must load some modules from the config!"

    if optional is None:
        optional = [False] * len(ordered_modules)

    with ConfigLoader(arguments) as c:  # load base config with remaining parameters
        # print(config_fname)
        logger.debug(f"Loading base config {base_config_name}...")
        common_params = import_config(base_config_name)

    logger.debug(f"Using modules: {ordered_modules}")
    assert len(ordered_modules) == len(optional), [ordered_modules, optional]
    for m, opt in zip(ordered_modules, optional):
        add_loadable_if_not_present(grouped_parser, m, common_params=common_params, optional=opt)

    # gets the arg structure without parsing everything
    grouped_parser.solve(args=grouped_parser_arguments)
    # sets the defaults from base config, when defaults not specified by each parser.
    grouped_parser.set_defaults_from_params(common_params)

    if debug:
        grouped_parser.print_help()

    if grouped_parser_arguments is None:
        args, unknown = grouped_parser.parse_known_args()
    else:
        args, unknown = grouped_parser.parse_known_args(args=grouped_parser_arguments)
    if len(unknown) > len(arguments):
        raise Exception(f"unparsed arguments: {set(unknown).difference(arguments)}")

    # convert to attrs, overriding conflicting names
    params = GroupedArgumentParser.to_attrs(args, common_params.leaf_copy())
    if debug:
        params.pprint()

    # loading order
    for group in ordered_modules:
        loaded_params = grouped_parser.get_child(group).params
        if not loaded_params.is_empty():
            params = (loaded_params >> "process_params")(group, params)
        else:
            logger.warn(f"Group {group} was not loadable")

    # now create everything.
    exp_name = params >> "exp_name"
    if isinstance(exp_name, Callable):
        exp_name = exp_name(params)

    assert isinstance(exp_name, str), f"Exp name should be a string: {exp_name}"

    return exp_name, args, params
