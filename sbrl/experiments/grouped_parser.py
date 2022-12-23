"""
A group-based parser for the command line, or for generic arg lists.

Supports user defined macros

From the command line:
        <arg1> <arg2> #<group1> g1.py ... #<group2> g2.py ..., etc
    where group1, group2, ... are registered children (supports nested params)
"""

import argparse
import collections
import re
import sys
from argparse import _ActionsContainer
from typing import Dict, Any, List, Tuple

# will load files automatically on command line if @file_name is used.
from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.utils.file_utils import file_path_with_default_dir, import_config
from sbrl.utils.python_utils import AttrDict, get_with_default

parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")

config: Dict[str, Any] = {}


class MacroEnabledArgumentParser(argparse.ArgumentParser):
    def __init__(self,
                 macros: Dict[str, str] = None,
                 macro_prefix: str = '!',
                 **kwargs):
        self._macros = macros
        self._macro_prefix = macro_prefix
        assert len(self._macro_prefix) > 0 or self._macros is None
        super(MacroEnabledArgumentParser, self).__init__(**kwargs)

        if self.fromfile_prefix_chars is not None:
            assert self._macro_prefix not in self.fromfile_prefix_chars, [self._macro_prefix,
                                                                          self.fromfile_prefix_chars]

    def _parse_known_args(self, arg_strings, namespace):
        if self._macros is not None:
            arg_strings = self._substitute_macros(arg_strings)

        return super(MacroEnabledArgumentParser, self)._parse_known_args(arg_strings, namespace)

    def _substitute_macros(self, arg_strings):
        incorrect_structure = r'%s{(?!.*})' % self._macro_prefix
        options = '|'.join(self._macros.keys())
        pattern = f'{self._macro_prefix}{{{options}}}'
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:
            # missing end bracket
            if re.search(incorrect_structure, arg_string) is not None:
                self.error(f"Unclosed brackets in argument: {arg_string}")

            # regex macro substitution, while matches exist
            while re.search(pattern, arg_string) is not None:
                for macro, value in self._macros.items():
                    # e.g., !{ROOT}
                    arg_string = re.sub(f'{self._macro_prefix}{{{macro}}}', str(value), arg_string)
            # for regular arguments, just add them back into the list
            new_arg_strings.append(arg_string)

        # return the modified argument list
        return new_arg_strings


class GroupedArgumentParser(MacroEnabledArgumentParser):
    """
    Groups arguments recursively, supports conversion to AttrDict.
    """

    children: Dict[str, argparse.ArgumentParser]

    def __init__(self, group_prefix_char="%", children: Dict[str, argparse.ArgumentParser] = None, **kwargs):
        assert len(group_prefix_char) == 1, "Single character only: %s" % group_prefix_char
        super(GroupedArgumentParser, self).__init__(**kwargs)
        self.children = collections.OrderedDict() if children is None else children
        assert isinstance(self.children, collections.OrderedDict), "Children must be an ordered dict!"
        self.group_prefix_char = group_prefix_char

        self._solved = False
        self._solved_local_args = []
        self._solved_nested_raw_args = []
        self._solved_nested_args = {}

    def invalidate(self):
        self._solved = False
        self._solved_local_args.clear()
        self._solved_nested_args.clear()
        for child_name, child in self.children.items():
            if isinstance(child, GroupedArgumentParser):
                child.invalidate()

    @property
    def raw_nested_args(self) -> List:
        return list(self._solved_nested_raw_args)

    def parse_local_args(self, args=None, namespace=None):
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = argparse.Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not argparse.SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not argparse.SUPPRESS:
                        setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        # partition the groups
        local_arg_strings, _ = self._split_local_and_groups(args, only_local=True)
        # now process things only at this level
        return super(GroupedArgumentParser, self)._parse_known_args(local_arg_strings, namespace)

    def format_groups(self, formatter=None):
        if formatter is None:
            formatter = self._get_formatter()

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        for child_name, child in self.children.items():
            formatter.start_section(child_name)
            if isinstance(child, GroupedArgumentParser):
                child.format_groups(formatter)
            else:
                formatter.add_arguments(child._actions)
            formatter.end_section()

    def format_usage(self) -> str:
        formatter = self._get_formatter()
        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        self.format_groups(formatter)
        # determine help from format above
        return formatter.format_help()

    def format_help(self):
        formatter = self._get_formatter()
        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        self.format_groups(formatter)

        # epilog
        formatter.add_text(self.epilog)
        # determine help from format above
        return formatter.format_help()

    def _parse_known_args(self, arg_strings, namespace):
        # solving will do optional behaviors like file loading, etc, before parsing the children.
        if not self._solved:
            self.solve(arg_strings)

        # partition the groups
        local_arg_strings, nested_child_arg_strings = list(self._solved_local_args), dict(self._solved_nested_args)
        # now process things at this level
        args, unknown = super(GroupedArgumentParser, self)._parse_known_args(local_arg_strings, namespace)

        child_namespaces = dict()
        child_unknowns = {key: [] for key in self.children.keys()}

        for group_name, child in self.children.items():
            logger.debug(f"[Parser] Loading --> {group_name}... ")
            child_namespaces[group_name], child_unknown = child.parse_known_args(args=nested_child_arg_strings[group_name])
            # known parameters are represented with slash notation
            for cu in child_unknown:
                child_unknowns[group_name].append(cu)

        assert not hasattr(args, 'child_unknowns') and not hasattr(args, 'child_namespaces')
        if len(child_namespaces) > 0:
            args.child_namespaces = child_namespaces
            args.child_unknowns = child_unknowns
        # TODO how to return nested unknowns inside the top level unknown list?
        return args, unknown

    def _split_local_and_groups(self, arg_strings, only_local=False, ret_raw_children=False):
        # these args are ordered
        # <arg1> <arg2> %<group1> g1.py ... %%<nested group> ... %<group2> g2.py ..., etc

        new_arg_strings = []
        nested_child_arg_strings = {key: [] for key in self.children.keys()}
        curr_group = None
        for i, arg in enumerate(arg_strings):
            # LOCAL ARGS
            if curr_group is None:
                if arg[0] == self.group_prefix_char:
                    if only_local:
                        return new_arg_strings, nested_child_arg_strings
                    else:
                        # check that the first group is not a subgroup (proper nesting)
                        assert len(arg) > 1 and arg[1] != self.group_prefix_char, arg
                else:
                    # skip the rest
                    new_arg_strings.append(arg)
                    # print(f"{arg} going to LOCAL")
                    continue

            # GROUPED ARGS
            if arg[0] == self.group_prefix_char:
                assert len(arg) > 1, "Floating %s!" % self.group_prefix_char

                if arg[1] == self.group_prefix_char:
                    if curr_group in nested_child_arg_strings.keys():
                        # nested group, remove prefix for sub groups and add to current children arg lists
                        nested_child_arg_strings[curr_group].append(arg[1:])
                    else:
                        logger.warn(f"Skipping nested group: {arg[1:]} since {curr_group} does not exist.")
                else:
                    # set the new group for future args
                    curr_group = arg[1:]
                    if curr_group not in self.children.keys():
                        logger.warn("Skippping unsupported group key: %s. Supported ones: %s" % (curr_group, str(list(self.children.keys()))))
            else:
                if curr_group in self.children.keys():
                    # print(f"{arg} going to {curr_group}")
                    nested_child_arg_strings[curr_group].append(arg)

        if ret_raw_children:
            return new_arg_strings, nested_child_arg_strings, arg_strings[len(new_arg_strings):]
        return new_arg_strings, nested_child_arg_strings

    def add_child(self, child_name, child: argparse.ArgumentParser):
        assert child_name not in self.children.keys(), child_name
        if isinstance(child, MacroEnabledArgumentParser) and self._macros is not None:
            # send your macros down
            if child._macros is None:
                child._macros = {}
            child._macros.update(self._macros)

        self.children[child_name] = child

    @staticmethod
    def set_default_no_override(container: _ActionsContainer, **kwargs):
        safe_kwargs = {}
        for key, item in kwargs.items():
            if container.get_default(key) is None:
                safe_kwargs[key] = item
        container.set_defaults(**safe_kwargs)

    def set_defaults_from_params(self, params: AttrDict, no_override=True) -> None:

        if len(self.children) == 0:
            # no children, so all params are good
            locals = params.leaf_copy()
        else:
            # keys that don't directly map to a known child
            locals = AttrDict()
            for k in params.keys():
                if k not in self.children.keys():
                    locals[k] = params[k]
            locals.pprint()

        if no_override:
            GroupedArgumentParser.set_default_no_override(self, **locals.as_dict())
        else:
            self.set_defaults(**locals.as_dict())

        for child_name, child in self.children.items():
            child_params = params << child_name
            if child_params is not None and isinstance(child_params, AttrDict):
                if isinstance(child, GroupedArgumentParser):
                    child.set_defaults_from_params(child_params, no_override=no_override)
                else:
                    if no_override:
                        GroupedArgumentParser.set_default_no_override(child, **child_params.as_dict())
                    else:
                        child.set_defaults(**child_params.as_dict())

    @staticmethod
    def to_attrs(namespace, out: AttrDict = AttrDict()) -> AttrDict:
        # locals
        for key in namespace.__dict__:
            # print(out, key)
            attr = getattr(namespace, key)
            if key not in ["child_namespaces", "child_unknowns"]:
                out[key] = attr

        # recurse
        if hasattr(namespace, "child_namespaces"):
            for child_name, child_ns in namespace.child_namespaces.items():
                # print("starting", child_name, out)
                # print(child_name, child_ns)
                GroupedArgumentParser.to_attrs(child_ns, out=out[child_name])

        return out

    def get_child(self, item: str) -> argparse.ArgumentParser:
        if '/' in item:
            item_split = item.split('/')
            curr_item = item_split[0]
            next_item = '/'.join(item_split[1:])
            assert isinstance(self.children[curr_item], GroupedArgumentParser)
            return self.children[curr_item].get_child(next_item)
        else:
            return self.children[item]

    def _solve_local(self, local_arg_strings) -> Tuple[List[str], bool]:
        self._solved = True
        return local_arg_strings, False

    def solve(self, args=None):
        # call this after adding all the children.

        assert not self._solved, "Calling solve twice is not allowed!"
        assert len(self._solved_nested_args) == 0, "Calling solve twice is not allowed!"
        assert len(self._solved_local_args) == 0, "Calling solve twice is not allowed!"

        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        local_arg_strings, nested_child_arg_strings, raw_child_strings = self._split_local_and_groups(args, ret_raw_children=True)
        # solve whatever this level needs. Might also add new children
        local_arg_strings, added_new_children = self._solve_local(local_arg_strings)

        # if new children were added, we need re-split the children args
        if added_new_children:
            _, nested_child_arg_strings, raw_child_strings = self._split_local_and_groups(args, ret_raw_children=True)

        # children solve
        for child_name, child in self.children.items():
            if isinstance(child, GroupedArgumentParser):
                child.solve(args=nested_child_arg_strings[child_name])
                assert child._solved, f"Child {child_name} was not able to solve!"

        # save these
        self._solved_local_args.extend(local_arg_strings)
        self._solved_nested_args.update(nested_child_arg_strings)
        self._solved_nested_raw_args.extend(raw_child_strings)


class LoadableGroupedArgumentParser(GroupedArgumentParser):
    """
    A specific type of grouped argument parser
    where the first argument is a file that exports the true parser:

    params = {
        declare_arguments: This enables python configs to add their own arguments
        process_params: This enables python configs to modify the params once loaded
            (e.g. using the user specified params to create a more specific config)
        (optional): submodules: AttrDict of LoadableGroupedArgumentParsers to add as children.
    }

    NOTE: process_params() is not called in the argument parser. Other scripts are responsible for calling this.
    """
    _params: AttrDict = AttrDict()

    def __init__(self, file_required: bool = True, allow_override=True, prepend_args=(), optional=False, submodules={}, **kwargs):
        self._file_required = file_required
        self._allow_override = allow_override
        self._prepend_args = list(prepend_args)
        self._optional = optional
        self._init_submodules = dict(submodules)  # will override file specified names with this.
        super(LoadableGroupedArgumentParser, self).__init__(**kwargs)

    def _load_parser(self, parser_path):
        # loads the parser from file, and declares its arguments
        if not self._allow_override:
            assert self._params.is_empty(), "Parser already loaded!"
        elif not self._params.is_empty():
            # actually override all the previously added actions
            self._actions.clear()
            self._option_string_actions.clear()
            self._action_groups.clear()
            self._mutually_exclusive_groups.clear()
            self._defaults.clear()
            self._has_negative_number_optionals.clear()

        # should contain at least {declare_arguments, process_params}
        parser_params = import_config(parser_path)
        # adds arguments from the file.
        self._params = parser_params.leaf_copy()
        self._submodules = get_with_default(parser_params, "submodules", AttrDict())
        for n, s in self._submodules.leaf_items():
            loc = parser_path
            if n in self._init_submodules.keys():
                s = self._submodules[n] = self._init_submodules[n]
                loc = "Parser.__init__"

            assert isinstance(s, LoadableGroupedArgumentParser), \
                f"Submodule {n} provided in {loc} is not a loadable arg parser, instead: {type(s)}"
            self.add_child(n, s)
        return (parser_params >> "declare_arguments")(self)

    def _solve_local(self, local_arg_strings) -> Tuple[List[str], bool]:
        initial_child_len = len(self.children)

        if self._optional and len(local_arg_strings) == 0:
            self._params = AttrDict.from_dict(self._defaults)
            logger.warn(f"Optional Group Missing!")
        else:
            if self._file_required:
                # LOAD parser
                assert len(local_arg_strings) >= 1, f"Must pass in at least one argument to Loadable Parser! {local_arg_strings}"
                # fill in any known macros since this arg won't be processed later
                file_name = local_arg_strings[0]
            else:
                # only load if there are args and first arg is a python file
                file_name = local_arg_strings[0] if len(local_arg_strings) >= 1 and local_arg_strings[0].endswith(".py") else None

            if file_name is not None:
                self._substitute_macros([local_arg_strings[0]]) if self._macros is not None else local_arg_strings[0]
                assert file_name.endswith(".py"), f"First argument must be a Python file: {file_name}"
                group_parser_path = file_path_with_default_dir(file_name, FileManager.base_dir)
                logger.debug(f"Loading group parser file: {group_parser_path}")
                # loading the parser might add new children.
                self._load_parser(group_parser_path)
                local_arg_strings = local_arg_strings[1:]

        self._solved = True
        # the remaining strings (1:, if file specified) and whatever hardcoded defaults
        return self._prepend_args + local_arg_strings, len(self.children) > initial_child_len

    @property
    def params(self):
        # mutable
        return self._params


class LoadedGroupedArgumentParser(LoadableGroupedArgumentParser):
    """
    File name specified from code, not command line.
    """
    def __init__(self, file_name, **kwargs):
        # depending on allow_overrides, will error on solve() if file is specified again.
        super(LoadedGroupedArgumentParser, self).__init__(file_required=False, **kwargs)
        # sets up arguments from the file
        self._load_parser(file_path_with_default_dir(file_name, FileManager.base_dir))


def matches_struct(prms: AttrDict, struct: AttrDict):
    all_required_nodes_and_leafs = struct.list_node_leaf_keys()
    return prms.has_node_leaf_keys(all_required_nodes_and_leafs)


def add_loadable_if_not_present(parser: GroupedArgumentParser, name, common_params, required_struct=AttrDict(cls=None, params=AttrDict()), optional=False):
    present = False
    child = None
    if name in common_params.keys():
        # loader already specified, we don't need to instantiate again
        if isinstance(common_params[name], LoadableGroupedArgumentParser):
            child = common_params[name]
        # instantiable
        elif matches_struct(common_params[name], required_struct):
            present = True

    child = child if child is not None else LoadableGroupedArgumentParser(file_required=not present, optional=optional)

    parser.add_child(name, child)
    if not present:
        common_params[name] = AttrDict()  # initialize empty if not present.
    return present


if __name__ == '__main__':
    parser = MacroEnabledArgumentParser(macros={'HOME': '/home', 'FINAL': '/other_folder', 'num': 1}, macro_prefix="!")
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    parser.add_argument('--another', type=str, default='none')
    parser.add_argument('--an_int', type=int, default=0)

    example = parser.parse_args(
        "!{HOME}/test/!{FINAL}/file.txt !{HOME}/test.py --another umm_!{num} --an_int !{num}".split())
    print(example)

    ### an example of nested parsers
    second_parser = MacroEnabledArgumentParser(macro_prefix="!")
    second_parser.add_argument('file1', type=str)
    second_parser.add_argument('file2', type=str)
    second_parser.add_argument('--another', type=str, default='none')
    second_parser.add_argument('--an_int', type=int, default=0)

    parent_parser = GroupedArgumentParser(macros={'HOME': '/home', 'FINAL': '/other_folder', 'num': 1}, macro_prefix="!")
    parent_parser.add_argument('file1', type=str)
    parent_parser.add_argument('--another', type=str, default='none')

    parent_parser.add_child("test_child", parser)
    parent_parser.add_child("second_child", second_parser)

    args = parent_parser.parse_args("group_file.txt --another yo_!{num} "
                                    "%test_child !{HOME}/test/!{FINAL}/file.txt !{HOME}/test.py --another umm_!{num} --an_int !{num} "
                                    "%second_child !{HOME}/second/!{FINAL}/file.txt !{HOME}/test2.py --another umm2_!{num} --an_int !{num}".split())

    # print AttrDict
    GroupedArgumentParser.to_attrs(args).pprint()
