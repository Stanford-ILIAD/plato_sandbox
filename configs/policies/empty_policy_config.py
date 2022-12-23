"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=Policy,
        params=AttrDict()
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
