"""
Example config module, loaded by LoadableGroupedArgumentParser.

 Config files should define two functions:

- declare_arguments(): returns a parser that will get the specific arguments required for this config.
<the calling script will then call parser.parse_args()>
- process_params(): takes in the parameterization (included by group in common_params), and makes any changes
                    to the global config based on all the loaded / default params.

Finally, they should export these as an AttrDict named "params"
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--device", "cuda")
    return parser


def process_params(group_name, common_params):
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
