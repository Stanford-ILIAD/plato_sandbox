"""
An example of how to integrate multiple separate parameterization files.
    this is the base set that doesn't change experiment to experiment.
    We can include default files by specifying loadable config files under each group here.
"""
from argparse import ArgumentParser

from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, default="test")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args(get_config_args())

params = GroupedArgumentParser.to_attrs(args)  #exp_name=get_exp_name, utils=utils)
