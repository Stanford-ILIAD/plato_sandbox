# Configurations

Config files specify experimental parameters, and each script will require different components be specified, in a specific order. For example, `scripts/train.py` requires the following groups:
- env_spec
- env_train
- dataset_train
- dataset_holdout
- model
- policy
- trainer

We split config loading into a `declare` phase, a `process` phase, and an `instantiate` phase.

### Declaring

Each group specifies a parser, which will get required and optional arguments from the command line or from files. We use the `argparse` package to parse the command line, using the custom `sbrl.experiments.grouped_parser.GroupedArgumentParser`, which reads group params in the following format (with `%` as the group prefix character):

`python <script> <args> %group_name <args> %next_group_name <args> ...`

All scripts defined in `scripts/` should support `LoadableArgumentParser`, which requires you to specify a single argument for each group that points to a file that will define the true parameters, with syntax:

`python <script> <args> %group1_name configs/group1_config.py <extra_args> ...`

Note that additional args can be specified for each loadable parser after the python file (e.g., `%group1 group1_config.py --render --debug`). Also note that if these extra arguments are specified in a .txt file, you can incorporate them by prefixing the filename by `@` (e.g., `%group1 group1_config.py @args.txt`, where `args.txt` contains `--render --debug`).

Also note that different groups exist in their own "containers," so do not worry about duplicate names across group boundaries.

Finally, many scripts require a base config, which defines a set of constants or functions that will be useful during processing.

### Processing

Once default arguments have been parsed from all groups, the calling script will process each group one by one. Groups will take in the global processed parameters until now, and make their own contributions.
Processing usually involves converting the set of arguments into an instantiable specification (e.g., in `train.py`, the `env_spec` group processing will specify what `sbrl.envs.env_spec.EnvSpec` to instantiate in python, and with what parameters).

### Instantiating

The script does this part, taking the final processed params after the last group, and instantiating all relevant python classes. Then, the script will do some work with each component.

## Config specification

There are three types of "two" files. 
The first is a base config, which can pre-specify or aggregate some groups and common arguments for modules to share.
The second is a module config, which will specify `declare_arguments` and `process_params` functions for the given module.

### Base config

This config is the "root" config of the experiment, which might do things like set some initial arguments for groups that it expects.
An example with comments can be seen in `configs/example/base_gym_config.py`, replicated here for convenience:

```python
## configs/example/base_gym_config.py ##

from argparse import ArgumentParser

from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser, GroupedArgumentParser
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.python_utils import AttrDict

# this has helper functions for the current set of experiments.
from configs.example import utils

# Set up a parser for just the base config parameters, these should be "global" fields that many modules will need.
parser = ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
args = parser.parse_args(get_config_args())


def get_exp_name(common_params):
    """
    The loading process will call this to figure out the experiment name (which determines where your experiment saves).
    """
    # lazily executed, gets the name from the overall common_params.
    # this function might be chained by future experiments to add on more details to the name.
    env_type = common_params >> "env_train/env_type"  # required for gym envs
    env_type_hr = env_type.lower()
    return f"gym/{env_type_hr}_b{common_params >> 'batch_size'}_h{common_params >> 'horizon'}"


# GLOBAL parameters, which will be the starting point for "common_params" seen in sub-groups
params = GroupedArgumentParser.to_attrs(args) & AttrDict(exp_name=get_exp_name, utils=utils)

# example of how to set default groups, here with the gym "env_spec" and "env_train"
# some useful flags for this parser:
#   prepend_args: Some initial arguments to pass in to the config, for more precise defaults.
#   allow_override: If False, this group's arguments will be FIXED.
params.env_spec = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_spec_config.py")
params.env_train = LoadedGroupedArgumentParser(file_name="configs/gym/gym_env_config.py")
```

**NOTE**: While arguments for each group are in their own name spaces, and thus can conflict without issue, the arguments provided to the base config should NOT conflict with arguments passed into the calling script.

For example, with the base config above, the script should not have any arguments for `device`, `horizon`, or `batch_size`.

**TODO**: In the future we will fix this bug by treating the base config as its own group, but for now be aware that this bug will not throw any errors...

### Module config

Module config files declare and parse a specific module (e.g., model). 
To support the "declare, process, instantiate" pipeline with `LoadableArgumentParser`, module config files should define two functions:

- `declare_arguments(parser)`: returns a parser that will get the specific arguments required for this config.
< the calling script will then call parser.parse_args()>
- `process_params(group_name, prms)`: takes in the parameterization (included by group in common_params), and makes any changes
                    to the global config based on all the loaded / default params.

Finally, they should export these as an `AttrDict` named `params`

### Directory format

`example_processed_params` functions will be specified within `src/`, to load each class with some rigid set of parameters. You can always override these in your own config files.
Calling these functions will be the responsibility of the `base_config.py` file. For training/eval experiments, we might structure the dir as follows:

```
- configs
    - <model_type>
        - specific_model_config.py
    - <env_type>
        - specific_env_config.py
    - exp_<exp_group>
        - base_config.py (with default env, env_spec, datasets)
        - all_args.txt
```

Here, `all_args.txt` might contain the command line call required to run an experiment. The base config can specify `LoadedGroupedArgumentParser` linking to specific files to load. These can be overrided from command line.


## Example: LfP

We will use Learning from Play, on the StackBlock2D env to demonstrate this config specification flow.

