"""
TODO
"""

import os

import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict
from sbrl.utils.script_utils import load_standard_ml_config

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str)
grouped_parser.add_argument('--model_file', type=str)
grouped_parser.add_argument('--no_model_file', action="store_true")
grouped_parser.add_argument('--random_policy', action="store_true")
grouped_parser.add_argument('--print_last_obs', action="store_true")
grouped_parser.add_argument('--print_policy_name', action="store_true")
local_args, unknown = grouped_parser.parse_local_args()
# this defines the required command line groups, and the defaults
# if in this list, we look for it

ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'policy']

config_fname = os.path.abspath(local_args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                 debug=False)

file_manager = ExperimentFileManager(exp_name, is_continue=True)

if args.model_file is not None:
    model_fname = args.model_file
    model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current eval: {}".format(model_fname))

env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)
env = params.env_train.cls(params.env_train.params, env_spec)
policy = params.policy.cls(params.policy.params, env_spec, env=env)

### warm start the planner
presets = AttrDict(do_gravity_compensation=True)
obs, goal = env.reset()
policy.reset_policy(next_obs=obs, next_goal=goal)
policy.warm_start(model, obs, goal)

### restore model
if not args.no_model_file:
    model.restore_from_file(model_fname)

### eval loop
done = [False]
rew = 0.
i = 0

while True:
    if done[0] or policy.is_terminated(model, obs, goal):
        rew = 0
        i = 0
        obs, goal = env.reset()
        policy.reset_policy(next_obs=obs, next_goal=goal)

    # two empty axes for (batch_size, horizon)
    expanded_obs = obs.leaf_apply(lambda arr: arr[:, None])
    expanded_goal = goal.leaf_apply(lambda arr: arr[:, None])
    with torch.no_grad():
        if args.random_policy:
            action = policy.get_random_action(model, expanded_obs, expanded_goal)
        else:
            action = policy.get_action(model, expanded_obs, expanded_goal)

        if i == 0 and args.print_policy_name and action.has_leaf_key("policy_name"):
            logger.info(f"Policy: {action.policy_name.item()}")

    obs, goal, done = env.step(action)
    i += 1

    if done and args.print_last_obs:
        logger.debug("Last obs:")
        obs.pprint()
