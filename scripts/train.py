import os
import sys

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.script_utils import load_standard_ml_config

if __name__ == '__main__':

    # things we can use from command line
    macros = {}

    grouped_parser = GroupedArgumentParser(macros=macros, fromfile_prefix_chars=['@'])
    grouped_parser.add_argument('config', type=str, help="common params for all modules.")
    grouped_parser.add_argument('--continue', action='store_true')
    grouped_parser.add_argument('--local_rank', type=int, default=None)
    grouped_parser.add_argument('--print_all', action='store_true')
    grouped_parser.add_argument('--do_holdout_env', action='store_true')
    local_args, unknown = grouped_parser.parse_local_args()
    # this defines the required command line groups, and the defaults
    # if in this list, we look for it

    logger.debug(f"Args: \n{' '.join(sys.argv)}")

    if local_args.local_rank is not None:
        import torch
        torch.cuda.set_device(local_args.local_rank)

    ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'dataset_holdout', 'policy', 'trainer']
    if local_args.do_holdout_env:
        ordered_modules.append('env_train_holdout')

    config_fname = os.path.abspath(local_args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules, debug=local_args.print_all)

    logger.debug(f"Using: {exp_name}")
    file_manager = ExperimentFileManager(exp_name,
                                         is_continue=getattr(args, 'continue'),
                                         log_fname='log_train.txt',
                                         config_fname=local_args.config,
                                         extra_args=unknown + grouped_parser.raw_nested_args)

    # instantiate classes from the params
    env_spec = params.env_spec.cls(params >> "env_spec/params")
    env_train = params.env_train.cls(params >> "env_train/params", env_spec)
    env_holdout = None if "env_holdout" not in params.keys() else params.env_holdout.cls(params >> "env_holdout/params", env_spec)

    dataset_train = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
    dataset_holdout = params.dataset_holdout.cls(params.dataset_holdout.params, env_spec, file_manager)

    # making model
    model = params.model.cls(params >> "model/params", env_spec, dataset_train)

    # policy
    policy = params.policy.cls(params.policy.params, env_spec, env=env_train)

    # trainer
    trainer = params.trainer.cls(params.trainer.params,
                                 file_manager=file_manager,
                                 model=model,
                                 policy=policy,
                                 dataset_train=dataset_train,
                                 dataset_holdout=dataset_holdout,
                                 env_train=env_train,
                                 env_holdout=env_holdout)

    # run training
    trainer.run()
