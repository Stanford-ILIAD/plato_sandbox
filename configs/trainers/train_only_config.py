"""
Train but no evaluations.

"""
from argparse import ArgumentParser

import numpy as np
import torch

from sbrl.datasets.data_augmentation import DataAugmentation
from sbrl.trainers.trainer import Trainer
from sbrl.utils.config_utils import default_process_env_step_output_fn
from sbrl.utils.python_utils import AttrDict as d
# TRAIN_EVERY_N_STEPS = 1
# HOLDOUT_EVERY_N_STEPS = 50
# STEP_ENV_EVERY_N_STEPS = 0  # epochs, effectively
# BLOCK_TRAINING_FOR_N_STEPS = 0  # STEP_ENV_EVERY_N_STEPS * math.ceil(BATCH_SIZE / NUM_ENVS)
# EPISODE_RETURN_BUFFER_LEN = 1  # averaging window
# MAX_GRAD_NORM = None
#
# RELOAD_STATS_EVERY_N_ENV_STEPS = 0  # 20 * 10 * NUM_ENVS * STEP_ENV_EVERY_N_STEPS
# LOG_EVERY_N = 1000
# SAVE_MODEL_EVERY_N = 20000
# SAVE_DATA_EVERY_N = 0
from sbrl.utils.torch_utils import to_torch


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--train_every_n_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=float, default=1e7)
    parser.add_argument("--holdout_every_n_steps", type=int, default=50)
    parser.add_argument("--step_env_every_n_steps", type=int, default=0)
    parser.add_argument("--log_every_n_steps", type=int, default=1000)
    parser.add_argument("--save_model_every_n_steps", type=int, default=20000)
    parser.add_argument("--save_checkpoint_every_n_steps", type=int, default=100000)
    parser.add_argument("--save_data_every_n_steps", type=int, default=0)
    parser.add_argument("--block_training_for_n_steps", type=int, default=0)
    parser.add_argument("--episode_return_buffer_len", type=int, default=1)
    parser.add_argument("--reload_stats_every_n_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--checkpoint_model_file", type=str, default="model.pt")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_absolute_lr", action='store_true', help="True = learning rate won't be multiplied by batch")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--augment_keys", type=str, nargs="*", default=[])
    parser.add_argument("--augment_stds", type=float, nargs="*", default=[],
                        help="gaussian noise with these std's. broadcast-able to true shape")
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert 'trainer' in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    # access to all the params for the current experiment here.
    prms = common_params >> group_name

    batch_size = common_params >> "batch_size"
    lr = prms >> 'learning_rate'
    lr = lr if prms >> "use_absolute_lr" else lr * batch_size

    # default type of data augmentation (adds noise)
    do_data_aug = False
    AUGMENT_KEYS = prms >> "augment_keys"
    AUGMENT_FNS = []
    aug_values = prms >> "augment_stds"
    assert len(aug_values) == len(AUGMENT_KEYS), [aug_values, AUGMENT_KEYS]
    if len(AUGMENT_KEYS) > 0:
        torch_aug_values = [to_torch(np.asarray(val)[None], device=common_params >> "device") for val in aug_values]

        def get_augment_fn(std):
            def fn(arr, ** kwargs):
                return arr + std * torch.randn_like(arr)
            return fn
        AUGMENT_FNS = [get_augment_fn(std) for std in torch_aug_values]
        do_data_aug = True

    common_params[group_name] = common_params[group_name] & d(
        cls=Trainer,
        params=d(
            # base
            max_steps=int(prms >> "max_train_steps") * (prms >> "train_every_n_steps"),
            train_every_n_steps=prms >> "train_every_n_steps",
            block_train_on_first_n_steps=prms >> "block_training_for_n_steps",
            step_train_env_every_n_steps=prms >> "step_env_every_n_steps",
            step_holdout_env_every_n_steps=0,
            holdout_every_n_steps=prms >> "holdout_every_n_steps",
            episode_return_buffer_len=prms >> "episode_return_buffer_len",  # sliding average window over last N episode rewards
            write_average_episode_returns_every_n_env_steps=20,  # number of ENV steps (not episodes)
            max_grad_norm=prms >> "max_grad_norm",
            # max_train_data_steps=0,
            # max_holdout_data_steps=0,

            data_augmentation_params=d(
                cls=DataAugmentation,
                params=d(
                    augment_keys=AUGMENT_KEYS,
                    augment_fns=AUGMENT_FNS,  # TODO
                )
            ),
            train_do_data_augmentation=do_data_aug,

            torchify_dataset=False,  # TODO
            torchify_device="cpu",  # careful with memory here...  # TODO

            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=prms >> "reload_stats_every_n_steps",
            log_every_n_steps=prms >> "log_every_n_steps",
            save_every_n_steps=prms >> "save_model_every_n_steps",
            save_checkpoint_every_n_steps=prms >> "save_checkpoint_every_n_steps",
            save_data_train_every_n_steps=prms >> "save_data_every_n_steps",
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file=prms >> "checkpoint_model_file",  # specifies starting model to load
            save_checkpoints=True,
            base_optimizer=lambda p: torch.optim.Adam(p, lr=lr, betas=prms >> "adam_betas", weight_decay=prms >> "decay"),
            # base_scheduler=get_schedule,

            process_env_step_output_fn=default_process_env_step_output_fn,  # VERY IMPORTANT, computes rewards, etc
        )
    )

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
