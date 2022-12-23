"""
Train but no evaluations.

"""
from argparse import ArgumentParser

import torch

from sbrl.datasets.data_augmentation import DataAugmentation
from sbrl.trainers.trainer_sac import TrainerSAC
from sbrl.utils.config_utils import prepend_next_process_env_step_output_fn
from sbrl.utils.python_utils import AttrDict as d, get_with_default

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
AUGMENT_KEYS = []
AUGMENT_FNS = []


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--train_every_n_steps", type=int, default=1)
    parser.add_argument("--holdout_every_n_steps", type=int, default=0)
    parser.add_argument("--step_env_every_n_steps", type=int, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=1000)
    parser.add_argument("--save_model_every_n_steps", type=int, default=20000)
    parser.add_argument("--save_data_every_n_steps", type=int, default=20000)
    parser.add_argument("--block_training_for_n_steps", type=int, default=None)
    parser.add_argument("--random_policy_on_first_n_steps", type=int, default=None)
    parser.add_argument("--episode_return_buffer_len", type=int, default=None)
    parser.add_argument("--reload_stats_every_n_steps", type=int, default=1000)
    parser.add_argument("--reload_stats_n_times", type=int, default=10)  # first 10 times
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--checkpoint_model_file", type=str, default="model.pt")
    parser.add_argument("--use_absolute_lr", action='store_true', help="True = learning rate won't be multiplied by batch_size")

    # specific to SAC
    parser.add_argument("--actor_lr", type=float, default=1e-4 / 1024)  # lr per batch size
    parser.add_argument("--critic_lr", type=float, default=1e-4 / 1024)
    parser.add_argument("--alpha_lr", type=float, default=1e-4 / 1024)
    parser.add_argument("--actor_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--critic_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--alpha_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--critic_tau", type=float, default=0.005)
    parser.add_argument("--actor_freq", type=float, default=1, help="How often to train actor, relative to critic.")
    parser.add_argument("--critic_target_freq", type=float, default=2, help="How often to update critic target, relative to critic.")
    #
    # parser.add_argument("--decay", type=float, default=5e-6)
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
    lrs = d()
    for lr_name in ['actor_lr', 'critic_lr', 'alpha_lr']:
        lr = prms >> lr_name
        lrs[lr_name] = lr if prms >> "use_absolute_lr" else lr * batch_size

    num_envs = common_params >> "env_train/num_envs"
    block_train_n_steps = get_with_default(prms, "block_training_for_n_steps", 10 * (common_params >> "batch_size"))
    random_policy_on_first_n_steps = get_with_default(prms, "random_policy_on_first_n_steps", block_train_n_steps)
    step_env_every_n_steps = get_with_default(prms, "step_env_every_n_steps", num_envs)
    episode_return_buffer_len = get_with_default(prms, "episode_return_buffer_len", max(num_envs, 4))

    common_params[group_name] = common_params[group_name] & d(
        cls=TrainerSAC,
        params=d(
            critic_tau=prms >> "critic_tau",
            actor_update_every_n_train_steps=prms >> "actor_freq",
            critic_target_update_every_n_train_steps=prms >> "critic_target_freq",
            # base
            max_steps=1e8 * (prms >> "train_every_n_steps"),
            train_every_n_steps=prms >> "train_every_n_steps",
            block_train_on_first_n_steps=block_train_n_steps,
            random_policy_on_first_n_steps=random_policy_on_first_n_steps,
            step_train_env_every_n_steps=step_env_every_n_steps,
            step_holdout_env_every_n_steps=0,
            holdout_every_n_steps=prms >> "holdout_every_n_steps",
            episode_return_buffer_len=episode_return_buffer_len,  # sliding average window over last N episode rewards
            write_average_episode_returns_every_n_env_steps=500,  # number of ENV steps (not episodes)
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
            train_do_data_augmentation=False,

            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=prms >> "reload_stats_every_n_steps",
            reload_statistics_n_times=prms >> "reload_stats_n_times",
            log_every_n_steps=prms >> "log_every_n_steps",
            save_every_n_steps=prms >> "save_model_every_n_steps",
            save_data_train_every_n_steps=prms >> "save_data_every_n_steps",
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file=prms >> "checkpoint_model_file",  # specifies starting model to load
            save_checkpoints=True,

            actor_optimizer=lambda p: torch.optim.Adam(p, lr=lrs >> "actor_lr", betas=prms >> "actor_betas"),
            critic_optimizer=lambda p: torch.optim.Adam(p, lr=lrs >> "critic_lr", betas=prms >> "critic_betas"),
            log_alpha_optimizer=lambda p: torch.optim.Adam(p, lr=lrs >> "alpha_lr", betas=prms >> "alpha_betas"),
            # base_scheduler=get_schedule,

            process_env_step_output_fn=prepend_next_process_env_step_output_fn,  # VERY IMPORTANT, populates next outputs, rewards, etc
        )
    )

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
