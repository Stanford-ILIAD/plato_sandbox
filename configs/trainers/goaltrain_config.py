# TODO this is wrong rn
from argparse import ArgumentParser

import torch

from sbrl.experiments import logger
from sbrl.experiments.grouped_parser import LoadedGroupedArgumentParser
from sbrl.metrics.metric import ExtractMetric
from sbrl.metrics.tracker import BufferedTracker
from sbrl.sandbox.new_trainer.goal_trainer import GoalTrainer
from sbrl.sandbox.new_trainer.optimizer import SingleOptimizer
from sbrl.utils.python_utils import AttrDict as d, AttrDict, get_with_default

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

submodules = d(
    data_augment=LoadedGroupedArgumentParser('configs/datasets/data_augmentation/key_da_config.py'),
)

def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--max_train_steps", type=float, default=1e7)
    parser.add_argument("--train_every_n_steps", type=int, default=1)
    parser.add_argument("--holdout_every_n_steps", type=int, default=50)

    parser.add_argument("--step_env_every_n_steps", type=int, default=0)
    parser.add_argument("--step_env_n_per_step", type=int, default=1, help="inner steps at the above frequency")
    parser.add_argument("--step_env_holdout_every_n_steps", type=int, default=0)
    parser.add_argument("--step_env_holdout_n_per_step", type=int, default=1, help="inner steps at the above frequency")
    parser.add_argument("--rollout_env_every_n_steps", type=int, default=0)
    parser.add_argument("--rollout_env_n_per_step", type=int, default=1, help="full rollouts at the above frequency")
    parser.add_argument("--rollout_env_holdout_every_n_steps", type=int, default=0)
    parser.add_argument("--rollout_env_holdout_n_per_step", type=int, default=1, help="full rollouts at the above frequency")

    parser.add_argument("--add_to_data_every_n_goals", type=int, nargs="+", default=0)  # default means only add at episode ends
    parser.add_argument("--log_every_n_steps", type=int, default=1000)
    parser.add_argument("--save_model_every_n_steps", type=int, default=20000)
    parser.add_argument("--save_checkpoint_every_n_steps", type=int, default=100000)
    parser.add_argument("--save_data_every_n_steps", type=int, default=0)
    parser.add_argument("--block_training_for_n_steps", type=int, default=0)
    parser.add_argument("--block_env_for_n_steps", type=int, default=0)
    parser.add_argument("--no_data_saving", action='store_true', help="True = no replay buffer")
    parser.add_argument("--random_policy_on_first_n_steps", type=int, default=None)
    parser.add_argument("--write_returns_every_n_env_steps", type=int, default=50,
                        help="writes tracker keys every n env steps")
    parser.add_argument("--return_buffer_len", type=int, default=10, help="writes tracker keys every n env steps")

    # parser.add_argument("--episode_return_buffer_len", type=int, default=1)
    parser.add_argument("--reload_stats_every_n_steps", type=int, default=0)
    parser.add_argument("--reload_stats_n_times", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--checkpoint_model_file", type=str, default="model.pt")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_absolute_lr", action='store_true', help="True = learning rate won't be multiplied by batch")
    parser.add_argument("--save_best_model", action='store_true', help="based on rollout returns")
    parser.add_argument("--reward_reduction", type=str, choices=['sum', 'max'], default="sum")
    parser.add_argument("--group_by_key", type=str, default=None)
    parser.add_argument("--groups", type=int, nargs="*", default=None)

    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--decay", type=float, default=5e-6)
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

    # load the type of data augmentation (adds noise)
    common_params = (submodules['data_augment'].params >> "process_params")(f"{group_name}/data_augment", common_params)

    data_augmentation_params = (common_params >> f"{group_name}/data_augment") < ['cls', 'params']
    do_data_aug = not data_augmentation_params.is_empty()
    if do_data_aug:
        # wrap one level up for merging below
        data_augmentation_params = d(data_augmentation_params=data_augmentation_params)
        logger.debug("Including data augmentation...")

    params = AttrDict(
        buffer_len=prms >> "return_buffer_len",  # return window is last N episodes.
        buffer_freq=0,  # only clear on resets
        time_agg_fn=(lambda k, a, b: a + b) if (prms >> "reward_reduction") == "sum" else (lambda k, a, b: max(a, b)),  # sum rewards
        metric=ExtractMetric('returns', key='reward', source=1),  # source is outputs.
        tracked_names=['returns'],
    )

    if prms >> "group_by_key" is not None:
        split_key = prms >> "group_by_key"
        assert prms >> "groups" is not None

        groups = prms >> "groups"
        logger.info(f"Goal Trainer Return Tracker using groups: {groups}")

        def get_group_fn(inputs, outputs, model_outputs):
            split_val = (inputs >> split_key)
            device = split_val.device if isinstance(split_val, torch.Tensor) else "cpu"
            split_val = split_val.item()
            # -1 group is added as first group
            if split_val not in groups:
                return torch.tensor([0], device=device), len(groups) + 1
            return torch.tensor([groups.index(split_val) + 1], device=device), len(groups) + 1

        params.compute_grouped = True
        params.metric_get_group_fn = get_group_fn

    reward_tracker = BufferedTracker(params)

    trackers = AttrDict.from_dict({
        'env_train/returns': reward_tracker,
        'env_holdout/returns': reward_tracker.duplicate()
    })

    block_train_n_steps = prms >> "block_training_for_n_steps"
    random_policy_on_first_n_steps = get_with_default(prms, "random_policy_on_first_n_steps", block_train_n_steps)

    # specifies what will be used to determine the "best" model.
    track_best_name = "env_train/returns" if prms >> 'save_best_model' else None
    track_best_key = "returns" if prms >> 'save_best_model' else None

    common_params[group_name] = common_params[group_name] & d(
        cls=GoalTrainer,
        params=d(
            # base
            max_steps=int(prms >> "max_train_steps") * (prms >> "train_every_n_steps"),
            train_every_n_steps=prms >> "train_every_n_steps",
            block_train_on_first_n_steps=block_train_n_steps,
            block_env_on_first_n_steps=prms >> "block_env_for_n_steps",
            random_policy_on_first_n_steps=random_policy_on_first_n_steps,
            step_train_env_every_n_steps=prms >> "step_env_every_n_steps",
            step_train_env_n_per_step=prms >> "step_env_n_per_step",
            step_holdout_env_every_n_steps=prms >> "step_env_holdout_every_n_steps",
            step_holdout_env_n_per_step=prms >> "step_env_holdout_n_per_step",
            add_to_data_train_every_n_goals=prms >> "add_to_data_every_n_goals",
            add_to_data_holdout_every_n_goals=prms >> "add_to_data_every_n_goals",
            holdout_every_n_steps=prms >> "holdout_every_n_steps",
            rollout_train_env_every_n_steps=prms >> "rollout_env_every_n_steps",
            rollout_train_env_n_per_step=prms >> "rollout_env_n_per_step",
            rollout_holdout_env_every_n_steps=prms >> "rollout_env_holdout_every_n_steps",
            rollout_holdout_env_n_per_step=prms >> "rollout_env_holdout_n_per_step",
            no_data_saving=prms >> "no_data_saving",
            # episode_return_buffer_len=prms >> "episode_return_buffer_len",  # sliding average window over last N episode rewards
            # max_train_data_steps=0,
            # max_holdout_data_steps=0,

            train_do_data_augmentation=do_data_aug,

            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=prms >> "reload_stats_every_n_steps",
            reload_statistics_n_times=prms >> "reload_stats_n_times",
            log_every_n_steps=prms >> "log_every_n_steps",
            save_every_n_steps=prms >> "save_model_every_n_steps",
            save_checkpoint_every_n_steps=prms >> "save_checkpoint_every_n_steps",
            save_data_train_every_n_steps=prms >> "save_data_every_n_steps",
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file=prms >> "checkpoint_model_file",  # specifies starting model to load
            save_checkpoints=True,

            optimizer=d(
                cls=SingleOptimizer,
                params=d(
                    max_grad_norm=prms >> "max_grad_norm",
                    get_base_optimizer=lambda p: torch.optim.Adam(p, lr=lr, betas=prms >> "adam_betas", weight_decay=prms >> "decay"),
                )
            ),

            trackers=trackers,
            write_average_episode_returns_every_n_env_steps=prms >> "write_returns_every_n_env_steps",  # number of ENV steps (not episodes)
            # base_scheduler=get_schedule,

            # for saving the "best" model
            track_best_name=track_best_name,
            track_best_key=track_best_key,
        ) & data_augmentation_params
    )

    logger.debug(f"Trainer using Adam(lr={lr}, decay={prms.decay}, betas={prms.adam_betas})")

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
    submodules=submodules,
)
