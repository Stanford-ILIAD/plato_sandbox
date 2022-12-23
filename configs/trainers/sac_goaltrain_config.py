"""
Train but no evaluations.

"""
from argparse import ArgumentParser
from typing import Iterable

import torch

from sbrl.metrics.metric import ExtractMetric
from sbrl.metrics.tracker import BufferedTracker
from sbrl.sandbox.new_trainer.goal_trainer import GoalTrainer
from sbrl.sandbox.new_trainer.sac_optimizer import SACOptimizer
from sbrl.utils.config_utils import prepend_next_process_env_step_output_fn
from sbrl.utils.math_utils import round_to_n
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
    parser.add_argument("--train_every_n_steps", type=int, nargs="+", default=1)
    parser.add_argument("--holdout_every_n_steps", type=int, nargs="+", default=0)
    parser.add_argument("--step_env_every_n_steps", type=int, default=None)
    parser.add_argument("--step_env_holdout_every_n_steps", type=int, default=None)
    parser.add_argument("--add_to_data_every_n_goals", type=int, nargs="+", default=0)  # default means only add at episode ends
    parser.add_argument("--log_every_n_steps", type=int, default=1000)
    parser.add_argument("--save_model_every_n_steps", type=int, default=20000)
    parser.add_argument("--save_data_every_n_steps", type=int, nargs="+", default=20000)
    parser.add_argument("--block_training_for_n_steps", type=int, default=None)
    parser.add_argument("--no_data_saving", action='store_true', help="True = no replay buffer")
    parser.add_argument("--random_policy_on_first_n_steps", type=int, default=None)
    parser.add_argument("--eval_every_n_steps", type=int, default=500)
    # parser.add_argument("--episode_return_buffer_len", type=int, default=None)
    parser.add_argument("--reload_stats_every_n_env_steps", type=int, default=None)  # default will be once at (block_training_for_n_steps)
    parser.add_argument("--reload_stats_n_times", type=int, default=1)  # first time
    parser.add_argument("--checkpoint_model_file", type=str, default="model.pt")
    parser.add_argument("--use_absolute_lr", action='store_true', help="True = learning rate won't be multiplied by batch_size")

    parser.add_argument("--reward_reduction", type=str, choices=['sum', 'max'], default="sum")
    parser.add_argument("--write_returns_every_n_env_steps", type=int, default=50,
                        help="writes tracker keys every n env steps")
    parser.add_argument("--return_buffer_len", type=int, default=10, help="writes tracker keys every n env steps")

    # specific to SAC
    parser.add_argument("--max_grad_norm", type=float, default=None)
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

    num_envs = get_with_default(common_params, "env_train/num_envs", 1)
    block_train_n_steps = get_with_default(prms, "block_training_for_n_steps", 10 * (common_params >> "batch_size"))
    reload_stats_every_n_env_steps = get_with_default(prms, "reload_stats_every_n_env_steps", max(block_train_n_steps, 1000))
    random_policy_on_first_n_steps = get_with_default(prms, "random_policy_on_first_n_steps", block_train_n_steps)
    step_env_every_n_steps = get_with_default(prms, "step_env_every_n_steps", num_envs)
    step_env_holdout_every_n_steps = get_with_default(prms, "step_env_holdout_every_n_steps", 0)  # default, no holdout env
    # episode_return_buffer_len = get_with_default(prms, "episode_return_buffer_len", max(num_envs, 4))

    # ORDER MATTERS
    opt_names = ["critic", "actor", "alpha"]
    m_names = ["critic", "actor", "log_alpha"]  # sub-model names
    get_params = lambda model, name: [getattr(model, name)] if name == "log_alpha" else getattr(model, name).parameters()

    train_every_n_steps = prms >> "train_every_n_steps"

    optimizer = d(
        cls=SACOptimizer,
        params=d(
            num_optimizers=3,
            max_grad_norm=prms >> "max_grad_norm",
            critic_tau=prms >> "critic_tau",
            actor_update_every_n_steps=prms >> "actor_freq",
            critic_target_update_every_n_steps=prms >> "critic_target_freq",

            get_optimizer=lambda sac_model, i: torch.optim.Adam(get_params(sac_model, m_names[i]),
                                                                lr=lrs >> f"{opt_names[i]}_lr",
                                                                betas=prms >> f"{opt_names[i]}_betas")
        )
    )

    # trackers
    params = d(
        buffer_len=prms >> "return_buffer_len",  # return window is last N episodes.
        buffer_freq=0,  # only clear on resets
        time_agg_fn=(lambda k, a, b: a + b) if (prms >> "reward_reduction") == "sum" else (lambda k, a, b: max(a, b)),
        # sum rewards
        metric=ExtractMetric('returns', key='reward', source=1),  # source is outputs.
        tracked_names=['returns'],
    )

    reward_tracker = BufferedTracker(params)

    trackers = d.from_dict({
        'env_train/returns': reward_tracker,
        'env_holdout/returns': reward_tracker.duplicate()
    })

    common_params[group_name] = common_params[group_name] & d(
        cls=GoalTrainer,
        params=d(
            optimizer=optimizer,
            max_steps=1e8 * (max(train_every_n_steps) if isinstance(train_every_n_steps, Iterable) else train_every_n_steps),
            train_every_n_steps=prms >> "train_every_n_steps",
            block_train_on_first_n_steps=block_train_n_steps,
            random_policy_on_first_n_steps=random_policy_on_first_n_steps,
            step_train_env_every_n_steps=step_env_every_n_steps,
            step_holdout_env_every_n_steps=step_env_holdout_every_n_steps,
            add_to_data_train_every_n_goals=prms >> "add_to_data_every_n_goals",
            add_to_data_holdout_every_n_goals=prms >> "add_to_data_every_n_goals",
            holdout_every_n_steps=prms >> "holdout_every_n_steps",
            eval_every_n_steps=prms >> "eval_every_n_steps",  # TODO remove, this doesn't do anything
            no_data_saving=prms >> "no_data_saving",
            # write_average_episode_returns_every_n_env_steps=500,  # number of ENV steps (not episodes)
            # max_train_data_steps=0,
            # max_holdout_data_steps=0,

            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=reload_stats_every_n_env_steps,
            reload_statistics_n_times=prms >> "reload_stats_n_times",
            log_every_n_steps=prms >> "log_every_n_steps",
            save_every_n_steps=prms >> "save_model_every_n_steps",
            save_data_train_every_n_steps=prms >> "save_data_every_n_steps",
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file=prms >> "checkpoint_model_file",  # specifies starting model to load
            save_checkpoints=True,

            trackers=trackers,
            write_average_episode_returns_every_n_env_steps=prms >> "write_returns_every_n_env_steps",  # number of ENV steps (not episodes)
            # base_scheduler=get_schedule,


            process_env_step_output_fn=prepend_next_process_env_step_output_fn,  # VERY IMPORTANT, populates next outputs, rewards, etc
        )
    )

    common_params.exp_name = wrap_get_exp_name(group_name, common_params >> "exp_name")
    common_params[group_name].random_policy_on_first_n_steps = random_policy_on_first_n_steps  # for exp_name fn
    return common_params

def wrap_get_exp_name(group_name, exp_name_fn):
    # modifies the experiment name with specific things from this level
    def get_exp_name(common_params):
        prms = common_params >> group_name
        mult = 1 if prms >> 'use_absolute_lr' else common_params >> 'batch_size'
        hr_aclr = str(round_to_n((prms >> "actor_lr") * mult)).replace('.', '_')
        hr_crlr = str(round_to_n((prms >> "critic_lr") * mult)).replace('.', '_')
        hr_allr = str(round_to_n((prms >> "alpha_lr") * mult)).replace('.', '_')
        NAME = exp_name_fn(common_params) + f"_lr-ac{hr_aclr}-cr{hr_crlr}-al{hr_allr}"
        NAME += "_rndinit" + str(prms >> "random_policy_on_first_n_steps")
        return NAME
    return get_exp_name

params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
