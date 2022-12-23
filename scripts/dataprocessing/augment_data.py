"""
This file is very similar to train; it's basically a subset of train
"""

import argparse
import os

import numpy as np
from sbrl.envs.bite_transfer import bt_utils

from sbrl.envs.env_spec import EnvSpec
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.models.model import Model
from sbrl.policies.policy import Policy
from sbrl.utils.file_utils import import_config, file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('output_config', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--no_model_file', action="store_true")

args = parser.parse_args()


# # TODO argumentize this, for now just edit this to do your data augmentation
# def process_fn(old_ep: AttrDict, model: Model, policy: Policy,
#                env_spec: EnvSpec, out_env_spec: EnvSpec):
#     # example adding noise
#     NUM_SAMPLED_EPS = 20  # 20x the data
#     POS_NOISE = 0.003 # 3 mm
#     F_NOISE = 0.05  # newtons
#     # DT = 0.1
#
#     all_new_eps = AttrDict()
#     for i in range(NUM_SAMPLED_EPS):
#         no = old_ep.fork_force.shape[0] + 1
#         delta_f = np.random.normal(0, F_NOISE, size=[no] + list(old_ep.fork_force.shape[1:]))
#         delta_pos = np.random.normal(0, POS_NOISE, size=[no] + list(old_ep.ee_position.shape[1:]))
#         # delta_act_lin = delta_pos / DT
#         ep_copy_i = old_ep.leaf_apply(lambda arr: np.copy(arr))
#         ep_copy_i.fork_force += delta_f[:-1]
#         ep_copy_i.next_fork_force += delta_f[1:]
#         ep_copy_i.ee_position += delta_pos[:-1]
#         ep_copy_i.next_ee_position += delta_pos[1:]
#
#         if "done" not in all_new_eps.leaf_keys():
#             all_new_eps = ep_copy_i  # first
#         else:
#             all_new_eps = AttrDict.leaf_combine_and_apply([all_new_eps, ep_copy_i],
#                               lambda vs: np.concatenate(vs, axis=0))
#     return all_new_eps


class_count = [0, 0, 0]
def process_fn(old_ep: AttrDict, model: Model, policy: Policy, env_spec: EnvSpec, out_env_spec: EnvSpec):
    OBS_TRAJ_MID = 0

    new_ep = AttrDict()
    for name in out_env_spec.names + out_env_spec.param_names + ["done", "rollout_timestep"]:
        new_ep[name] = old_ep[name]

    for name in out_env_spec.final_names:
        if name != "bite_trajectory_class":
            new_ep[name] = old_ep[name]

    assert "bite_trajectory_class" in out_env_spec.final_names
    assert "trajectory" in out_env_spec.param_names
    # 1mm tolerance for close pt, but the rest should be solidly past the mouth plane
    # if inmouth:
    #     print("inmouth")
    btc = bt_utils.bite_classify_basic(old_ep, OBS_TRAJ_MID)
    if "bite_trajectory_class" in old_ep.leaf_keys():
        if btc.value != old_ep.bite_trajectory_class[0,0]:
            import ipdb; ipdb.set_trace()  # this shouldn't happen
        new_ep.bite_trajectory_class = old_ep.bite_trajectory_class
    else:
        new_ep.bite_trajectory_class = np.array([[btc.value]])

    class_count[new_ep.bite_trajectory_class[0, 0] - 1] += 1

    new_ep.trajectory = old_ep.action[None]  # 1 x TRAJ LEN x ADIM
        # new_ep.final_orientation = old_ep.next_fork_orientation[OBS_TRAJ_MID][None]  # (1, 4)
    # if "done" not in all_new_eps.leaf_keys():
    #     logger.debug("First loading in process-fn")
    #     all_new_eps = new_ep  # first
    # else:
    #     all_new_eps = AttrDict.leaf_combine_and_apply([all_new_eps, new_ep], lambda vs: np.concatenate(vs, axis=0))
    return new_ep

# def process_fn(old_ep: AttrDict, model: Model, policy: Policy, env_spec: EnvSpec, out_env_spec: EnvSpec):
#     # add size x,y,z noise, along with orientation noise
#     NUM_SAMPLED_EPS = 1000
#
#     SIZE_SIG = 0.02
#     FORK_RPT_NOISE = np.deg2rad(5)  # degrees
#     FOOD_RPT_NOISE = np.deg2rad(5)
#
#     delta_size = np.random.normal(0, SIZE_SIG, size=(NUM_SAMPLED_EPS, 3))
#     delta_fork_rpt = np.random.normal(0, FORK_RPT_NOISE, size=(NUM_SAMPLED_EPS, 3))
#     delta_food_rpt = np.random.normal(0, FOOD_RPT_NOISE, size=(NUM_SAMPLED_EPS, 3))
#     delta_fork_quat = []
#     delta_food_quat = []
#     for i in range(NUM_SAMPLED_EPS):
#         delta_fork_quat.append(convert_rpt(*delta_fork_rpt[i]))
#         delta_food_quat.append(convert_rpt(*delta_food_rpt[i]))
#     delta_fork_quat = np.stack(delta_fork_quat)
#     delta_food_quat = np.stack(delta_food_quat)
#
#     all_new_eps = AttrDict()
#     for i in range(NUM_SAMPLED_EPS):
#         ep_copy_i = old_ep.leaf_apply(lambda arr: np.copy(arr))
#         ep_copy_i.food_size += delta_size[i]
#         ep_copy_i.next_food_size += delta_size[i]
#
#         # FOOD ON FORK
#         rpt = convert_quat_to_rpt(ep_copy_i.food_orientation_on_fork[0]) + delta_food_rpt[i]
#         ep_copy_i.food_orientation_on_fork[:] = convert_rpt(*rpt)[0]
#
#         rpt = convert_quat_to_rpt(ep_copy_i.next_food_orientation_on_fork[0]) + delta_food_rpt[i]
#         ep_copy_i.next_food_orientation_on_fork[:] = convert_rpt(*rpt)[0]
#
#         # EE / FORK
#         rpt = convert_quat_to_rpt(ep_copy_i.fork_orientation[0]) + delta_fork_rpt[i]
#         ep_copy_i.fork_orientation[:] = convert_rpt(*rpt)[0]
#
#         rpt = convert_quat_to_rpt(ep_copy_i.next_fork_orientation[0]) + delta_fork_rpt[i]
#         ep_copy_i.next_fork_orientation[:] = convert_rpt(*rpt)[0]
#
#         if "done" not in all_new_eps.leaf_keys():
#             all_new_eps = ep_copy_i  # first
#         else:
#             all_new_eps = AttrDict.leaf_combine_and_apply([all_new_eps, ep_copy_i], lambda vs: np.concatenate(vs, axis=0))
#
#     return all_new_eps

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

out_config_fname = os.path.abspath(args.output_config)
assert os.path.exists(out_config_fname), '{0} does not exist'.format(out_config_fname)
out_params = import_config(out_config_fname)

file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

out_path = file_path_with_default_dir(args.output_file, file_manager.exp_dir)

if args.model is not None:
    model_fname = os.path.abspath(args.model)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
elif "checkpoint_model_file" in params.trainer.params and params.trainer.params['checkpoint_model_file'] is not None:
    model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
    logger.debug("Using checkpoint model for current eval: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current eval: {}".format(model_fname))

exit_on_ctrl_c()
env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)
policy = params.policy.cls(params.policy.params, env_spec)

out_env_spec = out_params.env_spec.cls(out_params.env_spec.params)

assert params.dataset_train.params.get("horizon", 0)
params.dataset_train.params.file = args.file
dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
# params.dataset_train.params.file = None
# params.dataset_train.params.output_file = args.output_file
# dataset_output = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

### restore model
if not args.no_model_file:
    model.restore_from_file(model_fname)

all_names = env_spec.all_names + ['done', 'rollout_timestep']

out_ls = []

for i in range(dataset_input.get_num_episodes()):
    # two empty axes for (batch_size, horizon)
    datadict = dataset_input.get_episode(i, all_names)

    new_ep = process_fn(datadict, model, policy, env_spec, out_env_spec)
    out_ls.append(new_ep)

    if i % 10000 == 0:
        logger.debug("Loading sample %d" % i)

logger.debug("Combining all datasets")
out_datadict = AttrDict.leaf_combine_and_apply(out_ls, lambda vs: np.concatenate(vs, axis=0))

logger.debug("Saving dataset output to -> %s" % out_path)

logger.debug("Keys: " + str(list(out_datadict.leaf_keys())))
logger.debug("data len: %d" % len(out_datadict.done))
to_save = dict()
for name in out_datadict.leaf_keys():
    to_save[name] = out_datadict[name]

np.savez_compressed(out_path, **to_save)

# add any other print statements here:
logger.debug("Count per class " + str(class_count))
