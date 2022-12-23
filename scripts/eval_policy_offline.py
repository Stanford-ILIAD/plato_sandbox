"""
Replays data offline.
- Loads a dataset 'data_group_name' (e.g., dataset_train). will need to call get_episodes() on this
- Runs the policy on episodes sampled from the dataset in order, under 'num_proc' processes.

Saves dataset with policy output under 'data_save_group_name'

"""

import multiprocessing as mp
import os
import signal
import threading
import time

import numpy as np
import psutil
import sharedmem as shm
import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, timeit
from sbrl.utils.script_utils import is_next_cycle, load_standard_ml_config
from sbrl.utils.torch_utils import to_numpy, broadcast_dims, to_torch

# mp.set_start_method('spawn')

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str, help="common params for all modules.")
grouped_parser.add_argument('--device', type=str, default="cpu")
grouped_parser.add_argument('--output_file', type=str, default=None)  # where to save
grouped_parser.add_argument('--create_env', action='store_true', help='If true, will create the env to pass to the policy (will not call step though)')
grouped_parser.add_argument('--log_every_n_episodes', type=int, default=0)
grouped_parser.add_argument('--num_proc', type=int, default=1)
grouped_parser.add_argument('--data_group_name', type=str, default="dataset_train")
grouped_parser.add_argument('--data_save_group_name', type=str, default="dataset_train")
grouped_parser.add_argument('--env_spec_save_group_name', type=str, default=None, help='Default is same env spec for input and output datasets.')
grouped_parser.add_argument('--policy_sub_key', type=str, default=None,
                            help='Under what prefix to save policy_outs. Default is to override existing entries.')
grouped_parser.add_argument('--goal_policy_sub_key', type=str, default=None,
                            help='Under what prefix to save goal_policy_outs. Default is to override existing entries.')
grouped_parser.add_argument('--model_file', type=str)
grouped_parser.add_argument('--no_model_file', action="store_true")
grouped_parser.add_argument('--share_model', action="store_true", help='If true, will share the model across processes.')
grouped_parser.add_argument('--strict_model_load', action="store_true")
grouped_parser.add_argument('--use_goal', action='store_true', help='Goal policy will be used and queried once at episode start')
grouped_parser.add_argument('--print_all', action='store_true')
local_args, unknown = grouped_parser.parse_local_args()
# this defines the required command line groups, and the defaults
# if in this list, we look for it

logger.debug(f"Using dataset group: {local_args.data_group_name}")
logger.debug(f"Saving with dataset group: {local_args.data_save_group_name}")
ordered_modules = ['env_spec', 'env_train', 'model', local_args.data_group_name, local_args.data_save_group_name, 'policy']
if local_args.env_spec_save_group_name is not None:
    ordered_modules.append(local_args.env_spec_save_group_name)

if local_args.use_goal:
    ordered_modules.append('goal_policy')

config_fname = os.path.abspath(local_args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                 debug=local_args.print_all)

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
env_spec_save = env_spec
if local_args.env_spec_save_group_name is not None:
    ess = local_args.env_spec_save_group_name
    env_spec_save = params[ess].cls(params[ess].params)

if local_args.share_model:
    model = params.model.cls(params.model.params, env_spec, None)
    model.to(args.device)
    model.eval()
    model.share_memory()


def run_replay(arguments, proc_id, is_finished, ep_in_queue, policy_out_queue, num_proc, parent_proc=None):
    """
    Will replay episodes and send back new additions from the policy.

    :param arguments:
    :param proc_id:
    :param is_finished: sharedmem array of size 1, for auto terminating process.
    :param ep_in_queue: receive episodes here
    :param policy_out_queue: write back policy outputs here
    :param num_proc:
    :param parent_proc:
    :return:
    """
    logger.info(f"Beginning worker {proc_id}.")
    seed = int(time.time() * 1e9) % 1000
    np.random.seed(seed)
    if parent_proc is None:
        parent_proc = psutil.Process(os.getppid())

    import signal
    def handler(sig, frame):
        raise Exception(f"Worker {proc_id} received parent signal {sig}...")

    if num_proc > 1:  # if we aren't in the parent process
        signal.signal(signal.SIGHUP, handler)

    # make your own copy of the model, optionally
    if not local_args.share_model:
        model = params.model.cls(params.model.params, env_spec, None)
        model.restore_from_file(model_fname, strict=local_args.strict_model_load)
        model.to(arguments.device)
        model.eval()

    ### warm start the planner
    presets = AttrDict()  # TODO

    env = None
    if arguments.create_env:
        env = params.env_train.cls(params.env_train.params, env_spec)
        _, _ = env.reset(presets)  # TODO set the state of env to reflect obs
    policy = params.policy.cls(params.policy.params, env_spec, env=env)
    if local_args.use_goal:
        goal_policy = params.goal_policy.cls(params.goal_policy.params, env_spec, env=env)

    logger.debug(f"[{proc_id}] Beginning offline eval...")

    while not is_finished[0] and parent_proc.is_running():
        ep_idx, observations, goals = ep_in_queue.get()
        ep_len = observations.get_one().shape[0]

        # move episode to torch
        observations = observations.leaf_apply(lambda arr: to_torch(arr, device=model.device, check=True))
        goals = goals.leaf_apply(lambda arr: to_torch(arr, device=model.device, check=True))

        # B x H x ... for model
        obs = observations.leaf_apply(lambda arr: arr[None, 0])
        goal = goals.leaf_apply(lambda arr: arr[None, 0])
        policy.reset_policy(next_obs=obs, next_goal=goal)
        policy.warm_start(model, obs, goal)

        with torch.no_grad():
            goal_policy_out = AttrDict()
            if local_args.use_goal:
                goal_policy.reset_policy(next_obs=obs, next_goal=goal)
                goal_policy_out = goal_policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]), goal.leaf_apply(lambda arr: arr[:, None]))
                goal_policy_out.leaf_modify(lambda arr: arr[:, None])

        actions = []
        for t in range(ep_len):
            with torch.no_grad():
                obs = observations.leaf_apply(lambda arr: arr[None, t:t+1])
                goal = goals.leaf_apply(lambda arr: arr[None, t:t+1]) & goal_policy_out
                action = policy.get_action(model, obs, goal)
            action = action.leaf_arrays()
            actions.append(action)

        # H x ... for sending back
        policy_out = AttrDict.leaf_combine_and_apply(actions, lambda vs: to_numpy(torch.cat(vs, dim=0)))
        goal_policy_out = goal_policy_out.leaf_apply(lambda arr: broadcast_dims(arr[:, 0], [0], [ep_len]))

        policy_out_queue.put((proc_id, ep_idx, policy_out, goal_policy_out))

    logger.info(f"Ending worker {proc_id}.")


def run_aggregate(num_proc, is_finished, local_args, params, ep_send_queues, policy_out_queue, processes):
    # loade dataset to add to.
    dtparams = params[local_args.data_group_name].params.leaf_copy()
    dtsparams = params[local_args.data_save_group_name].params.leaf_copy()

    if local_args.output_file is not None:
        dtsparams.output_file = local_args.output_file

    assert dtsparams >> "output_file" is not None
    logger.warn(f"Output file set to: {dtsparams >> 'output_file'}")

    logger.debug(f"Loading Episode Dataset: {local_args.data_group_name}")
    dataset_train = params[local_args.data_group_name].cls(dtparams, env_spec, file_manager)

    logger.debug(f"Loading Dataset for Saving: {local_args.data_save_group_name}")
    dataset_save = params[local_args.data_save_group_name].cls(dtsparams, env_spec_save, file_manager)

    ### restore model
    if not local_args.no_model_file and local_args.share_model:
        model.restore_from_file(model_fname, strict=local_args.strict_model_load)

    episodes = 0
    curr_ep = 0
    steps = 0
    num_eps = dataset_train.get_num_episodes()
    counts = np.zeros(num_proc, dtype=int)
    save_on_exit = True
    free_workers = list(range(num_proc))

    received_ep_idx = []
    receieved_policy_outs = []
    receieved_goal_policy_outs = []

    assert len(ep_send_queues) == num_proc

    while True:
        if not any([proc.is_alive() for proc in processes]):
            # save only if all processes exited safely.
            save_on_exit = all([proc.exitcode == 0 for proc in processes])
            logger.warn("All child processes finished sending early. breaking")
            break

        # dispatch available work (if any)
        while len(free_workers) > 0 and curr_ep < num_eps:
            inputs, _ = dataset_train.get_episode(curr_ep, names=None, split=True)
            worker_id = free_workers.pop()
            ep_send_queues[worker_id].put((curr_ep, inputs, inputs < env_spec.goal_names))
            curr_ep += 1

        # wait to receive, then add to dataset
        which_proc, which_ep, policy_out, goal_policy_out = policy_out_queue.get()
        free_workers.append(which_proc)  # this proc is now free

        # update storage buffers
        received_ep_idx.append(which_ep)
        receieved_policy_outs.append(policy_out)
        receieved_goal_policy_outs.append(goal_policy_out)

        counts[which_proc] += 1
        steps += policy_out.get_one().shape[0]
        episodes += 1

        if is_next_cycle(episodes, local_args.log_every_n_episodes):
            logger.warn(
                "[%d episodes] Completed %d steps. eps per worker: %s" % (
                episodes, steps, counts))
            if num_proc == 1:
                # timeit will be accurate if multi-threading
                logger.debug(str(timeit))
                timeit.reset()

        if episodes >= num_eps:
            logger.debug(f"Collected {num_eps} episodes! Terminating.")
            is_finished[:] = True
            break

    if num_proc > 1:
        for proc in processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)

    # sorted episode order (low to high)

    logger.debug("Adding episodes to dataset_save in order...")
    for next_ep_idx in np.argsort(received_ep_idx):
        which_ep = received_ep_idx[next_ep_idx]
        policy_out = receieved_policy_outs[next_ep_idx]
        goal_policy_out = receieved_goal_policy_outs[next_ep_idx]
        ep_ins, ep_outs = dataset_train.get_episode(which_ep, names=None, split=True)
        assert ep_ins.get_one().shape[0] == policy_out.get_one().shape[0], [ep_ins.get_one().shape[0], policy_out.get_one().shape[0]]

        # combine
        if local_args.policy_sub_key is not None:
            ep_ins[local_args.policy_sub_key] = policy_out
        else:
            ep_ins.combine(policy_out)

        if local_args.goal_policy_sub_key is not None:
            ep_ins[local_args.goal_policy_sub_key] = goal_policy_out
        else:
            ep_ins.combine(goal_policy_out)

        dataset_save.add_episode(ep_ins, ep_outs)

    if save_on_exit:  # save if we added new data
        logger.debug("Saving...")
        dataset_train.save()
    else:
        logger.debug(f"Not saving. save_on_exit: {save_on_exit}, steps: {steps}, init_steps: {init_steps}")

    logger.debug("Done.")


if __name__ == '__main__':

    num_proc = local_args.num_proc
    assert num_proc > 0

    ep_send_queues = [mp.Queue() for _ in range(num_proc)]
    policy_out_queue = mp.Queue()
    is_finished = shm.empty((1,), dtype=bool)
    is_finished[:] = False
    # launch work

    logger.debug(f"launching {num_proc} processes...")
    processes = []
    # assert mp.get_start_method() == "fork", "Fork is required?"

    if num_proc > 1:
        for i in range(num_proc):
            proc = mp.Process(target=run_replay, args=(args, i, is_finished, ep_send_queues[i], policy_out_queue, num_proc), daemon=False)
            proc.start()
            processes.append(proc)

        run_aggregate(num_proc, is_finished, local_args, params, ep_send_queues, policy_out_queue, processes)
    else:
        proc = threading.Thread(target=run_aggregate, args=(num_proc, is_finished, local_args, params, ep_send_queues, policy_out_queue, [threading.current_thread()]))
        proc.start()

        # dc in the main thread
        encaps = AttrDict()
        encaps.is_running = lambda: proc.is_alive()
        run_replay(args, 0, is_finished, ep_send_queues[0], policy_out_queue, num_proc, parent_proc=encaps)

        proc.join()
