"""
TODO
"""

import multiprocessing as mp
import os
import queue
import signal
import threading
import time

import numpy as np
import psutil
import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager, FileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, timeit
from sbrl.utils.script_utils import is_next_cycle, load_standard_ml_config
from sbrl.utils.torch_utils import to_numpy, reduce_map_fn

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str, help="common params for all modules.")
grouped_parser.add_argument('--device', type=str, default="cpu")
grouped_parser.add_argument('--max_steps', type=int, required=True)
grouped_parser.add_argument('--save_every_n_episodes', type=int, required=True)
grouped_parser.add_argument('--reload_input', action='store_true')  # will use output as input
grouped_parser.add_argument('--no_input_file', action='store_true')  # defaults to not changing
grouped_parser.add_argument('--output_file', type=str, default=None)  # defaults to not changing
grouped_parser.add_argument('--log_every_n_episodes', type=int, default=0)
grouped_parser.add_argument('--min_ep_length', type=int, default=2)
grouped_parser.add_argument('--max_ep_length', type=int, default=np.inf, help="Equivalent to a timeout")
grouped_parser.add_argument('--num_proc', type=int, default=8)
grouped_parser.add_argument('--synchronize', action='store_true')  # waits until done processing episode before proceeding
grouped_parser.add_argument('--save_chunks', action='store_true')  
grouped_parser.add_argument('--save_start_ep', type=int, default=0)  
grouped_parser.add_argument('--keep_last_n_steps_only', type=int, default=0,
                            help="if nonzero, only saves the last N steps for each ep")
grouped_parser.add_argument('--data_group_name', type=str, default="dataset_train")
grouped_parser.add_argument('--model_file', type=str)
grouped_parser.add_argument('--no_model_file', action="store_true")
grouped_parser.add_argument('--strict_model_load', action="store_true")
grouped_parser.add_argument('--use_goal', action='store_true')
grouped_parser.add_argument('--print_all', action='store_true')
grouped_parser.add_argument('--return_keys', type=str, nargs="+", default=['reward'])
grouped_parser.add_argument('--track_returns', action='store_true')
grouped_parser.add_argument('--return_thresh', type=float, default=None, help='Any episodes less than this return will be tossed.')
grouped_parser.add_argument('--reduce_returns', type=str, default='sum', choices=list(reduce_map_fn.keys()),
                            help='If tracking returns, will apply this func to the returns before tracking..')
local_args, unknown = grouped_parser.parse_local_args()
# this defines the required command line groups, and the defaults
# if in this list, we look for it

print(local_args.data_group_name)
assert local_args.return_thresh is None or local_args.track_returns, "Must track returns in order to threshold the returns!"
ordered_modules = ['env_spec', 'env_train', 'model', local_args.data_group_name, 'policy']
if local_args.use_goal:
    ordered_modules.append('goal_policy')

config_fname = os.path.abspath(local_args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                 debug=local_args.print_all)

if os.path.exists(os.path.join(FileManager.base_dir, 'experiments', exp_name)):
    file_manager = ExperimentFileManager(exp_name, is_continue=True)
else:
    logger.warn(f"Experiment: 'experiments/{exp_name}' does not exist, using 'experiments/test' instead")
    file_manager = ExperimentFileManager('test', is_continue=True)


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
model.to(args.device)

### restore model
if not local_args.no_model_file:
    model.restore_from_file(model_fname, strict=local_args.strict_model_load)

model.share_memory()
model.eval()

reward_reduce = reduce_map_fn[local_args.reduce_returns]

shared_count = 0
assert not args.synchronize or args.num_proc == 1 

def run_data_collect(arguments, proc_id, iterations, ep_queue, num_proc, parent_proc=None):
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

    env = params.env_train.cls(params.env_train.params, env_spec)
    policy = params.policy.cls(params.policy.params, env_spec, env=env)
    if local_args.use_goal:
        goal_policy = params.goal_policy.cls(params.goal_policy.params, env_spec, env=env)

    ### warm start the planner
    presets = AttrDict()  # TODO
    obs, goal = env.reset(presets)

    logger.debug("Observation Keys: %s" % obs.list_leaf_keys())
    # print(obs.leaf_apply(lambda arr: arr.shape if isinstance(arr, np.ndarray) else None))
    policy.reset_policy(next_obs=obs, next_goal=goal)
    if local_args.use_goal:
        goal_policy.reset_policy(next_obs=obs, next_goal=goal)
        goal = goal & goal_policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]), goal.leaf_apply(lambda arr: arr[:, None]))
    policy.warm_start(model, obs, goal)

    logger.debug("Eval loop...")

    ### eval loop
    done = [False]
    steps = 0
    episodes = 0

    episode_inputs = []
    episode_outputs = []

    while steps < iterations and parent_proc.is_running():

        if done[0] or policy.is_terminated(model, obs, goal):

            logger.debug(f"[Worker {proc_id}] Resetting env....")
            obs, goal = env.reset()
            policy.reset_policy(next_obs=obs, next_goal=goal)
            if local_args.use_goal:
                goal_policy.reset_policy(next_obs=obs, next_goal=goal)
                goal = goal_policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]), goal.leaf_apply(lambda arr: arr[:, None]))
                # TODO goal termination check.

            if len(episode_inputs) == 0:
                logger.warn("Zero length episode!!")
            else:
                # TODO filter before concat?
                ep_ins = AttrDict.leaf_combine_and_apply(episode_inputs, func=lambda vs: np.concatenate(vs, axis=0))
                ep_outs = AttrDict.leaf_combine_and_apply(episode_outputs, func=lambda vs: np.concatenate(vs, axis=0))

                ep_outs.done[-1] = True  # in case policy terminated

                keep_length = len(ep_outs.done)
                if arguments.keep_last_n_steps_only > 0:
                    keep_length = max(min(keep_length, arguments.keep_last_n_steps_only), 1)  # must keep at least one step

                ep_ins = ep_ins.leaf_apply(lambda arr: arr[-keep_length:])
                ep_outs = ep_outs.leaf_apply(lambda arr: arr[-keep_length:])

                if len(ep_outs.done) > arguments.min_ep_length:
                    ## send to parent
                    old_shared_count = shared_count
                    ep_queue.put((proc_id, ep_ins, ep_outs))
                    # optional block until added to dataset
                    if arguments.synchronize and num_proc == 1:
                        while shared_count == old_shared_count:
                            time.sleep(0.1)

                steps += keep_length

                # dataset_train.add_episode(ep_ins, ep_outs)

                episode_inputs.clear()
                episode_outputs.clear()
                episodes += 1
            
            logger.debug(f"[Worker {proc_id}] Reset finished....")

        # two empty axes for (batch_size, horizon)
        policy_obs = obs.leaf_apply(lambda arr: arr[:, None])
        policy_goal = goal.leaf_apply(lambda arr: arr[:, None])
        with torch.no_grad():
            action = policy.get_action(model, policy_obs, policy_goal)
        next_obs, next_goal, done = env.step(action)

        if len(episode_inputs) >= local_args.max_ep_length - 1:
            done[:] = True

        # process inputs
        new_inputs = AttrDict()
        new_inputs.combine(obs)
        new_inputs.combine(goal)
        new_inputs.combine(action.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))

        # process outputs
        new_outputs = AttrDict()
        for key in env_spec.output_observation_names:
            if key[:5] in ["next_", "next/"]:
                if key[5:] in next_obs.leaf_keys():
                    new_outputs[key] = next_obs[key[5:]]
                elif key[5:] in next_goal.leaf_keys():
                    new_outputs[key] = next_goal[key[5:]]
            elif key in next_obs.leaf_keys():
                new_outputs[key] = next_obs[key]
            elif key in next_goal.leaf_keys():
                new_outputs[key] = next_goal[key]
        new_outputs.done = done
        episode_inputs.append(new_inputs)
        episode_outputs.append(new_outputs)

        # transition
        obs = next_obs
        if local_args.use_goal:
            goal = goal & next_goal  # there might be some info that isn't captured within the environment goal
        else:
            goal = next_goal

    # except BaseException as e:
    #     logger.warn(f"{type(e)}, {str(e)}")

    logger.info(f"Ending worker {proc_id}.")


def run_aggregate(num_proc, local_args, params, episode_queue, processes):
    global shared_count
    # loade dataset to add to.
    assert local_args.save_every_n_episodes > 0
    dtparams = params[local_args.data_group_name].params.leaf_copy()
    if local_args.no_input_file:
        dtparams.file = "none.npz"  # --> trickery
    elif local_args.reload_input:
        dtparams.file = file_path_with_default_dir(local_args.output_file, file_manager.exp_dir)

    if local_args.output_file is not None:
        dtparams.output_file = local_args.output_file
    dtparams.capacity = local_args.max_steps + 10  # override capacity so it works for max_steps
    assert dtparams >> "output_file" is not None
    logger.warn(f"Output file set to: {dtparams >> 'output_file'}")
    dataset_train = params[local_args.data_group_name].cls(dtparams, env_spec, file_manager)
    # work left to do, and capacity exists
    assert len(dataset_train) < local_args.max_steps <= int(dtparams.capacity), [len(dataset_train), local_args.max_steps, dtparams.capacity]

    STEPS_LEFT = local_args.max_steps - len(dataset_train)
    logger.info("Collecting max %d new samples..." % STEPS_LEFT)

    episodes = dataset_train.get_num_episodes()
    first_to_save_ep = episodes
    start_offset = args.save_start_ep
    logger.debug(f"Starting at episode {episodes}, naming offset = {start_offset}")
    steps = len(dataset_train)
    init_steps = steps
    counts = np.zeros(num_proc, dtype=int)
    save_on_exit = not local_args.save_chunks
    extra_returns = AttrDict.from_dict({k: [] for k in args.return_keys})
    return_thresh = local_args.return_thresh
    while steps < local_args.max_steps:
        # get an episode from a worker
        if any([proc.is_alive() for proc in processes]):
            with timeit('wait'):
                worker_id, ep_ins, ep_outs = episode_queue.get()
            # ep_ins.leaf_shapes().pprint()
        else:
            # save only if all processes exited safely.
            save_on_exit = all([proc.exitcode == 0 for proc in processes])
            logger.warn("All child processes finished sending early. breaking")
            break

        keep_episode = True
        if local_args.track_returns:
            for rew_key in args.return_keys:
                if ep_outs.has_leaf_key(rew_key):
                    rew = ep_outs >> rew_key
                elif ep_ins.has_leaf_key(rew_key):
                    rew = ep_ins >> rew_key
                else:
                    raise NotImplementedError(f"Reward key {rew_key} is not present...")

                # reduce rewards with method set via arguments.
                ret = reward_reduce(rew).item()
                extra_returns[rew_key].append(ret)

            for rew_key in args.return_keys:
                this_ret = extra_returns >> rew_key
                logger.debug(f"[{len(dataset_train)}] Episode {episodes} {rew_key}({local_args.reduce_returns}): {this_ret[-1]}")
                logger.debug(f"[{len(dataset_train)}] All {rew_key} returns (# = {episodes + 1}): mean = {np.mean(this_ret).item()}, std = {np.std(this_ret).item()}, #>0 = {np.sum(np.array(this_ret) > 1e-11).item()}")

            # thresh by the first key, if any.
            if return_thresh is not None:
                keep_episode = (extra_returns >> args.return_keys[0])[-1] >= return_thresh

        # this might be slow...
        if keep_episode:
            dataset_train.add_episode(ep_ins, ep_outs)
            steps += ep_ins.get_one().shape[0]
            counts[worker_id] += 1
            episodes += 1

            if is_next_cycle(episodes, local_args.log_every_n_episodes):
                logger.warn(
                    "[%d] After %d episodes, data len = %d. eps per worker: %s" % (
                    steps, episodes, len(dataset_train), counts))
                if num_proc == 1:
                    # timeit will be accurate if multi-threading
                    logger.debug(str(timeit))

            if is_next_cycle(episodes, local_args.save_every_n_episodes):
                logger.warn("[%d] Saving data after %d episodes, data len = %d" % (steps, episodes, len(dataset_train)))
                # saves in pieces, good for large arrays.
                if local_args.save_chunks:
                    if first_to_save_ep == episodes - 1:
                        suffix = f'_ep{start_offset + first_to_save_ep}'
                    elif first_to_save_ep < episodes - 1:
                        suffix = f'_ep{start_offset + first_to_save_ep}-{start_offset + episodes - 1}'
                    else:
                        raise NotImplementedError('first_save should always be less than episodes.')
                    # [inclusive, exclusive)
                    dataset_train.save(suffix=suffix, ep_range=(first_to_save_ep, episodes))
                    first_to_save_ep = episodes  # start at the next one next time.
                else:
                    dataset_train.save()
                logger.debug("Save finished..")
            
        else:
            logger.warn("Not keeping last episode!!")

        # signals when an episode can start is done
        shared_count += 1

    if num_proc > 1:
        for proc in processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)

    if len(dataset_train) > init_steps and save_on_exit:  # save if we added new data
        logger.warn("Reached max steps (%d). Done with final_len = %d" % (steps, len(dataset_train)))
        ep_lens = np.array([dataset_train.episode_length(i) for i in range(dataset_train.get_num_episodes())])
        logger.warn(f"Episode lengths: min: {np.min(ep_lens)}, max: {np.max(ep_lens)} mean: {np.mean(ep_lens)} std: {np.std(ep_lens)}")
        # NOTE: this will be extra work if save_chunks is True.
        dataset_train.save()
    else:
        logger.debug(f"Not saving. save_on_exit: {save_on_exit}, steps: {steps}, len(ds): {len(dataset_train)}, init_steps: {init_steps}")

    logger.debug("Done.")
    logger.debug(str(timeit))


if __name__ == '__main__':

    episode_queue = mp.Queue()
    # launch work

    num_proc = local_args.num_proc
    assert num_proc > 0

    if num_proc == 1:
        episode_queue = queue.Queue()

    logger.debug(f"launching {num_proc} processes...")
    processes = []
    assert mp.get_start_method() == "fork", "Fork is required?"

    if num_proc > 1:
        for i in range(num_proc):
            proc = mp.Process(target=run_data_collect, args=(args, i, local_args.max_steps, episode_queue, num_proc), daemon=False)
            proc.start()
            processes.append(proc)

        run_aggregate(num_proc, local_args, params, episode_queue, processes)
    else:
        proc = threading.Thread(target=run_aggregate, args=(num_proc, local_args, params, episode_queue, [threading.current_thread()]))
        proc.start()

        # dc in the main thread
        encaps = AttrDict()
        encaps.is_running = lambda: proc.is_alive()
        run_data_collect(args, 0, local_args.max_steps, episode_queue, num_proc, parent_proc=encaps)

        proc.join()
