import copy
import csv
import os
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sbrl.experiments import logger
from sbrl.metrics.metric import ExtractMetric
from sbrl.metrics.tracker import BufferedTracker, Tracker
from sbrl.utils.python_utils import get_with_default, AttrDict, timeit
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import to_numpy, add_horizon_dim, torch_mappable, to_torch


class BaseGoalTrainer:

    def __init__(self, params, file_manager,
                 model,
                 policy,
                 goal_policy,
                 env_train,
                 env_holdout,
                 policy_holdout=None,
                 goal_policy_holdout=None,
                 reward=None,
                 optimizer=None):
        """
        Don't use this directly. Does not implement run(), or train_step(). Mainly just a useful way to handle online env steps with multi-level policies.

        NOTE: env will be reset when either env is terminated or goal_policy is terminated. TODO option for terminate when policy is done

        :param params: parameters for training
        :param file_manager:
        :param model: a global model, with all the parameters necessary for computation at each level.
        :param policy: produces actions for train env
        :param goal_policy: produces goals for train policy/eng
        :param datasets_train: a list of training datasets to maintain
        :param datasets_holdout: a list of holdout datasets to maintain.
        :param env_train: the environment to step during training.
        :param env_holdout: the environment to step during holdout.
        :param reward: A Reward object to compute rewards, None means use the environment reward.
        :param optimizer: Optimizer object to step the model.
        :param policy_holdout: produces actions for the holdout env, None means same as train.
        :param goal_policy_holdout: produces goals for the holdout policy/env, None means same as train.
        """
        self._file_manager = file_manager
        self._model = model
        self._device = self._model.device

        self._policy = policy

        self._goal_policy = goal_policy
        self._goal_update_fn = get_with_default(params, "goal_update_fn", None)
        self._goal_id_name = get_with_default(params, "_goal_id_name", "goal_id")

        self._policy_holdout = policy_holdout if policy_holdout is not None else self._policy
        self._goal_policy_holdout = goal_policy_holdout if goal_policy_holdout is not None else self._goal_policy

        self._env_train = env_train
        self._env_holdout = env_holdout

        self._reward = reward
        # copy the reward for evaluating so that resets don't interfere
        self._reward_holdout = copy.deepcopy(reward)

        self._optimizer = optimizer
        if self._optimizer is None:
            self._optimizer = get_with_default(params, "optimizer", optimizer)  # look in params next

        self._env_train_memory = AttrDict()  # this will stay until reset.
        self._env_holdout_memory = AttrDict()  # this will stay until reset.

        # self._num_train_procs = 1  # multiprocessing TODO

        self._init_params_to_attrs(params)

        self._current_step = 0

        # steps since reset env
        # self._current_rollout_step = np.zeros(self._num_envs, dtype=int)  # TODO resolve this + vectorized
        self._current_env_train_step = 0
        self._current_env_holdout_step = 0

        self._current_env_train_ep = 0
        self._current_env_holdout_ep = 0

        # used with the default trackers.
        self._write_average_episode_returns_every_n_env_steps = int(
            get_with_default(params, "write_average_episode_returns_every_n_env_steps", 500))

        # TRACKING
        self._reward_names = get_with_default(params, "reward_names", ["reward"])
        if "reward" not in self._reward_names:
            self._reward_names.append("reward")

        default_trackers = {}
        for rew_name in self._reward_names:
            tracker = BufferedTracker(
                AttrDict(
                    buffer_len=10,  # return window is last 10 episodes.
                    buffer_freq=0,  # only clear on resets
                    time_agg_fn=lambda k, a, b: a + b,  # sum rewards
                    metric=ExtractMetric('sum_' + rew_name, key=rew_name, source=1),  # source is outputs.
                    tracked_names=['sum_' + rew_name],
                )
            )
            default_trackers['env_train/sum_' + rew_name] = tracker
            default_trackers['env_holdout/sum_' + rew_name] = tracker.duplicate()

        self._trackers = get_with_default(params, "trackers", AttrDict.from_dict(default_trackers))

        # tracked reward sums user specified
        self._tracker_write_frequencies = get_with_default(params, "tracker_write_frequences", AttrDict.from_dict({
            key: self._write_average_episode_returns_every_n_env_steps for key in self._trackers.leaf_keys()
        }))

        # default track all stats
        self._tracker_write_types = get_with_default(params, "tracker_write_types",
                                                     self._tracker_write_frequencies.leaf_apply(
                                                         lambda _: ['mean', 'max', 'min', 'std']))

        # initialize trackers.
        for tracker_name, tracker in self._trackers.leaf_items():
            if not isinstance(tracker, Tracker):
                assert isinstance(tracker, AttrDict), "Tracker must be passed in directly, or passed in as params."
                self._trackers[tracker_name] = (tracker >> "cls")(tracker >> "params")
                assert isinstance(tracker, Tracker), tracker

        # make sure tracker frequencies and types are provided for all trackers.
        assert set(self._trackers.leaf_keys()) == set(self._tracker_write_types.leaf_keys()), [
            self._trackers.list_leaf_keys(), self._tracker_write_types.list_leaf_keys()]
        assert set(self._trackers.leaf_keys()) == set(self._tracker_write_frequencies.leaf_keys()), [
            self._trackers.list_leaf_keys(), self._tracker_write_frequencies.list_leaf_keys()]
        # allowed keys to write.
        self._tracker_write_types.leaf_assert(lambda wt: all(t in ['mean', 'max', 'min', 'std'] for t in wt))

        # SAVING THE BEST MODEL (specify the tracker name to enable this)
        self._track_best_name = get_with_default(params, "track_best_name", None)
        self._track_best_key = get_with_default(params, "track_best_key", "returns")
        self._track_best_reduce_fn = get_with_default(params, "track_best_reduce_fn", lambda arr: arr.mean())

        self._last_best_tracked_val = -np.inf
        self._last_best_env_train_ep = self._current_env_train_ep

        if self._track_best_name is not None:
            logger.debug(f"Will attempt to save the best models using tracker: {self._track_best_name}, key: {self._track_best_key}")

        # loggers / writers
        self._enable_writers = get_with_default(params, "enable_writers", True)
        if self._enable_writers:
            l_path = os.path.join(self._file_manager.exp_dir, "loss.csv")
            csv_file = open(l_path, "a+")
            self._writer = csv.writer(csv_file, delimiter=',')
            self._summary_writer = SummaryWriter(self._file_manager.exp_dir)
        else:
            logger.warn("Writers are disabled!")
            self._writer = None
            self._summary_writer = None

        # scalars to log after each env_train step
        self._per_step_output_scalars_to_log = None if self._summary_writer is None else \
            get_with_default(params, "per_step_output_scalars_to_log", [])

        # optimizers
        self._init_optimizers(params)

        logger.debug(f"==== Training ====")
        if self._optimizer is not None:
            logger.debug(f"Optimizer: {self._optimizer}")
        logger.debug(f"Env: {self._env_train}")
        logger.debug(f"Policy: {self._policy}")
        logger.debug(f"Goal Policy: {self._goal_policy}")

    def _init_params_to_attrs(self, params):
        self._no_data_saving = bool(params >> "no_data_saving")

        self._random_policy_on_first_n_steps = int(get_with_default(params, "random_policy_on_first_n_steps", 0))

        self._load_statistics_initial = get_with_default(params, "load_statistics_initial", True)
        self._reload_statistics_every_n_env_steps = int(
            get_with_default(params, "reload_statistics_every_n_env_steps", 0))  # TODO
        self._reload_statistics_n_times = int(
            get_with_default(params, "reload_statistics_n_times", -1))  # -1 means never stop
        if self._reload_statistics_n_times > 0:
            assert self._reload_statistics_every_n_env_steps > 0
        self._max_grad_norm = get_with_default(params, "max_grad_norm", None)

        self._transfer_model_only = get_with_default(params, "transfer_model_only",
                                                     False)  # if true, loads only the model, not the training step
        self._checkpoint_model_file = get_with_default(params, "checkpoint_model_file", None)
        # self._save_checkpoints = get_with_default(params, "save_checkpoints", False)

        # these will be stored in env_memory.
        self._goal_policy_uses_last_n_inputs = get_with_default(params, "goal_policy_uses_last_n_inputs",
                                                                1)  # 1 means only the last episode
        assert self._goal_policy_uses_last_n_inputs > 0

        # fns
        # self._process_env_step_output_fn = get_with_default(params, "process_env_step_output_fn",
        #                                                     lambda env, o, g, next_obs, next_goal, po, ea, d: (
        #                                                         next_obs, next_goal))

        # by default does nothing
        self._env_action_from_policy_output = get_with_default(params, "env_action_from_policy_output",
                                                               lambda env, model, o, g, po: po)

    def _init_optimizers(self, params):
        self._optimizer = None
        raise NotImplementedError

    def _write_step(self, model, loss, inputs, outputs, meta: AttrDict = AttrDict(), **kwargs):
        for i, pg in enumerate(self._optimizer.param_groups):
            self._summary_writer.add_scalar("train/learning_rate_pg_%d" % i, pg['lr'], self._current_step)

    def _tracker_write_step(self, trackers, curr_step, force=False):
        for tracker_name, tracker in trackers.leaf_items():
            if (force or is_next_cycle(curr_step, self._tracker_write_frequencies[tracker_name])) and tracker.has_data():
                # print(f"tracker: {tracker_name} active")
                #  the tracker has time series output (e.g. buffered returns), which we will average
                ts_outputs = tracker.get_time_series().leaf_apply(lambda arr: np.asarray(arr)[None])  # T
                writing_types = self._tracker_write_types[tracker_name]
                for key, arr in ts_outputs.leaf_items():
                    if arr.size > 0:
                        with timeit("writer"):
                            if 'mean' in writing_types:
                                self._summary_writer.add_scalar(tracker_name + "/" + key + "_mean", arr.mean(),
                                                                self._current_step)
                            if 'max' in writing_types:
                                self._summary_writer.add_scalar(tracker_name + "/" + key + "_max", arr.max(),
                                                                self._current_step)
                            if 'min' in writing_types:
                                self._summary_writer.add_scalar(tracker_name + "/" + key + "_min", arr.min(),
                                                                self._current_step)
                            if 'std' in writing_types:
                                self._summary_writer.add_scalar(tracker_name + "/" + key + "_std", arr.std(),
                                                                self._current_step)

    def goal_step(self, model, datasets, obs, goal, env_memory: AttrDict, goal_policy, policy):
        # latest policy done, or first step.
        if env_memory >> "goal_rollout_count" == 0 or (env_memory >> "policy_done")[-1].item():
            # the passed in goal is from reset, and this function computes additional goals using the goal policy
            # env_memory.policy_done[-1][:] = False
            env_memory.goal_count += 1
            # TODO is this slow?
            # print((env_memory >> "history/obs"))
            stacked_obs = AttrDict.leaf_combine_and_apply(
                (env_memory >> "history/obs")[-self._goal_policy_uses_last_n_inputs:],
                lambda vs: np.concatenate(vs, axis=1))
            # if len(env_memory >> "history/goal") > 0:
            #     stacked_goal = AttrDict.leaf_combine_and_apply((env_memory >> "history/goal")[-self._goal_policy_uses_last_n_inputs:], lambda vs: torch.cat(vs, dim=1))
            # else:
            #     stacked_goal = AttrDict()

            ret_goal = goal & goal_policy.get_action(model, stacked_obs, goal)
        else:
            ret_goal = goal & (env_memory >> "history/goal")[-1].leaf_apply(lambda arr: arr[:, 0])  # latest goal

        # very first goal isn't stored, during reset, so store it now
        if env_memory >> "goal_rollout_count" == 0:
            env_memory.history.goal.append(ret_goal.leaf_arrays().leaf_apply(lambda arr: arr[:, None]))

        return ret_goal

    def policy_step(self, model, datasets, obs, goal, env_memory: AttrDict, policy, eval_step=False):
        if not eval_step and self._current_step < self._random_policy_on_first_n_steps:
            # policy takes in only ob
            return policy.get_random_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                            goal.leaf_apply(lambda arr: arr[:, None]))

        else:
            # policy takes in only ob
            return policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                     goal.leaf_apply(lambda arr: arr[:, None]), greedy_action=eval_step)

    def env_step(self, model, env, datasets, obs, goal, env_memory: AttrDict,
                 policy, goal_policy, reward=None, eval=False, add_data_every_n=0, curr_step=0, trackers=AttrDict(),
                 get_goal_done=False):
        """
        Do one step of the environment, calling policy / goal_policy and resetting all appropriately when done.

        :param model:
        :param env:
        :param datasets: a list of datasets, if env is saving, it will save to the first in this list.
        :param obs:
        :param goal:
        :param env_memory:
        :param policy:
        :param goal_policy:
        :param reward:
        :param eval:
        :param add_data_every_n:
        :param curr_step:
        :param trackers:
        :param get_goal_done:
        :return:
        """
        if eval:
            model.eval()

        if env_memory.is_empty():
            # do the reset tasks.
            obs, goal = self.env_reset(env, datasets, obs, goal, env_memory)
            policy.reset_policy(next_obs=obs, next_goal=goal)
            goal_policy.reset_policy(next_obs=obs, next_goal=goal)
            if reward is not None:
                reward.reset_reward()

        with torch.no_grad():
            with timeit("env_step/goal_policy"):
                goal = self.goal_step(model, datasets, obs, goal, env_memory, goal_policy, policy)

            with timeit("env_step/policy"):
                action = self.policy_step(model, datasets, obs, goal.leaf_arrays(), env_memory, policy,
                                          eval_step=get_goal_done)
                # action = self._env_action_from_policy_output(env, model, obs.leaf_apply(add_horizon_dim),
                #                                              goal.leaf_apply(add_horizon_dim), action)

            with timeit("env_step/step"):
                # DO the step
                next_obs, next_goal, done = env.step(action)

            # aggregate done's
            policy_done = np.array([self._policy.is_terminated(model, obs, goal, env_memory=env_memory)])
            goal_policy_done = np.array([self._goal_policy.is_terminated(model, next_obs, goal, env_memory=env_memory)])
            # e.g., for evaluation episode, add whether goal is reached for logging purposes
            if get_goal_done:
                next_obs["goal_reached"] = goal_policy_done.astype(np.float)

        done = np.logical_or(done, goal_policy_done)

        if reward is not None:
            with timeit("env_step/reward"):
                # supports the transition, or even longer scope rewards, or rewards that use models.
                reward_out = reward.get_reward(env, model, obs, goal, action, next_obs, next_goal,
                                               env_memory,
                                               policy_done, goal_policy_done, done, self._current_env_train_step)
                next_obs = next_obs & reward_out

        # update histories & reward
        env_memory.policy_done.append(policy_done[:, None])
        env_memory.goal_policy_done.append(goal_policy_done[:, None])
        env_memory.done.append(done[:, None])

        # add obs to history along dim=1 (append)
        # expand_append = lambda vs: vs[0].append(vs[1].leaf_apply(lambda arr: arr[:, None]))
        env_memory.history.goal.append(goal.leaf_arrays().leaf_apply(lambda arr: arr[:, None]))
        env_memory.history.acs.append(action.leaf_arrays().leaf_apply(lambda arr: arr[:, None]))
        env_memory.history.obs.append(next_obs.leaf_arrays().leaf_apply(lambda arr: arr[:, None]))
        # true history as well
        env_memory.true_history.goal.append(env_memory.history.goal[-1])
        env_memory.true_history.acs.append(env_memory.history.acs[-1])
        env_memory.true_history.obs.append(env_memory.history.obs[-1])

        # next_obs.leaf_arrays().leaf_apply(lambda arr: type(arr)).pprint()
        # action.leaf_arrays().leaf_apply(lambda arr: type(arr)).pprint()
        # next_goal.leaf_arrays().leaf_apply(lambda arr: type(arr)).pprint()

        # updating the env trackers

        # env tracker update
        inputs = obs & action & goal
        outputs = next_obs
        torch_inputs = inputs.leaf_arrays().leaf_apply(
            lambda arr: to_torch(arr, device=self._device, check=True) if torch_mappable(arr.dtype) else arr)
        torch_outputs = outputs.leaf_arrays().leaf_apply(
            lambda arr: to_torch(arr, device=self._device, check=True) if torch_mappable(arr.dtype) else arr)
        for _, tracker in trackers.leaf_items():
            tracker.compute_and_update(torch_inputs, torch_outputs, AttrDict())

        # log per env step scalars
        for name in self._per_step_output_scalars_to_log:
            val = float(outputs >> name)
            self._summary_writer.add_scalar("train_env_step/{}".format(name), val, self._current_env_train_step)

        # local steps
        env_memory.rollout_count += 1
        env_memory.goal_rollout_count += 1

        obs, goal, done = self._env_post_step(model, env, env_memory, datasets, obs, goal, done)

        goal_term = env_memory.goal_count > 0 and is_next_cycle(env_memory.goal_count, add_data_every_n)

        if done.item() or goal_term:

            # add to dataset if not an eval episode
            with timeit("store"):
                if not self._no_data_saving:
                    self.store_episode(datasets[0], env_memory)
                if goal_term and not done.item():
                    # reset without actually resetting the env, since we are storing data early.
                    next_obs, next_goal = self.clear_env_memory(next_obs, next_goal, env_memory, no_reset=True)

                if done.item():
                    # reset trackers that
                    for _, tracker in trackers.leaf_items():
                        tracker.reset_tracked_state()

                    env_memory.clear()  # clear so we know that we have to reset the environment

        # POST STEP ACTIONS (uses current_step+1)

        # reloading statistics
        if self._reload_statistics_n_times <= 0 or \
                curr_step <= self._reload_statistics_every_n_env_steps * self._reload_statistics_n_times:
            # only reload when train_step <= self._reload_stats * self._reload_stats_n_times
            if is_next_cycle(curr_step + 1, self._reload_statistics_every_n_env_steps):
                logger.warn("Reloading statistics from dataset")
                model.load_statistics()

        self._tracker_write_step(trackers, curr_step + 1)

        return next_obs, next_goal, done

    def _env_post_step(self, model, env, env_memory, datasets, obs, goal, done):
        """ Any actions before clearing the env_memory / handling done but after the step. """
        return obs, goal, done

    def clear_env_memory(self, obs, goal, env_memory, no_reset=False):
        """
        Clears internal state of env_memory (history) and if !no_reset, the rest.

        :param obs:
        :param goal:
        :param env_memory:
        :param no_reset: if True, means we are clearing in the middle of a rollout (just clear history)
        :return:
        """

        # initialize the env_memory/history/inputs for next run.
        expanded_obs = obs.leaf_apply(add_horizon_dim)

        if not no_reset:
            if not obs.has_leaf_key("reward"):
                # initial reward is 0.   B x 1
                for rew_name in self._reward_names:
                    obs[rew_name] = np.zeros((1, 1), dtype=np.float32)

            env_memory.rollout_count = 0  # first env step in rollout.
            env_memory.goal_count = 0  # number of goal action calls.
            # records the true history
            env_memory.true_history.obs = [expanded_obs]
            env_memory.true_history.acs = []
            env_memory.true_history.goal = []

        env_memory.goal_rollout_count = 0  # first env step since last goal reset

        # records what we should store in dataset
        env_memory.history.obs = [expanded_obs]
        env_memory.history.acs = []  # there is one less action than obs
        env_memory.history.goal = []

        # used for dataset as well
        env_memory.policy_done = []
        env_memory.goal_policy_done = []
        env_memory.done = []

        return obs, goal

    def env_reset(self, env, datasets, obs, goal, env_memory: AttrDict):
        obs, goal = env.reset()
        return self.clear_env_memory(obs, goal, env_memory)

    def env_rollout(self, model, env, datasets, obs, goal, env_memory, policy, goal_policy, reward, trackers,
                    curr_step_wrapper,
                    add_to_data_every_n=-1):
        """
        Simple wrapper function to rollout an environment based on episodes rather than steps
        """
        done = np.array([False])
        assert curr_step_wrapper.has_leaf_key("step")

        while not done[0]:
            obs, goal, done = self.env_step(model, env, datasets, obs, goal, env_memory, policy, goal_policy,
                                            reward=reward, eval=True,
                                            trackers=trackers,
                                            curr_step=curr_step_wrapper >> "step",
                                            add_data_every_n=add_to_data_every_n)
            curr_step_wrapper.step += 1

        assert env_memory.is_empty(), "Env memory must be empty at the end of episode!"

        return obs, goal, done

    # RUNNING SCRIPTS #

    def store_episode(self, dataset, env_memory):
        cat_fn = lambda vs: torch.cat(vs, dim=1) if isinstance(vs[0], torch.Tensor) else np.concatenate(vs, axis=1)
        # T x ...

        obs = AttrDict.leaf_combine_and_apply(env_memory >> "history/obs",
                                              lambda vs: to_numpy(cat_fn(vs), check=True)[0])
        assert obs.get_one().shape[
                   0] > 1, f"Must be at least 2 obs in episode to store it, but was only {obs.get_one().shape[0]}"

        # T-1 x ...
        acs = AttrDict.leaf_combine_and_apply(env_memory >> "history/acs",
                                              lambda vs: to_numpy(cat_fn(vs), check=True)[0])
        goal = AttrDict.leaf_combine_and_apply(env_memory >> "history/goal",
                                               lambda vs: to_numpy(cat_fn(vs), check=True)[0])

        input_obs = obs.leaf_apply(lambda arr: arr[:-1])
        output_obs = AttrDict(next=obs.leaf_apply(lambda arr: arr[1:]))  # prefixed by next/___

        output_obs.done = np.concatenate(env_memory >> "done", axis=1)[0]
        output_obs.policy_done = np.concatenate(env_memory >> "policy_done", axis=1)[0]

        # shift reward from next/reward to reward
        if output_obs.has_leaf_key("next/reward") and not output_obs.has_leaf_key("reward"):
            output_obs.reward = output_obs["next/reward"]
        elif input_obs.has_leaf_key("reward") and not output_obs.has_leaf_key("reward"):
            output_obs.reward = input_obs.reward  # copy the reward key over.

        # do the same thing with goals (these are basically just observations)
        input_goal = goal.leaf_apply(lambda arr: arr[:-1])
        output_goal = AttrDict(next=goal.leaf_apply(lambda arr: arr[1:]))

        logger.debug(f"Adding episode to dataset[0], ep_len={len(output_obs.done)}, len(ds)={len(dataset)}")
        # input_obs, actions, and goals are all inputs. outputs here include all next observations
        dataset.add_episode(input_obs & acs & input_goal, output_obs & output_goal)

    def _save(self, chkpt=False, best=False):
        base_fname = "model.pt"
        path = os.path.join(self._file_manager.models_dir, base_fname)
        save_data = {'step': self._current_step,
                     'model': self._model.state_dict()}
        save_data.update(self._get_save_meta_data())

        if self._track_best_name is not None:
            save_data[self._track_best_name] = self._last_best_tracked_val
        elif best:
            logger.warn("Could not save best value (best=True) to model, as there is no tracked data...")
        torch.save(save_data, path)
        logger.debug("Saved model")
        chkpt_base_fname = "chkpt_{:010d}.pt".format(self._current_step)
        if chkpt:
            shutil.copyfile(path, os.path.join(self._file_manager.models_dir, chkpt_base_fname))
            logger.debug(f"Saved checkpoint: {self._current_step}")
        if best:
            # save both so we know the best model and which checkpoints got rated as best.
            shutil.copyfile(path, os.path.join(self._file_manager.models_dir, "best_model.pt"))
            shutil.copyfile(path, os.path.join(self._file_manager.models_dir, f"best_{chkpt_base_fname}"))
            logger.debug(f"Saved best.")

    def _restore_checkpoint(self):
        if self._checkpoint_model_file is None:
            return False
        path = os.path.join(self._file_manager.models_dir, self._checkpoint_model_file)
        if os.path.isfile(path):
            checkpoint = torch.load(str(path))
            self._model.restore_from_checkpoint(checkpoint)
            if not self._transfer_model_only:
                self._current_step = checkpoint['step']
                self._restore_meta_data(checkpoint)
                if self._track_best_name in checkpoint.keys():
                    self._last_best_tracked_val = float(checkpoint[self._track_best_name])
            logger.debug(f"Loaded model from {path}, current train step: {self._current_step}")
            return True
        else:
            logger.warn("Unable to load model!")
            return False

    def _get_save_meta_data(self):
        return {}  # TODO

    def _restore_meta_data(self, checkpoint):
        pass

    def run_preamble(self):
        """
        CALL THIS FROM RUN()
        loading model, statistics, dataset setup, printing model params.
        """
        # NOTE: make sure you if you're experiment is killed that you can restart it where you left off
        load_success = self._restore_checkpoint()

        # TODO pre-training?

        if not load_success and self._load_statistics_initial:
            logger.info("Loading statistics for new model with given data_train")
            self._model.load_statistics()

        # log the model parameters
        logger.debug("Model parameters:")
        if hasattr(self._model, "print_parameters"):
            self._model.print_parameters()
        else:
            for p in self._model.parameters():
                logger.debug("PARAMETER: shape = %s, requires_grad = %s" % (p.shape, p.requires_grad))

    def run(self):
        raise NotImplementedError
