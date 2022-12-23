"""
The trainer is where everything is combined.
"""
import csv
import math
import os
import shutil
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sbrl.datasets.data_augmentation import DataAugmentation
from sbrl.envs.vectorize_env import VectorizedEnv
from sbrl.experiments import logger
from sbrl.metrics.metric import ExtractMetric
from sbrl.metrics.tracker import Tracker, BufferedTracker
from sbrl.utils.python_utils import timeit, AttrDict, get_with_default, get_cls_param_instance, is_array, get_required
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import to_numpy, to_torch


class Trainer(object):

    def __init__(self, params, file_manager, model, policy, dataset_train, dataset_holdout, env_train, env_holdout):
        self._file_manager = file_manager
        # supports multiple models, zero'th model would be the main
        if isinstance(model, OrderedDict):
            if len(model.keys()) > 0:
                logger.debug(f"Multiple models: {len(model)}")
            self._all_models = model
        else:
            self._all_models = OrderedDict({'model': model})
        self._model = list(self._all_models.values())[0]

        # supports multiple policies, zero'th policy would be the main
        if isinstance(policy, OrderedDict):
            if len(policy.keys()) > 0:
                logger.debug(f"Multiple policies: {len(policy)}")
            self._all_policies = policy
        else:
            self._all_policies = OrderedDict({'policy': policy})
        self._policy = list(self._all_policies.values())[0]
        self._dataset_train = dataset_train
        self._dataset_holdout = dataset_holdout
        self._env_train = env_train
        self._env_holdout = env_holdout

        self.data_augmentation_params: AttrDict = get_with_default(params, "data_augmentation_params", AttrDict())
        if isinstance(self.data_augmentation_params, DataAugmentation):
            self.data_augmentation: DataAugmentation = self.data_augmentation_params  # allow passing in data aug
        elif not self.data_augmentation_params.is_empty():
            self.data_augmentation: DataAugmentation = get_cls_param_instance(self.data_augmentation_params,
                                                                              "cls", "params", DataAugmentation)
        else:
            logger.info("Using no data augmentation.")
            self.data_augmentation = None
        self._train_do_data_augmentation = get_with_default(params, "train_do_data_augmentation", True)

        # env step will be parallelized based on this
        self._is_vectorized = isinstance(self._env_train, VectorizedEnv)
        self._num_envs = self._env_train.num_envs if self._is_vectorized else 1
        self._vectorized_input_history = []
        self._vectorized_output_history = []
        for i in range(self._num_envs):
            # instantiating history for each env
            self._vectorized_input_history.append(AttrDict())
            self._vectorized_output_history.append(AttrDict())
            for key in self._env_train.env_spec.observation_names + self._env_train.env_spec.action_names + self._env_train.env_spec.param_names:
                self._vectorized_input_history[i][key] = []
            for key in self._env_train.env_spec.output_observation_names + self._env_train.env_spec.final_names + [
                "done"]:
                self._vectorized_output_history[i][key] = []

        self._is_vectorize_policies = get_with_default(params, "vectorize_policies", False)
        # one policy maintained for each env
        if len(self._all_policies) == self._num_envs:
            logger.info(f"Using passed in policies for {self._num_envs} environment(s)")
            self._vectorized_policies = list(self._all_policies.values())
        elif self._is_vectorize_policies:
            logger.info(f"Duplicating policies")
            self._vectorized_policies = [self._policy] + [self._policy.duplicate() for _ in range(self._num_envs - 1)]
        else:
            self._vectorized_policies = [self._policy]

        # steps
        # NOTE: Define everything according to training steps. Quantities such as "rollouts" and "epochs"
        #       are bad because they are environment/data dependent.
        self._max_steps = int(params.max_steps)
        self._train_every_n_steps = int(params.train_every_n_steps)
        self._step_train_env_every_n_steps = int(params.step_train_env_every_n_steps)
        self._step_holdout_env_every_n_steps = int(params.step_holdout_env_every_n_steps)
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)
        self._save_data_train_every_n_steps = int(params.save_data_train_every_n_steps)
        self._save_data_holdout_every_n_steps = int(params.save_data_holdout_every_n_steps)
        self._block_train_on_first_n_steps = int(params.get("block_train_on_first_n_steps", 0))
        self._random_policy_on_first_n_steps = int(params.get("random_policy_on_first_n_steps", 0))
        self._write_to_tensorboard_every_n = int(
            get_with_default(params, "write_to_tensorboard_every_n_train_steps", 20))

        self._torchify_dataset = get_with_default(params, "torchify_dataset", False)
        self._torchify_device = get_with_default(params, "torchify_device", "cpu")

        self._load_statistics_initial = params.get("load_statistics_initial", True)
        self._reload_statistics_every_n_env_steps = int(params.get("reload_statistics_every_n_env_steps", 0))  # TODO
        self._reload_statistics_n_times = int(params.get("reload_statistics_n_times", -1))  # -1 means never stop
        if self._reload_statistics_n_times > 0:
            assert self._reload_statistics_every_n_env_steps > 0
        self._max_grad_norm = params.get("max_grad_norm", None)

        self._transfer_model_only = get_with_default(params, "transfer_model_only",
                                                     False)  # if true, loads only the model, not the training step
        self._checkpoint_model_file = params.get("checkpoint_model_file", None)
        # self._save_checkpoints = params.get("save_checkpoints", False)

        self._save_checkpoint_every_n_steps = get_with_default(params, "save_checkpoint_every_n_steps", 0)
        if self._save_checkpoint_every_n_steps > 0:
            assert self._save_checkpoint_every_n_steps % self._save_every_n_steps == 0, "Checkpointing steps should be a multiple of model save steps."
        self._use_all_devices = params.get("use_all_devices", torch.cuda.device_count() > 1)

        if self._use_all_devices and "cuda" in str(self._model.device):
            logger.debug(f"Using all available devices for model! GPU Count = {torch.cuda.device_count()}")
            # import torch.distributed
            # torch.distributed.init_process_group("nccl")
            # import ipdb; ipdb.set_trace()
            self._model.wrap_parallel()

        # fns
        self._process_env_step_output_fn = get_with_default(params, "process_env_step_output_fn",
                                                            lambda env, o, g, next_obs, next_goal, po, ea, d: (
                                                                next_obs, next_goal))

        # by default does nothing
        self._env_action_from_policy_output = get_with_default(params, "env_action_from_policy_output",
                                                               lambda env, model, o, g, po: po)

        self._current_step = 0
        self._current_train_step = 0
        self._current_train_loss = math.inf
        self._current_holdout_step = 0
        self._current_holdout_loss = math.inf

        # steps since reset env
        self._current_rollout_step = np.zeros(self._num_envs, dtype=int)  # TODO resolve this + vectorized
        self._current_env_train_step = 0

        # TODO phase this out
        if params.has_leaf_key('episode_return_buffer_len'):
            # an
            self._episode_return_buffer_len = int(params >> "episode_return_buffer_len")
            self._write_average_episode_returns_every_n_env_steps = int(
                get_required(params, "write_average_episode_returns_every_n_env_steps"))

            assert "trackers" not in params.keys(), "Tracker should not be specified with episodic returns as well"

            if self._episode_return_buffer_len > 0:
                assert self._episode_return_buffer_len >= self._num_envs  # this is a soft assertion, comment out if not necessary
                params.trackers = AttrDict(
                    returns=BufferedTracker(AttrDict(
                        buffer_len=int(np.round(self._episode_return_buffer_len / self._num_envs)),
                        buffer_freq=0,  # only clear on resets
                        time_agg_fn=lambda k, a, b: a + b,  # sum rewards
                        metric=ExtractMetric('returns', key='reward', source=1),
                        tracked_names=['returns'],
                    ))
                )
                params.tracker_write_frequencies = AttrDict(
                    returns=self._write_average_episode_returns_every_n_env_steps)
                params.tracker_is_batched = AttrDict(returns=True)
                params.tracker_write_types = AttrDict(returns=['mean', 'max', 'min', 'std'])

        # allows tracking multiple variables
        self._trackers = get_with_default(params, "trackers", AttrDict())
        self._tracker_is_batched = get_with_default(params, "tracker_is_batched",
                                                    self._trackers.leaf_apply(lambda _: True))
        self._tracker_write_frequencies = get_with_default(params, "tracker_write_frequencies", AttrDict())
        self._tracker_write_types = get_with_default(params, "tracker_write_types",
                                                     self._tracker_write_frequencies.leaf_apply(lambda _: ['mean']))

        # each element is a list, corresponding to the number of environments or 1 if not batched.
        self._batched_trackers = AttrDict()
        for name in self._trackers.leaf_keys():
            logger.debug(f'Tracker {name}')
            assert isinstance(self._trackers[name], Tracker)
            self._batched_trackers[name] = [self._trackers[name]]
            # replicate the tracker per env
            if self._tracker_is_batched and self._num_envs > 1:
                self._batched_trackers[name] += [self._trackers[name].duplicate() for _ in range(self._num_envs - 1)]

        # these all should have the same keys
        assert set(self._trackers.leaf_keys()) == set(self._tracker_write_frequencies.leaf_keys()) == \
               set(self._batched_trackers.leaf_keys()) == set(self._tracker_write_types.leaf_keys())

        # writing types are limited to this set:
        self._tracker_write_types.leaf_assert(lambda wt: all(t in ['mean', 'max', 'min', 'std'] for t in wt))

        # loggers / writers
        self._enable_writers = params.get("enable_writers", True)
        if self._enable_writers:
            l_path = os.path.join(self._file_manager.exp_dir, "loss.csv")
            csv_file = open(l_path, "a+")
            self._writer = csv.writer(csv_file, delimiter=',')
            self._summary_writer = SummaryWriter(self._file_manager.exp_dir)
        else:
            logger.warn("Writers are disabled!")
            self._writer = None
            self._summary_writer = None

        self._init_parameters(params)

        # optimizers
        self._init_optimizers(params)

        self._reset_curr_episode()

    # this allows training specific env reset functionality
    def _reset_curr_episode(self):
        pass

    # override this to make different optimizer schemes
    # NOTE: if this is overridden, need to override train_step as well
    def _init_optimizers(self, params):
        if len(list(self._model.parameters())) > 0:
            if not self._model.implements_train_step:
                # set up base optimizer / scheduler if model does not implement it.
                self._base_optimizer = params.base_optimizer(self._model.parameters())
                if "base_scheduler" in params.leaf_keys():
                    logger.debug("Using scheduler..")
                    self._base_scheduler = params.base_scheduler(self._base_optimizer)
                else:
                    self._base_scheduler = None
        else:
            logger.warn("Model has no parameters...")

    def _init_parameters(self, params):
        pass

    def run(self):
        """
        This is the main loop:
            - gather data
            - train the model
            - save the model
            - log progress
        """
        # NOTE: make sure you if you're experiment is killed that you can restart it where you left off
        load_success = self._restore_checkpoint()
        if not load_success and self._load_statistics_initial and len(self._dataset_train) > 0:
            logger.info("Loading statistics for new model with given data_train")
            self._model.load_statistics()

        if self._env_train is not None and self._step_train_env_every_n_steps > 0:
            obs_train, goal_train = self._env_train.reset()
        else:
            obs_train = AttrDict()
            goal_train = AttrDict()

        if self._env_holdout is not None and self._step_holdout_env_every_n_steps > 0:
            obs_holdout, goal_holdout = self._env_holdout.reset()
        else:
            obs_holdout = AttrDict()
            goal_holdout = AttrDict()

        # reset the policy or policies
        if self._is_vectorize_policies:
            for i in range(len(self._vectorized_policies)):
                self._vectorized_policies[i].reset_policy(next_obs=obs_train.leaf_apply(lambda arr: arr[None, i]),
                                                          next_goal=goal_train.leaf_apply(lambda arr: arr[None, i]))
        else:
            self._policy.reset_policy(next_obs=obs_train, next_goal=goal_train)

        # move data to torch / torch loader
        if self._torchify_dataset:
            self._dataset_train.torchify(self._torchify_device)  # TODO download more RAM

        # checking for data if we are training with no env
        if self._step_train_env_every_n_steps == 0 and len(self._dataset_train) == 0:
            raise Exception("Dataset is empty but no data is going to be collected")

        if self._save_data_train_every_n_steps > 0:
            self._dataset_train.create_save_dir()

        if self._save_data_holdout_every_n_steps > 0:
            self._dataset_holdout.create_save_dir()

        # log the model parameters
        logger.debug("Model parameters:")
        if hasattr(self._model, "print_parameters"):
            self._model.print_parameters()
        else:
            for p in self._model.parameters():
                logger.debug("PARAMETER: shape = %s, requires_grad = %s" % (p.shape, p.requires_grad))

        while self._current_step < self._max_steps:
            # NOTE: always have some form of timing so that you can find bugs
            with timeit('total_loop'):
                self._step_setup(obs_train, goal_train, obs_holdout, goal_holdout)

                # UPDATES
                if is_next_cycle(self._current_step, self._train_every_n_steps) and len(self._dataset_train) > 0:
                    if self._current_step > self._block_train_on_first_n_steps:
                        with timeit('train'):
                            self._train_step()

                if is_next_cycle(self._current_step, self._step_train_env_every_n_steps):
                    with timeit('train env'):
                        obs_train, goal_train = self._env_step(self._env_train, self._dataset_train,
                                                               obs_train, goal_train)

                if is_next_cycle(self._current_step, self._step_holdout_env_every_n_steps):
                    with timeit('holdout env'):
                        obs_holdout, goal_holdout = self._env_step(self._env_holdout, self._dataset_holdout,
                                                                   obs_holdout, goal_holdout)

                if is_next_cycle(self._current_step, self._holdout_every_n_steps) and len(self._dataset_holdout) > 0:
                    with timeit('holdout'):
                        self._holdout_step()

                # update step before saving, to match others
                self._current_step += 1

                # SAVE (optional checkpoint)
                if is_next_cycle(self._current_step, self._save_every_n_steps):
                    with timeit('save'):
                        self._save(chkpt=is_next_cycle(self._current_step, self._save_checkpoint_every_n_steps))

                if is_next_cycle(self._current_step, self._save_data_train_every_n_steps):
                    with timeit('save_data_train'):
                        self._dataset_train.save()

                if is_next_cycle(self._current_step, self._save_data_holdout_every_n_steps):
                    with timeit('save_data_holdout'):
                        self._dataset_holdout.save()

            if is_next_cycle(self._current_step, self._log_every_n_steps):
                self._log()

    def _step_setup(self, obs_train, goal_train, obs_holdout, goal_holdout):
        pass

    def _get_current_train_data_step(self):
        # raise NotImplementedError
        return len(self._dataset_train)

    def _get_current_holdout_data_step(self):
        # raise NotImplementedError
        return len(self._dataset_holdout)

    def _train_step(self):
        if len(self._dataset_train) == 0:
            logger.warn("Skipping training step since dataset is empty.")
            return

        # (B x H x ...)
        with timeit('train/get_batch'):
            with torch.no_grad():
                res = self._dataset_train.get_batch(torch_device=self._model.device)
                inputs, outputs = res[:2]
                meta = res[2] if len(res) == 3 else AttrDict()

        with timeit('train/data_augmentation'):
            if self._train_do_data_augmentation and self.data_augmentation is not None:
                inputs, outputs = self.data_augmentation.forward(inputs, outputs)

        self._model.train()

        sw = None
        if is_next_cycle(self._current_train_step, self._write_to_tensorboard_every_n):
            sw = self._summary_writer

        if self._model.implements_train_step:
            # Model defines train_step
            with timeit('train/model_train_step'):
                self._model.train_step(inputs, outputs, i=self._current_step, writer=sw,
                                       writer_prefix="train/", training=True, meta=meta)
        else:
            # default train step
            with timeit('train/loss'):
                loss = self._model.loss(inputs, outputs, i=self._current_step, writer=sw,
                                        writer_prefix="train/", training=True, meta=meta)

            with timeit('train/backprop'):
                self._base_optimizer.zero_grad()
                loss.backward()
                if self._max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)  # max
                self._base_optimizer.step()

            with timeit('train/detach_loss'):
                self._current_train_loss = loss.item()

            if self._base_scheduler is not None:
                with timeit("train/scheduler"):
                    self._base_scheduler.step()

        with timeit("writer"):
            if self._summary_writer is not None and is_next_cycle(self._current_train_step,
                                                                  self._write_to_tensorboard_every_n):
                self._summary_writer.add_scalar("train_step", self._current_train_step, self._current_step)
                if not self._model.implements_train_step:
                    for i, pg in enumerate(self._base_optimizer.param_groups):
                        self._summary_writer.add_scalar("train/learning_rate_pg_%d" % i, pg['lr'], self._current_step)
        self._current_train_step += 1

    def _holdout_step(self):
        # (B x H x ...)
        with timeit('total_holdout_step'):
            res = self._dataset_holdout.get_batch(torch_device=self._model.device)
            inputs, outputs = res[:2]
            meta = res[2] if len(res) == 3 else AttrDict()
            self._model.eval()
            loss = self._model.loss(inputs, outputs, i=self._current_step, writer=self._summary_writer,
                                    writer_prefix="holdout/", training=False, meta=meta)
            self._current_holdout_loss = loss.item()
        with timeit("writer"):
            if self._summary_writer is not None:
                self._summary_writer.add_scalar("holdout_step", self._current_holdout_step, self._current_step)
        self._current_holdout_step += 1

    def _env_step(self, env, dataset, obs, goal):
        # expand for horizon (all obs / goals have (B x ..) to start)
        with timeit("env_step/prep"):
            expanded_obs = obs.leaf_apply(lambda arr: to_torch(arr[:, None], device=self._model.device))
            expanded_goal = goal.leaf_apply(lambda arr: to_torch(arr[:, None], device=self._model.device))
        # actions will be (B x ..)

        self._model.eval()

        with torch.no_grad():
            # TODO clean this
            with timeit("env_step/get_action"):

                if self._current_step < self._random_policy_on_first_n_steps:
                    if self._is_vectorize_policies:
                        policy_outputs = [self._vectorized_policies[i].get_random_action(self._model,
                                                                                         expanded_obs.leaf_apply(
                                                                                             lambda arr: arr[None, i]),
                                                                                         expanded_goal.leaf_apply(
                                                                                             lambda arr: arr[None, i]))
                                          for i in range(self._num_envs)]
                        policy_output = AttrDict.leaf_combine_and_apply(policy_outputs,
                                                                        lambda vs: torch.cat(vs, dim=0)
                                                                        if all(is_array(a) for a in vs) else vs)
                    else:
                        policy_output = self._policy.get_random_action(self._model, expanded_obs, expanded_goal)
                else:
                    if self._is_vectorize_policies:
                        # slower possibly
                        policy_outputs = [self._vectorized_policies[i].get_action(self._model,
                                                                                  expanded_obs.leaf_apply(
                                                                                      lambda arr: arr[None, i]),
                                                                                  expanded_goal.leaf_apply(
                                                                                      lambda arr: arr[None, i]))
                                          for i in range(self._num_envs)]
                        policy_output = AttrDict.leaf_combine_and_apply(policy_outputs,
                                                                        lambda vs: torch.cat(vs, dim=0)
                                                                        if all(is_array(a) for a in vs) else vs)
                    else:
                        policy_output = self._policy.get_action(self._model, expanded_obs, expanded_goal)

        with timeit("env_step/get_env_action"):
            # filter torch action to np
            np_policy_output = policy_output.leaf_apply(
                lambda arr: to_numpy(arr) if isinstance(arr, torch.Tensor) else arr)
            np_policy_output = env.env_spec.map_to_types(np_policy_output, skip_keys=True)

            # this gets the "environment" action (e.g. if policy outputs relative, but env takes absolute)
            np_env_action = self._env_action_from_policy_output(env, self._model, obs, goal, np_policy_output)
            all_action = np_policy_output & np_env_action

        with timeit("env_step/step"):
            # actual step
            next_obs, next_goal, done = env.step(all_action)
            if self._is_vectorize_policies:
                policy_done = [self._vectorized_policies[i].is_terminated(self._model, obs, goal) for i in
                               range(self._num_envs)]
            else:
                policy_done = [self._policy.is_terminated(self._model, obs, goal)] * self._num_envs
            done = np.logical_or(done, policy_done)

        with timeit("env_step/map_and_process_transition"):
            # np_env_action = env_action.leaf_apply(lambda arr: to_numpy(arr) if isinstance(arr, torch.Tensor) else arr)
            np_env_action = env.env_spec.map_to_types(np_env_action, skip_keys=True)

            # inputs, outputs should be (1 x ...) for compatibility with dataset
            inputs, outputs, next_obs, next_goal = self._process_transition(env, obs, goal, next_obs, next_goal,
                                                                            np_policy_output, np_env_action, done)

        # TODO ADDING REWARDS
        # if 'reward' in outputs.leaf_keys():
        #     self._current_episode_returns += outputs.reward[:, 0]
        # for name, tracker_ls in self._batched_trackers.leaf_items():
        #     for tracker in tracker_ls:
        #         tracker.compute_and_update(inputs)

        # adds keys that aren't present to new dc. Modifies TODO put this in a better spot
        def safe_add_keys_to_match(original: AttrDict, dc: AttrDict, default_fn=lambda key: list()):
            not_present = set(list(original.leaf_keys())).difference(list(set(dc.leaf_keys())))
            for key in not_present:
                dc[key] = default_fn(key)
            return not_present

        def np_concat_ignore_empty(vs):
            updated_vs = [v for v in vs if len(v) > 0]
            return np.concatenate(updated_vs)

        with timeit("env_step/dataset_add"):
            for i in range(self._num_envs):
                inputs_i = inputs.leaf_arrays().leaf_apply(lambda arr: arr[None, i])  # just get this
                outputs_i = outputs.leaf_arrays().leaf_apply(lambda arr: arr[None, i])  # just get this

                # tracker update
                for tracker_name, tracker_ls in self._batched_trackers.leaf_items():
                    # non-batched trackers have 1 element in ^^
                    if self._tracker_is_batched[tracker_name] or i == 0:
                        tracker_ls[i].compute_and_update(inputs_i, outputs_i, AttrDict())

                if self._is_vectorized:
                    # TODO only add if not vectorized or episodes are individually finished (also smarter done handling)

                    # makes sure missing keys are replaced with empty lists, for compatibility with concatenation later
                    safe_add_keys_to_match(self._vectorized_input_history[i], inputs_i)
                    safe_add_keys_to_match(self._vectorized_output_history[i], outputs_i)

                    # extend list of episode returns
                    self._vectorized_input_history[i] = AttrDict.leaf_combine_and_apply(
                        [self._vectorized_input_history[i], inputs_i], lambda vs: vs[0] + [vs[1]], match_keys=False)
                    self._vectorized_output_history[i] = AttrDict.leaf_combine_and_apply(
                        [self._vectorized_output_history[i], outputs_i], lambda vs: vs[0] + [vs[1]], match_keys=False)
                    # add episode and clear if we are at done == True
                    if self._vectorized_output_history[i].done[-1][0]:  # latest done element, index into (1,) done
                        # list of (1, ...) => np.array (L, ...)
                        episode_inputs = self._vectorized_input_history[i].leaf_apply(np_concat_ignore_empty)
                        episode_outputs = self._vectorized_output_history[i].leaf_apply(np_concat_ignore_empty)
                        # add full episode to dataset
                        logger.debug(
                            "Adding episode from env %d to dataset of length: %d" % (i, episode_outputs.done.shape[0]))
                        dataset.add_episode(episode_inputs, episode_outputs)
                        # clear entries for this env
                        self._vectorized_input_history[i].leaf_call(lambda ls: ls.clear())  # reset in place
                        self._vectorized_output_history[i].leaf_call(lambda ls: ls.clear())  # reset in place
                else:
                    dataset.add(inputs, outputs, rollout_step=self._current_rollout_step[0])

        # mem profile dataset???
        # if self._current_env_train_step % 100 == 0:
        #     print("Trainer", file_utils.get_size(self))
        #     print("Model", file_utils.get_size(self._model))
        #     print("Train data", file_utils.get_size(self._dataset_train))
        #     print("Holdout data", file_utils.get_size(self._dataset_holdout))
        #     print("Env train", file_utils.get_size(self._env_train))
        #     print("Writer", file_utils.get_size(self._writer))

        #     from pympler import muppy, summary
        #     all_objects = muppy.get_objects()
        #     sum1 = summary.summarize(all_objects)
        #     summary.print_(sum1)
        # s = pickle.dumps(self._dataset_train)
        # print("**** Dataset train size: ", len(s))

        done = outputs.done  # (num_envs,)
        done_idxs = done.nonzero()[0]

        # reset all the trackers where the episode finished.
        for idx in done_idxs:
            for tracker_name, tracker_ls in self._batched_trackers.leaf_items():
                if self._tracker_is_batched[idx] or idx == 0:
                    tracker_ls[idx].reset_tracked_state()

        # new_returns = self._current_episode_returns[done_idxs].tolist()
        # self._episode_return_buffer.extend(new_returns)  # record these returns
        # self._episode_return_buffer = self._episode_return_buffer[-self._episode_return_buffer_len:]  # keep the newest
        # self._current_episode_returns[done_idxs] = 0  # clear current returns if done

        # redundant kinda
        with timeit("env_step/reset"):
            if not self._is_vectorized:
                if done[0]:
                    next_obs, next_goal = env.reset()
                    # single policy reset only happens when training on 1 thread (doesn't make sense otherwise)
                    self._policy.reset_policy(next_obs=next_obs, next_goal=next_goal)
            else:
                # TODO this will break for episode lengths that aren't fixed
                next_obs_partial, next_goal_partial = env.reset_where(done_idxs)

                assert len(done_idxs) == 0 or len(list(next_obs_partial.leaf_items())) > 0
                assert len(self._env_train.env_spec.final_names) == 0 or len(done_idxs) in [0,
                                                                                            self._num_envs], "TODO support async envs + final/start this"

                # only reset each policy if env is done and there are separate policies for each env
                if self._is_vectorize_policies:
                    for i, idx in enumerate(done_idxs):
                        self._vectorized_policies[idx].reset_policy(
                            next_obs=next_obs_partial.leaf_apply(lambda arr: arr[None, i]),
                            next_goal=next_goal_partial.leaf_apply(lambda arr: arr[None, i]))

                def combine_func(vals):
                    arr = np.copy(vals[0])
                    arr[done_idxs] = vals[1]
                    return arr

                if len(done_idxs) > 0:
                    spec = self._env_train.env_spec
                    # this needs to happen since next_obs might contain output obs names or final names... no bueno
                    next_obs = AttrDict.leaf_combine_and_apply(
                        [next_obs.leaf_filter_keys(spec.observation_names + spec.param_names),
                         next_obs_partial.leaf_filter_keys(spec.observation_names + spec.param_names)],
                        func=combine_func)
                    next_goal = AttrDict.leaf_combine_and_apply([next_goal, next_goal_partial], func=combine_func)

        self._current_rollout_step += 1
        self._current_rollout_step[done_idxs] = 0

        ## WRITING

        # do things that happen on cycle for env_train here (since environments can be vectorized)
        for _ in range(self._num_envs):

            # TODO writing
            # with timeit("writer"):
            #     if self._summary_writer is not None:
            #         if is_next_cycle(self._current_env_train_step, self._write_average_episode_returns_every_n_env_steps):
            #             self._summary_writer.add_scalar("env/mean_returns", np.array(self._episode_return_buffer).mean(),
            #                                             self._current_step)
            #             self._summary_writer.add_scalar("env/max_returns", max(self._episode_return_buffer),
            #                                             self._current_step)
            #             self._summary_writer.add_scalar("env/min_returns", min(self._episode_return_buffer),
            #                                             self._current_step)
            # writing returns

            for tracker_name, tracker_ls in self._batched_trackers.leaf_items():
                if is_next_cycle(self._current_env_train_step, self._tracker_write_frequencies[tracker_name]) and \
                        any(tr.has_data() for tr in tracker_ls):
                    # each of the batched trackers for this name has some time series output, which we will average
                    ts_outputs = AttrDict.from_kvs(tracker_ls[0].tracked_names,
                                                   [[] for _ in range(len(tracker_ls[0].tracked_names))])
                    for idx, tracker in enumerate(tracker_ls):
                        if tracker.has_data():
                            ts_data = tracker.get_time_series().leaf_apply(lambda arr: np.asarray(arr)[None])  # 1 x T
                            ts_outputs = AttrDict.leaf_combine_and_apply([ts_outputs, ts_data],
                                                                         lambda vs: vs[0] + [vs[1]])

                    # B x T, after concat
                    ts_outputs = ts_outputs.leaf_apply(lambda vs: np.concatenate(vs, axis=0))
                    writing_types = self._tracker_write_types[tracker_name]
                    for key, arr in ts_outputs.leaf_items():
                        if len(arr) > 0:
                            if 'mean' in writing_types:
                                self._summary_writer.add_scalar("env/" + tracker_name + "/" + key + "_mean", arr.mean(),
                                                                self._current_step)
                            if 'max' in writing_types:
                                self._summary_writer.add_scalar("env/" + tracker_name + "/" + key + "_max", arr.max(),
                                                                self._current_step)
                            if 'min' in writing_types:
                                self._summary_writer.add_scalar("env/" + tracker_name + "/" + key + "_min", arr.min(),
                                                                self._current_step)
                            if 'std' in writing_types:
                                self._summary_writer.add_scalar("env/" + tracker_name + "/" + key + "_std", arr.std(),
                                                                self._current_step)

            # reloading statistics
            if self._reload_statistics_n_times <= 0 or \
                    self._current_env_train_step <= self._reload_statistics_every_n_env_steps * self._reload_statistics_n_times:
                # only reload when train_step <= self._reload_stats * self._reload_stats_n_times
                if is_next_cycle(self._current_env_train_step + 1, self._reload_statistics_every_n_env_steps):
                    logger.warn("Reloading statistics from dataset")
                    self._model.load_statistics()
            self._current_env_train_step += 1  # this might need to be changed, hacky

        return next_obs, next_goal

    def _restore_checkpoint(self):
        if self._checkpoint_model_file is None:
            return False
        path = os.path.join(self._file_manager.models_dir, self._checkpoint_model_file)
        if os.path.isfile(path):
            checkpoint = torch.load(str(path))
            self._model.restore_from_checkpoint(checkpoint)
            if not self._transfer_model_only:
                self._current_step = checkpoint['step']
                self._current_train_step = checkpoint['train_step']  # todo add env
                self._current_holdout_step = checkpoint['holdout_step']
                self._current_train_loss = checkpoint['train_loss']
                self._current_holdout_loss = checkpoint['holdout_loss']
            logger.debug("Loaded model from {}, current train step: {}".format(path, self._current_step))
            return True
        else:
            logger.warn("Unable to load model!")
            return False

    # takes the output of env step, and turns it into inputs and outputs for dataset
    # NOTE: onetime (param & final) keys, should be in inputs and outputs if they are present
    def _process_transition(self, env, obs, goal, next_obs, next_goal, policy_output, env_action, done):
        # gets the next_obs (output_obs_names) and next_goal
        outputs_next_obs, outputs_next_goal = self._process_env_step_output_fn(env, obs, goal, next_obs, next_goal,
                                                                               policy_output,
                                                                               env_action, done)

        # policy should output what we want, if env does not
        for name in env.env_spec.output_observation_names:
            if name not in outputs_next_obs.leaf_keys():
                assert name in policy_output.leaf_keys() or name in env_action.leaf_keys(), name
                copy_from = policy_output if name in policy_output.leaf_keys() else env_action
                outputs_next_obs[name] = to_numpy(copy_from[name], check=True).astype(
                    env.env_spec.names_to_dtypes[name])

        inputs = obs.leaf_copy()
        inputs.combine(policy_output)
        inputs.combine(env_action)

        outputs = outputs_next_obs.leaf_copy() & outputs_next_goal
        if "done" not in outputs.leaf_keys():
            outputs.done = done

        return inputs, outputs, next_obs, next_goal

    def _save(self, chkpt=False):
        base_fname = "model.pt"
        path = os.path.join(self._file_manager.models_dir, base_fname)
        torch.save({'step': self._current_step,
                    'train_step': self._current_train_step,
                    'holdout_step': self._current_holdout_step,
                    'train_loss': self._current_train_loss,
                    'holdout_loss': self._current_holdout_loss,
                    'model': self._model.state_dict()}, path)
        logger.debug("Saved model")
        if chkpt:
            chkpt_base_fname = "chkpt_{:010d}.pt".format(self._current_step)
            shutil.copyfile(path, os.path.join(self._file_manager.models_dir, chkpt_base_fname))
            logger.debug(f"Saved checkpoint: {self._current_step}")

    def _log(self):
        logger.info('[{}] (steps, loss) -> TRAIN: ({}, {}), HOLDOUT: ({}, {})'  # , AVG RETURN: {}±{}' TODO
                    .format(self._current_step,
                            self._current_train_step, self._current_train_loss,
                            self._current_holdout_step, self._current_holdout_loss,
                            # np.asarray(self._episode_return_buffer).mean(),
                            # np.asarray(self._episode_return_buffer).std(),
                            )
                    )
        # try:
        #     msg = str(timeit)
        # except Exception as e:
        #     msg = "Timeit unavailable: %s" % e
        logger.debug(timeit)
        timeit.reset()
