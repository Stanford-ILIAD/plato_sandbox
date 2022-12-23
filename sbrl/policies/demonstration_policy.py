"""
The policy runs actions from a pre-recorded sequence, then

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""
import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import AttrDict


class DemonstrationPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # self.sample_action = params.sample_action
        # takes in model, obs, goal  (all will have shape (B, H, ...) -> action (B, ...)
        self.input_files = params.input_files
        self.random_episode = params.get("random_episode", True)

        # this function converts the sampled episode -> actions at each step (useful if you want to ignore/modify recorded actions)
        self.actions_from_datadict_fn = params.get("actions_from_datadict_fn", DemonstrationPolicy._actions_from_datadict)
        assert isinstance(self._file_manager, ExperimentFileManager), "DemonstrationPolicy needs the file manager"

        self.valid_episode_indices = params.valid_episode_indices

    def _init_setup(self):
        params = AttrDict(
            file=self.input_files,
            output_file="/tmp/empty.npz",
            capacity=1e3,
            horizon=1,
            batch_size=10
        )
        self._dataset = NpDataset(params, self._env_spec, self._file_manager)
        print("Loaded with %d samples" % len(self._dataset))
        print("Split IDXS: %s" % self._dataset.split_indices())
        num_eps = len(self._dataset.split_indices())
        assert num_eps > 0, "Need more than 1 complete episode in file"
        logger.debug("Number of episodes loaded: %d" % num_eps)
        if not isinstance(self.valid_episode_indices, list) and not isinstance(self.valid_episode_indices, np.ndarray):
            logger.debug("Policy using all valid episodes in file")
            self.valid_episode_indices = np.arange(len(self._dataset.split_indices()))
        else:
            self.valid_episode_indices = np.array(self.valid_episode_indices)
            logger.debug("Policy using the following episodes: %s" % self.valid_episode_indices)

        self.curr_episode_idx = -1
        self.curr_idx_in_episode = 0
        self.curr_episode = AttrDict()
        self.memory = AttrDict()

    def reset_policy(self, **kwargs):
        self.memory = AttrDict()

    def warm_start(self, model, observation, goal):
        pass

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)
        note: kwargs are for model forward fn

        :return action: AttrDict (B x ...)
        """
        actions = self.actions_from_datadict_fn(self, model, self.curr_episode, observation, self.curr_idx_in_episode)
        actions = actions.leaf_apply(lambda arr: arr[None])
        # print(actions)
        self._env_spec.clip(actions, self._env_spec.action_names)
        actions = self._env_spec.map_to_types(actions)
        ep_len = self._dataset.episode_length(self.valid_episode_indices[self.curr_episode_idx])
        # never advance past the last idx
        self.curr_idx_in_episode += 1
        if self.curr_idx_in_episode >= ep_len:
            self.curr_idx_in_episode = ep_len - 1

        return actions

    def reset_policy(self, **kwargs):
        # pick new episode
        if self.random_episode:
            self.curr_episode_idx = np.random.choice(len(self.valid_episode_indices))
        else:
            self.curr_episode_idx = (self.curr_episode_idx + 1) % len(self.valid_episode_indices)  # rolling

        self.curr_episode = self._dataset.get_episode(self.valid_episode_indices[self.curr_episode_idx], self._env_spec.all_names)
        self.curr_idx_in_episode = 0

    # gets action at step idx
    def _actions_from_datadict(self, model, sampled_datadict: AttrDict, observation: AttrDict, idx) -> AttrDict:
        return sampled_datadict.leaf_filter(func=lambda k, v: k in self._env_spec.action_names).leaf_apply(lambda arr: arr[idx])

