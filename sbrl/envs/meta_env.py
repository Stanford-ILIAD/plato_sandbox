"""
This is where actions actually get executed on the robot, and observations are received.

NOTE: Why is there no reward? Are we even doing reinforcement learning???
      The reward is just another observation! Viewing it this way is much more flexible,
      especially with model-based RL

"""

from sbrl.envs.env import Env
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_required, get_with_default


class MetaEnv(Env):

    def __init__(self, params, env_spec):
        super(MetaEnv, self).__init__(params, env_spec)

        # lower level policy params
        self._all_envs = list(get_required(params, "all_envs"))
        self._obs_histories = [[] for _ in range(len(self._all_envs))]
        self._goal_histories = [[] for _ in range(len(self._all_envs))]
        logger.debug("Using envs:")
        for i, env in enumerate(self._all_envs):
            if isinstance(env, AttrDict):
                pr = env.pprint(str_max_len=50, ret_string=True)
            else:
                pr = str(env)
            logger.debug(f"[{i}] --------------------------------------\n{pr}")

        # takes (idx, last obs histories)
        self._next_param_fn = get_with_default(params, "next_param_fn",
                                               lambda idx, *args, **kwargs: (0 if idx is None else (idx + 1) % len(self._all_envs),
                                                                             AttrDict()))

        assert len(self._all_envs) > 0

        self._debug = False

        for i, p in enumerate(self._all_envs):
            # create or load policies
            if isinstance(p, AttrDict):
                self._all_envs[i] = p.cls(p.params, self._env_spec)
            assert isinstance(self._all_envs[i], Env), type(self._all_envs[i])

        self._curr_env_idx = None
        self._curr_env = None

        self._env_done = False

    def step(self, action, **kwargs):
        assert self._curr_env is not None, "Call reset before stepping"
        next_obs, next_goal, done = self._curr_env.step(action, **kwargs)
        self.record(next_obs, next_goal, self._curr_env_idx)
        return next_obs, next_goal, done

    def get_current_idx(self):
        return self._curr_env_idx

    def record(self, obs: AttrDict, goal: AttrDict, idx: int):
        self._obs_histories[idx].append(obs)
        self._goal_histories[idx].append(goal)

    def clear_history(self, idx: int):
        self._obs_histories[idx].clear()
        self._goal_histories[idx].clear()

    def reset(self, presets: AttrDict = AttrDict()):
        """
        presets: some episodes support this
         - disable_user: Bool, if you are using

         returns the next episode obs, goal.
        """

        # update to the next env
        self._curr_env_idx, next_presets = self._next_param_fn(self._curr_env_idx, self._obs_histories)
        self._curr_env = self._all_envs[self._curr_env_idx]

        # remove history for the new idx after transition (do this after next_param_fn)
        self.clear_history(self._curr_env_idx)

        # reset the current env
        obs, goal = self._curr_env.reset(presets.combine(next_presets, ret=True))
        # register the obs to the old idx (for next run of this env)
        self.record(obs, goal, self._curr_env_idx)
        return obs, goal
