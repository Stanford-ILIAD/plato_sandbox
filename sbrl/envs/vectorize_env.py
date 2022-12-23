"""
This is where actions actually get executed on the robot, and observations are received.

"""
import time
from multiprocessing import Process, Pipe
from typing import Callable

import numpy as np

from sbrl.envs.env import Env
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.point_mass_env import PointMassEnv
from sbrl.experiments import logger
from sbrl.utils.core_utils import CloudPickleWrapper
from sbrl.utils.python_utils import AttrDict, get_with_default, is_array, get_required
from sbrl.utils.torch_utils import to_numpy


class VectorizedEnv(Env):
    """
    Creates multiple envs with their own parameters in different processes
    synchronizes steps and episode runs
    """

    def __init__(self, params, env_spec):
        # shared env spec
        super().__init__(params, env_spec)
        self.num_envs = get_required(params, "num_envs")  # num process
        self.base_env_params = get_required(params, "env_params")
        self.step_outputs_as_list = get_with_default(params, "step_outputs_as_list", False)

        # self.all_envs = []
        # customization of parameters
        each_env_params = [params.env_params for _ in range(self.num_envs)]
        # must return a new attr dict!!
        if isinstance(params.get("modify_env_params_by_index", None), Callable):
            for i in range(self.num_envs):
                # construct env params
                each_env_params[i] = params.modify_env_params_by_index(each_env_params[i])
                # create env here (to avoid failures)
                # self.all_envs.append(each_env_params[i].cls(each_env_params[i].params, env_spec))

        self._waiting = False
        self._closed = False

        self._obs = AttrDict()
        self._goals = AttrDict()
        self._dones = np.array([False] * self.num_envs)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [Process(target=VectorizedEnv.process_work, args=(work_remote, remote, CloudPickleWrapper(env_params), CloudPickleWrapper(env_spec)))
                   for (work_remote, remote, env_params) in zip(self.work_remotes, self.remotes, each_env_params)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    @staticmethod
    def process_work(remote, parent_remote, env_params_wrapper, env_spec_wrapper):
        parent_remote.close()
        seed = int(time.time() * 1e9) % 1000
        print("Random seed:", seed)
        np.random.seed(seed)
        env_params = env_params_wrapper.x
        env = env_params.cls(env_params.params, env_spec_wrapper.x)
        logger.debug("Env created: %s" % env)
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, goal, done = env.step(data, )
                remote.send((ob, goal, done))
            elif cmd == 'reset':
                ob, goal = env.reset(data)
                remote.send((ob, goal))
            elif cmd == 'close':
                remote.close()
                break
            elif hasattr(env, cmd):
                remote.send(getattr(env, cmd)(data))
            else:
                raise NotImplementedError
        logger.debug("Env closing: %s" % env)

    def step(self, action):
        """
        :param action: AttrDict (B x ...)

        :return obs, goal, done: AttrDict (B x ...)
        """
        self.step_async(action)
        return self.step_wait()

    def step_async(self, actions):
        for i in range(self.num_envs):
            remote = self.remotes[i]
            action = actions.leaf_filter(lambda k, v: is_array(v))\
                .leaf_apply(lambda arr: to_numpy(arr[i:i + 1], check=True))  # (1, ...)
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, goals, dones = zip(*results)
        # envs might return varying lengths (e.g. episodic)
        self._dones = np.concatenate(dones, axis=0)
        if not self.step_outputs_as_list:
            self._obs = AttrDict.leaf_combine_and_apply(obs, func=lambda arr_list: np.concatenate(arr_list, axis=0))
            self._goals = AttrDict.leaf_combine_and_apply(goals, func=lambda arr_list: np.concatenate(arr_list, axis=0))
        else:
            # return list, don't try to concatenate, remove the (1 x ) dimension
            return [dc.leaf_apply(lambda arr: np.asarray(arr)[0]) for dc in obs], \
                   [dc.leaf_apply(lambda arr: np.asarray(arr)[0]) for dc in goals], \
                   self._dones

        return self._obs, self._goals, self._dones

    # do not call this unless you want to force reset everything
    def reset(self, presets: AttrDict = AttrDict()):
        logger.warn("Calling reset on ALL idxs. ur gonna break something, just sayin")
        for remote in self.remotes:
            remote.send(('reset', presets))
        results = np.stack([remote.recv() for remote in self.remotes])
        obs, goals = zip(*results)
        self._obs = AttrDict.leaf_combine_and_apply(obs, func=lambda arr_list: np.concatenate(arr_list, axis=0))
        self._goals = AttrDict.leaf_combine_and_apply(goals, func=lambda arr_list: np.concatenate(arr_list, axis=0))

        return self._obs, self._goals

    def reset_where(self, idxs, presets: AttrDict = AttrDict()):
        idxs = idxs.astype(int)
        if idxs.shape[0] == 0:
            return self._obs, self._goals
        assert idxs.shape[0] <= self.num_envs
        assert np.unique(idxs).shape[0] == idxs.shape[0]  # no duplicates
        for i in idxs:
            self.remotes[i].send(('reset', presets))
        results = np.stack([self.remotes[i].recv() for i in idxs])
        obs, goals = zip(*results)

        self._obs = AttrDict.leaf_combine_and_apply(obs, func=lambda arr_list: np.concatenate(arr_list, axis=0))
        self._goals = AttrDict.leaf_combine_and_apply(goals, func=lambda arr_list: np.concatenate(arr_list, axis=0))

        return self._obs, self._goals

    def close(self):
        if self._closed:
            return
        if self._waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self._closed = True

    def call_sync(self, cmd, data, i=None):
        if i is not None:
            self.remotes[i].send((cmd, data))
            return self.remotes[i].recv()
        else:
            results = []
            for remote in self.remotes:
                remote.send((cmd, data))
                res = remote.recv()
                results.append(res)
            return results

    def __len__(self):
        return self.num_envs


if __name__ == '__main__':
    params = AttrDict(
        num_envs=5,
        modify_env_params_by_index=lambda p: p.copy(),
        env_params=AttrDict(
            cls=PointMassEnv,
            params=AttrDict(
                render=False,
                num_steps=100,
                noise_std=0.,
                theta_noise_std=1.0,
                target_speed=0.5,  # max speed
            )
        )
    )
    env_spec = ParamEnvSpec(AttrDict(
        names_shapes_limits_dtypes=[
            ('obs', (4,), (0, 1), np.float32),
            ('next_obs', (4,), (0, 1), np.float32),

            ('reward', (1,), (-np.inf, np.inf), np.float32),

            ('action', (2,), (-1, 1), np.float32),
        ],
        output_observation_names=['next_obs', 'reward'],
        observation_names=['obs'],
        action_names=['action'],
        goal_names=[],
    ))

    venv = VectorizedEnv(params, env_spec)

    print(venv.all_envs)

    obs, goal = venv.reset()

    assert obs.obs.shape == (5, 4)
    assert obs.reward.shape == (5, 1)

    actions = AttrDict(action=np.stack([1 + np.ones(2) * i for i in range(5)]))

    next_obs, next_goal, dones = venv.step(actions)

    assert next_obs.obs.shape == (5, 4)
    assert next_obs.reward.shape == (5, 1)
    assert dones.shape == (5,)

    venv.close()


