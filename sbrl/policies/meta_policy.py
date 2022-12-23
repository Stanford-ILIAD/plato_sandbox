"""
Meta-Policies interact with an environment every so often, over a sequence of lower level policies.
The meta-policy never calls step on the underlying env, but rather chooses a low level policy for each step until that policy has "finished"
All base-policies must have a termination condition implemented, or we won't cycle
"""
import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import AttrDict, get_required, get_with_default


class MetaPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # lower level policy params
        self._all_policies = list(get_required(params, "all_policies"))
        self._max_steps_per_policy = get_with_default(params, "max_steps_per_policy", None)

        logger.debug("Using policies:")
        for i, pol in enumerate(self._all_policies):
            if isinstance(pol, AttrDict):
                pr = pol.pprint(str_max_len=50, ret_string=True)
            else:
                pr = str(pol)
            logger.debug(f"[{i}] --------------------------------------\n{pr}")

        # returns policy idx and params to reset with
        self._next_param_fn = get_with_default(params, "next_param_fn",
                                               lambda idx, *args, **kwargs: ((idx + 1) % len(self._all_policies),
                                                                             AttrDict()))

        assert len(self._all_policies) > 0

        self._curr_policy_idx = None
        self._last_action = None
        self._curr_policy = None
        self._curr_policy_step = 0

        self._policy_done = False
        self._memory = AttrDict()

        self._debug = False

    def _init_setup(self):
        for i, p in enumerate(self._all_policies):
            # create or load policies
            if isinstance(p, AttrDict):
                self._all_policies[i] = p.cls(p.params, self._env_spec, self._file_manager, env=self._env)
            assert isinstance(self._all_policies[i], Policy), type(self._all_policies[i])

    def warm_start(self, model, observation, goal):
        for p in self._all_policies:
            p.warm_start(model, observation, goal)

    def get_next_policy(self, policy_idx, model, observation, goal, **kwargs):
        next_policy_idx, next_policy_presets = self._next_param_fn(policy_idx, model, observation, goal,
                                                                   memory=self._memory, env=self._env, **kwargs)
        next_policy = None
        if next_policy_idx is not None:
            next_policy = self._all_policies[next_policy_idx]
            next_policy.reset_policy(**next_policy_presets.as_dict())  # reset the policy with new information

        return next_policy_idx, next_policy, next_policy_presets

    # override if meta controller has a stronger move-on condition, use low level policy's by default
    def is_curr_policy_terminated(self, model, observation, goal, **kwargs) -> bool:
        if self._max_steps_per_policy is not None and self._curr_policy_step >= self._max_steps_per_policy:
            return True
        return self._curr_policy.is_terminated(model, observation, goal, **kwargs)

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        # assert self._curr_policy is not None, "Must instantiate a base-policy before calling get action"
        self._curr_policy_step += 1

        policy_switch = False  # true when a new policy is loaded
        if self._curr_policy is None or self.is_curr_policy_terminated(model, observation, goal, **kwargs):

            if self._debug:
                logger.debug("Moving to next policy..")
            self._curr_policy_idx, self._curr_policy, presets = self.get_next_policy(self._curr_policy_idx, model, observation, goal, **kwargs)

            if self._curr_policy_idx is None:
                self._policy_done = True
                self._curr_policy = None
                # if self._curr_policy is None:  # edge case where there was no previous policy, default to 0th
                #     self._curr_policy = self._all_policies[0]  # default starting policy
                #     self._curr_policy.reset_policy(**presets.as_dict())
            else:
                self._curr_policy_step = 0  # back to zero
                policy_switch = True
                self._curr_policy = self._all_policies[self._curr_policy_idx]
                self._curr_policy.reset_policy(**presets.as_dict())  # reset the policy with new information

        if self._curr_policy is not None:
            self._last_action = self._curr_policy.get_action(model, observation, goal, **kwargs)
        elif self._last_action is not None:
            self._last_action = self._last_action.leaf_copy()
        else:
            logger.warn("No policy specified and no last action available, returning zeros!")
            self._last_action = self._env.env_spec.get_zeros(self._env.env_spec.action_names, 1)

        sample_ac = self._last_action.get_one()  # getting return type, to add policy switch bool
        if isinstance(sample_ac, torch.Tensor):
            return self._last_action & AttrDict(
                policy_switch=torch.BoolTensor([policy_switch], device=sample_ac.device)[None],
                policy_step=torch.LongTensor([self._curr_policy_step], device=sample_ac.device)[None],
            )
        else:
            return self._last_action & AttrDict(policy_switch=np.array([policy_switch])[None],
                                                policy_step=np.array([self._curr_policy_step])[None],
                                                )

    def reset_policy(self, **kwargs):
        # # TODO does this make sense?
        # for p in self._all_policies:
        #     p.reset_policy(**kwargs)

        self._curr_policy_idx = None
        self._curr_policy = None
        self._last_action = None

        self._policy_done = False
        self._memory.clear()

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._policy_done

    @property
    def curr_policy_idx(self):
        return self._curr_policy_idx if self._curr_policy_idx is not None else -1

    @property
    def curr_policy(self):
        return self._curr_policy
