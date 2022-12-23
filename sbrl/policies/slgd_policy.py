"""
Stochastic Langevin Gradient Descent -- based on https://github.com/google-research/ibc/blob/master/ibc/agents/
"""
import numpy as np
import torch

from sbrl.policies.random_shooting import OptimizerPolicy
from sbrl.utils.python_utils import AttrDict as d, get_with_default, timeit
from sbrl.utils.torch_utils import broadcast_dims, torch_disable_grad, torch_clip, to_torch, \
    unsqueeze_then_gather


def langevin_step(energy_network, observation: d, sampled_action: d, max_action: d, min_action: d,
                  delta_action_clip=0.1, l_lambda=1., noise_scale=1.0,
                  expand_obs=True, energy_key="energy", copy=False, prefix='', **kwargs):
    """
    Takes a langevin step through energy network:  act - l_lambda ( 0.5 * d_energy / d_act + noise), noise ~ N(0, noise_scale)

    :param energy_network:
    :param observation: (B, ...) or (B, N, ...) if expand_obs=False, torch.Tensor
    :param sampled_action: (B, N, ...), where N is the number of samples, torch.Tensor
    :param l_lambda: langevin update size
    :param noise_scale: noise sigma to add to gradient (TODO allow tensors here)
    :param expand_obs: if True, will expand obs to match ac shape in dim 1
    :param energy_key: model output name
    :param copy: if True, will copy the action tensors first, before gradients
    :param kwargs: extra model arguments
    :return:
    """

    B, N = sampled_action.get_one().shape[:2]
    if expand_obs:
        observation = observation.leaf_apply(lambda arr: broadcast_dims(arr[:, None], [1], [N]))

    if copy:
        sampled_action = sampled_action.leaf_apply(lambda arr: arr.clone())

    with torch.enable_grad():
        # make actions require_gradient and clear their gradients. equivalent to optimizer.zero_grad() on actions
        for arr in sampled_action.leaf_values():
            arr.requires_grad = True
            arr.grad = None

        with timeit(f'{prefix}langevin_forward'):
            inputs = observation & sampled_action
            # prevent energy_network param gradients
            with torch_disable_grad(energy_network, eval_mode=False):
                outs = energy_network(inputs, **kwargs)
            energy_arr = outs >> energy_key

        with timeit(f'{prefix}langevin_backward'):
            # should get us de / dact, which we want to minimize
            energy_arr.sum().backward()

    # TODO grad clip?

    sampled_grad = sampled_action.leaf_apply(lambda arr: arr.grad)
    # langevin dynamics
    action_update = sampled_grad.leaf_apply(lambda g: (l_lambda * 0.5 * g + torch.randn_like(g) * noise_scale))

    out_ac = d()
    for key, update in action_update.leaf_items():
        mna = min_action[key]
        mxa = max_action[key]

        if delta_action_clip is not None:
            # clipping update to be some fraction of the action range
            delta_action = delta_action_clip * 0.5 * (mxa - mna)
            mid_action = 0.5 * (mxa + mna)
            update = torch_clip(update, mid_action - delta_action, mid_action + delta_action)

        # langevin step, new actions must be in bounds
        out_ac[key] = torch_clip(sampled_action[key] - update, mna, mxa)

    # print("- ", energy_arr[0, 0])

    return out_ac, energy_arr


class SLGDPolicy(OptimizerPolicy):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        super(SLGDPolicy, self)._init_params_to_attrs(params)
        self._max_iters = get_with_default(params, "max_iters", 25)  # max iterations of optimization
        assert self._max_iters > 0

        self._num_action_samples = get_with_default(params, "num_action_samples", 256)
        self._noise_scale = get_with_default(params, "noise_scale", 1.0)  # gradient noise sigma
        self._init_l_lambda = get_with_default(params, "init_l_lambda", 0.1)  # update step size for SLGD
        self._l_lambda_scale_fn = get_with_default(params, "l_lambda_scale_fn", lambda l_lambda, iter: 1.0)
        self._energy_key = get_with_default(params, "energy_key", "energy")

        self._autoregressive = get_with_default(params, "autoregressive", False)
        if self._autoregressive:
            raise NotImplementedError

        # which actions to optimize with SLGD
        self._optim_action_names = get_with_default(params, "optimize_action_names", self._env.env_spec.action_names)
        self._delta_action_clip = get_with_default(params, "delta_action_clip", 0.1)
        lims = self._env.env_spec.limits(self._optim_action_names, as_dict=True)
        self._act_min = lims.leaf_apply(lambda tup: to_torch(tup[0], device="cpu"))
        self._act_max = lims.leaf_apply(lambda tup: to_torch(tup[1], device="cpu"))

        self._timeout = get_with_default(params, "timeout", np.inf)  # when is the policy done

    # @abstract.overrides
    def _init_setup(self):
        super(SLGDPolicy, self)._init_setup()
        self._curr_iter = 0
        self._step_count = 0
        self.cache = d()  # general purpose cache

    # @abstract.overrides
    def warm_start(self, model, observation, goal):
        pass

    # @abstract.overrides
    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...) with action_sequence, scores, order, action (initial_act)
        """

        # observation/goal for first horizon element (not multi-step supporting yet...)
        observation = observation.leaf_apply(lambda x: to_torch(x[:, 0], device=model.device, check=True))
        goal = goal.leaf_apply(lambda x: to_torch(x[:, 0], device=model.device, check=True))
        joint_obs = observation & goal

        self._act_min = self._act_min.leaf_apply(
            lambda arr: arr.to(device=model.device) if model.device != arr.device else arr)
        self._act_max = self._act_max.leaf_apply(
            lambda arr: arr.to(device=model.device) if model.device != arr.device else arr)

        B = joint_obs.get_one().shape[0]

        # uniform in the action space (B, N, ...)
        sampled_action = d()
        for key in self._optim_action_names:
            unif = torch.rand([B, self._num_action_samples] + list(self._act_max[key].shape), device=model.device)  # 0 -> 1
            sampled_action[key] = self._act_min[key] + unif * (self._act_max[key] - self._act_min[key])

        # SLGD loop
        curr_l_lambda = self._init_l_lambda
        with timeit("policy/slgd"):
            for self._curr_iter in range(self._max_iters):
                sampled_action, energies = langevin_step(model, joint_obs, sampled_action, self._act_max, self._act_min,
                                                         delta_action_clip=self._delta_action_clip,
                                                         l_lambda=curr_l_lambda, noise_scale=self._noise_scale,
                                                         energy_key=self._energy_key)
                curr_l_lambda *= self._l_lambda_scale_fn(curr_l_lambda, self._curr_iter)

        # argmax over self._num_action_samples
        best_indices = torch.argmin(energies.reshape((B, self._num_action_samples)), dim=1)

        # (B x ...) TODO sample from categorical w/ probabilities... need to know how energies convert to probs (within model)
        best_action = sampled_action.leaf_apply(lambda ac: unsqueeze_then_gather(ac, best_indices, 1))
        best_energies = unsqueeze_then_gather(energies, best_indices, 1)

        self._step_count += 1
        print(best_energies)

        return d.from_dict({
            f'{self._energy_key}': energies,
            f'best/{self._energy_key}': best_energies,
            f'best/idx': best_indices,
        }) & best_action

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return super(SLGDPolicy, self).is_terminated(model, observation, goal, **kwargs) \
            or self._step_count >= self._timeout

    def reset_policy(self, **kwargs):
        self._step_count = 0
