"""
Stochastic Langevin Gradient Descent -- based on https://github.com/google-research/ibc/blob/master/ibc/agents/
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from sbrl.policies.random_shooting import OptimizerPolicy
from sbrl.utils.python_utils import AttrDict as d, get_with_default, timeit
from sbrl.utils.torch_utils import broadcast_dims, torch_clip, to_torch, \
    unsqueeze_then_gather, concatenate, combine_after_dim


def get_default_compute_logits_fn(nll_key="energy"):
    def compute_logits_fn(network, observation: d, sampled_action: d, expand_obs=True, copy=False, **kwargs):
        B, N = sampled_action.get_one().shape[:2]
        if expand_obs:
            observation = observation.leaf_apply(lambda arr: broadcast_dims(arr[:, None], [1], [N]))

        if copy:
            sampled_action = sampled_action.leaf_apply(lambda arr: arr.clone())

        with torch.no_grad():
            inputs = observation & sampled_action
            outs = network(inputs, **kwargs)

        return -(outs >> nll_key)

    return compute_logits_fn


def fit_and_resampling_step(network, compute_logits_fn, observation: d, sampled_action: d, max_action: d,
                            min_action: d,
                            temperature=1.0, noise_scale=0.33, add_noise=True,
                            expand_obs=True, copy=False, prefix='', **kwargs):
    """
    Evaluates sampled action, replaces with new actions, adds noise if noise=True

    :param network: must return a negative_log_prob (TODO allow for others)
    :param observation: (B, ...) or (B, N, ...) if expand_obs=False, torch.Tensor
    :param sampled_action: (B, N, ...), where N is the number of samples, torch.Tensor
    :param l_lambda: langevin update size
    :param noise_scale: noise sigma to add to gradient (TODO allow tensors here)
    :param expand_obs: if True, will expand obs to match ac shape in dim 1
    :param nll_key: model output name
    :param copy: if True, will copy the action tensors first, before gradients
    :param kwargs: extra model arguments
    :return:
    """

    B, N = sampled_action.get_one().shape[:2]

    # resample by probabilites
    with torch.no_grad():
        with timeit(f"{prefix}compute_logits"):
            logits = compute_logits_fn(network, observation, sampled_action, expand_obs=expand_obs, copy=copy, **kwargs)
            logits = logits.view(-1, N) / temperature

        with timeit(f"{prefix}categorical_sample"):
            out = Categorical(logits=logits).sample((N,))  # B x N

        with timeit(f"{prefix}bin_count"):
            # (B x N) + (B x 1) offsets
            step_idxs = out + torch.arange(0, B * N, step=N, dtype=out.dtype, device=out.device)[:, None]
            bin_count = torch.bincount(step_idxs.view(-1), minlength=B * N)  # (B*N, ) how many repeats per element
            flat_bin = bin_count.view(-1)

        with timeit(f"{prefix}repeat_interleave"):
            # actions are (B x N x ...), repeat along dim=1, with repeats shaped (B, N)
            # new shape will be (B x N x ...) for actions
            sampled_action = sampled_action.leaf_apply(
                lambda arr: torch.repeat_interleave(arr.view(B * N, -1), flat_bin, dim=0).view(B, N, *arr.shape[2:])
            )

    if add_noise:
        with timeit(f"{prefix}add_noise"):
            out_ac = d()
            for key, update in sampled_action.leaf_items():
                mna = min_action[key]
                mxa = max_action[key]

                # add noise
                ac = sampled_action[key]
                ac = ac + torch.randn_like(ac) * noise_scale
                out_ac[key] = torch_clip(ac, mna, mxa)
    else:
        # no need to clip, previous actions will be assumed to already be in range
        out_ac = sampled_action

    # logits for previous samples (which were used to generate out_ac), and also returns new samples (out_ac)
    return out_ac, logits


def autoregressive_fit_and_resampling_step(network, compute_logits_fn, observation: d, sampled_action: d, max_action: d,
                                           min_action: d, action_names, num_action_dim,
                                           temperature=1.0, noise_scale=0.33, add_noise=True,
                                           expand_obs=True, copy=False, prefix='', **kwargs):
    """
    Evaluates sampled action, replaces with new actions, adds noise if noise=True

    :param network: must return a negative_log_prob (TODO allow for others)
    :param observation: (B, ...) or (B, N, ...) if expand_obs=False, torch.Tensor
    :param sampled_action: (B, N, ...), where N is the number of samples, torch.Tensor
    :param l_lambda: langevin update size
    :param noise_scale: noise sigma to add to gradient (TODO allow tensors here)
    :param expand_obs: if True, will expand obs to match ac shape in dim 1
    :param nll_key: model output name
    :param copy: if True, will copy the action tensors first, before gradients
    :param kwargs: extra model arguments
    :return:
    """

    B, N = sampled_action.get_one().shape[:2]

    flat_action = concatenate(sampled_action.leaf_apply(lambda arr: combine_after_dim(arr, 2)), action_names, dim=2)
    flat_action_min = concatenate(min_action.leaf_apply(lambda arr: combine_after_dim(arr, 0)), action_names, dim=0)
    flat_action_max = concatenate(max_action.leaf_apply(lambda arr: combine_after_dim(arr, 0)), action_names, dim=0)

    sampled_action = network.env_spec.parse_view_from_concatenated_flat(flat_action, action_names)

    # resample by probabilites
    with torch.no_grad():
        for n in range(num_action_dim):
            # flat view, all
            logits = compute_logits_fn(network, observation, sampled_action, expand_obs=expand_obs, copy=copy, **kwargs)
            assert logits.shape[-1] == num_action_dim
            logits = logits[..., n]  # n'th prediction corresponds to p(x_{<=n})
            logits = logits.view(-1, N) / temperature

            out = Categorical(logits=logits).sample((N,))  # B x N

            # (B x N) + (B x 1) offsets
            step_idxs = out + torch.arange(0, B * N, step=N, dtype=out.dtype, device=out.device)[:, None]
            bin_count = torch.bincount(step_idxs.view(-1), minlength=B * N)  # (B*N, ) how many repeats per element
            flat_bin = bin_count.view(-1)

            # actions (B x N x ...), repeat 1 action dim (n) along sampling dim (1), with repeats shaped (B, N)
            # new shape will be (B x N, n+1) for resampling ac dim n+1
            flat_action[..., :n+1] = torch.repeat_interleave(flat_action[..., :n+1].view(B * N, n+1), flat_bin, dim=0).view(B, N, n+1)

            if add_noise:
                # only to dimension n
                ac = flat_action[..., n]
                ac = ac + torch.randn_like(ac) * noise_scale
                flat_action[..., n] = torch.clip(ac, flat_action_min[n], flat_action_max[n])

            # import ipdb; ipdb.set_trace()

            # this shouldn't be necessary, since all modifications are in place...
            sampled_action = network.env_spec.parse_view_from_concatenated_flat(flat_action, action_names)

    # logits for last autoregressive layer samples (which were used to generate out_ac), and also returns new samples (out_ac)
    return sampled_action, logits


class MultinomialCEMPolicy(OptimizerPolicy):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        super(MultinomialCEMPolicy, self)._init_params_to_attrs(params)
        self._max_iters = get_with_default(params, "max_iters", 3)  # max iterations of optimization
        assert self._max_iters > 0

        self._num_action_samples = get_with_default(params, "num_action_samples", 1024)
        self._noise_scale = get_with_default(params, "noise_scale", 0.33)  # perturbation N(0, this), init
        self._shrink_noise_scale = get_with_default(params, "shrink_noise_scale", 0.5)  # scales noise down every iter
        self._temperature = get_with_default(params, "temperature", 1.0)  # temperature for computing probs

        # runs autoregressive DFO
        self._autoregressive = get_with_default(params, "autoregressive", False)

        # which actions to optimize with SLGD
        self._optim_action_names = get_with_default(params, "optimize_action_names", self._env_spec.action_names)

        self._energy_key = get_with_default(params, "energy_key", "energy")
        # (network, observation: d, sampled_action: d, expand_obs=True, copy=False, **kwargs) -> Tensor
        self._compute_logits_fn = get_with_default(params, "compute_logits_fn",
                                                   get_default_compute_logits_fn(nll_key=self._energy_key))

        lims = self._env_spec.limits(self._optim_action_names, as_dict=True)
        self._act_min = lims.leaf_apply(lambda tup: to_torch(tup[0], device="cpu"))
        self._act_max = lims.leaf_apply(lambda tup: to_torch(tup[1], device="cpu"))

        self._act_dim = self._env_spec.dim(self._optim_action_names)

        self._timeout = get_with_default(params, "timeout", np.inf)  # when is the policy done

    # @abstract.overrides
    def _init_setup(self):
        super(MultinomialCEMPolicy, self)._init_setup()
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
            unif = torch.rand([B, self._num_action_samples] + list(self._act_max[key].shape),
                              device=model.device)  # 0 -> 1
            sampled_action[key] = self._act_min[key] + unif * (self._act_max[key] - self._act_min[key])

        # Fit and resampling loop
        curr_noise = self._noise_scale
        with timeit("policy/mcem"):
            for self._curr_iter in range(self._max_iters):
                if self._autoregressive:
                    sampled_action, _ = autoregressive_fit_and_resampling_step(model, self._compute_logits_fn, joint_obs,
                                                                            sampled_action, self._act_max,
                                                                            self._act_min,
                                                                            self._optim_action_names, self._act_dim,
                                                                            noise_scale=curr_noise,
                                                                            temperature=self._temperature,
                                                                            add_noise=self._curr_iter < self._max_iters - 1,
                                                                            prefix='policy/mcem/')
                else:
                    # add noise to sampled actions for all except the last iter
                    sampled_action, _ = fit_and_resampling_step(model, self._compute_logits_fn, joint_obs,
                                                                sampled_action, self._act_max, self._act_min,
                                                                noise_scale=curr_noise,
                                                                temperature=self._temperature,
                                                                add_noise=self._curr_iter < self._max_iters - 1,
                                                                prefix='policy/mcem/')
                curr_noise *= self._shrink_noise_scale

        # argmax over self._num_action_samples
        logits = self._compute_logits_fn(model, observation, sampled_action).view((B, self._num_action_samples, -1))[..., -1]
        probs = F.softmax(logits / self._temperature, dim=-1)
        best_indices = torch.argmax(logits, dim=1)

        # (B x ...) TODO sample from categorical w/ probabilities... need to know how energies convert to probs (within model)
        best_action = sampled_action.leaf_apply(lambda ac: unsqueeze_then_gather(ac, best_indices, 1))
        best_probs = unsqueeze_then_gather(probs, best_indices, 1)

        self._step_count += 1

        # if "action" in observation.leaf_keys():  # TODO remove
        #     sampled_action = observation > ['action']
        #     true_logits = self._compute_logits_fn(model, observation, sampled_action, expand_obs=False)
        #
        #     print(
        #         f"[{self._step_count}] True logit: {true_logits.mean()}, Best logit: {unsqueeze_then_gather(logits, best_indices, 1).mean()}")
        #     print(f"[{self._step_count}] True action: {sampled_action.action[0]}, Best action: {best_action.action[0]}")
        #
        # print(unsqueeze_then_gather(logits, best_indices, 1))

        return d.from_dict({
            f'prob': probs,
            f'logit': logits,
            f'best/prob': best_probs,
            f'best/idx': best_indices,
        }) & best_action

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return super(MultinomialCEMPolicy, self).is_terminated(model, observation, goal, **kwargs) \
               or self._step_count >= self._timeout

    def reset_policy(self, **kwargs):
        self._step_count = 0
