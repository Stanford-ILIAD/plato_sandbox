import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.policies.random_shooting import OptimizerPolicy
# should not be used in traditional training loop
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import split_dim, to_torch, torch_clip, to_numpy


def performCEM(observation, goal, model, score_fn, pop_size, horizon, act_lows, act_highs, max_iters, num_elites, epsilon, alpha):
    """
    do CEM using a given starting point (obs, goal), resampling "actions" from list act_names

    @param observation: AttrDict (B x MH x ...) starting observation (might include actions)
    @param goal: AttrDict (B x ...) goal (might include
    @param model:
    @param score_fn: CEM score for a given sequence
    @param pop_size:
    @param horizon:
    @param act_lows: AttrDict with names to sample -> low bounds, np broadcastable, non inf
    @param act_highs: AttrDict with names to sample -> high bounds, np broadcastable, non inf
    @param max_iters:
    @param num_elites:
    @param epsilon:
    @param alpha:
    @return:
    """
    # score_fn: takes in model, cache, inputs, goal, -> scores, model_outputs
    act_names = list(act_lows.leaf_keys())
    curr_iter = 0
    cache = AttrDict()  # general purpose cache

    # generate random sequence
    observation = observation.leaf_apply(lambda x: to_torch(x, device=model.device, check=True))
    goal = goal.leaf_apply(lambda x: to_torch(x, device=model.device, check=True))

    # properly scale up to fit popsize
    batch_size = list(observation.leaf_values())[0].shape[0]  # any obs will do
    # assert horizon == list(observation.leaf_values())[0].shape[1]  # any obs will do
    observation.leaf_modify(lambda arr: arr.repeat_interleave(pop_size, dim=0))
    goal.leaf_modify(lambda arr: arr.repeat_interleave(pop_size, dim=0))

    # just happens once (reasonable seed
    action_sequence = AttrDict()
    torch_lows, torch_highs = AttrDict(), AttrDict()
    for i, name in enumerate(act_names):
        action_sequence[name] = np.random.uniform(act_lows[name], act_highs[name],
                                                  size=[batch_size * pop_size, horizon] + list(act_lows[name].shape)).astype(act_lows[name].dtype)
        # action_sequence[name] = action_sequence[name].astype(low[i].dtype)
        torch_lows[name] = to_torch(act_lows[name], device=model.device, check=True)
        torch_highs[name] = to_torch(act_highs[name], device=model.device, check=True)

    action_sequence.leaf_modify(func=lambda arr: to_torch(arr, check=True, device=model.device))

    def resample_and_flatten(key, vs):
        old_acseq = vs[0]
        mean, std = vs[1], vs[2]
        sample = torch.randn_like(old_acseq) * std + mean
        d = AttrDict({key: sample})
        d[key] = torch_clip(d[key], torch_lows[key], torch_highs[key])
        return d[key].view([-1] + list(old_acseq.shape[2:]))

    best_initial_act = None
    ret_dict = None
    for curr_iter in range(max_iters):

        # run the model
        # inputs = action_sequence.copy()
        # inputs.combine(observation)

        scores, model_outputs = score_fn(model, cache, observation, action_sequence, goal)
        # view as (B, Pop, ...)
        action_sequence.leaf_modify(lambda x: split_dim(x, 0, [batch_size, pop_size]))
        # model_outputs.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]) if isinstance(x, torch.Tensor) else x)
        scores = split_dim(scores, 0, [batch_size, pop_size])  # (B*P,) -> (B,P)

        ret_dict = AttrDict()
        ret_dict.combine(model_outputs)
        ret_dict['order'] = torch.argsort(scores, dim=1, descending=True)  # highest to lowest (best to worst)
        ret_dict['scores'] = scores

        best = ret_dict.order[:, :num_elites]

        def get_best(arr):
            act_dim = list(arr.shape[3:])
            best_local = best.unsqueeze(-1)
            for _ in range(len(act_dim)):
                best_local = best_local.unsqueeze(-1)
            best_local = best_local.expand((-1, -1, horizon, *act_dim))
            return torch.gather(arr, 1, best_local)

        best_act_seq = action_sequence.leaf_apply(get_best)
        best_initial_act = best_act_seq.leaf_apply(lambda x: x[:, 0, 0])  # where x is (B, Pop, H ...)
        means = best_act_seq.leaf_apply(lambda x: x.mean(dim=1, keepdim=True))
        stds = best_act_seq.leaf_apply(lambda x: x.std(dim=1, keepdim=True))

        if curr_iter < max_iters - 1:
            # resampling
            action_sequence = AttrDict.leaf_combine_and_apply([action_sequence, means, stds], resample_and_flatten, pass_in_key_to_func=True)

    ret_dict.combine(best_initial_act)
    ret_dict['action_sequence'] = action_sequence
    return ret_dict


class CEMPolicy(OptimizerPolicy):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._pop_size = int(params.pop_size)
        self._action_horizon = int(params.horizon)  # optimize over this many actions
        self._max_iters = params.max_iters  # max iterations of optimization / resampling
        self._num_elites = params.num_elites  # top candidate to target next distribution
        self._epsilon = params.epsilon  # minimum allowed variance
        self._alpha = params.alpha  # momentum per iter

        self._act_dim = self._env_spec.names_to_shapes['action']
        self._action_names = params.get("action_names", self._env_spec.action_names)
        assert len(self._action_names) == 1, "Doesn't yet support multiple action names"
        self._action_names_unoptimized = [a for a in self._env_spec.action_names if a not in self._action_names]
        self._num_actions = len(self._action_names)

        self._score_fn = params.score_fn  # takes in model, optimizer_policy, inputs, goal, -> scores, model_outputs
        self.curr_iter = 0
        self.cache = AttrDict()  # general purpose cache

    # @abstract.overrides
    def _init_setup(self):
        pass

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

        # generate random sequence
        observation = observation.leaf_apply(lambda x: to_torch(x, device=model.device, check=True))
        goal = goal.leaf_apply(lambda x: to_torch(x, device=model.device, check=True))

        batch_size = list(observation.leaf_values())[0].shape[0]  # any obs will do
        assert self._action_horizon == list(observation.leaf_values())[0].shape[1]  # any obs will do
        observation.leaf_modify(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))
        goal.leaf_modify(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))

        # just happens once
        action_sequence = self._env_spec.get_uniform(self._action_names,
                                                     batch_size=batch_size * self._pop_size * self._action_horizon,
                                                     torch_device=model.device)
        action_sequence.leaf_modify(lambda x: split_dim(x, dim=0,
                                                        new_shape=[batch_size * self._pop_size, self._action_horizon]))

        def resample_and_flatten(vs):
            old_acseq = vs[0]
            mean, std = vs[1], vs[2]
            sample = torch.randn_like(old_acseq) * std + mean
            d = AttrDict(action=sample)
            self._env_spec.clip(d, self._action_names)
            return d.action.view([-1] + list(old_acseq.shape[2:]))

        best_initial_act = None
        ret_dict = None
        for self.curr_iter in range(self._max_iters):

            # run the model
            inputs = action_sequence.copy()
            inputs.combine(observation)

            scores, model_outputs = self._score_fn(model, self, inputs, goal)

            # view as (B, Pop, ...)
            action_sequence.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))
            # model_outputs.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]) if isinstance(x, torch.Tensor) else x)
            scores = split_dim(scores, 0, [batch_size, self._pop_size])  # (B*P,) -> (B,P)

            ret_dict = AttrDict()
            ret_dict['order'] = torch.argsort(scores, dim=1, descending=True)  # highest to lowest (best to worst)

            best = ret_dict.order[:, :self._num_elites]
            best = best.unsqueeze(-1)
            for _ in self._act_dim:
                best = best.unsqueeze(-1)
            best = best.expand((-1, -1, self._action_horizon, *self._act_dim))
            best_act_seq = action_sequence.leaf_apply(lambda x: torch.gather(x, 1, best))
            best_initial_act = best_act_seq.leaf_apply(lambda x: x[:, 0, 0])  # where x is (B, Pop, H ...)
            means = best_act_seq.leaf_apply(lambda x: x.mean(dim=1, keepdim=True))
            stds = best_act_seq.leaf_apply(lambda x: x.std(dim=1, keepdim=True))

            if self.curr_iter < self._max_iters - 1:
                # resampling
                action_sequence = AttrDict.leaf_combine_and_apply([action_sequence, means, stds], resample_and_flatten)

        ret_dict.combine(best_initial_act)
        ret_dict['action_sequence'] = action_sequence

        print("Highest Score:", scores[:, 0])

        return ret_dict


if __name__ == '__main__':
    from sbrl.models.model import Model
    # tests CEM functionality with simple point mass example.
    observation = AttrDict(state=np.random.random((64, 1, 3)))
    goal = AttrDict()
    model = Model(AttrDict(ignore_inputs=True), None, None)
    pop_size = 100
    horizon = 1
    act_lows = AttrDict(act=np.zeros((3,)))
    act_high = AttrDict(act=np.ones((3,)))
    max_iters = 10
    num_elites = 10
    epsilon = 0.001
    alpha = 0.25
    def score_fn(model, cache, obs, act_seq, g):
        # neg. l1 mean distance
        return -((act_seq >> "act") - (obs >> "state")).abs().mean([-1, -2]), AttrDict()

    ret_dict = performCEM(observation, goal, model, score_fn, pop_size, horizon, act_lows, act_high, max_iters=max_iters, num_elites=num_elites, epsilon=epsilon, alpha=alpha)

    best_initial = to_numpy(ret_dict >> "act")[:, None]

    logger.debug("Final optimization distance (L1): %f" % np.abs(best_initial - (observation >> "state")).mean())
