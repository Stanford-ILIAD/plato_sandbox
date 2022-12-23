import torch

from sbrl.policies.policy import Policy
# should not be used in traditional training loop
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import split_dim, to_torch


class OptimizerPolicy(Policy):
    pass


class RandomShooting(OptimizerPolicy):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._pop_size = int(params.pop_size)
        self._horizon = int(params.horizon)  # optimize over this many actions
        self._act_dim = self._env_spec.names_to_shapes['action']
        self._action_names = params.get("action_names", self._env_spec.action_names)
        assert len(self._action_names) == 1, "Doesn't yet support multiple action names"
        self._action_names_unoptimized = [a for a in self._env_spec.action_names if a not in self._action_names]
        self._num_actions = len(self._action_names)

        self._score_fn = params.score_fn  # takes in model, optimizer_policy, inputs, goal, -> scores, model_outputs

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
        assert self._horizon == list(observation.leaf_values())[0].shape[1]  # any obs will do
        observation.leaf_modify(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))
        goal.leaf_modify(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))

        # just happens once
        action_sequence = self._env_spec.get_uniform(self._action_names,
                                                     batch_size=batch_size * self._pop_size * self._horizon,
                                                     torch_device=model.device)
        action_sequence.leaf_modify(lambda x: split_dim(x, dim=0,
                                                        new_shape=[batch_size * self._pop_size, self._horizon]))

        # run the model
        inputs = action_sequence.copy()
        inputs.combine(observation)

        scores, model_outputs = self._score_fn(model, self, inputs, goal)

        # view as (B, Pop, ...)
        action_sequence.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))
        model_outputs.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))
        scores = split_dim(scores, 0, [batch_size, self._pop_size])  # (B*P,) -> (B,P)

        ret_dict = AttrDict()
        ret_dict['scores'] = scores
        ret_dict['order'] = torch.argsort(scores, dim=1, descending=True)  # highest to lowest (best to worst)
        best = ret_dict.order[:, :1].unsqueeze(-1)
        for _ in self._act_dim:
            best = best.unsqueeze(-1)
        best = best.expand((-1, -1, self._horizon, *self._act_dim))
        best_act_seq = action_sequence.leaf_apply(lambda x: torch.gather(x, 1, best))
        best_initial_act = best_act_seq.leaf_apply(lambda x: x[:, 0, 0])  # where x is (B, Pop, H ...)
        ret_dict.combine(best_initial_act)
        ret_dict['action_sequence'] = action_sequence

        return ret_dict
