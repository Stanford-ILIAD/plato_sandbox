"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
import os
from typing import List

import torch

from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.param_utils import SequentialParams, LayerParams
from sbrl.utils.python_utils import AttrDict, get_required, timeit
from sbrl.utils.torch_utils import concatenate


class BasicModel(Model):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self.inputs = get_required(params, "model_inputs")
        self.output = str(get_required(params, "model_output"))
        self.net = (params >> "network").to_module_list(as_sequential=True).to(self.device)

        self.concat_dim = int(params.get("concat_dim", -1))
        self.concat_dtype = params.get("concat_dtype", torch.float32)

    # @abstract.overrides
    def _init_setup(self):
        pass

    # @abstract.overrides
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    @staticmethod
    def concat_forward(model: Model, net: torch.nn.Module, inputs, input_names: List[str], output_name: str,
                       concat_dim: int, concat_dtype,
                       training=False, preproc_fn=None, postproc_fn=None, **kwargs):
        if not isinstance(inputs, AttrDict):
            if isinstance(inputs, torch.Tensor):
                inputs = [inputs]
            assert len(inputs) == len(model.inputs), [model.inputs, len(inputs)]
            inputs = AttrDict.from_dict({k: v for k, v in zip(model.inputs, inputs)})
        else:
            inputs = inputs.leaf_copy()

        if model.normalize_inputs:
            inputs = model.normalize_by_statistics(inputs, model.normalization_inputs, shared_dtype=concat_dtype)

        with timeit("basic_model/preproc"):
            if preproc_fn:
                inputs = preproc_fn(inputs)

        with timeit("basic_model/cat"):
            obs = concatenate(inputs.node_leaf_filter_keys_required(input_names)
                              .leaf_apply(lambda arr: arr.to(dtype=concat_dtype)),
                              input_names, dim=concat_dim)
        # print(self.inputs, self.output)

        # assert not torch.any(torch.isnan(obs)), inputs.leaf_apply(lambda arr: torch.isnan(arr).any())

        with timeit("basic_model/forward"):
            out = net(obs)
        out = AttrDict({output_name: out})

        return postproc_fn(inputs, out) if postproc_fn else out

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        return BasicModel.concat_forward(self, self.net, inputs, self.inputs, self.output, self.concat_dim, self.concat_dtype, training=training,
                                         preproc_fn=self._preproc_fn if preproc else None,
                                         postproc_fn=self._postproc_fn if postproc else None, **kwargs)

    @staticmethod
    def get_default_mem_policy_forward_fn(*args, add_goals_in_hor=False, **kwargs):

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: BasicModel, obs: AttrDict, goal: AttrDict, memory: AttrDict,
                                        root_model: Model=None, **inner_kwargs):
            obs = obs.leaf_copy()
            if memory.is_empty():
                memory.count = 0

            if not add_goals_in_hor and not goal.is_empty():
               obs.goal_states = goal

            memory.count += 1

            # normal policy w/ fixed plan, we use prior, doesn't really matter here tho since run_plan=False
            base_model = (model if root_model is None else root_model)
            out = base_model.forward(obs, **inner_kwargs)
            return base_model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[BasicModel]")
        for p in self.parameters():
            print_fn(prefix + "[BasicModel] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    md = BasicModel(AttrDict(model_inputs=["ins"], model_output="out", device="cpu", normalization_inputs=[],
                             network=SequentialParams([
                                 LayerParams("linear", in_features=5, out_features=10),
                                 LayerParams("relu"),
                                 LayerParams("linear", in_features=10, out_features=10),
                                 LayerParams("gaussian_dist_cap", params=AttrDict(use_log_sig=False))
                             ]), postproc_fn=lambda inps, outs: outs.out),
                    ParamEnvSpec(AttrDict()), None)

    writer = SummaryWriter(os.path.expanduser("~/.test"))

    print("Adding graph...")
    writer.add_graph(md, input_to_model=torch.zeros((10, 5)), verbose=True)

    print("DONE.")
