"""
Structure is customizeable, but basically:

*if rnn_before_network = True*

inputs ---> recurrent_network     ---->   network  ---> output
        |                           |
        ->         parallel_model --                 -> non_recurrent_output

*else*

inputs --->  network  --->  recurrent_network   ---->  output
        |                                       |
         --->      parallel_model           ----  ->  non_recurrent_output

"""
from collections import Callable

import torch

from sbrl.experiments import logger
from sbrl.models.basic_model import BasicModel
from sbrl.models.model import Model
from sbrl.utils.param_utils import LayerParams
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit
from sbrl.utils.script_utils import is_next_cycle
from sbrl.utils.torch_utils import concatenate


# 1 input, split into N outputs via a split function
class RnnModel(BasicModel):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        # initializes inputs, outputs, net
        super(RnnModel, self)._init_params_to_attrs(params)

        self.recurrent_net = params.recurrent_network.to_module_list().to(self.device)

        self.rnn_before_net = get_with_default(params, "rnn_before_net", True)
        # an extra network to take static inputs
        self.parallel_model = get_with_default(params, "parallel_model", None)
        if self.parallel_model is not None:
            self.parallel_model = (self.parallel_model >> "cls")(self.parallel_model >> "params", self.env_spec, self._dataset_train)
            # default merge (after rnn) does not include outputs in the final net's inputs.
            self.merge_parallel_outputs_fn = get_with_default(params, "merge_parallel_outputs_fn", lambda new_in, pout: new_in)
            assert isinstance(self.merge_parallel_outputs_fn, Callable)
        self.if_after_net_use_hidden = get_with_default(params, "net_use_hidden", False)
        self.tuple_hidden = get_with_default(params, "tuple_hidden", False)

        self.rnn_output_name = get_with_default(params, "rnn_output_name", self.output + "_rnn_output")
        self.hidden_name = get_with_default(params, "hidden_name", "hidden")

        # this is the masking name to look for in inputs for posterior / policy.
        #   if not None, this means the rnn_model will receive a packed sequence.
        self.mask_name = params << "mask_name"
        if self.mask_name is not None:
            logger.info(f"RnnModel using mask: {self.mask_name}")

    # @abstract.overrides
    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def rnn_forward(self, obs, h0, pad_mask=None):
        """

        :param obs: tensor, (B, H, ...)
        :param h0: tensor
        :param pad_mask: tensor, (B, H)
        :return:
        """
        if pad_mask is not None:
            with timeit("rnn/packing_time"):
                assert list(pad_mask.shape) == list(obs.shape[:2]), [obs.shape[:2], pad_mask.shape]
                ep_lengths = torch.count_nonzero(~pad_mask, 1).cpu()
                obs = torch.nn.utils.rnn.pack_padded_sequence(obs, ep_lengths, batch_first=True, enforce_sorted=False)

        out, h_n = self.recurrent_net(obs, h0)

        if pad_mask is not None:
            with timeit("rnn/padding_time"):
                # bring it back to padded form.
                out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out, h_n

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, rnn_hidden_init=None, do_normalize=None, parallel_kwargs=None, **kwargs):
        """
        Runs self.net and self.rnn_net, not necessarily in that order

        :param inputs: (AttrDict)  (B x SEQ LEN x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn
        :param rnn_hidden_init: (B x layers x hidden) the initial rnn state, must be same shape as the output of self.net(inputs) will be
        :param do_normalize: (bool) forces normalizing, if not None.
        :return model_outputs: (AttrDict)  [rnn_output_name]: (B x SEQ_LEN x num_directions*hidden_size)
                                           [hidden_name]: (num_layers*num_directions x B x hidden_size)
                               for batch_first=True, for example
        """

        outputs = AttrDict()

        inputs = inputs.leaf_copy()

        # overriding arg
        do_normalize = do_normalize if do_normalize is not None else self.normalize_inputs
        if do_normalize:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        # parallel model to run on the inputs in parallel to recurrent network.
        parallel_outputs = AttrDict()
        if self.parallel_model is not None:
            parallel_kwargs = {} if parallel_kwargs is None else parallel_kwargs
            parallel_outputs = self.parallel_model(inputs, **parallel_kwargs, **kwargs)
            outputs.combine(parallel_outputs)

        if preproc:
            inputs = self._preproc_fn(inputs)

        # likely, B x H x ...
        # a sequence of inputs.
        obs = concatenate(inputs, self.inputs, dim=self.concat_dim)
        # print(self.inputs, obs.shape)
        if not self.rnn_before_net:
            obs = self.net(obs)
            outputs[self.output] = obs

        h_0 = None
        if rnn_hidden_init is not None:
            if self.tuple_hidden:
                h_0 = tuple(x.transpose(0, 1).contiguous() for x in rnn_hidden_init)
            else:
                h_0 = torch.transpose(rnn_hidden_init, 0, 1).contiguous()  # permute to (nl*nd x B x hidden)

        # RUN RNN with optional mask
        rnn_out, h_n = self.rnn_forward(obs, h_0, pad_mask=(inputs << self.mask_name) if self.mask_name is not None else None)

        outputs[self.rnn_output_name] = rnn_out
        outputs[self.hidden_name] = tuple(x.transpose(0,1) for x in h_n) if self.tuple_hidden else torch.transpose(h_n, 0, 1)

        if self.rnn_before_net:
            if self.if_after_net_use_hidden:
                new_ins = h_n  # (B x nl*nd x hidden)
            else:
                new_ins = rnn_out
            if self.parallel_model is not None:
                new_ins = self.merge_parallel_outputs_fn(new_ins, parallel_outputs)
            outputs[self.output] = self.net(new_ins)
            # print(rnn_out.shape)

        return self._postproc_fn(inputs, outputs) if postproc else outputs

    def print_parameters(self, prefix="", print_fn=logger.debug):

        print_fn(prefix + "[RNN]")
        for k, p in self.torch_means.items():
            print_fn(prefix + "[RNN means] %s: <%s> (requires_grad = %s" % (k, list(p.shape), p.requires_grad))
        for k, p in self.torch_stds.items():
            print_fn(prefix + "[RNN stds]  %s: <%s> (requires_grad = %s" % (k, list(p.shape), p.requires_grad))
        if self.parallel_model is not None:
            self.parallel_model.print_parameters(prefix=prefix + "[RNN Parallel]", print_fn=print_fn)
        for p in self.net.parameters():
            print_fn(prefix + "[RNN non-recurrent] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))
        for p in self.recurrent_net.parameters():
            print_fn(prefix + "[RNN recurrent] param <%s> (requires_grad = %s)" % (list(p.shape), p.requires_grad))

    @staticmethod
    def get_default_mem_policy_forward_fn(*args, add_goals_in_hor=False, flush_horizon=0, **kwargs):
        assert isinstance(flush_horizon, int) and flush_horizon >= 0, flush_horizon
        if flush_horizon == 0:
            logger.warn("Note: RNN will never flush the hidden state online (flush_horizon=0)! This can cause issues "
                        "online.")

        def submodel_apply(model, pmem, fn):
            if not isinstance(model, RnnModel):
                return
            fn(model, pmem)
            if isinstance(model.parallel_model, RnnModel):
                submodel_apply(model.parallel_model, pmem.parallel_model, fn)

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: RnnModel, obs: AttrDict, goal: AttrDict, memory: AttrDict,
                                        root_model: Model=None, **inner_kwargs):
            obs = obs.leaf_copy()
            if memory.is_empty():
                memory.count = 0  # total steps
                memory.flush_count = 0  # just for flushing the rnn state

                # traverse down all rnn models in parallel tree
                submodel_apply(model, memory, lambda mod, mem: setattr(mem, 'policy_rnn_h0', None))

            if is_next_cycle(memory.flush_count, flush_horizon):
                submodel_apply(model, memory, lambda mod, mem: setattr(mem, 'policy_rnn_h0', None))

            if not add_goals_in_hor and not goal.is_empty():
               obs.goal_states = goal

            memory.count += 1
            memory.flush_count += 1

            pkwargs = {}

            # recursively fill with rnn hidden states
            pk = pkwargs
            mem = memory
            while mem.has_leaf_key('parallel_model/policy_rnn_h0'):
                pk['rnn_hidden_init'] = mem >> 'parallel_model/policy_rnn_h0'
                pk['parallel_kwargs'] = {}
                mem = mem.parallel_model
                pk = pk['parallel_kwargs']

            # normal policy w/ fixed plan, we use prior, doesn't really matter here tho since run_plan=False
            base_model = (model if root_model is None else root_model)
            out = base_model.forward(obs, rnn_hidden_init=memory >> "policy_rnn_h0", parallel_kwargs=pkwargs, **inner_kwargs)
            # NEXT OUTPUT

            submodel_apply(model, memory, lambda mod, mem: setattr(mem, 'policy_rnn_h0', out >> mod.hidden_name))

            # default online postproc defined in BasicModel (parent class)
            return base_model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn


if __name__ == '__main__':
    #TODO
    DEVICE = "cpu"
    state_dim = 4

    # (H-1, VIS + PROPRIO + PLAN) -> (H-1, ACTION)
    policy_params = AttrDict(
        cls=RnnModel,
        params=AttrDict(
            device=DEVICE,
            recurrent_net=LayerParams("relu"),  # for testing
            net_use_hidden=False,
            model_inputs=['state'],
            model_output='action_dist',
            rnn_output_name='rnn_output',
            hidden_name='rnn_hidden',
            forward_fn=lambda model, inps: AttrDict(
                action_dist=torch.distributions.Normal(inps.state, scale=1e-11)
            )
        )
    )
