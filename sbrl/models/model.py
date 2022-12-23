"""
This is where all the neural network magic happens!

Models are applied on AttrDicts, while Layers/Networks are applied on tensors (Models call Layers)
"""
import abc
from typing import Callable, List

import torch
from torch import nn

from sbrl.experiments import logger
from sbrl.metrics.metric import Metric
from sbrl.utils.dist_utils import DistributionCap
from sbrl.utils.python_utils import AttrDict, get_with_default, get_required
from sbrl.utils.torch_utils import to_torch


class Model(torch.nn.Module):
    implements_train_step = False

    def __init__(self, params, env_spec, dataset_train):
        super(Model, self).__init__()

        self._env_spec = env_spec
        self._dataset_train = dataset_train
        self.params = params.leaf_copy()

        self._ignore_inputs = get_with_default(params, "ignore_inputs", False)

        if self._ignore_inputs:
            self._device = None
            self._preproc_fn = lambda inps: inps
            self._postproc_fn = lambda inps, outs: inps
        else:
            device = get_required(params, "device")
            if "cuda" not in str(device) or torch.cuda.is_available():
                self._device = torch.device(device)
            else:
                self._device = torch.device("cpu")

            # TODO logger
            logger.debug("Model using device: {}".format(self._device))

            keys = list(params.leaf_keys())
            # first thing called in model forward (inputs) -> new_inputs
            self.set_fn("_preproc_fn", get_with_default(params, "preproc_fn", lambda x: x),
                        Callable[[AttrDict], AttrDict])  # mutation okay

            # updates model outputs (inputs, model_outputs) -> new_model_outputs
            self.set_fn("_postproc_fn", get_with_default(params, "postproc_fn", lambda x, y: y),
                        Callable[[AttrDict, AttrDict], AttrDict])  # mutation okay

            # gets model loss (inputs, outputs, model_outputs) -> loss torch tensor
            self.set_fn("_loss_fn", get_with_default(params, "loss_fn",
                                                     lambda *args, **kwargs: torch.tensor(0., device=self.device)),
                        Callable[[AttrDict, AttrDict, AttrDict], torch.Tensor])

            # special args to provide when calling forward() from the loss fn
            self._loss_forward_kwargs = get_with_default(params, "loss_forward_kwargs", {})

            # TODO this should probably go elsewhere
            # these are things to compute using inputs, outputs, model_outputs
            # if these are specified, loss_fn will not be used.
            self.metrics: List[Metric] = get_with_default(params, "metrics", [])
            self.metric_compute_groups: List[bool] = get_with_default(params, "metric_compute_groups",
                                                                      [False] * len(self.metrics))
            assert len(self.metrics) == len(self.metric_compute_groups)

            self.metric_get_group_fn = get_with_default(params, "metric_get_group_fn", None)
            self.metric_names = [metric.agg_metric_field for metric in self.metrics]
            self.write_only_metric = get_with_default(params, "write_only_metric",
                                                      [False for _ in self.metrics])  # these get computed once per iter
            self.metric_dict = AttrDict.from_kvs(self.metric_names, self.metrics)
            # this metric is used as the loss (default is a sum over all keys)
            self.metric_agg_fn = get_with_default(params, "metric_agg_fn", lambda ms: ms.leaf_reduce(sum))

            # can be overriden, should be all the names you might want to use (especially important to reduce param size)
            self.normalize_inputs = params.get("normalize_inputs", False)
            self.default_normalize_sigma = get_with_default(params, "default_normalize_sigma", 1.)  # number of standard deviations to include in normalization (-normalize_sigma*std, normalize_sigma*std)
            self.do_minmax_norm = params.get("do_minmax_norm", False)  # TODO if true, will normalize to be -1 to 1 with min and max in dataset.

            # to actually use
            self.normalization_inputs = get_with_default(params, "normalization_inputs", [])
            # to compute the norms for (superset)
            self.save_normalization_inputs = get_with_default(params, "save_normalization_inputs",
                                                              self.normalization_inputs)
            # these names will store the MAX std across the dimension, not one per axis.
            self.max_std_normalization_inputs = get_with_default(params, "max_std_normalization_inputs", [])
            assert set(self.max_std_normalization_inputs).issubset(self.save_normalization_inputs), [self.max_std_normalization_inputs, self.save_normalization_inputs]
            assert set(self.normalization_inputs).issubset(self.save_normalization_inputs), \
                f"{self.normalization_inputs} must be a subset of save names: {self.save_normalization_inputs}"

            if len(self.max_std_normalization_inputs) > 0:
                logger.debug(f"Will use maximum std for these keys: {self.max_std_normalization_inputs}")

            self._init_params_to_attrs(params)
            self._init_setup()

            if env_spec is not None:
                # means 'n stds
                initial_means_t = env_spec.get_zeros(self.save_normalization_inputs, 1, torch_device=self._device)
                initial_stds_t = env_spec.get_ones(self.save_normalization_inputs, 1, torch_device=self._device)
                initial_means_t = initial_means_t.leaf_apply(lambda arr: nn.Parameter(arr[0], requires_grad=False))
                initial_stds_t = initial_stds_t.leaf_apply(lambda arr: nn.Parameter(arr[0], requires_grad=False))

                # some simple statistics that may or may not be filled in by trainer. add more of these in init_setup / load_statistics
                self.torch_means = nn.ParameterDict(initial_means_t.as_dict())
                self.torch_stds = nn.ParameterDict(initial_stds_t.as_dict())

        # used by get_default_mem_policy_forward_fn
        # (model, out, obs, goal, memory, **kwargs) -> new_out
        self.online_postproc_fn = get_with_default(params, "online_postproc_fn", lambda model, out, *args, **kwargs: out)

    # proper function typing
    def set_fn(self, name, func, ftype):
        if func is None:
            raise NotImplementedError

        def _internal_setter(fn: ftype):
            self.__setattr__(name, fn)

        _internal_setter(func)

    def has_fn(self, name):
        return hasattr(self, name) and isinstance(getattr(self, name), Callable)

    def normalize_by_statistics(self, inputs: AttrDict, names, shared_dtype=None, check_finite=True, inverse=False,
                                shift_mean=True, normalize_sigma=None):
        # TODO case where inputs come from CAttrDict
        normalize_sigma = self.default_normalize_sigma if normalize_sigma is None else normalize_sigma
        assert inputs.has_leaf_keys(names), list(set(names).difference(inputs.list_leaf_keys()))
        inputs = inputs.leaf_copy()
        for name in names:
            assert name in self.torch_means.keys(), [name, names, self.torch_means.keys()]
            # print(name)
            if shared_dtype is not None:
                inputs[name] = inputs[name].to(dtype=shared_dtype)  # TODO
            # normalize_sig = how many sigmas to normalize (2.5 = 95% of data will fall between -1 and 1)
            std_typed = normalize_sigma * self.torch_stds[name][None].to(dtype=inputs[name].dtype)
            if inverse:
                # unnormalize
                normalized = inputs[name] * std_typed
                if shift_mean:
                    normalized += self.torch_means[name][None].to(dtype=inputs[name].dtype)
            else:
                # normalize
                if shift_mean:
                    inputs[name] = inputs[name] - self.torch_means[name][None].to(dtype=inputs[name].dtype)
                normalized = torch.div(inputs[name], std_typed)
            if check_finite:
                inputs[name] = torch.where(normalized.isfinite(), normalized,
                                           inputs[name])  # do not keep NaN std's, this is bad
            else:
                inputs[name] = normalized
        return inputs

    def load_statistics(self, dd=None):
        logger.debug("Loading statistics for model's dataset")
        if self._ignore_inputs or len(self.save_normalization_inputs) == 0:
            logger.warn("No stats loaded for model!")
            return dd

        if len(self._dataset_train) == 0:
            logger.warn("No data in dataset but tried to load statistics.")
            return dd

        if dd is None:
            dd = AttrDict()
            missing = self.save_normalization_inputs
        else:
            missing = list(set(self.save_normalization_inputs).difference((dd >> 'mean').list_leaf_keys()))

        dd = dd & self._dataset_train.get_statistics(missing).leaf_apply(
            lambda arr: to_torch(arr, device=self._device))

        assert "mean" in dd.keys() and "std" in dd.keys()
        for name in self.save_normalization_inputs:
            self.torch_means[name].data.copy_(dd.mean >> name)
            assert torch.all(dd.std[name] >= 0)
            if torch.any(dd.std[name] == 0):
                logger.warn("Zero mean for key %s, shape %s. This could cause issues with normalization" % (
                name, dd.std[name].shape))
            if name in self.max_std_normalization_inputs:
                print(name, dd.std[name].shape)
                self.torch_stds[name].data.copy_(torch.max(dd.std[name]))  # the MAXIMUM std for this name will be used
            else:
                self.torch_stds[name].data.copy_(dd.std[name])  # there might be zero entries

        for key in self.save_normalization_inputs:
            logger.debug("--> [%s] means: %s, Stds: %s" % (key, self.torch_means[key].data, self.torch_stds[key].data))

        return dd

    def wrap_parallel(self, device_ids=None):
        # make sure this is called after initializing all models.
        assert "cuda" in str(self._device), "only works for multi-gpu"
        def helper(m, required=False, key=None):
            wrap = False
            found = False

            if isinstance(m, Model):
                m.wrap_parallel(device_ids=device_ids)
                found = True
            elif isinstance(m, nn.DataParallel):
                found = True
            elif isinstance(m, nn.Module):
                # any_grads = any(p.requires_grad for p in m.parameters())
                # if isinstance(m, nn.ParameterDict) and any_grads:
                if isinstance(m, nn.ParameterDict):
                    for k in list(m.keys()):
                        if isinstance(m[k], nn.Module):
                            m[k] = helper(m[k], key=k, required=True)  # requires a return val
                elif isinstance(m, nn.ParameterList):
                    for i in range(len(m)):
                        if isinstance(m[i], nn.Module):
                            m[i] = helper(m[i], key=i, required=True)
                else:
                    wrap = True

                found = True

            if wrap:
                # m is a Module
                if isinstance(m, nn.Sequential):
                    # special check for networks that end in distributions.
                    children = list(m.children())
                    if isinstance(children[-1], DistributionCap):
                        wrapped_initial = nn.DataParallel(nn.Sequential(*children[:-1]), device_ids=device_ids)  # base case
                        return nn.Sequential(wrapped_initial, children[-1])  # data parallel stacked below

                return nn.DataParallel(m, device_ids=device_ids)  # base case
            elif found:
                return m  # if we don't need to wrap, just return the same thing.
            elif required:
                raise Exception(str(m))  # required means we must be able to find the parameter.

            return None

        # objects = {name: getattr(self, name) for name in dir(self)}
        # for attr in dir(self):
        #     if isinstance(getattr(type(self), attr, None), property):
        #         print(attr, "is a property")

        for name in list(dir(self)):
            sm = getattr(self, name)
            if not isinstance(getattr(type(self), name, None), property):  # skip properties
                out = helper(sm, key=name)
                if out is not None:
                    # print(type(self), name, type(out))
                    setattr(self, name, out)

    @property
    def device(self):
        return self._device

    @property
    def env_spec(self):
        return self._env_spec

    @abc.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abc.abstractmethod
    def _init_setup(self):
        pass

    @abc.abstractmethod
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        return inputs

    # override this if you want your own loss
    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, ret_dict=False, meta=AttrDict(), **kwargs):
        """

        :param inputs: (AttrDict)  (B x H x ...)
        :param outputs: (AttrDict)  (B x H x ...)
        :param i: (int) current step
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)

        :return loss: (torch.Tensor)  (1,)
        """
        model_outputs = self.forward(inputs, training=training, **self._loss_forward_kwargs, **kwargs)
        if len(self.metrics) == 0:
            assert hasattr(self, "_loss_fn"), "Model needs a user defined loss function to call loss()"
            loss = self._loss_fn(self, model_outputs, inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix,
                                 ret_dict=ret_dict, meta=meta, **kwargs)
        else:
            all_metrics = []
            all_metric_keys = []
            for mi, m in enumerate(self.metrics):
                if not self.write_only_metric[mi] or writer is not None:
                    if self.metric_compute_groups[mi]:
                        # get groups for each entry (B,) and num groups to compute
                        g, n_groups = self.metric_get_group_fn(inputs, outputs, model_outputs, mi)
                        results = m.compute_group_wise(inputs, outputs, model_outputs, g, n_groups)
                    else:
                        results = m.compute(inputs, outputs, model_outputs)
                    if writer:
                        for k, l in results.leaf_items():
                            writer.add_scalar(writer_prefix + k, l.item(), i)
                    all_metrics.append(results)
                    all_metric_keys.extend(results.leaf_keys())
            assert len(set(all_metric_keys)) == len(all_metric_keys), f"Non unique keys returned: {all_metric_keys}"

            all_metric_dict = AttrDict()
            for r in all_metrics:
                all_metric_dict.combine(r)
            loss = self.metric_agg_fn(all_metric_dict)

        if not ret_dict:
            loss = loss.mean()
            if writer:
                writer.add_scalar(writer_prefix + "loss", loss.item(), i)

        return loss

    def train_step(self, inputs, outputs, i=0, writer=None, writer_prefix="", ret_dict=False, optimizer=None, **kwargs):
        # if you override this, make sure to set implements_train_step=True for Trainer to know
        raise NotImplementedError

    def restore_from_checkpoint(self, checkpoint, strict=False):
        self.load_state_dict(checkpoint['model'], strict=strict)

    def restore_from_file(self, fname, strict=False):
        self.restore_from_checkpoint(torch.load(fname, map_location=self.device), strict=strict)
