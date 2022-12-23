from typing import List, Union

import torch

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.param_utils import LayerParams
from sbrl.utils.python_utils import AttrDict as d, get_or_instantiate_cls


class GroupedModel(Model):
    """
    Abstract base class for an grouping of models. NOTE: this is basically an aggregation of models in some standard way.
        This does NOT implement training schedules or individual optimizer groups

    GroupedModel consist of...
    - models: (Model) these are all the models we run forward and/or backward on
    - TODO metrics: (Metric) these are all the metrics we might need to compute for this algorithm
        - loss_metrics: specific to computing loss
        - log_metrics: things that get logged
    - parameter group access for combinations of model parameters specific to algorithm

    """

    required_models = []

    def _init_params_to_attrs(self, params: d):
        # all models passed in as AttrDict
        self._models = params >> "models"
        assert isinstance(self._models, d), "GroupedModel requires all models to be passed in as AttrDict"

        self._sorted_model_order = sorted(self._models.keys())
        logger.debug("Model names: " + str(self._sorted_model_order))

        # instantiate all
        for model_name, m in self._models.items():
            if not isinstance(m, Model):
                if isinstance(m, LayerParams):
                    logger.debug(f"Instantiating {model_name} from {m.__class__}")
                    self._models[model_name] = m.to_module_list(as_sequential=True)
                else:
                    assert isinstance(self._models[model_name],
                                      d), f"All models must be instantiable, but \'{model_name}\' was not."
                    logger.debug(
                        f"Instantiating \"{model_name}\" with params {m.pprint(ret_string=True)}")
                    self._models[model_name] = get_or_instantiate_cls(self._models, model_name, Model,
                                                                      constructor=lambda cls, prms:
                                                                      cls(prms, self.env_spec, self._dataset_train)
                                                                      )
            # assign them locally for parameter linking (prefixed by "_")
            setattr(self, "_" + model_name, self._models[model_name])

            assert not isinstance(m, GroupedModel), "GroupedModel recursive set not allowed"

        if len(self.required_models) > 0:
            assert set(self.required_models).issubset(self._sorted_model_order), \
                f"{self.__class__} is missing models {set(self.required_models).difference(self._sorted_model_order)}"

    def load_statistics(self, dd=None):
        dd = super(GroupedModel, self).load_statistics(dd)
        for model_name in self._sorted_model_order:
            m = self._models[model_name]
            if isinstance(m, Model):
                dd = m.load_statistics(dd)
        return dd

    def normalize_by_statistics(self, inputs: d, names, shared_dtype=None, check_finite=True, inverse=False,
                                shift_mean=True):
        this_level_names = list(set(names).intersection(self.save_normalization_inputs))
        names = list(set(names).difference(self.save_normalization_inputs))        # missing ones
        if len(this_level_names) > 0:
            inputs = super(GroupedModel, self).normalize_by_statistics(inputs, this_level_names,
                                                                       shared_dtype=shared_dtype,
                                                                       check_finite=check_finite, inverse=inverse,
                                                                       shift_mean=shift_mean)

        for model in self._models.values():
            if len(names) == 0:
                break
            next_level_names = list(set(names).intersection(model.save_normalization_inputs))
            names = list(set(names).difference(model.save_normalization_inputs))        # missing ones
            if len(next_level_names) > 0:
                inputs = model.normalize_by_statistics(inputs, next_level_names,
                                                       shared_dtype=shared_dtype,
                                                       check_finite=check_finite, inverse=inverse,
                                                       shift_mean=shift_mean)

        if len(names) > 0:
            raise ValueError(f"Missing names to normalize: {names}")

        return inputs

    def parse_kwargs_for_method(self, method, kwargs):
        dc = getattr(self, f"{method.lower()}_parsed_kwargs").copy()
        intersection_keys = set(kwargs.keys()).intersection(dc.keys())
        for key in intersection_keys:
            dc[key] = kwargs[key]
        return dc

    # def update(self, inputs: d, outputs: d, i=0, writer=None, writer_prefix="", **kwargs):
    #     """
    #     Updates the grouped model. This offloads work from the Trainer class, and must be implemented for subclasses
    #
    #     :param inputs:
    #     :param outputs:
    #     :param i:
    #     :param writer:
    #     :param writer_prefix:
    #     :param kwargs:
    #     :return:
    #     """
    #     pass

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[GroupedModel]")
        for model_name in self._sorted_model_order:
            m = self._models[model_name]
            print_fn(prefix + f"\t[{model_name}] type <{m.__class__}>")
            if hasattr(m, "print_parameters"):
                m.print_parameters(prefix + "\t\t", print_fn=print_fn)
            else:
                for p in m.parameters():
                    print_fn(prefix + f"\t\tparam <{list(p.shape)}> (requires_grad = {p.requires_grad})")

    # normal restoring behavior loads all parameters for the model. we might want to split loading between files
    def restore_from_checkpoints(self, model_names, checkpoints, strict=False):
        assert len(model_names) == len(checkpoints), "Same number of files must be passed in as models"
        models = self._models.get_keys_required(model_names)
        for m, chkpt in zip(models, checkpoints):
            if isinstance(m, Model):
                m.restore_from_checkpoint(chkpt, strict=strict)
            else:
                m.load_state_dict(chkpt['model'], strict=strict)

    def restore_from_files(self, model_names: List[str], file_names: Union[str, List[str]]):
        if isinstance(file_names, str):
            file_names = [file_names] * len(model_names)

        chkpts = [torch.load(f, map_location=self.device) for f in file_names]
        self.restore_from_checkpoints(model_names, chkpts)

    @property
    def all_models(self) -> d:
        return self._models.leaf_copy()

    @staticmethod
    def get_staticmethod_for_model(*args, static_fn_name="get_fn", model_name="model", **kwargs):
        base_fn = None

        def fn(model, *inner_args, **inner_kwargs):
            nonlocal base_fn
            if not base_fn:
                submodel = getattr(model, model_name)
                base_fn = getattr(submodel, static_fn_name)(*args, **kwargs)

            return base_fn(model, *inner_args, **inner_kwargs)

        return fn
