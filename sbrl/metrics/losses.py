import torch

from sbrl.metrics.metric import Metric, ElementwiseMetric, MultiTaskMetric, maximum, minimum
from sbrl.utils.python_utils import get_required, get_with_default, AttrDict as d, get_from_ls


class Loss(Metric):
    """
    Generic loss that takes dicts
    """
    def __init__(self, params):
        self.loss_fn = get_required(params, "loss_fn")
        name = get_with_default(params, 'name', 'loss')
        super().__init__(name=name)

    def _compute(self, inputs: d, outputs: d, model_outputs: d):
        return self.loss_fn(inputs, outputs, model_outputs)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)


class SingleTensorLoss(Loss):
    """
    Specific wrapper for single key pair loss (e.g. BCE(y_pred, y_true))
    """
    def __init__(self, params):
        super().__init__(params)

        self.pred_key = get_required(params, "pred_key")
        self.true_key = get_with_default(params, "true_key", self.pred_key)

        self.true_key_source = get_from_ls(params, "true_key_source", ["inputs", "outputs"], default_idx=1, map_fn=lambda x: str(x).lower())

    def _compute(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        if self.true_key_source == "inputs":
            return self.loss_fn(model_outputs >> self.pred_key, inputs >> self.true_key)
        else:
            return self.loss_fn(model_outputs >> self.pred_key, outputs >> self.true_key)


class ElementwiseSingleTensorLoss(SingleTensorLoss, ElementwiseMetric):

    def _compute_element_wise(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        self._compute(inputs, outputs, model_outputs)


# TODO this is broken (see parent)
class MultiTaskSingleTensorLoss(SingleTensorLoss, MultiTaskMetric):

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        elif isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            flattened_y_true = flattened_y_true.long()
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss


class Accuracy(ElementwiseMetric):
    def __init__(self, params):
        self.pred_fn = get_with_default(params, "pred_fn", None)
        self.true_fn = get_with_default(params, "true_fn", self.pred_fn)
        name = get_with_default(params, 'name', 'acc')
        super().__init__(name=name)

        self.pred_key = get_required(params, "pred_key")
        self.true_key = get_with_default(params, "true_key", self.pred_key)

        self.true_key_source = get_from_ls(params, "true_key_source", ["inputs", "outputs"], default_idx=1, map_fn=lambda x: str(x).lower())

    def _compute_element_wise(self, inputs: d, outputs: d, model_outputs: d):
        y_pred = model_outputs >> self.pred_key
        y = (inputs if self.true_key_source == "inputs" else outputs) >> self.true_key
        if self.pred_fn is not None:
            y_pred = self.pred_fn(y_pred)
        if self.true_fn is not None:
            y = self.true_fn(y)

        return (y_pred == y).float()

    def worst(self, metrics):
        return minimum(metrics)
