from typing import List

import numpy as np

from sbrl.datasets.dataset import Dataset
from sbrl.utils.python_utils import AttrDict


class BatchProcessor:
    """
    Batch processors take a batch of inputs, outputs, and process them for a model, e.g. by adding sampling info.
    These are often specific to dataset.
    """
    def __init__(self, params: AttrDict):
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params: AttrDict):
        pass

    def setup(self, dataset: Dataset):
        pass

    def forward(self, inputs: AttrDict, outputs: AttrDict, indices: np.ndarray, period_indices: np.ndarray, episode_indices: np.ndarray, names_to_get: List[str], chunk_lengths: np.ndarray, **kwargs):
        """

        :param inputs:
        :param outputs:
        :param indices:
        :param period_indices:
        :param episode_indices:
        :param names_to_get:
        :param chunk_lengths:
        :param kwargs:

        :return: new_inputs, new_outputs, new_metadata
        """
        return inputs, outputs, AttrDict()

