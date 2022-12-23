from typing import List

import numpy as np

from sbrl.datasets.dataset import Dataset
from sbrl.datasets.preprocess.batch_processor import BatchProcessor
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit


class DataAugmentKeys(BatchProcessor):
    """
    Batch processors take a batch of inputs, outputs, and process them for a model, e.g. by adding sampling info.
    These are often specific to dataset.
    """
    def _init_params_to_attrs(self, params: AttrDict):
        self._add_noise_keys = get_with_default(params, "add_noise_keys", [], map_fn=list)
        self._add_noise_stds = get_with_default(params, "add_noise_stds", [], map_fn=list)
        self._add_noise_stds = [np.asarray(s) for s in self._add_noise_stds]
        assert len(self._add_noise_keys) == len(self._add_noise_stds)
        self._add_noise = len(self._add_noise_stds) > 0
        self._use_base_dataset_stds = get_with_default(params, "use_base_dataset_stds", True)
        self._same_noise = get_with_default(params, "same_noise", True)  # noise will be generated over batch size of 1 for efficiency

        assert all(np.all(std > 0) for std in self._add_noise_stds)

        if self._add_noise:
            logger.debug(f"Adding noise: {AttrDict.from_kvs(self._add_noise_keys, self._add_noise_stds).pprint(ret_string=True)} (raw = {not self._use_base_dataset_stds})")

    def setup(self, dataset: Dataset):
        self._dataset = dataset

        if self._use_base_dataset_stds:
            stats = self._dataset.get_statistics(self._add_noise_keys)
            dset_stds = stats >> "std"
            self._add_noise_stds = [std * (dset_stds >> key) for key, std in zip(self._add_noise_keys, self._add_noise_stds)]

            logger.debug(f"Updated stds: {AttrDict.from_kvs(self._add_noise_keys, self._add_noise_stds).pprint(ret_string=True)}")

        # self._pre_alloc = AttrDict.from_kvs(self._add_noise_keys, [None] * len(self._add_noise_keys))

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
        join_inp_outs = inputs & outputs
        assert join_inp_outs.has_leaf_keys(self._add_noise_keys), join_inp_outs.leaf_key_symmetric_difference(self._add_noise_keys)
        assert isinstance(inputs.get_one(), np.ndarray), "np arrays required here."

        # the rest of the forward call will be mutating
        inputs = inputs.leaf_copy()
        outputs = outputs.leaf_copy()

        with timeit("data_augment_keys/forward"):
            for key, std in zip(self._add_noise_keys, self._add_noise_stds):
                src = inputs if key in inputs.leaf_keys() else outputs
                if self._same_noise:
                    noise = np.random.randn(1, *src[key].shape[1:])
                else:
                    noise = np.random.randn(*src[key].shape)
                noise = noise * std
                new_src = src[key] + noise
                assert list(new_src.shape) == list(src[key].shape), key  # maintain shape!
                src[key] = new_src

        return inputs, outputs, AttrDict()
