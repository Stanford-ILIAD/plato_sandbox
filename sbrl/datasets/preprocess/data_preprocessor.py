import numpy as np

from sbrl.envs.env_spec import EnvSpec
from sbrl.utils.python_utils import AttrDict, get_with_default


# TODO
class DataPreprocessor:
    def __init__(self, params: AttrDict, env_spec: EnvSpec, dataset=None):
        self._init_params_to_attrs(params)
        self._dataset = dataset  # optional
        self._env_spec = env_spec

    def _init_params_to_attrs(self, params: AttrDict):
        self._name: str = get_with_default(params, "name", "generic_preprocessor")

        # (datadict, onetime, ep_idx, self) -> (datadict, onetime)
        self._episode_preproc_fn = get_with_default(params, "episode_preproc_fn", None)

    def forward(self, dataset, datadict, onetime_datadict, split_indices, **kwargs):
        """
        Takes a datadict, a onetime_datadict, split_indices (numpy), and does some work on it.

        :return: all the above, modified.
        """
        if len(split_indices) == 0:
            return datadict, onetime_datadict, split_indices

        if self._episode_preproc_fn is not None:
            start = 0
            i = 0
            all_ep = []
            all_ep_onetime = []
            new_splits = []
            while i < len(split_indices):
                end = split_indices[i]

                new_ep, new_ep_onetime = self._episode_preproc_fn(datadict.leaf_apply(lambda arr: arr[start:end]),
                                                                   onetime_datadict.leaf_apply(
                                                                       lambda arr: arr[i:i + 1]), i)
                new_ep_onetime.leaf_assert(lambda arr: len(arr) == 1)
                all_ep.append(new_ep)
                all_ep_onetime.append(new_ep_onetime)
                new_splits.append(len(new_ep >> 'done'))

                start = end
                i += 1

            new_datadict = AttrDict.leaf_combine_and_apply(all_ep, np.concatenate)
            new_onetime_datadict = AttrDict.leaf_combine_and_apply(all_ep_onetime, np.concatenate)
            return new_datadict, new_onetime_datadict, np.cumsum(new_splits)

        else:

            raise NotImplementedError

    @property
    def name(self):
        return self._name
#
#
#
# class TransformAugmentation(DataAugmentation):
#     def _init_params_to_attrs(self, params: AttrDict):
#         super(TransformAugmentation, self)._init_params_to_attrs(params)
#         self.transforms: AttrDict = get_with_default(params, "transforms")
