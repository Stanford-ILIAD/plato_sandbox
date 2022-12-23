from typing import List

from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict, get_with_default


# TODO
class DataAugmentation:
    def __init__(self, params: AttrDict, dataset=None):
        self._init_params_to_attrs(params)
        self._dataset = dataset  # optional

    def _init_params_to_attrs(self, params: AttrDict):
        self.augment_keys: List[str] = get_with_default(params, "augment_keys", [])
        # (arr, torch_device=torch_device, memory=memory, **kwargs) -> arr_augmented
        self.augment_fns = get_with_default(params, "augment_fns", [])

        # will run this on the dataset to edit it beforehand, if a dataset gets linked.
        # (dataset, DataAugmentation) -> base memory AttrDict
        self.preproc_dataset_fn = get_with_default(params, "preproc_dataset_fn", None)

        # inputs, outputs -> inputs, outputs. optional additional fn.
        self.preproc_fn = get_with_default(params, "preproc_fn", lambda inputs, outputs, *args, **kwargs: (inputs, outputs))
        self.postproc_fn = get_with_default(params, "postproc_fn", lambda inputs, outputs, *args, **kwargs: (inputs, outputs))
        self.post_mask_name = params << "post_mask_name"

        assert len(self.augment_fns) == len(self.augment_keys), "Mismatch in augment keys / fns"

        self.augment_dict = AttrDict()
        self._base_memory = AttrDict()

        for i, key in enumerate(self.augment_keys):
            self.augment_dict[key] = self.augment_fns[i]

        self._read_keys = get_with_default(params, "read_keys", [], map_fn=list)

    def link_dataset(self, dataset):
        self._dataset = dataset

        if self.preproc_dataset_fn is not None:
            logger.debug("DataAugmentation: Running preprocessing on whole dataset...")
            self._base_memory = self.preproc_dataset_fn(self._dataset, self)

    def forward(self, inputs: AttrDict = None, outputs: AttrDict = None, torch_device=None, allow_missing=False, **kwargs):
        """
        All the inputs are optional, returns the data augmentation on a batch of data (see NpDataset)

        :param inputs: optional input keys
        :param outputs: optional output keys
        :param torch_device: If this is None, the input is NP array, otherwise do transforms on this device
        :param allow_missing: will skip keys that aren't in input / output. otherwise requires them
        :param kwargs:
        :return: optional inputs, optional outputs, modified
        """

        input_keys_present, output_keys_present = [], []
        if inputs is not None:
            inputs = inputs.copy()
            input_keys_present += inputs.list_leaf_keys()
        if outputs is not None:
            outputs = outputs.copy()
            output_keys_present += outputs.list_leaf_keys()

        key_overlap = list(set(input_keys_present + output_keys_present).intersection(set(self.augment_keys)))

        if not allow_missing:
            assert len(key_overlap) == len(self.augment_keys), \
                "Missing keys from set %s, given input set %s, output set %s" % (self.augment_keys, input_keys_present, output_keys_present)

        # this can be used or ignored by each fn (e.g. for correlated noise, etc)
        memory = self._base_memory.copy()
        if len(self._read_keys) > 0:
            memory.read_inputs = (inputs & outputs) > self._read_keys

        inputs, outputs = self.preproc_fn(inputs, outputs, memory=memory, **kwargs)

        # default behavior is just apply all functions
        for key in self.augment_keys:
            if key in input_keys_present:
                inputs[key] = self.augment_dict[key](inputs[key], torch_device=torch_device, memory=memory, key=key,
                                                     inputs=inputs, outputs=outputs, **kwargs)
            if key in output_keys_present:
                outputs[key] = self.augment_dict[key](outputs[key], torch_device=torch_device, memory=memory, key=key,
                                                      inputs=inputs, outputs=outputs, **kwargs)

        return self.postproc_fn(inputs, outputs, mask=memory << self.post_mask_name, memory=memory, **kwargs)
#
#
#
# class TransformAugmentation(DataAugmentation):
#     def _init_params_to_attrs(self, params: AttrDict):
#         super(TransformAugmentation, self)._init_params_to_attrs(params)
#         self.transforms: AttrDict = get_with_default(params, "transforms")


