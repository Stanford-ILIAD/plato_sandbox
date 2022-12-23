from argparse import ArgumentParser

import numpy as np
import torch

from sbrl.datasets.data_augmentation import DataAugmentation
from sbrl.utils.python_utils import AttrDict as d
from sbrl.utils.torch_utils import to_torch


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument("--keys", type=str, nargs="*", default=[])
    parser.add_argument("--stds", type=float, nargs="*", default=[],
                        help="gaussian noise with these std's. broadcast-able to true shape")


def process_params(group_name, common_params):
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    # access to all the params for the current experiment here.
    prms = common_params >> group_name

    # default type of data augmentation (adds noise)
    AUGMENT_KEYS = prms >> "keys"
    AUGMENT_FNS = []
    aug_values = prms >> "stds"
    assert len(aug_values) == len(AUGMENT_KEYS), [aug_values, AUGMENT_KEYS]
    if len(AUGMENT_KEYS) > 0:
        torch_aug_values = [to_torch(np.asarray(val)[None], device=common_params >> "device") for val in aug_values]

        def get_augment_fn(std):
            def fn(arr, **kwargs):
                return arr + std * torch.randn_like(arr)

            return fn

        AUGMENT_FNS = [get_augment_fn(std) for std in torch_aug_values]

    common_params[group_name] = d()
    # only add if
    if len(AUGMENT_KEYS) > 0:
        common_params[group_name].combine(d(
            cls=DataAugmentation,
            params=d(
                augment_keys=AUGMENT_KEYS,
                augment_fns=AUGMENT_FNS,  # TODO
            )
        ))

    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
