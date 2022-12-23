"""
This file is very similar to train; it's basically a subset of train
"""

import os
from typing import List, Tuple

import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, is_array
from sbrl.utils.script_utils import load_standard_ml_config
from sbrl.utils.torch_utils import to_numpy, combine_dims_np

if __name__ == '__main__':
    parser = GroupedArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--no_model_file', action="store_true")
    # parser.add_argument('--batch_size', type=int, default=32)  # ignore for now
    parser.add_argument('--override_horizon', type=int, default=0)  # if 0, use data train's horizon
    parser.add_argument('--num_iters', type=int, default=10000)  # number of batches to sample
    parser.add_argument('--do_model_forward', action="store_true")
    parser.add_argument('--model_forward_kwargs', type=str, default="{}")
    parser.add_argument('--do_loss', action="store_true")
    parser.add_argument('--do_policy_forward', action="store_true")
    # if horizon > 1, eval len will be different than data len, pad eval with zeros
    parser.add_argument('--pad_eval_zeros', action="store_true")
    parser.add_argument('--depad_done_key', type=str, default="done", help="Key to de-pad at the end. only last element can be true per batch.")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--combine_first_n_dims', type=int, default=0,
                        help="will squash the first N dims of the outputs before saving")
    args, unknown = parser.parse_local_args()

    ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'policy']

    config_fname = os.path.abspath(args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, new_args, params = load_standard_ml_config(config_fname, unknown, parser, ordered_modules,
                                                     debug=False)

    file_manager = ExperimentFileManager(exp_name, is_continue=True)

    if args.model_file is not None:
        model_fname = args.model_file
        model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
        assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
        logger.debug("Using model: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))

    exit_on_ctrl_c()
    env_spec = params.env_spec.cls(params.env_spec.params)
    model = params.model.cls(params.model.params, env_spec, None)
    policy = params.policy.cls(params.policy.params, env_spec)

    if args.device is not None:
        model.to(args.device)

    if args.override_horizon > 0:
        HORIZON = args.override_horizon
        logger.warn("Overriding horizon in dset with %d" % HORIZON)
    else:
        HORIZON = params.dataset_train.params.get("horizon", 0)

    assert HORIZON > 0
    params.dataset_train.params.file = args.file
    params.dataset_train.params.horizon = HORIZON
    dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
    # params.dataset_train.params.file = None
    # params.dataset_train.params.output_file = args.output_file
    # dataset_output = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

    output_file = file_path_with_default_dir(args.output_file, file_manager.exp_dir, expand_user=True)

    ### restore model
    if not args.no_model_file:
        model.restore_from_file(model_fname)

    all_names = env_spec.all_names + ["done"]

    eval_datadict = []

    if args.do_model_forward:
        import json
        model_forward_kwargs = json.loads(args.model_forward_kwargs)

    def get_np_concatable(vs, dim):
        if isinstance(vs[0], np.ndarray):
            pre = [vs[i].shape[:dim] for i in range(len(vs))]
            post = [vs[i].shape[dim+1:] for i in range(len(vs))]
            last_pre = pre[0]
            last_post = post[0]
            for i in range(1, len(vs)):
                if last_pre != pre[i] or last_post != post[i]:
                    return False
        return True


    def cat_fn(vs):
        if (isinstance(vs[0], np.ndarray) and get_np_concatable(vs, 0)):
            return np.concatenate(vs, axis=0)
        return None
        # else:
        #     return np.array(vs, dtype=object)

    assert args.num_iters > 0, "Pass in nonzero number of batch iterations"

    all_policy_forward = []
    all_model_forward = []
    all_losses = []

    for i in range(args.num_iters):
        # (N x ...) where N is episode length
        res = dataset_input.get_batch(torch_device=model.device)
        inputs, outputs = res[:2]
        meta = res[2] if len(res) == 3 else AttrDict()
        # print(datadict.done.shape, HORIZON)

        # model.eval()
        with torch.no_grad():
            policy_forward = AttrDict()
            if args.do_policy_forward:
                policy_forward = policy.get_action(model, inputs, inputs)  # should output torch device

            model_forward = AttrDict()
            if args.do_model_forward:
                model_forward = model.forward(inputs, **model_forward_kwargs, meta=meta)  # TODO sample as a param

            loss = AttrDict()
            if args.do_loss:
                loss_out = model.loss(inputs, outputs, ret_dict=True, meta=meta)
                if isinstance(loss_out, List) or isinstance(loss_out, Tuple):
                    for j in range(len(loss_out)):
                        loss["loss_%d" % j] = loss_out[j]
                elif isinstance(loss_out, AttrDict):
                    loss = loss_out
                else:
                    loss.loss = loss_out[None]  # single loss

                loss.pprint()

        all_policy_forward.append(policy_forward)
        all_model_forward.append(model_forward)
        all_losses.append(loss)

        evaluations = AttrDict()
        evaluations.model = model_forward.leaf_apply(lambda arr: to_numpy(arr, check=True) if is_array(arr) else arr)
        evaluations.policy = policy_forward.leaf_apply(lambda arr: to_numpy(arr, check=True) if is_array(arr) else arr)
        evaluations.losses = loss.leaf_apply(lambda arr: to_numpy(arr, check=True) if is_array(arr) else arr)

        new_datadict = AttrDict()
        new_datadict.combine(inputs.leaf_apply(lambda arr: to_numpy(arr, check=True)))
        new_datadict.combine(outputs.leaf_apply(lambda arr: to_numpy(arr, check=True)))
        new_datadict.evaluations = evaluations

        # batch dones might be padded. make sure there is only one "done" at the end.
        done = new_datadict >> args.depad_done_key
        done[:, :-1] = False
        if args.combine_first_n_dims > 1:
            # we need to ensure done = True for last element, since things will be combined...
            done[:, -1] = True

        # print(obs.ee_position.shape, new_action.action.shape, out_obs.next_ee_position.shape)

        eval_datadict.append(new_datadict)
        if i == 0:
            logger.info("Using keys for first batch: " + str(list(new_datadict.leaf_keys())))
            for key, item in new_datadict.leaf_items():
                logger.debug("%s : type: %s, shape: %s" % (key, type(item), item.shape if isinstance(item, np.ndarray) else []))



    eval_datadict = AttrDict.leaf_combine_and_apply(eval_datadict, cat_fn)

    if args.combine_first_n_dims > 1:
        logger.info(f"Combining first {args.combine_first_n_dims} dimensions of output shape..")
        eval_datadict.leaf_modify(lambda arr: combine_dims_np(arr, 0, num_axes=args.combine_first_n_dims))

    logger.debug("Saving dataset output to -> %s" % output_file)

    # logger.debug("Keys: " + str(list(eval_datadict.leaf_keys())))
    for key, item in eval_datadict.leaf_items():
        logger.debug("%s : type: %s, shape: %s" % (key, type(item), item.shape if isinstance(item, np.ndarray) else []))

    to_save = dict()
    for name in eval_datadict.leaf_keys():
        to_save[name] = eval_datadict[name]

    np.savez_compressed(output_file, **to_save)

    logger.debug("done.")
