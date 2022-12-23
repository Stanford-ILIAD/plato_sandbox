"""
This file is very similar to train; it's basically a subset of train
"""

import math
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, get_with_default
from sbrl.utils.script_utils import load_standard_ml_config
from sbrl.utils.torch_utils import to_numpy, get_horizon_chunks, to_torch

if __name__ == '__main__':

    parser = GroupedArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--eval_batch_size', type=int, default=32)  # ignore for now
    parser.add_argument('--eval_horizon', type=int, default=0)  # if 0, use data train's horizon
    parser.add_argument('--do_model_forward', action="store_true")
    parser.add_argument('--do_loss', action="store_true")
    parser.add_argument('--do_policy_forward', action="store_true")
    # if horizon > 1, eval len will be different than data len, pad eval with zeros
    parser.add_argument('--pad_eval_zeros', action="store_true")
    local_args, unknown = parser.parse_local_args()

    logger.debug(f"Args: \n{' '.join(sys.argv)}")

    ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'policy']

    config_fname = os.path.abspath(local_args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, args, params = load_standard_ml_config(config_fname, unknown, parser, ordered_modules,
                                                     debug=False)

    logger.debug(f"Retrieving: {exp_name}")
    file_manager = ExperimentFileManager(exp_name, is_continue=True)

    if args.model_file is not None:
        model_fname = os.path.abspath(args.model_file)
        if not os.path.exists(model_fname):
            model_fname = file_path_with_default_dir(args.model_file, file_manager.exp_dir)
            assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)

        logger.debug("Using model: {}".format(model_fname))

    # elif "checkpoint_model_file" in params.trainer.params and params.trainer.params['checkpoint_model_file'] is not None:
    #     model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
    #     logger.debug("Using checkpoint model for current eval: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))

    exit_on_ctrl_c()
    env_spec = params.env_spec.cls(params.env_spec.params)
    model = params.model.cls(params.model.params, env_spec, None)
    policy = params.policy.cls(params.policy.params, env_spec)

    if args.eval_horizon > 0:
        HORIZON = args.eval_horizon
        logger.warn("Overriding horizon in dset with %d" % HORIZON)
    else:
        HORIZON = get_with_default(params.dataset_train.param, "horizon", 0)

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

    all_names = env_spec.all_names + ["done", "rollout_timestep"]

    eval_datadict = []


    def get_concat_if_concatable(dim):
        def concat_fn(vs):
            if isinstance(vs[0], np.ndarray):
                return np.concatenate(vs, axis=dim)
            elif isinstance(vs[0], torch.Tensor):
                return torch.cat(vs, dim=dim)
            else:
                return vs
        return concat_fn


    for i in range(dataset_input.get_num_episodes()):
        # (N x ...) where N is episode length
        inputs, outputs = dataset_input.get_episode(i, all_names, split=True)
        # print(datadict.done.shape, HORIZON)

        if len(outputs.done) < HORIZON:
            logger.warn("Episode %d does not have enough data pts (%d instead of %d). Skipping." % (i, len(outputs.done), HORIZON))
            continue

        # # (inclusive range), new shape is (N - H + 1, H, ...)
        inputs_chunked = inputs\
            .leaf_apply(lambda arr: get_horizon_chunks(arr, HORIZON, 0, len(outputs.done) - HORIZON, dim=0, stack_dim=0))
        outputs_chunked = outputs\
            .leaf_apply(lambda arr: get_horizon_chunks(arr, HORIZON, 0, len(outputs.done) - HORIZON, dim=0, stack_dim=0))
        #
        # # print(datadict_episodic.done.shape)
        #
        # # these were (1,  ...), now (N - H + 1, H, ...)
        # onetime_datadict_episodic = datadict\
        #     .leaf_filter_keys(env_spec.param_names + env_spec.final_names)\
        #     .leaf_apply(lambda arr: broadcast_dims_np(arr[:, None], [0, 1], [len(datadict.done) - HORIZON + 1, HORIZON]))

        # model.eval()
        with torch.no_grad():
            policy_forward = []
            model_forward = []
            losses = []

            num_chunks = len(outputs_chunked.done)
            batch_size = min(max(args.eval_batch_size, 1), num_chunks)
            num_batches = math.ceil(num_chunks / batch_size)
            for j in range(num_batches):
                start, end = j * batch_size, min((j+1) * batch_size, num_chunks)
                # convert to torch
                inputs_t = inputs_chunked.leaf_apply(lambda arr: to_torch(arr, device=model.device))
                outputs_t = outputs_chunked.leaf_apply(lambda arr: to_torch(arr, device=model.device))

                # for policy (np)
                inputs_j = inputs_chunked.leaf_apply(lambda arr: arr[start:end])
                goals_j = inputs_chunked.leaf_apply(lambda arr: arr[start:end])
                outputs_j = outputs_chunked.leaf_apply(lambda arr: arr[start:end])

                # for model (torch)
                inputs_t_j = inputs_t.leaf_apply(lambda arr: arr[start:end])
                outputs_t_j = inputs_t.leaf_apply(lambda arr: arr[start:end])

                policy_forward_j = AttrDict()
                if args.do_policy_forward:
                    policy_forward_j = policy.get_action(model, inputs_j, goals_j)  # should output torch device

                model_forward_j = AttrDict()
                if args.do_model_forward:
                    model_forward_j = model.forward(inputs_t_j, )  # TODO sample as a param

                loss_j = AttrDict()
                if args.do_loss:
                    loss_out = model.loss(inputs_t_j, outputs_t_j, ret_dict=False)
                    if isinstance(loss_out, List) or isinstance(loss_out, Tuple):
                        for j in range(len(loss_out)):
                            loss_j["loss_%d" % j] = loss_out[j]
                    elif isinstance(loss_out, AttrDict):
                        loss_j = loss_out
                    else:
                        loss_j.loss = loss_out[None]  # single loss

                    loss_j.pprint()

                policy_forward.append(policy_forward_j)
                model_forward.append(model_forward_j)
                losses.append(loss_j)

            # evaluations for this episode
            model_forward = AttrDict.leaf_combine_and_apply(model_forward, get_concat_if_concatable(0))
            policy_forward = AttrDict.leaf_combine_and_apply(policy_forward, get_concat_if_concatable(0))
            losses = AttrDict.leaf_combine_and_apply(losses, get_concat_if_concatable(0))

            def to_np_pad(arr):
                arr = to_numpy(arr, check=True)
                # do this to make sure everything is aligned and the same length
                if args.pad_eval_zeros:
                    padding = [[0, 0] for _ in range(len(arr.shape))]
                    padding[0][1] = len(outputs.done) - num_chunks
                    return np.pad(arr, padding)
                return arr

            evaluations = AttrDict()
            evaluations.model = model_forward.leaf_apply(lambda arr: to_np_pad(arr) if isinstance(arr, torch.Tensor) else arr)
            evaluations.policy = policy_forward.leaf_apply(lambda arr: to_np_pad(arr) if isinstance(arr, torch.Tensor) else arr)
            evaluations.losses = losses.leaf_apply(lambda arr: to_np_pad(arr) if isinstance(arr, torch.Tensor) else arr)

        new_datadict = AttrDict()
        new_datadict.combine(inputs)
        new_datadict.combine(outputs)
        new_datadict.evaluations = evaluations

        # print(obs.ee_position.shape, new_action.action.shape, out_obs.next_ee_position.shape)

        eval_datadict.append(new_datadict)
        if i == 0:
            logger.info("Using keys for first batch: " + str(list(new_datadict.leaf_keys())))
            for key, item in new_datadict.leaf_items():
                logger.debug("%s : type: %s, shape: %s" % (key, type(item), item.shape if isinstance(item, np.ndarray) else []))

    eval_datadict = AttrDict.leaf_combine_and_apply(eval_datadict, lambda vs: np.concatenate(vs, axis=0) if isinstance(vs[0], np.ndarray) else np.array(vs, dtype=object))


    logger.debug("Saving dataset output to -> %s" % output_file)

    logger.debug("Keys: " + str(list(eval_datadict.leaf_keys())))
    to_save = dict()
    for name in eval_datadict.leaf_keys():
        to_save[name] = eval_datadict[name]

    np.savez_compressed(output_file, **to_save)
