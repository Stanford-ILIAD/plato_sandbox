"""
This file is very similar to train; it's basically a subset of train
"""

import os

import cv2
import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict
from sbrl.utils.script_utils import load_standard_ml_config

if __name__ == '__main__':

    parser = GroupedArgumentParser()
    parser.add_argument('config', type=str)
    # parser.add_argument('--model_file', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    # parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--image_key', type=str, required=True)  # ignore for now
    parser.add_argument('--use_episodes', action="store_true")
    parser.add_argument('--flip_imgs', action="store_true")
    # parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--horizon', type=int, default=None)  # ignore for now

    # file_manager = ExperimentFileManager(params.exp_name, is_continue=True)
    local_args, unknown = parser.parse_local_args()
    # register_config_args(unknown)

    ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train']

    config_fname = os.path.abspath(local_args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, args, params = load_standard_ml_config(config_fname, unknown, parser, ordered_modules, debug=True)

    # logger.debug(f"Loading: {exp_name}")
    file_manager = ExperimentFileManager("test", is_continue=True)

    # if local_args.model_file is not None:
    #     model_fname = os.path.abspath(local_args.model_file)
    #     assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    #     logger.debug("Using model: {}".format(model_fname))
    # elif "checkpoint_model_file" in params.trainer.params and params.trainer.params[
    #     'checkpoint_model_file'] is not None:
    #     model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
    #     logger.debug("Using checkpoint model for current eval: {}".format(model_fname))
    # else:
    #     model_fname = os.path.join(file_manager.models_dir, "model.pt")
    #     logger.debug("Using default model for current eval: {}".format(model_fname))

    exit_on_ctrl_c()

    env_spec = params.env_spec.cls(params.env_spec.params)
    # model = params.model.cls(params.model.params, env_spec, None)
    # policy = params.policy.cls(params.policy.params, env_spec)

    if local_args.horizon is not None:
        params.dataset_train.params.horizon = local_args.horizon

    horizon = params.dataset_train.params.horizon

    params.dataset_train.params.file = local_args.file
    params.dataset_train.params.batch_size = 1  # single episode, get_batch returns (1, H, ...)
    dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
    sampler_in = dataset_input.sampler
    # params.dataset_train.params.file = None
    # params.dataset_train.params.output_file = args.output_file
    # dataset_output = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

    ### restore model
    # if not local_args.no_model_file:
    #     model.restore_from_file(model_fname)

    # all_names = env_spec.all_names + ["done"]

    eval_datadict = AttrDict()

    cv2.namedWindow("sequence_image", cv2.WINDOW_AUTOSIZE)

    running = True
    while running:
        # (1 x H x ...)
        if args.use_episodes:
            inputs, outputs = dataset_input.get_episode(np.random.randint(dataset_input.get_num_episodes()), names=None, split=True)
            inputs.leaf_modify(lambda arr: arr[None])
            outputs.leaf_modify(lambda arr: arr[None])
            meta = AttrDict()
        else:
            res = dataset_input.get_batch(indices=sampler_in.get_indices())
            inputs, outputs = res[:2]
            meta = res[2] if len(res) == 3 else AttrDict()
            inputs = inputs & (meta < [local_args.image_key])
            # todo why do we need this
            # if inputs << "policy_type" is not None:
            #     # only pulling
            #     if 1 not in np.unique(inputs.policy_type[0]):
            #         continue

        assert inputs.has_leaf_key(local_args.image_key), [inputs.list_leaf_keys(), local_args.image_key]

        batch_len = min(outputs.done.shape[1], inputs[local_args.image_key].shape[1])

        start_inputs = inputs.leaf_apply(lambda arr: arr[0, 0])
        end_inputs = inputs.leaf_apply(lambda arr: arr[0, -1])

        i = 0
        while i < batch_len:
            logger.debug("[%d] press n = next, p = prev, q = next batch, esc = end.." % i)
            img = inputs[local_args.image_key][0, i]
            if local_args.flip_imgs:
                img = img[..., ::-1]
            cv2.imshow("sequence_image", img.astype(np.uint8))
            # print("done:", outputs.done[0, i])
            # print("bp:", (inputs >> "block_positions")[0, i, 0])
            ret = cv2.waitKey(0)
            if ret == ord('n'):  # next
                i = (i + 1) % batch_len
            elif ret == ord('p'):  # prev
                i = (i - 1) % batch_len
            elif ret == ord('q'):  # next batch
                i = batch_len
            elif ret == 27:  # esc
                i = batch_len
                running = False









    #
    # for i in range(dataset_input.get_num_episodes()):
    #     # two empty axes for (batch_size, horizon)
    #     datadict = dataset_input.get_episode(i, all_names)
    #     obs = datadict.leaf_filter_keys(env_spec.observation_names)
    #     goal = datadict.leaf_filter_keys(env_spec.goal_names)
    #     prms = datadict.leaf_filter_keys(env_spec.param_names)
    #     finals = datadict.leaf_filter_keys(env_spec.final_names)
    #     old_actions = datadict.leaf_filter_keys(env_spec.action_names)
    #     out_obs = datadict.leaf_filter_keys(env_spec.output_observation_names)
    #     with torch.no_grad():
    #         # might contain multiple different outputs
    #         inps = obs.leaf_apply(lambda arr: arr[:, None])
    #         inps.combine(old_actions.leaf_apply(lambda arr: arr[:, None]))
    #         inps.combine(prms.leaf_apply(lambda arr: np.repeat(arr, len(datadict.done), axis=0)))  # need to tile onetime data
    #         new_outs = policy.get_action(model, inps, goal.leaf_apply(lambda arr: arr[:, None]))
    #         new_outs = new_outs.leaf_filter(func=lambda key,val: isinstance(val, torch.Tensor) or isinstance(val, np.ndarray))
    #         new_outs = new_outs.leaf_apply(lambda arr: to_numpy(arr, check=True))
    #     obs.combine(goal)
    #     obs.combine(old_actions)
    #     obs.combine(new_outs)
    #     out_obs.done = datadict.done
    #     obs.combine(out_obs)
    #     obs.combine(prms)
    #     obs.combine(finals)
    #     new_datadict = obs
    #     # print(obs.ee_position.shape, new_action.action.shape, out_obs.next_ee_position.shape)
    #
    #     if i == 0:
    #         eval_datadict = new_datadict
    #         logger.info("Using keys: " + str(list(eval_datadict.leaf_keys())))
    #     else:
    #         eval_datadict = AttrDict.leaf_combine_and_apply([eval_datadict, new_datadict], lambda vs: np.concatenate([vs[0], vs[1]], axis=0))
    #
    #
    # logger.debug("Saving dataset output to -> exp_dir/%s" % args.output_file)
    #
    # path = os.path.join(file_manager.exp_dir, args.output_file)
    # logger.debug("Keys: " + str(list(eval_datadict.leaf_keys())))
    # to_save = dict()
    # for name in eval_datadict.leaf_keys():
    #     to_save[name] = eval_datadict[name]
    #
    # np.savez_compressed(path, **to_save)
