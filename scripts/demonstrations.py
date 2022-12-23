"""
This file is for collecting demonstrations (no training). similar to eval but we use train dataset
"""

import datetime
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.input_utils import KeyInput as KI, wait_for_keydown_from_set, query_string_from_set, ProcessUserInput
from sbrl.utils.pygame_utils import PygameOnlyKeysInput, TextFillPygameDisplay
from sbrl.utils.python_utils import timeit, exit_on_ctrl_c, AttrDict
from sbrl.utils.script_utils import load_standard_ml_config
from sbrl.utils.torch_utils import to_numpy

if __name__ == '__main__':

    grouped_parser = GroupedArgumentParser()
    grouped_parser.add_argument('config', type=str)
    grouped_parser.add_argument('--model_file', type=str)
    grouped_parser.add_argument('--file', type=str, nargs="*", default=[], help="1 or more input files")
    grouped_parser.add_argument('--record_external_video', type=str, default=None)
    grouped_parser.add_argument('--record_external_video_size', type=str, default="1080x640")
    grouped_parser.add_argument('--video_file_base', type=str, default="vid_output")
    grouped_parser.add_argument('--ds_out_name', type=str, default="dataset_out")
    grouped_parser.add_argument('--output_file', type=str, default="demonstrations.npz")
    grouped_parser.add_argument('--no_model_file', action="store_true")
    grouped_parser.add_argument('--do_input_process', action="store_true")
    grouped_parser.add_argument('--save_every_ep', action='store_true')
    grouped_parser.add_argument('--save_start_ep', type=int, default=0)
    grouped_parser.add_argument('--do_sequence_select', action="store_true", help="Select a sequence using sequence key of horizon length")
    grouped_parser.add_argument('--sequence_key', type=str, default="image")
    grouped_parser.add_argument('--sequence_length', type=int, default=0, help="default uses dataset horizon")
    grouped_parser.add_argument('--add_episode_labels', action="store_true", help="Allow episode labels in dataset (ints)")
    grouped_parser.add_argument('--episode_label_as_final', action="store_true", help="Add episode labels in dataset as final, instead of extra (ints)")
    grouped_parser.add_argument('--episode_label_set', type=str, nargs="*", default=[], help="valid episode labels")
    grouped_parser.add_argument('--use_env_display', action="store_true", help="If true, will use display defined in env (may not be process compatible)")
    local_args, unknown = grouped_parser.parse_local_args()

    ordered_modules = ['env_spec', 'env_train', 'model', local_args.ds_out_name, 'policy']

    exit_on_ctrl_c()  # in case of infinite waiting

    config_fname = os.path.abspath(local_args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
    exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
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

    env_spec = params.env_spec.cls(params.env_spec.params)
    env = params.env_train.cls(params.env_train.params, env_spec)
    model = params.model.cls(params.model.params, env_spec, None)

    running = True
    def quit(*local_args):
        global running
        running = False
        sys.exit(0)

    ### pygame display details
    if local_args.use_env_display:
        # environment owns display
        display = env.display
    else:
        # script owns display
        display = TextFillPygameDisplay(AttrDict())

    assert display is not None, "Env (%s) doesn't have a display set up" % type(env)

    ### handling user input
    empty_handler = lambda ui, ki: None
    if local_args.do_input_process:
        input_handle_params = AttrDict(
            base_ui_cls=PygameOnlyKeysInput,
            base_ui_params=AttrDict(),
            display=display,
        )
        input_handle = ProcessUserInput(input_handle_params, {})
    else:
        input_handle = PygameOnlyKeysInput(AttrDict(display=display), {})

    input_handle.register_callback(KI("r", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("y", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("n", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("q", KI.ON.down), quit)

    if local_args.do_input_process:
        input_handle.run()

    # input handle is required if using a teleop policy
    policy = params.policy.cls(params.policy.params, env_spec, file_manager=file_manager, env=env, input_handle=input_handle)

    ## demonstrations (must be Np dataset for now)
    dtparams = params[local_args.ds_out_name].params.copy()
    assert params[local_args.ds_out_name].cls == NpDataset, params[local_args.ds_out_name].cls
    if len(local_args.file) > 0:  # user provided files, otherwise empty
        dtparams.file = local_args.file
    else:
        dtparams.file = None

    dtparams.output_file = file_path_with_default_dir(local_args.output_file, file_manager.exp_dir, expand_user=True)

    dataset = params[local_args.ds_out_name].cls(dtparams, env_spec, file_manager)

    ### sequence selection

    select_seq = local_args.do_sequence_select
    seq_len = local_args.sequence_length
    if select_seq:
        global cv2
        import cv2
        assert local_args.sequence_key in env_spec.observation_names, "%s not in %s" % (local_args.sequence_key, env_spec.observation_names)
        logger.debug("Using sequence selection.")
        if seq_len <= 0:
            seq_len = dataset.horizon
            logger.warn("Overriding sequences length to horizon %d" % seq_len)
        else:
            logger.info("Setting sequences length to %d" % seq_len)

    if local_args.add_episode_labels:
        assert len(local_args.episode_label_set) > 0
        logger.debug("Using episode labels from set: %s" % local_args.episode_label_set)
    else:
        assert not local_args.episode_label_as_final

    if local_args.episode_label_as_final:
        assert 'label' in env_spec.final_names, env_spec.final_names

    ### restore model
    if not local_args.no_model_file:
        logger.info("Restoring model from %s" % model_fname)
        model.restore_from_file(model_fname)


    ### eval loop
    done = [False]
    steps = 0
    start_time = time.time()
    episode_count = 1
    latest_save = 0
    next_save_ep = 0
    episode_inputs = None
    episode_outputs = None

    extra_data = AttrDict()

    record_video_external = local_args.record_external_video
    vid_cmd_base = None
    vid_file_base = local_args.video_file_base
    proc = None
    if record_video_external is not None:
        rate = int(1 / env.dt)
        vid_cmd_base = "streamer -q -c %s -f rgb24 -t 01:00:00 -r %d -s %s " % (record_video_external, rate, local_args.record_external_video_size)
        logger.debug("Video file base to %s (e.g. %s_0.avi)" % (vid_file_base, vid_file_base))


    def sequence_select_from_episode(e_in: AttrDict, e_out: AttrDict):
        assert e_out.has_leaf_key("done")
        assert e_in.has_leaf_key(local_args.sequence_key)
        total_length = len(e_out.done)
        start_idx_valid_range = list(range(total_length - seq_len))
        start_idx = None
        i = 0
        logger.info("Press.. \'n\': go forward (looping)")
        logger.info("        \'p\': go back (looping)")
        logger.info("        \'s\': mark start idx")
        logger.info("      \'esc\': quit with no start idx (trash)")

        cv2.namedWindow("sequence_image", cv2.WINDOW_AUTOSIZE)
        while start_idx is None:
            cv2.imshow("sequence_image", e_in[local_args.sequence_key][i])  # i'th image
            ret = cv2.waitKey(0)
            print(i)
            if ret == ord('n'):  # n for next
                i = (i + 1) % total_length
            elif ret == ord('p'):  # p for prev
                i = (i - 1) % total_length
            elif ret == 27:  # esc
                start_idx = -1  # no save
            elif ret == ord('s'):  # s to save
                if i not in start_idx_valid_range:
                    logger.warn("[%d/%d] Not a valid start idx for seq len %d" % (i, total_length, seq_len))
                else:
                    start_idx = i  # save
            else:
                logger.warn("[%d/%d] Not a supported key: %s" % (i, total_length, chr(ret)))
        cv2.destroyWindow("sequence_image")

        if start_idx == -1:
            return start_idx, -1

        return start_idx, start_idx + seq_len


    def save_episode(e_in: AttrDict, e_out: AttrDict, extra_data: AttrDict):
        global latest_save, next_save_ep
        e_out.done[-1] = True  # end of episode
        dataset.add_episode(e_in, e_out)
        if local_args.save_every_ep:
            suffix = f'_ep{local_args.save_start_ep + next_save_ep}'
            dataset.save(extra_data, suffix=suffix, ep_range=(dataset.get_num_episodes() - 1, dataset.get_num_episodes()))
        else:
            dataset.save(extra_data)
        latest_save = len(dataset)
        next_save_ep = dataset.get_num_episodes()


    # gets called at the right time
    def extra_reset_fn(**kwargs):
        global episode_inputs, episode_outputs
        if episode_inputs is not None and len(episode_outputs >> "done") > 0:
            done = False
            while not done:
                ep_inputs = episode_inputs.leaf_apply(np.concatenate)
                ep_outputs = episode_outputs.leaf_apply(np.concatenate)
                # ep_inputs = episode_inputs.leaf_copy()
                # ep_outputs = episode_outputs.leaf_copy()
                ok_to_save = True  # modified by sequence selector, doesn't guarantee saving (that is do_save)
                if select_seq:
                    logger.info("Prompting user with full episode for pruning ...")
                    start_idx, end_idx = sequence_select_from_episode(ep_inputs, ep_outputs)
                    ok_to_save = start_idx != -1
                    if ok_to_save:
                        assert end_idx > start_idx, "%d must be greater than %d" % (end_idx, start_idx)
                        logger.info("Selected sequence [%d:%d]" % (start_idx, end_idx))
                        ep_inputs.leaf_modify(lambda arr: arr[start_idx:end_idx])
                        ep_outputs.leaf_modify(lambda arr: arr[start_idx:end_idx])

                if ok_to_save:
                    logger.info("UI: Save [y] or Trash [n]")
                    if hasattr(env, "populate_display_fn"):
                        env.populate_display_fn("UI: Save [y] or Trash [n]")
                    res = wait_for_keydown_from_set(input_handle, [KI('y', KI.ON.down), KI('n', KI.ON.down)], do_async=local_args.do_input_process)
                    do_save = res.key == 'y'
                else:
                    do_save = False

                ## save or trash
                if do_save:
                    logger.warn("Saving!")

                    if local_args.add_episode_labels:
                        if not extra_data.has_leaf_key("label"):
                            extra_data.label = []

                        label = query_string_from_set("[KB] episode label from %s?" % local_args.episode_label_set, local_args.episode_label_set, lower=False)
                        extra_data.label.append(np.array([label]))  # add the label to extra data

                    # merge all lists into stack or rows
                    concat_extra = extra_data.leaf_apply(lambda list_of_arrs: np.stack(list_of_arrs))

                    if local_args.episode_label_as_final:
                        save_episode(ep_inputs, ep_outputs & concat_extra, AttrDict())
                    else:
                        save_episode(ep_inputs, ep_outputs, concat_extra)
                else:
                    logger.warn("Trashing!")

                ## exit loop or repeat
                if select_seq:
                    logger.info("UI: Select another [y] or Done [n]")
                    select_again = wait_for_keydown_from_set(input_handle, [KI('y', KI.ON.down), KI('n', KI.ON.down)], do_async=local_args.do_input_process)
                    # do again
                    done = select_again.key == 'n'
                else:
                    done = True

            logger.debug("Done saving for this episode.")
        else:
            logger.warn("No data in latest episode!")


    ### warm start the planner
    presets = AttrDict(do_gravity_compensation=True)  # gravity compensation if it is an option
    obs, goal = env.user_input_reset(input_handle, extra_reset_fn, presets=presets)
    policy.reset_policy(next_obs=obs, next_goal=goal)
    policy.warm_start(model, obs, goal)
    logger.info("Starting episode %d" % episode_count)


    # if record_video_external is not None:
    #     tm = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    #     save_to_file = os.path.join(file_manager.exp_dir, vid_file_base + "_ep_%d_on_%s.avi" % (episode_count, tm))
    #     cmd = vid_cmd_base + "-o %s" % save_to_file
    #     logger.debug("Calling: " + cmd)
    #     proc = subprocess.Popen(shlex.split(cmd), shell=False)


    # this function gets called repeatedly
    def pygame_input_user_callback(handle: PygameOnlyKeysInput):
        global done, proc, episode_inputs, episode_outputs, episode_count, steps, start_time, running, obs, goal
        with timeit('pre_actions'):
            do_reset = done[0] or policy.is_terminated(model, obs, goal)
            key_states = handle.read_input()
            for key, on_states in key_states.items():
                do_reset = do_reset or (key == 'r' and KI.ON.down in on_states)

        if do_reset:
            handle.running = False
        else:
            new_inputs = obs.copy()
            new_inputs.combine(goal)
            # two empty axes for (batch_size, horizon)
            obs = obs.leaf_apply(lambda arr: arr[:, None])
            goal = goal.leaf_apply(lambda arr: arr[:, None])
            with timeit('policy'):
                with torch.no_grad():
                    action = policy.get_action(model, obs, goal, user_input_state=key_states)

            with timeit('step'):
                # STEP
                obs, goal, done = env.step(action)

            with timeit('post_actions'):
                action_np = action.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True))

                # get the inputs/outputs for this step
                new_inputs.combine(action_np)
                new_outputs = AttrDict()
                for key in env_spec.observation_names:
                    new_outputs["next_" + key] = obs[key]
                for key in env_spec.output_observation_names:
                    if key not in new_outputs.leaf_keys() and key in obs.leaf_keys():
                        new_outputs[key] = obs[key]

                new_outputs.done = done
                new_outputs.reward = np.zeros((1, 1))

                # add to running list
                if episode_inputs is None:
                    episode_inputs = new_inputs.leaf_apply(lambda arr: [arr])  # start the list
                    episode_outputs = new_outputs.leaf_apply(lambda arr: [arr])  # start the list
                else:
                    episode_inputs = AttrDict.leaf_combine_and_apply([episode_inputs, new_inputs], func=lambda vs: vs[0] + [vs[1]])
                    episode_outputs = AttrDict.leaf_combine_and_apply([episode_outputs, new_outputs], func=lambda vs: vs[0] + [vs[1]])

                steps += 1

        # if steps % np.ceil(5 / env.dt) == 0:
        #     logger.debug("[ep = %d] steps = %d, avg time per step = %f" % (episode_count, steps, (time.time() - start_time) / steps))


    # environment handles rate limiting, callback should step the environment
    timeit.reset()

    while running:
        # save video
        if record_video_external is not None:
            tm = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            save_to_file = os.path.join(file_manager.exp_dir, vid_file_base + "_ep_%d_on_%s.avi" % (episode_count, tm))
            cmd = vid_cmd_base + "-o %s" % save_to_file
            logger.debug("Calling: " + cmd)
            proc = subprocess.Popen(shlex.split(cmd), shell=False)

        # does one episode
        if not local_args.do_input_process:
            input_handle.run(dt=env.dt, user_callback=pygame_input_user_callback, rate_limited=False)
        else:
            do_reset = False
            done[0] = False
            while not do_reset and not done[0]:
                key_states = input_handle.read_input()
                for key, on_states in key_states.items():
                    do_reset = do_reset or (key == 'r' and KI.ON.down in on_states)

                new_inputs = obs.copy()
                new_inputs.combine(goal)
                # two empty axes for (batch_size, horizon)
                obs = obs.leaf_apply(lambda arr: arr[:, None])
                goal = goal.leaf_apply(lambda arr: arr[:, None])
                with torch.no_grad():
                    action = policy.get_action(model, obs, goal, user_input_state=key_states)

                # STEP
                obs, goal, done = env.step(action, )

                action_np = action.leaf_apply(lambda arr: to_numpy(arr, check=True))

                # get the inputs/outputs for this step
                new_inputs.combine(action_np)

                new_outputs = AttrDict()
                for key in env_spec.observation_names:
                    new_outputs["next_" + key] = obs[key]
                for key in env_spec.output_observation_names:
                    if key not in new_outputs.leaf_keys() and key in obs.leaf_keys():
                        new_outputs[key] = obs[key]

                new_outputs.done = done
                new_outputs.reward = np.zeros((1, 1))

                if episode_inputs is None:
                    episode_inputs = new_inputs.leaf_apply(lambda arr: [arr])  # start the list
                    episode_outputs = new_outputs.leaf_apply(lambda arr: [arr])  # start the list
                else:
                    episode_inputs = AttrDict.leaf_combine_and_apply([episode_inputs, new_inputs], func=lambda vs: vs[0] + [vs[1]])
                    episode_outputs = AttrDict.leaf_combine_and_apply([episode_outputs, new_outputs], func=lambda vs: vs[0] + [vs[1]])

                steps += 1

                if steps % np.ceil(5 / env.dt) == 0:
                    logger.debug("[ep = %d] steps = %d, avg time per step = %f" % (
                    episode_count, steps, (time.time() - start_time) / steps))

        logger.debug("Demonstration resetting...")
        steps = 0
        done[:] = False
        start_time = time.time()
        if proc is not None:
            proc.terminate()
            proc = None
        obs, goal = env.user_input_reset(input_handle, reset_action_fn=extra_reset_fn)
        policy.reset_policy(next_obs=obs, next_goal=goal)
        episode_count += 1
        logger.info("Starting episode %d" % episode_count)
        episode_inputs = None
        episode_outputs = None

    logger.warn("Ending demonstrations, saved: %d steps to %s" % (latest_save, dtparams.output_file))

