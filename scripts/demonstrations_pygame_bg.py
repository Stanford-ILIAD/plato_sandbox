"""
This file is for collecting demonstrations (no training). similar to eval but we use train dataset
"""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import threading

import numpy as np
import pygame
import torch

from sbrl.datasets.np_dataset import NpDataset
from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager
from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.file_utils import import_config
from sbrl.utils.input_utils import KeyInput as KI, wait_for_keydown_from_set
from sbrl.utils.pygame_utils import PygameOnlyKeysInput
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict
from sbrl.utils.torch_utils import to_numpy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--file', type=str, nargs="*", default=[], help="1 or more input files")
    parser.add_argument('--record_external_video', type=str, default=None)
    parser.add_argument('--video_file_base', type=str, default="vid_output")
    parser.add_argument('--output_file', type=str, default="demonstrations.npz")
    parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--use_teleop_policy', action="store_true", help="False = use demonstration policy in params. Otherwise runs BasicPolicy")
    args = parser.parse_args()

    exit_on_ctrl_c()  # in case of infinite waiting

    config_fname = os.path.abspath(args.config)
    assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

    params = import_config(config_fname)

    file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

    if args.model is not None:
        model_fname = os.path.abspath(args.model)
        assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
        logger.debug("Using model: {}".format(model_fname))
    elif "checkpoint_model_file" in params.trainer.params and params.trainer.params['checkpoint_model_file'] is not None:
        model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
        logger.debug("Using checkpoint model for current eval: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))

    env_spec = params.env_spec.cls(params.env_spec.params)
    env = params.env.cls(params.env.params, env_spec)
    model = params.model.cls(params.model.params, env_spec, None)

    ### PYGAME
    if not pygame.get_init():
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((400, 300))
        screen.fill((255,255,255))
        pygame.display.set_caption("Demonstrations")
        myfont = pygame.font.SysFont('Arial', 20)
        textline1 = myfont.render('In window, press..', False, (0, 0, 0))
        textline_y = myfont.render('y: yes', False, (0, 0, 0))
        textline_n = myfont.render('n: no', False, (0, 0, 0))
        textline_r = myfont.render('r: reset', False, (0, 0, 0))
        textline_q = myfont.render('q: quit', False, (0, 0, 0))
        screen.blit(textline1, (10,10))
        screen.blit(textline_y, (10,50))
        screen.blit(textline_n, (10,90))
        screen.blit(textline_r, (10,130))
        screen.blit(textline_q, (10,170))
        pygame.display.flip()

    running = True
    def quit(*args):
        global running
        running = False
        sys.exit(0)

    ### handling user input
    empty_handler = lambda ui, ki: None
    input_handle = PygameOnlyKeysInput(AttrDict(), {})
    input_handle.register_callback(KI("r", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("y", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("n", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("q", KI.ON.down), quit)

    if not args.use_teleop_policy:
        ## demonstration policy
        assert "demonstration_policy" in params.keys()
        ppolicy = params.demonstration_policy
    else:
        ## teleop policy
        ppolicy = params.policy
        assert hasattr(env, "get_default_teleop_model_forward_fn"), "Need to implement \"get_default_teleop_model_forward_fn\" in order to run demonstration collection"
        ppolicy.params.mem_policy_model_forward_fn = env.get_default_teleop_model_forward_fn(input_handle)  # teleop control loop
        ppolicy.cls = BasicPolicy

    policy = ppolicy.cls(ppolicy.params, env_spec, file_manager=file_manager)

    ## demonstrations (must be Np dataset for now)
    dtparams = params.dataset_train.params.copy()
    assert params.dataset_train.cls == NpDataset, params.dataset_train.cls
    if len(args.file) > 0:  # user provided files, otherwise empty
        dtparams.file = args.file
    else:
        dtparams.file = None

    dtparams.output_file = args.output_file

    dataset_train = params.dataset_train.cls(dtparams, env_spec, file_manager)

    ### restore model
    if not args.no_model_file:
        model.restore_from_file(model_fname)

    # bg_thread = threading.Thread(target=lambda: None, daemon=True)
    bg_thread = threading.Thread(target=input_handle.run, daemon=True)  # visualization is slow
    bg_thread.start()  # reads

    ### eval loop
    done = [False]
    steps = 0
    episode_count = 1
    latest_save = 0
    episode_inputs = None
    episode_outputs = None

    record_video_external = args.record_external_video
    vid_cmd_base = None
    vid_file_base = args.video_file_base
    proc = None
    if record_video_external is not None:
        vid_cmd_base = "streamer -q -c %s -f rgb24 -t 01:00:00 -r 3 " % record_video_external
        logger.debug("Video file base to %s (e.g. %s_0.avi)" % (vid_file_base, vid_file_base))


    # gets called at the right time
    def extra_reset_fn(**kwargs):
        if episode_inputs is not None and len(episode_outputs.done) > 0:
            episode_outputs.done[-1] = True  # end of episode
            logger.info("UI: Save [y] or Trash [n]")
            res = wait_for_keydown_from_set(input_handle, [KI('y', KI.ON.down), KI('n', KI.ON.down)])
            if res.key == 'y':
                dataset_train.add_episode(episode_inputs, episode_outputs)
                dataset_train.save()
                latest_save = len(dataset_train)
            else:
                logger.warn("Trashing!")
        else:
            logger.warn("No data in latest episode!")


    ### warm start the planner
    presets = AttrDict(do_gravity_compensation=True)  # gravity compensation if it is an option
    obs, goal = env.user_input_reset(input_handle, extra_reset_fn, presets=presets)
    policy.reset_policy(next_obs=obs, next_goal=goal)
    policy.warm_start(model, obs, goal)
    logger.info("Starting episode %d" % episode_count)

    if record_video_external is not None:
        tm = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        save_to_file = os.path.join(file_manager.exp_dir, vid_file_base + "_ep_%d_on_%s.avi" % (episode_count, tm))
        cmd = vid_cmd_base + "-o %s" % save_to_file
        logger.debug("Calling: " + cmd)
        proc = subprocess.Popen(shlex.split(cmd), shell=False)

    while running:
        do_reset = done[0]
        key_states = input_handle.read_input()
        for key, on_states in key_states.items():
            do_reset = do_reset or (key == 'r' and KI.ON.down in on_states)

        if do_reset:
            logger.debug("Demonstration resetting...")
            steps = 0
            if proc is not None:
                proc.terminate()
                proc = None
            obs, goal = env.user_input_reset(input_handle, reset_action_fn=extra_reset_fn)
            policy.reset_policy(next_obs=obs, next_goal=goal)
            episode_count += 1
            logger.info("Starting episode %d" % episode_count)
            episode_inputs = None
            episode_outputs = None

            if record_video_external is not None:
                tm = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
                save_to_file = os.path.join(file_manager.exp_dir, vid_file_base + "_ep_%d_on_%s.avi" % (episode_count, tm))
                cmd = vid_cmd_base + "-o %s" % save_to_file
                logger.debug("Calling: " + cmd)
                proc = subprocess.Popen(shlex.split(cmd), shell=False)

        new_inputs = obs.copy()
        new_inputs.combine(goal)
        # two empty axes for (batch_size, horizon)
        obs = obs.leaf_apply(lambda arr: arr[:, None])
        goal = goal.leaf_apply(lambda arr: arr[:, None])
        with torch.no_grad():
            action = policy.get_action(model, obs, goal, user_input_state=key_states)

        # STEP
        obs, goal, done = env.step(action, )

        # get the inputs/outputs for this step
        new_inputs.combine(action.leaf_apply(lambda arr: to_numpy(arr, check=True)))
        new_outputs = AttrDict()
        for key in env_spec.observation_names:
            new_outputs["next_" + key] = obs[key]
        for key in env_spec.output_observation_names:
            if key not in new_outputs.leaf_keys() and key in obs.leaf_keys():
                new_outputs[key] = obs[key]

        new_outputs.done = done
        new_outputs.reward = np.zeros((1,1))

        # add to running list
        if episode_inputs is None:
            episode_inputs = new_inputs
            episode_outputs = new_outputs
        else:
            episode_inputs = AttrDict.leaf_combine_and_apply([episode_inputs, new_inputs], func=lambda vs: np.concatenate([vs[0], vs[1]], axis=0))
            episode_outputs = AttrDict.leaf_combine_and_apply([episode_outputs, new_outputs], func=lambda vs: np.concatenate([vs[0], vs[1]], axis=0))

        steps += 1

    logger.warn("Ending demonstrations, saved: %d steps to %s" % (latest_save, dtparams.output_file))
    input_handle.end()
    bg_thread.join()
