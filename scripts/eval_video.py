"""
TODO
"""

import os

import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt, animation

from sbrl.experiments import logger
from sbrl.experiments.file_manager import ExperimentFileManager, FileManager
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, timeit
from sbrl.utils.script_utils import is_next_cycle, load_standard_ml_config

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str)
# eval keys
grouped_parser.add_argument('--max_steps', type=int, default=np.inf)
grouped_parser.add_argument('--log_every_n_eps', type=int, default=1)
grouped_parser.add_argument('--model_file', type=str)
grouped_parser.add_argument('--no_model_file', action="store_true")
grouped_parser.add_argument('--animate', action="store_true", help='If true, will show images through matplotlib animation, otherwise will save them to file')
grouped_parser.add_argument('--random_policy', action="store_true")
grouped_parser.add_argument('--print_last_obs', action="store_true")
grouped_parser.add_argument('--print_policy_name', action="store_true")
grouped_parser.add_argument('--draw_reward', action="store_true")
grouped_parser.add_argument('--draw_action_mode_key', type=str, default=None)
grouped_parser.add_argument('--track_returns', action="store_true")
grouped_parser.add_argument('--keep_return_above', type=float, default=None)
grouped_parser.add_argument('--keep_return_below', type=float, default=None)
# image / video keys
grouped_parser.add_argument('--image_key', type=str, default='image')
grouped_parser.add_argument('--fps', type=int, default=10)
grouped_parser.add_argument('--raw', action='store_true', help='if False, will flip images BGR -> RGB before saving')
grouped_parser.add_argument('--save_file', type=str, default=None)

local_args, unknown = grouped_parser.parse_local_args()
# this defines the required command line groups, and the defaults
# if in this list, we look for it

ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'policy']

config_fname = os.path.abspath(local_args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                 debug=False)

if os.path.exists(os.path.join(FileManager.base_dir, 'experiments', exp_name)):
    file_manager = ExperimentFileManager(exp_name, is_continue=True)
else:
    logger.warn(f"Experiment: 'experiments/{exp_name}' does not exist, using 'experiments/test' instead")
    file_manager = ExperimentFileManager('test', is_continue=True)


if args.model_file is not None:
    model_fname = args.model_file
    model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current eval: {}".format(model_fname))

env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)
env = params.env_train.cls(params.env_train.params, env_spec)
policy = params.policy.cls(params.policy.params, env_spec, env=env)

### warm start the planner
presets = AttrDict(do_gravity_compensation=True)
obs, goal = env.reset()
policy.reset_policy(next_obs=obs, next_goal=goal)
policy.warm_start(model, obs, goal)

img_key = local_args.image_key
img = obs >> img_key

### restore model
if not local_args.no_model_file:
    model.restore_from_file(model_fname)

### eval loop
done = [False]
rew = 0.
i = 0
steps = 0
eps = 0
returns = 0.

filter_returns = local_args.keep_return_above is not None or local_args.keep_return_below is not None
track_returns = filter_returns or local_args.track_returns
if filter_returns:
    keep_above = -np.inf if local_args.keep_return_above is None else local_args.keep_return_above
    keep_below = np.inf if local_args.keep_return_below is None else local_args.keep_return_below
    logger.debug(f"Will keep returns above {keep_above} and below {keep_below}")

if local_args.raw:
    postprocess = lambda x: x
else:
    postprocess = lambda x: np.flip(x, axis=-1)  # BGR -> RGB

get_reward_fn = lambda env: (obs >> 'reward').item()
if local_args.draw_reward:
    env.setup_draw_reward(get_reward_fn)

if local_args.animate:
    fig, ax = plt.subplots()
    im = plt.imshow(postprocess(img[0]), animated=True)

def step(*unused):
    global done, obs, goal, img, i, steps, eps, rew

    if done[0] or policy.is_terminated(model, obs, goal):
        rew = 0
        i = 0
        eps += 1
        obs, goal = env.reset()
        policy.reset_policy(next_obs=obs, next_goal=goal)

        if local_args.draw_reward:
            env.setup_draw_reward(get_reward_fn)

        if is_next_cycle(eps, local_args.log_every_n_eps):
            logger.debug(f"[{steps}]: Num episodes: {eps}")
            logger.debug(timeit)
            timeit.reset()

    # two empty axes for (batch_size, horizon)
    expanded_obs = obs.leaf_apply(lambda arr: arr[:, None])
    expanded_goal = goal.leaf_apply(lambda arr: arr[:, None])
    with torch.no_grad():
        if local_args.random_policy:
            action = policy.get_random_action(model, expanded_obs, expanded_goal)
        else:
            action = policy.get_action(model, expanded_obs, expanded_goal)

        if i == 0 and local_args.print_policy_name and action.has_leaf_key("policy_name"):
            logger.info(f"Policy: {action.policy_name.item()}")

    obs, goal, d = env.step(action)
    img = obs >> img_key

    # hacky
    if local_args.draw_action_mode_key is not None:
        mode = (action >> local_args.draw_action_mode_key).item()
        img = img.astype(np.uint8)
        # show the marker in magenta at the top...
        img[0] = cv2.putText(img[0], f"M = {mode}", (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 255))

    i += 1
    steps += 1

    if d and args.print_last_obs:
        logger.debug("Last obs:")
        obs.pprint()

    done[0] = d.item()
    if track_returns:
        rew = get_reward_fn(env)

    if local_args.animate:
        im.set_array(postprocess(img[0]))
        return im,

## start animation
if local_args.animate:
    assert not track_returns, "Returns cannot be filtered with animations!"
    if local_args.save_file is not None:
        logger.warn("Cannot save an animation! save_file will be ignored.")
    ani = animation.FuncAnimation(fig, step, interval=int(1000. / local_args.fps), frames=local_args.max_steps)

    plt.show()
else:
    save_path = file_path_with_default_dir(local_args.save_file, file_manager.exp_video_dir)
    logger.debug(f"Video will save to {save_path}")
    logger.debug(f"Beginning evaluation...")
    imgs = [img.copy()]
    dones = [False]
    returns = [0.]
    ep_steps = 0

    # save loop w/ tracking of returns.
    while steps < local_args.max_steps:
        step()
        imgs.append(img.copy())
        ep_steps += 1

        true_done = done[0] or policy.is_terminated(model, obs, goal)

        # termination
        if track_returns:
            returns[-1] += rew
            if true_done:
                logger.debug(f'After {len(returns)} Episodes: Return = {returns[-1]}, avg = {np.mean(returns)}, #>0 = {np.sum(np.asarray(returns) > 0)}')
                if filter_returns and (keep_above > returns[-1] or returns[-1] > keep_below):
                    logger.debug(f"Discarding episode.")
                    imgs = imgs[:-ep_steps]
                    returns.pop(-1)
                    steps -= ep_steps

                returns.append(0.)  # start fresh for next time

        if true_done:
            ep_steps = 0

    imgs = np.concatenate(imgs, axis=0)  # (H x ...)

    logger.debug(f"Images output shape: {imgs.shape}")

    if save_path is not None:
        logger.debug("Saving video of length %d, fps %d to file -> %s" % (len(imgs), local_args.fps, save_path))

        imgs = postprocess(imgs)

        imageio.mimsave(save_path, imgs.astype(np.uint8), format='mp4', fps=local_args.fps)

        logger.debug("Saved.")
    