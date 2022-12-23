import argparse
import os

from moviepy.editor import ImageSequenceClip
from sbrl.datasets.torch_dataset import TorchDataset

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager, ExperimentFileManager
from sbrl.utils.config_utils import register_config_args
from sbrl.utils.file_utils import import_config, file_path_with_default_dir
from sbrl.utils.python_utils import exit_on_ctrl_c, AttrDict, get_with_default

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--horizon', type=int, default=0)  # if 0, use data train's horizon
parser.add_argument('--img_key', type=str, default="image")  # if 0, use data train's horizon
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to run, [] for all")  # TODO
parser.add_argument('--max_eps', type=int, default=0)  # TODO
parser.add_argument('--gif_dir', type=str, required=True)  # TODO
parser.add_argument('--gif_prefix', type=str, default="episode_")  # TODO
parser.add_argument('--fps', type=int, default=10)  # TODO



args, unknown = parser.parse_known_args()

if len(unknown) > 0:
    register_config_args(unknown)

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

params = import_config(config_fname)

file_manager = ExperimentFileManager(params.exp_name, is_continue=True)

exit_on_ctrl_c()
env_spec = params.env_spec.cls(params.env_spec.params)

if args.horizon > 0:
    HORIZON = args.horizon
    logger.warn("Overriding horizon in dset with %d" % HORIZON)
else:
    HORIZON = get_with_default(params.dataset_train.params, "horizon", 0)
    logger.info(f"Horizon = {HORIZON}")

img_key = args.img_key

assert HORIZON > 0
params.dataset_train.params.file = args.file
params.dataset_train.params.horizon = HORIZON
dataset_input = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
# use underlying dataset
if isinstance(dataset_input, TorchDataset):
    dataset_input = dataset_input.base_dataset

# params.dataset_train.params.file = None
# params.dataset_train.params.output_file = args.output_file
# dataset_output = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)

plt_dir = file_path_with_default_dir(args.gif_dir, FileManager.plot_dir, expand_user=True)
os.makedirs(plt_dir, exist_ok=True)

logger.debug(f"Plotting to {plt_dir}")

all_names = env_spec.all_names + ["done"]
num_eps = len(dataset_input.split_indices())

assert num_eps != 0, "No data to run"
assert all([args.ep_idxs[i] < num_eps for i in range(len(args.ep_idxs))]), "Bad idxs: %s" % args.ep_idxs

episodes = args.ep_idxs if len(args.ep_idxs) > 0 else list(range(num_eps))

# ### handling user input
# empty_handler = lambda ui, ki: None
# input_handle = PygameOnlyKeysInput(AttrDict(), {})
# input_handle.register_callback(KI("r", KI.ON.down), empty_handler)
# input_handle.register_callback(KI("y", KI.ON.down), empty_handler)
# input_handle.register_callback(KI("n", KI.ON.down), empty_handler)
# input_handle.register_callback(KI("q", KI.ON.down), lambda ui, ki: sys.exit(0))

for i in range(dataset_input.get_num_episodes()):
    if i >= args.max_eps > 0:
        break

    # (N x ...) where N is episode length
    # inps, outs = dataset_input.get_batch()
    # datadict = inps.leaf_apply(lambda arr: arr[0])
    # ep_len = outs.done.shape[0]
    datadict: AttrDict = dataset_input.get_episode(i, [img_key])
    # datadict, _ = dataset_input.get_batch(np.asarray([i]))
    ep_len = dataset_input.episode_length(i)
    # save first whole batch item
    images = (datadict >> img_key)
    logger.debug("Saving episode %d of length=%d, dlen=%d" % (i, ep_len, images.shape[0]))

    clip = ImageSequenceClip(list(images), fps=args.fps)

    clip.write_gif(os.path.join(plt_dir, args.gif_prefix + f"{i}.gif"), fps=args.fps)
