import imageio
import numpy as np

from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.experiments.grouped_parser import MacroEnabledArgumentParser
from sbrl.utils.file_utils import file_path_with_default_dir, postpend_to_base_name
from sbrl.utils.np_utils import np_split_dataset_by_key
from sbrl.utils.python_utils import AttrDict, exit_on_ctrl_c

parser = MacroEnabledArgumentParser()
parser.add_argument('--file', type=str, required=True, help="input file")
parser.add_argument('--img_key', type=str, default="image")
parser.add_argument('--done_key', type=str, default=None)  # if None, whole video will be saved
parser.add_argument('--ep_range', type=int, nargs=2, default=None, help="episodes to run, range")  # TODO
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[], help="episodes to run, [] for all. done must be specified")  # TODO
parser.add_argument('--save_file', type=str, default=None, help="Save video here, ")
parser.add_argument('--fps', type=int, default=10, help="frame rate for video saves")

parser.add_argument('--raw', action="store_true", help="will not convert BGR -> RGB")
parser.add_argument('--flatten', action="store_true", help="when specifying done key, choose to collapse all eps (similar to not specifying done key)")

parser.add_argument('--show', action="store_true", help="show videos")
# parser.add_argument('--avg', action="store_true", help="add avg line to plots")

args = parser.parse_args()

exit_on_ctrl_c()  # in case of infinite waiting
path = file_path_with_default_dir(args.file, FileManager.base_dir, expand_user=True)
logger.debug("File Path: %s" % path)

img_key = args.img_key
done_key = args.done_key
logger.debug("Image Name: %s" % img_key)
if done_key is not None:
    logger.debug("Episode Done Name: %s" % done_key)

data = AttrDict.from_dict(dict(np.load(path, allow_pickle=True)))
logger.debug(list(data.leaf_keys()))

to_save = {}
if args.done_key is not None:
    # split by key, then save each ep
    done = data >> done_key
    splits, data_ls, _ = np_split_dataset_by_key(data > [img_key], AttrDict(), done, complete=True)
    if args.ep_range is not None:
        if len(args.ep_range) == 1:
            args.ep_range = [0, args.ep_range[0]]
        assert args.ep_range[0] < args.ep_range[1] <= len(splits), f"Ep range is invalid: {args.ep_range}"
        ep_idxs = list(range(args.ep_range[0], args.ep_range[1]))
    else:
        ep_idxs = list(range(len(splits))) if len(args.ep_idxs) == 0 else args.ep_idxs
    
    if not args.flatten:
        for ep in ep_idxs:
            to_save[f"_{ep}"] = data_ls[ep] >> img_key
    else:
        # flatten into one "ep"
        to_save[''] = np.concatenate([data_ls[i] >> img_key for i in ep_idxs])

else:
    # all of them
    to_save[''] = data >> img_key

if args.save_file is not None:
    save_path = file_path_with_default_dir(args.save_file, FileManager.video_dir)
    for key, imgs in to_save.items():
        img_path = postpend_to_base_name(save_path, key)
        logger.debug("Saving video of length %d to file -> %s" % (len(imgs), img_path))
        # BGR -> RGB
        if not args.raw:
            imgs = np.flip(imgs, axis=-1)
        imageio.mimsave(img_path, imgs.astype(np.uint8), format='mp4', fps=args.fps)
#
# if args.show:
#     plt.show()
