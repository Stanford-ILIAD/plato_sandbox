import io
import queue
import threading
import warnings

import PIL
import PIL.Image
import numba
import numpy as np
import torch
from numba import NumbaPendingDeprecationWarning

from sbrl.experiments import logger
from sbrl.utils.cv_utils import cv2
from sbrl.utils.python_utils import AttrDict, timeit

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.dot(v1, v1) / np.dot(v2, v2))

def clip_norm(arr, norm, axis=None):
    return arr * np.minimum(norm / (np.linalg.norm(arr, axis=axis, keepdims=axis is not None) + 1e-11), 1)

def clip_scale(arr, scale_max):
    # scale down so all terms are less than max
    scale = np.maximum(np.max(np.abs(arr) / scale_max, axis=-1), 1)
    if len(arr.shape) > len(scale.shape):
        scale = scale[..., None]
    return arr / (scale + 1e-11)

def imrectify_fisheye(img, K, D, balance=0.0):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    dim = img.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    assert (len(shape) == 3)
    assert (shape[-1] == 1 or shape[-1] == 3)
    assert (image.shape[0] / image.shape[1] == shape[0] / shape[1]) # maintain aspect ratio
    height, width, channels = shape

    if len(image.shape) > 2 and image.shape[2] == 1:
        image = image[:,:,0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert (im.shape == tuple(shape))

    return im


def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


# Buffer-less VideoCapture -- courtesy of StackOverflow :: https://stackoverflow.com/questions/54460797/
class VideoCapture:
    def __init__(self, name, H=480, W=640):
        import cv2
        self.cap = cv2.VideoCapture(name)
        assert self.cap.isOpened(), "Webcam doesn't appear to be online!"

        # Set Width and Height so we get Webcam Full Resolution
        print('[*] Setting Proper WebCam Resolutions...')
        self.cap.set(3, W)
        self.cap.set(4, H)

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one!
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# @numba.jit(cache=True, nopython=True, parallel=True)
def np_pad_sequence(list_of_arr):
    first = list_of_arr[0]
    pad_sizes = np.array([arr.shape[0] for arr in list_of_arr])
    max_size = int(np.max(pad_sizes))
    # extra_sizes = max_size - pad_sizes

    shape = (len(list_of_arr), max_size, *first.shape[1:])
    # for i in range(len(shape)):
    #     shape[i] = int(shape[i])
    # print(type(shape))
    cat = np.zeros(shape, dtype=first.dtype)
    # pads = [(0, 0) for _ in range(len(list_of_arr[0].shape))]
    for i, arr in enumerate(list_of_arr):
        # pads[0] = (0, extra)
        # assign to output, sequentially
        # slices.append(slice(arr.shape[0]))

        cat[i, :arr.shape[0]] = arr

    return cat


@numba.jit(cache=True, nopython=True)
def np_pad_packed_sequence(arr, lengths):
    sequences = np.split(arr, np.cumsum(lengths)[:-1].astype(int), axis=0)
    return np_pad_sequence(sequences)

def np_concat_if_same_size(list_of_arr, axis=0):
    # TODO doesn't work
    shapes = [list(arr.shape[:axis]) + list(arr.shape[axis+1:]) for arr in list_of_arr]
    main_sh = None
    matching = True
    for sh in shapes:
        if main_sh is None:
            main_sh = sh
        elif sh != main_sh:
            matching = False
            break

    if matching:
        return np.concatenate(list_of_arr, axis=0)
    else:
        return np.array(list_of_arr, dtype=np.object)


def not_idxs(idxs, total_len):
    idxs = np.asarray(idxs)
    assert len(idxs.shape) == 1
    np_where = np.ones((total_len,), dtype=bool)
    np_where[idxs] = False

    return np.arange(total_len)[np_where]


def np_add_to_buffer(buffer, new, max_len=0, end=True):
    assert list(buffer.shape[1:]) == list(new.shape), "Shapes do not match: %s should be %s" % (new.shape, buffer.shape[:1])
    max_len = buffer.shape[0] + 1 if max_len <= 0 else max_len
    if end:
        buffer = np.concatenate([buffer, new[None]], axis=0)[-max_len:]
    else:
        buffer = np.concatenate([new[None], buffer], axis=0)[:max_len]
    return buffer


# TODO try jit
def np_idx_range_between(start_idxs, end_idxs, length, spill_end_idxs=None, ret_idxs=True):
    # start, end, and spill_end are inclusive.
    # end is the last sampling start, spill end is the last idx that we can possible sample from (just in case |range| < L)
    if spill_end_idxs is None:
        spill_end_idxs = end_idxs

    assert length > 0, length
    #
    endp1_idxs = end_idxs + 1
    # should not be less than start_idxs (e.g, if 0, 49, L=30, last_start = 20)
    last_sample_start = np.maximum(endp1_idxs - length, start_idxs)
    sampled_start = np.random.randint(start_idxs, last_sample_start + 1)

    if ret_idxs:
        # N x L corresponding to sampled ranges, truncated if necessary.
        return np.minimum(sampled_start[:, None] + np.arange(length)[None], spill_end_idxs[:, None])
    else:
        sampled_end = np.minimum(sampled_start + length, spill_end_idxs)  # can't go past spill_end
        return sampled_start, sampled_end

# TODO implement fully
class DynamicNpArrayWrapper:
    def __init__(self, shape, capacity=0, dtype=np.float32):
        self._capacity = capacity
        self._size = 0
        self._data = np.empty((capacity, *list(shape)), dtype=dtype)

    def dynamic_add(self, x):
        if self._size == self._capacity:
            self._capacity *= 4
            newdata = np.zeros((self._capacity,))
            newdata[:self._size] = self._data
            self._data = newdata

        self._data[self._size] = x
        self._size += 1


def np_split_dataset_by_key(data: AttrDict, onetime_data: AttrDict, done_arr: np.ndarray, complete=True):
    assert len(done_arr.shape) == 1
    done_arr = done_arr.astype(bool)
    data.leaf_assert(lambda arr: arr.shape[0] == done_arr.shape[0])
    if not complete:
        last_true = np.argwhere(done_arr[::-1])[0]
        logger.debug(f"Cutting length {len(done_arr)} down to {last_true + 1} elements")
        done_arr = done_arr[:last_true + 1]
        data = data.leaf_modify(lambda arr: arr[:last_true + 1])

    assert done_arr[-1], "Last value must be true"
    last_idxs_per_ep = np.flatnonzero(done_arr)
    # boundaries for splitting
    splits = last_idxs_per_ep + 1
    onetime_data.leaf_assert(lambda arr: arr.shape[0] == len(splits))
    data_ep_tup = data.leaf_apply(lambda arr: np.split(arr, splits[:-1], axis=0))
    onetime_data_ep_tup = onetime_data.leaf_apply(lambda arr: np.split(arr, len(splits), axis=0))
    data_ep = []
    onetime_data_ep = []
    for ep in range(len(splits)):
        data_ep.append(data_ep_tup.leaf_apply(lambda vs: vs[ep]))
        onetime_data_ep.append(onetime_data_ep_tup.leaf_apply(lambda vs: vs[ep]))
    return splits, data_ep, onetime_data_ep


if __name__ == '__main__':
    lengths = np.random.randint(20, 100, (1024,))
    sumH = lengths.sum()
    big_arr = np.zeros((sumH, 15))
    sequences = np.split(big_arr, np.cumsum(lengths)[:-1].astype(int), axis=0)

    timeit.reset()
    with timeit("loop"):
        for i in range(500):
            padded = np_pad_sequence(sequences)

    from torch.nn.utils.rnn import pad_sequence
    torch_arr = torch.from_numpy(big_arr)
    torch_seqs = torch.split_with_sizes(torch_arr, lengths.tolist())

    print(padded.shape)
    print(timeit)

    timeit.reset()
    with timeit("torch_loop"):
        for i in range(500):
            padded = pad_sequence(torch_seqs)

    print(padded.shape)
    print(timeit)

