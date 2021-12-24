
import os
import sys
import math
import numpy as np





def append_default_keys_dict(default_dict, dest_dict):
    if not dest_dict:
        return default_dict
    for key in default_dict.keys():
        if key not in dest_dict.keys():
            dest_dict[key] = default_dict[key]
    return dest_dict


def check_key_in_dict(dictionary, keys):
    for key in keys:
        if key not in dictionary.keys():
            raise ValueError("{} must be defined".format(key))


def preprocess_paths(paths):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None




def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    return [transcript.decode(encoding) for transcript in array]


def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] < window_size:
            slice_ = np.pad(
                slice_, (0, window_size - slice_.shape[0]), 'constant', constant_values=0.0)
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)



def merge_slices_numpy(slices: np.ndarray):
    # slices shape = [batch, window_size]
    return np.reshape(slices, [-1])


def get_num_batches(samples, batch_size):
    return math.ceil(float(samples) / float(batch_size))




def print_test_info(*args, **kwargs):
    print_one_line("[Test] Batches: ", kwargs["batches"], ", ", *args)





