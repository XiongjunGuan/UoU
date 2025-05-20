import numpy as np


def pts_normalization(arr):
    if len(arr):
        arr = arr[None] if arr.ndim == 1 else arr
    return arr


def load_minutiae_complete(fname, return_header=False):
    num_core = np.loadtxt(fname, skiprows=2, max_rows=1).astype(int)
    num_delta = np.loadtxt(fname, skiprows=3, max_rows=1).astype(int)
    num_minu = np.loadtxt(fname, skiprows=4, max_rows=1).astype(int)

    if num_core:
        core_arr = np.loadtxt(fname, skiprows=5, max_rows=num_core)
        core_arr = pts_normalization(core_arr)
    else:
        core_arr = np.zeros((0, 4))
    if num_delta:
        delta_arr = np.loadtxt(fname, skiprows=5 + num_core, max_rows=num_delta)
        delta_arr = pts_normalization(delta_arr)
    else:
        delta_arr = np.zeros((0, 6))
    if num_minu:
        mnt_arr = np.loadtxt(fname, skiprows=5 + num_core + num_delta)
        mnt_arr = pts_normalization(mnt_arr)
    else:
        mnt_arr = np.zeros((0, 4))

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return core_arr, delta_arr, mnt_arr, header
    else:
        return core_arr, delta_arr, mnt_arr


def load_singular(fname, return_header=False):
    num_core = np.loadtxt(fname, skiprows=2, max_rows=1).astype(int)
    num_delta = np.loadtxt(fname, skiprows=3, max_rows=1).astype(int)
    core_arr = np.loadtxt(fname, skiprows=5, max_rows=num_core)
    delta_arr = np.loadtxt(fname, skiprows=5 + num_core, max_rows=num_delta)

    core_arr = pts_normalization(core_arr)
    delta_arr = pts_normalization(delta_arr)

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return core_arr, delta_arr, header
    else:
        return core_arr, delta_arr


def load_minutiae(fname, return_header=False):
    """load minutiae file

    Parameters:
        [None]
    Returns:
        mnt_array[, img_size(width, height)]
    """
    try:
        num_sp = np.loadtxt(fname, skiprows=2, max_rows=2)
        mnt_arr = np.loadtxt(fname, skiprows=5 + num_sp.sum().astype(int))

        mnt_arr = pts_normalization(mnt_arr)

        if return_header:
            header = np.loadtxt(fname, max_rows=2).astype(int)
            return mnt_arr, header
        else:
            return mnt_arr
    except:
        if return_header:
            return None, None
        else:
            return None
