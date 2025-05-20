'''
Description: 
Author: Xiongjun Guan
Date: 2025-05-15 14:09:56
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-05-19 16:17:33

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np


def norm_vf_mnt(arr, h, w):
    if arr.shape[0] == 0:
        return np.zeros((0, 4))

    new_arr = np.zeros((arr.shape[0], 4))
    new_arr[:, 0] = arr[:, 0] / w  # x -> (0,1)
    new_arr[:, 1] = arr[:, 1] / h  # y -> (0,1)
    angles = (arr[:, 2] % 360) / 360  # theta [0, 360] -> (0,1)
    new_arr[:, 2] = (np.sin(np.deg2rad(angles)) + 1) / 2  # sin -> (0,1)
    new_arr[:, 3] = (np.cos(np.deg2rad(angles)) + 1) / 2  # cos -> (0,1)
    return new_arr


def norm_img_sz(array, target_shape, padding_value=255):
    """
    Resize a 2D array to the target shape. If the array is larger, crop it centrally.
    If the array is smaller, pad it centrally with a specified padding value.

    Parameters:
        array (np.ndarray): Input 2D array.
        target_shape (tuple): Target shape as (rows, cols).
        padding_value (int, float): Value to use for padding if the array is smaller.

    Returns:
        np.ndarray: Resized array with the target shape.
    """
    # Get the input array shape
    input_rows, input_cols = array.shape
    target_rows, target_cols = target_shape

    fit_row = (target_rows - input_rows) // 2
    fit_col = (target_cols - input_cols) // 2

    # Initialize the output array
    output_array = None

    # Case 1: Crop if the array is larger than target
    if input_rows > target_rows or input_cols > target_cols:
        start_row = max((input_rows - target_rows) // 2, 0)
        start_col = max((input_cols - target_cols) // 2, 0)
        end_row = start_row + target_rows
        end_col = start_col + target_cols
        output_array = array[start_row:end_row, start_col:end_col]

    # ---
    array = output_array
    input_rows, input_cols = array.shape
    target_rows, target_cols = target_shape

    # Case 2: Pad if the array is smaller than target
    if input_rows < target_rows or input_cols < target_cols:
        pad_top = (target_rows - input_rows) // 2
        pad_bottom = target_rows - input_rows - pad_top
        pad_left = (target_cols - input_cols) // 2
        pad_right = target_cols - input_cols - pad_left

        output_array = np.pad(
            array,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=padding_value,
        )

    # Case 3: If the array matches the target shape, return it as is
    else:
        output_array = array

    return output_array, fit_row, fit_col
