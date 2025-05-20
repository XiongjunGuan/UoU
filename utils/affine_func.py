"""
Description:
Author: Xiongjun Guan
Date: 2024-06-13 10:31:54
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-07-09 15:23:52

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as sndi


def fliplr_mnt(arr, w):
    new_arr = np.zeros_like(arr)
    new_arr[:, 0] = w - arr[:, 0]
    new_arr[:, 1] = arr[:, 1]
    new_arr[:, 2] = (180 - arr[:, 2]) % 360
    return new_arr


def rotate_points(points, center, angle_deg):
    """
    Rotate a set of 2D points around a given center by a specified angle.

    Parameters:
        points (np.ndarray): Array of shape (N, 2), where N is the number of points.
        center (tuple): The rotation center as (xc, yc).
        angle_deg (float): The rotation angle in degrees.

    Returns:
        np.ndarray: Array of rotated points with shape (N, 2).
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ])

    # Shift points to the origin (relative to center)
    shifted_points = points - np.array(center)

    # Apply rotation
    rotated_shifted_points = np.dot(shifted_points, rotation_matrix.T)

    # Shift points back to the original center
    rotated_points = rotated_shifted_points + np.array(center)

    return rotated_points


def affine_mnt(mnt, dx, dy, dtheta, h, w):
    mnt[:, 0] += dx
    mnt[:, 1] += dy
    mnt[:, :2] = rotate_points(mnt[:, :2], (h // 2, w // 2), -dtheta)
    mnt[:, 2] = (mnt[:, 2] - dtheta) % 360

    valid_indices = ((mnt[:, 0] > 0) * (mnt[:, 0] < w) * (mnt[:, 1] > 0) *
                     (mnt[:, 1] < h))
    mnt = mnt[valid_indices]
    return mnt


def select_mask_mnt(points, mask):
    # Ensure points array is not empty
    if points.shape[0] == 0:
        return points  # Return empty array if input points are empty

    # Extract x and y coordinates of the points
    x_coords, y_coords = points[:, 0], points[:, 1]

    # Check if the mask value at each point's location is 1
    valid_indices = mask[y_coords, x_coords] == 1

    # Filter points based on valid indices
    filtered_points = points[valid_indices]

    return filtered_points


def affine_img(img, dx, dy, dtheta, pad_width=0, fit_value=255):
    """translation -> rotation

    Args:
        img (_type_): _description_
        dx (_type_): col pixel
        dy (_type_): row pixel
        theta (_type_): degree
        pad_width (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if pad_width > 0:
        img = np.pad(
            img,
            [[pad_width, pad_width], [pad_width, pad_width]],
            "constant",
            constant_values=fit_value,
        )

    h, w = img.shape[:2]

    # translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img,
                         M, (w, h),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=fit_value)

    # rotation
    center = (h // 2, w // 2)
    M = cv2.getRotationMatrix2D(center, dtheta, 1.0)
    img = cv2.warpAffine(img,
                         M, (w, h),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=fit_value)
    if pad_width > 0:
        img = img[pad_width:-pad_width, pad_width:-pad_width]
    return img
