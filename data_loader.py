"""
Description:
Author: Xiongjun Guan
Date: 2025-05-15
version: 0.0.1
"""

import copy
import logging
import random

import cv2
import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset

from utils.affine_func import (affine_img, affine_mnt, fliplr_mnt,
                               select_mask_mnt)
from utils.minutiae_func import load_minutiae_complete
from utils.norm_func import norm_img_sz, norm_vf_mnt
from utils.visual_func import draw_minutia_on_finger


class load_dataset_train(Dataset):

    def __init__(
        self,
        fp_lst: None,
        mnt_lst: None,
        img_sz=512,
        apply_aug=False,
        trans_aug=50,
        rot_aug=180,
    ):
        self.fp_lst = fp_lst
        self.mnt_lst = mnt_lst
        self.img_sz = img_sz
        self.apply_aug = apply_aug
        self.trans_aug = trans_aug
        self.rot_aug = rot_aug

    def __len__(self):
        return len(self.fp_lst)

    def __getitem__(self, idx):
        # --- load img
        fp_path = self.fp_lst[idx]
        img = cv2.imread(fp_path, 0).astype(np.float32)

        # --- load mnt
        mnt_path = self.mnt_lst[idx]
        core_arr, delta_arr, mnt_arr = load_minutiae_complete(mnt_path)
        core_arr = core_arr[:, 0:3]
        delta_arr = delta_arr[:, 0:3]
        mnt_arr = mnt_arr[:, 0:3]

        # --- set img size
        img, dy, dx = norm_img_sz(img, self.img_sz)
        mask = np.ones_like(img)
        core_arr[:, 0] += dx
        core_arr[:, 1] += dy
        delta_arr[:, 0] += dx
        delta_arr[:, 1] += dy
        mnt_arr[:, 0] += dx
        mnt_arr[:, 1] += dy

        # --- data augmentation
        if self.apply_aug:
            # --- fliplr
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
                core_arr = fliplr_mnt(core_arr, img.shape[1])
                delta_arr = fliplr_mnt(delta_arr, img.shape[1])
                mnt_arr = fliplr_mnt(mnt_arr, img.shape[1])

            # --- translation and rotation
            dx = random.randint(-self.trans_aug, self.trans_aug)
            dy = random.randint(-self.trans_aug, self.trans_aug)
            dtheta = random.randint(-self.rot_aug, self.rot_aug)

            img = affine_img(img, dx, dy, dtheta, pad_width=0, fit_value=255)
            mask = affine_img(mask, dx, dy, dtheta, pad_width=0, fit_value=0)
            mask = mask > 0.5

            h, w = img.shape
            core_arr = affine_mnt(core_arr, dx, dy, dtheta, h,
                                  w).astype(np.int32)
            delta_arr = affine_mnt(delta_arr, dx, dy, dtheta, h,
                                   w).astype(np.int32)
            mnt_arr = affine_mnt(mnt_arr, dx, dy, dtheta, h,
                                 w).astype(np.int32)

            # --- mask some area
            if np.random.rand() < 0.2:
                rect_yc = random.randint(h // 5, 4 * h // 5)
                rect_xc = random.randint(w // 5, 4 * w // 5)
                rect_h = random.randint(100, 180)
                rect_w = random.randint(100, 180)
                rect_y1 = max(0, rect_yc - rect_h // 2)
                rect_x1 = max(0, rect_xc - rect_w // 2)
                rect_y2 = min(h, rect_yc + rect_h // 2)
                rect_x2 = min(w, rect_xc + rect_w // 2)
                mask[rect_y1:rect_y2, rect_x1:rect_x2] = 0
                img[mask == 0] = 255
                core_arr = select_mask_mnt(core_arr, mask)
                delta_arr = select_mask_mnt(delta_arr, mask)
                mnt_arr = select_mask_mnt(mnt_arr, mask)

        # --- show
        save_path = "/data/guanxiongjun/UniFiNet/tmp/show.png"
        draw_minutia_on_finger(img, mnt_arr, save_path=save_path)

        img = (255.0 - img) / 255.0  # 0 to background
        img = img[None, :, :]

        core_arr = norm_vf_mnt(core_arr)
        delta_arr = norm_vf_mnt(delta_arr)
        mnt_arr = norm_vf_mnt(mnt_arr)

        # (x,y,theta,sin,cos) -> (0,1)
        pts_coord = np.vstack((core_arr, delta_arr, mnt_arr))
        # [0,1,2,2,...,2]
        pts_cls = np.concatenate((
            np.full(core_arr.shape[0], 0),
            np.full(delta_arr.shape[0], 1),
            np.full(mnt_arr.shape[0], 2),
        ))

        target = {}
        target["boxes"] = pts_coord
        target["labels"] = pts_cls

        return img, target


def get_dataloader_train(
    fp_lst: None,
    mnt_lst: None,
    img_sz=512,
    apply_aug=False,
    trans_aug=50,
    rot_aug=180,
    batch_size=1,
    shuffle=True,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            fp_lst=fp_lst,
            mnt_lst=mnt_lst,
            img_sz=img_sz,
            apply_aug=apply_aug,
            trans_aug=trans_aug,
            rot_aug=rot_aug,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )
    logging.info(f"n_train:{len(dataset)}")

    return train_loader


def get_dataloader_valid(
    fp_lst: None,
    mnt_lst: None,
    img_sz=512,
    apply_aug=False,
    trans_aug=50,
    rot_aug=180,
    batch_size=1,
    shuffle=False,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            fp_lst=fp_lst,
            mnt_lst=mnt_lst,
            img_sz=img_sz,
            apply_aug=apply_aug,
            trans_aug=trans_aug,
            rot_aug=rot_aug,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )
    logging.info(f"n_valid:{len(dataset)}")

    return valid_loader


if __name__ == "__main__":
    img_path = "/data/panzhiyu/fingerprint/FVC04DB1/image/query/15_2.tif"
    mnt_path = "/data/panzhiyu/fingerprint/FVC04DB1/mnt/query/15_2.mnt"

    img_sz = (512, 512)

    img = cv2.imread(img_path, 0)

    core_arr, delta_arr, mnt_arr = load_minutiae_complete(mnt_path)
    core_arr = core_arr[:, 0:3]
    delta_arr = delta_arr[:, 0:3]
    mnt_arr = mnt_arr[:, 0:3]

    save_path = "/data/guanxiongjun/UniFiNet/tmp/0.png"
    draw_minutia_on_finger(img, mnt_arr, save_path=save_path)

    # --- set img size
    img, dy, dx = norm_img_sz(img, img_sz)
    mask = np.ones_like(img)

    core_arr[:, 0] += dx
    core_arr[:, 1] += dy
    delta_arr[:, 0] += dx
    delta_arr[:, 1] += dy
    mnt_arr[:, 0] += dx
    mnt_arr[:, 1] += dy

    save_path = "/data/guanxiongjun/UniFiNet/tmp/1.png"
    draw_minutia_on_finger(img, mnt_arr, save_path=save_path)

    # --- fliplr
    img = np.fliplr(img)
    mnt_arr = fliplr_mnt(mnt_arr, img.shape[1])

    save_path = "/data/guanxiongjun/UniFiNet/tmp/2.png"
    draw_minutia_on_finger(img, mnt_arr, save_path=save_path)

    # --- translation and rotation
    dx = 0
    dy = 0
    dtheta = 0
    img = affine_img(img, dx, dy, dtheta, pad_width=0, fit_value=255)
    mask = affine_img(mask, dx, dy, dtheta, pad_width=0, fit_value=0)
    mask = mask > 0.5
    h, w = img.shape
    core_arr = affine_mnt(core_arr, dx, dy, dtheta, h, w).astype(np.int32)
    delta_arr = affine_mnt(delta_arr, dx, dy, dtheta, h, w).astype(np.int32)
    mnt_arr = affine_mnt(mnt_arr, dx, dy, dtheta, h, w).astype(np.int32)

    save_path = "/data/guanxiongjun/UniFiNet/tmp/2.png"
    draw_minutia_on_finger(img, mnt_arr, save_path=save_path)

    # --- mask some area
    rect_yc = random.randint(h // 4, 3 * h // 4)
    rect_xc = random.randint(w // 4, 3 * w // 4)
    rect_h = random.randint(0, 10)
    rect_w = random.randint(0, 10)
    rect_y1 = max(0, rect_yc - rect_h // 2)
    rect_x1 = max(0, rect_xc - rect_w // 2)
    rect_y2 = min(h, rect_yc + rect_h // 2)
    rect_x2 = min(w, rect_xc + rect_w // 2)
    mask[rect_y1:rect_y2, rect_x1:rect_x2] = 0
    img[mask == 0] = 255
    core_arr = select_mask_mnt(core_arr, mask)
    delta_arr = select_mask_mnt(delta_arr, mask)
    mnt_arr = select_mask_mnt(mnt_arr, mask)

    pts = np.vstack((core_arr, delta_arr, mnt_arr))
    cls = np.concatenate((
        np.full(core_arr.shape[0], 1),  # 对于 points1 的点，标记为 1
        np.full(delta_arr.shape[0], 2),  # 对于 points2 的点，标记为 2
        np.full(mnt_arr.shape[0], 3),  # 对于 points3 的点，标记为 3
    ))

    save_path = "/data/guanxiongjun/UniFiNet/tmp/3.png"
    draw_minutia_on_finger(img, mnt_arr, save_path=save_path)
