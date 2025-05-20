"""
Description:
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-06 10:35:12

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import argparse
import datetime
import logging
import os
import os.path as osp
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from tqdm import tqdm

from args import dict_to_namespace, get_args
from data_loader import get_dataloader_train, get_dataloader_valid
from losses.criterion import SetCriterion, reduce_dict
from losses.matcher import build_matcher
from models.DETR import UniFiNet_DETR


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model(model, save_path):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        model_state,
        osp.join(save_path),
    )
    return


def train(
    model,
    criterion,
    train_dataloader,
    valid_dataloader,
    device,
    cfg,
    save_dir=None,
    save_checkpoint=1e6,
):
    # -------------- init settings-------------- #
    lr = cfg.train_cfg.lr
    end_lr = cfg.train_cfg.end_lr
    optim = cfg.train_cfg.optimizer
    scheduler_type = cfg.train_cfg.scheduler_type
    num_epoch = cfg.train_cfg.epochs
    max_norm = cfg.train_cfg.clip_max_norm

    if valid_dataloader is None:
        valid = False
    else:
        valid = True

    # -------------- select optimizer -------------- #
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=0,
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-3,
        )
    elif optim == "adamW":
        optimizer = torch.optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-2,
        )

    # -------------- select scheduler -------------- #
    best_error = None
    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=num_epoch,
                                      eta_min=end_lr)
    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer,
                           np.round(num_epoch / (1 + np.log10(lr / end_lr))),
                           0.1)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode="min",
                                      factor=0.1,
                                      patience=10)

    # -------------- train & valid -------------- #
    for epoch in range(num_epoch):
        if ("epoch_stop" in cfg["train_cfg"].keys()
                and epoch > cfg["train_cfg"]["epoch_stop"]):
            break

        # -------------- train phase
        model.train()
        train_losses = {"total": 0}
        logging.info("epoch: {}, lr:{:.8f}".format(
            epoch,
            optimizer.state_dict()["param_groups"][0]["lr"]))
        pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")
        for img, targets in pbar:
            img = img.float().to(device)
            targets = [{
                k: v.float().to(device)
                for k, v in t.items()
            } for t in targets]

            outputs = model(img)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f'{k}_unscaled': v
                for k, v in loss_dict_reduced.items()
            }
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k]
                for k, v in loss_dict_reduced.items() if k in weight_dict
            }
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            train_losses["total"] += loss_value / len(train_dataloader)
            klist = loss_dict_reduced_unscaled.keys()
            for k in klist:
                if k in train_losses:
                    train_losses[k] += loss_dict_reduced_unscaled[k] / len(
                        train_dataloader)
                else:
                    train_losses[k] = loss_dict_reduced_unscaled[k] / len(
                        train_dataloader)
            klist = loss_dict_reduced_scaled.keys()
            for k in klist:
                if k in train_losses:
                    train_losses[k] += loss_dict_reduced_scaled[k] / len(
                        train_dataloader)
                else:
                    train_losses[k] = loss_dict_reduced_scaled[k] / len(
                        train_dataloader)

            pbar.set_postfix(**{"loss": losses.item()})

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            del loss

        pbar.close()

        klist = train_losses.keys()
        logging_info = "\tTRAIN: ".format(epoch)
        for k in klist:
            logging_info = logging_info + "{}:{:.4f}, ".format(
                k, train_losses[k])
        logging.info(logging_info)

        # -------------- valid phase
        if valid is False:
            continue
        model.eval()
        with torch.no_grad():
            valid_losses = {
                "total": 0,
            }
            pbar = tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")

            for img, targets in pbar:
                img = img.float().to(device)
                targets = [{
                    k: v.float().to(device)
                    for k, v in t.items()
                } for t in targets]

                outputs = model(img)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {
                    f'{k}_unscaled': v
                    for k, v in loss_dict_reduced.items()
                }
                loss_dict_reduced_scaled = {
                    k: v * weight_dict[k]
                    for k, v in loss_dict_reduced.items() if k in weight_dict
                }
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                train_losses["total"] += loss_value / len(train_dataloader)
                klist = loss_dict_reduced_unscaled.keys()
                for k in klist:
                    if k in train_losses:
                        train_losses[k] += loss_dict_reduced_unscaled[k] / len(
                            train_dataloader)
                    else:
                        train_losses[k] = loss_dict_reduced_unscaled[k] / len(
                            train_dataloader)
                klist = loss_dict_reduced_scaled.keys()
                for k in klist:
                    if k in train_losses:
                        train_losses[k] += loss_dict_reduced_scaled[k] / len(
                            train_dataloader)
                    else:
                        train_losses[k] = loss_dict_reduced_scaled[k] / len(
                            train_dataloader)

            pbar.close()

            klist = valid_losses.keys()
            logging_info = "\tVALID: ".format(epoch)
            for k in klist:
                logging_info = logging_info + "{}:{:.4f}, ".format(
                    k, valid_losses[k])
            logging.info(logging_info)

        # -------------- scheduler
        scheduler.step()

        # save
        if save_dir is not None:
            if (not np.isnan(valid_losses["total"])) and ((
                (best_error is None)) or (valid_losses["total"] < best_error)):
                best_error = valid_losses["total"]
                save_model(model, osp.join(save_dir, f"best.pth"))
                logging.info("SAVE BEST MODEL!")

            if scheduler_type == "ReduceLROnPlateau":
                if optimizer.state_dict()["param_groups"][0]["lr"] < end_lr:
                    return
            elif epoch >= save_checkpoint:
                save_model(model, osp.join(save_dir, f"epoch_{epoch}.pth"))

    return


if __name__ == "__main__":
    set_seed(seed=7)

    # --- load args
    args = get_args()

    cuda_ids = args.cuda_ids
    if "," in cuda_ids:
        cuda_ids = [int(x) for x in args.cuda_ids.split(",")]
    else:
        cuda_ids = [int(cuda_ids)]
    args.cuda_ids = cuda_ids

    # --- load config
    config_path = f"./configs/{args.config_name}.yaml"
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    # --- update param from `args` to `config`
    args = vars(args)
    for part in cfg.keys():
        for k in cfg[part].keys():
            if k in args.keys():
                cfg[part][k] = args[k]

    # --- [dict] -> [namespace]
    cfg = dict_to_namespace(cfg)

    # --- set save dir
    save_basedir = cfg.save_cfg.save_basedir
    if cfg.save_cfg.save_title == "time":
        now = datetime.datetime.now()
        save_title = now.strftime("%Y-%m-%d-%H-%M-%S")
    else:
        save_title = cfg.save_cfg.save_title
    save_basedir = osp.join(save_basedir, save_title)

    if not osp.exists(save_basedir):
        os.makedirs(save_basedir)

    # --- save cfg
    with open(osp.join(save_basedir, "config.yaml"), "w") as file:
        yaml.dump(cfg, file, default_flow_style=False)

    # --- load database info
    train_info_path = cfg["db_cfg"]["train_info_path"]
    valid_info_path = cfg["db_cfg"]["valid_info_path"]
    train_info = np.load(train_info_path, allow_pickle=True).item()
    valid_info = np.load(valid_info_path, allow_pickle=True).item()

    # --- logging
    logging_path = osp.join(save_basedir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )

    # --- set dataloader
    train_loader = get_dataloader_train(
        fp_lst=train_info["fp_lst"],
        mnt_lst=train_info["mnt_lst"],
        img_sz=cfg.model_cfg.img_sz,
        apply_aug=cfg.train_cfg.apply_aug,
        trans_aug=cfg.train_cfg.trans_aug,
        rot_aug=cfg.train_cfg.rot_aug,
        batch_size=cfg.train_cfg.batch_size,
        shuffle=True,
    )

    valid_loader = get_dataloader_valid(
        fp_lst=valid_info["fp_lst"],
        mnt_lst=valid_info["mnt_lst"],
        img_sz=cfg.model_cfg.img_sz,
        apply_aug=cfg.train_cfg.apply_aug,
        trans_aug=cfg.train_cfg.trans_aug,
        rot_aug=cfg.train_cfg.rot_aug,
        batch_size=cfg.train_cfg.batch_size,
        shuffle=False,
    )

    # --- set models
    device = torch.device(
        "cuda:{}".format(str(cfg["train_cfg"]["cuda_ids"][0])
                         ) if torch.cuda.is_available() else "cpu")
    if cfg.model_cfg.name == "UniFiNet_DETR":
        model = UniFiNet_DETR(cfg.model_cfg)

    model = torch.nn.DataParallel(
        model,
        device_ids=cfg.train_cfg.cuda_ids,
        output_device=cfg.train_cfg.cuda_ids[0],
    ).to(device)
    logging.info("Model: {}".format(cfg.model_cfg.name))

    # --- set loss
    matcher = build_matcher(cfg.matcher_cfg)
    weight_dict = {'loss_ce': 1, 'loss_bbox': cfg.loss_cfg.bbox_loss_coef}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes=cfg.train_cfg.num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=args.loss_cfg.eos_coef,
                             losses=losses)
    criterion.to(device)

    logging.info("******** begin training ********")
    train(
        model=model,
        criterion=criterion,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cfg=cfg,
        save_dir=save_basedir,
        save_checkpoint=cfg.train_cfg.epochs - cfg.train_cfg.ckpts_num,
    )
