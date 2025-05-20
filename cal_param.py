'''
Description: 
Author: Xiongjun Guan
Date: 2025-05-18 17:33:28
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-05-18 17:38:35

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import yaml

from args import dict_to_namespace, get_args
from models.DETR import UniFiNet_DETR

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

model = UniFiNet_DETR(cfg.model_cfg)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params: (M)', n_parameters / 1e6)
