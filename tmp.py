from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from models.DETR import UniFiNet_DETR

# 测试模型
if __name__ == "__main__":
    config_path = "/data/guanxiongjun/UniFiNet/configs/config.yaml"
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)
    model_cfgs = SimpleNamespace(**cfg["model_cfg"])

    model = UniFiNet_DETR(model_cfgs)

    # 模拟输入
    batch_size = 8
    image_size = model_cfgs.img_sz
    inputs = torch.randn(batch_size, 3, *image_size)  # 假设输入图像大小为 (3, 256, 256)

    outputs = model(inputs)  # (B, num_tokens, num_classes)
    print("输出形状:", outputs.shape)
