'''
Description: 
Author: Xiongjun Guan
Date: 2025-05-13 17:50:37
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-05-18 17:38:14

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


class BackboneRes(nn.Module):
    """
    使用 ResNet 作为图像编码器，提取二维图像特征。
    """

    def __init__(
        self,
        specify_model="resnet34",
        pretrained=False,
    ):
        super(BackboneRes, self).__init__()
        # 使用 ResNet，并去掉最后的全连接层
        if specify_model == "resnet18":
            resnet = resnet18(pretrained=False)
            if pretrained is not False:
                state_dict = torch.load(pretrained)
                resnet.load_state_dict(state_dict)
            self.feat_dim = 512
            self.downsample_rate = 32
        elif specify_model == "resnet34":
            resnet = resnet34(pretrained=False)
            if pretrained is not False:
                state_dict = torch.load(pretrained)
                resnet.load_state_dict(state_dict)
            self.feat_dim = 512
            self.downsample_rate = 32

        self.stem = nn.Sequential(
            resnet.conv1,  # 7x7 Conv with stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 3x3 MaxPool with stride 2
        )
        # Use only layers until the third stage (16x downsample)
        self.res_layers = nn.Sequential(
            resnet.layer1,  # No downsampling
            resnet.layer2,  # Downsample by 2
            resnet.layer3,  # Downsample by 2
            resnet.layer4,  # Downsample by 2
        )

    def forward(self, x):
        """
        输入: x, 二维图像 (B, 3, H, W)
        输出: 图像特征 token (B, c, h, w)
        """
        x = self.stem(x)  # Initial stem
        x = self.res_layers(x)  # ResNet layers

        return x


if __name__ == "__main__":
    model = BackboneRes()
    inp = torch.randn(4, 3, 512, 512)
    res = model(inp)
