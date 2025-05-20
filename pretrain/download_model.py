import os

from torchvision.models import ResNet34_Weights, resnet34

# 设置自定义路径
os.environ["TORCH_HOME"] = "/data/guanxiongjun/UniFiNet/pretrain/"

# 加载预训练模型，权重会被下载到上述路径
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

# 打印模型结构以验证
print(model)
