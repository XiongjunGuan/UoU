import torch
import torch.nn as nn

from models.backbone import BackboneRes
from models.head import MLP
from models.pos_emb import PositionEmbeddingSine
from models.transformer import Transformer


class UniFiNet_DETR(nn.Module):
    """
    主模型，包含编码器、解码器和任务头。
    """

    def __init__(
        self,
        cfgs,
    ):
        super(UniFiNet_DETR, self).__init__()
        self.encoder = BackboneRes(
            specify_model=cfgs.specify_model,
            pretrained=cfgs.backbone_pretrained,
        )

        self.input_proj = nn.Conv2d(
            self.encoder.feat_dim, cfgs.hidden_dim,
            kernel_size=1)  # 将 ResNet 输出的 channel 调整到 output_dim

        self.pos_emb = PositionEmbeddingSine(
            cfgs.hidden_dim,
            cfgs.img_sz[0] // self.encoder.downsample_rate,
            cfgs.img_sz[1] // self.encoder.downsample_rate,
        )[None, :, :, :]

        self.query_embed = nn.Embedding(cfgs.num_queries, cfgs.hidden_dim)
        self.transformer = Transformer(
            d_model=cfgs.hidden_dim,
            nhead=cfgs.nhead,
            num_encoder_layers=cfgs.num_encoder_layers,
            num_decoder_layers=cfgs.num_decoder_layers,
            dim_feedforward=cfgs.dim_feedforward,
            dropout=cfgs.dropout,
        )

        # Prediction heads
        self.class_embed = nn.Linear(
            cfgs.hidden_dim,
            cfgs.num_classes)  # should continue "no object" class
        self.bbox_embed = MLP(cfgs.hidden_dim,
                              cfgs.hidden_dim,
                              output_dim=4,
                              num_layers=3)

    def forward(self, x):
        """
        输入:
            x: 二维图像 (B, C, H, W)
        输出:
            任务头输出 (B, num_tokens, num_classes)
        """
        # Extract features from backbone
        features = self.encoder(x)  # (B, C, H, W)
        src = self.input_proj(features)

        mask = None

        # sine position embedding
        pos_emb = self.pos_emb.repeat(src.shape[0], 1, 1, 1)

        # Transformer forward pass
        # - hs: query feature, (b, q, c)
        # - memory: image feature after Transformer Encoder (b, c, h, w)
        hs, memory = self.transformer(src, mask, self.query_embed.weight,
                                      pos_emb)

        # Prediction heads
        pred_class = self.class_embed(
            hs)  # [batch_size, num_queries, num_classes+1]
        pred_coord = self.bbox_embed(
            hs).sigmoid()  # 0~1 [batch_size, num_queries, 4]
        return {"pred_logits": pred_class, "pred_boxes": pred_coord}


# 测试模型
if __name__ == "__main__":
    # 模拟输入
    batch_size = 8
    image_size = (512, 512)
    num_classes = 60

    model = UniFiNet_DETR(
        img_sz=image_size,
        feature_dim=256,
        num_heads=8,
        num_layers=4,
        num_tokens=10,
        num_classes=num_classes,
    )
    inputs = torch.randn(batch_size, 3, *image_size)  # 假设输入图像大小为 (3, 256, 256)

    outputs = model(inputs)  # (B, num_tokens, num_classes)
    print("输出形状:", outputs.shape)
