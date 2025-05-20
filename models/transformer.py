import torch
import torch.nn as nn


# ---------------- Transformer -----------------
class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout
            ),
            num_layers=num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout
            ),
            num_layers=num_decoder_layers,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_pos, pos_emb):
        """_summary_

        Args:
            src (_type_): image feature, (b, c, h, w)
            mask (_type_): _description_
            query_pos (_type_): position embedding of object queries, (n, c)
            pos_embed (_type_): position embedding of image feature, (b, c, h, w)

        Returns:
            _type_: _description_
        """
        # Flatten spatial dimensions
        bs, c, h, w = src.shape
        pos_emb = pos_emb.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        src += pos_emb

        query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)  # [N_queries, B, C]

        # Transformer Encoder
        memory = self.encoder(src, src_key_padding_mask=mask)

        memory += pos_emb

        # Transformer Decoder
        tgt = torch.zeros_like(query_pos) + query_pos
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
        )
        return hs.permute(1, 0, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
