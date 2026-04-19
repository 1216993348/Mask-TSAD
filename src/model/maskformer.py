"""
MaskFormer 模型 - 时间序列版（修正）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class MaskFormerTS(nn.Module):
    """
    MaskFormer 风格时间序列模型
    直接预测每个 query 的 mask（时间步）
    """

    def __init__(self, input_dim, d_model=256, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, num_queries=100, num_classes=6,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))

        # 输出头
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        # Mask 预测直接用 query 和 memory 点积，不需要额外线性层

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            pred_class: (B, Q, C+1)
            pred_mask: (B, Q, T)
        """
        B, T, D = x.shape

        # 编码
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        memory = self.encoder(x)  # (B, T, d_model)

        # 解码
        query = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # (B, Q, d_model)
        hs = self.decoder(query, memory)  # (B, Q, d_model)

        # 类别预测
        pred_class = self.class_embed(hs)  # (B, Q, C+1)

        # Mask 预测：query 和 memory 点积（原版 MaskFormer 做法）
        # (B, Q, d_model) @ (B, d_model, T) -> (B, Q, T)
        pred_mask = torch.bmm(hs, memory.transpose(1, 2))  # (B, Q, T)

        return pred_class, pred_mask


def build_model(input_dim, **kwargs):
    """构建模型"""
    return MaskFormerTS(input_dim=input_dim, **kwargs)