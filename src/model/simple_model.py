"""
带原型机制的异常检测器
- K 个可学习原型
- 序列与原型通过交叉注意力交互
- 输出整体类别 + 逐点掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
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


class SimplePrototypeAnomalyDetector(nn.Module):
    """
    基于原型的异常检测器（简化版）
    - 使用交叉注意力让原型与序列交互
    - 输出整体类别 + 逐点掩码
    """
    def __init__(self, input_dim, d_model=128, num_prototypes=16,
                 num_classes=5, nhead=4, num_encoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 序列编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 可学习原型
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        nn.init.xavier_uniform_(self.prototypes)

        # 交叉注意力：让原型与序列交互
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 原型更新后的 FFN
        self.proto_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 分类头（全局）
        self.class_head = nn.Sequential(
            nn.Linear(d_model + num_prototypes, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes + 1)
        )

        # Mask 头（逐点）
        self.mask_head = nn.Sequential(
            nn.Linear(d_model + num_prototypes, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) 输入序列
        Returns:
            pred_class: (B, C+1) 整体类别
            pred_mask: (B, T) 逐点异常概率
        """
        B, T, _ = x.shape

        # 1. 编码序列
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.encoder(x)  # (B, T, D)

        # 2. 扩展原型到 batch 维度
        prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # 3. 交叉注意力：原型作为 Query，序列作为 Key/Value
        #   让每个原型从序列中聚合相关信息
        proto_updated, attn_weights = self.cross_attention(
            query=prototypes,      # (B, K, D)
            key=memory,            # (B, T, D)
            value=memory           # (B, T, D)
        )  # proto_updated: (B, K, D), attn_weights: (B, K, T)

        # 4. FFN 更新原型
        proto_updated = self.proto_ffn(proto_updated)  # (B, K, D)

        # 5. 计算每个时间步与原型的相似度（用于 Mask 头）
        # 使用更新后的原型
        memory_norm = F.normalize(memory, dim=-1)
        proto_norm = F.normalize(proto_updated, dim=-1)
        proto_sim = torch.matmul(memory_norm, proto_norm.transpose(1, 2))  # (B, T, K)

        # 6. 全局聚合：每个原型的最大相似度
        proto_agg = proto_sim.max(dim=1)[0]  # (B, K) - 每个原型在序列上的最大响应
        global_feat = memory.mean(dim=1)     # (B, D)

        # 7. 分类
        class_feat = torch.cat([global_feat, proto_agg], dim=-1)  # (B, D + K)
        pred_class = self.class_head(class_feat)  # (B, C+1)

        # 8. Mask 预测
        mask_feat = torch.cat([memory, proto_sim], dim=-1)  # (B, T, D + K)
        pred_mask = self.mask_head(mask_feat).squeeze(-1)   # (B, T)
        pred_mask = torch.sigmoid(pred_mask)

        return pred_class, pred_mask


class SimpleAnomalyDetector(nn.Module):
    """
    最简单的异常检测器：没有原型，直接用 Transformer 编码后输出
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_encoder_layers=4,
                 num_classes=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes + 1)
        )

        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.encoder(x)

        global_feat = memory.mean(dim=1)
        pred_class = self.class_head(global_feat)

        pred_mask = self.mask_head(memory).squeeze(-1)
        pred_mask = torch.sigmoid(pred_mask)

        return pred_class, pred_mask


def build_model(input_dim, model_type='simple', **kwargs):
    """
    构建模型

    Args:
        input_dim: 输入维度
        model_type: 'simple' (原型版) 或 'basic' (无原型版)
        **kwargs: 其他参数

    Returns:
        model: nn.Module
    """
    if model_type == 'basic':
        return SimpleAnomalyDetector(input_dim=input_dim, **kwargs)
    else:  # 'simple' 或默认
        return SimplePrototypeAnomalyDetector(input_dim=input_dim, **kwargs)