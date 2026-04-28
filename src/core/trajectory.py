from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrackRepresentation:
    """
    轨迹表示类，包含序列特征、深度特征、原始物理特征、物理特征和融合特征。
    """
    sequence_features: torch.Tensor # 序列特征
    deep_features: torch.Tensor # 深度特征
    raw_physical_features: torch.Tensor # 原始物理特征
    physical_features: torch.Tensor # 物理特征
    fused_features: torch.Tensor # 融合特征


class TransformerTrackEncoder(nn.Module):
    """
    基于 Transformer 的轨迹编码器，用于将轨迹序列转换为固定长度的表示。
    """
    def __init__(
        self,
        input_dim: int, # 输入特征维度(TRACK_FEATURE_COLUMNS)
        model_dim: int = 128, # 模型维度
        num_heads: int = 4, # 头数，从四个角度看数据，四个头，每个头分32维
        num_layers: int = 2, # 层数
        dropout: float = 0.1, # Dropout 率，随机10%失活
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim) # 输入投影层
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, # 模型维度
            nhead=num_heads, # 头数
            dim_feedforward=model_dim * 4, # 前馈层维度
            dropout=dropout, # Dropout 率
            batch_first=True, # 输入和输出的批次维度在第一个位置
            activation="gelu", # 激活函数
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.input_proj.__call__(features) # 输入投影层
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask) # Transformer 编码器
        return self.norm.__call__(hidden) # 层归一化


class TrackRepresentationBuilder(nn.Module):
    """
    轨迹表示构建器，将序列特征和物理特征转换为轨迹表示。
    """
    def __init__(
        self,
        sequence_input_dim: int, # 序列输入特征维度(TRACK_FEATURE_COLUMNS)
        physical_input_dim: int, # 物理输入特征维度(physical_features)
        model_dim: int = 128, # 模型维度
        dropout: float = 0.1, # Dropout 率，随机10%失活
    ) -> None:
        super().__init__()
        self.sequence_encoder = TransformerTrackEncoder(
            input_dim=sequence_input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.physical_norm = nn.LayerNorm(physical_input_dim)
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_input_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim + 32, model_dim + 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim + 32),
        )

    def forward(
        self,
        sequence_features: torch.Tensor, # 序列特征
        physical_features: torch.Tensor, # 物理特征
        padding_mask: torch.Tensor | None = None,
    ) -> TrackRepresentation:
        encoded = self.sequence_encoder(sequence_features, padding_mask=padding_mask)
        if padding_mask is None:
            deep_features = encoded.mean(dim=1)
        else:
            valid = (~padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            deep_features = (encoded * valid).sum(dim=1) / denom
        physical_proj = self.physical_proj(self.physical_norm(physical_features))
        fused = self.fusion(torch.cat([deep_features, physical_proj], dim=-1))
        return TrackRepresentation(
            sequence_features=encoded,
            deep_features=deep_features,
            raw_physical_features=physical_features,
            physical_features=physical_proj,
            fused_features=fused,
        )
