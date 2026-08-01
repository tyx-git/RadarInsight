from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


TEMPORAL_ENCODING_MODES = {"none", "sinusoidal_index_v1"}
TEMPORAL_POOLING_MODES = {"mean", "last_valid", "attention"}
TEMPORAL_POSITION_REFERENCE = "first_observed_token_index"


class TrajectorySequencePool(nn.Module):
    """Pool encoded trajectory tokens without using future context."""

    def __init__(
        self,
        input_dim: int,
        mode: str = "mean",
        attention_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in TEMPORAL_POOLING_MODES:
            raise ValueError(
                "unsupported temporal pooling mode: "
                f"{mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
            )
        self.mode = normalized_mode
        if normalized_mode == "attention":
            hidden_dim = int(max(1, attention_hidden_dim))
            self.score = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        sequence: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, feature]")
        batch_size, time_steps, _ = sequence.shape
        if time_steps < 1:
            return sequence.new_zeros((batch_size, sequence.shape[-1]))
        if padding_mask is None:
            padding = torch.zeros(
                (batch_size, time_steps),
                dtype=torch.bool,
                device=sequence.device,
            )
        else:
            if padding_mask.shape != (batch_size, time_steps):
                raise ValueError("padding_mask must have shape [batch, time]")
            padding = padding_mask.bool()
        valid = ~padding
        has_valid = valid.any(dim=1)

        if self.mode == "last_valid":
            last_indices = valid.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = sequence[
                torch.arange(batch_size, device=sequence.device),
                last_indices,
            ]
        elif self.mode == "attention":
            safe_padding = padding.clone()
            if bool((~has_valid).any()):
                safe_padding[~has_valid] = False
            weights = self.score(sequence).masked_fill(
                safe_padding.unsqueeze(-1),
                -1.0e9,
            )
            weights = torch.softmax(weights, dim=1)
            pooled = torch.sum(sequence * weights, dim=1)
        else:
            valid_float = valid.to(dtype=sequence.dtype).unsqueeze(-1)
            pooled = (sequence * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        return torch.where(has_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))


def _sinusoidal_index_encoding(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return positions reset at token zero of each input frame."""
    if length < 1:
        return torch.zeros((0, dimension), device=device, dtype=dtype)
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_indices = torch.arange(0, dimension, 2, device=device, dtype=dtype)
    frequencies = torch.exp(-math.log(10000.0) * even_indices / max(1, dimension))
    encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    odd_count = encoding[:, 1::2].shape[1]
    if odd_count:
        encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_count])
    return encoding


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
        temporal_encoding_mode: str = "none",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.temporal_encoding_mode = str(temporal_encoding_mode).strip().lower()
        if self.temporal_encoding_mode not in TEMPORAL_ENCODING_MODES:
            raise ValueError(
                "unsupported temporal encoding mode: "
                f"{temporal_encoding_mode!r}; expected one of {sorted(TEMPORAL_ENCODING_MODES)}"
            )
        self.input_proj = nn.Linear(input_dim, model_dim) # 输入投影层
        if self.temporal_encoding_mode == "sinusoidal_index_v1":
            # Enabled and disabled checkpoints have different state keys, so
            # a temporal-contract mismatch fails at strict checkpoint load.
            self.temporal_scale = nn.Parameter(torch.ones(model_dim))
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
        if self.temporal_encoding_mode == "sinusoidal_index_v1":
            position = _sinusoidal_index_encoding(
                int(hidden.shape[1]),
                int(hidden.shape[2]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden = hidden + position.unsqueeze(0) * self.temporal_scale.to(
                device=hidden.device,
                dtype=hidden.dtype,
            ).view(1, 1, -1)
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
        physical_representation_contract_version: str = "legacy_v2",
        temporal_encoding_mode: str = "none",
        temporal_pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.sequence_encoder = TransformerTrackEncoder(
            input_dim=sequence_input_dim,
            model_dim=model_dim,
            dropout=dropout,
            temporal_encoding_mode=temporal_encoding_mode,
        )
        self.temporal_encoding_mode = self.sequence_encoder.temporal_encoding_mode
        self.temporal_pooling_mode = str(temporal_pooling_mode).strip().lower()
        self.sequence_pool = TrajectorySequencePool(
            model_dim,
            mode=self.temporal_pooling_mode,
            attention_hidden_dim=max(32, model_dim // 2),
        )
        supported_physical_contracts = {"legacy_v2", "source_robust_v3"}
        if physical_representation_contract_version not in supported_physical_contracts:
            raise ValueError(
                "unsupported physical representation contract: "
                f"{physical_representation_contract_version}"
            )
        self.physical_representation_contract_version = physical_representation_contract_version
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
        deep_features = self.sequence_pool(encoded, padding_mask=padding_mask)
        if self.physical_representation_contract_version == "legacy_v2":
            physical_input = self.physical_norm(physical_features)
        else:
            # source_robust_v3 receives signed-log, train-only source-wise
            # normalized continuous values with binary masks left untouched.
            physical_input = physical_features
        physical_proj = self.physical_proj(physical_input)
        fused = self.fusion(torch.cat([deep_features, physical_proj], dim=-1))
        return TrackRepresentation(
            sequence_features=encoded,
            deep_features=deep_features,
            raw_physical_features=physical_features,
            physical_features=physical_proj,
            fused_features=fused,
        )
