from typing import Sequence

import torch
from torch import nn


def _make_divisible(value: int, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction_ratio: float = 0.0625):
        super().__init__()
        reduced_channels = max(8, int(channels * reduction_ratio))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.reduce = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.expand = nn.Conv3d(reduced_channels, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.pool(inputs)
        scale = self.activation(self.reduce(scale))
        scale = self.gate(self.expand(scale))
        return inputs * scale


class RadarMicroMotionEnhancer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.range_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.doppler_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.harmonic_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 1, 9), padding=(1, 0, 4), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        merged_channels = hidden_channels * 5
        self.fusion = nn.Sequential(
            nn.Conv3d(merged_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.se = SqueezeExcitation3D(in_channels, reduction_ratio=0.25)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enhanced = torch.cat(
            [
                self.local_branch(inputs),
                self.temporal_branch(inputs),
                self.range_branch(inputs),
                self.doppler_branch(inputs),
                self.harmonic_branch(inputs),
            ],
            dim=1,
        )
        enhanced = self.fusion(enhanced)
        enhanced = self.se(enhanced)
        return self.activation(inputs + enhanced)


class AxisGate1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.gate(self.pointwise(self.projection(inputs)))


class SpectralAxisRefiner(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 12, dropout: float = 0.1):
        super().__init__()
        self.harmonic_short = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(1, 1, 5), padding=(0, 0, 2), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.harmonic_long = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(1, 1, 9), padding=(0, 0, 4), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.harmonic_dilated = nn.Sequential(
            nn.Conv3d(
                channels,
                hidden_channels,
                kernel_size=(1, 1, 5),
                padding=(0, 0, 4),
                dilation=(1, 1, 2),
                bias=False,
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal_doppler = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(3, 1, 7), padding=(1, 0, 3), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        merged_channels = hidden_channels * 4
        self.fusion = nn.Sequential(
            nn.Conv3d(merged_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.range_gate = AxisGate1D(channels, kernel_size=5)
        self.doppler_gate = AxisGate1D(channels, kernel_size=7)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spectral_features = torch.cat(
            [
                self.harmonic_short(inputs),
                self.harmonic_long(inputs),
                self.harmonic_dilated(inputs),
                self.temporal_doppler(inputs),
            ],
            dim=1,
        )
        refined = self.fusion(spectral_features)

        range_descriptor = inputs.mean(dim=(2, 4))
        doppler_descriptor = inputs.mean(dim=(2, 3))
        range_attention = self.range_gate(range_descriptor).unsqueeze(2).unsqueeze(-1)
        doppler_attention = self.doppler_gate(doppler_descriptor).unsqueeze(2).unsqueeze(3)

        refined = refined * range_attention * doppler_attention
        return self.activation(inputs + refined)


class BackgroundContrastRefiner(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.background_branch = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 1, 7), padding=(1, 0, 3), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.contrast_branch = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(5, 1, 5), padding=(2, 0, 2), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.se = SqueezeExcitation3D(out_channels, reduction_ratio=0.25)
        self.activation = nn.SiLU(inplace=True)
        self.local_pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

    @staticmethod
    def _resolve_amplitude_proxy(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) >= 3:
            return inputs[:, 2:3]
        if inputs.size(1) >= 2:
            return torch.log1p(torch.sqrt(inputs[:, 0:1] ** 2 + inputs[:, 1:2] ** 2 + 1e-6))
        return inputs[:, :1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        amplitude = self._resolve_amplitude_proxy(inputs)
        background = amplitude.median(dim=2, keepdim=True).values
        background_suppressed = torch.relu(amplitude - background)
        local_contrast = amplitude - self.local_pool(amplitude)

        background_feature = self.background_branch(background_suppressed)
        contrast_feature = self.contrast_branch(local_contrast)
        refined = self.fusion(torch.cat([background_feature, contrast_feature], dim=1))
        refined = self.se(refined)
        return self.activation(inputs + refined)


class X3DStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_xy = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
        )
        self.conv_t = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            groups=out_channels,
            bias=False,
        )
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv_xy(inputs)
        outputs = self.conv_t(outputs)
        return self.activation(self.norm(outputs))


class X3DBottleneckTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        expansion_factor: float,
        se_ratio: float,
        dropout: float,
    ):
        super().__init__()
        hidden_channels = _make_divisible(int(out_channels * expansion_factor))
        self.conv_a = nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm_a = nn.BatchNorm3d(hidden_channels)
        self.conv_b = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm_b = nn.BatchNorm3d(hidden_channels)
        self.se = SqueezeExcitation3D(hidden_channels, reduction_ratio=se_ratio)
        self.conv_c = nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.norm_c = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.norm_a(self.conv_a(inputs)))
        outputs = self.activation(self.norm_b(self.conv_b(outputs)))
        outputs = self.se(outputs)
        outputs = self.dropout(self.norm_c(self.conv_c(outputs)))
        return outputs


class X3DResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        expansion_factor: float,
        se_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.branch = X3DBottleneckTransform(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expansion_factor=expansion_factor,
            se_ratio=se_ratio,
            dropout=dropout,
        )
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.shortcut(inputs) + self.branch(inputs))


class X3DResStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        spatial_stride: int,
        temporal_stride: int,
        expansion_factor: float,
        se_ratio: float,
        dropout: float,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for block_index in range(depth):
            stride = (
                temporal_stride if block_index == 0 else 1,
                spatial_stride if block_index == 0 else 1,
                spatial_stride if block_index == 0 else 1,
            )
            blocks.append(
                X3DResBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion_factor=expansion_factor,
                    se_ratio=se_ratio,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class X3DHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(features)
        temporal_tokens = projected.mean(dim=(-1, -2)).transpose(1, 2)
        token_embeddings = self.token_norm(temporal_tokens)
        temporal_mean = token_embeddings.mean(dim=1)
        temporal_std = token_embeddings.std(dim=1, unbiased=False)
        pooled_embedding = self.global_projection(torch.cat([temporal_mean, temporal_std], dim=-1))
        return token_embeddings, pooled_embedding


class X3DMicroMotionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 24,
        stage_channels: Sequence[int] = (32, 64, 128, 192),
        stage_depths: Sequence[int] = (2, 3, 5, 3),
        spatial_strides: Sequence[int] = (1, 2, 2, 2),
        temporal_strides: Sequence[int] = (1, 1, 2, 2),
        embedding_dim: int = 256,
        expansion_factor: float = 2.25,
        se_ratio: float = 0.0625,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not (
            len(stage_channels)
            == len(stage_depths)
            == len(spatial_strides)
            == len(temporal_strides)
        ):
            raise ValueError("X3D stage configuration lengths must match.")

        self.output_dim = embedding_dim
        enhancer_channels = max(8, min(32, stem_channels // 2))
        self.micro_motion_enhancer = RadarMicroMotionEnhancer(
            in_channels=in_channels,
            hidden_channels=enhancer_channels,
            dropout=dropout,
        )
        self.spectral_axis_refiner = SpectralAxisRefiner(
            channels=in_channels,
            hidden_channels=max(8, min(24, stem_channels // 2)),
            dropout=dropout,
        )
        self.background_contrast_refiner = BackgroundContrastRefiner(
            out_channels=in_channels,
            hidden_channels=max(8, min(16, stem_channels // 2)),
            dropout=dropout,
        )
        self.stem = X3DStem(in_channels=in_channels, out_channels=stem_channels)

        stages: list[nn.Module] = []
        current_channels = stem_channels
        for depth, out_channels, spatial_stride, temporal_stride in zip(
            stage_depths,
            stage_channels,
            spatial_strides,
            temporal_strides,
        ):
            stage = X3DResStage(
                in_channels=current_channels,
                out_channels=out_channels,
                depth=depth,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                expansion_factor=expansion_factor,
                se_ratio=se_ratio,
                dropout=dropout,
            )
            stages.append(stage)
            current_channels = stage.out_channels
        self.stages = nn.ModuleList(stages)
        self.head = X3DHead(in_channels=current_channels, embedding_dim=embedding_dim, dropout=dropout)

    def forward(self, rd_cube: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rd_cube.ndim == 4:
            features = rd_cube.unsqueeze(1)
        elif rd_cube.ndim == 5:
            features = rd_cube
        else:
            raise ValueError("RD input must have shape (B, T, H, W) or (B, C, T, H, W).")

        features = self.micro_motion_enhancer(features)
        features = self.spectral_axis_refiner(features)
        features = self.background_contrast_refiner(features)
        features = self.stem(features)
        for stage in self.stages:
            features = stage(features)
        return self.head(features)


RDConvEncoder = X3DMicroMotionEncoder
