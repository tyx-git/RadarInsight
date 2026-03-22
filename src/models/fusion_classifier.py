from typing import Any, Sequence

import torch
from torch import nn

from src.models.cnn_encoder import X3DMicroMotionEncoder
from src.models.temporal_encoder import TrackTemporalEncoder


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.track_to_rd = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.rd_to_track = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.track_norm1 = nn.LayerNorm(embedding_dim)
        self.rd_norm1 = nn.LayerNorm(embedding_dim)
        self.track_norm2 = nn.LayerNorm(embedding_dim)
        self.rd_norm2 = nn.LayerNorm(embedding_dim)
        self.track_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.rd_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        track_tokens: torch.Tensor,
        rd_tokens: torch.Tensor,
        rd_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rd_mask is None:
            modality_mask = track_tokens.new_ones((track_tokens.size(0), 1, 1))
        else:
            modality_mask = rd_mask.to(track_tokens.dtype).view(-1, 1, 1)

        track_cross, _ = self.track_to_rd(track_tokens, rd_tokens, rd_tokens, need_weights=False)
        rd_cross, _ = self.rd_to_track(rd_tokens, track_tokens, track_tokens, need_weights=False)

        track_tokens = self.track_norm1(track_tokens + track_cross * modality_mask)
        rd_tokens = self.rd_norm1(rd_tokens + rd_cross * modality_mask)

        track_tokens = self.track_norm2(track_tokens + self.track_ffn(track_tokens))
        rd_tokens = self.rd_norm2(rd_tokens + self.rd_ffn(rd_tokens))
        rd_tokens = rd_tokens * modality_mask
        return track_tokens, rd_tokens


class FusionRadarClassifier(nn.Module):
    def __init__(
        self,
        track_input_dim: int,
        num_classes: int,
        include_rd: bool = True,
        track_projection_dim: int = 128,
        tcn_channels: Sequence[int] = (128, 128, 256),
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        max_sequence_length: int = 64,
        x3d_in_channels: int = 1,
        x3d_stem_channels: int = 24,
        x3d_stage_channels: Sequence[int] = (32, 64, 128, 192),
        x3d_stage_depths: Sequence[int] = (2, 3, 5, 3),
        x3d_spatial_strides: Sequence[int] = (1, 2, 2, 2),
        x3d_temporal_strides: Sequence[int] = (1, 1, 2, 2),
        x3d_bottleneck_expansion: float = 2.25,
        x3d_se_ratio: float = 0.0625,
        cross_attention_layers: int = 2,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.1,
        fusion_hidden_size: int = 256,
    ):
        super().__init__()
        self.include_rd = include_rd
        self.fusion_hidden_size = fusion_hidden_size

        self.track_encoder = TrackTemporalEncoder(
            input_dim=track_input_dim,
            projection_dim=track_projection_dim,
            tcn_channels=tcn_channels,
            transformer_layers=transformer_layers,
            attention_heads=transformer_heads,
            dropout=transformer_dropout,
            max_sequence_length=max_sequence_length,
            output_dim=fusion_hidden_size,
        )
        self.track_pool = AttentionPooling(fusion_hidden_size)
        self.track_classifier = nn.Sequential(
            nn.LayerNorm(fusion_hidden_size),
            nn.Linear(fusion_hidden_size, num_classes),
        )

        if include_rd:
            self.rd_encoder: X3DMicroMotionEncoder | None = X3DMicroMotionEncoder(
                in_channels=x3d_in_channels,
                stem_channels=x3d_stem_channels,
                stage_channels=x3d_stage_channels,
                stage_depths=x3d_stage_depths,
                spatial_strides=x3d_spatial_strides,
                temporal_strides=x3d_temporal_strides,
                embedding_dim=fusion_hidden_size,
                expansion_factor=x3d_bottleneck_expansion,
                se_ratio=x3d_se_ratio,
                dropout=cross_attention_dropout,
            )
            self.rd_pool = AttentionPooling(fusion_hidden_size)
            self.rd_classifier = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.Linear(fusion_hidden_size, num_classes),
            )
            self.fusion_blocks = nn.ModuleList(
                [
                    CrossAttentionFusionBlock(
                        embedding_dim=fusion_hidden_size,
                        num_heads=cross_attention_heads,
                        dropout=cross_attention_dropout,
                    )
                    for _ in range(cross_attention_layers)
                ]
            )
            classifier_input_dim = fusion_hidden_size * 4
        else:
            self.rd_encoder = None
            self.rd_pool = None
            self.rd_classifier = None
            self.fusion_blocks = nn.ModuleList()
            classifier_input_dim = fusion_hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, fusion_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(cross_attention_dropout),
            nn.Linear(fusion_hidden_size * 2, num_classes),
        )

    def forward(
        self,
        track_sequence: torch.Tensor,
        rd_cube: torch.Tensor | None = None,
        rd_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        track_tokens, track_feature = self.track_encoder(track_sequence)
        track_logits = self.track_classifier(track_feature)

        if not self.include_rd:
            return {
                "logits": track_logits,
                "track_logits": track_logits,
                "rd_logits": None,
            }

        rd_encoder = self.rd_encoder
        rd_pool = self.rd_pool
        rd_classifier = self.rd_classifier
        if rd_encoder is None or rd_pool is None or rd_classifier is None:
            raise RuntimeError("RD branch is enabled but not initialized.")

        if rd_mask is None:
            modality_mask = track_feature.new_ones((track_feature.size(0), 1))
        else:
            modality_mask = rd_mask.to(track_feature.dtype).view(-1, 1)

        if rd_cube is None:
            rd_tokens = track_tokens.new_zeros((track_tokens.size(0), 1, self.fusion_hidden_size))
            rd_feature = track_feature.new_zeros((track_feature.size(0), self.fusion_hidden_size))
        else:
            rd_tokens, rd_feature = rd_encoder(rd_cube)

        rd_feature = rd_feature * modality_mask
        rd_logits = rd_classifier(rd_feature)

        fused_track_tokens = track_tokens
        fused_rd_tokens = rd_tokens * modality_mask.view(-1, 1, 1)
        for block in self.fusion_blocks:
            fused_track_tokens, fused_rd_tokens = block(fused_track_tokens, fused_rd_tokens, modality_mask)

        fused_track_feature = self.track_pool(fused_track_tokens)
        fused_rd_feature = rd_pool(fused_rd_tokens) * modality_mask
        fused_feature = torch.cat(
            [
                fused_track_feature,
                fused_rd_feature,
                torch.abs(fused_track_feature - fused_rd_feature),
                fused_track_feature * fused_rd_feature,
            ],
            dim=-1,
        )
        logits = self.classifier(fused_feature)

        return {
            "logits": logits,
            "track_logits": track_logits,
            "rd_logits": rd_logits,
        }
