from typing import Any, Sequence

import torch
from torch import nn

from src.models.cnn_encoder import X3DMicroMotionEncoder
from src.models.graph_temporal_encoder import TrackGraphEncoder
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


class CosineClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, scale: float):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale))))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = torch.nn.functional.normalize(self.feature_norm(features), dim=-1)
        normalized_weight = torch.nn.functional.normalize(self.weight, dim=-1)
        return torch.exp(self.log_scale) * (normalized_features @ normalized_weight.t())


class HierarchicalClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_coarse_classes: int,
        dropout: float,
        detail_cosine_scale: float,
    ):
        super().__init__()
        self.coarse_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.detail_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.detail_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.coarse_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_coarse_classes),
        )
        self.detail_classifier = CosineClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            scale=detail_cosine_scale,
        )
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(
        self,
        fused_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_feature = self.coarse_projection(fused_feature)
        detail_feature = self.detail_projection(fused_feature)
        detail_gate = self.detail_gate(torch.cat([coarse_feature, detail_feature], dim=-1))
        detail_feature = detail_feature + detail_gate * coarse_feature
        coarse_logits = self.coarse_classifier(coarse_feature)
        detail_logits = self.detail_classifier(detail_feature)
        logits = self.final_classifier(torch.cat([coarse_feature, detail_feature], dim=-1))
        return logits, coarse_logits, detail_logits, coarse_feature, detail_feature


class FusionRadarClassifier(nn.Module):
    def __init__(
        self,
        track_input_dim: int,
        track_global_feature_dim: int,
        num_classes: int,
        num_coarse_classes: int | None = None,
        include_track: bool = True,
        include_rd: bool = True,
        track_projection_dim: int = 128,
        tcn_channels: Sequence[int] = (128, 128, 256),
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        max_sequence_length: int = 64,
        short_sequence_length: int = 8,
        mid_sequence_length: int = 16,
        long_sequence_length: int = 32,
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
        detail_logit_residual_weight: float = 0.2,
        detail_cosine_scale: float = 16.0,
        num_notes: int = 0,
        track_backbone_type: str = "temporal",
        temporal_2d_branch_channels: int = 96,
        bird_class_index: int | None = None,
        surface_class_index: int | None = None,
        aircraft_class_index: int | None = None,
        drone_class_index: int | None = None,
        bird_surface_logit_residual_weight: float = 0.2,
        ship_specialist_logit_residual_weight: float = 0.08,
        confusion_pair_handler_enabled: bool = False,
        confusion_pair_handler_hidden_dim: int = 64,
        confusion_pair_handler_dropout: float = 0.1,
        confusion_pair_handler_weight: float = 0.0,
        confusion_pair_handler_gate_threshold: float = 0.55,
        confusion_ship_drone_handler_weight: float = 0.0,
        confusion_bird_ship_handler_weight: float = 0.0,
        rd_dominant_fusion_bias: float = 1.25,
        rd_unimodal_logit_residual_weight: float = 0.35,
        min_rd_fusion_weight: float = 0.15,
        ambiguous_rd_fusion_boost: float = 0.15,
        surface_rd_fusion_boost: float = 0.1,
        use_weighted_fusion_feature: bool = False,
    ):
        super().__init__()
        self.include_track = include_track
        self.include_rd = include_rd
        self.num_classes = num_classes
        self.fusion_hidden_size = fusion_hidden_size
        self.detail_logit_residual_weight = float(detail_logit_residual_weight)
        self.num_notes = int(num_notes)
        resolved_num_coarse_classes = num_classes if num_coarse_classes is None else num_coarse_classes

        self.track_backbone_type = str(track_backbone_type).lower()
        self.bird_class_index = bird_class_index
        self.surface_class_index = surface_class_index
        self.aircraft_class_index = aircraft_class_index
        self.drone_class_index = drone_class_index
        self.bird_surface_logit_residual_weight = float(bird_surface_logit_residual_weight)
        self.ship_specialist_logit_residual_weight = float(ship_specialist_logit_residual_weight)
        self.confusion_pair_handler_enabled = bool(confusion_pair_handler_enabled)
        self.confusion_pair_handler_weight = float(confusion_pair_handler_weight)
        self.confusion_pair_handler_gate_threshold = float(confusion_pair_handler_gate_threshold)
        self.confusion_ship_drone_handler_weight = float(confusion_ship_drone_handler_weight)
        self.confusion_bird_ship_handler_weight = float(confusion_bird_ship_handler_weight)
        self.rd_dominant_fusion_bias = float(rd_dominant_fusion_bias)
        self.rd_unimodal_logit_residual_weight = float(rd_unimodal_logit_residual_weight)
        self.min_rd_fusion_weight = float(min_rd_fusion_weight)
        self.ambiguous_rd_fusion_boost = float(ambiguous_rd_fusion_boost)
        self.surface_rd_fusion_boost = float(surface_rd_fusion_boost)
        self.use_weighted_fusion_feature = bool(use_weighted_fusion_feature)
        if self.track_backbone_type in {"graph", "dggcn"}:
            self.track_encoder = TrackGraphEncoder(
                input_dim=track_input_dim,
                global_feature_dim=track_global_feature_dim,
                projection_dim=track_projection_dim,
                graph_layers=max(3, transformer_layers + 1),
                dropout=transformer_dropout,
                output_dim=fusion_hidden_size,
                temporal_2d_branch_channels=temporal_2d_branch_channels,
            )
        else:
            self.track_encoder = TrackTemporalEncoder(
                input_dim=track_input_dim,
                global_feature_dim=track_global_feature_dim,
                projection_dim=track_projection_dim,
                tcn_channels=tcn_channels,
                transformer_layers=transformer_layers,
                attention_heads=transformer_heads,
                dropout=transformer_dropout,
                max_sequence_length=max_sequence_length,
                short_sequence_length=short_sequence_length,
                mid_sequence_length=mid_sequence_length,
                long_sequence_length=long_sequence_length,
                output_dim=fusion_hidden_size,
                temporal_2d_branch_channels=temporal_2d_branch_channels,
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
            self.track_reliability_head = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
                nn.GELU(),
                nn.Linear(fusion_hidden_size // 2, 1),
            )
            self.rd_reliability_head = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
                nn.GELU(),
                nn.Linear(fusion_hidden_size // 2, 1),
            )
            self.unimodal_logit_residual = nn.Parameter(torch.tensor(0.25))
            classifier_input_dim = fusion_hidden_size * 4
        else:
            self.rd_encoder = None
            self.rd_pool = None
            self.rd_classifier = None
            self.fusion_blocks = nn.ModuleList()
            self.track_reliability_head = None
            self.rd_reliability_head = None
            self.unimodal_logit_residual = nn.Parameter(torch.tensor(0.15))
            classifier_input_dim = fusion_hidden_size

        self.hierarchical_head = HierarchicalClassifierHead(
            input_dim=classifier_input_dim,
            hidden_dim=fusion_hidden_size,
            num_classes=num_classes,
            num_coarse_classes=resolved_num_coarse_classes,
            dropout=cross_attention_dropout,
            detail_cosine_scale=detail_cosine_scale,
        )
        self.notes_head = nn.Linear(classifier_input_dim if include_rd else fusion_hidden_size, num_notes) if num_notes > 0 else None
        self.track_router_head = nn.Linear(fusion_hidden_size, 3)
        self.bird_drone_head = nn.Linear(classifier_input_dim if include_rd else fusion_hidden_size, 1)
        self.bird_surface_head = nn.Linear(classifier_input_dim if include_rd else fusion_hidden_size, 1)
        self.bird_rd_head = nn.Linear(fusion_hidden_size, 1)
        self.bird_surface_rd_head = nn.Linear(fusion_hidden_size, 1)
        self.rd_bird_drone_head = nn.Linear(fusion_hidden_size, 2)
        self.rd_local_ship_projection = nn.Sequential(
            nn.LayerNorm(max(8, fusion_hidden_size // 2)),
            nn.Linear(max(8, fusion_hidden_size // 2), fusion_hidden_size),
            nn.GELU(),
        )
        self.rd_ship_drone_head = nn.Linear(fusion_hidden_size, 2)
        self.rd_ship_bird_head = nn.Linear(fusion_hidden_size, 2)
        self.rd_ship_air_head = nn.Linear(fusion_hidden_size, 2)
        if self.confusion_pair_handler_enabled:
            pair_input_dim = fusion_hidden_size * 3
            self.ship_drone_handler = nn.Sequential(
                nn.LayerNorm(pair_input_dim),
                nn.Linear(pair_input_dim, confusion_pair_handler_hidden_dim),
                nn.GELU(),
                nn.Dropout(confusion_pair_handler_dropout),
                nn.Linear(confusion_pair_handler_hidden_dim, 2),
            )
            self.bird_ship_handler = nn.Sequential(
                nn.LayerNorm(pair_input_dim),
                nn.Linear(pair_input_dim, confusion_pair_handler_hidden_dim),
                nn.GELU(),
                nn.Dropout(confusion_pair_handler_dropout),
                nn.Linear(confusion_pair_handler_hidden_dim, 2),
            )
        else:
            self.ship_drone_handler = None
            self.bird_ship_handler = None

    def set_runtime_fusion_controls(
        self,
        min_rd_fusion_weight: float | None = None,
        ambiguous_rd_fusion_boost: float | None = None,
        surface_rd_fusion_boost: float | None = None,
    ) -> None:
        if min_rd_fusion_weight is not None:
            self.min_rd_fusion_weight = float(min_rd_fusion_weight)
        if ambiguous_rd_fusion_boost is not None:
            self.ambiguous_rd_fusion_boost = float(ambiguous_rd_fusion_boost)
        if surface_rd_fusion_boost is not None:
            self.surface_rd_fusion_boost = float(surface_rd_fusion_boost)

    def forward(
        self,
        track_sequence: torch.Tensor,
        track_global_features: torch.Tensor | None = None,
        track_short_sequence: torch.Tensor | None = None,
        track_long_sequence: torch.Tensor | None = None,
        rd_cube: torch.Tensor | None = None,
        rd_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.include_track:
            track_tokens, track_feature = self.track_encoder(track_sequence, track_global_features, track_short_sequence, track_long_sequence)
            track_logits = self.track_classifier(track_feature)
        else:
            batch_size = rd_cube.size(0) if rd_cube is not None else track_sequence.size(0)
            device = rd_cube.device if rd_cube is not None else track_sequence.device
            dtype = rd_cube.dtype if rd_cube is not None else track_sequence.dtype
            track_tokens = torch.zeros((batch_size, 1, self.fusion_hidden_size), device=device, dtype=dtype)
            track_feature = torch.zeros((batch_size, self.fusion_hidden_size), device=device, dtype=dtype)
            track_logits = torch.zeros((batch_size, self.num_classes), device=device, dtype=dtype)

        if not self.include_rd:
            logits, coarse_logits, detail_logits, coarse_feature, detail_feature = self.hierarchical_head(track_feature)
            unimodal_logit_residual = self.unimodal_logit_residual
            if unimodal_logit_residual is not None:
                logits = logits + unimodal_logit_residual * track_logits
            logits = logits + self.detail_logit_residual_weight * detail_logits
            notes_logits = self.notes_head(track_feature) if self.notes_head is not None else None
            track_router_logits = self.track_router_head(track_feature)
            bird_drone_logits = self.bird_drone_head(track_feature)
            bird_surface_logits = self.bird_surface_head(track_feature)
            if self.bird_class_index is not None and self.surface_class_index is not None:
                expert_delta = self.bird_surface_logit_residual_weight * bird_surface_logits.view(-1)
                logits[:, self.bird_class_index] = logits[:, self.bird_class_index] + expert_delta
                logits[:, self.surface_class_index] = logits[:, self.surface_class_index] - expert_delta
            return {
                "logits": logits,
                "track_logits": track_logits,
                "rd_logits": None,
                "coarse_logits": coarse_logits,
                "detail_logits": detail_logits,
                "coarse_feature": coarse_feature,
                "detail_feature": detail_feature,
                "track_weight": torch.full((track_feature.size(0), 1), 1.0 if self.include_track else 0.0, dtype=track_feature.dtype, device=track_feature.device),
                "rd_weight": torch.zeros((track_feature.size(0), 1), dtype=track_feature.dtype, device=track_feature.device),
                "notes_logits": notes_logits,
                "track_router_logits": track_router_logits,
                "bird_drone_logits": bird_drone_logits,
                "bird_surface_logits": bird_surface_logits,
                "rd_bird_drone_logits": None,
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
            rd_tokens, rd_feature, rd_local_feature = rd_encoder(rd_cube)

        rd_feature = rd_feature * modality_mask
        rd_local_feature = rd_local_feature * modality_mask
        rd_logits = rd_classifier(rd_feature)

        track_reliability_head = self.track_reliability_head
        rd_reliability_head = self.rd_reliability_head
        if track_reliability_head is None or rd_reliability_head is None:
            raise RuntimeError("Reliability heads are not initialized for RD fusion.")

        if self.include_track:
            track_reliability = track_reliability_head(track_feature)
        else:
            track_reliability = track_feature.new_full((track_feature.size(0), 1), -1e4)
        rd_reliability = rd_reliability_head(rd_feature).masked_fill(modality_mask <= 0.0, -1e4)
        reliability_logits = torch.cat([track_reliability, rd_reliability + self.rd_dominant_fusion_bias], dim=-1)
        modality_weights = torch.softmax(reliability_logits, dim=-1)
        raw_track_weight = modality_weights[:, :1]
        raw_rd_weight = modality_weights[:, 1:2] * modality_mask

        track_router_logits = self.track_router_head(track_feature)
        track_router_prob = torch.softmax(track_router_logits, dim=-1)
        router_surface = track_router_prob[:, 0:1]
        router_ambiguous = track_router_prob[:, 2:3]

        rd_floor = (
            float(self.min_rd_fusion_weight)
            + float(self.ambiguous_rd_fusion_boost) * router_ambiguous
            + float(self.surface_rd_fusion_boost) * router_surface
        ) * modality_mask
        rd_floor = rd_floor.clamp_min(0.0).clamp_max(0.45)
        rd_weight = torch.maximum(raw_rd_weight, rd_floor) * modality_mask
        track_weight = raw_track_weight
        weight_sum = (track_weight + rd_weight).clamp_min(1e-6)
        track_weight = track_weight / weight_sum
        rd_weight = rd_weight / weight_sum * modality_mask

        fused_track_tokens = track_tokens
        fused_rd_tokens = rd_tokens * modality_mask.view(-1, 1, 1)
        for block in self.fusion_blocks:
            fused_track_tokens, fused_rd_tokens = block(fused_track_tokens, fused_rd_tokens, modality_mask)

        fused_track_feature = self.track_pool(fused_track_tokens)
        fused_rd_feature = rd_pool(fused_rd_tokens) * modality_mask
        if self.use_weighted_fusion_feature:
            weighted_track_feature = fused_track_feature * track_weight
            weighted_rd_feature = fused_rd_feature * rd_weight
        else:
            weighted_track_feature = fused_track_feature
            weighted_rd_feature = fused_rd_feature
        fused_feature = torch.cat(
            [
                weighted_track_feature,
                weighted_rd_feature,
                torch.abs(weighted_track_feature - weighted_rd_feature),
                weighted_track_feature * weighted_rd_feature,
            ],
            dim=-1,
        )
        logits, coarse_logits, detail_logits, coarse_feature, detail_feature = self.hierarchical_head(fused_feature)
        notes_logits = self.notes_head(fused_feature) if self.notes_head is not None else None
        bird_drone_logits = self.bird_drone_head(fused_feature)
        bird_surface_logits = self.bird_surface_head(fused_feature)
        bird_rd_logits = self.bird_rd_head(rd_feature)
        bird_surface_rd_logits = self.bird_surface_rd_head(rd_feature)
        rd_bird_drone_logits = self.rd_bird_drone_head(rd_feature)
        rd_local_ship_feature = self.rd_local_ship_projection(rd_local_feature)
        rd_ship_drone_logits = self.rd_ship_drone_head(rd_local_ship_feature)
        rd_ship_bird_logits = self.rd_ship_bird_head(rd_local_ship_feature)
        rd_ship_air_logits = self.rd_ship_air_head(rd_local_ship_feature)

        unimodal_logit_residual = self.unimodal_logit_residual
        if unimodal_logit_residual is not None:
            logits = logits + unimodal_logit_residual * track_logits * track_weight
            logits = logits + self.rd_unimodal_logit_residual_weight * rd_logits * rd_weight
        logits = logits + self.detail_logit_residual_weight * detail_logits
        router_surface = track_router_logits[:, 0]
        router_aircraft = track_router_logits[:, 1]
        router_ambiguous = track_router_logits[:, 2]
        if self.surface_class_index is not None:
            logits[:, self.surface_class_index] = logits[:, self.surface_class_index] + 0.08 * router_surface
        if self.aircraft_class_index is not None:
            logits[:, self.aircraft_class_index] = logits[:, self.aircraft_class_index] + 0.08 * router_aircraft
        if self.bird_class_index is not None:
            logits[:, self.bird_class_index] = logits[:, self.bird_class_index] + 0.04 * router_ambiguous
        if self.drone_class_index is not None:
            logits[:, self.drone_class_index] = logits[:, self.drone_class_index] + 0.04 * router_ambiguous
        if self.bird_class_index is not None and self.surface_class_index is not None:
            expert_delta = self.bird_surface_logit_residual_weight * bird_surface_logits.view(-1)
            logits[:, self.bird_class_index] = logits[:, self.bird_class_index] + expert_delta
            logits[:, self.surface_class_index] = logits[:, self.surface_class_index] - expert_delta
        if self.bird_class_index is not None and self.drone_class_index is not None:
            logits[:, self.bird_class_index] = logits[:, self.bird_class_index] + 0.08 * rd_bird_drone_logits[:, 0]
            logits[:, self.drone_class_index] = logits[:, self.drone_class_index] + 0.08 * rd_bird_drone_logits[:, 1]
        if self.surface_class_index is not None and self.drone_class_index is not None:
            logits[:, self.surface_class_index] = logits[:, self.surface_class_index] + self.ship_specialist_logit_residual_weight * rd_ship_drone_logits[:, 0]
            logits[:, self.drone_class_index] = logits[:, self.drone_class_index] + self.ship_specialist_logit_residual_weight * rd_ship_drone_logits[:, 1]
        if self.surface_class_index is not None and self.bird_class_index is not None:
            logits[:, self.surface_class_index] = logits[:, self.surface_class_index] + self.ship_specialist_logit_residual_weight * rd_ship_bird_logits[:, 0]
            logits[:, self.bird_class_index] = logits[:, self.bird_class_index] + self.ship_specialist_logit_residual_weight * rd_ship_bird_logits[:, 1]

        ship_drone_handler_logits = None
        bird_ship_handler_logits = None
        if self.confusion_pair_handler_enabled and self.ship_drone_handler is not None and self.bird_ship_handler is not None:
            pair_feature = torch.cat([track_feature, rd_feature, rd_local_ship_feature], dim=-1)
            ship_drone_handler_logits = self.ship_drone_handler(pair_feature)
            bird_ship_handler_logits = self.bird_ship_handler(pair_feature)
            class_probabilities = torch.softmax(logits, dim=-1)
            if self.surface_class_index is not None and self.drone_class_index is not None:
                ship_drone_gate = (class_probabilities[:, self.surface_class_index] + class_probabilities[:, self.drone_class_index]) >= self.confusion_pair_handler_gate_threshold
                if ship_drone_gate.any():
                    ship_drone_weight = self.confusion_ship_drone_handler_weight if self.confusion_ship_drone_handler_weight > 0 else self.confusion_pair_handler_weight
                    logits[ship_drone_gate, self.surface_class_index] = logits[ship_drone_gate, self.surface_class_index] + ship_drone_weight * ship_drone_handler_logits[ship_drone_gate, 0]
                    logits[ship_drone_gate, self.drone_class_index] = logits[ship_drone_gate, self.drone_class_index] + ship_drone_weight * ship_drone_handler_logits[ship_drone_gate, 1]
            if self.bird_class_index is not None and self.surface_class_index is not None:
                bird_ship_gate = (class_probabilities[:, self.bird_class_index] + class_probabilities[:, self.surface_class_index]) >= self.confusion_pair_handler_gate_threshold
                if bird_ship_gate.any():
                    bird_ship_weight = self.confusion_bird_ship_handler_weight if self.confusion_bird_ship_handler_weight > 0 else self.confusion_pair_handler_weight
                    logits[bird_ship_gate, self.bird_class_index] = logits[bird_ship_gate, self.bird_class_index] + bird_ship_weight * bird_ship_handler_logits[bird_ship_gate, 0]
                    logits[bird_ship_gate, self.surface_class_index] = logits[bird_ship_gate, self.surface_class_index] + bird_ship_weight * bird_ship_handler_logits[bird_ship_gate, 1]

        return {
            "logits": logits,
            "track_logits": track_logits,
            "rd_logits": rd_logits,
            "coarse_logits": coarse_logits,
            "detail_logits": detail_logits,
            "coarse_feature": coarse_feature,
            "detail_feature": detail_feature,
            "track_weight": track_weight,
            "rd_weight": rd_weight,
            "notes_logits": notes_logits,
            "track_router_logits": track_router_logits,
            "bird_drone_logits": bird_drone_logits,
            "bird_surface_logits": bird_surface_logits,
            "bird_rd_logits": bird_rd_logits,
            "bird_surface_rd_logits": bird_surface_rd_logits,
            "rd_bird_drone_logits": rd_bird_drone_logits,
            "rd_ship_drone_logits": rd_ship_drone_logits,
            "rd_ship_bird_logits": rd_ship_bird_logits,
            "rd_ship_air_logits": rd_ship_air_logits,
            "ship_drone_handler_logits": ship_drone_handler_logits,
            "bird_ship_handler_logits": bird_ship_handler_logits,
            "rd_feature": rd_feature,
            "rd_local_feature": rd_local_feature,
            "rd_local_ship_feature": rd_local_ship_feature,
        }
