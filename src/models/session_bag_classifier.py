from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.fusion_classifier import FusionRadarClassifier


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_positions: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_positions, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(features.size(1), device=features.device, dtype=torch.long)
        return features + self.embedding(positions).unsqueeze(0)


class MaskedSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = mask <= 0.0
        normalized = self.norm1(features)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        features = features + attended
        features = features + self.ffn(self.norm2(features))
        return features


class MaskedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(features).squeeze(-1)
        scores = scores.masked_fill(mask <= 0.0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * features).sum(dim=1)
        return pooled, weights


class SessionBagRadarClassifier(nn.Module):
    def __init__(
        self,
        clip_model: FusionRadarClassifier,
        num_classes: int,
        embedding_dim: int,
        bag_hidden_dim: int = 256,
        dropout: float = 0.1,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        use_positional_encoding: bool = True,
        max_positions: int = 128,
        use_stat_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.clip_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, bag_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_encoding = LearnedPositionalEncoding(max_positions=max_positions, embedding_dim=bag_hidden_dim) if use_positional_encoding else None
        self.session_encoder = nn.ModuleList(
            [
                MaskedSelfAttentionBlock(
                    embedding_dim=bag_hidden_dim,
                    hidden_dim=bag_hidden_dim * 2,
                    num_heads=attention_heads,
                    dropout=dropout,
                )
                for _ in range(max(int(encoder_layers), 0))
            ]
        )
        self.use_stat_pooling = bool(use_stat_pooling)
        self.session_pool = MaskedAttentionPool(bag_hidden_dim, bag_hidden_dim, dropout)
        pooled_dim = bag_hidden_dim * 3 if self.use_stat_pooling else bag_hidden_dim
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, bag_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.session_classifier = nn.Sequential(
            nn.LayerNorm(bag_hidden_dim),
            nn.Linear(bag_hidden_dim, bag_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bag_hidden_dim, num_classes),
        )
        self.session_track_router_head = nn.Linear(bag_hidden_dim, 3)
        self.session_ship_drone_head = nn.Linear(bag_hidden_dim, 2)
        self.session_ship_bird_head = nn.Linear(bag_hidden_dim, 2)
        self.session_bird_drone_head = nn.Linear(bag_hidden_dim, 2)
        self.session_bird_surface_head = nn.Linear(bag_hidden_dim, 1)
        self.session_people_cyclist_head = nn.Linear(bag_hidden_dim, 2)
        self.session_bird_clutter_other_head = nn.Linear(bag_hidden_dim, 3)

    def forward(
        self,
        track_features: torch.Tensor,
        track_global_features: torch.Tensor,
        track_short_features: torch.Tensor | None,
        track_long_features: torch.Tensor | None,
        rd_features: torch.Tensor | None,
        rd_mask: torch.Tensor | None,
        clip_mask: torch.Tensor,
    ) -> dict[str, Any]:
        batch_size, num_clips = track_features.shape[:2]

        flat_track_features = track_features.reshape(batch_size * num_clips, *track_features.shape[2:])
        flat_track_global = track_global_features.reshape(batch_size * num_clips, *track_global_features.shape[2:])
        flat_track_short = None if track_short_features is None else track_short_features.reshape(batch_size * num_clips, *track_short_features.shape[2:])
        flat_track_long = None if track_long_features is None else track_long_features.reshape(batch_size * num_clips, *track_long_features.shape[2:])
        flat_rd_features = None if rd_features is None else rd_features.reshape(batch_size * num_clips, *rd_features.shape[2:])
        flat_rd_mask = None if rd_mask is None else rd_mask.reshape(batch_size * num_clips, 1)

        clip_outputs = self.clip_model(
            flat_track_features,
            flat_track_global,
            flat_track_short,
            flat_track_long,
            flat_rd_features,
            flat_rd_mask,
        )

        def _reshape_clip_output(name: str):
            value = clip_outputs.get(name)
            if value is None:
                return None
            return value.reshape(batch_size, num_clips, *value.shape[1:])

        clip_logits = _reshape_clip_output('logits')
        coarse_feature = clip_outputs.get('coarse_feature')
        detail_feature = clip_outputs.get('detail_feature')
        if coarse_feature is not None and detail_feature is not None:
            clip_embedding = torch.cat([coarse_feature, detail_feature], dim=-1).reshape(batch_size, num_clips, -1)
        else:
            clip_embedding = clip_logits

        encoded_clips = self.clip_projection(clip_embedding)
        if self.position_encoding is not None:
            encoded_clips = self.position_encoding(encoded_clips)
        for block in self.session_encoder:
            encoded_clips = block(encoded_clips, clip_mask)

        session_attention_embedding, attention_weights = self.session_pool(encoded_clips, clip_mask)
        if self.use_stat_pooling:
            clip_mask_expanded = clip_mask.unsqueeze(-1)
            valid_counts = clip_mask_expanded.sum(dim=1).clamp_min(1.0)
            masked_encoded = encoded_clips * clip_mask_expanded
            mean_embedding = masked_encoded.sum(dim=1) / valid_counts
            centered = (encoded_clips - mean_embedding.unsqueeze(1)) * clip_mask_expanded
            var_embedding = (centered.pow(2).sum(dim=1) / valid_counts).clamp_min(1e-8)
            std_embedding = torch.sqrt(var_embedding)
            pooled_features = torch.cat([session_attention_embedding, mean_embedding, std_embedding], dim=-1)
        else:
            pooled_features = session_attention_embedding
        session_embedding = self.pool_projection(pooled_features)
        session_logits = self.session_classifier(session_embedding)
        session_track_router_logits = self.session_track_router_head(session_embedding)
        session_ship_drone_logits = self.session_ship_drone_head(session_embedding)
        session_ship_bird_logits = self.session_ship_bird_head(session_embedding)
        session_bird_drone_logits = self.session_bird_drone_head(session_embedding)
        session_bird_surface_logits = self.session_bird_surface_head(session_embedding).squeeze(-1)
        session_people_cyclist_logits = self.session_people_cyclist_head(session_embedding)
        session_bird_clutter_other_logits = self.session_bird_clutter_other_head(session_embedding)
        return {
            'logits': session_logits,
            'session_logits': session_logits,
            'clip_logits': clip_logits,
            'clip_embedding': clip_embedding,
            'encoded_clip_embedding': encoded_clips,
            'session_embedding': session_embedding,
            'session_attention_embedding': session_attention_embedding,
            'session_attention_weights': attention_weights,
            'session_track_router_logits': session_track_router_logits,
            'session_ship_drone_logits': session_ship_drone_logits,
            'session_ship_bird_logits': session_ship_bird_logits,
            'session_bird_drone_logits': session_bird_drone_logits,
            'session_bird_surface_logits': session_bird_surface_logits,
            'session_people_cyclist_logits': session_people_cyclist_logits,
            'session_bird_clutter_other_logits': session_bird_clutter_other_logits,
            'clip_track_weight': _reshape_clip_output('track_weight'),
            'clip_rd_weight': _reshape_clip_output('rd_weight'),
            'clip_track_router_logits': _reshape_clip_output('track_router_logits'),
            'clip_bird_surface_logits': _reshape_clip_output('bird_surface_logits'),
            'clip_bird_rd_logits': _reshape_clip_output('bird_rd_logits'),
            'clip_bird_surface_rd_logits': _reshape_clip_output('bird_surface_rd_logits'),
            'clip_rd_ship_drone_logits': _reshape_clip_output('rd_ship_drone_logits'),
            'clip_rd_ship_bird_logits': _reshape_clip_output('rd_ship_bird_logits'),
            'clip_rd_ship_air_logits': _reshape_clip_output('rd_ship_air_logits'),
        }
