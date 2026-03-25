from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _build_temporal_norm(num_channels: int) -> nn.Module:
    for num_groups in (8, 4, 2):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = _build_temporal_norm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = _build_temporal_norm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.dropout(self.activation(self.norm1(self.conv1(inputs))))
        outputs = self.dropout(self.norm2(self.conv2(outputs)))
        return self.activation(outputs + residual)


class TemporalPatternBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal_network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False),
            _build_temporal_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False),
            _build_temporal_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(sequence)
        pattern_tokens = self.temporal_network(projected.transpose(1, 2)).transpose(1, 2)
        attention_scores = self.attention(pattern_tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled_pattern = (attention_weights.unsqueeze(-1) * pattern_tokens).sum(dim=1)
        return pattern_tokens, pooled_pattern


class TrackTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_feature_dim: int = 0,
        projection_dim: int = 128,
        tcn_channels: Sequence[int] = (128, 128, 256),
        transformer_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        self.global_feature_dim = global_feature_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        temporal_layers_list: list[nn.Module] = []
        current_channels = projection_dim
        for layer_index, next_channels in enumerate(tcn_channels):
            temporal_layers_list.append(
                ResidualTemporalBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
            )
            current_channels = next_channels
        self.temporal_network = nn.Sequential(*temporal_layers_list)

        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, current_channels))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=current_channels,
            nhead=attention_heads,
            dim_feedforward=current_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.token_projection = nn.Sequential(
            nn.Linear(current_channels, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 1),
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.continuity_branch = TemporalPatternBranch(
            input_dim=input_dim,
            hidden_dim=output_dim,
            kernel_size=5,
            dropout=dropout,
        )
        self.smoothness_branch = TemporalPatternBranch(
            input_dim=input_dim,
            hidden_dim=output_dim,
            kernel_size=5,
            dropout=dropout,
        )
        self.maneuver_branch = TemporalPatternBranch(
            input_dim=input_dim * 2,
            hidden_dim=output_dim,
            kernel_size=3,
            dropout=dropout,
        )
        self.pattern_fusion = nn.Sequential(
            nn.LayerNorm(output_dim * 3),
            nn.Linear(output_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pattern_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.pattern_token_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        if global_feature_dim > 0:
            self.global_projection: nn.Module | None = nn.Sequential(
                nn.Linear(global_feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.global_gate: nn.Module | None = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            self.global_projection = None
            self.global_gate = None
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _position_encoding(self, sequence_length: int) -> torch.Tensor:
        if sequence_length <= self.max_sequence_length:
            return self.position_embedding[:, :sequence_length]

        resized = F.interpolate(
            self.position_embedding.transpose(1, 2),
            size=sequence_length,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    @staticmethod
    def _temporal_difference(sequence: torch.Tensor) -> torch.Tensor:
        previous = torch.cat([sequence[:, :1], sequence[:, :-1]], dim=1)
        return sequence - previous

    @staticmethod
    def _smoothed_sequence(sequence: torch.Tensor) -> torch.Tensor:
        smoothed = F.avg_pool1d(sequence.transpose(1, 2), kernel_size=3, stride=1, padding=1)
        return smoothed.transpose(1, 2)

    def forward(
        self,
        track_sequence: torch.Tensor,
        track_global_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(track_sequence)
        temporal_features = self.temporal_network(projected.transpose(1, 2)).transpose(1, 2)
        position_encoding = self._position_encoding(temporal_features.size(1))
        transformer_output = self.transformer(temporal_features + position_encoding)

        token_embeddings = self.token_projection(transformer_output)

        continuity_tokens, continuity_feature = self.continuity_branch(track_sequence)
        smoothness_source = track_sequence - self._smoothed_sequence(track_sequence)
        _, smoothness_feature = self.smoothness_branch(smoothness_source)
        first_difference = self._temporal_difference(track_sequence)
        second_difference = self._temporal_difference(first_difference)
        maneuver_source = torch.cat([first_difference, second_difference], dim=-1)
        maneuver_tokens, maneuver_feature = self.maneuver_branch(maneuver_source)

        pattern_feature = self.pattern_fusion(
            torch.cat([continuity_feature, smoothness_feature, maneuver_feature], dim=-1)
        )
        pattern_token_source = (continuity_tokens.mean(dim=1) + maneuver_tokens.mean(dim=1) + pattern_feature) / 3.0
        pattern_token = self.pattern_token_projection(pattern_token_source).unsqueeze(1)
        token_embeddings = torch.cat([token_embeddings, pattern_token], dim=1)

        attention_scores = self.attention(token_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attentive_context = (attention_weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
        sequence_mean = token_embeddings.mean(dim=1)

        raw_mean = track_sequence.mean(dim=1)
        raw_std = track_sequence.std(dim=1, unbiased=False)
        raw_min = track_sequence.min(dim=1).values
        raw_max = track_sequence.max(dim=1).values
        summary_feature = self.summary_projection(torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=-1))

        if self.global_projection is not None and self.global_gate is not None:
            if track_global_features is None:
                track_global_features = torch.zeros(
                    (track_sequence.size(0), self.global_feature_dim),
                    dtype=track_sequence.dtype,
                    device=track_sequence.device,
                )
            global_feature = self.global_projection(track_global_features)
            global_gate = self.global_gate(torch.cat([summary_feature, global_feature], dim=-1))
            merged_summary = summary_feature + global_gate * global_feature
        else:
            merged_summary = summary_feature

        pattern_gate = self.pattern_gate(torch.cat([merged_summary, pattern_feature], dim=-1))
        boundary_pattern_feature = pattern_feature + pattern_gate * pattern_feature

        pooled_embedding = self.output_projection(
            torch.cat([sequence_mean, attentive_context, merged_summary, boundary_pattern_feature], dim=-1)
        )
        return token_embeddings, pooled_embedding
