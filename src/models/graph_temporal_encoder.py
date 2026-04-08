from __future__ import annotations

import math

import torch
from torch import nn


class DynamicEdgeNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        source = node_features.unsqueeze(2)
        target = node_features.unsqueeze(1)
        absolute_delta = torch.abs(source - target)
        pairwise = torch.cat([source.expand_as(absolute_delta), target.expand_as(absolute_delta), absolute_delta], dim=-1)
        return self.network(pairwise).squeeze(-1)


class DGGCNBlock(nn.Module):
    def __init__(self, feature_dim: int, dropout: float):
        super().__init__()
        self.local_message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.similarity_message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dynamic_message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        local_adjacency: torch.Tensor,
        similarity_adjacency: torch.Tensor,
        dynamic_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        local_context = local_adjacency @ self.local_message(node_features)
        similarity_context = similarity_adjacency @ self.similarity_message(node_features)
        dynamic_context = dynamic_adjacency @ self.dynamic_message(node_features)
        gate_logits = self.scale_gate(torch.cat([node_features, local_context, similarity_context, dynamic_context], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused_context = (
            gate_weights[..., 0:1] * local_context
            + gate_weights[..., 1:2] * similarity_context
            + gate_weights[..., 2:3] * dynamic_context
        )
        return node_features + self.update(torch.cat([node_features, fused_context], dim=-1))


class TrackDGGCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_feature_dim: int = 0,
        projection_dim: int = 128,
        graph_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
        temporal_2d_branch_channels: int = 96,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.global_feature_dim = global_feature_dim
        self.temporal_2d_branch_channels = temporal_2d_branch_channels

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.motion_projection = nn.Sequential(
            nn.Linear(input_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.edge_network = DynamicEdgeNetwork(projection_dim, projection_dim, dropout=dropout)
        self.blocks = nn.ModuleList([DGGCNBlock(projection_dim, dropout=dropout) for _ in range(graph_layers)])

        self.local_scale = nn.Parameter(torch.tensor(1.0))
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))
        self.dynamic_scale = nn.Parameter(torch.tensor(1.0))

        half_channels = max(8, temporal_2d_branch_channels // 2)
        self.sequence_2d_encoder = nn.Sequential(
            nn.Conv2d(1, half_channels, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(half_channels),
            nn.GELU(),
            nn.Conv2d(half_channels, temporal_2d_branch_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(temporal_2d_branch_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.sequence_2d_projection = nn.Sequential(
            nn.Linear(temporal_2d_branch_channels * 16, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.token_projection = nn.Sequential(
            nn.Linear(projection_dim, output_dim),
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
        )
        self.pattern_projection = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.GELU(),
        )
        self.short_context_projection = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.GELU(),
        )
        self.long_context_projection = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.GELU(),
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim * 3),
            nn.Sigmoid(),
        )
        if global_feature_dim > 0:
            self.global_projection = nn.Sequential(
                nn.Linear(global_feature_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.global_gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            self.global_projection = None
            self.global_gate = None
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 7, output_dim),
            nn.GELU(),
        )

    @staticmethod
    def _temporal_difference(sequence: torch.Tensor) -> torch.Tensor:
        previous = torch.roll(sequence, shifts=1, dims=1)
        previous[:, 0, :] = sequence[:, 0, :]
        return sequence - previous

    def _build_local_adjacency(self, sequence_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device, dtype=dtype)
        pairwise_distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        local = torch.exp(-(pairwise_distance ** 2) / max(float(sequence_length), 1.0))
        local = local + torch.eye(sequence_length, device=device, dtype=dtype)
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return local

    def _build_similarity_adjacency(self, node_features: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.normalize(node_features, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        similarity = torch.softmax(self.similarity_scale * similarity, dim=-1)
        return similarity

    def _build_dynamic_adjacency(self, node_features: torch.Tensor) -> torch.Tensor:
        dynamic_logits = self.edge_network(node_features)
        dynamic_adjacency = torch.softmax(self.dynamic_scale * dynamic_logits, dim=-1)
        return dynamic_adjacency

    def forward(
        self,
        track_sequence: torch.Tensor,
        track_global_features: torch.Tensor | None = None,
        track_short_sequence: torch.Tensor | None = None,
        track_long_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if track_short_sequence is None:
            track_short_sequence = track_sequence[:, : max(1, track_sequence.size(1) // 2)]
        if track_long_sequence is None:
            track_long_sequence = track_sequence

        first_difference = self._temporal_difference(track_sequence)
        second_difference = self._temporal_difference(first_difference)
        motion_source = torch.cat([first_difference, second_difference], dim=-1)

        projected = self.input_projection(track_sequence)
        motion_feature = self.motion_projection(motion_source)
        node_features = projected + motion_feature

        sequence_length = track_sequence.size(1)
        local_template = self._build_local_adjacency(sequence_length, track_sequence.device, track_sequence.dtype)
        local_adjacency = (self.local_scale.sigmoid() * local_template).unsqueeze(0).expand(track_sequence.size(0), -1, -1)
        local_adjacency = local_adjacency / local_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity_adjacency = self._build_similarity_adjacency(node_features)
        dynamic_adjacency = self._build_dynamic_adjacency(node_features)

        graph_tokens = node_features
        for block in self.blocks:
            graph_tokens = block(graph_tokens, local_adjacency, similarity_adjacency, dynamic_adjacency)

        token_embeddings = self.token_projection(graph_tokens)
        attention_scores = self.attention(token_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attentive_context = (attention_weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
        sequence_mean = token_embeddings.mean(dim=1)

        raw_mean = track_sequence.mean(dim=1)
        raw_std = track_sequence.std(dim=1, unbiased=False)
        raw_min = track_sequence.min(dim=1).values
        raw_max = track_sequence.max(dim=1).values
        summary_feature = self.summary_projection(torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=-1))

        abs_first_mean = first_difference.abs().mean(dim=1)
        abs_second_mean = second_difference.abs().mean(dim=1)
        first_std = first_difference.std(dim=1, unbiased=False)
        second_std = second_difference.std(dim=1, unbiased=False)
        pattern_feature = self.pattern_projection(torch.cat([abs_first_mean, abs_second_mean, first_std, second_std], dim=-1))

        short_mean = track_short_sequence.mean(dim=1)
        short_std = track_short_sequence.std(dim=1, unbiased=False)
        long_mean = track_long_sequence.mean(dim=1)
        long_std = track_long_sequence.std(dim=1, unbiased=False)
        short_context = self.short_context_projection(torch.cat([short_mean, short_std], dim=-1))
        long_context = self.long_context_projection(torch.cat([long_mean, long_std], dim=-1))
        scale_gates = self.scale_gate(torch.cat([pattern_feature, short_context, long_context], dim=-1))
        gate_pattern, gate_short, gate_long = torch.chunk(scale_gates, chunks=3, dim=-1)
        gated_pattern = gate_pattern * pattern_feature
        gated_short = gate_short * short_context
        gated_long = gate_long * long_context

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

        sequence_2d_feature = self.sequence_2d_projection(self.sequence_2d_encoder(track_sequence.unsqueeze(1)))

        pooled_embedding = self.output_projection(
            torch.cat([
                sequence_mean,
                attentive_context,
                merged_summary,
                gated_pattern,
                sequence_2d_feature,
                gated_short,
                gated_long,
            ], dim=-1)
        )
        return token_embeddings, pooled_embedding


TrackGraphEncoder = TrackDGGCNEncoder
