from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


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
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
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


class TrackTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
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

    def forward(self, track_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(track_sequence)
        temporal_features = self.temporal_network(projected.transpose(1, 2)).transpose(1, 2)
        position_encoding = self._position_encoding(temporal_features.size(1))
        transformer_output = self.transformer(temporal_features + position_encoding)

        token_embeddings = self.token_projection(transformer_output)
        attention_scores = self.attention(token_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attentive_context = (attention_weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
        sequence_mean = token_embeddings.mean(dim=1)

        raw_mean = track_sequence.mean(dim=1)
        raw_std = track_sequence.std(dim=1, unbiased=False)
        raw_min = track_sequence.min(dim=1).values
        raw_max = track_sequence.max(dim=1).values
        summary_feature = self.summary_projection(torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=-1))

        pooled_embedding = self.output_projection(
            torch.cat([sequence_mean, attentive_context, summary_feature], dim=-1)
        )
        return token_embeddings, pooled_embedding
