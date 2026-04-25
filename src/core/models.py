from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SequenceClassifierOutput:
    logits: torch.Tensor


class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> SequenceClassifierOutput:
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, features.size(0), self.hidden_dim)
            last_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)
        else:
            last_hidden = hidden[-1]
        logits = self.head(self.norm(last_hidden))
        return SequenceClassifierOutput(logits=logits)
