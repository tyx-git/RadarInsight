"""Independent causal Bird-route knownness head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BirdRouteKnownnessHead(nn.Module):
    """Causal GRU and 32D projection for Bird versus Bird-route Unknown."""

    def __init__(self, *, hidden_dim: int = 48, projection_dim: int = 32, stable8_enabled: bool) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.stable8_enabled = bool(stable8_enabled)
        self.summary_dim = 672
        self.event_dim = 355
        self.task_fact_dim = 7
        self.stable8_dim = 38
        input_dim = self.event_dim + self.task_fact_dim + (self.stable8_dim if self.stable8_enabled else 0)
        projection_input_dim = self.summary_dim + self.hidden_dim + self.task_fact_dim + (self.stable8_dim if self.stable8_enabled else 0)
        self.trajectory_encoder = nn.GRU(input_dim, self.hidden_dim, batch_first=True)
        self.knownness_projection = nn.Sequential(
            nn.Linear(projection_input_dim, self.projection_dim),
            nn.SiLU(),
        )
        self.knownness_head = nn.Linear(self.projection_dim, 1)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r5-bird-head-v1",
            "experiment_id": "V27d-r5-bird-route",
            "head": "bird_route_knownness",
            "hidden_dim": self.hidden_dim,
            "projection_dim": self.projection_dim,
            "summary_dim": self.summary_dim,
            "event_trajectory_dim": self.event_dim,
            "task_fact_dim": self.task_fact_dim,
            "stable8_dim": self.stable8_dim,
            "stable8_enabled": self.stable8_enabled,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "route_used_as_continuous_input": False,
            "padding_policy": "suffix_masked_packed_sequence",
        }

    def forward(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        stable8_causal_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, length = padding_mask.shape
        expected = {
            "summary_features": (batch, length, self.summary_dim),
            "event_trajectory": (batch, length, self.event_dim),
            "task_facts_sequence": (batch, length, self.task_fact_dim),
            "stable8_causal_features": (batch, length, self.stable8_dim),
        }
        values = {
            "summary_features": summary_features,
            "event_trajectory": event_trajectory,
            "task_facts_sequence": task_facts_sequence,
            "stable8_causal_features": stable8_causal_features,
        }
        if padding_mask.dtype != torch.bool or padding_mask.ndim != 2:
            raise ValueError("V27d-r5 padding mask is invalid")
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"V27d-r5 head input differs: {name}")
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27d-r5 padded input is non-zero: {name}")
        lengths = (~padding_mask).long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V27d-r5 head received an empty sequence")
        pieces = [event_trajectory, task_facts_sequence]
        if self.stable8_enabled:
            pieces.append(stable8_causal_features)
        trajectory_input = torch.cat(pieces, dim=-1)
        packed = pack_padded_sequence(
            trajectory_input,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.trajectory_encoder(packed)
        trajectory, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=length)
        projection_parts = [summary_features, trajectory, task_facts_sequence]
        if self.stable8_enabled:
            projection_parts.append(stable8_causal_features)
        projection = self.knownness_projection(torch.cat(projection_parts, dim=-1))
        logit = self.knownness_head(projection).squeeze(-1)
        projection = projection.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        trajectory = trajectory.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        logit = logit.masked_fill(padding_mask, 0.0)
        if not all(torch.isfinite(value).all() for value in (projection, trajectory, logit)):
            raise FloatingPointError("V27d-r5 head output is non-finite")
        rows = torch.arange(batch, device=padding_mask.device)
        return {
            "projection_sequence": projection,
            "trajectory_sequence": trajectory,
            "knownness_logit_sequence": logit,
            "projection_final": projection[rows, lengths - 1],
            "knownness_logit_final": logit[rows, lengths - 1],
        }


def build_head(*, stable8_enabled: bool, seed: int, hidden_dim: int = 48, projection_dim: int = 32) -> BirdRouteKnownnessHead:
    if int(seed) < 0:
        raise ValueError("V27d-r5 seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return BirdRouteKnownnessHead(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            stable8_enabled=stable8_enabled,
        )


__all__ = ["BirdRouteKnownnessHead", "build_head"]
