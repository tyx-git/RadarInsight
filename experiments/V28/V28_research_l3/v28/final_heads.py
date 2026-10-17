"""Final V28 auxiliary heads used by the frozen research L3 chain."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .protocol import V28Context


class FinalTemporalKnownnessHead(nn.Module):
    """Knownness and quality heads over causal Liquid summaries."""

    def __init__(self, context: V28Context, *, task: str, input_dim: int, temporal_dim: int, quality_input_dim: int) -> None:
        super().__init__()
        self.task = str(task)
        self.input_dim = int(input_dim)
        self.temporal_dim = int(temporal_dim)
        self.knownness_input_dim = self.input_dim
        hidden_dim = int(context.config.heads["knownness_hidden_dim"])
        projection_dim = int(context.config.heads["projection_dim"])
        quality_hidden = int(context.config.heads["quality_hidden_dim"])
        self.knownness_projection = nn.Sequential(
            nn.Linear(self.knownness_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
        )
        self.target_knownness_head = nn.Linear(projection_dim, 1)
        self.clutter_knownness_head = nn.Linear(projection_dim, 1)
        self.quality_head = nn.Sequential(
            nn.Linear(int(quality_input_dim), quality_hidden),
            nn.SiLU(),
            nn.Linear(quality_hidden, 1),
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V28-knownness-temporal-head-v3",
            "task": self.task,
            "knownness_input": "baseline_causal_trace_plus_temporal_consistency_summaries",
            "knownness_input_dim": self.knownness_input_dim,
            "knownness_outputs": "target_knownness_and_clutter_knownness" if self.task == "l1_known_gate" else "target_knownness",
            "temporal_suffix_dim": self.temporal_dim,
            "quality_input": "baseline_causal_quality_facts_only",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
            "classifier_endpoint_frozen": True,
        }

    def forward_features(self, knownness_features: torch.Tensor, quality_features: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if knownness_features.ndim != 3 or knownness_features.shape[-1] != self.knownness_input_dim:
            raise ValueError("final temporal knownness feature shape differs")
        if quality_features.ndim != 3 or quality_features.shape[:2] != knownness_features.shape[:2] or padding_mask.shape != knownness_features.shape[:2]:
            raise ValueError("final temporal auxiliary feature shapes differ")
        projection = self.knownness_projection(knownness_features)
        target_knownness = self.target_knownness_head(projection).squeeze(-1)
        clutter_knownness = self.clutter_knownness_head(projection).squeeze(-1)
        quality = self.quality_head(quality_features).squeeze(-1)
        projection = projection.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        target_knownness = target_knownness.masked_fill(padding_mask, 0.0)
        clutter_knownness = clutter_knownness.masked_fill(padding_mask, 0.0)
        quality = quality.masked_fill(padding_mask, 0.0)
        return {
            "knownness_projection": projection,
            "target_knownness_logits": target_knownness,
            "clutter_knownness_logits": clutter_knownness,
            "knownness_logits": target_knownness,
            "quality_logits": quality,
        }


class FinalL2ProfileResidualHead(nn.Module):
    """Frozen final temporal head plus the selected causal RD/profile residual."""

    def __init__(self, baseline: nn.Module, *, profile_dim: int, hidden_dim: int = 32, branch_hidden_dim: int = 16) -> None:
        super().__init__()
        self.baseline = baseline
        self.profile_dim = int(profile_dim)
        if self.profile_dim <= 0:
            raise ValueError("final L2 profile dimension must be positive")
        self.profile_branch = nn.Sequential(
            nn.Linear(self.profile_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(branch_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(branch_hidden_dim), 1),
        )
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V28-knownness-l2-profile-head-v4",
            "knownness_input": "frozen_temporal_norm_head_plus_causal_l2_event_profile_residual",
            "profile_feature_dim": self.profile_dim,
            "profile_sources": ["rd_physical_features", "micro_doppler_features", "rd_physical_valid", "micro_doppler_valid"],
            "baseline_frozen": True,
            "quality_head": "frozen_baseline_diagnostic_only",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
        }

    def forward_features(self, knownness_features: torch.Tensor, quality_features: torch.Tensor, padding_mask: torch.Tensor, profile_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if knownness_features.ndim != 3 or quality_features.ndim != 3 or profile_features.ndim != 3:
            raise ValueError("final L2 profile feature tensors must be [B,T,D]")
        if knownness_features.shape[:2] != quality_features.shape[:2] or profile_features.shape[:2] != knownness_features.shape[:2] or profile_features.shape[-1] != self.profile_dim or padding_mask.shape != knownness_features.shape[:2]:
            raise ValueError("final L2 profile feature shapes differ")
        self.baseline.eval()
        with torch.no_grad():
            base = self.baseline.forward_features(knownness_features, quality_features, padding_mask)
        residual = self.profile_branch(profile_features).squeeze(-1).masked_fill(padding_mask, 0.0)
        knownness = (base["knownness_logits"] + residual).masked_fill(padding_mask, 0.0)
        return {
            "knownness_projection": base["knownness_projection"].detach(),
            "knownness_logits": knownness,
            "quality_logits": base["quality_logits"].detach(),
            "profile_residual_logits": residual,
        }


__all__ = ["FinalTemporalKnownnessHead", "FinalL2ProfileResidualHead"]
