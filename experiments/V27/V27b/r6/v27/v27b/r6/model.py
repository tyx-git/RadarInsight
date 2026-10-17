"""Small independent knownness heads used by the V27b-r6 ablation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .protocol import R6Config


class R6KnownnessHead(nn.Module):
    """Read frozen causal features without ever consuming classifier logits.

    Both arms have the same trajectory and summary capacity.  The treatment
    alone receives the deterministic stable8 stability feature; this keeps the
    treatment variable explicit and prevents the V27a classifier residual from
    being silently reused as an OOD score.
    """

    def __init__(self, config: R6Config, *, arm: str) -> None:
        super().__init__()
        if arm not in {"control", "stable8_treatment"}:
            raise ValueError("V27b-r6 knownness arm is invalid")
        self.config = config
        self.arm = arm
        model = config.model
        self.event_projection = nn.Sequential(
            nn.Linear(config.data.event_trajectory_dim, model.trajectory_projection_dim),
            nn.LayerNorm(model.trajectory_projection_dim),
            nn.SiLU(),
        )
        self.trajectory_encoder = nn.GRU(
            model.trajectory_projection_dim,
            model.trajectory_hidden_dim,
            batch_first=True,
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(config.data.summary_feature_dim + config.data.task_fact_dim, model.summary_projection_dim),
            nn.LayerNorm(model.summary_projection_dim),
            nn.SiLU(),
        )
        self.stable_feature_dim = config.data.stable8_dim * 4 + 6
        # Both arms own an identical projection and fusion width.  Control
        # receives a deterministic all-zero stable input so any difference is
        # attributable to reliable stable8 evidence, rather than extra capacity.
        self.stable_projection = nn.Sequential(
            nn.Linear(self.stable_feature_dim, model.stable_projection_dim),
            nn.LayerNorm(model.stable_projection_dim),
            nn.SiLU(),
        )
        fusion_dim = model.trajectory_hidden_dim + model.summary_projection_dim
        fusion_dim += model.stable_projection_dim
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, model.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.fusion_hidden_dim, 1),
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27b-r6-knownness-head-v1",
            "experiment_id": "V27b-r6",
            "arm": self.arm,
            "event_trajectory_dim": self.config.data.event_trajectory_dim,
            "summary_feature_dim": self.config.data.summary_feature_dim,
            "task_fact_dim": self.config.data.task_fact_dim,
            "stable_feature_dim": self.stable_feature_dim,
            "classifier_logits_input": False,
            "frozen_classifier_endpoint": "not_an_input",
            "stable8_input": self.arm == "stable8_treatment",
            "stable8_control_input": "all_zero_matched_capacity",
            "stable8_reliability": "frame_feature_reliable_not_rd_present",
            "causality": "prefix_local_features_only",
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        event_trajectory: torch.Tensor,
        summary_features: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        stable_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            event_trajectory.ndim != 3
            or event_trajectory.shape[-1] != self.config.data.event_trajectory_dim
            or summary_features.shape
            != (*event_trajectory.shape[:2], self.config.data.summary_feature_dim)
            or task_facts_sequence.shape
            != (*event_trajectory.shape[:2], self.config.data.task_fact_dim)
            or stable_features.shape
            != (*event_trajectory.shape[:2], self.stable_feature_dim)
            or padding_mask.shape != event_trajectory.shape[:2]
            or padding_mask.dtype != torch.bool
        ):
            raise ValueError("V27b-r6 knownness feature shapes differ")
        values = (
            event_trajectory,
            summary_features,
            task_facts_sequence,
            stable_features,
        )
        if not all(value.is_floating_point() and torch.isfinite(value).all() for value in values):
            raise ValueError("V27b-r6 knownness features must be finite floats")
        if any(value.device != event_trajectory.device for value in values):
            raise ValueError("V27b-r6 knownness features must share a device")
        if any(value.dtype != event_trajectory.dtype for value in values):
            raise ValueError("V27b-r6 knownness features must share a dtype")
        for value in values:
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError("V27b-r6 padded knownness features must be zero")
        lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)
        if bool((lengths <= 0).any()):
            raise ValueError("V27b-r6 knownness received an empty parent")
        projected_events = self.event_projection(event_trajectory)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected_events,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.trajectory_encoder(packed)
        trajectory = hidden[-1]
        rows = torch.arange(lengths.shape[0], device=lengths.device)
        final = lengths - 1
        summary = self.summary_projection(
            torch.cat((summary_features[rows, final], task_facts_sequence[rows, final]), dim=-1)
        )
        stable_input = stable_features[rows, final]
        if self.arm == "control":
            stable_input = torch.zeros_like(stable_input)
        pieces = [trajectory, summary, self.stable_projection(stable_input)]
        return self.output(torch.cat(pieces, dim=-1)).squeeze(-1)


def build_r6_model(config: R6Config, *, arm: str, seed: int) -> R6KnownnessHead:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27b-r6 initialization seed is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return R6KnownnessHead(config, arm=arm)


__all__ = ["R6KnownnessHead", "build_r6_model"]
