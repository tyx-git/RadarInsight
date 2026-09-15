"""Independent V28 knownness and quality/maturity heads.

The knownness branch consumes only the raw causal Liquid trajectory and
explicit event/task facts.  Classifier logits, softmax, labels and class
identity are deliberately absent from its input signature.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .protocol import V28Context, V28_L1_TASK


def _causal_mean_std(value: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = active.to(dtype=value.dtype).unsqueeze(-1)
    count = weights.cumsum(dim=1).clamp_min(1.0)
    mean = (value * weights).cumsum(dim=1) / count
    second = (value.square() * weights).cumsum(dim=1) / count
    return mean, (second - mean.square()).clamp_min(0.0).sqrt()


class V28TrajectoryHeads(nn.Module):
    """Independent auxiliary heads; no recurrent classifier branch is added."""

    def __init__(self, context: V28Context, *, task: str) -> None:
        super().__init__()
        model = context.config.model
        hidden = int(model["liquid_hidden_dim"])
        self.task = str(task)
        self.event_fact_dim = 19
        self.task_fact_dim = 8 if task == V28_L1_TASK else 7
        # Seven causal trajectories: raw state plus five Liquid diagnostics and
        # tau. Each contributes a prefix mean and std. Explicit facts add their
        # prefix mean and current-event value.
        self.knownness_input_dim = hidden * 14 + self.event_fact_dim * 2 + self.task_fact_dim
        projection_dim = int(context.config.heads["projection_dim"])
        hidden_dim = int(context.config.heads["knownness_hidden_dim"])
        quality_hidden = int(context.config.heads["quality_hidden_dim"])
        self.knownness_projection = nn.Sequential(nn.Linear(self.knownness_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, projection_dim), nn.LayerNorm(projection_dim), nn.SiLU())
        self.knownness_head = nn.Linear(projection_dim, 1)
        quality_input_dim = self.event_fact_dim * 2 + self.task_fact_dim
        self.quality_head = nn.Sequential(nn.Linear(quality_input_dim, quality_hidden), nn.SiLU(), nn.Linear(quality_hidden, 1))

    def model_contract(self) -> dict[str, Any]:
        return {"schema_version": "RADAR-LN3-V28-head-contract-v2", "task": self.task, "knownness_input": "raw_causal_liquid_trace_plus_explicit_quality_facts", "knownness_input_dim": self.knownness_input_dim, "quality_input": "causal_quality_facts_only", "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"], "recurrent_auxiliary_branch": "none"}

    def build_features(self, model_output: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        raw = model_output["raw_sequence"]
        trace = model_output["trace"]
        padding = model_output["padding_mask"]
        event_facts = model_output["event_facts_sequence"]
        task_facts = model_output["task_facts_sequence"]
        active = ~padding
        trajectories = [raw, trace["candidate_innovation"], trace["mixed_memory_innovation"], trace["actual_update"], trace["gate"], trace["alpha"], trace["tau"]]
        summaries: list[torch.Tensor] = []
        for trajectory in trajectories:
            mean, std = _causal_mean_std(trajectory, active)
            summaries.extend((mean, std))
        fact_mean, _ = _causal_mean_std(event_facts, active)
        features = torch.cat((*summaries, fact_mean, event_facts, task_facts), dim=-1)
        if features.shape[-1] != self.knownness_input_dim:
            raise RuntimeError(f"V28 knownness feature width differs: {features.shape[-1]} != {self.knownness_input_dim}")
        features = features.masked_fill(padding.unsqueeze(-1), 0.0)
        quality_features = torch.cat((fact_mean, event_facts, task_facts), dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
        return {"knownness_features": features, "quality_features": quality_features}

    def forward_features(self, knownness_features: torch.Tensor, quality_features: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if knownness_features.ndim != 3 or knownness_features.shape[-1] != self.knownness_input_dim or quality_features.ndim != 3 or quality_features.shape[:2] != knownness_features.shape[:2] or padding_mask.shape != knownness_features.shape[:2]:
            raise ValueError("V28 auxiliary feature shapes differ")
        projection = self.knownness_projection(knownness_features)
        knownness = self.knownness_head(projection).squeeze(-1)
        quality = self.quality_head(quality_features).squeeze(-1)
        projection = projection.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        knownness = knownness.masked_fill(padding_mask, 0.0)
        quality = quality.masked_fill(padding_mask, 0.0)
        return {"knownness_projection": projection, "knownness_logits": knownness, "quality_logits": quality}

    def forward(self, model_output: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        features = self.build_features(model_output)
        return {**features, **self.forward_features(features["knownness_features"], features["quality_features"], model_output["padding_mask"])}


def deterministic_quality_target(
    context: V28Context,
    *,
    task: str,
    task_facts: torch.Tensor,
    event_facts: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the causal quality/maturity fact gate for every prefix."""

    if task_facts.shape[-1] != (8 if task == V28_L1_TASK else 7) or event_facts.shape[-1] != 19:
        raise ValueError("V28 quality target fact dimensions differ")
    if task_facts.shape[:2] != padding_mask.shape or event_facts.shape[:2] != padding_mask.shape:
        raise ValueError("V28 quality target tensors have incompatible time shapes")
    active = ~padding_mask
    event_number = torch.arange(
        1,
        padding_mask.shape[1] + 1,
        device=padding_mask.device,
    ).unsqueeze(0)
    rd_fraction = task_facts[..., 1]
    dt_fraction = task_facts[..., 3]
    aux_fraction = task_facts[..., 7] if task == V28_L1_TASK else torch.minimum(task_facts[..., 5], task_facts[..., 6])
    heads = context.config.heads
    return active & (
        (event_number >= int(heads["quality_min_events"]))
        & (rd_fraction >= float(heads["quality_min_rd_fraction"]))
        & (dt_fraction >= float(heads["quality_min_dt_fraction"]))
        & (aux_fraction >= float(heads["quality_min_aux_fraction"]))
    )


__all__ = ["V28TrajectoryHeads", "deterministic_quality_target"]
