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


def trajectory_consistency_score(model_output: Mapping[str, Any]) -> torch.Tensor:
    """Return a causal trajectory consistency score in ``[0, 1]``."""

    raw = model_output["raw_sequence"]
    trace = model_output["trace"]
    facts = model_output["event_facts_sequence"]
    padding = model_output["padding_mask"]
    active = ~padding
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    pair_valid = active & previous_active
    previous_raw = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, :-1]), dim=1)
    previous_update = torch.cat((torch.zeros_like(trace["actual_update"][:, :1]), trace["actual_update"][:, :-1]), dim=1)
    previous_facts = torch.cat((torch.zeros_like(facts[:, :1]), facts[:, :-1]), dim=1)

    def norm(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=-1).clamp_min(0.0).sqrt()

    def cosine(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        denominator = current.norm(dim=-1) * previous.norm(dim=-1)
        return ((current * previous).sum(dim=-1) / denominator.clamp_min(1.0e-6)).masked_fill(~pair_valid, 0.0).clamp(-1.0, 1.0)

    count = active.to(raw.dtype).cumsum(dim=1).clamp_min(1.0).unsqueeze(-1)
    prefix_mean = (raw * active.to(raw.dtype).unsqueeze(-1)).cumsum(dim=1) / count
    state_deviation = norm(raw - prefix_mean)
    event_delta = norm(facts - previous_facts).masked_fill(~pair_valid, 0.0)
    update_norm = norm(trace["actual_update"])
    score = 0.25 * (
        ((cosine(raw, previous_raw) + 1.0) * 0.5)
        + ((cosine(trace["actual_update"], previous_update) + 1.0) * 0.5)
        + torch.reciprocal(1.0 + state_deviation.clamp_min(0.0))
        + torch.reciprocal(1.0 + event_delta.clamp_min(0.0))
    )
    score = (score * pair_valid.to(raw.dtype) + 0.5 * (~pair_valid).to(raw.dtype)).masked_fill(padding, 0.0)
    if not torch.isfinite(score).all():
        raise FloatingPointError("V28 trajectory consistency contains non-finite values")
    return score.clamp(0.0, 1.0)


def route_knownness_score(
    target_score: torch.Tensor | np.ndarray,
    consistency: torch.Tensor | np.ndarray | None,
    *,
    route: str,
    consistency_weight: float,
) -> torch.Tensor | np.ndarray:
    """Combine target knownness with causal consistency on the Bird route."""

    if route != "BirdTarget" or consistency is None:
        return target_score
    weight = float(consistency_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("V28 Bird consistency weight must be within [0, 1]")
    return (1.0 - weight) * target_score + weight * consistency


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
        # L1 has two distinct open-set routes.  Keeping the two projections
        # separate prevents one scalar from being forced to model DBB and
        # Clutter with the same operating point.  L2 uses the target branch;
        # its route-specific thresholds are calibrated downstream.
        self.target_knownness_head = nn.Linear(projection_dim, 1)
        self.clutter_knownness_head = nn.Linear(projection_dim, 1)
        quality_input_dim = self.event_fact_dim * 2 + self.task_fact_dim
        self.quality_head = nn.Sequential(nn.Linear(quality_input_dim, quality_hidden), nn.SiLU(), nn.Linear(quality_hidden, 1))

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V28-head-contract-v3",
            "task": self.task,
            "knownness_input": "raw_causal_liquid_trace_plus_explicit_quality_facts",
            "knownness_input_dim": self.knownness_input_dim,
            "knownness_outputs": (
                "target_knownness_and_clutter_knownness"
                if self.task == V28_L1_TASK
                else "target_knownness"
            ),
            "quality_input": "causal_quality_facts_only",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
            "recurrent_auxiliary_branch": "none",
        }

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
        return {
            "knownness_features": features,
            "quality_features": quality_features,
            "trajectory_consistency_sequence": trajectory_consistency_score(model_output),
        }

    def forward_features(self, knownness_features: torch.Tensor, quality_features: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if knownness_features.ndim != 3 or knownness_features.shape[-1] != self.knownness_input_dim or quality_features.ndim != 3 or quality_features.shape[:2] != knownness_features.shape[:2] or padding_mask.shape != knownness_features.shape[:2]:
            raise ValueError("V28 auxiliary feature shapes differ")
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
            # L2 callers use this stable alias.  L1 calibration must use the
            # explicit route outputs above.
            "knownness_logits": target_knownness,
            "quality_logits": quality,
        }

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
    # ``tr_dt_valid`` means positive delta only.  A zero-delta event has a
    # defined Liquid V2 update and is semantically usable for quality/maturity
    # decisions.  Recompute the quality fact from the reason bits instead of
    # treating zero-delta observations as missing time information.
    active_count = active.to(dtype=event_facts.dtype).cumsum(dim=1).clamp_min(1.0)
    dt_semantic = (event_facts[..., 10] + event_facts[..., 11]).clamp(max=1.0)
    dt_fraction = (dt_semantic * active.to(dtype=event_facts.dtype)).cumsum(dim=1) / active_count
    aux_fraction = task_facts[..., 7] if task == V28_L1_TASK else torch.minimum(task_facts[..., 5], task_facts[..., 6])
    heads = context.config.heads
    return active & (
        (event_number >= int(heads["quality_min_events"]))
        & (rd_fraction >= float(heads["quality_min_rd_fraction"]))
        & (dt_fraction >= float(heads["quality_min_dt_fraction"]))
        & (aux_fraction >= float(heads["quality_min_aux_fraction"]))
    )


__all__ = ["V28TrajectoryHeads", "deterministic_quality_target", "route_knownness_score", "trajectory_consistency_score"]
