"""冻结 endpoint representation 上的独立 open-set knownness head。

该模块不读取 classifier logits，也不修改 L1/L2 endpoint。输入是已经由
冻结 endpoint 产生的 causal trajectory/features；每个 route 输出一个独立
的 binary knownness logit。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


KNOWNNESS_HEAD_SCHEMA = "RADAR-LN3-KNOWNNESS-HEAD-V7"
KNOWNNESS_HEAD_SCHEMA_V3 = "RADAR-LN3-KNOWNNESS-HEAD-V8"
KNOWNNESS_HEAD_SCHEMA_V4 = "RADAR-LN3-KNOWNNESS-HEAD-V9"
KNOWNNESS_HEAD_SCHEMA_V5 = "RADAR-LN3-KNOWNNESS-HEAD-V10"
KNOWNNESS_HEAD_SCHEMA_V6 = "RADAR-LN3-KNOWNNESS-HEAD-V11"
MICRO_DOPPLER_TEMPORAL_STABILITY_FEATURE_NAMES = (
    "prefix_mean_resolution_cell_peak_count",
    "prefix_mean_top4_resolution_cell_concentration",
    "prefix_std_pulse_envelope_rms",
    "prefix_std_modulation_600_1200hz_rms",
    "prefix_std_modulation_1200_2000hz_rms",
    "prefix_std_modulation_entropy",
)


def build_micro_doppler_temporal_stability_features(
    evidence_features: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Build six event-prefix summaries from current and previous valid RD events."""

    if (
        evidence_features.ndim != 3
        or evidence_features.shape[-1] != 47
        or padding_mask.shape != evidence_features.shape[:2]
        or padding_mask.dtype != torch.bool
        or not evidence_features.is_floating_point()
        or not torch.isfinite(evidence_features).all()
    ):
        raise ValueError("micro-Doppler temporal evidence must be finite [B,T,47]")
    if bool(evidence_features.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
        raise ValueError("padded micro-Doppler evidence must be zero")
    active = (evidence_features[..., 46] > 0.0) & ~padding_mask
    weights = active.to(dtype=evidence_features.dtype)
    count = weights.cumsum(dim=1)
    denominator = count.clamp_min(1.0)
    event_means = evidence_features[..., :23]

    def prefix_mean(index: int) -> torch.Tensor:
        values = event_means[..., index] * weights
        return values.cumsum(dim=1) / denominator

    def prefix_std(index: int) -> torch.Tensor:
        values = event_means[..., index]
        mean = (values * weights).cumsum(dim=1) / denominator
        second = (values.square() * weights).cumsum(dim=1) / denominator
        return (second - mean.square()).clamp_min(0.0).sqrt()

    features = torch.stack(
        (
            prefix_mean(12),
            prefix_mean(11),
            prefix_std(14),
            prefix_std(16),
            prefix_std(17),
            prefix_std(21),
        ),
        dim=-1,
    )
    has_evidence = count > 0.0
    return features.masked_fill(padding_mask.unsqueeze(-1) | ~has_evidence.unsqueeze(-1), 0.0)


def _causal_mean_std(value: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate prefix-only mean/std for a padded causal sequence."""

    weights = active.to(dtype=value.dtype).unsqueeze(-1)
    count = weights.cumsum(dim=1).clamp_min(1.0)
    mean = (value * weights).cumsum(dim=1) / count
    second = (value.square() * weights).cumsum(dim=1) / count
    return mean, (second - mean.square()).clamp_min(0.0).sqrt()


def build_temporal_knownness_features(
    event_trajectory: torch.Tensor,
    task_facts: torch.Tensor,
    padding_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the temporal knownness representation from frozen endpoint output.

    ``event_trajectory`` is the existing radar_ln3 causal layout:
    seven hidden trajectories followed by 19 event facts.  This keeps the
    package independent while keeping all features prefix-only.
    """

    if event_trajectory.ndim != 3 or task_facts.ndim != 3:
        raise ValueError("temporal inputs must be [B,T,D]")
    if event_trajectory.shape[:2] != task_facts.shape[:2] or padding_mask.shape != event_trajectory.shape[:2]:
        raise ValueError("temporal input time dimensions differ")
    if padding_mask.dtype != torch.bool or not torch.isfinite(event_trajectory).all() or not torch.isfinite(task_facts).all():
        raise ValueError("temporal inputs must be finite with a boolean padding mask")
    hidden = (int(event_trajectory.shape[-1]) - 19) // 7
    if hidden <= 0 or hidden * 7 + 19 != int(event_trajectory.shape[-1]):
        raise ValueError("event trajectory width differs from the registered layout")
    if bool(event_trajectory.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()) or bool(task_facts.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
        raise ValueError("padded temporal features must be zero")
    trajectories = [
        event_trajectory[..., index * hidden : (index + 1) * hidden]
        for index in range(7)
    ]
    event_facts = event_trajectory[..., 7 * hidden :]
    active = ~padding_mask
    summaries: list[torch.Tensor] = []
    for trajectory in trajectories:
        mean, std = _causal_mean_std(trajectory, active)
        summaries.extend((mean, std))
    fact_mean, _ = _causal_mean_std(event_facts, active)
    base = torch.cat((*summaries, fact_mean, event_facts, task_facts), dim=-1)

    raw, candidate, mixed, update, gate, alpha, tau = trajectories
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    pair_valid = active & previous_active
    previous_raw = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, :-1]), dim=1)
    previous_update = torch.cat((torch.zeros_like(update[:, :1]), update[:, :-1]), dim=1)
    previous_facts = torch.cat((torch.zeros_like(event_facts[:, :1]), event_facts[:, :-1]), dim=1)

    def norm(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=-1).clamp_min(0.0).sqrt()

    def cosine(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        denominator = current.norm(dim=-1) * previous.norm(dim=-1)
        return ((current * previous).sum(dim=-1) / denominator.clamp_min(1.0e-6)).masked_fill(~pair_valid, 0.0).clamp(-1.0, 1.0)

    prefix_mean, _ = _causal_mean_std(raw, active)
    update_norm = norm(update)
    cumulative_update = update_norm.cumsum(dim=1).clamp_min(1.0e-6)
    event_delta = norm(event_facts - previous_facts).masked_fill(~pair_valid, 0.0)
    scalars = torch.stack(
        (
            norm(raw),
            update_norm,
            norm(candidate),
            norm(mixed),
            cosine(raw, previous_raw),
            cosine(update, previous_update),
            norm(raw - prefix_mean),
            update_norm / cumulative_update,
            update_norm.cummax(dim=1).values / cumulative_update,
            cosine(event_facts, previous_facts),
            event_delta,
            gate.mean(dim=-1),
            alpha.mean(dim=-1),
            tau.mean(dim=-1),
            pair_valid.to(dtype=raw.dtype),
        ),
        dim=-1,
    ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    temporal_mean, temporal_std = _causal_mean_std(scalars, active)
    temporal = torch.cat((temporal_mean, temporal_std, scalars), dim=-1).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    knownness = torch.cat((base, temporal), dim=-1).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    if not torch.isfinite(knownness).all():
        raise FloatingPointError("temporal features contain non-finite values")
    return {
        "knownness_features": knownness,
        "temporal_features": temporal,
    }


def build_profile_knownness_features(
    rd_physical: torch.Tensor,
    rd_physical_valid: torch.Tensor,
    micro_doppler: torch.Tensor,
    micro_doppler_valid: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the 44-dimensional causal RD/micro-Doppler profile."""

    if rd_physical.ndim != 3 or micro_doppler.ndim != 3 or rd_physical.shape[:2] != micro_doppler.shape[:2] or padding_mask.shape != rd_physical.shape[:2]:
        raise ValueError("profile input shapes differ")
    if rd_physical_valid.shape != rd_physical.shape[:2] or micro_doppler_valid.shape != micro_doppler.shape[:2]:
        raise ValueError("profile validity shapes differ")
    active = ~padding_mask
    values = torch.cat((rd_physical, micro_doppler), dim=-1)
    valid = torch.cat(
        (
            rd_physical_valid.unsqueeze(-1).expand_as(rd_physical),
            micro_doppler_valid.unsqueeze(-1).expand_as(micro_doppler),
        ),
        dim=-1,
    ) & active.unsqueeze(-1)
    event = torch.cat(
        (
            values,
            rd_physical_valid.to(dtype=values.dtype).unsqueeze(-1),
            micro_doppler_valid.to(dtype=values.dtype).unsqueeze(-1),
        ),
        dim=-1,
    ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    mean, std = _causal_mean_std(event, active)
    previous = torch.cat((torch.zeros_like(event[:, :1]), event[:, :-1]), dim=1)
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    pair_valid = active & previous_active
    denominator = event.norm(dim=-1) * previous.norm(dim=-1)
    cosine = ((event * previous).sum(dim=-1) / denominator.clamp_min(1.0e-6)).masked_fill(~pair_valid, 0.0).clamp(-1.0, 1.0)
    delta = (event - previous).norm(dim=-1).masked_fill(~pair_valid, 0.0)
    profile = torch.cat((mean, std, event, cosine.unsqueeze(-1), delta.unsqueeze(-1)), dim=-1).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    if profile.shape[-1] != 44 or not torch.isfinite(profile).all():
        raise ValueError("profile feature width or values differ")
    return profile


def knownness_representation_contract(
    *,
    task: str,
    summary_dim: int,
    event_trajectory_dim: int,
    task_fact_dim: int,
    extra_dim: int,
) -> dict[str, Any]:
    """返回 head 输入 contract；明确禁止 logits/label 旁路。"""

    if task not in {"l1", "l2"}:
        raise ValueError("knownness representation task 必须为 l1 或 l2")
    dimensions = {
        "summary_features": int(summary_dim),
        "event_trajectory": int(event_trajectory_dim),
        "task_facts_sequence": int(task_fact_dim),
        "extra_features": int(extra_dim),
    }
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError("knownness representation dimensions 必须为正")
    return {
        "schema_version": "RADAR-LN3-L3-REPRESENTATION-V3",
        "task": task,
        "dimensions": dimensions,
        "extra_feature_names": [
            "rd_physical4",
            "rd_physical_valid",
            "micro_doppler8",
            "micro_doppler_valid",
            "stable8_summary8",
            "stable8_valid",
            "motion9",
        ],
        "causal_prefix_only": True,
        "future_events": "forbidden",
        "endpoint_logits": "forbidden",
        "softmax_values": "forbidden",
        "label_identity": "forbidden",
        "source": "frozen_endpoint_raw_liquid_trajectory_and_causal_facts",
    }


class TemporalKnownnessHead(nn.Module):
    """Temporal knownness head over frozen causal features."""

    def __init__(
        self,
        input_dim: int,
        *,
        route_names: Sequence[str],
        hidden_dim: int = 48,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        names = tuple(str(value) for value in route_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("knownness route names must be non-empty and unique")
        if min(int(input_dim), int(hidden_dim), int(projection_dim)) <= 0:
            raise ValueError("knownness dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.route_names = names
        self.knownness_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.SiLU(),
        )
        self.route_heads = nn.ModuleDict(
            {name: nn.Linear(self.projection_dim, 1) for name in names}
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": KNOWNNESS_HEAD_SCHEMA,
            "class_path": "radar_ln3.model.l3.knownness.TemporalKnownnessHead",
            "knownness_input_dim": self.input_dim,
            "knownness_hidden_dim": self.hidden_dim,
            "projection_dim": self.projection_dim,
            "route_names": list(self.route_names),
            "knownness_input": "frozen_causal_trace_plus_temporal_features",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
            "classifier_endpoint_frozen": True,
        }

    def forward_features(
        self,
        knownness_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            knownness_features.ndim != 3
            or knownness_features.shape[-1] != self.input_dim
            or padding_mask.shape != knownness_features.shape[:2]
            or padding_mask.dtype != torch.bool
        ):
            raise ValueError("knownness feature shapes differ")
        if not torch.isfinite(knownness_features).all():
            raise ValueError("knownness features must be finite")
        projection = self.knownness_projection(knownness_features)
        route_logits = torch.cat(
            [self.route_heads[name](projection) for name in self.route_names], dim=-1
        )
        projection = projection.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        route_logits = route_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return {
            "knownness_projection": projection,
            "route_logits": route_logits,
            "route_probabilities": torch.sigmoid(route_logits).masked_fill(padding_mask.unsqueeze(-1), 0.0),
        }

    def forward(
        self,
        knownness_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_features(knownness_features, padding_mask)

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "TemporalKnownnessHead":
        if contract.get("schema_version") != KNOWNNESS_HEAD_SCHEMA:
            raise ValueError("knownness contract schema differs")
        return cls(
            int(contract["knownness_input_dim"]),
            route_names=tuple(str(value) for value in contract["route_names"]),
            hidden_dim=int(contract["knownness_hidden_dim"]),
            projection_dim=int(contract["projection_dim"]),
        )


class ProfileResidualKnownnessHead(nn.Module):
    """Frozen temporal head plus a causal profile residual.

    旧 L2 head 只有一个 route。实验性按类别 head 可以为每个目标类别
    暴露独立 residual logit，同时保持旧 contract 可加载。
    """

    def __init__(self, baseline: TemporalKnownnessHead, *, profile_dim: int = 44) -> None:
        super().__init__()
        self.baseline = baseline
        self.profile_dim = int(profile_dim)
        self.profile_branch = nn.Sequential(
            nn.Linear(self.profile_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, len(baseline.route_names)),
        )
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": KNOWNNESS_HEAD_SCHEMA,
            "class_path": "radar_ln3.model.l3.knownness.ProfileResidualKnownnessHead",
            "profile_feature_dim": self.profile_dim,
            "baseline_contract": self.baseline.model_contract(),
            "baseline_frozen": True,
            "knownness_input": "frozen_temporal_head_plus_causal_rd_profile_residual",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
        }

    def forward(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if profile_features.ndim != 3 or profile_features.shape[-1] != self.profile_dim or profile_features.shape[:2] != knownness_features.shape[:2]:
            raise ValueError("profile residual feature shape differs")
        with torch.no_grad():
            baseline = self.baseline.forward_features(knownness_features, padding_mask)
        residual = self.profile_branch(profile_features).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        route_logits = baseline["route_logits"].clone()
        if route_logits.shape[-1] != residual.shape[-1]:
            raise ValueError("profile residual route width differs from baseline")
        route_logits = (route_logits + residual).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return {
            "knownness_projection": baseline["knownness_projection"].detach(),
            "route_logits": route_logits,
            "route_probabilities": torch.sigmoid(route_logits).masked_fill(padding_mask.unsqueeze(-1), 0.0),
            "profile_residual_logits": residual,
        }


class BirdProfileResidualKnownnessHead(ProfileResidualKnownnessHead):
    """在同一个 L2 knownness checkpoint 中增加 Bird/Unknown residual。

    原 temporal/profile 分支保持冻结和可复用；Bird residual 使用已有
    causal profile 加 representation contract 尾部的 32 维物理、微多普勒、
    stable8 和 motion 特征。它只在 L2 route 为 Bird 时由决策层使用。
    """

    def __init__(self, baseline: TemporalKnownnessHead, *, profile_dim: int = 44, extra_dim: int = 32) -> None:
        super().__init__(baseline, profile_dim=profile_dim)
        self.extra_dim = int(extra_dim)
        if self.extra_dim <= 0:
            raise ValueError("Bird residual extra_dim 必须为正数")
        self.bird_branch = nn.Sequential(
            nn.Linear(self.profile_dim + self.extra_dim, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.bird_residual_enabled = True
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.profile_branch.parameters():
            parameter.requires_grad_(False)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": KNOWNNESS_HEAD_SCHEMA_V3,
            "class_path": "radar_ln3.model.l3.knownness.BirdProfileResidualKnownnessHead",
            "profile_feature_dim": self.profile_dim,
            "extra_feature_dim": self.extra_dim,
            "baseline_contract": self.baseline.model_contract(),
            "baseline_frozen": True,
            "profile_branch_frozen": True,
            "bird_input": "causal_profile_plus_representation_contract_extra_tail",
            "forbidden_inputs": ["l1_logits", "l2_logits", "softmax", "label_identity"],
        }

    def bird_features(self, knownness_features: torch.Tensor, profile_features: torch.Tensor) -> torch.Tensor:
        if (
            knownness_features.ndim != 3
            or profile_features.ndim != 3
            or profile_features.shape[:2] != knownness_features.shape[:2]
            or profile_features.shape[-1] != self.profile_dim
            or knownness_features.shape[-1] < self.extra_dim
        ):
            raise ValueError("Bird residual feature shape differs")
        return torch.cat((profile_features, knownness_features[..., -self.extra_dim:]), dim=-1)

    def forward(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(knownness_features, profile_features, padding_mask)
        bird_logits = self.bird_branch(self.bird_features(knownness_features, profile_features))
        bird_logits = bird_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output["bird_residual_logits"] = bird_logits
        output["bird_residual_probabilities"] = torch.sigmoid(bird_logits).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        if self.bird_residual_enabled and output["route_probabilities"].shape[-1] == 1:
            base = output["route_probabilities"]
            bird = output["bird_residual_probabilities"]
            output["route_probabilities"] = torch.cat((base, bird, base), dim=-1)
        return output


class BirdProfileResidualFusionKnownnessHead(BirdProfileResidualKnownnessHead):
    """以原 Bird knownness logit 为基线的增量 head。"""

    def __init__(self, baseline: TemporalKnownnessHead, *, profile_dim: int = 44, extra_dim: int = 32) -> None:
        super().__init__(baseline, profile_dim=profile_dim, extra_dim=extra_dim)
        nn.init.zeros_(self.bird_branch[-1].weight)
        nn.init.zeros_(self.bird_branch[-1].bias)

    def model_contract(self) -> dict[str, Any]:
        contract = super().model_contract()
        contract["schema_version"] = KNOWNNESS_HEAD_SCHEMA_V4
        contract["class_path"] = "radar_ln3.model.l3.knownness.BirdProfileResidualFusionKnownnessHead"
        contract["fusion"] = "sigmoid(logit(baseline_bird_score) + bird_residual_logit)"
        return contract

    def forward(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = ProfileResidualKnownnessHead.forward(
            self, knownness_features, profile_features, padding_mask
        )
        residual = self.bird_branch(self.bird_features(knownness_features, profile_features))
        base_logits = output["route_logits"]
        base_probability = output["route_probabilities"][..., :1].clamp(1.0e-5, 1.0 - 1.0e-5)
        fused_bird = torch.logit(base_probability) + residual
        fused_bird = fused_bird.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        route_logits = torch.cat((base_logits, fused_bird, base_logits), dim=-1)
        route_probabilities = torch.sigmoid(route_logits).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        output["route_logits"] = route_logits
        output["route_probabilities"] = route_probabilities
        output["bird_residual_logits"] = residual.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output["bird_residual_probabilities"] = torch.sigmoid(
            output["bird_residual_logits"]
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return output


class BirdProfileResidualEvidenceFusionKnownnessHead(BirdProfileResidualFusionKnownnessHead):
    """在 V4 Bird 残差上增加完整微多普勒证据摘要。

    temporal/profile baseline 与 V4 的 Bird residual 都保持冻结语义；新增分支
    只读取因果的 23 维 evidence 均值、标准差和有效帧比例，并以零初始化
    残差起步，确保训练开始时与原 Bird baseline 完全一致。
    """

    def __init__(
        self,
        baseline: TemporalKnownnessHead,
        *,
        profile_dim: int = 44,
        extra_dim: int = 32,
        evidence_dim: int = 47,
    ) -> None:
        super().__init__(baseline, profile_dim=profile_dim, extra_dim=extra_dim)
        self.evidence_dim = int(evidence_dim)
        if self.evidence_dim <= 0:
            raise ValueError("Bird evidence residual evidence_dim 必须为正数")
        self.evidence_branch = nn.Sequential(
            nn.Linear(self.evidence_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        # 证据残差从零开始，保留 V4 的完整 Bird profile baseline。
        nn.init.zeros_(self.evidence_branch[-1].weight)
        nn.init.zeros_(self.evidence_branch[-1].bias)
        self.bird_residual_enabled = True
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.profile_branch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.bird_branch.parameters():
            parameter.requires_grad_(False)

    def model_contract(self) -> dict[str, Any]:
        contract = super().model_contract()
        contract["schema_version"] = KNOWNNESS_HEAD_SCHEMA_V5
        contract["class_path"] = (
            "radar_ln3.model.l3.knownness."
            "BirdProfileResidualEvidenceFusionKnownnessHead"
        )
        contract["evidence_feature_dim"] = self.evidence_dim
        contract["bird_input"] = (
            "causal_profile_plus_representation_contract_extra_tail_plus_"
            "causal_micro_doppler_evidence_summary"
        )
        contract["evidence_summary"] = "23-dim mean + 23-dim std + valid-frame-fraction"
        contract["evidence_branch_zero_initialized"] = True
        return contract

    def bird_features_with_evidence(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        evidence_features: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_features.ndim != 3 or evidence_features.shape[:2] != knownness_features.shape[:2]:
            raise ValueError("Bird evidence residual evidence shape differs")
        if evidence_features.shape[-1] != self.evidence_dim:
            raise ValueError("Bird evidence residual evidence width differs")
        return evidence_features

    def forward(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
        evidence_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(
            knownness_features, profile_features, padding_mask
        )
        residual = self.evidence_branch(
            self.bird_features_with_evidence(
                knownness_features, profile_features, evidence_features
            )
        )
        base_logits = output["route_logits"]
        fused_bird = (base_logits[..., 1:2] + residual).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        route_logits = torch.cat(
            (base_logits[..., 0:1], fused_bird, base_logits[..., 2:3]), dim=-1
        )
        route_probabilities = torch.sigmoid(route_logits).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        output["route_logits"] = route_logits
        output["route_probabilities"] = route_probabilities
        output["evidence_residual_logits"] = residual.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        output["evidence_residual_probabilities"] = torch.sigmoid(
            output["evidence_residual_logits"]
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return output


class BirdTemporalStabilityResidualKnownnessHead(
    BirdProfileResidualEvidenceFusionKnownnessHead
):
    """在 V5 Bird score 上增加六个冻结基线的因果前缀统计残差。"""

    def __init__(
        self,
        baseline: TemporalKnownnessHead,
        *,
        profile_dim: int = 44,
        extra_dim: int = 32,
        evidence_dim: int = 47,
        standardization_mean: Sequence[float],
        standardization_scale: Sequence[float],
        ridge_alpha: float,
        cv_fold_count: int,
    ) -> None:
        super().__init__(
            baseline,
            profile_dim=profile_dim,
            extra_dim=extra_dim,
            evidence_dim=evidence_dim,
        )
        mean = torch.as_tensor(standardization_mean, dtype=torch.float32)
        scale = torch.as_tensor(standardization_scale, dtype=torch.float32)
        if (
            mean.shape != (len(MICRO_DOPPLER_TEMPORAL_STABILITY_FEATURE_NAMES),)
            or scale.shape != mean.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or bool((scale <= 0.0).any())
        ):
            raise ValueError("micro-Doppler temporal feature standardization 无效")
        if not torch.isfinite(torch.tensor(float(ridge_alpha))) or float(ridge_alpha) <= 0.0:
            raise ValueError("L2_18 ridge alpha 必须为正有限值")
        if int(cv_fold_count) < 2:
            raise ValueError("L2_18 cross-validation 至少需要两折")
        self.register_buffer("temporal_standardization_mean", mean)
        self.register_buffer("temporal_standardization_scale", scale)
        self.ridge_alpha = float(ridge_alpha)
        self.cv_fold_count = int(cv_fold_count)
        self.temporal_residual = nn.Linear(
            len(MICRO_DOPPLER_TEMPORAL_STABILITY_FEATURE_NAMES), 1
        )
        nn.init.zeros_(self.temporal_residual.weight)
        nn.init.zeros_(self.temporal_residual.bias)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.profile_branch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.bird_branch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.evidence_branch.parameters():
            parameter.requires_grad_(False)

    def model_contract(self) -> dict[str, Any]:
        contract = super().model_contract()
        contract["schema_version"] = KNOWNNESS_HEAD_SCHEMA_V6
        contract["class_path"] = (
            "radar_ln3.model.l3.knownness."
            "BirdTemporalStabilityResidualKnownnessHead"
        )
        contract["temporal_residual_feature_names"] = list(
            MICRO_DOPPLER_TEMPORAL_STABILITY_FEATURE_NAMES
        )
        contract["temporal_residual_aggregation"] = (
            "inclusive_causal_prefix_over_valid_event_level_23d_frame_means"
        )
        contract["temporal_residual_training_view"] = (
            "train_only_l1_target_and_l2_bird_predictions"
        )
        contract["temporal_residual_fit"] = "stratified_parent_5_fold_ridge_logistic"
        contract["temporal_residual_ridge_alpha"] = self.ridge_alpha
        contract["temporal_residual_cv_fold_count"] = self.cv_fold_count
        contract["temporal_residual_standardization_mean"] = (
            self.temporal_standardization_mean.detach().cpu().tolist()
        )
        contract["temporal_residual_standardization_scale"] = (
            self.temporal_standardization_scale.detach().cpu().tolist()
        )
        contract["temporal_residual_zero_initialized_before_fit"] = True
        return contract

    def forward(
        self,
        knownness_features: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
        evidence_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(
            knownness_features,
            profile_features,
            padding_mask,
            evidence_features,
        )
        temporal_features = build_micro_doppler_temporal_stability_features(
            evidence_features, padding_mask
        )
        normalized = (
            temporal_features - self.temporal_standardization_mean
        ) / self.temporal_standardization_scale
        has_evidence = (evidence_features[..., 46] > 0.0) & ~padding_mask
        has_evidence = has_evidence.cumsum(dim=1) > 0
        normalized = normalized.masked_fill(
            padding_mask.unsqueeze(-1) | ~has_evidence.unsqueeze(-1), 0.0
        )
        residual = self.temporal_residual(normalized).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        base_logits = output["route_logits"]
        bird_logits = (base_logits[..., 1:2] + residual).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        route_logits = torch.cat(
            (base_logits[..., 0:1], bird_logits, base_logits[..., 2:3]), dim=-1
        )
        route_probabilities = torch.sigmoid(route_logits).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        output["route_logits"] = route_logits
        output["route_probabilities"] = route_probabilities
        output["temporal_residual_logits"] = residual
        output["temporal_residual_probabilities"] = torch.sigmoid(residual).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return output


__all__ = [
    "KNOWNNESS_HEAD_SCHEMA",
    "KNOWNNESS_HEAD_SCHEMA_V3",
    "KNOWNNESS_HEAD_SCHEMA_V4",
    "KNOWNNESS_HEAD_SCHEMA_V5",
    "KNOWNNESS_HEAD_SCHEMA_V6",
    "BirdProfileResidualKnownnessHead",
    "BirdProfileResidualFusionKnownnessHead",
    "BirdProfileResidualEvidenceFusionKnownnessHead",
    "BirdTemporalStabilityResidualKnownnessHead",
    "MICRO_DOPPLER_TEMPORAL_STABILITY_FEATURE_NAMES",
    "build_micro_doppler_temporal_stability_features",
    "TemporalKnownnessHead",
    "ProfileResidualKnownnessHead",
    "build_temporal_knownness_features",
    "build_profile_knownness_features",
    "knownness_representation_contract",
]
