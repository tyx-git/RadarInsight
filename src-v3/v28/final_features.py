"""Final causal features for the V28 knownness heads."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .heads import V28TrajectoryHeads


FEATURE_VERSION = "RADAR-LN3-V28-final-knownness-temporal-features-v1"
TEMPORAL_SUFFIX_DIM = 45
PROFILE_RAW_DIM = 14
PROFILE_DIM = PROFILE_RAW_DIM * 3 + 2
PROFILE_NAMES = (
    "rd_physical_0", "rd_physical_1", "rd_physical_2", "rd_physical_3",
    "micro_doppler_0", "micro_doppler_1", "micro_doppler_2", "micro_doppler_3",
    "micro_doppler_4", "micro_doppler_5", "micro_doppler_6", "micro_doppler_7",
    "rd_physical_valid", "micro_doppler_valid",
)
PROFILE_FEATURE_NAMES = (
    *(f"{name}_prefix_mean" for name in PROFILE_NAMES),
    *(f"{name}_prefix_std" for name in PROFILE_NAMES),
    *(f"{name}_current" for name in PROFILE_NAMES),
    "profile_adjacent_cosine", "profile_adjacent_delta_norm",
)


def concat_batches(arrays: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length causal batches to one deterministic tensor cache."""

    if not arrays or any(not values for values in arrays.values()):
        raise ValueError("final representation batches are empty")
    max_length = max(int(value.shape[1]) for value in arrays["knownness_features"])
    result: dict[str, torch.Tensor] = {}
    for name, values in arrays.items():
        if name == "lengths":
            result[name] = torch.cat(values, dim=0)
            continue
        padded: list[torch.Tensor] = []
        for value in values:
            if value.ndim < 2 or int(value.shape[1]) > max_length:
                raise ValueError(f"final representation shape differs: {name}")
            width = max_length - int(value.shape[1])
            if width:
                suffix_shape = (int(value.shape[0]), width, *value.shape[2:])
                suffix = torch.full(suffix_shape, True if name == "padding_mask" else 0, dtype=value.dtype)
                value = torch.cat((value, suffix), dim=1)
            padded.append(value)
        result[name] = torch.cat(padded, dim=0)
    return result


def _norm(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean(dim=-1).clamp_min(0.0).sqrt()


def _cosine(current: torch.Tensor, previous: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denominator = current.norm(dim=-1) * previous.norm(dim=-1)
    value = (current * previous).sum(dim=-1) / denominator.clamp_min(1.0e-6)
    return value.masked_fill(~valid, 0.0).clamp(-1.0, 1.0)


def causal_mean_std(value: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = active.to(dtype=value.dtype)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    count = weights.cumsum(dim=1).clamp_min(1.0)
    mean = (value * weights).cumsum(dim=1) / count
    second = (value.square() * weights).cumsum(dim=1) / count
    return mean, (second - mean.square()).clamp_min(0.0).sqrt()


def _temporal_scalars(model_output: Mapping[str, Any]) -> tuple[torch.Tensor, tuple[str, ...]]:
    raw = model_output["raw_sequence"]
    trace = model_output["trace"]
    event_facts = model_output["event_facts_sequence"]
    padding = model_output["padding_mask"]
    active = ~padding
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    pair_valid = active & previous_active
    previous_raw = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, :-1]), dim=1)
    previous_update = torch.cat((torch.zeros_like(trace["actual_update"][:, :1]), trace["actual_update"][:, :-1]), dim=1)
    previous_facts = torch.cat((torch.zeros_like(event_facts[:, :1]), event_facts[:, :-1]), dim=1)
    update_norm = _norm(trace["actual_update"])
    cumulative_update = update_norm.cumsum(dim=1).clamp_min(1.0e-6)
    prefix_state_mean, _ = causal_mean_std(raw, active)
    state_prefix_deviation = _norm(raw - prefix_state_mean)
    event_fact_delta = _norm(event_facts - previous_facts)
    values = torch.stack([
        _norm(raw), update_norm, _norm(trace["candidate_innovation"]), _norm(trace["mixed_memory_innovation"]),
        _cosine(raw, previous_raw, pair_valid), _cosine(trace["actual_update"], previous_update, pair_valid),
        state_prefix_deviation, update_norm / cumulative_update,
        update_norm.cummax(dim=1).values / cumulative_update,
        _cosine(event_facts, previous_facts, pair_valid), event_fact_delta.masked_fill(~pair_valid, 0.0),
        trace["gate"].mean(dim=-1), trace["alpha"].mean(dim=-1), trace["tau"].mean(dim=-1), pair_valid.to(raw.dtype),
    ], dim=-1).masked_fill(~active.unsqueeze(-1), 0.0)
    names = (
        "state_norm", "actual_update_norm", "candidate_innovation_norm", "mixed_innovation_norm",
        "state_adjacent_cosine", "update_adjacent_cosine", "state_prefix_deviation", "current_update_share",
        "largest_update_share", "event_fact_adjacent_cosine", "event_fact_delta_norm", "gate_mean", "alpha_mean",
        "tau_mean", "adjacent_pair_valid",
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("final temporal features contain non-finite values")
    return values, names


def build_temporal_features(model_output: Mapping[str, Any], base_heads: V28TrajectoryHeads) -> dict[str, torch.Tensor | tuple[str, ...]]:
    base = base_heads.build_features(model_output)
    scalars, names = _temporal_scalars(model_output)
    active = ~model_output["padding_mask"]
    mean, std = causal_mean_std(scalars, active)
    temporal = torch.cat((mean, std, scalars), dim=-1).masked_fill(model_output["padding_mask"].unsqueeze(-1), 0.0)
    return {
        "knownness_features": torch.cat((base["knownness_features"], temporal), dim=-1),
        "quality_features": base["quality_features"],
        "temporal_features": temporal,
        "temporal_feature_names": names,
    }


def raw_profile(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padding = batch["tr_padding_mask"].bool()
    active = ~padding
    physical = batch["rd_physical_features"].float()
    micro = batch["micro_doppler_features"].float()
    physical_valid = batch["rd_physical_valid"].bool()
    micro_valid = batch["micro_doppler_valid"].bool()
    values = torch.cat((physical, micro), dim=-1)
    valid = torch.cat((physical_valid.unsqueeze(-1).expand_as(physical), micro_valid.unsqueeze(-1).expand_as(micro)), dim=-1)
    event = torch.cat((values, physical_valid.to(values.dtype).unsqueeze(-1), micro_valid.to(values.dtype).unsqueeze(-1)), dim=-1)
    event = event.masked_fill(~active.unsqueeze(-1), 0.0)
    valid = valid & active.unsqueeze(-1)
    if event.shape[-1] != PROFILE_RAW_DIM or not torch.isfinite(event).all() or bool(event.masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("final L2 raw profile violates shape, finite or padding contract")
    return event, valid, padding


def profile_features(event: torch.Tensor, valid: torch.Tensor, padding: torch.Tensor, means: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    normalized = torch.where(valid, (event[..., :12] - means.view(1, 1, -1)) / scales.view(1, 1, -1), torch.zeros_like(event[..., :12]))
    profile_event = torch.cat((normalized, event[..., 12:]), dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
    active = ~padding
    mean, std = causal_mean_std(profile_event, active)
    previous = torch.cat((torch.zeros_like(profile_event[:, :1]), profile_event[:, :-1]), dim=1)
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    pair_valid = active & previous_active
    denominator = profile_event.norm(dim=-1) * previous.norm(dim=-1)
    cosine = ((profile_event * previous).sum(dim=-1) / denominator.clamp_min(1.0e-6)).masked_fill(~pair_valid, 0.0).clamp(-1.0, 1.0)
    delta = (profile_event - previous).norm(dim=-1).masked_fill(~pair_valid, 0.0)
    result = torch.cat((mean, std, profile_event, cosine.unsqueeze(-1), delta.unsqueeze(-1)), dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
    if result.shape[-1] != PROFILE_DIM or not torch.isfinite(result).all():
        raise ValueError("final L2 profile feature width differs")
    return result


__all__ = ["FEATURE_VERSION", "TEMPORAL_SUFFIX_DIM", "PROFILE_DIM", "PROFILE_FEATURE_NAMES", "causal_mean_std", "concat_batches", "build_temporal_features", "raw_profile", "profile_features"]
