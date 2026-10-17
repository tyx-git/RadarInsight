"""Causal prefix-score aggregation used by V27d-r12."""

from __future__ import annotations

import math

import torch


def aggregate_prefix_quantile(
    score_sequence: torch.Tensor,
    final_scores: torch.Tensor,
    prefix_route_mask: torch.Tensor,
    candidate_population: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """Aggregate eligible causal prefix scores without looking beyond the final observation."""

    if score_sequence.ndim != 2 or final_scores.ndim != 1:
        raise ValueError("V27d-r12 score tensors have invalid rank")
    if score_sequence.shape[0] != final_scores.shape[0]:
        raise ValueError("V27d-r12 score tensor rows differ")
    if (
        prefix_route_mask.shape != score_sequence.shape
        or padding_mask.shape != score_sequence.shape
        or candidate_population.shape != final_scores.shape
    ):
        raise ValueError("V27d-r12 prefix aggregation shapes differ")
    if (
        prefix_route_mask.dtype != torch.bool
        or candidate_population.dtype != torch.bool
        or padding_mask.dtype != torch.bool
    ):
        raise ValueError("V27d-r12 prefix aggregation masks must be boolean")
    if not 0.0 < float(quantile) < 1.0 or not math.isfinite(float(quantile)):
        raise ValueError("V27d-r12 quantile must be in (0,1)")
    if not score_sequence.is_floating_point() or not final_scores.is_floating_point():
        raise TypeError("V27d-r12 scores must be floating point")
    if not torch.isfinite(score_sequence).all() or not torch.isfinite(final_scores).all():
        raise ValueError("V27d-r12 scores must be finite")
    if bool((prefix_route_mask & padding_mask).any()):
        raise ValueError("V27d-r12 padded prefix is eligible")

    eligible = prefix_route_mask & candidate_population.unsqueeze(1)
    result = final_scores.detach().clone()
    for row in range(score_sequence.shape[0]):
        if not bool(candidate_population[row]):
            continue
        values = score_sequence[row][eligible[row]]
        if not values.numel():
            raise ValueError(f"V27d-r12 candidate row {row} has no eligible causal prefix")
        result[row] = torch.quantile(values, float(quantile))
    if not torch.isfinite(result).all():
        raise FloatingPointError("V27d-r12 aggregated scores are non-finite")
    return result


__all__ = ["aggregate_prefix_quantile"]
