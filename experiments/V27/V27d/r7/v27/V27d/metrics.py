"""Deterministic binary-ranking metrics used by V27d diagnostics."""

from __future__ import annotations

import torch


def _validated_inputs(scores: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or target.ndim != 1 or scores.shape != target.shape:
        raise ValueError("V27d binary metric inputs must be aligned one-dimensional tensors")
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("V27d binary metric scores must be finite floating-point values")
    return scores.detach().cpu().to(dtype=torch.float64), target.detach().cpu().bool()


def _descending_average_ranks(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = scores[order]
    if not len(ordered):
        return order, ordered
    starts = torch.cat(
        (torch.ones(1, dtype=torch.bool), ordered[1:] != ordered[:-1])
    )
    group_ids = starts.to(dtype=torch.long).cumsum(0) - 1
    group_sizes = torch.bincount(group_ids)
    ends = group_sizes.cumsum(0).to(dtype=torch.float64)
    begins = ends - group_sizes.to(dtype=torch.float64) + 1.0
    average_ranks = (begins + ends) / 2.0
    return order, average_ranks[group_ids]


def binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    """Return descending-score AUROC with average ranks for ties."""

    scores, target = _validated_inputs(scores, target)
    positive = int(target.sum().item())
    negative = int((~target).sum().item())
    if positive == 0 or negative == 0:
        return None
    order, ranks = _descending_average_ranks(scores)
    rank_sum = ranks[target[order]].sum().item()
    return float(
        1.0
        - (rank_sum - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


__all__ = ["binary_auroc"]
