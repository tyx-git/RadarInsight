from __future__ import annotations

import torch
from torch import nn


def five_class_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("five-class logits must have shape [batch, 5]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if not torch.isfinite(logits).all():
        raise ValueError("five-class logits must be finite")
    if labels.dtype not in (torch.int64, torch.long):
        labels = labels.long()
    if bool((labels < 0).any()) or bool((labels >= 5).any()):
        raise ValueError("five-class labels are outside [0, 4]")
    if class_weights is not None:
        if class_weights.shape != (5,) or not torch.isfinite(class_weights).all() or bool((class_weights <= 0.0).any()):
            raise ValueError("class_weights must be finite positive values with shape [5]")
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    value = nn.functional.cross_entropy(logits, labels, weight=class_weights)
    if not torch.isfinite(value):
        raise ValueError("five-class loss is non-finite")
    return value


def causal_prefix_loss(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise every causal prefix while giving each parent equal weight.

    ``sequence_logits[:, t]`` is the prediction available after event ``t``.
    Padding is excluded, and the per-prefix loss is averaged inside each
    parent before the parent losses are averaged across the batch.  This
    avoids making long tracks dominate the window objective.
    """

    if sequence_logits.ndim != 3 or sequence_logits.shape[-1] != 5:
        raise ValueError("sequence_logits must have shape [batch, sequence, 5]")
    if labels.ndim != 1 or labels.shape[0] != sequence_logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if padding_mask.shape != sequence_logits.shape[:2]:
        raise ValueError("padding_mask must have shape [batch, sequence]")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("sequence_logits must be finite")
    if class_weights is not None:
        if class_weights.shape != (5,) or not torch.isfinite(class_weights).all() or bool((class_weights <= 0.0).any()):
            raise ValueError("class_weights must be finite positive values with shape [5]")
        class_weights = class_weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    flat_labels = labels.to(device=sequence_logits.device, dtype=torch.long).unsqueeze(1).expand(-1, sequence_logits.shape[1])
    per_prefix = nn.functional.cross_entropy(
        sequence_logits.reshape(-1, 5),
        flat_labels.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape(sequence_logits.shape[:2])
    valid = (~padding_mask).to(dtype=per_prefix.dtype)
    counts = valid.sum(dim=1)
    if bool((counts <= 0.0).any()):
        raise ValueError("causal prefix loss received an empty parent sequence")
    parent_means = (per_prefix * valid).sum(dim=1) / counts
    value = parent_means.mean()
    if not torch.isfinite(value).item():
        raise ValueError("causal prefix loss is non-finite")
    return value


__all__ = ["causal_prefix_loss", "five_class_loss"]
