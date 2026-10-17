from __future__ import annotations


import torch
from torch import nn






def parent_final_fixed_parent_count_ce(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """V27's only training objective.

    Exactly one final causal state is selected for every parent.  The
    train-only class weight is applied to that parent's CE and the sum is
    divided by the fixed number of parents in the batch.  In particular, the
    denominator never changes with the classes present in a minibatch.  No
    prefix, Unknown, quality, metric, or open-set term is included here.
    """

    if sequence_logits.ndim != 3:
        raise ValueError("V27 sequence_logits must have shape [B,T,C]")
    if labels.shape != (sequence_logits.shape[0],):
        raise ValueError("V27 labels must have shape [B]")
    if padding_mask.shape != sequence_logits.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("V27 padding_mask must be boolean with shape [B,T]")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("V27 sequence_logits must be finite")
    class_count = int(sequence_logits.shape[-1])
    if class_weights.shape != (class_count,) or not torch.isfinite(class_weights).all():
        raise ValueError("V27 class_weights have the wrong shape or are non-finite")
    if bool((class_weights <= 0.0).any()):
        raise ValueError("V27 class_weights must be positive")
    targets = labels.to(device=sequence_logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= class_count)).any()):
        raise ValueError("V27 labels are outside the configured class range")
    lengths = (~padding_mask).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("V27 loss received an empty parent sequence")
    batch_indices = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
    final_logits = sequence_logits[batch_indices, lengths - 1]
    per_parent = nn.functional.cross_entropy(final_logits, targets, reduction="none")
    weights = class_weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    fixed_parent_count = torch.tensor(
        float(targets.numel()), dtype=sequence_logits.dtype, device=sequence_logits.device
    )
    value = (per_parent * weights[targets]).sum() / fixed_parent_count
    effective_mass = torch.zeros(
        (class_count,), dtype=sequence_logits.dtype, device=sequence_logits.device
    )
    effective_mass.scatter_add_(0, targets, weights[targets])
    if not torch.isfinite(value).item():
        raise ValueError("V27 fixed-parent-count loss is non-finite")
    return {
        "loss": value,
        "final_state_loss": value,
        "prefix_auxiliary_loss": value.new_zeros(()),
        "fixed_parent_count": fixed_parent_count,
        "effective_class_mass": effective_mass,
        "final_logits": final_logits,
    }












__all__ = ["parent_final_fixed_parent_count_ce"]