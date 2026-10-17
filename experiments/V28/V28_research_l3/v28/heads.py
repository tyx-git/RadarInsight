"""Independent V28 knownness and quality/maturity heads.

The knownness branch consumes only the raw causal Liquid trajectory and
explicit event/task facts.  Classifier logits, softmax, labels and class
identity are deliberately absent from its input signature.
"""

from __future__ import annotations


import torch

from .protocol import V28Context, V28_L1_TASK










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


__all__ = ["deterministic_quality_target"]