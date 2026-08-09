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


def final_valid_state_loss(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise exactly one final causal state per parent.

    This is the product-aligned parent-final objective.  Padding never enters
    the loss and an empty parent is rejected instead of silently selecting a
    padded state.
    """

    if sequence_logits.ndim != 3:
        raise ValueError("sequence_logits must have shape [batch, sequence, classes]")
    if labels.shape != (sequence_logits.shape[0],):
        raise ValueError("labels must have shape [batch]")
    if padding_mask.shape != sequence_logits.shape[:2]:
        raise ValueError("padding_mask must have shape [batch, sequence]")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("sequence_logits must be finite")
    class_count = int(sequence_logits.shape[-1])
    if class_count < 2:
        raise ValueError("final state loss requires at least two classes")
    if class_weights is not None:
        if (
            class_weights.shape != (class_count,)
            or not torch.isfinite(class_weights).all()
            or bool((class_weights <= 0.0).any())
        ):
            raise ValueError("class_weights must be finite positive values matching the class count")
        class_weights = class_weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    lengths = (~padding_mask).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("final state loss received an empty parent sequence")
    batch_indices = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
    final_logits = sequence_logits[batch_indices, lengths - 1]
    value = nn.functional.cross_entropy(
        final_logits,
        labels.to(device=sequence_logits.device, dtype=torch.long),
        weight=class_weights,
    )
    if not torch.isfinite(value).item():
        raise ValueError("final state loss is non-finite")
    return value


def hierarchical_final_state_nll(
    gate_sequence_logits: torch.Tensor,
    dbb_sequence_logits: torch.Tensor,
    cu_sequence_logits: torch.Tensor,
    base_labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    prefix_auxiliary_weight: float = 0.0,
    prefix_auxiliary_min_events: int = 1,
    gate_class_weights: torch.Tensor | None = None,
    dbb_class_weights: torch.Tensor | None = None,
    cu_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Product-level three-layer objective at the final valid state.

    Labels use the frozen five-class order Drone/Bird/Balloon/Clutter/Unknown.
    The final-state NLL is factorized by the actual route:

    * DBB: ``P(DBB) * P(subclass | DBB)``;
    * Clutter: ``P(CU) * P(Clutter | CU)``;
    * Unknown: ``P(CU) * P(reject | CU)``.

    Unknown supervision therefore trains the CU rejection branch, but the
    runtime action remains distinct from a generic closed-set class label.
    Prefix supervision is optional, explicit, and never the main objective.
    """

    shapes = (
        gate_sequence_logits.shape[:-1],
        dbb_sequence_logits.shape[:-1],
        cu_sequence_logits.shape[:-1],
    )
    if len(set(shapes)) != 1 or padding_mask.shape != shapes[0]:
        raise ValueError("hierarchical logits and padding mask shapes do not match")
    if (
        gate_sequence_logits.shape[-1] != 2
        or dbb_sequence_logits.shape[-1] != 3
        or cu_sequence_logits.shape[-1] != 2
    ):
        raise ValueError("hierarchical loss expects gate=2, DBB=3 and CU=2 logits")
    if base_labels.shape != (gate_sequence_logits.shape[0],):
        raise ValueError("base_labels must have shape [batch]")
    labels = base_labels.to(device=gate_sequence_logits.device, dtype=torch.long)
    if bool(((labels < 0) | (labels >= 5)).any()):
        raise ValueError("hierarchical base labels must be in [0, 4]")

    def normalized_weights(
        value: torch.Tensor | None,
        class_count: int,
        name: str,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if value.shape != (class_count,) or not torch.isfinite(value).all() or bool((value <= 0.0).any()):
            raise ValueError(f"{name} must contain {class_count} finite positive values")
        return value.to(device=reference.device, dtype=reference.dtype)

    gate_weights = normalized_weights(gate_class_weights, 2, "gate_class_weights", gate_sequence_logits)
    dbb_weights = normalized_weights(dbb_class_weights, 3, "dbb_class_weights", dbb_sequence_logits)
    cu_weights = normalized_weights(cu_class_weights, 2, "cu_class_weights", cu_sequence_logits)
    prefix_weight = float(prefix_auxiliary_weight)
    if not 0.0 <= prefix_weight <= 1.0:
        raise ValueError("prefix_auxiliary_weight must be in [0, 1]")
    minimum_prefix_events = int(prefix_auxiliary_min_events)
    if minimum_prefix_events < 1:
        raise ValueError("prefix_auxiliary_min_events must be positive")
    lengths = (~padding_mask).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("hierarchical loss received an empty parent sequence")
    batch_indices = torch.arange(labels.shape[0], device=labels.device)
    final_indices = lengths - 1
    gate_final = gate_sequence_logits[batch_indices, final_indices]
    dbb_final = dbb_sequence_logits[batch_indices, final_indices]
    cu_final = cu_sequence_logits[batch_indices, final_indices]
    gate_targets = (labels >= 3).long()
    gate_nll = nn.functional.cross_entropy(
        gate_final, gate_targets, weight=gate_weights, reduction="none"
    )
    specialist_nll = torch.zeros_like(gate_nll)
    dbb_mask = labels < 3
    cu_mask = ~dbb_mask
    if bool(dbb_mask.any()):
        specialist_nll[dbb_mask] = nn.functional.cross_entropy(
            dbb_final[dbb_mask], labels[dbb_mask], weight=dbb_weights, reduction="none"
        )
    if bool(cu_mask.any()):
        specialist_nll[cu_mask] = nn.functional.cross_entropy(
            cu_final[cu_mask], labels[cu_mask] - 3, weight=cu_weights, reduction="none"
        )
    route_balance = (
        torch.ones_like(specialist_nll)
        if gate_weights is None
        else gate_weights[gate_targets]
    )
    specialist_nll = specialist_nll * route_balance
    final_nll = (gate_nll + specialist_nll).mean()

    prefix_nll = final_nll.new_zeros(())
    if prefix_weight > 0.0:
        event_numbers = torch.arange(
            1,
            gate_sequence_logits.shape[1] + 1,
            device=gate_sequence_logits.device,
        ).unsqueeze(0)
        valid = (
            (~padding_mask)
            & (event_numbers >= minimum_prefix_events)
            & (event_numbers < lengths.unsqueeze(1))
        )
        expanded = labels.unsqueeze(1).expand_as(valid)
        gate_targets_all = (expanded >= 3).long()
        gate_all = nn.functional.cross_entropy(
            gate_sequence_logits.reshape(-1, 2),
            gate_targets_all.reshape(-1),
            weight=gate_weights,
            reduction="none",
        ).reshape_as(valid)
        specialist_all = torch.zeros_like(gate_all)
        dbb_all = expanded < 3
        cu_all = ~dbb_all
        if bool((valid & dbb_all).any()):
            specialist_all[dbb_all] = nn.functional.cross_entropy(
                dbb_sequence_logits[dbb_all], expanded[dbb_all], weight=dbb_weights, reduction="none"
            )
        if bool((valid & cu_all).any()):
            specialist_all[cu_all] = nn.functional.cross_entropy(
                cu_sequence_logits[cu_all], expanded[cu_all] - 3, weight=cu_weights, reduction="none"
            )
        if gate_weights is not None:
            specialist_all = specialist_all * gate_weights[gate_targets_all]
        valid_float = valid.to(dtype=gate_all.dtype)
        prefix_counts = valid.sum(dim=1)
        eligible_parents = prefix_counts > 0
        if bool(eligible_parents.any()):
            parent_prefix_nll = ((gate_all + specialist_all) * valid_float).sum(dim=1) / prefix_counts.clamp_min(1)
            prefix_nll = parent_prefix_nll[eligible_parents].mean()
    total = final_nll + prefix_weight * prefix_nll
    if not all(torch.isfinite(value).item() for value in (total, final_nll, gate_nll.mean(), specialist_nll.mean(), prefix_nll)):
        raise ValueError("hierarchical loss produced a non-finite value")
    return {
        "loss": total,
        "final_hierarchical_nll": final_nll,
        "final_gate_nll": gate_nll.mean(),
        "final_specialist_nll": specialist_nll.mean(),
        "prefix_auxiliary_nll": prefix_nll,
        "dbb_parent_count": dbb_mask.sum(),
        "cu_parent_count": cu_mask.sum(),
    }


__all__ = [
    "causal_prefix_loss",
    "final_valid_state_loss",
    "five_class_loss",
    "hierarchical_final_state_nll",
]
