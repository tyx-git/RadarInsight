from __future__ import annotations


import torch
from torch import nn








def local_task_effective_class_mass(
    targets: torch.Tensor,
    *,
    class_count: int,
    class_weights: torch.Tensor | None = None,
    class_weight_normalization: str = "batch_weight_sum",
) -> torch.Tensor:
    """Return the actual per-class coefficient mass for one local-task reduction.

    The returned vector is intentionally detached from model logits.  It is a
    compact audit of the scalar loss semantics, not an additional training
    signal.  In particular, ``present_class_mean`` gives every class present
    in the contributing parent set equal total mass, irrespective of its
    count or the legacy inverse-frequency weights.
    """

    if targets.ndim != 1 or targets.numel() <= 0:
        raise ValueError("local task effective class mass requires a non-empty [parents] target tensor")
    if int(class_count) < 2:
        raise ValueError("local task effective class mass requires at least two classes")
    targets = targets.to(dtype=torch.long)
    if bool(((targets < 0) | (targets >= int(class_count))).any()):
        raise ValueError("local task effective class mass targets are outside the configured class range")
    if class_weight_normalization not in {
        "batch_weight_sum",
        "fixed_parent_count",
        "present_class_mean",
    }:
        raise ValueError(
            "local task class_weight_normalization must be batch_weight_sum, "
            "fixed_parent_count or present_class_mean"
        )
    weights = class_weights
    if weights is not None:
        if (
            weights.shape != (int(class_count),)
            or not torch.isfinite(weights).all()
            or bool((weights <= 0.0).any())
        ):
            raise ValueError("local task class_weights must match the class count and be finite positive")
        weights = weights.to(device=targets.device, dtype=torch.float32)
    counts = torch.bincount(targets, minlength=int(class_count)).to(dtype=torch.float32)
    present = counts > 0.0
    if class_weight_normalization == "present_class_mean":
        return present.to(dtype=torch.float32) / present.sum().to(dtype=torch.float32)
    if weights is None:
        raw_mass = counts
    else:
        raw_mass = counts * weights
    if class_weight_normalization == "batch_weight_sum":
        return raw_mass / raw_mass.sum()
    # ``fixed_parent_count`` preserves the declared inverse-frequency mass;
    # its total may differ from one when a minibatch is class-skewed.
    return raw_mass / float(targets.numel())


def _present_class_mean(
    per_parent_loss: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_count: int,
) -> torch.Tensor:
    """Average contributing parents inside each present class, then classes."""

    if per_parent_loss.ndim != 1 or per_parent_loss.shape != targets.shape:
        raise ValueError("present-class mean loss requires aligned one-dimensional parent losses and targets")
    class_means = []
    for class_id in range(int(class_count)):
        class_losses = per_parent_loss[targets == class_id]
        if class_losses.numel():
            class_means.append(class_losses.mean())
    if not class_means:
        raise ValueError("present-class mean loss received no contributing class")
    return torch.stack(class_means).mean()


def parent_final_task_ce_with_prefix_auxiliary(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    prefix_auxiliary_weight: float = 0.0,
    prefix_auxiliary_min_events: int = 3,
    class_weight_normalization: str = "batch_weight_sum",
) -> dict[str, torch.Tensor]:
    """Train one local hierarchy task with an explicit product-aligned target.

    The main objective supervises exactly one final causal state per parent.
    The optional auxiliary objective starts only after the configured evidence
    count, excludes each parent's final state, averages eligible prefixes
    inside a parent, and then averages parents.  Consequently neither padding
    nor parent length changes a parent's contribution to the auxiliary term.

    ``batch_weight_sum`` preserves the historical PyTorch weighted-mean
    semantics: the weighted numerator is divided by the sum of target weights
    present in the current minibatch.  In a single-class minibatch this cancels
    the class weight completely.  ``fixed_parent_count`` instead divides by the
    number of contributing parents.  With dataset inverse-frequency weights
    whose dataset mean is one, this is an unbiased minibatch estimate of the
    declared parent/class-balanced objective and does not renormalize a rare
    class away merely because the other class is absent from that minibatch.
    ``present_class_mean`` first averages per-parent CE within each class
    currently represented in the contributing set, then averages those class
    means.  This makes a mixed two-class batch exactly 50/50 while retaining
    numerical equivalence with the historical objective for a pure-class batch.
    """

    if sequence_logits.ndim != 3:
        raise ValueError("local task sequence_logits must have shape [batch, sequence, classes]")
    if labels.shape != (sequence_logits.shape[0],):
        raise ValueError("local task labels must have shape [batch]")
    if padding_mask.shape != sequence_logits.shape[:2]:
        raise ValueError("local task padding_mask must have shape [batch, sequence]")
    if padding_mask.dtype != torch.bool:
        raise ValueError("local task padding_mask must be boolean")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("local task sequence_logits must be finite")
    class_count = int(sequence_logits.shape[-1])
    if class_count < 2:
        raise ValueError("local task loss requires at least two classes")
    targets = labels.to(device=sequence_logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= class_count)).any()):
        raise ValueError("local task labels are outside the configured class range")
    weights = class_weights
    if weights is not None:
        if (
            weights.shape != (class_count,)
            or not torch.isfinite(weights).all()
            or bool((weights <= 0.0).any())
        ):
            raise ValueError("local task class_weights must match the class count and be finite positive")
        weights = weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    prefix_weight = float(prefix_auxiliary_weight)
    if not 0.0 <= prefix_weight <= 1.0:
        raise ValueError("local task prefix_auxiliary_weight must be in [0,1]")
    minimum_events = int(prefix_auxiliary_min_events)
    if minimum_events < 2:
        raise ValueError("local task prefix auxiliary supervision must exclude the first evidence event")
    if class_weight_normalization not in {
        "batch_weight_sum",
        "fixed_parent_count",
        "present_class_mean",
    }:
        raise ValueError(
            "local task class_weight_normalization must be batch_weight_sum, "
            "fixed_parent_count or present_class_mean"
        )

    lengths = (~padding_mask).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("local task loss received an empty parent sequence")
    batch_indices = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
    final_indices = lengths - 1
    final_logits = sequence_logits[batch_indices, final_indices]
    final_effective_class_mass = local_task_effective_class_mass(
        targets,
        class_count=class_count,
        class_weights=weights,
        class_weight_normalization=class_weight_normalization,
    ).to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    if class_weight_normalization == "present_class_mean":
        final_per_parent = nn.functional.cross_entropy(
            final_logits,
            targets,
            reduction="none",
        )
        final_loss = _present_class_mean(
            final_per_parent,
            targets,
            class_count=class_count,
        )
    elif weights is None or class_weight_normalization == "batch_weight_sum":
        final_loss = nn.functional.cross_entropy(final_logits, targets, weight=weights)
    else:
        final_per_parent = nn.functional.cross_entropy(
            final_logits,
            targets,
            reduction="none",
        )
        final_loss = (final_per_parent * weights[targets]).sum() / targets.numel()

    prefix_loss = final_loss.new_zeros(())
    eligible_prefix_count = final_indices.new_zeros(())
    eligible_parent_count = final_indices.new_zeros(())
    prefix_effective_class_mass = final_effective_class_mass.new_zeros(class_count)
    if prefix_weight > 0.0:
        event_numbers = torch.arange(
            1,
            sequence_logits.shape[1] + 1,
            device=sequence_logits.device,
        ).unsqueeze(0)
        eligible = (
            (~padding_mask)
            & (event_numbers >= minimum_events)
            & (event_numbers < lengths.unsqueeze(1))
        )
        eligible_prefix_count = eligible.sum()
        prefix_counts = eligible.sum(dim=1)
        eligible_parents = prefix_counts > 0
        eligible_parent_count = eligible_parents.sum()
        if bool(eligible_parents.any()):
            expanded_targets = targets.unsqueeze(1).expand(-1, sequence_logits.shape[1])
            per_event = nn.functional.cross_entropy(
                sequence_logits.reshape(-1, class_count),
                expanded_targets.reshape(-1),
                weight=(
                    weights
                    if class_weight_normalization == "batch_weight_sum"
                    else None
                ),
                reduction="none",
            ).reshape(sequence_logits.shape[:2])
            parent_means = (
                (per_event * eligible.to(dtype=per_event.dtype)).sum(dim=1)
                / prefix_counts.clamp_min(1)
            )
            selected_parent_means = parent_means[eligible_parents]
            selected_targets = targets[eligible_parents]
            prefix_effective_class_mass = local_task_effective_class_mass(
                selected_targets,
                class_count=class_count,
                class_weights=weights,
                class_weight_normalization=class_weight_normalization,
            ).to(device=sequence_logits.device, dtype=sequence_logits.dtype)
            if class_weight_normalization == "present_class_mean":
                # ``per_event`` is deliberately unweighted in this branch:
                # class balance is expressed by the outer present-class mean,
                # after each parent's eligible causal prefixes are averaged.
                prefix_loss = _present_class_mean(
                    selected_parent_means,
                    selected_targets,
                    class_count=class_count,
                )
            elif weights is None:
                prefix_loss = selected_parent_means.mean()
            elif class_weight_normalization == "batch_weight_sum":
                selected_parent_weights = weights[selected_targets]
                prefix_loss = selected_parent_means.sum() / selected_parent_weights.sum()
            else:
                selected_parent_weights = weights[selected_targets]
                prefix_loss = (
                    selected_parent_means * selected_parent_weights
                ).sum() / eligible_parent_count

    total_loss = final_loss + prefix_weight * prefix_loss
    if not all(torch.isfinite(value).item() for value in (total_loss, final_loss, prefix_loss)):
        raise ValueError("local task loss produced a non-finite value")
    return {
        "loss": total_loss,
        "final_state_loss": final_loss,
        "prefix_auxiliary_loss": prefix_loss,
        "eligible_prefix_count": eligible_prefix_count,
        "eligible_prefix_parent_count": eligible_parent_count,
        "final_effective_class_mass": final_effective_class_mass,
        "prefix_effective_class_mass": prefix_effective_class_mass,
    }






__all__ = ["local_task_effective_class_mass", "parent_final_task_ce_with_prefix_auxiliary"]