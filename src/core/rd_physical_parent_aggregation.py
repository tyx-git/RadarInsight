"""Leakage-safe parent aggregation shared by RD-only reports and selection.

Parent IDs are used only to group already-produced RD probabilities and to
verify a parent has one label.  The identifier itself is never transformed
into a feature, loss weight, sampler value, or prediction input.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def mean_parent_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
    parent_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert logits to probabilities, then average probabilities per parent.

    Averaging logits and averaging probabilities are not equivalent.  This
    helper fixes the report/selection contract to the latter so flat,
    specialist, and hierarchy reports are comparable.
    """

    if logits.ndim != 2 or labels.shape != (logits.shape[0],) or len(parent_ids) != logits.shape[0]:
        raise ValueError("parent aggregation shape contract failed")
    if logits.shape[0] == 0:
        return logits, labels.to(dtype=torch.long), 0
    probabilities = torch.softmax(logits, dim=-1)
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for probability, label, parent_id in zip(probabilities, labels.tolist(), parent_ids):
        key = str(parent_id)
        grouped[key].append(probability)
        prior = grouped_labels.setdefault(key, int(label))
        if prior != int(label):
            raise ValueError(f"parent has inconsistent labels: {key}")
    parent_probabilities = torch.stack([torch.stack(values).mean(0) for values in grouped.values()])
    parent_labels = torch.tensor(list(grouped_labels.values()), dtype=torch.long, device=parent_probabilities.device)
    return parent_probabilities, parent_labels, len(grouped)


__all__ = ["mean_parent_probabilities"]
