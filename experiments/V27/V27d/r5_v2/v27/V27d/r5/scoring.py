"""Bird projection prototype scores and threshold-free diagnostics."""

from __future__ import annotations

from typing import Any

import torch


from v27.V27d.metrics import binary_auroc


















def ranking_metrics(scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> dict[str, Any]:
    mask = positive | negative
    target = positive[mask]
    selected = scores[mask]
    return {
        "support": int(mask.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc_higher_is_known": binary_auroc(selected, target),
        "score_min": float(selected.min().item()) if selected.numel() else None,
        "score_max": float(selected.max().item()) if selected.numel() else None,
        "score_mean_positive": float(scores[positive].mean().item()) if bool(positive.any()) else None,
        "score_mean_negative": float(scores[negative].mean().item()) if bool(negative.any()) else None,
    }


__all__ = ["ranking_metrics"]