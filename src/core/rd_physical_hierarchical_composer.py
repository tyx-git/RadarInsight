from __future__ import annotations

from dataclasses import dataclass

import torch

from .data.rd_physical_five_dataset import RD_FIVE_LABELS


@dataclass(frozen=True)
class HierarchicalComposition:
    probabilities: torch.Tensor | None
    logits: torch.Tensor | None
    available: torch.Tensor
    mode: str
    reason: tuple[str, ...]


def _probs(values: torch.Tensor | None, *, batch: int, classes: int, name: str) -> torch.Tensor | None:
    if values is None:
        return None
    if values.ndim != 2 or tuple(values.shape) != (batch, classes) or not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite logits with shape [{batch},{classes}]")
    return torch.softmax(values, dim=-1)


def compose_rd_specialists(
    parent_logits: torch.Tensor | None,
    dbb_logits: torch.Tensor | None,
    cu_logits: torch.Tensor | None,
    *,
    flat_logits: torch.Tensor | None = None,
    mode: str = "predicted_soft",
) -> HierarchicalComposition:
    """Compose independent ParentGate/DBB/CU logits without hard routing.

    ``flat_logits`` is an explicitly supplied, already validated RD-only
    fallback.  Missing specialist inputs never become zero-logit predictions.
    """
    if mode not in {"predicted_soft", "oracle_parent", "hard_routing"}:
        raise ValueError("mode must be predicted_soft, oracle_parent, or hard_routing")
    tensors = [value for value in (parent_logits, dbb_logits, cu_logits, flat_logits) if value is not None]
    if not tensors:
        return HierarchicalComposition(None, None, torch.zeros(0, dtype=torch.bool), mode, ("all_specialists_missing",))
    batch = int(tensors[0].shape[0])
    for value in tensors:
        if value.ndim != 2 or value.shape[0] != batch:
            raise ValueError("specialist logits must share batch dimension")
    parent = _probs(parent_logits, batch=batch, classes=2, name="parent_logits")
    dbb = _probs(dbb_logits, batch=batch, classes=3, name="dbb_logits")
    cu = _probs(cu_logits, batch=batch, classes=2, name="cu_logits")
    flat = _probs(flat_logits, batch=batch, classes=5, name="flat_logits")
    if mode == "predicted_soft":
        if parent is None or dbb is None or cu is None:
            if flat is None:
                return HierarchicalComposition(None, None, torch.zeros(batch, dtype=torch.bool), mode, ("specialist_missing_and_no_flat_fallback",) * batch)
            return HierarchicalComposition(flat, torch.log(flat.clamp_min(1e-12)), torch.ones(batch, dtype=torch.bool, device=flat.device), "flat_fallback", ("specialist_missing",) * batch)
        probabilities = torch.cat([parent[:, 0:1] * dbb, parent[:, 1:2] * cu], dim=1)
    elif mode == "oracle_parent":
        if dbb is None or cu is None or parent is None:
            raise ValueError("oracle_parent requires all specialist logits; oracle parent labels are supplied by caller only in evaluation")
        probabilities = torch.cat([parent[:, 0:1] * dbb, parent[:, 1:2] * cu], dim=1)
    else:
        if parent is None or dbb is None or cu is None:
            raise ValueError("hard_routing requires all specialist logits")
        selected = parent.argmax(dim=1)
        probabilities = torch.zeros((batch, 5), dtype=parent.dtype, device=parent.device)
        dbb_rows = selected.eq(0); cu_rows = ~dbb_rows
        probabilities[dbb_rows, :3] = dbb[dbb_rows]
        probabilities[cu_rows, 3:] = cu[cu_rows]
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("hierarchical probabilities are invalid")
    if not torch.allclose(probabilities.sum(1), torch.ones(batch, device=probabilities.device), atol=1e-6):
        raise ValueError("hierarchical probabilities do not sum to one")
    return HierarchicalComposition(probabilities, torch.log(probabilities.clamp_min(1e-12)), torch.ones(batch, dtype=torch.bool, device=probabilities.device), mode, ("all_specialists_available",) * batch)


def compose_with_oracle_parent(parent_labels: torch.Tensor, dbb_logits: torch.Tensor, cu_logits: torch.Tensor) -> HierarchicalComposition:
    if parent_labels.ndim != 1 or dbb_logits.ndim != 2 or cu_logits.ndim != 2 or dbb_logits.shape[0] != parent_labels.shape[0] or cu_logits.shape[0] != parent_labels.shape[0]:
        raise ValueError("oracle parent composition shape contract failed")
    dbb = torch.softmax(dbb_logits, -1); cu = torch.softmax(cu_logits, -1); batch = parent_labels.shape[0]
    out = torch.zeros((batch, len(RD_FIVE_LABELS)), dtype=dbb.dtype, device=dbb.device)
    out[parent_labels.eq(0), :3] = dbb[parent_labels.eq(0)]
    out[parent_labels.eq(1), 3:] = cu[parent_labels.eq(1)]
    return HierarchicalComposition(out, torch.log(out.clamp_min(1e-12)), torch.ones(batch, dtype=torch.bool, device=out.device), "oracle_parent", ("true_parent_for_diagnostic_only",) * batch)


__all__ = ["HierarchicalComposition", "compose_rd_specialists", "compose_with_oracle_parent"]
