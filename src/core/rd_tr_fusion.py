from __future__ import annotations

from dataclasses import dataclass

import torch

from .data.rd_physical_five_dataset import RD_FIVE_LABELS


@dataclass(frozen=True)
class RDFusionDecision:
    """Result of routing one batch of five-class modality scores.

    ``logits`` is returned only when every row has a real modality score.  A
    mixed batch (or an all-missing batch) gets ``None`` so an accidental
    ``argmax`` cannot turn a zero sentinel into a prediction.  Per-row
    probabilities remain available for rows whose ``logits_available`` flag is
    true; abstained rows are all-zero and must be excluded from metrics.
    """

    logits: torch.Tensor | None
    probabilities: torch.Tensor | None
    logits_available: torch.Tensor
    selected_source: tuple[str, ...]
    availability_state: tuple[str, ...]
    abstain: torch.Tensor
    reason: tuple[str, ...]

    def require_all_logits(self) -> torch.Tensor:
        """Return logits only when every row has a real modality score."""

        if self.logits is None or not bool(self.logits_available.all()):
            raise ValueError("fusion logits are unavailable for at least one row; handle abstain first")
        return self.logits


def _validate_scores(name: str, values: torch.Tensor, batch_size: int) -> None:
    if values.ndim != 2 or values.shape != (batch_size, len(RD_FIVE_LABELS)):
        raise ValueError(f"{name} must have shape [batch, 5] in LOW_ALTITUDE_CLASSES order")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def route_rd_tr_five(
    rd_logits: torch.Tensor | None,
    tr_logits: torch.Tensor | None,
    *,
    rd_available: torch.Tensor,
    tr_available: torch.Tensor,
    rd_weight: float = 0.5,
) -> RDFusionDecision:
    """Availability-aware late fusion.

    The availability masks are operational state only; no ID, label, pairing
    result, or path can alter routing. Missing modalities are never replaced by
    zero tensors. Single-modality rows strictly pass through that modality.
    """
    if rd_available.ndim != 1 or tr_available.ndim != 1 or rd_available.shape != tr_available.shape:
        raise ValueError("availability masks must have the same [batch] shape")
    batch_size = int(rd_available.shape[0])
    rd_mask = rd_available.bool()
    tr_mask = tr_available.bool()
    if rd_logits is not None:
        _validate_scores("rd_logits", rd_logits, batch_size)
    if tr_logits is not None:
        _validate_scores("tr_logits", tr_logits, batch_size)
    if bool((rd_mask & (rd_logits is None)).any()) or bool((tr_mask & (tr_logits is None)).any()):
        raise ValueError("available modality is missing its score tensor")
    if not 0.0 <= float(rd_weight) <= 1.0:
        raise ValueError("rd_weight must be in [0, 1]")
    reference = rd_logits if rd_logits is not None else tr_logits
    if reference is None:
        reference = torch.empty((batch_size, len(RD_FIVE_LABELS)), dtype=torch.float32)
    output = reference.new_empty((batch_size, len(RD_FIVE_LABELS)))
    abstain = ~(rd_mask | tr_mask)
    sources: list[str] = []
    states: list[str] = []
    reasons: list[str] = []
    for index in range(batch_size):
        has_rd = bool(rd_mask[index])
        has_tr = bool(tr_mask[index])
        if has_rd and has_tr:
            assert rd_logits is not None and tr_logits is not None
            output[index] = float(rd_weight) * rd_logits[index] + (1.0 - float(rd_weight)) * tr_logits[index]
            sources.append("rd+tr")
            states.append("both")
            reasons.append("both_available")
        elif has_rd:
            assert rd_logits is not None
            output[index] = rd_logits[index]
            sources.append("rd")
            states.append("rd_only")
            reasons.append("tr_missing")
        elif has_tr:
            assert tr_logits is not None
            output[index] = tr_logits[index]
            sources.append("tr")
            states.append("tr_only")
            reasons.append("rd_missing")
        else:
            output[index] = 0.0
            sources.append("none")
            states.append("none")
            reasons.append("both_missing")
    # The zero values assigned to both-missing rows are an internal temporary
    # used only to keep tensor shapes uniform while routing.  They are never
    # exposed as logits for a mixed or all-missing batch.
    probabilities = torch.softmax(output, dim=-1)
    probabilities = torch.where(abstain[:, None], torch.zeros_like(probabilities), probabilities)
    return RDFusionDecision(
        logits=output if not bool(abstain.any()) else None,
        probabilities=probabilities,
        logits_available=~abstain,
        selected_source=tuple(sources),
        availability_state=tuple(states),
        abstain=abstain,
        reason=tuple(reasons),
    )


__all__ = ["RDFusionDecision", "route_rd_tr_five"]
