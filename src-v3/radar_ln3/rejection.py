from __future__ import annotations

from dataclasses import dataclass

import torch

from . import LABELS


REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-rejection-calibration-v1"
TRAIN_PARENT_GROUPED_OOF_SOURCE = "train_parent_grouped_oof"
MAX_CLASS_PROBABILITY_SCORE = "max_class_probability"


@dataclass(frozen=True)
class RejectionCalibration:
    """A threshold contract fitted without validation/test target leakage."""

    schema_version: str
    source: str
    score_name: str
    acceptance_threshold: float

    def __post_init__(self) -> None:
        if self.schema_version != REJECTION_CALIBRATION_SCHEMA:
            raise ValueError(f"unsupported rejection calibration schema: {self.schema_version!r}")
        if self.source != TRAIN_PARENT_GROUPED_OOF_SOURCE:
            raise ValueError(
                "rejection calibration must come from train parent-grouped OOF predictions"
            )
        if self.score_name != MAX_CLASS_PROBABILITY_SCORE:
            raise ValueError(f"unsupported rejection score: {self.score_name!r}")
        threshold = float(self.acceptance_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("rejection acceptance_threshold must be in (0, 1)")


@dataclass(frozen=True)
class RejectionDecision:
    """Classification and reject action are deliberately orthogonal fields."""

    class_label_id: int
    class_label_name: str
    class_confidence: float
    decision_state: str
    rejected: bool | None
    rejection_score: float | None
    calibration_source: str | None


class RejectionPolicy:
    """Guarded maximum-probability rejection policy.

    This class only defines the interface and safety boundary.  It does not
    estimate a threshold.  Until a train parent-grouped OOF calibration is
    supplied and explicitly enabled, it returns ``rejected=None``.
    """

    def __init__(
        self,
        calibration: RejectionCalibration | None = None,
        *,
        enabled: bool = False,
    ) -> None:
        if enabled and calibration is None:
            raise RuntimeError(
                "rejection cannot be enabled without train parent-grouped OOF calibration"
            )
        self.calibration = calibration
        self.enabled = bool(enabled)

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def decide(self, probabilities: torch.Tensor) -> RejectionDecision:
        values = probabilities.detach().float().reshape(-1)
        if values.shape != (len(LABELS),) or not torch.isfinite(values).all():
            raise ValueError(f"rejection expects {len(LABELS)} finite class probabilities")
        if bool((values < 0.0).any()):
            raise ValueError("rejection probabilities must be non-negative")
        total = float(values.sum().item())
        if total <= 0.0 or abs(total - 1.0) > 1.0e-4:
            raise ValueError("rejection expects normalized class probabilities")
        class_label_id = int(values.argmax().item())
        class_confidence = float(values[class_label_id].item())
        if not self.enabled:
            return RejectionDecision(
                class_label_id=class_label_id,
                class_label_name=LABELS[class_label_id],
                class_confidence=class_confidence,
                decision_state="not_calibrated" if self.calibration is None else "disabled",
                rejected=None,
                rejection_score=None,
                calibration_source=None if self.calibration is None else self.calibration.source,
            )
        if self.calibration is None:  # Defensive guard for future mutable wrappers.
            raise RuntimeError("rejection was enabled without a calibration contract")
        rejected = class_confidence < float(self.calibration.acceptance_threshold)
        return RejectionDecision(
            class_label_id=class_label_id,
            class_label_name=LABELS[class_label_id],
            class_confidence=class_confidence,
            decision_state="rejected" if rejected else "accepted",
            rejected=rejected,
            rejection_score=1.0 - class_confidence,
            calibration_source=self.calibration.source,
        )


__all__ = [
    "MAX_CLASS_PROBABILITY_SCORE",
    "REJECTION_CALIBRATION_SCHEMA",
    "TRAIN_PARENT_GROUPED_OOF_SOURCE",
    "RejectionCalibration",
    "RejectionDecision",
    "RejectionPolicy",
]
