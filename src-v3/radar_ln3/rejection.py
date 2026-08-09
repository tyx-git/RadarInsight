from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import LABELS


REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-rejection-calibration-v1"
TRAIN_PARENT_GROUPED_OOF_SOURCE = "train_parent_grouped_oof"
MAX_CLASS_PROBABILITY_SCORE = "max_class_probability"
HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-V13-hierarchical-rejection-calibration-v1"
HIERARCHICAL_TASK_NAMES = ("dbb_cu_gate", "dbb_specialist", "cu_specialist")
GATE_DBB_PROBABILITY_SCORE = "gate_dbb_probability_and_margin"
CU_CLUTTER_PROBABILITY_SCORE = "cu_clutter_probability_and_margin"


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


@dataclass(frozen=True)
class HierarchicalRejectionCalibration:
    """Two-stage acceptance boundaries fitted only from train-parent OOF."""

    schema_version: str
    source: str
    gate_score_name: str
    cu_score_name: str
    gate_accept_dbb_threshold: float
    gate_accept_dbb_margin: float
    cu_accept_clutter_threshold: float
    cu_accept_clutter_margin: float
    min_valid_points: int = 1
    min_track_quality: float = 0.0
    max_gap_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA:
            raise ValueError(f"unsupported hierarchical calibration schema: {self.schema_version!r}")
        if self.source != TRAIN_PARENT_GROUPED_OOF_SOURCE:
            raise ValueError("hierarchical rejection calibration must come from train parent-grouped OOF")
        if self.gate_score_name != GATE_DBB_PROBABILITY_SCORE:
            raise ValueError(f"unsupported Gate rejection score: {self.gate_score_name!r}")
        if self.cu_score_name != CU_CLUTTER_PROBABILITY_SCORE:
            raise ValueError(f"unsupported CU rejection score: {self.cu_score_name!r}")
        for name, value in (
            ("gate_accept_dbb_threshold", self.gate_accept_dbb_threshold),
            ("cu_accept_clutter_threshold", self.cu_accept_clutter_threshold),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be in (0,1)")
        for name, value in (
            ("gate_accept_dbb_margin", self.gate_accept_dbb_margin),
            ("cu_accept_clutter_margin", self.cu_accept_clutter_margin),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if int(self.min_valid_points) < 1:
            raise ValueError("min_valid_points must be positive")
        if not 0.0 <= float(self.min_track_quality) <= 1.0:
            raise ValueError("min_track_quality must be in [0,1]")
        if not 0.0 <= float(self.max_gap_fraction) <= 1.0:
            raise ValueError("max_gap_fraction must be in [0,1]")


@dataclass(frozen=True)
class HierarchicalActionDecision:
    """One event's product action with class, route and reject kept distinct."""

    decision_state: str
    calibrated: bool
    gate_action: str
    conditional_action: str
    final_action: str
    diagnostic_gate_route: str
    diagnostic_label_id: int
    diagnostic_label_name: str
    accepted_label_id: int | None
    accepted_label_name: str | None
    rejected_to_unknown: bool | None
    gate_dbb_probability: float
    gate_margin: float
    specialist_label_id: int
    specialist_confidence: float
    specialist_margin: float
    valid_point_count: int
    track_quality: float
    gap_fraction: float
    trigger_reasons: tuple[str, ...]
    calibration_source: str | None


def _normalized_probabilities(values: torch.Tensor, class_count: int, name: str) -> torch.Tensor:
    probabilities = values.detach().float().reshape(-1)
    if probabilities.shape != (class_count,) or not torch.isfinite(probabilities).all():
        raise ValueError(f"{name} expects {class_count} finite probabilities")
    if bool((probabilities < 0.0).any()):
        raise ValueError(f"{name} probabilities must be non-negative")
    total = float(probabilities.sum().item())
    if total <= 0.0 or abs(total - 1.0) > 1.0e-4:
        raise ValueError(f"{name} expects normalized probabilities")
    return probabilities


class HierarchicalRejectionPolicy:
    """DBB acceptance followed by Clutter acceptance/Unknown rejection.

    Disabled policies expose only a diagnostic hard route and never return an
    accepted or rejected product action. This prevents the ordinary CU-head
    Unknown label from being reported as validated open-set rejection.
    """

    def __init__(
        self,
        calibration: HierarchicalRejectionCalibration | None = None,
        *,
        enabled: bool = False,
    ) -> None:
        if enabled and calibration is None:
            raise RuntimeError("hierarchical rejection cannot be enabled without train-parent OOF calibration")
        self.calibration = calibration
        self.enabled = bool(enabled)

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def decide(
        self,
        gate_probabilities: torch.Tensor,
        dbb_probabilities: torch.Tensor,
        cu_probabilities: torch.Tensor,
        *,
        valid_point_count: int,
        track_quality: float,
        gap_fraction: float,
    ) -> HierarchicalActionDecision:
        gate = _normalized_probabilities(gate_probabilities, 2, "hierarchical Gate")
        dbb = _normalized_probabilities(dbb_probabilities, 3, "hierarchical DBB specialist")
        cu = _normalized_probabilities(cu_probabilities, 2, "hierarchical CU specialist")
        points = int(valid_point_count)
        quality = float(track_quality)
        gaps = float(gap_fraction)
        if points < 1 or not 0.0 <= quality <= 1.0 or not 0.0 <= gaps <= 1.0:
            raise ValueError("hierarchical quality inputs are outside their contract")
        gate_dbb = float(gate[0].item())
        gate_margin = float((gate[0] - gate[1]).item())
        diagnostic_route = "DBB" if int(gate.argmax().item()) == 0 else "CU"
        if diagnostic_route == "DBB":
            specialist_id = int(dbb.argmax().item())
            diagnostic_label_id = specialist_id
            specialist = dbb
        else:
            specialist_id = int(cu.argmax().item())
            diagnostic_label_id = 3 + specialist_id
            specialist = cu
        specialist_confidence = float(specialist[specialist_id].item())
        if specialist.shape[0] == 2:
            other = 1 - specialist_id
            specialist_margin = float((specialist[specialist_id] - specialist[other]).item())
        else:
            top = torch.topk(specialist, k=2).values
            specialist_margin = float((top[0] - top[1]).item())

        if not self.enabled:
            return HierarchicalActionDecision(
                decision_state="not_calibrated" if self.calibration is None else "disabled",
                calibrated=self.calibrated,
                gate_action="NOT_CALIBRATED",
                conditional_action="NOT_CALIBRATED",
                final_action="NOT_CALIBRATED",
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=None,
                accepted_label_name=None,
                rejected_to_unknown=None,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=specialist_id,
                specialist_confidence=specialist_confidence,
                specialist_margin=specialist_margin,
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=("missing_train_parent_grouped_oof_calibration",),
                calibration_source=None if self.calibration is None else self.calibration.source,
            )
        calibration = self.calibration
        if calibration is None:  # Defensive guard for future mutable wrappers.
            raise RuntimeError("hierarchical rejection was enabled without calibration")
        if points < int(calibration.min_valid_points):
            pending_action = "DBB_PENDING" if diagnostic_route == "DBB" else "CU_PENDING"
            return HierarchicalActionDecision(
                decision_state="pending",
                calibrated=True,
                gate_action=pending_action,
                conditional_action="PENDING_MINIMUM_EVIDENCE",
                final_action=pending_action,
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=None,
                accepted_label_name=None,
                rejected_to_unknown=None,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=specialist_id,
                specialist_confidence=specialist_confidence,
                specialist_margin=specialist_margin,
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=("insufficient_valid_points",),
                calibration_source=calibration.source,
            )
        quality_reasons: list[str] = []
        if quality < float(calibration.min_track_quality):
            quality_reasons.append("track_quality_below_threshold")
        if gaps > float(calibration.max_gap_fraction):
            quality_reasons.append("gap_fraction_above_threshold")
        if quality_reasons:
            return HierarchicalActionDecision(
                decision_state="pending",
                calibrated=True,
                gate_action="CU_PENDING",
                conditional_action="PENDING_QUALITY_GUARD",
                final_action="CU_PENDING",
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=None,
                accepted_label_name=None,
                rejected_to_unknown=None,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=specialist_id,
                specialist_confidence=specialist_confidence,
                specialist_margin=specialist_margin,
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=tuple(quality_reasons),
                calibration_source=calibration.source,
            )
        gate_accepts_dbb = (
            gate_dbb >= float(calibration.gate_accept_dbb_threshold)
            and gate_margin >= float(calibration.gate_accept_dbb_margin)
        )
        if gate_accepts_dbb:
            dbb_id = int(dbb.argmax().item())
            dbb_top = torch.topk(dbb, k=2).values
            return HierarchicalActionDecision(
                decision_state="accepted",
                calibrated=True,
                gate_action="DBB_ACCEPTED",
                conditional_action=f"{LABELS[dbb_id]}_ACCEPTED",
                final_action="DBB_ACCEPTED",
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=dbb_id,
                accepted_label_name=LABELS[dbb_id],
                rejected_to_unknown=False,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=dbb_id,
                specialist_confidence=float(dbb[dbb_id].item()),
                specialist_margin=float((dbb_top[0] - dbb_top[1]).item()),
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=("gate_dbb_probability_and_margin_accepted",),
                calibration_source=calibration.source,
            )

        cu_clutter = float(cu[0].item())
        cu_margin = float((cu[0] - cu[1]).item())
        clutter_accepted = (
            cu_clutter >= float(calibration.cu_accept_clutter_threshold)
            and cu_margin >= float(calibration.cu_accept_clutter_margin)
        )
        gate_reasons = list(quality_reasons)
        if gate_dbb < float(calibration.gate_accept_dbb_threshold):
            gate_reasons.append("gate_dbb_probability_below_threshold")
        if gate_margin < float(calibration.gate_accept_dbb_margin):
            gate_reasons.append("gate_dbb_margin_below_threshold")
        if clutter_accepted:
            return HierarchicalActionDecision(
                decision_state="accepted",
                calibrated=True,
                gate_action="ROUTED_TO_CU",
                conditional_action="CLUTTER_ACCEPTED",
                final_action="CLUTTER_ACCEPTED",
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=3,
                accepted_label_name=LABELS[3],
                rejected_to_unknown=False,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=0,
                specialist_confidence=cu_clutter,
                specialist_margin=cu_margin,
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=tuple(gate_reasons + ["cu_clutter_probability_and_margin_accepted"]),
                calibration_source=calibration.source,
            )
        reject_reasons = gate_reasons
        if cu_clutter < float(calibration.cu_accept_clutter_threshold):
            reject_reasons.append("cu_clutter_probability_below_threshold")
        if cu_margin < float(calibration.cu_accept_clutter_margin):
            reject_reasons.append("cu_clutter_margin_below_threshold")
        return HierarchicalActionDecision(
            decision_state="rejected",
            calibrated=True,
            gate_action="ROUTED_TO_CU",
            conditional_action="CLUTTER_NOT_ACCEPTED",
            final_action="UNKNOWN_REJECTED",
            diagnostic_gate_route=diagnostic_route,
            diagnostic_label_id=diagnostic_label_id,
            diagnostic_label_name=LABELS[diagnostic_label_id],
            accepted_label_id=None,
            accepted_label_name=None,
            rejected_to_unknown=True,
            gate_dbb_probability=gate_dbb,
            gate_margin=gate_margin,
            specialist_label_id=1,
            specialist_confidence=float(cu[1].item()),
            specialist_margin=float((cu[1] - cu[0]).item()),
            valid_point_count=points,
            track_quality=quality,
            gap_fraction=gaps,
            trigger_reasons=tuple(reject_reasons),
            calibration_source=calibration.source,
        )


def load_hierarchical_rejection_calibration(
    path: str | Path,
) -> tuple[HierarchicalRejectionCalibration, dict[str, Any]]:
    """Load a versioned threshold artifact with mandatory OOF provenance."""

    calibration_path = Path(path).resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source",
        "gate_score_name",
        "cu_score_name",
        "gate_accept_dbb_threshold",
        "gate_accept_dbb_margin",
        "cu_accept_clutter_threshold",
        "cu_accept_clutter_margin",
        "min_valid_points",
        "min_track_quality",
        "max_gap_fraction",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"hierarchical calibration artifact is missing fields: {missing}")
    provenance = dict(payload["provenance"])
    if (
        provenance.get("fit_split") != "train"
        or provenance.get("grouping") != "parent"
        or provenance.get("prediction_protocol") != "out_of_fold"
        or provenance.get("validation_used_for_threshold_fitting") is not False
        or provenance.get("test_used") is not False
    ):
        raise ValueError("hierarchical calibration provenance is not train parent-grouped OOF only")
    hashes = dict(provenance.get("data_contract_hashes") or {})
    if set(hashes) != {"index_sha256", "contract_sha256", "stats_sha256"}:
        raise ValueError("hierarchical calibration lacks exact data contract hashes")
    _normalize_hierarchical_deployment_identity(
        provenance.get("deployment_checkpoint_set"),
        field_name="calibration provenance deployment_checkpoint_set",
    )
    calibration = HierarchicalRejectionCalibration(
        schema_version=str(payload["schema_version"]),
        source=str(payload["source"]),
        gate_score_name=str(payload["gate_score_name"]),
        cu_score_name=str(payload["cu_score_name"]),
        gate_accept_dbb_threshold=float(payload["gate_accept_dbb_threshold"]),
        gate_accept_dbb_margin=float(payload["gate_accept_dbb_margin"]),
        cu_accept_clutter_threshold=float(payload["cu_accept_clutter_threshold"]),
        cu_accept_clutter_margin=float(payload["cu_accept_clutter_margin"]),
        min_valid_points=int(payload["min_valid_points"]),
        min_track_quality=float(payload["min_track_quality"]),
        max_gap_fraction=float(payload["max_gap_fraction"]),
    )
    return calibration, payload


def _normalize_hierarchical_deployment_identity(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    required = {
        "task_config_identity_sha256",
        "checkpoint_sha256",
        "epoch",
        "global_step",
        "product_selection_sha256",
    }
    if set(value) != required:
        raise ValueError(f"{field_name} fields do not match the V13 deployment identity contract")

    def hash_mapping(name: str) -> dict[str, str]:
        mapping = value.get(name)
        if not isinstance(mapping, dict) or set(mapping) != set(HIERARCHICAL_TASK_NAMES):
            raise ValueError(f"{field_name}.{name} must cover exactly the three V13 tasks")
        normalized = {str(key): str(item).lower() for key, item in mapping.items()}
        if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in normalized.values()):
            raise ValueError(f"{field_name}.{name} contains an invalid SHA-256")
        return normalized

    selection_hash = str(value.get("product_selection_sha256", "")).lower()
    if len(selection_hash) != 64 or any(char not in "0123456789abcdef" for char in selection_hash):
        raise ValueError(f"{field_name}.product_selection_sha256 is invalid")
    epoch = int(value.get("epoch", -1))
    global_step = int(value.get("global_step", -1))
    if epoch < 1 or global_step < 1:
        raise ValueError(f"{field_name} epoch/global_step must be positive")
    return {
        "task_config_identity_sha256": hash_mapping("task_config_identity_sha256"),
        "checkpoint_sha256": hash_mapping("checkpoint_sha256"),
        "epoch": epoch,
        "global_step": global_step,
        "product_selection_sha256": selection_hash,
    }


def validate_hierarchical_calibration_deployment(
    payload: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a calibration artifact to target the exact deployed trio."""

    provenance = dict(payload.get("provenance") or {})
    actual = _normalize_hierarchical_deployment_identity(
        provenance.get("deployment_checkpoint_set"),
        field_name="calibration provenance deployment_checkpoint_set",
    )
    expected = _normalize_hierarchical_deployment_identity(
        dict(expected_identity),
        field_name="evaluated deployment checkpoint set",
    )
    if actual != expected:
        raise ValueError(
            "hierarchical calibration was not frozen for the evaluated V13 checkpoint set"
        )
    return actual


__all__ = [
    "MAX_CLASS_PROBABILITY_SCORE",
    "CU_CLUTTER_PROBABILITY_SCORE",
    "GATE_DBB_PROBABILITY_SCORE",
    "HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA",
    "HIERARCHICAL_TASK_NAMES",
    "REJECTION_CALIBRATION_SCHEMA",
    "TRAIN_PARENT_GROUPED_OOF_SOURCE",
    "RejectionCalibration",
    "RejectionDecision",
    "RejectionPolicy",
    "HierarchicalActionDecision",
    "HierarchicalRejectionCalibration",
    "HierarchicalRejectionPolicy",
    "load_hierarchical_rejection_calibration",
    "validate_hierarchical_calibration_deployment",
]
