from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from . import LABELS


REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-rejection-calibration-v1"
TRAIN_PARENT_GROUPED_OOF_SOURCE = "train_parent_grouped_oof"
MAX_CLASS_PROBABILITY_SCORE = "max_class_probability"
HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-V13-hierarchical-rejection-calibration-v1"
STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1 = "RADAR-LN3-V14-strict-open-set-rejection-calibration-v1"
STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA = "RADAR-LN3-V14-strict-open-set-rejection-calibration-v2"
V15_CONTROL_REJECTION_CALIBRATION_SCHEMA = (
    "RADAR-LN3-V15-linear-control-rejection-calibration-v1"
)
V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA = (
    "RADAR-LN3-V15-angular-treatment-rejection-calibration-v1"
)
ANGULAR_KNOWNNESS_REJECTION_CALIBRATION_SCHEMA = V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA
STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMAS = frozenset(
    {
        STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1,
        STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        V15_CONTROL_REJECTION_CALIBRATION_SCHEMA,
        V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA,
    }
)
CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE = "deployment_eligible"
CALIBRATION_QUALITY_AUDIT_ONLY = "audit_only"
CALIBRATION_PURPOSE_DEPLOYMENT = "deployment"
CALIBRATION_PURPOSE_AUDIT = "audit"
HIERARCHICAL_TASK_NAMES = ("dbb_cu_gate", "dbb_specialist", "cu_specialist")
GATE_DBB_PROBABILITY_SCORE = "gate_dbb_probability_and_margin"
CU_CLUTTER_PROBABILITY_SCORE = "cu_clutter_probability_and_margin"
DBB_KNOWN_PROBABILITY_SCORE = "dbb_top_probability_and_margin"
DBB_KNOWN_COSINE_SCORE = "dbb_top_cosine_and_cosine_gap"


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
class StrictOpenSetRejectionCalibration:
    """Three known-only acceptance boundaries fitted from train-parent OOF."""

    schema_version: str
    source: str
    gate_score_name: str
    dbb_known_score_name: str
    cu_score_name: str
    gate_accept_dbb_threshold: float
    gate_accept_dbb_margin: float
    dbb_accept_known_threshold: float
    dbb_accept_known_margin: float
    cu_accept_clutter_threshold: float
    cu_accept_clutter_margin: float
    calibration_quality_status: str
    deployment_eligible: bool
    model_schema_version: str | None = None
    paired_arm: str | None = None
    min_valid_points: int = 1
    min_track_quality: float = 0.0
    max_gap_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version not in STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMAS:
            raise ValueError(f"unsupported V14 calibration schema: {self.schema_version!r}")
        if self.source != TRAIN_PARENT_GROUPED_OOF_SOURCE:
            raise ValueError("V14 rejection calibration must come from train parent-grouped OOF")
        if self.gate_score_name != GATE_DBB_PROBABILITY_SCORE:
            raise ValueError(f"unsupported V14 Gate score: {self.gate_score_name!r}")
        if self.schema_version == V15_CONTROL_REJECTION_CALIBRATION_SCHEMA:
            expected_model_schema = "RADAR-LN3-V15"
            expected_arm = "control"
            allowed_dbb_scores = {DBB_KNOWN_PROBABILITY_SCORE}
        elif self.schema_version == V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA:
            expected_model_schema = "RADAR-LN3-V15"
            expected_arm = "treatment"
            allowed_dbb_scores = {DBB_KNOWN_COSINE_SCORE}
        else:
            expected_model_schema = None
            expected_arm = None
            allowed_dbb_scores = {DBB_KNOWN_PROBABILITY_SCORE}
        if self.model_schema_version != expected_model_schema or self.paired_arm != expected_arm:
            raise ValueError(
                "strict-open-set calibration model schema/paired arm does not match its calibration schema"
            )
        if self.dbb_known_score_name not in allowed_dbb_scores:
            raise ValueError(
                f"unsupported strict-open-set DBB knownness score: {self.dbb_known_score_name!r}"
            )
        if self.cu_score_name != CU_CLUTTER_PROBABILITY_SCORE:
            raise ValueError(f"unsupported V14 Clutter score: {self.cu_score_name!r}")
        if self.calibration_quality_status not in {
            CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE,
            CALIBRATION_QUALITY_AUDIT_ONLY,
        }:
            raise ValueError(
                "V14 calibration_quality_status must be deployment_eligible or audit_only"
            )
        if bool(self.deployment_eligible) != (
            self.calibration_quality_status == CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE
        ):
            raise ValueError("V14 calibration quality status and deployment_eligible disagree")
        for name, value in (
            ("gate_accept_dbb_threshold", self.gate_accept_dbb_threshold),
            ("dbb_accept_known_threshold", self.dbb_accept_known_threshold),
            ("cu_accept_clutter_threshold", self.cu_accept_clutter_threshold),
        ):
            lower = (
                -1.0
                if name == "dbb_accept_known_threshold"
                and self.dbb_known_score_name == DBB_KNOWN_COSINE_SCORE
                else 0.0
            )
            if not lower < float(value) < 1.0:
                raise ValueError(f"{name} must be in ({lower},1)")
        for name, value in (
            ("gate_accept_dbb_margin", self.gate_accept_dbb_margin),
            ("dbb_accept_known_margin", self.dbb_accept_known_margin),
            ("cu_accept_clutter_margin", self.cu_accept_clutter_margin),
        ):
            upper = (
                2.0
                if name == "dbb_accept_known_margin"
                and self.dbb_known_score_name == DBB_KNOWN_COSINE_SCORE
                else 1.0
            )
            if not 0.0 <= float(value) < upper:
                raise ValueError(f"{name} must be in [0,{upper})")
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
    dbb_known_confidence: float | None = None
    dbb_known_margin: float | None = None
    clutter_accept_confidence: float | None = None
    clutter_accept_margin: float | None = None


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


class StrictOpenSetRejectionPolicy:
    """V14 product action: Gate, DBB-knownness and Clutter acceptance.

    Unknown is never a learned branch.  A DBB route whose specialist evidence
    is below the knownness boundary and a CU route whose Clutter evidence is
    below its boundary both converge on ``UNKNOWN_REJECTED``.
    """

    def __init__(
        self,
        calibration: StrictOpenSetRejectionCalibration | None = None,
        *,
        enabled: bool = False,
        allow_audit_only: bool = False,
    ) -> None:
        if enabled and calibration is None:
            raise RuntimeError("V14 rejection cannot be enabled without known-only train-parent OOF calibration")
        if (
            enabled
            and calibration is not None
            and not calibration.deployment_eligible
            and not allow_audit_only
        ):
            raise PermissionError(
                "V14 audit-only calibration cannot enable a deployment rejection policy"
            )
        self.calibration = calibration
        self.enabled = bool(enabled)
        self.allow_audit_only = bool(allow_audit_only)

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def decide(
        self,
        gate_probabilities: torch.Tensor,
        dbb_probabilities: torch.Tensor,
        cu_probabilities: torch.Tensor,
        *,
        dbb_knownness_scores: torch.Tensor | None = None,
        valid_point_count: int,
        track_quality: float,
        gap_fraction: float,
    ) -> HierarchicalActionDecision:
        gate = _normalized_probabilities(gate_probabilities, 2, "V14 Gate")
        dbb = _normalized_probabilities(dbb_probabilities, 3, "V14 DBB specialist")
        cu = _normalized_probabilities(cu_probabilities, 2, "V14 Clutter acceptance head")
        points = int(valid_point_count)
        quality = float(track_quality)
        gaps = float(gap_fraction)
        if points < 1 or not 0.0 <= quality <= 1.0 or not 0.0 <= gaps <= 1.0:
            raise ValueError("V14 quality inputs are outside their contract")
        gate_dbb = float(gate[0].item())
        gate_margin = float((gate[0] - gate[1]).item())
        dbb_id = int(dbb.argmax().item())
        if (
            self.calibration is not None
            and self.calibration.dbb_known_score_name == DBB_KNOWN_COSINE_SCORE
            and dbb_knownness_scores is None
        ):
            raise ValueError("cosine-knownness calibration requires explicit DBB cosine scores")
        knownness = dbb if dbb_knownness_scores is None else torch.as_tensor(dbb_knownness_scores, dtype=torch.float32)
        if knownness.shape != (3,) or not torch.isfinite(knownness).all():
            raise ValueError("strict-open-set DBB knownness scores must be three finite values")
        if bool((knownness < -1.0 - 1.0e-6).any()) or bool((knownness > 1.0 + 1.0e-6).any()):
            raise ValueError("strict-open-set DBB knownness scores must be bounded to [-1,1]")
        if int(knownness.argmax().item()) != dbb_id:
            raise ValueError("DBB class logits and knownness scores disagree on the top class")
        dbb_top = torch.topk(knownness, k=2).values
        dbb_confidence = float(knownness[dbb_id].item())
        dbb_margin = float((dbb_top[0] - dbb_top[1]).item())
        cu_clutter = float(cu[0].item())
        cu_margin = float((cu[0] - cu[1]).item())
        diagnostic_route = "DBB" if int(gate.argmax().item()) == 0 else "CU"
        diagnostic_label_id = dbb_id if diagnostic_route == "DBB" else 3
        diagnostic_specialist_id = dbb_id if diagnostic_route == "DBB" else 0
        diagnostic_confidence = dbb_confidence if diagnostic_route == "DBB" else cu_clutter
        diagnostic_margin = dbb_margin if diagnostic_route == "DBB" else cu_margin

        def decision(
            *,
            decision_state: str,
            gate_action: str,
            conditional_action: str,
            final_action: str,
            accepted_label_id: int | None,
            rejected_to_unknown: bool | None,
            reasons: tuple[str, ...],
            specialist_id: int = diagnostic_specialist_id,
            specialist_confidence: float = diagnostic_confidence,
            specialist_margin: float = diagnostic_margin,
        ) -> HierarchicalActionDecision:
            return HierarchicalActionDecision(
                decision_state=decision_state,
                calibrated=self.calibrated,
                gate_action=gate_action,
                conditional_action=conditional_action,
                final_action=final_action,
                diagnostic_gate_route=diagnostic_route,
                diagnostic_label_id=diagnostic_label_id,
                diagnostic_label_name=LABELS[diagnostic_label_id],
                accepted_label_id=accepted_label_id,
                accepted_label_name=None if accepted_label_id is None else LABELS[accepted_label_id],
                rejected_to_unknown=rejected_to_unknown,
                gate_dbb_probability=gate_dbb,
                gate_margin=gate_margin,
                specialist_label_id=specialist_id,
                specialist_confidence=specialist_confidence,
                specialist_margin=specialist_margin,
                valid_point_count=points,
                track_quality=quality,
                gap_fraction=gaps,
                trigger_reasons=reasons,
                calibration_source=None if self.calibration is None else self.calibration.source,
                dbb_known_confidence=dbb_confidence,
                dbb_known_margin=dbb_margin,
                clutter_accept_confidence=cu_clutter,
                clutter_accept_margin=cu_margin,
            )

        if not self.enabled:
            return decision(
                decision_state="not_calibrated" if self.calibration is None else "disabled",
                gate_action="NOT_CALIBRATED",
                conditional_action="NOT_CALIBRATED",
                final_action="NOT_CALIBRATED",
                accepted_label_id=None,
                rejected_to_unknown=None,
                reasons=("missing_known_only_train_parent_grouped_oof_calibration",),
            )
        calibration = self.calibration
        if calibration is None:
            raise RuntimeError("V14 rejection was enabled without calibration")
        dbb_score_is_cosine = calibration.dbb_known_score_name == DBB_KNOWN_COSINE_SCORE
        dbb_accept_reason = (
            "dbb_known_cosine_and_gap_accepted"
            if dbb_score_is_cosine
            else "dbb_known_probability_and_margin_accepted"
        )
        dbb_confidence_below_reason = (
            "dbb_known_cosine_below_threshold"
            if dbb_score_is_cosine
            else "dbb_known_probability_below_threshold"
        )
        dbb_margin_below_reason = (
            "dbb_known_cosine_gap_below_threshold"
            if dbb_score_is_cosine
            else "dbb_known_margin_below_threshold"
        )
        if points < int(calibration.min_valid_points):
            pending = "DBB_PENDING" if diagnostic_route == "DBB" else "CU_PENDING"
            return decision(
                decision_state="pending",
                gate_action=pending,
                conditional_action="PENDING_MINIMUM_EVIDENCE",
                final_action=pending,
                accepted_label_id=None,
                rejected_to_unknown=None,
                reasons=("insufficient_valid_points",),
            )
        quality_reasons: list[str] = []
        if quality < float(calibration.min_track_quality):
            quality_reasons.append("track_quality_below_threshold")
        if gaps > float(calibration.max_gap_fraction):
            quality_reasons.append("gap_fraction_above_threshold")
        if quality_reasons:
            return decision(
                decision_state="pending",
                gate_action="CU_PENDING",
                conditional_action="PENDING_QUALITY_GUARD",
                final_action="CU_PENDING",
                accepted_label_id=None,
                rejected_to_unknown=None,
                reasons=tuple(quality_reasons),
            )

        gate_accepts_dbb = (
            gate_dbb >= float(calibration.gate_accept_dbb_threshold)
            and gate_margin >= float(calibration.gate_accept_dbb_margin)
        )
        if gate_accepts_dbb:
            dbb_accepts_known = (
                dbb_confidence >= float(calibration.dbb_accept_known_threshold)
                and dbb_margin >= float(calibration.dbb_accept_known_margin)
            )
            if dbb_accepts_known:
                return decision(
                    decision_state="accepted",
                    gate_action="DBB_ROUTE_ACCEPTED",
                    conditional_action=f"{LABELS[dbb_id]}_ACCEPTED",
                    final_action="DBB_ACCEPTED",
                    accepted_label_id=dbb_id,
                    rejected_to_unknown=False,
                    reasons=(
                        "gate_dbb_probability_and_margin_accepted",
                        dbb_accept_reason,
                    ),
                    specialist_id=dbb_id,
                    specialist_confidence=dbb_confidence,
                    specialist_margin=dbb_margin,
                )
            reject_reasons = ["gate_dbb_probability_and_margin_accepted"]
            if dbb_confidence < float(calibration.dbb_accept_known_threshold):
                reject_reasons.append(dbb_confidence_below_reason)
            if dbb_margin < float(calibration.dbb_accept_known_margin):
                reject_reasons.append(dbb_margin_below_reason)
            return decision(
                decision_state="rejected",
                gate_action="DBB_ROUTE_ACCEPTED",
                conditional_action="DBB_KNOWN_NOT_ACCEPTED",
                final_action="UNKNOWN_REJECTED",
                accepted_label_id=None,
                rejected_to_unknown=True,
                reasons=tuple(reject_reasons),
                specialist_id=dbb_id,
                specialist_confidence=dbb_confidence,
                specialist_margin=dbb_margin,
            )

        gate_reasons: list[str] = []
        if gate_dbb < float(calibration.gate_accept_dbb_threshold):
            gate_reasons.append("gate_dbb_probability_below_threshold")
        if gate_margin < float(calibration.gate_accept_dbb_margin):
            gate_reasons.append("gate_dbb_margin_below_threshold")
        clutter_accepted = (
            cu_clutter >= float(calibration.cu_accept_clutter_threshold)
            and cu_margin >= float(calibration.cu_accept_clutter_margin)
        )
        if clutter_accepted:
            return decision(
                decision_state="accepted",
                gate_action="ROUTED_TO_CU",
                conditional_action="CLUTTER_ACCEPTED",
                final_action="CLUTTER_ACCEPTED",
                accepted_label_id=3,
                rejected_to_unknown=False,
                reasons=tuple(gate_reasons + ["cu_clutter_probability_and_margin_accepted"]),
                specialist_id=0,
                specialist_confidence=cu_clutter,
                specialist_margin=cu_margin,
            )
        reject_reasons = list(gate_reasons)
        if cu_clutter < float(calibration.cu_accept_clutter_threshold):
            reject_reasons.append("cu_clutter_probability_below_threshold")
        if cu_margin < float(calibration.cu_accept_clutter_margin):
            reject_reasons.append("cu_clutter_margin_below_threshold")
        return decision(
            decision_state="rejected",
            gate_action="ROUTED_TO_CU",
            conditional_action="CLUTTER_NOT_ACCEPTED",
            final_action="UNKNOWN_REJECTED",
            accepted_label_id=None,
            rejected_to_unknown=True,
            reasons=tuple(reject_reasons),
            specialist_id=0,
            specialist_confidence=cu_clutter,
            specialist_margin=cu_margin,
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


def load_strict_open_set_rejection_calibration(
    path: str | Path,
    *,
    purpose: Literal["deployment", "audit"] = CALIBRATION_PURPOSE_DEPLOYMENT,
) -> tuple[StrictOpenSetRejectionCalibration, dict[str, Any]]:
    """Load a schema-bound strict-open-set calibration with a safety boundary."""

    calibration_path = Path(path).resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source",
        "gate_score_name",
        "dbb_known_score_name",
        "cu_score_name",
        "gate_accept_dbb_threshold",
        "gate_accept_dbb_margin",
        "dbb_accept_known_threshold",
        "dbb_accept_known_margin",
        "cu_accept_clutter_threshold",
        "cu_accept_clutter_margin",
        "min_valid_points",
        "min_track_quality",
        "max_gap_fraction",
        "provenance",
        "selection_diagnostics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"strict-open-set calibration artifact is missing fields: {missing}")
    schema_version = str(payload["schema_version"])
    is_v15 = schema_version in {
        V15_CONTROL_REJECTION_CALIBRATION_SCHEMA,
        V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA,
    }
    if is_v15:
        v15_required = {
            "model_schema_version",
            "paired_arm",
            "calibration_quality_status",
            "deployment_eligible",
            "quality_gate",
        }
        v15_missing = sorted(v15_required - set(payload))
        if v15_missing:
            raise ValueError(f"V15 calibration artifact is missing fields: {v15_missing}")
    provenance = dict(payload["provenance"])
    expected_prediction_protocol = "nested_out_of_fold" if is_v15 else "out_of_fold"
    if (
        provenance.get("fit_split") != "train"
        or provenance.get("fit_population") != "known_only"
        or provenance.get("grouping") != "parent"
        or provenance.get("prediction_protocol") != expected_prediction_protocol
        or provenance.get("unknown_training_exposure") != 0
        or provenance.get("unknown_threshold_fit_count") != 0
        or provenance.get("validation_used_for_threshold_fitting") is not False
        or provenance.get("test_used") is not False
    ):
        raise ValueError("strict-open-set calibration provenance is not known-only train parent-grouped OOF")
    if is_v15 and (
        provenance.get("epoch_protocol")
        != "inner_parent_selection_then_outer_train_retrain"
        or provenance.get("outer_held_out_used_for_normalization") is not False
        or provenance.get("outer_held_out_used_for_epoch_selection") is not False
        or int(provenance.get("outer_held_out_prediction_pass_count", -1)) != 1
    ):
        raise ValueError("V15 calibration provenance violates the nested OOF execution contract")
    known_parent_count = int(provenance.get("known_parent_count", -1))
    oof_parent_count = int(provenance.get("oof_prediction_parent_count", -2))
    oof_unique_parent_count = int(provenance.get("oof_unique_parent_count", -3))
    if known_parent_count <= 0 or oof_parent_count != known_parent_count or oof_unique_parent_count != known_parent_count:
        raise ValueError("V14 calibration does not cover every known train parent exactly once")
    hashes = dict(provenance.get("data_contract_hashes") or {})
    if set(hashes) != {"index_sha256", "contract_sha256", "stats_sha256"}:
        raise ValueError("V14 calibration lacks exact data contract hashes")
    _normalize_hierarchical_deployment_identity(
        provenance.get("deployment_checkpoint_set"),
        field_name="V14 calibration provenance deployment_checkpoint_set",
    )
    normalized_payload = dict(payload)
    quality = strict_open_set_calibration_quality(normalized_payload)
    normalized_payload.update(quality)
    if purpose not in {CALIBRATION_PURPOSE_DEPLOYMENT, CALIBRATION_PURPOSE_AUDIT}:
        raise ValueError(f"unsupported V14 calibration purpose: {purpose!r}")
    if purpose == CALIBRATION_PURPOSE_DEPLOYMENT and not quality["deployment_eligible"]:
        reasons = ", ".join(quality["quality_gate"]["failure_reasons"])
        raise PermissionError(
            "V14 calibration is audit_only and cannot be loaded for deployment; "
            f"failed quality gates: {reasons}"
        )
    calibration = StrictOpenSetRejectionCalibration(
        schema_version=str(normalized_payload["schema_version"]),
        source=str(normalized_payload["source"]),
        gate_score_name=str(normalized_payload["gate_score_name"]),
        dbb_known_score_name=str(normalized_payload["dbb_known_score_name"]),
        cu_score_name=str(normalized_payload["cu_score_name"]),
        gate_accept_dbb_threshold=float(normalized_payload["gate_accept_dbb_threshold"]),
        gate_accept_dbb_margin=float(normalized_payload["gate_accept_dbb_margin"]),
        dbb_accept_known_threshold=float(normalized_payload["dbb_accept_known_threshold"]),
        dbb_accept_known_margin=float(normalized_payload["dbb_accept_known_margin"]),
        cu_accept_clutter_threshold=float(normalized_payload["cu_accept_clutter_threshold"]),
        cu_accept_clutter_margin=float(normalized_payload["cu_accept_clutter_margin"]),
        calibration_quality_status=str(normalized_payload["calibration_quality_status"]),
        deployment_eligible=bool(normalized_payload["deployment_eligible"]),
        model_schema_version=(
            None
            if normalized_payload.get("model_schema_version") is None
            else str(normalized_payload["model_schema_version"])
        ),
        paired_arm=(
            None
            if normalized_payload.get("paired_arm") is None
            else str(normalized_payload["paired_arm"])
        ),
        min_valid_points=int(normalized_payload["min_valid_points"]),
        min_track_quality=float(normalized_payload["min_track_quality"]),
        max_gap_fraction=float(normalized_payload["max_gap_fraction"]),
    )
    return calibration, normalized_payload


def strict_open_set_calibration_quality(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive or verify all predeclared strict-open-set OOF quality gates."""

    schema_version = str(payload.get("schema_version", ""))
    if schema_version not in STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMAS:
        raise ValueError(f"unsupported V14 calibration schema: {schema_version!r}")
    diagnostics = payload.get("selection_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("V14 calibration lacks selection_diagnostics for quality gating")
    component_fields = {
        "gate": "gate",
        "dbb_knownness": "dbb_knownness",
        "clutter_acceptance": "clutter_acceptance",
    }
    component_targets_passed: dict[str, bool] = {}
    for output_name, diagnostic_name in component_fields.items():
        diagnostic = diagnostics.get(diagnostic_name)
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"V14 calibration lacks {diagnostic_name} quality diagnostics")
        value = diagnostic.get("all_precision_recall_targets_passed")
        if not isinstance(value, bool):
            raise ValueError(
                f"V14 calibration {diagnostic_name} quality result must be boolean"
            )
        component_targets_passed[output_name] = value
    all_required_targets_passed = all(component_targets_passed.values())
    failure_reasons = [
        f"{name}_precision_recall_target_not_met"
        for name, passed in component_targets_passed.items()
        if not passed
    ]
    expected_status = (
        CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE
        if all_required_targets_passed
        else CALIBRATION_QUALITY_AUDIT_ONLY
    )
    expected_gate = {
        "required_components": list(component_fields),
        "component_targets_passed": component_targets_passed,
        "all_required_targets_passed": all_required_targets_passed,
        "failure_reasons": failure_reasons,
    }
    explicit_quality_schemas = {
        STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        V15_CONTROL_REJECTION_CALIBRATION_SCHEMA,
        V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA,
    }
    if schema_version in explicit_quality_schemas:
        if payload.get("calibration_quality_status") != expected_status:
            raise ValueError("V14 calibration quality status disagrees with OOF diagnostics")
        if payload.get("deployment_eligible") is not all_required_targets_passed:
            raise ValueError("V14 deployment_eligible disagrees with OOF diagnostics")
        if payload.get("quality_gate") != expected_gate:
            raise ValueError("V14 quality_gate disagrees with OOF diagnostics")
        source = "explicit_quality_verified_against_selection_diagnostics"
    else:
        source = "derived_from_legacy_v1_selection_diagnostics"
    return {
        "calibration_quality_status": expected_status,
        "deployment_eligible": all_required_targets_passed,
        "quality_gate": expected_gate,
        "quality_status_source": source,
    }


def validate_strict_open_set_calibration_model_binding(
    calibration: StrictOpenSetRejectionCalibration,
    *,
    model_schema_version: str,
    paired_arm: str | None,
    dbb_known_score_name: str,
) -> None:
    """Reject V14/V15, paired-arm or DBB-score calibration mixing."""

    model_schema = str(model_schema_version)
    arm = None if paired_arm is None else str(paired_arm)
    score = str(dbb_known_score_name)
    if model_schema == "RADAR-LN3-V14":
        allowed_schemas = {
            STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1,
            STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        }
        expected_arm = None
        expected_score = DBB_KNOWN_PROBABILITY_SCORE
    elif model_schema == "RADAR-LN3-V15" and arm == "control":
        allowed_schemas = {V15_CONTROL_REJECTION_CALIBRATION_SCHEMA}
        expected_arm = "control"
        expected_score = DBB_KNOWN_PROBABILITY_SCORE
    elif model_schema == "RADAR-LN3-V15" and arm == "treatment":
        allowed_schemas = {V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA}
        expected_arm = "treatment"
        expected_score = DBB_KNOWN_COSINE_SCORE
    else:
        raise ValueError(
            "strict-open-set calibration binding requires V14 or a declared V15 paired arm"
        )
    if (
        calibration.schema_version not in allowed_schemas
        or calibration.model_schema_version != (None if model_schema == "RADAR-LN3-V14" else model_schema)
        or calibration.paired_arm != expected_arm
        or calibration.dbb_known_score_name != expected_score
        or score != expected_score
    ):
        raise ValueError(
            "strict-open-set calibration does not match the evaluated model version, paired arm, or DBB knownness score"
        )


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
    "DBB_KNOWN_COSINE_SCORE",
    "DBB_KNOWN_PROBABILITY_SCORE",
    "GATE_DBB_PROBABILITY_SCORE",
    "CALIBRATION_PURPOSE_AUDIT",
    "CALIBRATION_PURPOSE_DEPLOYMENT",
    "CALIBRATION_QUALITY_AUDIT_ONLY",
    "CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE",
    "HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA",
    "STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA",
    "STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1",
    "STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMAS",
    "V15_CONTROL_REJECTION_CALIBRATION_SCHEMA",
    "V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA",
    "ANGULAR_KNOWNNESS_REJECTION_CALIBRATION_SCHEMA",
    "HIERARCHICAL_TASK_NAMES",
    "REJECTION_CALIBRATION_SCHEMA",
    "TRAIN_PARENT_GROUPED_OOF_SOURCE",
    "RejectionCalibration",
    "RejectionDecision",
    "RejectionPolicy",
    "HierarchicalActionDecision",
    "HierarchicalRejectionCalibration",
    "HierarchicalRejectionPolicy",
    "StrictOpenSetRejectionCalibration",
    "StrictOpenSetRejectionPolicy",
    "load_hierarchical_rejection_calibration",
    "load_strict_open_set_rejection_calibration",
    "strict_open_set_calibration_quality",
    "validate_strict_open_set_calibration_model_binding",
    "validate_hierarchical_calibration_deployment",
]
