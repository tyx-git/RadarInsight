"""Fail-closed dynamic L3 action resolver for V27d."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .scorers import classifier_conformal_p_values, conformal_p_values


L3_REGISTRY_SCHEMA = "RADAR-LN3-V27d-dynamic-calibration-registry-v1"


def _sigmoid_scalar(value: float | torch.Tensor) -> float:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
    if tensor.ndim != 0 or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError("V27d L3 scalar logit must be a finite scalar tensor")
    return float(torch.sigmoid(tensor).item())


def resolve_l3_action(
    *,
    registry: Mapping[str, Any] | None,
    l1_endpoint_logits: torch.Tensor,
    l1_knownness_logit: torch.Tensor,
    l1_quality_logit: torch.Tensor,
    l2_endpoint_logits: torch.Tensor,
    l2_knownness_projection: torch.Tensor,
    l2_knownness_logit: torch.Tensor,
    l2_quality_logit: torch.Tensor,
    deterministic_quality_eligible: bool | None = None,
    l1_deterministic_quality_eligible: bool | None = None,
    l2_deterministic_quality_eligible: bool | None = None,
) -> dict[str, Any]:
    """Resolve one parent without allowing an unfitted/ambiguous state to pass."""

    if registry is not None and not isinstance(registry, Mapping):
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": None,
            "conformal_set": [],
            "calibration_error": "registry must be a mapping or None",
        }

    if l1_deterministic_quality_eligible is None:
        l1_deterministic_quality_eligible = deterministic_quality_eligible
    if l2_deterministic_quality_eligible is None:
        l2_deterministic_quality_eligible = deterministic_quality_eligible
    if not isinstance(l1_deterministic_quality_eligible, bool) or not isinstance(
        l2_deterministic_quality_eligible, bool
    ):
        raise TypeError("V27d L3 deterministic quality verdicts must be bool")
    scalar_inputs = {
        "l1_knownness_logit": l1_knownness_logit,
        "l1_quality_logit": l1_quality_logit,
        "l2_knownness_logit": l2_knownness_logit,
        "l2_quality_logit": l2_quality_logit,
    }
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or not value.is_floating_point()
        for value in scalar_inputs.values()
    ):
        raise ValueError("V27d L3 scalar logits must be finite floating-point scalars")
    if (
        l1_endpoint_logits.shape != (2,)
        or l2_endpoint_logits.shape != (3,)
        or not l1_endpoint_logits.is_floating_point()
        or not l2_endpoint_logits.is_floating_point()
    ):
        raise ValueError("V27d L3 endpoint logit shapes differ")
    if not l2_knownness_projection.is_floating_point() or l2_knownness_projection.ndim != 1:
        raise ValueError("V27d L3 knownness projection must be one-dimensional")
    if not all(
        torch.isfinite(value).all()
        for value in (
            l1_endpoint_logits,
            l1_knownness_logit.reshape(1),
            l1_quality_logit.reshape(1),
            l2_endpoint_logits,
            l2_knownness_projection,
            l2_knownness_logit.reshape(1),
            l2_quality_logit.reshape(1),
        )
    ):
        raise ValueError("V27d L3 inputs must be finite")
    if not l1_deterministic_quality_eligible:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "LOW_QUALITY_OR_IMMATURE",
            "knownness_score": None,
            "conformal_set": [],
        }
    if registry is None or registry.get("status") != "fitted":
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": None,
            "conformal_set": [],
        }
    l1_score = _sigmoid_scalar(l1_knownness_logit)
    try:
        l1_threshold = float(registry["l1"]["knownness_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": None,
            "conformal_set": [],
            "calibration_error": str(exc),
        }
    if not l1_score >= l1_threshold:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "OOD_NONCONFORMING",
            "knownness_score": l1_score,
            "conformal_set": [],
        }
    try:
        l1_class_names = tuple(str(value) for value in registry["l1"]["class_names"])
        l1_references = {
            class_name: torch.tensor(
                registry["l1_classifier_calibration_nonconformity"][class_name],
                dtype=l1_endpoint_logits.dtype,
            )
            for class_name in l1_class_names
        }
        l1_p_values = classifier_conformal_p_values(
            l1_endpoint_logits.reshape(1, -1),
            l1_references,
            class_names=l1_class_names,
        )[0]
        l1_p_threshold = float(registry["l1"]["classifier_conformal_p_threshold"])
        l1_conformal_set = [
            name
            for index, name in enumerate(l1_class_names)
            if float(l1_p_values[index]) >= l1_p_threshold
        ]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": l1_score,
            "conformal_set": [],
            "calibration_error": str(exc),
        }
    if not l1_conformal_set:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "OOD_NONCONFORMING",
            "knownness_score": l1_score,
            "conformal_set": [],
            "p_values": l1_p_values.tolist(),
        }
    if len(l1_conformal_set) != 1:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "L1_ROUTE_CONFLICT",
            "knownness_score": l1_score,
            "conformal_set": l1_conformal_set,
            "p_values": l1_p_values.tolist(),
        }
    l1_predicted_name = l1_class_names[int(torch.argmax(l1_endpoint_logits).item())]
    if l1_predicted_name != l1_conformal_set[0]:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "L1_ROUTE_CONFLICT",
            "knownness_score": l1_score,
            "conformal_set": l1_conformal_set,
            "p_values": l1_p_values.tolist(),
        }
    if l1_conformal_set[0] == "Clutter":
        return {
            "action": "Clutter",
            "reason_code": "L1_CLUTTER_ACCEPTED",
            "knownness_score": l1_score,
            "conformal_set": l1_conformal_set,
            "p_values": l1_p_values.tolist(),
        }
    if not l2_deterministic_quality_eligible:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "LOW_QUALITY_OR_IMMATURE",
            "knownness_score": l1_score,
            "conformal_set": [],
        }
    scorer = registry.get("scorer")
    if not isinstance(scorer, Mapping):
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": None,
            "conformal_set": [],
        }
    try:
        projection_dim = int(scorer["projection_dim"])
        l2_threshold = float(registry["l2"]["knownness_threshold"])
        p_threshold = float(registry["l2"]["conformal_p_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "CALIBRATION_NOT_FITTED",
            "knownness_score": None,
            "conformal_set": [],
            "calibration_error": str(exc),
        }
    if l2_knownness_projection.shape != (projection_dim,):
        raise ValueError("V27d L3 knownness projection width differs")
    p_values = conformal_p_values(
        l2_knownness_projection.reshape(1, -1).cpu(),
        scorer,
        registry["scorer_calibration_nonconformity"],
    )[0]
    conformal_set = [
        name
        for index, name in enumerate(registry["l2"]["class_names"])
        if float(p_values[index]) >= p_threshold
    ]
    l2_score = _sigmoid_scalar(l2_knownness_logit)
    if not l2_score >= l2_threshold:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "OOD_NONCONFORMING",
            "knownness_score": l2_score,
            "conformal_set": conformal_set,
            "p_values": p_values.tolist(),
        }
    if not conformal_set:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "OOD_NONCONFORMING",
            "knownness_score": l2_score,
            "conformal_set": [],
            "p_values": p_values.tolist(),
        }
    if len(conformal_set) != 1:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "KNOWN_CLASS_AMBIGUOUS",
            "knownness_score": l2_score,
            "conformal_set": conformal_set,
            "p_values": p_values.tolist(),
        }
    predicted = registry["l2"]["class_names"][int(torch.argmax(l2_endpoint_logits).item())]
    if predicted != conformal_set[0]:
        return {
            "action": "UNKNOWN_REJECTED",
            "reason_code": "KNOWN_CLASS_AMBIGUOUS",
            "knownness_score": l2_score,
            "conformal_set": conformal_set,
            "p_values": p_values.tolist(),
        }
    return {
        "action": predicted,
        "reason_code": "KNOWN_DBB_CLASS_ACCEPTED",
        "knownness_score": l2_score,
        "conformal_set": conformal_set,
        "p_values": p_values.tolist(),
    }


__all__ = ["L3_REGISTRY_SCHEMA", "resolve_l3_action"]