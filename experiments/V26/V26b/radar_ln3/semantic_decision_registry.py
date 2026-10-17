"""Immutable L3 semantic-decision registry contracts.

L3 is deliberately a decision/calibration layer, not an alternative storage
location for L1/L2 training hyperparameters.  Each model keeps its own config
and checkpoint as the authoritative training record; an L3 registry binds
those artifacts by path and SHA-256, then stores only the score definitions,
thresholds and action rules that are fitted after model training.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any, Mapping

from .open_set_architecture import (
    ARCHITECTURE_SCHEMA,
    DBB_SPECIALIST_TASK_NAME,
    KNOWN_GATE_TASK_NAME,
    SEMANTIC_DECISION_LAYER_NAME,
)
from .runtime import resolve_project_path, verify_file_hash


SEMANTIC_DECISION_REGISTRY_SCHEMA = (
    "RADAR-LN3-semantic-decision-layer-registry-v1"
)
UNFITTED_CALIBRATION_STATUS = "unfitted_raw_routing_audit_only"
FITTED_CALIBRATION_STATUS = "fitted_cross_validated"


def validate_semantic_decision_registry(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    verify_artifact_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a minimal immutable L3 registry without loading model weights.

    The validator intentionally permits an *unfitted* registry.  It is useful
    for proving which frozen L1/L2 endpoints a raw-routing audit used, but it
    cannot authorize a final product action or an open-set claim until a later
    cross-fitted calibration writes non-null acceptance thresholds.
    """

    value = dict(payload)
    if value.get("schema_version") != SEMANTIC_DECISION_REGISTRY_SCHEMA:
        raise ValueError("semantic-decision registry schema is invalid")
    if value.get("layer_name") != SEMANTIC_DECISION_LAYER_NAME:
        raise ValueError("semantic-decision registry layer name is invalid")
    if value.get("architecture_schema") != ARCHITECTURE_SCHEMA:
        raise ValueError("semantic-decision registry architecture schema is invalid")
    if value.get("unknown_is_learned_class") is not False:
        raise ValueError("semantic-decision registry must preserve strict open-set Unknown")
    if value.get("pending_is_unknown_rejection_success") is not False:
        raise ValueError("semantic-decision registry must not count Pending as rejection")

    components = dict(value.get("component_bindings") or {})
    required_components = {"l1_known_gate", "l2_dbb_specialist"}
    if set(components) != required_components:
        raise ValueError("semantic-decision registry component bindings are incomplete")
    expected_task_names = {
        "l1_known_gate": KNOWN_GATE_TASK_NAME,
        "l2_dbb_specialist": DBB_SPECIALIST_TASK_NAME,
    }
    for key, task_name in expected_task_names.items():
        binding = dict(components[key] or {})
        if binding.get("task_name") != task_name:
            raise ValueError(f"semantic-decision registry {key} task is invalid")
        for field in (
            "config_path",
            "config_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "training_summary_path",
            "training_summary_sha256",
        ):
            field_value = binding.get(field)
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"semantic-decision registry {key} lacks immutable {field}"
                )
        for field in (
            "config_sha256",
            "checkpoint_sha256",
            "training_summary_sha256",
        ):
            digest = str(binding[field])
            if len(digest) != 64 or any(character not in string.hexdigits for character in digest):
                raise ValueError(
                    f"semantic-decision registry {key} has an invalid {field}"
                )
        if verify_artifact_hashes:
            if project_root is None:
                raise ValueError("project_root is required for artifact hash verification")
            for path_field, hash_field in (
                ("config_path", "config_sha256"),
                ("checkpoint_path", "checkpoint_sha256"),
                ("training_summary_path", "training_summary_sha256"),
            ):
                artifact_path = resolve_project_path(
                    project_root,
                    binding[path_field],
                    must_exist=True,
                )
                verify_file_hash(
                    artifact_path,
                    binding[hash_field],
                    field=f"{key}.{hash_field}",
                )
            optional_readiness_path = binding.get("training_readiness_path")
            optional_readiness_hash = binding.get("training_readiness_sha256")
            if (optional_readiness_path is None) != (optional_readiness_hash is None):
                raise ValueError(
                    f"semantic-decision registry {key} readiness path/hash must be paired"
                )
            if optional_readiness_path is not None:
                verify_file_hash(
                    resolve_project_path(
                        project_root,
                        optional_readiness_path,
                        must_exist=True,
                    ),
                    optional_readiness_hash,
                    field=f"{key}.training_readiness_sha256",
                )

    calibration = dict(value.get("calibration") or {})
    status = calibration.get("status")
    if status not in {UNFITTED_CALIBRATION_STATUS, FITTED_CALIBRATION_STATUS}:
        raise ValueError("semantic-decision registry calibration status is invalid")
    score_definitions = dict(calibration.get("score_definitions") or {})
    required_scores = {
        "l1_known_target_acceptance",
        "l1_known_clutter_acceptance",
        "l2_subclass_acceptance",
    }
    if set(score_definitions) != required_scores:
        raise ValueError("semantic-decision registry score definitions are incomplete")
    thresholds = dict(calibration.get("thresholds") or {})
    if set(thresholds) != required_scores:
        raise ValueError("semantic-decision registry thresholds are incomplete")
    if status == UNFITTED_CALIBRATION_STATUS:
        if any(thresholds[name] is not None for name in required_scores):
            raise ValueError("unfitted L3 registry must not contain acceptance thresholds")
        if value.get("final_actions_enabled") is not False:
            raise ValueError("unfitted L3 registry cannot enable final actions")
    else:
        if any(thresholds[name] is None for name in required_scores):
            raise ValueError("fitted L3 registry requires all acceptance thresholds")
        if value.get("final_actions_enabled") is not True:
            raise ValueError("fitted L3 registry must explicitly enable final actions")

    semantics = dict(value.get("action_semantics") or {})
    if semantics.get("resolver") != "resolve_semantic_action":
        raise ValueError("semantic-decision registry action resolver is invalid")
    if semantics.get("minimum_evidence_events") != 3:
        raise ValueError("semantic-decision registry eligibility rule is invalid")
    return value




__all__ = ["FITTED_CALIBRATION_STATUS", "SEMANTIC_DECISION_REGISTRY_SCHEMA", "UNFITTED_CALIBRATION_STATUS", "validate_semantic_decision_registry"]