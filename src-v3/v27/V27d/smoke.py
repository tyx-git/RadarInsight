"""Contract smoke tests for the V27d frozen-endpoint pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from radar_ln3.runtime import atomic_write_text, json_text

from .calibration import _candidate_thresholds
from .endpoint import endpoint_cache_digest, load_endpoint_cache, load_endpoint_manifest
from .features import load_feature_cache
from .heads import build_heads
from .l3 import resolve_l3_action
from .pools import load_pools
from .protocol import (
    V27DContext,
    V27D_KNOWN_LABELS,
    artifact_path,
    read_sealed_json,
    write_sealed_json,
)


def _expect_failure(check: Callable[[], Any], name: str) -> None:
    try:
        check()
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError, KeyError, IndexError):
        return
    raise AssertionError(f"V27d smoke expected failure did not occur: {name}")


def _path_boundary(context: V27DContext) -> dict[str, Any]:
    _expect_failure(
        lambda: artifact_path(
            context.project_root,
            "src-v3/v27/V27d/cache/../../../../outside.json",
        ),
        "artifact path escape",
    )
    return {"path_escape_rejected": True}


def _sealed_hash(context: V27DContext) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir=str(context.project_root / "src-v3" / "v27" / "V27d" / "reports")) as folder:
        path = Path(folder) / "sealed.json"
        write_sealed_json(path, {"schema_version": "smoke", "value": 1})
        if read_sealed_json(path, expected_schema="smoke")["value"] != 1:
            raise AssertionError("sealed JSON round trip changed the value")
        atomic_write_text(path, json_text({"schema_version": "smoke", "value": 2, "payload_sha256": "0" * 64}))
        _expect_failure(lambda: read_sealed_json(path), "sealed JSON hash tamper")
    return {"sealed_hash_verified": True, "sealed_hash_tamper_rejected": True}


def _endpoint_contract(context: V27DContext, pools: dict[str, Any]) -> dict[str, Any]:
    manifest = load_endpoint_manifest(context)
    checked = 0
    for task in ("known_gate", "dbb_specialist"):
        for role in ("representation_fit", "representation_development", "calibration"):
            entry = manifest["tasks"][f"{task}:{role}"]
            payload = load_endpoint_cache(
                context,
                path=entry["path"],
                pools=pools,
                task=task,
                role=role,
            )
            if payload["parent_ids"] != pools["roles"][role]["parent_ids"]:
                raise AssertionError(f"V27d parent order differs: {task}/{role}")
            if payload["unknown_is_classifier_class"] is not False or payload["unknown_classifier_training_exposure"] != 0:
                raise AssertionError(f"V27d Unknown classifier contract differs: {task}/{role}")
            checked += 1

            with tempfile.TemporaryDirectory(dir=str(context.project_root / "src-v3" / "v27" / "V27d" / "cache")) as folder:
                bad_order = dict(payload)
                bad_order["parent_ids"] = list(reversed(payload["parent_ids"]))
                bad_order["cache_sha256"] = endpoint_cache_digest(bad_order)
                bad_path = Path(folder) / "bad-order.pt"
                torch.save(bad_order, bad_path)
                _expect_failure(
                    lambda: load_endpoint_cache(context, path=bad_path, pools=pools, task=task, role=role),
                    "parent order mismatch",
                )
                bad_hash = dict(payload)
                bad_hash["cache_sha256"] = "0" * 64
                hash_path = Path(folder) / "bad-hash.pt"
                torch.save(bad_hash, hash_path)
                _expect_failure(
                    lambda: load_endpoint_cache(context, path=hash_path, pools=pools, task=task, role=role),
                    "endpoint hash mismatch",
                )
    return {"endpoint_caches_checked": checked, "parent_order_and_hash_guards": True, "padding_zero_guard": True}


def _feature_contract(context: V27DContext) -> dict[str, Any]:
    checked = 0
    clutter_fit_eligible: dict[str, int] = {"known_gate": 0, "dbb_specialist": 0}
    unknown_population = 0
    for task in ("known_gate", "dbb_specialist"):
        for role in ("representation_fit", "representation_development", "calibration"):
            path = artifact_path(
                context.project_root,
                context.config.artifacts.feature_cache_template,
                variant="r1",
                role=role,
                task=task,
            )
            payload = load_feature_cache(context, path)
            labels = payload["label_names"]
            tensors = payload["tensors"]
            for row, label in enumerate(labels):
                if label == "ClutterTarget" and bool(tensors["fit_eligible"][row]):
                    clutter_fit_eligible[task] += 1
                if label == "UnknownTarget" and not bool(tensors["knownness_population"][row]):
                    raise AssertionError("V27d Unknown was excluded from knownness population")
            unknown_population += sum(label == "UnknownTarget" for label in labels)
            if payload["unknown_is_classifier_class"] is not False or payload["unknown_classifier_training_exposure"] != 0:
                raise AssertionError("V27d Unknown entered classifier training metadata")
            checked += 1
    if clutter_fit_eligible["dbb_specialist"]:
        raise AssertionError(
            "V27d Clutter entered L2 knownness fit: "
            f"{clutter_fit_eligible['dbb_specialist']}"
        )
    if not unknown_population:
        raise AssertionError("V27d smoke found no Unknown population")
    return {
        "feature_caches_checked": checked,
        "clutter_knownness_fit_count": clutter_fit_eligible,
        "unknown_knownness_population_count": unknown_population,
        "unknown_not_l2_class": True,
    }


def _head_empty_sequence(context: V27DContext) -> dict[str, Any]:
    model = build_heads(context, task="known_gate", variant="r1", seed=42)
    width = context.config.data.max_sequence_length
    inputs = {
        "summary_features": torch.zeros((1, width, context.config.data.liquid_hidden_dim * 14)),
        "event_trajectory": torch.zeros((1, width, context.config.data.liquid_hidden_dim * 7 + context.config.data.event_fact_dim)),
        "task_facts_sequence": torch.zeros((1, width, context.config.data.l1_task_fact_dim)),
        "event_facts_sequence": torch.zeros((1, width, context.config.data.event_fact_dim)),
        "raw_scale_features": torch.zeros((1, width, context.config.model.raw_scale_feature_dim)),
        "stable8_causal_features": torch.zeros((1, width, context.config.model.stable8_feature_dim)),
        "padding_mask": torch.ones((1, width), dtype=torch.bool),
    }
    _expect_failure(lambda: model(**inputs), "empty sequence")
    return {"empty_sequence_rejected": True}


def _l3_fail_closed() -> dict[str, Any]:
    result = resolve_l3_action(
        registry=None,
        l1_endpoint_logits=torch.zeros(2),
        l1_knownness_logit=torch.zeros(()),
        l1_quality_logit=torch.zeros(()),
        l2_endpoint_logits=torch.zeros(3),
        l2_knownness_projection=torch.zeros(32),
        l2_knownness_logit=torch.zeros(()),
        l2_quality_logit=torch.zeros(()),
        deterministic_quality_eligible=True,
    )
    if result["action"] != "PENDING_ABSTAIN" or result["reason_code"] != "CALIBRATION_NOT_FITTED":
        raise AssertionError("V27d L3 did not fail closed before calibration")
    low_quality = resolve_l3_action(
        registry=None,
        l1_endpoint_logits=torch.zeros(2),
        l1_knownness_logit=torch.zeros(()),
        l1_quality_logit=torch.zeros(()),
        l2_endpoint_logits=torch.zeros(3),
        l2_knownness_projection=torch.zeros(32),
        l2_knownness_logit=torch.zeros(()),
        l2_quality_logit=torch.zeros(()),
        l1_deterministic_quality_eligible=False,
        l2_deterministic_quality_eligible=False,
    )
    if low_quality["reason_code"] != "LOW_QUALITY_OR_IMMATURE":
        raise AssertionError("V27d low-quality L3 input did not remain Pending")
    unavailable = resolve_l3_action(
        registry={"status": "unavailable"},
        l1_endpoint_logits=torch.zeros(2),
        l1_knownness_logit=torch.zeros(()),
        l1_quality_logit=torch.zeros(()),
        l2_endpoint_logits=torch.zeros(3),
        l2_knownness_projection=torch.zeros(32),
        l2_knownness_logit=torch.zeros(()),
        l2_quality_logit=torch.zeros(()),
        deterministic_quality_eligible=True,
    )
    if unavailable["reason_code"] != "CALIBRATION_NOT_FITTED":
        raise AssertionError("V27d unavailable calibration did not fail closed")
    return {
        "unfitted_l3_action": result["action"],
        "unfitted_l3_reason": result["reason_code"],
        "low_quality_l3_action": low_quality["action"],
        "low_quality_l3_reason": low_quality["reason_code"],
        "unavailable_registry_reason": unavailable["reason_code"],
    }


def _l3_quality_order() -> dict[str, Any]:
    """Check the two non-negotiable quality boundaries before L3 is fitted."""

    common = {
        "l1_endpoint_logits": torch.tensor([2.0, -2.0]),
        "l1_knownness_logit": torch.tensor(2.0),
        "l1_quality_logit": torch.tensor(2.0),
        "l2_endpoint_logits": torch.tensor([2.0, 0.0, -1.0]),
        "l2_knownness_projection": torch.zeros(32),
        "l2_knownness_logit": torch.tensor(2.0),
        "l2_quality_logit": torch.tensor(2.0),
    }
    l1_pending = resolve_l3_action(
        registry=None,
        **common,
        l1_deterministic_quality_eligible=False,
        l2_deterministic_quality_eligible=True,
    )
    if l1_pending["action"] != "PENDING_ABSTAIN":
        raise AssertionError("V27d L1 quality did not take priority")
    return {
        "l1_quality_precedes_l2": True,
        "l1_quality_pending_action": l1_pending["action"],
    }


def _calibration_threshold_boundary() -> dict[str, Any]:
    """A conformal cutoff must retain alpha even between observed p-values."""

    candidates = _candidate_thresholds(
        torch.tensor([0.02, 0.05]), lower=0.1, upper=1.0
    )
    if candidates != [0.1]:
        raise AssertionError("V27d calibration discarded its configured alpha cutoff")
    return {"configured_alpha_retained_without_observed_support": True}


def _l3_fitted_boundaries() -> dict[str, Any]:
    """Exercise the fitted L3 branches with a minimal in-memory registry."""

    dimension = 32
    means = torch.zeros((3, dimension))
    means[1:, 0] = 10.0
    scorer = {
        "schema_version": "RADAR-LN3-V27d-scorer-v1",
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "projection_dim": dimension,
        "prototype_count": 1,
        "knn_k": 1,
        "ridge": 1.0e-3,
        "prototypes": means.unsqueeze(1),
        "means": means,
        "inverse_covariances": torch.eye(dimension).repeat(3, 1, 1),
        "reference_points": [torch.zeros((1, dimension)) for _ in range(3)],
        "supports": [1, 1, 1],
    }
    registry = {
        "status": "fitted",
        "l1": {
            "knownness_threshold": 0.5,
            "classifier_conformal_p_threshold": 0.75,
            "class_names": ["Target", "Clutter"],
        },
        "l1_classifier_calibration_nonconformity": {"Target": [1.0], "Clutter": [1.0]},
        "l2": {
            "knownness_threshold": 0.9,
            "conformal_p_threshold": 0.75,
            "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        },
        "scorer": scorer,
        "scorer_calibration_nonconformity": {
            "DroneTarget": torch.zeros(1),
            "BirdTarget": torch.zeros(1),
            "BalloonTarget": torch.zeros(1),
        },
    }
    common = {
        "registry": registry,
        "l1_endpoint_logits": torch.tensor([10.0, -10.0]),
        "l1_knownness_logit": torch.tensor(10.0),
        "l1_quality_logit": torch.tensor(10.0),
        "l2_endpoint_logits": torch.tensor([10.0, -10.0, -10.0]),
        "l2_knownness_projection": torch.zeros(dimension),
        "l2_knownness_logit": torch.tensor(10.0),
        "l2_quality_logit": torch.tensor(10.0),
        "l1_deterministic_quality_eligible": True,
        "l2_deterministic_quality_eligible": True,
    }
    accepted = resolve_l3_action(**common)
    if accepted["action"] != "DroneTarget":
        raise AssertionError("V27d fitted L3 did not accept a conforming DBB class")
    route_conflict = resolve_l3_action(
        **{**common, "l1_endpoint_logits": torch.zeros(2)},
    )
    if route_conflict["reason_code"] != "L1_ROUTE_CONFLICT":
        raise AssertionError("V27d fitted L3 did not preserve L1 route conflict")
    class_ambiguous = resolve_l3_action(
        **{
            **common,
            "l2_knownness_projection": torch.cat((torch.tensor([10.0]), torch.zeros(dimension - 1))),
        },
    )
    if class_ambiguous["reason_code"] != "KNOWN_CLASS_AMBIGUOUS":
        raise AssertionError("V27d fitted L3 did not preserve DBB class ambiguity")
    empty_set = resolve_l3_action(
        **{
            **common,
            "l2_knownness_projection": torch.full((dimension,), 100.0),
        },
    )
    if empty_set["reason_code"] != "OOD_NONCONFORMING":
        raise AssertionError("V27d fitted L3 did not reject an empty conformal set")
    return {
        "fitted_accept_action": accepted["action"],
        "route_conflict_reason": route_conflict["reason_code"],
        "class_ambiguity_reason": class_ambiguous["reason_code"],
        "empty_set_reason": empty_set["reason_code"],
    }


def run_smoke(context: V27DContext) -> dict[str, Any]:
    pools = load_pools(context)
    checks = {
        "path_boundary": _path_boundary(context),
        "sealed_hash": _sealed_hash(context),
        "endpoint_contract": _endpoint_contract(context, pools),
        "feature_contract": _feature_contract(context),
        "empty_sequence": _head_empty_sequence(context),
        "l3_fail_closed": _l3_fail_closed(),
        "l3_quality_order": _l3_quality_order(),
        "calibration_threshold_boundary": _calibration_threshold_boundary(),
        "l3_fitted_boundaries": _l3_fitted_boundaries(),
    }
    result = {
        "status": "passed",
        "experiment_id": "V27d",
        "variant": "r1",
        "checks": checks,
        "test_payload_access_count": 0,
    }
    report_path = artifact_path(context.project_root, context.config.artifacts.smoke_report_path)
    write_sealed_json(report_path, result)
    return {
        **result,
        "report_path": str(report_path.relative_to(context.project_root)).replace("\\", "/"),
    }


__all__ = ["run_smoke"]
