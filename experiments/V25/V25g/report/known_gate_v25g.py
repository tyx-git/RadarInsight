"""Evaluate V25g's frozen causal ACCEPT/PENDING policy on V25a epoch14."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v25f import _source
from build.known_gate_v25g import (
    CONTRACT_SCHEMA,
    EXPECTED_TRAIN_PARENTS,
    EXPECTED_VALIDATION_CLUTTER,
    EXPECTED_VALIDATION_PARENTS,
    EXPECTED_V25A_BEST_EPOCH,
    EXPERIMENT_CONTRACT,
    MAX_V25G_JSON_BYTES,
    READINESS,
    REPORT,
    V25G_SCHEMA,
)
from radar_ln3.known_gate import (
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    known_gate_device,
    sha256_file,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v25g_policy import V25G_BASE_LABELS, V25G_MAX_THRESHOLD_CANDIDATES, V25G_MINIMUM_EVIDENCE_EVENTS, apply_accept_pending_policy, class_metric, directional_action_changes, first_causal_decisions, select_train_only_threshold, summarize_policy_records
from report.known_gate_v25f import _verify_epoch_payload


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 25071
BASELINE_THRESHOLD = 0.5
EXPECTED_VALIDATION_WINDOW_RECORDS = 4452
EXPECTED_VALIDATION_ELIGIBLE_PREFIX_RECORDS = 3756
MAX_TRAIN_CALIBRATION_RECORDS = 200_000
MAX_VALIDATION_RECORDS = 50_000
MAX_RD_TENSOR_CACHE_ITEMS = 1


def _slim_calibration_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only fields required by threshold fitting before releasing train data."""

    required = (
        "parent_sample_id",
        "track_id",
        "source_name",
        "label_name",
        "true_id",
        "pred_id",
        "confidence",
        "event_index",
        "window_length",
        "product_action_eligible",
    )
    output: list[dict[str, Any]] = []
    for row in records:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"V25g calibration record is missing fields: {missing}")
        output.append({name: row[name] for name in required})
    if len(output) > MAX_TRAIN_CALIBRATION_RECORDS:
        raise MemoryError(
            "V25g train calibration record count exceeds the fail-closed limit: "
            f"{len(output)} > {MAX_TRAIN_CALIBRATION_RECORDS}"
        )
    return output


def _release_memory(*, model: torch.nn.Module | None = None) -> None:
    """Release CPU references and allocator blocks between split evaluations."""

    if model is not None:
        model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _harden_dataset_tensor_cache(dataset: Any) -> None:
    """Bound the raw RD tensor LRU for this read-only report."""

    base = getattr(dataset, "base", None)
    cache = getattr(base, "_tensor_cache", None)
    items = getattr(cache, "_items", None)
    if cache is None or items is None or not hasattr(cache, "max_size"):
        raise RuntimeError("V25g dataset does not expose the expected RD tensor cache")
    cache.max_size = MAX_RD_TENSOR_CACHE_ITEMS
    items.clear()


def _clear_dataset_tensor_cache(dataset: Any) -> None:
    base = getattr(dataset, "base", None)
    cache = getattr(base, "_tensor_cache", None)
    items = getattr(cache, "_items", None)
    if items is not None:
        items.clear()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V25g JSON source is missing: {path}")
    if path.stat().st_size > MAX_V25G_JSON_BYTES:
        raise ValueError(f"V25g JSON source exceeds the size limit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"V25g JSON source must contain an object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_V25G_JSON_BYTES:
        raise ValueError(f"V25g report exceeds the size limit: {path}")
    if path.exists():
        if path.stat().st_size > MAX_V25G_JSON_BYTES:
            raise ValueError(f"existing V25g report exceeds the size limit: {path}")
        if path.read_text(encoding="utf-8") == text:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_preparation(root: Path) -> dict[str, Any]:
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    contract = _load_json(contract_path)
    readiness = _load_json(readiness_path)
    checks = dict(readiness.get("checks") or {})
    contract_policy = dict(contract.get("policy") or {})
    contract_safety = dict(contract.get("resource_safety") or {})
    expected_checks = {
        "v25a_source_is_completed_seed43_epoch14",
        "v25a_has_sealed_parent_orders_and_checkpoint",
        "v25a_test_not_executed",
        "v25a_unknown_training_exposure_is_zero",
        "v25g_has_zero_training_and_optimizer_steps",
        "v25g_does_not_create_or_mutate_feature_cache",
        "pending_is_separate_from_unknown",
        "prefix_minimum_events_is_frozen_at_three",
        "validation_is_not_used_for_threshold_fit",
        "exact_oof_is_not_claimed",
    }
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != V25G_SCHEMA
        or contract.get("policy_schema") != readiness.get("policy_schema")
        or readiness.get("training") != "not_executed"
        or readiness.get("calibration") != "train_only_frozen_checkpoint_in_sample"
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or readiness.get("validation_used_for_threshold_fit") is not False
        or set(checks) != expected_checks
        or not all(value is True for value in checks.values())
        or contract.get("deployment_eligible") is not False
        or contract.get("calibration", {}).get("is_oof") is not False
        or contract_policy.get("threshold_candidates")
        != "0.5 plus deterministic quantile representatives of train-only eligible-prefix confidence values, capped at 256"
        or int(contract_safety.get("max_threshold_candidates", -1))
        != V25G_MAX_THRESHOLD_CANDIDATES
        or contract_safety.get("sequential_split_loading") is not True
        or int(contract_safety.get("max_train_calibration_records", -1)) != MAX_TRAIN_CALIBRATION_RECORDS
        or int(contract_safety.get("max_validation_records", -1)) != MAX_VALIDATION_RECORDS
    ):
        raise ValueError("V25g preparation artifacts are stale or outside the frozen scope")
    if dict(readiness.get("experiment_contract") or {}).get("sha256") != sha256_file(contract_path):
        raise ValueError("V25g readiness is not bound to the experiment contract")
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
    }


def _raw_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    records = dict(evaluation.get("records") or {})
    parent = [dict(row) for row in list(records.get("parent") or ())]
    eligible = [dict(row) for row in list(records.get("eligible_prefix") or ())]
    window = [dict(row) for row in list(records.get("window") or ())]
    return {
        "parent_final": summarize_known_gate_records(parent, scope="parent_final"),
        "eligible_prefix_event_pooled": summarize_known_gate_records(
            eligible, scope="eligible_causal_prefix_auxiliary_excluding_final"
        ),
        "eligible_prefix_parent_balanced": summarize_parent_balanced_eligible_prefix_records(
            eligible,
            scope="parent_balanced_eligible_causal_prefix_auxiliary_excluding_final",
        ),
        "window_event_pooled": summarize_known_gate_records(
            window, scope="causal_prefix_windows"
        ),
    }


def _policy_metrics(evaluation: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    records = dict(evaluation.get("records") or {})
    eligible = [dict(row) for row in list(records.get("eligible_prefix") or ())]
    if not eligible:
        raise ValueError("V25g validation has no eligible prefix records")
    policy = apply_accept_pending_policy(
        eligible,
        threshold=threshold,
        minimum_events=V25G_MINIMUM_EVIDENCE_EVENTS,
    )
    decisions = first_causal_decisions(policy)
    return {
        "threshold": float(threshold),
        "event_pooled": summarize_policy_records(
            policy,
            scope="eligible_causal_prefix_policy_actions",
        ),
        "parent_balanced": summarize_policy_records(
            policy,
            scope="parent_balanced_eligible_causal_prefix_policy_actions",
            parent_balanced=True,
        ),
        "first_causal_decision": summarize_policy_records(
            decisions,
            scope="first_eligible_causal_accept_or_pending_per_parent",
        ),
        "records": {
            "eligible_prefix": policy,
            "first_causal_decision": decisions,
        },
        "directional_changes_vs_local_argmax": directional_action_changes(
            eligible,
            policy,
        ),
    }


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("V25g bootstrap values are empty or non-finite")
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(
    *,
    parent_decisions: Sequence[Mapping[str, Any]],
    eligible_policy: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    parent_by_id = {
        str(row["parent_sample_id"]): dict(row) for row in parent_decisions
    }
    if len(parent_by_id) != len(parent_decisions):
        raise ValueError("V25g bootstrap parent decisions repeat keys")
    prefixes: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_policy:
        prefixes.setdefault(str(row["parent_sample_id"]), []).append(dict(row))
    strata = {
        label: [
            parent_id
            for parent_id, row in parent_by_id.items()
            if str(row.get("label_name")) == label
        ]
        for label in V25G_BASE_LABELS
    }
    expected = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": EXPECTED_VALIDATION_CLUTTER,
    }
    if {label: len(ids) for label, ids in strata.items()} != expected:
        raise ValueError("V25g validation bootstrap strata differ from frozen support")
    rng = np.random.default_rng(seed)
    parent_clutter_precision: list[float] = []
    parent_clutter_recall: list[float] = []
    parent_clutter_f1: list[float] = []
    prefix_clutter_precision: list[float] = []
    prefix_clutter_recall: list[float] = []
    prefix_clutter_f1: list[float] = []
    prefix_target_recall: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            str(parent_id)
            for label in V25G_BASE_LABELS
            for parent_id in rng.choice(
                np.asarray(strata[label], dtype=object),
                size=expected[label],
                replace=True,
            )
        ]
        sampled_parents: list[dict[str, Any]] = []
        sampled_prefix: list[dict[str, Any]] = []
        for index, parent_id in enumerate(sampled_ids):
            synthetic_id = f"{parent_id}#v25g_bootstrap_{index}"
            parent_row = dict(parent_by_id[parent_id])
            parent_row["parent_sample_id"] = synthetic_id
            sampled_parents.append(parent_row)
            for prefix_row in prefixes.get(parent_id, ()):
                cloned = dict(prefix_row)
                cloned["parent_sample_id"] = synthetic_id
                sampled_prefix.append(cloned)
        parent_metrics = summarize_policy_records(
            sampled_parents,
            scope="v25g_parent_decision_bootstrap",
        )
        prefix_metrics = summarize_policy_records(
            sampled_prefix,
            scope="v25g_parent_balanced_prefix_bootstrap",
            parent_balanced=True,
        )
        parent_clutter_precision.append(
            class_metric(parent_metrics, "KNOWN_CLUTTER_EVIDENCE", "precision")
        )
        parent_clutter_recall.append(
            class_metric(parent_metrics, "KNOWN_CLUTTER_EVIDENCE", "recall")
        )
        parent_clutter_f1.append(
            class_metric(parent_metrics, "KNOWN_CLUTTER_EVIDENCE", "f1")
        )
        prefix_clutter_precision.append(
            class_metric(prefix_metrics, "KNOWN_CLUTTER_EVIDENCE", "precision")
        )
        prefix_clutter_recall.append(
            class_metric(prefix_metrics, "KNOWN_CLUTTER_EVIDENCE", "recall")
        )
        prefix_clutter_f1.append(
            class_metric(prefix_metrics, "KNOWN_CLUTTER_EVIDENCE", "f1")
        )
        prefix_target_recall.append(
            class_metric(prefix_metrics, "KNOWN_TARGET_EVIDENCE", "recall")
        )
    return {
        "method": "stratified parent bootstrap; Pending remains a separate non-accepted action",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "confidence_level": 0.95,
        "stratum_support": expected,
        "first_causal_parent_decision": {
            "clutter_precision": _ci(parent_clutter_precision),
            "clutter_recall": _ci(parent_clutter_recall),
            "clutter_f1": _ci(parent_clutter_f1),
        },
        "parent_balanced_eligible_prefix": {
            "clutter_precision": _ci(prefix_clutter_precision),
            "clutter_recall": _ci(prefix_clutter_recall),
            "clutter_f1": _ci(prefix_clutter_f1),
            "known_target_recall": _ci(prefix_target_recall),
        },
    }


def _clutter_trajectories(policy_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in policy_records:
        if str(row.get("label_name")) == "ClutterTarget":
            grouped.setdefault(str(row["parent_sample_id"]), []).append(dict(row))
    if len(grouped) != EXPECTED_VALIDATION_CLUTTER:
        raise ValueError("V25g Clutter trajectory support differs from 9 parents")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["event_index"]))
    return grouped


def _observation_strata(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        rows_by_parent.setdefault(str(row["parent_sample_id"]), []).append(row)
    parent_rows: list[dict[str, Any]] = []
    for decision in decisions:
        parent_id = str(decision["parent_sample_id"])
        rows = rows_by_parent.get(parent_id, [])
        parent_length = max((int(row.get("parent_length", 0)) for row in rows), default=0)
        rd_multiple_rate = (
            sum(bool(row.get("rd_multiple_frames")) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        capture = next(
            (
                str(row.get("capture_name"))
                for row in rows
                if str(row.get("capture_name", ""))
            ),
            "unknown",
        )
        parent_rows.append(
            {
                **dict(decision),
                "parent_length": parent_length,
                "parent_length_bin": (
                    "short_<=8"
                    if parent_length <= 8
                    else "medium_9_16"
                    if parent_length <= 16
                    else "long_>=17"
                ),
                "rd_multiple_frames_rate": rd_multiple_rate,
                "capture_name": capture,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in parent_rows:
        grouped.setdefault(
            f"capture={row['capture_name']}|length={row['parent_length_bin']}",
            [],
        ).append(row)
    return {
        "available_fields_only": True,
        "note": "same-second multiplicity is not inferred from absent record fields; rd_multiple_frames is reported when available",
        "groups": {
            key: {
                "parent_count": len(rows),
                "pending_parent_count": sum(bool(row["policy_pending"]) for row in rows),
                "accepted_parent_count": sum(bool(row["policy_accepted"]) for row in rows),
                "clutter_parent_count": sum(str(row["label_name"]) == "ClutterTarget" for row in rows),
                "known_target_parent_count": sum(str(row["label_name"]) != "ClutterTarget" for row in rows),
                "mean_rd_multiple_frames_rate": float(
                    sum(float(row["rd_multiple_frames_rate"]) for row in rows) / len(rows)
                ),
            }
            for key, rows in grouped.items()
        },
    }


def run(project_root: Path, *, output: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    preparation = _verify_preparation(root)
    source = _source(root)
    config = source["config"]
    config_path = source["paths"]["config"]
    checkpoint_path = source["epoch_paths"][EXPECTED_V25A_BEST_EPOCH]
    checkpoint = _verify_epoch_payload(
        checkpoint_path,
        epoch=EXPECTED_V25A_BEST_EPOCH,
        config=config,
        config_path=config_path,
        readiness_sha256=source["hashes"]["readiness"],
        expected_order_sha256=source["orders"][str(EXPECTED_V25A_BEST_EPOCH)],
    )
    model, _, pairing = build_known_gate_model_for_config(root, config)
    if pairing.get("new_fusion_columns_exactly_zero") is not True:
        raise ValueError("V25g model initialization pairing is not the frozen V25a pair")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    device = known_gate_device(config)
    model.to(device)

    # The feature-cache loader materializes each split in memory.  Evaluate
    # train first, retain only the small calibration projection, and release
    # the train cache before opening validation.  This prevents the two NPZ
    # payloads from overlapping at peak memory.
    train_dataset = build_known_gate_dataset(root, config, split_name="train")
    _harden_dataset_tensor_cache(train_dataset)
    train_evaluation = evaluate_known_gate(
        model,
        train_dataset,
        config=config,
        device=device,
        batch_size=1,
        pin_memory=False,
    )
    train_records = dict(train_evaluation.get("records") or {})
    train_eligible_records = list(train_records.get("eligible_prefix") or ())
    calibration_records = _slim_calibration_records(train_eligible_records)
    train_parent_count = len(train_dataset)
    if train_parent_count != EXPECTED_TRAIN_PARENTS:
        raise ValueError("V25g train parent count differs from the frozen V25a contract")
    del train_records, train_eligible_records, train_evaluation, train_dataset
    _release_memory(model=model)

    validation_dataset = build_known_gate_dataset(root, config, split_name="val")
    _harden_dataset_tensor_cache(validation_dataset)
    if validation_dataset.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 223,
        "KNOWN_CLUTTER_EVIDENCE": EXPECTED_VALIDATION_CLUTTER,
    }:
        raise ValueError("V25g validation support differs from frozen V25a")
    validation_evaluation = evaluate_known_gate(
        model,
        validation_dataset,
        config=config,
        device=device,
        batch_size=1,
        pin_memory=False,
    )
    validation_records = dict(validation_evaluation.get("records") or {})
    validation_counts = {
        "parent": len(list(validation_records.get("parent") or ())),
        "eligible_prefix": len(list(validation_records.get("eligible_prefix") or ())),
        "window": len(list(validation_records.get("window") or ())),
    }
    if (
        validation_counts["parent"] != EXPECTED_VALIDATION_PARENTS
        or validation_counts["eligible_prefix"] != EXPECTED_VALIDATION_ELIGIBLE_PREFIX_RECORDS
        or validation_counts["window"] != EXPECTED_VALIDATION_WINDOW_RECORDS
        or any(value > MAX_VALIDATION_RECORDS for value in validation_counts.values())
    ):
        raise ValueError("V25g validation record counts differ from the frozen V25a contract")
    calibration = select_train_only_threshold(calibration_records)
    selected = calibration.get("selected")
    threshold = (
        float(selected["threshold"])
        if isinstance(selected, Mapping)
        else BASELINE_THRESHOLD
    )
    policy = _policy_metrics(validation_evaluation, threshold)
    baseline_policy = _policy_metrics(validation_evaluation, BASELINE_THRESHOLD)
    validation_raw = _raw_metrics(validation_evaluation)
    policy_records = list(policy["records"]["eligible_prefix"])
    decisions = list(policy["records"]["first_causal_decision"])
    bootstrap = _bootstrap(
        parent_decisions=decisions,
        eligible_policy=policy_records,
        seed=BOOTSTRAP_SEED,
    )
    parent_metrics = policy["first_causal_decision"]
    clutter_precision = class_metric(parent_metrics, "KNOWN_CLUTTER_EVIDENCE", "precision")
    clutter_recall = class_metric(parent_metrics, "KNOWN_CLUTTER_EVIDENCE", "recall")
    target_recall = class_metric(parent_metrics, "KNOWN_TARGET_EVIDENCE", "recall")
    baseline_target_recall = class_metric(
        baseline_policy["first_causal_decision"],
        "KNOWN_TARGET_EVIDENCE",
        "recall",
    )
    prefix_target_recall = class_metric(
        policy["parent_balanced"], "KNOWN_TARGET_EVIDENCE", "recall"
    )
    raw_prefix_target_recall = class_metric(
        validation_raw["eligible_prefix_parent_balanced"],
        "KNOWN_TARGET_EVIDENCE",
        "recall",
    )
    validation_gate = {
        "clutter_support_is_9": len(
            [row for row in decisions if str(row["label_name"]) == "ClutterTarget"]
        )
        == EXPECTED_VALIDATION_CLUTTER,
        "clutter_parent_recall_at_least_8_of_9": clutter_recall + 1.0e-12 >= 8.0 / 9.0,
        "clutter_parent_precision_at_least_0_90": clutter_precision + 1.0e-12 >= 0.90,
        "first_decision_known_target_recall_not_below_baseline_policy": target_recall + 1.0e-12 >= baseline_target_recall,
        "eligible_prefix_known_target_recall_not_below_v25a_local_argmax": prefix_target_recall + 1.0e-12 >= raw_prefix_target_recall,
        "pending_is_not_counted_as_unknown": True,
    }
    clutter_trajectories = _clutter_trajectories(policy_records)
    observation_strata = _observation_strata(policy_records, decisions)
    baseline_policy_report = {
        key: value
        for key, value in baseline_policy.items()
        if key != "records"
    }
    selected_policy_report = {
        key: value
        for key, value in policy.items()
        if key != "records"
    }
    # Do not retain duplicate event-level policy records in the report.  The
    # required per-parent Clutter trajectories remain below; metrics are enough
    # for the full baseline/selected comparison.
    del policy["records"], baseline_policy["records"]
    _clear_dataset_tensor_cache(validation_dataset)
    del validation_records, validation_evaluation, validation_dataset
    _release_memory(model=model)
    formal_adoption_eligible = bool(all(validation_gate.values()) and bool(calibration.get("is_oof")))
    report = {
        "status": "completed_development_policy_diagnostic",
        "schema_version": V25G_SCHEMA,
        "policy_schema": calibration.get("selected", {}).get("policy_schema", None)
        if isinstance(calibration.get("selected"), Mapping)
        else "RADAR-LN3-V25g-causal-accept-pending-policy-v1",
        "experiment": "V25g frozen V25a epoch14 causal ACCEPT/PENDING policy diagnostic",
        "scope": "KnownGate development only; no training, Unknown, test, OOF, L3, cascade, rejection, stable or P95",
        "single_factor": {
            "changed_component": "prefix local argmax action interpretation only",
            "training_step_count": 0,
            "optimizer_step_count": 0,
            "frozen_checkpoint_epoch": EXPECTED_V25A_BEST_EPOCH,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "source_v25a": {
            "config_path": str(config_path),
            "config_sha256": source["hashes"]["config"],
            "config_identity_sha256": known_gate_config_identity_sha256(config),
            "summary_path": str(source["paths"]["summary"]),
            "summary_sha256": source["hashes"]["summary"],
            "legacy_best_epoch": EXPECTED_V25A_BEST_EPOCH,
            "validation_parent_count": EXPECTED_VALIDATION_PARENTS,
            "validation_clutter_parent_support": EXPECTED_VALIDATION_CLUTTER,
        },
        "calibration": {
            "protocol": calibration["protocol"],
            "is_oof": calibration["is_oof"],
            "oof_status": "not_executed",
            "train_parent_count": train_parent_count,
            "train_eligible_prefix_event_count": len(calibration_records),
            "selected_threshold": threshold,
            "selection_status": (
                "selected_train_only_candidate"
                if selected is not None
                else "no_feasible_train_only_candidate_fallback_to_0_5"
            ),
            "selection": calibration,
            "validation_or_test_used_for_threshold_fit": False,
            "formal_adoption_requires_exact_oof": True,
        },
        "validation": {
            "raw_v25a_local_argmax": validation_raw,
            "baseline_policy_threshold_0_5": baseline_policy_report,
            "selected_policy": selected_policy_report,
            "validation_gate": validation_gate,
            "formal_adoption_eligible": formal_adoption_eligible,
            "parent_balanced_prefix_target_recall_delta_vs_local_argmax": float(
                prefix_target_recall - raw_prefix_target_recall
            ),
        },
        "paired_parent_bootstrap": bootstrap,
        "clutter_parent_trajectories": clutter_trajectories,
        "observation_strata": observation_strata,
        "resource_safety": {
            "new_feature_cache_created": False,
            "checkpoint_mutated": False,
            "sequential_split_loading": True,
            "rd_tensor_cache_max_items": MAX_RD_TENSOR_CACHE_ITEMS,
            "pin_memory": False,
            "evaluation_batch_size": 1,
            "max_threshold_candidates": V25G_MAX_THRESHOLD_CANDIDATES,
            "max_train_calibration_records": MAX_TRAIN_CALIBRATION_RECORDS,
            "max_validation_records": MAX_VALIDATION_RECORDS,
            "train_calibration_evaluation_pass_count": 1,
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        "decision": (
            "not_formal_until_exact_parent_grouped_oof_calibration"
            if not formal_adoption_eligible
            else "development_gate_passed_but_held_out_confirmation_still_required"
        ),
        "conclusion": (
            "The frozen action layer satisfies the local validation gates, but this V25g result is not formal because threshold calibration is train-only rather than exact OOF."
            if formal_adoption_eligible
            else "The frozen action layer does not satisfy every local validation gate, or exact OOF calibration is unavailable; do not promote it."
        ),
        "training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "Unknown": "not_used",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    destination = root / output if output is not None else root / REPORT
    _write_json(destination, report)
    return {
        "status": report["status"],
        "report": str(destination),
        "selected_threshold": threshold,
        "calibration_is_oof": False,
        "formal_adoption_eligible": formal_adoption_eligible,
        "training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V25g frozen causal policy diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()