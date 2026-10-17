"""Scan V25a checkpoints with V25f causal-prefix utility selectors."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v25f import (
    CONTRACT_SCHEMA,
    EXPECTED_EPOCHS,
    EXPERIMENT_CONTRACT,
    MAX_V25F_CHECKPOINT_BYTES,
    MAX_V25F_CHECKPOINT_MEMBERS,
    MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES,
    MAX_V25F_JSON_BYTES,
    READINESS,
    REPORT,
    _source,
    validate_v25f_checkpoint_file,
)
from radar_ln3.known_gate import (
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    known_gate_device,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v25f_selection import (
    V25F_SELECTOR_SCHEMA,
    assess_checkpoint,
    metric_delta,
    metrics_from_records,
    reference_from_records,
    select_best,
)
from radar_ln3.known_gate import sha256_file


REPORT_SCHEMA = (
    "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-"
    "utility-selection-report-v1"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 25051
BASE_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        existing_size = path.stat().st_size
        if existing_size > MAX_V25F_JSON_BYTES:
            raise ValueError(
                "existing V25f report exceeds the fail-closed JSON size limit: "
                f"{path}={existing_size} > {MAX_V25F_JSON_BYTES} bytes"
            )
        if path.read_text(encoding="utf-8") == text:
            return
    if len(text.encode("utf-8")) > MAX_V25F_JSON_BYTES:
        raise ValueError(
            "new V25f report exceeds the fail-closed JSON size limit: "
            f"{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        # Keep an interrupted report write from becoming a false complete
        # artifact or consuming space across repeated scans.
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V25f JSON source is not a file: {path}")
    size = path.stat().st_size
    if size > MAX_V25F_JSON_BYTES:
        raise ValueError(
            "V25f JSON source exceeds the fail-closed size limit: "
            f"{path}={size} > {MAX_V25F_JSON_BYTES} bytes"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"V25f JSON source must contain an object: {path}")
    return dict(payload)


def _verify_preparation(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25f contract/readiness is incomplete")
    contract = _load_json(contract_path)
    readiness = _load_json(readiness_path)
    source_binding = dict(readiness.get("source_v25a") or {})
    experiment_binding = dict(readiness.get("experiment_contract") or {})
    checks = dict(readiness.get("checks") or {})
    product_safe = dict(dict(contract.get("threshold_policy") or {}).get("product_safe") or {})
    diagnostic = dict(dict(contract.get("threshold_policy") or {}).get("diagnostic_frontier") or {})
    prefix_semantics = dict(contract.get("prefix_action_semantics") or {})
    evaluation = dict(contract.get("evaluation") or {})
    single_factor = dict(contract.get("single_factor") or {})
    expected_readiness_checks = {
        "source_v25a_training_completed",
        "source_v25a_has_all_twenty_epoch_checkpoints",
        "source_v25a_parent_orders_are_sealed",
        "source_v25a_test_not_executed",
        "source_v25a_unknown_training_exposure_is_zero",
        "no_v25f_training_or_optimizer_steps",
        "product_safe_and_diagnostic_thresholds_are_distinct",
        "fixed_two_pp_gate_is_not_used",
    }
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("selector_schema") != V25F_SELECTOR_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("selector_schema") != V25F_SELECTOR_SCHEMA
        or experiment_binding.get("sha256") != sha256_file(contract_path)
        or source_binding.get("config_sha256") != source["hashes"]["config"]
        or source_binding.get("readiness_sha256") != source["hashes"]["readiness"]
        or source_binding.get("summary_sha256") != source["hashes"]["summary"]
        or readiness.get("training") != "not_executed"
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or set(checks) != expected_readiness_checks
        or not all(value is True for value in checks.values())
        or single_factor.get("changed_component") != "checkpoint selection rule only"
        or int(single_factor.get("training_step_count", -1)) != 0
        or int(single_factor.get("optimizer_step_count", -1)) != 0
        or product_safe.get("clutter_parent_support") != 9
        or product_safe.get("clutter_true_positive_minimum") != 8
        or abs(float(product_safe.get("clutter_recall_minimum", -1.0)) - (8.0 / 9.0)) > 1.0e-12
        or abs(float(product_safe.get("clutter_precision_minimum", -1.0)) - 0.90) > 1.0e-12
        or product_safe.get("known_target_parent_recall")
        != "must_not_decrease_vs_V25a_legacy"
        or product_safe.get("known_target_prefix_recall")
        != "must_not_decrease_vs_V25a_legacy"
        or diagnostic.get("maximum_known_target_parent_loss") != 1
        or diagnostic.get("purpose")
        != "locate the selection tradeoff only; never an adoption gate"
        or prefix_semantics.get("minimum_evidence_events") != 3
        or prefix_semantics.get("events_before_minimum") != "PENDING_ABSTAIN"
        or prefix_semantics.get("eligible_prefix_excludes_final_event") is not True
        or prefix_semantics.get("prefix_loss_target") != "full parent-final KnownGate label"
        or prefix_semantics.get("local_argmax_is_product_action") is not False
        or prefix_semantics.get("unknown_probability_output") is not False
        or evaluation.get("validation_parent_count") != 232
        or evaluation.get("validation_clutter_parent_support") != 9
        or contract.get("training") != "not_executed_by_contract"
        or contract.get("deployment_eligible") is not False
    ):
        raise ValueError("V25f preparation artifacts are stale or outside selector-only scope")
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
    }


def _class_row(metrics: Mapping[str, Any], class_name: str) -> dict[str, Any]:
    for row in list(metrics.get("per_class") or ()):
        if str(row.get("class")) == class_name:
            return dict(row)
    raise ValueError(f"V25f metrics lack class {class_name}")


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("V25f bootstrap values are empty or non-finite")
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _records_by_parent(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        parent_id = str(row["parent_sample_id"])
        if parent_id in result:
            raise ValueError("V25f parent record keys are repeated")
        result[parent_id] = dict(row)
    return result


def _stratified_parent_bootstrap(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    parent = _records_by_parent(records["parent"])
    prefixes: dict[str, list[dict[str, Any]]] = {}
    for row in records["eligible_prefix"]:
        prefixes.setdefault(str(row["parent_sample_id"]), []).append(dict(row))
    strata = {
        label: [
            parent_id
            for parent_id, row in parent.items()
            if str(row["label_name"]) == label
        ]
        for label in BASE_LABELS
    }
    expected = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }
    if {key: len(value) for key, value in strata.items()} != expected:
        raise ValueError("V25f bootstrap parent strata differ from frozen support")
    rng = np.random.default_rng(seed)
    parent_clutter_precision: list[float] = []
    parent_clutter_recall: list[float] = []
    parent_clutter_f1: list[float] = []
    parent_macro_f1: list[float] = []
    prefix_clutter_precision: list[float] = []
    prefix_clutter_recall: list[float] = []
    prefix_clutter_f1: list[float] = []
    prefix_known_target_recall: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            str(parent_id)
            for label in BASE_LABELS
            for parent_id in rng.choice(
                np.asarray(strata[label], dtype=object),
                size=expected[label],
                replace=True,
            )
        ]
        sampled_parent: list[dict[str, Any]] = []
        sampled_prefix: list[dict[str, Any]] = []
        for index, parent_id in enumerate(sampled_ids):
            synthetic_id = f"{parent_id}#v25f_bootstrap_{index}"
            parent_row = dict(parent[parent_id])
            parent_row["parent_sample_id"] = synthetic_id
            sampled_parent.append(parent_row)
            for prefix_row in prefixes.get(parent_id, ()):
                cloned = dict(prefix_row)
                cloned["parent_sample_id"] = synthetic_id
                sampled_prefix.append(cloned)
        parent_metrics = summarize_known_gate_records(
            sampled_parent, scope="v25f_parent_bootstrap"
        )
        prefix_metrics = summarize_parent_balanced_eligible_prefix_records(
            sampled_prefix, scope="v25f_parent_balanced_prefix_bootstrap"
        )
        parent_clutter = _class_row(parent_metrics, "KNOWN_CLUTTER_EVIDENCE")
        prefix_clutter = _class_row(prefix_metrics, "KNOWN_CLUTTER_EVIDENCE")
        prefix_target = _class_row(prefix_metrics, "KNOWN_TARGET_EVIDENCE")
        parent_clutter_precision.append(float(parent_clutter["precision"]))
        parent_clutter_recall.append(float(parent_clutter["recall"]))
        parent_clutter_f1.append(float(parent_clutter["f1"]))
        parent_macro_f1.append(float(parent_metrics["macro_f1"]))
        prefix_clutter_precision.append(float(prefix_clutter["precision"]))
        prefix_clutter_recall.append(float(prefix_clutter["recall"]))
        prefix_clutter_f1.append(float(prefix_clutter["f1"]))
        prefix_known_target_recall.append(float(prefix_target["recall"]))
    return {
        "method": "stratified parent bootstrap; duplicated parents retain all causal prefixes with 1/n parent weighting",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "confidence_level": 0.95,
        "stratum_support": expected,
        "parent_final": {
            "clutter_precision": _ci(parent_clutter_precision),
            "clutter_recall": _ci(parent_clutter_recall),
            "clutter_f1": _ci(parent_clutter_f1),
            "macro_f1": _ci(parent_macro_f1),
        },
        "parent_balanced_eligible_prefix": {
            "clutter_precision": _ci(prefix_clutter_precision),
            "clutter_recall": _ci(prefix_clutter_recall),
            "clutter_f1": _ci(prefix_clutter_f1),
            "known_target_recall": _ci(prefix_known_target_recall),
        },
    }


def _directional_changes(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    left = {
        (str(row["parent_sample_id"]), int(row["event_index"])): dict(row)
        for row in before
    }
    right = {
        (str(row["parent_sample_id"]), int(row["event_index"])): dict(row)
        for row in after
    }
    if set(left) != set(right):
        raise ValueError("V25f paired record keys differ")
    result = {
        "prediction_changed": 0,
        "known_to_clutter": 0,
        "clutter_to_known": 0,
        "rescued": 0,
        "harmed": 0,
    }
    for key in left:
        old = left[key]
        new = right[key]
        if int(old["pred_id"]) == int(new["pred_id"]):
            continue
        result["prediction_changed"] += 1
        if int(old["pred_id"]) == 0 and int(new["pred_id"]) == 1:
            result["known_to_clutter"] += 1
        if int(old["pred_id"]) == 1 and int(new["pred_id"]) == 0:
            result["clutter_to_known"] += 1
        old_correct = int(old["pred_id"]) == int(old["true_id"])
        new_correct = int(new["pred_id"]) == int(new["true_id"])
        result["rescued"] += int(not old_correct and new_correct)
        result["harmed"] += int(old_correct and not new_correct)
    return result


def _clutter_trajectories(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    parents = {
        str(row["parent_sample_id"]): dict(row)
        for row in records["parent"]
        if str(row["label_name"]) == "ClutterTarget"
    }
    result: dict[str, Any] = {}
    for parent_id, parent_row in parents.items():
        result[parent_id] = {
            "parent_final": parent_row,
            "eligible_prefix_actions": [],
        }
    for row in records["eligible_prefix"]:
        parent_id = str(row["parent_sample_id"])
        if parent_id in result:
            result[parent_id]["eligible_prefix_actions"].append(dict(row))
    for payload in result.values():
        payload["eligible_prefix_actions"].sort(key=lambda row: int(row["event_index"]))
    if len(result) != 9:
        raise ValueError("V25f Clutter trajectory support differs from 9 parents")
    return result


def _action_alignment(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    records = dict(evaluation.get("records") or {})
    window = list(records.get("window") or ())
    eligible = list(records.get("eligible_prefix") or ())
    pending = [row for row in window if not bool(row.get("product_action_eligible"))]
    if any(bool(row.get("product_action_eligible")) for row in pending):
        raise ValueError("V25f pending/action eligibility records are inconsistent")
    return {
        "window_event_count": len(window),
        "pending_abstain_event_count": len(pending),
        "eligible_prefix_event_count": len(eligible),
        "final_parent_action_count": len(records.get("parent") or ()),
        "minimum_evidence_events": int(
            dict(evaluation.get("prefix_action_contract") or {}).get(
                "minimum_evidence_events", -1
            )
        ),
        "events_before_minimum_are_pending_abstain": all(
            int(row["event_index"]) < 2 for row in pending
        ),
        "prefix_loss_target_is_parent_final_label": True,
        "local_argmax_is_product_action": False,
        "unknown_probability_output": bool(evaluation.get("unknown_probability_output")),
    }


def _verify_epoch_payload(
    path: Path,
    *,
    epoch: int,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    expected_order_sha256: str,
) -> Mapping[str, Any]:
    validate_v25f_checkpoint_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V25f epoch checkpoint payload must be a mapping: {path}")
    metrics = dict(payload.get("metrics") or {})
    if (
        payload.get("checkpoint_schema") != config.train.checkpoint_schema
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != config.variant
        or int(payload.get("epoch", -1)) != epoch
        or payload.get("config_identity_sha256")
        != known_gate_config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
        or int(metrics.get("epoch", -1)) != epoch
        or metrics.get("parent_order_sha256") != expected_order_sha256
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError(f"V25f epoch checkpoint is stale or invalid: {path}")
    return payload


def _selected_payload(
    candidate: Mapping[str, Any] | None,
    *,
    records: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    legacy_records: Mapping[str, Sequence[Mapping[str, Any]]],
    legacy_metrics: Mapping[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    epoch = int(candidate["epoch"])
    selected_records = records[epoch]
    metrics = dict(candidate["metrics"])
    return {
        "mode": mode,
        "epoch": epoch,
        "metrics": metrics,
        "delta_vs_v25a_legacy": metric_delta(
            legacy_metrics["selector_metrics"], metrics
        ),
        "paired_parent_bootstrap": _stratified_parent_bootstrap(
            selected_records, seed=BOOTSTRAP_SEED + epoch
        ),
        "directional_changes_vs_v25a_legacy": {
            scope: _directional_changes(
                legacy_records[scope], selected_records[scope]
            )
            for scope in ("parent", "eligible_prefix", "window")
        },
        "clutter_parent_trajectories": _clutter_trajectories(selected_records),
    }


def run(project_root: Path, *, output: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    source = _source(root)
    preparation = _verify_preparation(root, source)
    config = source["config"]
    config_path = source["paths"]["config"]
    readiness_sha256 = source["hashes"]["readiness"]
    detail_records = {
        scope: [
            dict(row)
            for row in list(dict(source["detail"].get("records") or {}).get(scope) or ())
        ]
        for scope in ("parent", "eligible_prefix", "window")
    }
    legacy_metrics = metrics_from_records(detail_records)
    legacy_reference = reference_from_records(
        detail_records,
        epoch=int(source["summary"]["best_epoch"]),
        source={
            "variant": config.variant,
            "summary_path": str(source["paths"]["summary"]),
            "summary_sha256": source["hashes"]["summary"],
            "legacy_best_epoch": int(source["summary"]["best_epoch"]),
            "sample_detail_path": str(source["paths"]["sample_detail"]),
            "sample_detail_sha256": source["hashes"]["sample_detail"],
        },
    )
    legacy_metrics_for_delta = {
        "selector_metrics": {
            "parent_final": legacy_metrics["parent_metrics"],
            "eligible_prefix_event_pooled": legacy_metrics["eligible_prefix_metrics"],
            "eligible_prefix_parent_balanced": legacy_metrics[
                "parent_balanced_eligible_prefix_metrics"
            ],
            "window_event_pooled": legacy_metrics["window_metrics"],
        }
    }
    validation = build_known_gate_dataset(root, config, split_name="val")
    if validation.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 223,
        "KNOWN_CLUTTER_EVIDENCE": 9,
    }:
        raise ValueError("V25f validation support differs from V25a")
    device = known_gate_device(config)
    model, _, pairing = build_known_gate_model_for_config(root, config)
    if pairing.get("new_fusion_columns_exactly_zero") is not True:
        raise ValueError("V25f model initialization pairing is not frozen V25a")
    model.to(device)
    candidates: list[dict[str, Any]] = []
    action_alignment_reference: dict[str, Any] | None = None
    action_alignment_epoch_count = 0
    for epoch in EXPECTED_EPOCHS:
        checkpoint_path = source["epoch_paths"][epoch]
        checkpoint = _verify_epoch_payload(
            checkpoint_path,
            epoch=epoch,
            config=config,
            config_path=config_path,
            readiness_sha256=readiness_sha256,
            expected_order_sha256=source["orders"][str(epoch)],
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        evaluation = evaluate_known_gate(model, validation, config=config, device=device)
        alignment = _action_alignment(evaluation)
        if action_alignment_reference is None:
            action_alignment_reference = alignment
        elif alignment != action_alignment_reference:
            raise ValueError("V25f epoch action alignment changed unexpectedly")
        if alignment["minimum_evidence_events"] != 3:
            raise ValueError("V25f prefix action contract changed during epoch scan")
        records = {
            scope: [dict(row) for row in list(evaluation["records"][scope])]
            for scope in ("parent", "eligible_prefix", "window")
        }
        metrics = metrics_from_records(records)
        assessment = assess_checkpoint(
            metrics,
            reference=legacy_reference,
            epoch=epoch,
            prefix_action_contract=evaluation["prefix_action_contract"],
        )
        assessment["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }
        assessment["action_alignment"] = alignment
        candidates.append(assessment)
        action_alignment_epoch_count += 1
        # The first pass deliberately retains only scalar summaries.  Keeping
        # all 4,452 event records for 20 epochs caused avoidable multi-GB
        # process growth during this read-only selector scan.
        del evaluation, records, metrics, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if action_alignment_reference is None or action_alignment_epoch_count != 20:
        raise ValueError("V25f epoch action alignment is not invariant")
    strict_selected = select_best(candidates, mode="strict_product_safe")
    diagnostic_selected = select_best(candidates, mode="diagnostic_utility")
    selected_epochs = {
        int(candidate["epoch"])
        for candidate in (strict_selected, diagnostic_selected)
        if candidate is not None
    }
    selected_epoch_records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for epoch in sorted(selected_epochs):
        checkpoint_path = source["epoch_paths"][epoch]
        checkpoint = _verify_epoch_payload(
            checkpoint_path,
            epoch=epoch,
            config=config,
            config_path=config_path,
            readiness_sha256=readiness_sha256,
            expected_order_sha256=source["orders"][str(epoch)],
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        selected_evaluation = evaluate_known_gate(
            model, validation, config=config, device=device
        )
        selected_epoch_records[epoch] = {
            scope: [
                dict(row)
                for row in list(selected_evaluation["records"][scope])
            ]
            for scope in ("parent", "eligible_prefix", "window")
        }
        del selected_evaluation, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_strict = _selected_payload(
        strict_selected,
        records=selected_epoch_records,
        legacy_records=detail_records,
        legacy_metrics=legacy_metrics_for_delta,
        mode="strict_product_safe",
    )
    selected_diagnostic = _selected_payload(
        diagnostic_selected,
        records=selected_epoch_records,
        legacy_records=detail_records,
        legacy_metrics=legacy_metrics_for_delta,
        mode="diagnostic_utility",
    )
    strict_delta = (
        selected_strict["delta_vs_v25a_legacy"] if selected_strict is not None else None
    )
    diagnostic_delta = (
        selected_diagnostic["delta_vs_v25a_legacy"]
        if selected_diagnostic is not None
        else None
    )
    if strict_selected is None:
        strict_conclusion = (
            "No product-safe V25f epoch exists under the pre-registered Clutter "
            "8/9 and precision 0.90 floors plus no KnownTarget parent/prefix recall loss."
        )
    elif int(strict_selected["epoch"]) == int(source["summary"]["best_epoch"]):
        strict_conclusion = (
            "Strict product-safe selection returns the V25a legacy epoch; "
            "checkpoint selection alone does not improve causal-prefix actions."
        )
    else:
        strict_conclusion = (
            "Strict product-safe selection found a different feasible epoch; "
            "this remains validation-only until parent-grouped confirmation."
        )
    report = {
        "status": "completed_development_selector_only",
        "schema_version": REPORT_SCHEMA,
        "selector_schema": V25F_SELECTOR_SCHEMA,
        "experiment": "V25f causal-prefix checkpoint utility selection on frozen V25a",
        "scope": "KnownGate development only; no retraining, Unknown, OOF, test, L3, cascade, rejection, stable or P95",
        "single_factor": {
            "changed_component": "checkpoint selection rule only",
            "training_step_count": 0,
            "optimizer_step_count": 0,
            "frozen_source": "V25a self-developed causal-course 20-epoch trajectory",
        },
        "source_v25a": {
            "config_path": str(config_path),
            "config_sha256": source["hashes"]["config"],
            "config_identity_sha256": known_gate_config_identity_sha256(config),
            "readiness_path": str(source["paths"]["readiness"]),
            "readiness_sha256": source["hashes"]["readiness"],
            "training_summary_path": str(source["paths"]["summary"]),
            "training_summary_sha256": source["hashes"]["summary"],
            "legacy_best_epoch": int(source["summary"]["best_epoch"]),
            "epoch_checkpoint_count": 20,
            "all_parent_orders_reproduced": True,
        },
        "resource_safety": {
            "source_json_max_bytes": MAX_V25F_JSON_BYTES,
            "checkpoint_max_bytes": MAX_V25F_CHECKPOINT_BYTES,
            "checkpoint_max_declared_uncompressed_bytes": MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES,
            "checkpoint_max_zip_members": MAX_V25F_CHECKPOINT_MEMBERS,
            "checkpoint_validation_before_torch_load": True,
            "epoch_records_retained_during_first_pass": False,
            "new_feature_cache_created": False,
        },
        "preparation": preparation,
        "threshold_policy": {
            "strict_product_safe": {
                "clutter_parent_recall_minimum": 8 / 9,
                "clutter_parent_precision_minimum": 0.90,
                "known_target_parent_recall_loss_allowed": 0,
                "known_target_prefix_recall_loss_allowed": 0,
            },
            "diagnostic_utility_frontier": {
                "known_target_parent_loss_allowed": 1,
                "not_an_adoption_gate": True,
            },
            "fixed_two_pp_gate_used": False,
        },
        "prefix_supervision_semantic_audit": {
            "prefix_loss_target": "full parent-final KnownGate label",
            "minimum_evidence_events": 3,
            "events_before_minimum": "PENDING_ABSTAIN",
            "eligible_prefix_excludes_final_event": True,
            "local_argmax_is_product_action": False,
            "unknown_probability_output": False,
            "action_alignment": action_alignment_reference,
            "invariant_across_epoch_count": action_alignment_epoch_count,
        },
        "legacy_v25a_reference": {
            "epoch": int(source["summary"]["best_epoch"]),
            "metrics": legacy_metrics_for_delta["selector_metrics"],
        },
        "epoch_candidates": candidates,
        "strict_product_safe_feasible_epoch_count": sum(
            bool(candidate["strict_product_safe_feasible"]) for candidate in candidates
        ),
        "diagnostic_utility_feasible_epoch_count": sum(
            bool(candidate["diagnostic_utility_feasible"]) for candidate in candidates
        ),
        "strict_product_safe_selection": selected_strict,
        "diagnostic_utility_selection": selected_diagnostic,
        "strict_product_safe_delta": strict_delta,
        "diagnostic_utility_delta": diagnostic_delta,
        "conclusion": strict_conclusion,
        "diagnostic_interpretation": (
            "A diagnostic-only epoch may expose a prefix/final tradeoff, but cannot be promoted without a new pre-registered product policy and held-out confirmation."
            if diagnostic_selected is not None
            else "No diagnostic frontier epoch satisfies the relaxed one-parent-loss bound."
        ),
        "validation_used_for_epoch_scan": True,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    destination = (root / output) if output is not None else root / REPORT
    _write_json(destination, report)
    return {
        "status": report["status"],
        "report": str(destination),
        "strict_product_safe_feasible_epoch_count": report[
            "strict_product_safe_feasible_epoch_count"
        ],
        "diagnostic_utility_feasible_epoch_count": report[
            "diagnostic_utility_feasible_epoch_count"
        ],
        "strict_selected_epoch": (
            None if strict_selected is None else int(strict_selected["epoch"])
        ),
        "diagnostic_selected_epoch": (
            None if diagnostic_selected is None else int(diagnostic_selected["epoch"])
        ),
        "training": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V25f frozen checkpoint utility scan.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()