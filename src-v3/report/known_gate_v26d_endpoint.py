"""Evaluate V26d's fixed epoch-20 L1 endpoint against the V26c epoch-14 OOF.

V26d is intentionally an inference-only diagnostic.  It materializes one
epoch-20 L1 held-out prediction per existing V26c fold, reuses the sealed
V26c L2 scores, and compares three views:

* V26c epoch-14 control with its original cross-fitted thresholds;
* epoch-20 scores under those same thresholds (pure endpoint sensitivity);
* epoch-20 scores with the same, freshly cross-fitted threshold protocol.

No outer-fold label is used to choose a checkpoint or a threshold.  Unknown,
validation, and test payloads remain inaccessible.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v26d_endpoint import require_prepared
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    build_known_gate_model_for_config,
    build_known_gate_oof_dataset,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    known_gate_device,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v26c_oof import (
    FOLD_COUNT,
    L1_CHECKPOINT_SCHEMA,
    L1_EPOCH,
    MINIMUM_EVIDENCE_EVENTS,
    fold_record,
    l1_checkpoint_path,
    l1_held_out_feature_cache_path,
    l1_last_checkpoint_path,
    l1_prediction_path,
    l1_training_summary_path,
    sequence_sha256,
)
from radar_ln3.known_gate_v26d_endpoint import (
    CONTROL_EPOCH,
    CONTRACT,
    PARENT_DETAIL_CSV,
    READINESS,
    REPORT,
    REPORT_SCHEMA,
    TREATMENT_EPOCH,
    TREATMENT_PREDICTION_SCHEMA,
    relative_path,
    sha256_file,
    treatment_prediction_path,
    write_new_or_identical,
)
from report.known_gate_v26c_oof import (
    DBB_LABELS,
    LABELS,
    _actions as _v26c_actions,
    _annotate_provenance,
    _assignment_by_parent,
    _fit_thresholds,
    _joint_rows,
    _metrics,
    _prediction as _v26c_prediction,
)
from train.known_gate_v21 import (
    _state_dict_sha256,
)
from train.known_gate_v26c_oof import (
    _add_fold_fields,
    _dataset_scope,
    _load_checkpoint,
    _record_parent_ids,
    _runtime_l1_config,
    _source_protocols,
    _require_prepared as _v26c_require_prepared,
)


MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_PREDICTION_BYTES = 12 * 1024 * 1024
MAX_DETAIL_BYTES = 2 * 1024 * 1024
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 26064
ACTION_NAMES = (*DBB_LABELS, "ClutterTarget", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
RAW_SCOPES = ("parent", "eligible_prefix", "window")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26d expected JSON object: {path}")
    return value


def _json_text(payload: Mapping[str, Any], *, limit: int, description: str) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > limit:
        raise MemoryError(f"V26d {description} exceeds its bounded size")
    return text


def _write_text_once(path: Path, text: str, *, description: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26d {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        if temporary.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to replace a different interrupted V26d {description}: {temporary}"
            )
    else:
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def _write_report(root: Path, payload: Mapping[str, Any]) -> str:
    return _write_text_once(
        root / REPORT,
        _json_text(payload, limit=MAX_REPORT_BYTES, description="report"),
        description="report",
    )


def _write_prediction(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_text_once(
        path,
        _json_text(payload, limit=MAX_PREDICTION_BYTES, description="treatment prediction"),
        description="treatment prediction",
    )


def _checkpoint_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "config_identity_sha256",
        "config_file_sha256",
        "fold_contract_sha256",
        "readiness_sha256",
        "source_config_sha256",
        "initial_model_state_sha256",
    )
    return {field: payload.get(field) for field in fields}


def _validated_probability(value: Any, *, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"V26d probability is non-finite or out of range: {field}")
    return result


def _validate_l1_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fold_id: int,
    expected_parent_ids: Sequence[str],
    source_name: str,
) -> None:
    """Validate the complete L1 record domain before any metric is computed."""

    allowed_labels = {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"}
    expected_ids = {str(value) for value in expected_parent_ids}
    parent_lengths: dict[str, int] = {}
    scope_parent_ids: dict[str, set[str]] = {}
    for scope in RAW_SCOPES:
        rows = list(records.get(scope) or ())
        if not rows and scope != "eligible_prefix":
            raise ValueError(f"V26d {source_name} has no {scope} records: fold={fold_id}")
        keys = [
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"V26d {source_name} repeats a {scope} record: fold={fold_id}")
        if scope == "parent":
            parent_ids = {key[0] for key in keys}
            if parent_ids != expected_ids or len(parent_ids) != len(rows):
                raise ValueError(f"V26d {source_name} parent coverage differs: fold={fold_id}")
        elif not {key[0] for key in keys} <= expected_ids:
            raise ValueError(f"V26d {source_name} {scope} contains an unexpected parent: fold={fold_id}")
        scope_parent_ids[scope] = {key[0] for key in keys}
        for row in rows:
            if int(row.get("fold_id", -1)) != fold_id:
                raise ValueError(f"V26d {source_name} fold binding is invalid: fold={fold_id}")
            if str(row.get("prediction_scope", "")) != scope:
                raise ValueError(f"V26d {source_name} prediction scope is invalid: fold={fold_id}")
            label = str(row.get("label_name", ""))
            if label not in allowed_labels:
                raise ValueError(f"V26d {source_name} has an invalid label: fold={fold_id}")
            target = _validated_probability(
                row.get("known_target_probability"),
                field=f"{source_name}.known_target_probability",
            )
            clutter = _validated_probability(
                row.get("known_clutter_probability"),
                field=f"{source_name}.known_clutter_probability",
            )
            if abs(target + clutter - 1.0) > 1.0e-5:
                raise ValueError(
                    f"V26d {source_name} L1 probabilities do not sum to one: fold={fold_id}"
                )
            true_id = int(row.get("true_id", -1))
            expected_true_id = 1 if label == "ClutterTarget" else 0
            if true_id != expected_true_id:
                raise ValueError(f"V26d {source_name} true label is inconsistent: fold={fold_id}")
            pred_id = int(row.get("pred_id", -1))
            expected_pred_id = 0 if target >= clutter else 1
            if pred_id != expected_pred_id:
                raise ValueError(f"V26d {source_name} pred_id is not the L1 argmax: fold={fold_id}")
            confidence = _validated_probability(
                row.get("confidence"),
                field=f"{source_name}.confidence",
            )
            if abs(confidence - max(target, clutter)) > 1.0e-6:
                raise ValueError(f"V26d {source_name} confidence is inconsistent: fold={fold_id}")
            event_index = int(row.get("event_index", -1))
            window_length = int(row.get("window_length", 0))
            parent_length = int(row.get("parent_length", 0))
            if (
                window_length < 1
                or parent_length < window_length
                or event_index != window_length - 1
                or event_index < 0
            ):
                raise ValueError(f"V26d {source_name} event/window bounds are invalid: fold={fold_id}")
            if not isinstance(row.get("product_action_eligible"), bool):
                raise ValueError(f"V26d {source_name} eligibility type is invalid: fold={fold_id}")
            eligible = bool(row.get("product_action_eligible"))
            if eligible != (window_length >= MINIMUM_EVIDENCE_EVENTS):
                raise ValueError(f"V26d {source_name} eligibility is inconsistent: fold={fold_id}")
            if scope == "parent" and window_length != parent_length:
                raise ValueError(f"V26d {source_name} parent record is not final: fold={fold_id}")
            if scope == "eligible_prefix" and (
                not eligible or window_length >= parent_length
            ):
                raise ValueError(f"V26d {source_name} eligible-prefix bounds are invalid: fold={fold_id}")
            parent_id = str(row["parent_sample_id"])
            prior_length = parent_lengths.get(parent_id)
            if prior_length is not None and prior_length != parent_length:
                raise ValueError(f"V26d {source_name} parent length changes across scopes: fold={fold_id}")
            parent_lengths[parent_id] = parent_length

    if scope_parent_ids.get("parent", set()) != expected_ids:
        raise ValueError(f"V26d {source_name} parent scope coverage is incomplete: fold={fold_id}")
    for parent_id in expected_ids:
        parent_rows = [
            row for row in records["parent"]
            if str(row.get("parent_sample_id", "")) == parent_id
        ]
        if len(parent_rows) != 1:
            raise ValueError(f"V26d {source_name} has repeated parent-final rows: fold={fold_id}")
        parent_length = int(parent_rows[0]["parent_length"])
        expected_window = set(range(parent_length))
        actual_window = {
            int(row["event_index"])
            for row in records["window"]
            if str(row["parent_sample_id"]) == parent_id
        }
        if actual_window != expected_window:
            raise ValueError(f"V26d {source_name} window event coverage is invalid: fold={fold_id}")
        expected_prefix = set(range(MINIMUM_EVIDENCE_EVENTS - 1, max(MINIMUM_EVIDENCE_EVENTS - 1, parent_length - 1)))
        actual_prefix = {
            int(row["event_index"])
            for row in records["eligible_prefix"]
            if str(row["parent_sample_id"]) == parent_id
        }
        if actual_prefix != expected_prefix:
            raise ValueError(f"V26d {source_name} eligible-prefix event coverage is invalid: fold={fold_id}")


def _validate_treatment_prediction(
    payload: Mapping[str, Any],
    *,
    root: Path,
    fold_id: int,
    fold: Mapping[str, Any],
    fold_binding: Mapping[str, Any],
    treatment_checkpoint: Path,
    control_prediction: Mapping[str, Any],
) -> dict[str, Any]:
    records = {
        scope: [dict(row) for row in list(dict(payload.get("records") or {}).get(scope) or ())]
        for scope in RAW_SCOPES
    }
    expected_ids = list(fold["held_out_parent_ids"])
    parent_ids = sorted(str(row.get("parent_sample_id", "")) for row in records["parent"])
    expected_paths = dict(fold_binding.get("paths") or {})
    expected_hashes = dict(fold_binding.get("sha256") or {})
    source_lineage = dict(fold_binding.get("shared_checkpoint_lineage") or {})
    if (
        payload.get("schema_version") != TREATMENT_PREDICTION_SCHEMA
        or int(payload.get("fold_id", -1)) != fold_id
        or int(payload.get("endpoint_epoch", -1)) != TREATMENT_EPOCH
        or payload.get("held_out_checkpoint_selection") is not False
        or int(payload.get("held_out_forward_count", -1)) != 1
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("unknown_oof_prediction_count", -1)) != 0
        or int(payload.get("validation_access_count", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or set(dict(payload.get("records") or {})) != set(RAW_SCOPES)
        or payload.get("v26d_contract_sha256") != sha256_file(root / CONTRACT)
        or payload.get("config_path") != expected_paths.get("l1_config")
        or payload.get("config_sha256") != expected_hashes.get("l1_config")
        or payload.get("checkpoint_path") != expected_paths.get("treatment_checkpoint")
        or payload.get("checkpoint_sha256") != sha256_file(treatment_checkpoint)
        or payload.get("checkpoint_sha256") != expected_hashes.get("treatment_checkpoint")
        or payload.get("control_prediction_path") != expected_paths.get("control_prediction")
        or payload.get("control_prediction_sha256")
        != sha256_file(root / l1_prediction_path(fold_id))
        or payload.get("control_prediction_sha256") != expected_hashes.get("control_prediction")
        or payload.get("fold_contract_sha256") != source_lineage.get("fold_contract_sha256")
        or payload.get("readiness_sha256") != source_lineage.get("readiness_sha256")
        or payload.get("source_config_sha256") != source_lineage.get("source_config_sha256")
        or payload.get("config_identity_sha256") != source_lineage.get("config_identity_sha256")
        or payload.get("initial_model_state_sha256") != source_lineage.get("initial_model_state_sha256")
        or parent_ids != expected_ids
        or len(parent_ids) != len(set(parent_ids))
        or any(not records[scope] for scope in RAW_SCOPES)
    ):
        raise ValueError(f"V26d treatment prediction binding is invalid: fold={fold_id}")
    _validate_l1_records(
        records,
        fold_id=fold_id,
        expected_parent_ids=expected_ids,
        source_name="treatment_epoch20",
    )
    control_records = {
        scope: [dict(row) for row in list(dict(control_prediction.get("records") or {}).get(scope) or ())]
        for scope in RAW_SCOPES
    }
    for scope in RAW_SCOPES:
        treatment_keys = {
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in records[scope]
        }
        control_keys = {
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in control_records[scope]
        }
        if treatment_keys != control_keys or len(treatment_keys) != len(records[scope]):
            raise ValueError(f"V26d treatment/control {scope} coverage differs: fold={fold_id}")
    return records


def _materialize_treatment_prediction(
    root: Path,
    *,
    contract: Mapping[str, Any],
    fold_id: int,
    control_prediction: Mapping[str, Any],
) -> dict[str, Any]:
    source_contract_path = root / Path(
        str(dict(contract["source_v26c"])["paths"]["contract"])
    )
    source_contract = _read(source_contract_path)
    fold = fold_record(source_contract, fold_id)
    output_path = root / treatment_prediction_path(fold_id)
    treatment_checkpoint = root / l1_last_checkpoint_path(fold_id)
    if output_path.is_file():
        return _validate_treatment_prediction(
            _read(output_path),
            root=root,
            fold_id=fold_id,
            fold=fold,
            fold_binding=next(
                dict(item)
                for item in list(contract.get("folds") or ())
                if int(dict(item).get("fold_id", -1)) == fold_id
            ),
            treatment_checkpoint=treatment_checkpoint,
            control_prediction=control_prediction,
        )

    prepared = _v26c_require_prepared(root, fold_id)
    binding = dict(prepared["l1_binding"])
    config, config_path, _ = _runtime_l1_config(root, fold_id, binding)
    source_config, _, source_config_sha, _ = _source_protocols(root)
    dataset = build_known_gate_oof_dataset(
        root,
        config,
        parent_ids=fold["held_out_parent_ids"],
        feature_cache_path=root / l1_held_out_feature_cache_path(fold_id),
    )
    held_out_scope = _dataset_scope(
        dataset,
        fold["held_out_parent_ids"],
        ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"),
        "v26d_l1_held_out",
    )
    model, reference, pairing = build_known_gate_model_for_config(root, source_config)
    if reference is not None:
        del reference
    model.config = config
    initial_state_sha = _state_dict_sha256(model.state_dict())
    config_identity = known_gate_config_identity_sha256(config)
    device = known_gate_device(config)
    model.to(device)
    checkpoint_payload = _load_checkpoint(
        treatment_checkpoint,
        task="known_gate",
        schema=L1_CHECKPOINT_SCHEMA,
        fold_id=fold_id,
        model=model,
        optimizer=None,
        config_identity=config_identity,
        config_path=config_path,
        contract_sha256=prepared["contract_sha256"],
        readiness_sha256=prepared["readiness_sha256"],
        source_config_sha256=source_config_sha,
        initial_state_sha256=initial_state_sha,
        device=device,
    )
    if int(checkpoint_payload.get("epoch", -1)) != TREATMENT_EPOCH:
        raise ValueError(f"V26d treatment checkpoint is not epoch 20: fold={fold_id}")
    evaluation = evaluate_known_gate(model, dataset, config=config, device=device)
    _add_fold_fields(evaluation, fold_id=fold_id, task="v26d_l1_epoch20")
    if _record_parent_ids(evaluation) != list(fold["held_out_parent_ids"]):
        raise ValueError(f"V26d treatment parent coverage differs: fold={fold_id}")
    payload = {
        "status": "completed",
        "schema_version": TREATMENT_PREDICTION_SCHEMA,
        "task_name": "known_gate",
        "fold_id": fold_id,
        "role": "V26d fixed epoch-20 endpoint held-out inference; not checkpoint selection",
        "endpoint_epoch": TREATMENT_EPOCH,
        "control_epoch": CONTROL_EPOCH,
        "held_out_checkpoint_selection": False,
        "held_out_forward_count": 1,
        "v26d_contract_sha256": sha256_file(root / CONTRACT),
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "config_path": relative_path(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity,
        "fold_contract_sha256": prepared["contract_sha256"],
        "readiness_sha256": prepared["readiness_sha256"],
        "source_config_sha256": source_config_sha,
        "checkpoint_path": relative_path(root, treatment_checkpoint),
        "checkpoint_sha256": sha256_file(treatment_checkpoint),
        "control_prediction_path": relative_path(root, root / l1_prediction_path(fold_id)),
        "control_prediction_sha256": sha256_file(root / l1_prediction_path(fold_id)),
        "initial_model_state_sha256": initial_state_sha,
        "final_model_state_sha256": _state_dict_sha256(model.state_dict()),
        "source_initialization": pairing,
        "held_out_scope": held_out_scope,
        "evaluation": {key: value for key, value in evaluation.items() if key != "records"},
        "records": evaluation["records"],
        "deployment_eligible": False,
    }
    _write_prediction(output_path, payload)
    del model, dataset, evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return _validate_treatment_prediction(
        payload,
        root=root,
        fold_id=fold_id,
        fold=fold,
        fold_binding=next(
            dict(item)
            for item in list(contract.get("folds") or ())
            if int(dict(item).get("fold_id", -1)) == fold_id
        ),
        treatment_checkpoint=treatment_checkpoint,
        control_prediction=control_prediction,
    )


def _compact_raw_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metric.get(key)
        for key in (
            "scope",
            "support",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
            "per_class",
            "confusion_matrix_rows_true_columns_pred",
        )
    }


def _raw_views(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "parent": _compact_raw_metric(
            summarize_known_gate_records(rows["parent"], scope="parent_final_raw_l1")
        ),
        "eligible_prefix": _compact_raw_metric(
            summarize_known_gate_records(rows["eligible_prefix"], scope="eligible_prefix_raw_l1")
        ),
        "window": _compact_raw_metric(
            summarize_known_gate_records(rows["window"], scope="window_raw_l1")
        ),
        "eligible_prefix_parent_balanced": _compact_raw_metric(
            summarize_parent_balanced_eligible_prefix_records(
                rows["eligible_prefix"],
                scope="eligible_prefix_parent_balanced_raw_l1",
            )
        ),
    }


def _compact_action_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metric.get(key)
        for key in (
            "scope",
            "event_support",
            "parent_support",
            "aggregation",
            "per_class",
            "known_target_recall",
            "clutter_precision",
            "clutter_recall",
            "known_to_clutter_error_rate",
            "clutter_to_known_error_rate",
            "unknown_rejected_known_count",
            "pending_known_count",
            "accepted_action_accuracy",
            "selective_accuracy",
            "action_coverage",
            "confusion_rows_true_known_label_columns_semantic_action",
        )
    }


def _action_views(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "parent": _compact_action_metric(_metrics(rows["parent"], scope="parent_final")),
        "eligible_prefix": _compact_action_metric(
            _metrics(rows["eligible_prefix"], scope="eligible_prefix")
        ),
        "window": _compact_action_metric(_metrics(rows["window"], scope="window")),
        "eligible_prefix_parent_balanced": _compact_action_metric(
            _metrics(
                rows["eligible_prefix"],
                scope="eligible_prefix_parent_balanced",
                parent_balanced=True,
            )
        ),
    }


def _calibrate(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    label: str,
) -> tuple[dict[int, dict[str, float]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    parent_rows = list(rows["parent"])
    thresholds_by_fold: dict[int, dict[str, float]] = {}
    cross_fit: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fit_rows = [row for row in parent_rows if int(row["fold_id"]) != fold_id]
        held_rows = [row for row in parent_rows if int(row["fold_id"]) == fold_id]
        if not fit_rows or not held_rows or any(int(row["fold_id"]) == fold_id for row in fit_rows):
            raise ValueError(f"V26d {label} threshold fold separation is invalid: {fold_id}")
        thresholds, fit = _fit_thresholds(fit_rows)
        thresholds_by_fold[fold_id] = thresholds
        held_parent_actions = _v26c_actions(held_rows, thresholds)
        held_scope_metrics = {
            "parent": _compact_action_metric(
                _metrics(held_parent_actions, scope=f"{label}_outer_fold_{fold_id}_parent")
            )
        }
        cross_fit.append(
            {
                "fold_id": fold_id,
                "fit_parent_count": len(fit_rows),
                "fit_dbb_parent_count": sum(str(row["label_name"]) in DBB_LABELS for row in fit_rows),
                "held_out_parent_count": len(held_rows),
                "thresholds": thresholds,
                "fit": fit,
                "held_out_metrics": held_scope_metrics["parent"],
            }
        )
    calibrated = {
        scope: [
            action
            for row in rows[scope]
            for action in _v26c_actions([row], thresholds_by_fold[int(row["fold_id"])])
        ]
        for scope in RAW_SCOPES
    }
    return thresholds_by_fold, {"cross_fit": cross_fit}, calibrated


def _apply_thresholds(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds_by_fold: Mapping[int, Mapping[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        scope: [
            action
            for row in rows[scope]
            for action in _v26c_actions([row], thresholds_by_fold[int(row["fold_id"])])
        ]
        for scope in RAW_SCOPES
    }


def _all_action_views(calibrated: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return _action_views(calibrated)


def _metric_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    metric_names: Sequence[str],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in metric_names:
        left, right = treatment.get(name), control.get(name)
        result[name] = None if left is None or right is None else float(left) - float(right)
    return result


def _action_metric_deltas(
    treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, float | None]:
    return _metric_delta(
        treatment,
        control,
        metric_names=(
            "known_target_recall",
            "clutter_precision",
            "clutter_recall",
            "clutter_to_known_error_rate",
            "known_to_clutter_error_rate",
            "accepted_action_accuracy",
            "action_coverage",
        ),
    )


def _raw_parent_transition(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right) or len(left) != len(control) or len(right) != len(treatment):
        raise ValueError("V26d raw parent pairing is not one-to-one")
    transitions: Counter[tuple[str, str, str]] = Counter()
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if str(a["label_name"]) != str(b["label_name"]):
            raise ValueError(f"V26d raw paired labels differ: {parent_id}")
        transitions[(str(a["label_name"]), str(a["pred_id"]), str(b["pred_id"]))] += 1
    changed = sum(count for (_, before, after), count in transitions.items() if before != after)
    return {
        "parent_count": len(left),
        "changed_parent_count": changed,
        "unchanged_parent_count": len(left) - changed,
        "transition_counts": [
            {"label_name": label, "control_pred_id": int(before), "treatment_pred_id": int(after), "count": count}
            for (label, before, after), count in sorted(transitions.items())
        ],
    }


def _action_transition(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right) or len(left) != len(control) or len(right) != len(treatment):
        raise ValueError("V26d action parent pairing is not one-to-one")
    transitions: Counter[tuple[str, str, str]] = Counter()
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if str(a["label_name"]) != str(b["label_name"]):
            raise ValueError(f"V26d action paired labels differ: {parent_id}")
        transitions[(str(a["label_name"]), str(a["semantic_action"]), str(b["semantic_action"]))] += 1
    changed = sum(count for (_, before, after), count in transitions.items() if before != after)
    return {
        "parent_count": len(left),
        "changed_parent_count": changed,
        "unchanged_parent_count": len(left) - changed,
        "transition_counts": [
            {"label_name": label, "control_action": before, "treatment_action": after, "count": count}
            for (label, before, after), count in sorted(transitions.items())
        ],
    }


def _label_action_distribution(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, int]:
    values = Counter(str(row["semantic_action"]) for row in rows if str(row["label_name"]) == label)
    return {name: int(values.get(name, 0)) for name in ACTION_NAMES if values.get(name, 0)}


def _component_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    parents = {str(row["parent_sample_id"]): dict(row) for row in rows}
    if len(parents) != len(rows) or len(parents) != 986:
        raise ValueError("V26d paired bootstrap requires one row per Known parent")
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: dict[str, Counter[str]] = defaultdict(Counter)
    for row in parents.values():
        component = str(row.get("provenance_component_id", ""))
        if not component:
            raise ValueError("V26d parent row has no provenance component")
        components[component].append(row)
        signatures[component][str(row["label_name"])] += 1
    strata: dict[str, list[str]] = defaultdict(list)
    for component, counts in signatures.items():
        signature = "+".join(f"{name}:{counts[name]}" for name in sorted(counts))
        strata[signature].append(component)
    if len(components) != 975:
        raise ValueError("V26d provenance component count differs from V26c")
    return components, dict(strata)


def _paired_bootstrap(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_by_id = {str(row["parent_sample_id"]): dict(row) for row in control}
    treatment_by_id = {str(row["parent_sample_id"]): dict(row) for row in treatment}
    if set(control_by_id) != set(treatment_by_id):
        raise ValueError("V26d paired bootstrap parent sets differ")
    control_components, strata = _component_groups(control)
    treatment_components, treatment_strata = _component_groups(treatment)
    if set(control_components) != set(treatment_components) or strata != treatment_strata:
        raise ValueError("V26d paired bootstrap provenance strata differ")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = (
        "known_target_recall",
        "clutter_precision",
        "clutter_recall",
        "clutter_to_known_error_rate",
        "known_to_clutter_error_rate",
        "accepted_action_accuracy",
        "action_coverage",
    )
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected: list[str] = []
        for component_ids in strata.values():
            selected.extend(
                str(value)
                for value in rng.choice(component_ids, size=len(component_ids), replace=True).tolist()
            )
        sampled_control = [row for component in selected for row in control_components[component]]
        sampled_treatment = [row for component in selected for row in treatment_components[component]]
        control_metrics = _metrics(sampled_control, scope="v26d_bootstrap_control")
        treatment_metrics = _metrics(sampled_treatment, scope="v26d_bootstrap_treatment")
        for name in metric_names:
            before, after = control_metrics.get(name), treatment_metrics.get(name)
            if before is not None and after is not None:
                values[name].append(float(after) - float(before))
    return {
        "method": "paired provenance-component bootstrap; cross-fitted actions fixed before resampling",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "component_count": len(control_components),
        "stratum_component_counts": {name: len(items) for name, items in sorted(strata.items())},
        "delta_metrics": {
            name: {
                "median": float(np.median(values[name])),
                "lower_95": float(np.quantile(values[name], 0.025)),
                "upper_95": float(np.quantile(values[name], 0.975)),
                "finite_replicates": len(values[name]),
            }
            for name in sorted(values)
            if values[name]
        },
    }


def _fold_raw_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for fold_id in range(FOLD_COUNT):
        fold_rows = [row for row in rows if int(row["fold_id"]) == fold_id]
        metric = summarize_known_gate_records(fold_rows, scope=f"fold_{fold_id}_parent_final_raw_l1")
        clutter = next(item for item in metric["per_class"] if item["class"] == KNOWN_GATE_CLASS_NAMES[1])
        target = next(item for item in metric["per_class"] if item["class"] == KNOWN_GATE_CLASS_NAMES[0])
        result.append(
            {
                "fold_id": fold_id,
                "support": len(fold_rows),
                "known_target": {key: target[key] for key in ("precision", "recall", "f1", "support")},
                "known_clutter": {key: clutter[key] for key in ("precision", "recall", "f1", "support")},
            }
        )
    return result


def _fold_action_metrics(rows: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    result = []
    for fold_id in range(FOLD_COUNT):
        fold_rows = [row for row in rows if int(row["fold_id"]) == fold_id]
        metric = _compact_action_metric(_metrics(fold_rows, scope=f"{label}_fold_{fold_id}_parent_final"))
        result.append({"fold_id": fold_id, "metrics": metric})
    return result


def _parent_detail_rows(
    control_raw: Sequence[Mapping[str, Any]],
    treatment_raw: Sequence[Mapping[str, Any]],
    control_actions: Sequence[Mapping[str, Any]],
    treatment_frozen: Sequence[Mapping[str, Any]],
    treatment_refit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result = {str(row["parent_sample_id"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("V26d parent detail rows repeat a parent")
        return result

    indexes = [index(rows) for rows in (control_raw, treatment_raw, control_actions, treatment_frozen, treatment_refit)]
    if any(set(indexes[0]) != set(value) for value in indexes[1:]):
        raise ValueError("V26d parent detail coverage differs")
    rows: list[dict[str, Any]] = []
    for parent_id in sorted(indexes[0]):
        c, t, ca, tf, tr = (mapping[parent_id] for mapping in indexes)
        rows.append(
            {
                "fold_id": int(c["fold_id"]),
                "parent_sample_id": parent_id,
                "label_name": str(c["label_name"]),
                "control_raw_pred_id": int(c["pred_id"]),
                "treatment_raw_pred_id": int(t["pred_id"]),
                "control_target_probability": float(c["known_target_probability"]),
                "control_clutter_probability": float(c["known_clutter_probability"]),
                "treatment_target_probability": float(t["known_target_probability"]),
                "treatment_clutter_probability": float(t["known_clutter_probability"]),
                "control_action": str(ca["semantic_action"]),
                "treatment_action_frozen_threshold": str(tf["semantic_action"]),
                "treatment_action_refit_threshold": str(tr["semantic_action"]),
                "control_clutter_threshold": float(ca["l1_known_clutter_acceptance_threshold"]),
                "treatment_clutter_threshold_frozen": float(tf["l1_known_clutter_acceptance_threshold"]),
                "treatment_clutter_threshold_refit": float(tr["l1_known_clutter_acceptance_threshold"]),
            }
        )
    return rows


def _write_parent_details(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "fold_id",
        "parent_sample_id",
        "label_name",
        "control_raw_pred_id",
        "treatment_raw_pred_id",
        "control_target_probability",
        "control_clutter_probability",
        "treatment_target_probability",
        "treatment_clutter_probability",
        "control_action",
        "treatment_action_frozen_threshold",
        "treatment_action_refit_threshold",
        "control_clutter_threshold",
        "treatment_clutter_threshold_frozen",
        "treatment_clutter_threshold_refit",
    )
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    text = stream.getvalue()
    if len(text.encode("utf-8")) > MAX_DETAIL_BYTES:
        raise MemoryError("V26d parent detail CSV exceeds its bounded size")
    return _write_text_once(root / PARENT_DETAIL_CSV, text, description="parent detail CSV")


def _compare_control_to_v26c(
    recomputed: Mapping[str, Any],
    source_report: Mapping[str, Any],
    thresholds: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    source_metrics = dict(source_report.get("aggregate_metrics") or {})
    metric_names = (
        "event_support",
        "parent_support",
        "known_target_recall",
        "clutter_precision",
        "clutter_recall",
        "known_to_clutter_error_rate",
        "clutter_to_known_error_rate",
        "unknown_rejected_known_count",
        "pending_known_count",
        "accepted_action_accuracy",
        "selective_accuracy",
        "action_coverage",
    )
    mismatches: list[str] = []
    for scope, metric in recomputed.items():
        source_metric = dict(source_metrics.get(scope) or {})
        for name in metric_names:
            left, right = metric.get(name), source_metric.get(name)
            if left is None or right is None:
                if left != right:
                    mismatches.append(f"{scope}.{name}")
            elif abs(float(left) - float(right)) > 1.0e-12:
                mismatches.append(f"{scope}.{name}")
    source_cross_fit = list(source_report.get("cross_fit") or ())
    threshold_matches = True
    if len(source_cross_fit) != FOLD_COUNT:
        threshold_matches = False
    else:
        for entry in source_cross_fit:
            fold_id = int(entry.get("fold_id", -1))
            if fold_id not in thresholds or dict(entry.get("thresholds") or {}) != dict(thresholds[fold_id]):
                threshold_matches = False
    return {
        "metrics_match": not mismatches,
        "thresholds_match": threshold_matches,
        "mismatches": mismatches,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract, readiness = require_prepared(root)
    source_contract_path = root / Path(
        str(dict(contract["source_v26c"])["paths"]["contract"])
    )
    source_contract = _read(source_contract_path)
    source_report_path = root / Path(str(dict(contract["source_v26c"])["paths"]["report"]))
    source_report = _read(source_report_path)

    control_rows: dict[str, list[dict[str, Any]]] = {scope: [] for scope in RAW_SCOPES}
    treatment_rows: dict[str, list[dict[str, Any]]] = {scope: [] for scope in RAW_SCOPES}
    control_raw_rows: dict[str, list[dict[str, Any]]] = {scope: [] for scope in RAW_SCOPES}
    treatment_raw_rows: dict[str, list[dict[str, Any]]] = {scope: [] for scope in RAW_SCOPES}
    fold_bindings = {int(row["fold_id"]): row for row in list(contract.get("folds") or ())}
    for fold_id in range(FOLD_COUNT):
        if fold_id not in fold_bindings:
            raise ValueError(f"V26d contract lacks fold {fold_id}")
        control_prediction, l2_prediction = _v26c_prediction(root, fold_id=fold_id)
        control_joint = _joint_rows(control_prediction, l2_prediction, fold_id)
        treatment_fold_records = _materialize_treatment_prediction(
            root,
            contract=contract,
            fold_id=fold_id,
            control_prediction=control_prediction,
        )
        # The L2 joint rows are intentionally built once from V26c's sealed
        # output.  Only the two L1 probabilities are substituted below.
        for scope in RAW_SCOPES:
            control_raw_scope = [
                dict(row)
                for row in list(dict(control_prediction.get("records") or {}).get(scope) or ())
            ]
            control_scope = [dict(row) for row in control_joint[scope]]
            treatment_scope = [dict(row) for row in treatment_fold_records[scope]]
            control_raw_by_key = {
                (str(row["parent_sample_id"]), int(row["event_index"])): row
                for row in control_raw_scope
            }
            treatment_by_key = {
                (str(row["parent_sample_id"]), int(row["event_index"])): row
                for row in treatment_scope
            }
            if set(control_raw_by_key) != set(treatment_by_key) or len(control_raw_by_key) != len(control_raw_scope):
                raise ValueError(f"V26d raw L1 {scope} coverage differs: fold={fold_id}")
            for row in control_scope:
                key = (str(row["parent_sample_id"]), int(row["event_index"]))
                treatment = treatment_by_key[key]
                control_raw = control_raw_by_key[key]
                static_fields = (
                    "parent_sample_id",
                    "label_name",
                    "event_index",
                    "window_length",
                    "product_action_eligible",
                )
                for field in static_fields:
                    if row.get(field) != treatment.get(field):
                        raise ValueError(
                            f"V26d endpoint changed non-L1 field {field}: fold={fold_id}; key={key}"
                        )
                for field in (
                    "true_id",
                    "parent_sample_id",
                    "track_id",
                    "source_name",
                    "label_name",
                    "event_index",
                    "window_length",
                    "product_action_eligible",
                    "rd_frame_count",
                    "rd_present",
                    "rd_multiple_frames",
                    "parent_length",
                    "rd_quality_missing",
                    "rd_quality_frame_count_scaled",
                ):
                    if control_raw.get(field) != treatment.get(field):
                        raise ValueError(
                            f"V26d endpoint changed raw L1 metadata {field}: fold={fold_id}; key={key}"
                        )
                replacement = dict(row)
                replacement["l1_known_target_probability"] = float(treatment["known_target_probability"])
                replacement["l1_known_clutter_probability"] = float(treatment["known_clutter_probability"])
                replacement["l1_endpoint_epoch"] = TREATMENT_EPOCH
                control_rows[scope].append(row)
                treatment_rows[scope].append(replacement)
            control_raw_rows[scope].extend(control_raw_scope)
            treatment_raw_rows[scope].extend(treatment_scope)
        _validate_l1_records(
            {scope: list(dict(control_prediction.get("records") or {}).get(scope) or ()) for scope in RAW_SCOPES},
            fold_id=fold_id,
            expected_parent_ids=fold_record(source_contract, fold_id)["held_out_parent_ids"],
            source_name="control_epoch14",
        )
    all_parent_ids = sorted(str(row["parent_sample_id"]) for row in control_rows["parent"])
    if len(all_parent_ids) != 986 or len(all_parent_ids) != len(set(all_parent_ids)):
        raise ValueError("V26d control OOF coverage is not exactly 986 Known parents")
    treatment_parent_ids = sorted(str(row["parent_sample_id"]) for row in treatment_rows["parent"])
    if treatment_parent_ids != all_parent_ids:
        raise ValueError("V26d treatment OOF parent coverage differs from control")
    assignment = _assignment_by_parent(root, all_parent_ids)
    control_rows = {scope: _annotate_provenance(rows, assignment) for scope, rows in control_rows.items()}
    treatment_rows = {scope: _annotate_provenance(rows, assignment) for scope, rows in treatment_rows.items()}

    control_thresholds, control_fit_info, control_calibrated = _calibrate(control_rows, label="control")
    treatment_thresholds, treatment_fit_info, treatment_recalibrated = _calibrate(treatment_rows, label="treatment_epoch20")
    treatment_frozen = _apply_thresholds(treatment_rows, control_thresholds)
    control_action_views = _all_action_views(control_calibrated)
    treatment_frozen_views = _all_action_views(treatment_frozen)
    treatment_refit_views = _all_action_views(treatment_recalibrated)
    control_reproduction = _compare_control_to_v26c(
        {scope: control_action_views[scope] for scope in ("parent", "eligible_prefix", "window", "eligible_prefix_parent_balanced")},
        source_report,
        control_thresholds,
    )
    if not control_reproduction["metrics_match"] or not control_reproduction["thresholds_match"]:
        raise AssertionError(
            "V26d recomputed control no longer reproduces V26c: "
            + json.dumps(control_reproduction, ensure_ascii=False)
        )

    raw_control = _raw_views(control_raw_rows)
    raw_treatment = _raw_views(treatment_raw_rows)
    raw_deltas = {
        scope: _metric_delta(
            raw_treatment[scope],
            raw_control[scope],
            metric_names=("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"),
        )
        for scope in ("parent", "eligible_prefix", "window")
    }
    parent_control_actions = control_calibrated["parent"]
    parent_treatment_frozen = treatment_frozen["parent"]
    parent_treatment_refit = treatment_recalibrated["parent"]
    parent_details = _parent_detail_rows(
        control_raw_rows["parent"],
        treatment_raw_rows["parent"],
        parent_control_actions,
        parent_treatment_frozen,
        parent_treatment_refit,
    )
    _write_parent_details(root, parent_details)
    detail_status = "immutable_verified"
    true_clutter = {
        "control": _label_action_distribution(parent_control_actions, "ClutterTarget"),
        "treatment_frozen_threshold": _label_action_distribution(parent_treatment_frozen, "ClutterTarget"),
        "treatment_refit_threshold": _label_action_distribution(parent_treatment_refit, "ClutterTarget"),
    }
    report = {
        "status": "completed_v26d_l1_epoch20_endpoint_diagnostic",
        "schema_version": REPORT_SCHEMA,
        "scope": "Known-only provenance-grouped OOF endpoint diagnostic; no training, Unknown, validation, test or deployment",
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "source_v26c_report_path": relative_path(root, source_report_path),
        "source_v26c_report_sha256": sha256_file(source_report_path),
        "known_parent_count": 986,
        "known_label_parent_support": {
            "BalloonTarget": 238,
            "BirdTarget": 239,
            "ClutterTarget": 33,
            "DroneTarget": 476,
        },
        "checkpoint_protocol": {
            "control_epoch": CONTROL_EPOCH,
            "treatment_epoch": TREATMENT_EPOCH,
            "treatment_source": "V26c same-fold l1/last_model.pt after epoch 20",
            "outer_held_out_checkpoint_selection": False,
            "per_fold_epoch_selection": False,
            "l2_endpoint_changed": False,
            "l2_outputs_reused": True,
        },
        "integrity": {
            "control_reproduces_v26c_metrics": control_reproduction["metrics_match"],
            "control_reproduces_v26c_thresholds": control_reproduction["thresholds_match"],
            "control_reproduction_mismatches": control_reproduction["mismatches"],
            "all_parent_ids_unique_and_paired": True,
            "only_l1_probability_columns_replaced": True,
            "l2_scores_and_metadata_reused": True,
        },
        "raw_l1": {
            "control_epoch14": {
                "metrics": raw_control,
                "fold_parent_metrics": _fold_raw_metrics(control_raw_rows["parent"]),
            },
            "treatment_epoch20": {
                "metrics": raw_treatment,
                "fold_parent_metrics": _fold_raw_metrics(treatment_raw_rows["parent"]),
            },
            "delta_treatment_minus_control": raw_deltas,
            "paired_parent_transition": _raw_parent_transition(
                control_raw_rows["parent"], treatment_raw_rows["parent"]
            ),
        },
        "l3_action_views": {
            "control_epoch14_cross_fitted": {
                "metrics": control_action_views,
                "fold_parent_metrics": _fold_action_metrics(parent_control_actions, "control"),
                "cross_fit": control_fit_info["cross_fit"],
                "thresholds_by_fold": control_thresholds,
                "true_clutter_parent_outcomes": true_clutter["control"],
            },
            "treatment_epoch20_frozen_control_thresholds": {
                "metrics": treatment_frozen_views,
                "fold_parent_metrics": _fold_action_metrics(parent_treatment_frozen, "treatment_frozen"),
                "thresholds_by_fold": control_thresholds,
                "delta_vs_control_parent": _action_metric_deltas(
                    treatment_frozen_views["parent"], control_action_views["parent"]
                ),
                "true_clutter_parent_outcomes": true_clutter["treatment_frozen_threshold"],
                "paired_parent_transition": _action_transition(
                    parent_control_actions, parent_treatment_frozen
                ),
            },
            "treatment_epoch20_recalibrated_cross_fitted": {
                "metrics": treatment_refit_views,
                "fold_parent_metrics": _fold_action_metrics(parent_treatment_refit, "treatment_refit"),
                "cross_fit": treatment_fit_info["cross_fit"],
                "thresholds_by_fold": treatment_thresholds,
                "delta_vs_control_parent": _action_metric_deltas(
                    treatment_refit_views["parent"], control_action_views["parent"]
                ),
                "true_clutter_parent_outcomes": true_clutter["treatment_refit_threshold"],
                "paired_parent_transition": _action_transition(
                    parent_control_actions, parent_treatment_refit
                ),
            },
        },
        "paired_component_bootstrap": {
            "frozen_control_threshold_delta": _paired_bootstrap(
                parent_control_actions, parent_treatment_frozen
            ),
            "recalibrated_threshold_delta": _paired_bootstrap(
                parent_control_actions, parent_treatment_refit
            ),
        },
        "parent_detail_csv": relative_path(root, root / PARENT_DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "unknown_threshold_fit_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "training_executed": False,
        "mixed_unknown_audit": "not_executed",
        "full_cascade": "not_executed",
        "deployment_eligible": False,
    }
    report_write = _write_report(root, report)
    return {
        "status": report["status"],
        "report": str(root / REPORT),
        "report_sha256": sha256_file(root / REPORT),
        "report_write": report_write,
        "parent_detail_csv": str(root / PARENT_DETAIL_CSV),
        "control_reproduces_v26c": control_reproduction["metrics_match"] and control_reproduction["thresholds_match"],
        "training_executed": False,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V26d's fixed epoch-20 endpoint diagnostic.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
