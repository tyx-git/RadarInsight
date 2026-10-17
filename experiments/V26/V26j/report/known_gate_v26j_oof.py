"""Calibrate V26j causal-GRU L1 epoch-32 OOF with frozen V26c L2 reuse."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate_v26c_oof import (
    L2_EPOCH as V26C_L2_EPOCH,
    L2_PREDICTION_SCHEMA as V26C_L2_PREDICTION_SCHEMA,
    l1_prediction_path as v26c_l1_prediction_path,
    l2_checkpoint_path as v26c_l2_checkpoint_path,
    l2_prediction_path as v26c_l2_prediction_path,
    l2_training_summary_path as v26c_l2_training_summary_path,
)
from radar_ln3.known_gate_v26j_oof import CONTRACT, CONTRACT_SCHEMA, FOLD_COUNT, L1_EPOCH, L1_PREDICTION_SCHEMA, L1_SOURCE_CONFIG, L3_REGISTRY, L3_REPORT, L3_REPORT_SCHEMA, L1_KNOWN_TARGET_ACTION_RECALL_FLOOR, L2_CORRECT_ACTION_RECALL_FLOOR, CLUTTER_PRECISION_FLOOR, CLUTTER_RECALL_FLOOR, MINIMUM_EVIDENCE_EVENTS, PARENT_DETAIL_CSV, READINESS, READINESS_SCHEMA, V21E_ASSIGNMENT, V21E_FOLD_CONTRACT, fold_record, l1_checkpoint_path, l1_config_path, l1_prediction_path, l1_training_summary_path, relative_path, require_bound_file, sequence_sha256, sha256_file, validate_frozen_v26c_sources
from train.known_gate_v26j_oof import verify_l1_fold
from report.known_gate_v26c_oof import (
    DBB_LABELS,
    _actions,
    _annotate_provenance,
    _assignment_by_parent,
    _bootstrap,
    _fit_thresholds,
    _joint_rows,
    _metrics,
    _write_detail_csv,
)


MAX_REPORT_BYTES = 6 * 1024 * 1024
IMPLEMENTATION_FILES = (
    Path("experiments/V26/V26j/build/known_gate_v26j_oof.py"),
    Path("experiments/V26/V26j/train/known_gate_v26j_oof.py"),
    Path("experiments/V26/V26j/report/known_gate_v26j_oof.py"),
    Path("experiments/V26/V26j/radar_ln3/known_gate_v26j_oof.py"),
    Path("src-v3/radar_ln3/known_gate.py"),
    Path("src-v3/radar_ln3/known_gate_features.py"),
    Path("src-v3/radar_ln3/model.py"),
    Path("experiments/V26/V26j/radar_ln3/known_gate_v26_audit.py"),
    Path("experiments/V26/V26j/radar_ln3/known_gate_v26j_audit.py"),
    Path("experiments/V26/V26j/build/known_gate_v26j_audit.py"),
    Path("experiments/V26/V26j/report/known_gate_v26j_audit.py"),
    Path("experiments/V26/V26j/scripts/v26j_known_gate_rd_causal_gru_oof_pipeline.ps1"),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26j expected JSON object: {path}")
    return value


def _write_report(root: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise MemoryError("V26j report exceeds the bounded size")
    path = root / L3_REPORT
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26j report: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to replace a different V26j report: {temporary}")
    if not temporary.exists():
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def _implementation_sha256(root: Path) -> dict[str, str]:
    return {
        relative_path(root, require_bound_file(root, path, f"V26j implementation {path}")): sha256_file(root / path)
        for path in IMPLEMENTATION_FILES
    }


def _require_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract, readiness = _read(contract_path), _read(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks or not all(value is True for value in checks.values())
        or contract.get("single_variable", {}).get("field") != "model.rd_frame_aggregation"
        or contract.get("single_variable", {}).get("control") != "masked_mean"
        or contract.get("single_variable", {}).get("treatment") != "causal_gru"
        or int(contract.get("l1", {}).get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
    ):
        raise ValueError("V26j prepared contract/readiness is invalid")
    validate_frozen_v26c_sources(root, contract)
    return contract, readiness


def _rebind_l2_to_v26j_l1(root: Path, *, l1: Mapping[str, Any], l2: Mapping[str, Any], fold_id: int, l1_path: Path) -> dict[str, Any]:
    result = copy.deepcopy(dict(l2))
    joint = dict(result.get("joint_l1_l2_scores") or {})
    l1_records, joint_records = dict(l1.get("records") or {}), dict(joint.get("records") or {})
    scopes = {"parent", "eligible_prefix", "window"}
    if set(l1_records) != scopes or set(joint_records) != scopes:
        raise ValueError(f"V26j fold {fold_id} joint score scopes are incomplete")
    for scope in scopes:
        l1_index: dict[tuple[str, int], dict[str, Any]] = {}
        for original in list(l1_records[scope]):
            row = dict(original); key = (str(row.get("parent_sample_id")), int(row.get("event_index", -1)))
            if key in l1_index:
                raise ValueError(f"V26j fold {fold_id} repeats L1 key: {scope} {key}")
            l1_index[key] = row
        rows, seen = [], set()
        for original in list(joint_records[scope]):
            row = dict(original); key = (str(row.get("parent_sample_id")), int(row.get("event_index", -1)))
            if key in seen or key not in l1_index:
                raise ValueError(f"V26j fold {fold_id} L1/L2 key differs: {scope} {key}")
            seen.add(key); source = l1_index[key]
            for field in ("label_name", "fold_id", "window_length", "product_action_eligible"):
                if str(row.get(field)) != str(source.get(field)):
                    raise ValueError(f"V26j fold {fold_id} L1/L2 metadata differs: {scope} {key} {field}")
            target, clutter = float(source["known_target_probability"]), float(source["known_clutter_probability"])
            if not np.isfinite(target) or not np.isfinite(clutter) or abs(target + clutter - 1.0) > 1e-5:
                raise ValueError(f"V26j fold {fold_id} L1 probability pair is invalid: {key}")
            row["l1_known_target_probability"], row["l1_known_clutter_probability"] = target, clutter
            rows.append(row)
        if seen != set(l1_index):
            raise ValueError(f"V26j fold {fold_id} L1/L2 coverage differs: {scope}")
        joint_records[scope] = rows
    joint["records"] = joint_records
    joint["l1_prediction_path"], joint["l1_prediction_sha256"] = relative_path(root, l1_path), sha256_file(l1_path)
    result["joint_l1_l2_scores"] = joint
    return result


def _load_fold_predictions(root: Path, *, fold_id: int, contract_fold: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Reconstruct the causal-GRU model on CPU and bind the checkpoint to the
    # full 32-epoch train-only history before reading its JSON projection.
    # This prevents a stale JSON artifact from being calibrated as V26j.
    completed = verify_l1_fold(root, fold_id=fold_id)
    l1_path, summary_path = root / l1_prediction_path(fold_id), root / l1_training_summary_path(fold_id)
    l2_path, l2_summary_path = root / v26c_l2_prediction_path(fold_id), root / v26c_l2_training_summary_path(fold_id)
    l1, summary, l2, l2_summary = _read(l1_path), _read(summary_path), _read(l2_path), _read(l2_summary_path)
    binding, l2_binding = dict(contract_fold.get("l1") or {}), dict(contract_fold.get("l2") or {})
    config_path, checkpoint_path, l2_checkpoint_path = root / l1_config_path(fold_id), root / l1_checkpoint_path(fold_id), root / v26c_l2_checkpoint_path(fold_id)
    for path, description in ((l1_path, "L1 prediction"), (summary_path, "L1 summary"), (l2_path, "L2 prediction"), (l2_summary_path, "L2 summary"), (config_path, "L1 config"), (checkpoint_path, "L1 checkpoint"), (l2_checkpoint_path, "L2 checkpoint")):
        require_bound_file(root, path, f"V26j fold {fold_id} {description}")
    runtime = dict(l1.get("rd_runtime") or {})
    if (
        l1.get("schema_version") != L1_PREDICTION_SCHEMA
        or int(l1.get("fold_id", -1)) != fold_id
        or int(l1.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or l1.get("prediction_protocol") != "out_of_fold_fixed_epoch_32"
        or l1.get("held_out_checkpoint_selection") is not False
        or int(l1.get("unknown_training_exposure", -1)) != 0
        or int(l1.get("validation_access_count", -1)) != 0
        or int(l1.get("test_payload_access_count", -1)) != 0
        or l1.get("config_path") != relative_path(root, config_path)
        or l1.get("config_sha256") != sha256_file(config_path)
        or binding.get("config_sha256") != sha256_file(config_path)
        or l1.get("fold_contract_sha256") != sha256_file(root / CONTRACT)
        or l1.get("readiness_sha256") != sha256_file(root / READINESS)
        or l1.get("source_config_sha256") != sha256_file(root / L1_SOURCE_CONFIG)
        or l1.get("checkpoint_path") != relative_path(root, checkpoint_path)
        or l1.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or l1.get("checkpoint_sha256") != completed.get("checkpoint_sha256")
        or runtime.get("rd_frame_aggregation_config") != "causal_gru"
        or runtime.get("rd_frame_aggregation_runtime") != "causal_gru"
        or runtime.get("rd_temporal_module") != "GRUCell"
        or int(runtime.get("rd_temporal_parameter_count", 0)) <= 0
        or not list(runtime.get("rd_temporal_state_keys") or ())
        or summary.get("status") != "completed"
        or dict(summary.get("rd_runtime") or {}) != runtime
        or summary.get("checkpoint_sha256") != completed.get("checkpoint_sha256")
        or summary.get("prediction_sha256") != completed.get("prediction_sha256")
        or sorted(str(row["parent_sample_id"]) for row in list(dict(l1.get("records") or {}).get("parent") or ())) != list(contract_fold["held_out_parent_ids"])
    ):
        raise ValueError(f"V26j fold {fold_id} L1 prediction/summary binding differs")
    if (
        l2.get("schema_version") != V26C_L2_PREDICTION_SCHEMA
        or int(l2.get("fold_id", -1)) != fold_id
        or int(l2.get("fixed_snapshot_epoch", -1)) != V26C_L2_EPOCH
        or l2.get("held_out_l1_scores_reused_from_sealed_prediction") is not True
        or int(l2.get("held_out_l1_forward_count", -1)) != 0
        or int(l2.get("held_out_l2_forward_count", -1)) != 1
        or l2_binding.get("checkpoint_sha256") != sha256_file(l2_checkpoint_path)
        or l2_summary.get("status") != "completed"
    ):
        raise ValueError(f"V26j fold {fold_id} V26c L2 reuse binding differs")
    original_joint = dict(l2.get("joint_l1_l2_scores") or {})
    if original_joint.get("l1_prediction_path") != relative_path(root, root / v26c_l1_prediction_path(fold_id)):
        raise ValueError(f"V26j fold {fold_id} sealed V26c joint L1 binding differs")
    rebound = _rebind_l2_to_v26j_l1(root, l1=l1, l2=l2, fold_id=fold_id, l1_path=l1_path)
    return l1, rebound, {
        "records": _joint_rows(l1, rebound, fold_id),
        "l2_path": relative_path(root, l2_path),
        "rd_runtime": runtime,
        "completed_fold_verification": completed,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _raw_l1_metrics(rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        true_id, pred_id = int(row["true_id"]), int(row["pred_id"])
        if true_id not in (0, 1) or pred_id not in (0, 1):
            raise ValueError(f"V26j raw L1 {scope} has invalid class id")
        matrix[true_id][pred_id] += 1
    names, per_class = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE"), []
    for class_id, name in enumerate(names):
        tp, support, predicted = matrix[class_id][class_id], sum(matrix[class_id]), matrix[0][class_id] + matrix[1][class_id]
        precision, recall = _safe_divide(tp, predicted), _safe_divide(tp, support)
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": _safe_divide(2 * precision * recall, precision + recall), "support": support, "major_confusion_to": names[1-class_id], "major_confusion_count": matrix[class_id][1-class_id]})
    return {"scope": scope, "support": len(rows), "accuracy": _safe_divide(matrix[0][0] + matrix[1][1], len(rows)), "macro_precision": float(np.mean([row["precision"] for row in per_class])), "macro_recall": float(np.mean([row["recall"] for row in per_class])), "macro_f1": float(np.mean([row["f1"] for row in per_class])), "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])), "per_class": per_class, "confusion_matrix_rows_true_columns_pred": matrix}


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract, readiness = _require_prepared(root)
    all_joint = {scope: [] for scope in ("parent", "eligible_prefix", "window")}
    all_l1 = {scope: [] for scope in all_joint}
    fold_raw, l2_reuse, runtimes = [], [], []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        l1, _rebound, loaded = _load_fold_predictions(root, fold_id=fold_id, contract_fold=fold)
        fold_l1_rows = {
            scope: [dict(row) for row in list(dict(l1["records"])[scope])]
            for scope in all_joint
        }
        for scope in all_joint:
            all_joint[scope].extend(loaded["records"][scope])
            all_l1[scope].extend(fold_l1_rows[scope])
        fold_raw.append(
            {
                "fold_id": fold_id,
                **{
                    scope: _raw_l1_metrics(fold_l1_rows[scope], scope=scope)
                    for scope in all_joint
                },
            }
        )
        l2_reuse.append(
            {
                "fold_id": fold_id,
                "prediction_path": loaded["l2_path"],
                "fixed_snapshot_epoch": V26C_L2_EPOCH,
                "training_executed_in_v26j": False,
                "v26j_completed_fold_verification": loaded["completed_fold_verification"],
            }
        )
        runtimes.append({"fold_id": fold_id, **loaded["rd_runtime"]})
    parent_ids = [str(row["parent_sample_id"]) for row in all_joint["parent"]]
    if len(parent_ids) != 986 or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V26j parent OOF coverage must contain each Known parent once")
    assignment = _assignment_by_parent(root, parent_ids)
    if (contract.get("assignment_reuse", {}).get("source") != relative_path(root, root / V21E_ASSIGNMENT) or contract.get("assignment_reuse", {}).get("sha256") != sha256_file(root / V21E_ASSIGNMENT) or contract.get("assignment_reuse", {}).get("source_contract_sha256") != sha256_file(root / V21E_FOLD_CONTRACT)):
        raise ValueError("V26j contract is not bound to frozen V21e assignment")
    all_joint = {scope: _annotate_provenance(rows, assignment) for scope, rows in all_joint.items()}
    support = Counter(str(row["label_name"]) for row in all_joint["parent"])
    if dict(sorted(support.items())) != {"BalloonTarget": 238, "BirdTarget": 239, "ClutterTarget": 33, "DroneTarget": 476}:
        raise ValueError("V26j Known OOF label support differs")
    thresholds_by_fold, cross_fit = {}, []
    for fold_id in range(FOLD_COUNT):
        fit_rows = [row for row in all_joint["parent"] if int(row["fold_id"]) != fold_id]
        held_rows = [row for row in all_joint["parent"] if int(row["fold_id"]) == fold_id]
        thresholds, fit = _fit_thresholds(fit_rows)
        thresholds_by_fold[fold_id] = thresholds
        cross_fit.append({"fold_id": fold_id, "fit_parent_count": len(fit_rows), "fit_dbb_parent_count": sum(str(row["label_name"]) in DBB_LABELS for row in fit_rows), "held_out_parent_count": len(held_rows), "thresholds": thresholds, "fit": fit, "held_out_metrics": _metrics(_actions(held_rows, thresholds), scope=f"outer_fold_{fold_id}_parent_final")})
    calibrated = {scope: [action for row in rows for action in _actions([row], thresholds_by_fold[int(row["fold_id"])])] for scope, rows in all_joint.items()}
    metrics = {"parent": _metrics(calibrated["parent"], scope="parent_final_cross_fitted"), "eligible_prefix": _metrics(calibrated["eligible_prefix"], scope="eligible_prefix_cross_fitted"), "window": _metrics(calibrated["window"], scope="window_cross_fitted"), "eligible_prefix_parent_balanced": _metrics(calibrated["eligible_prefix"], scope="eligible_prefix_parent_balanced_cross_fitted", parent_balanced=True)}
    pooled_thresholds, pooled_fit = _fit_thresholds(all_joint["parent"])
    _write_detail_csv(root / PARENT_DETAIL_CSV, calibrated["parent"])
    report = {"status": "completed_v26j_rd_causal_gru_epoch32_known_only_l3_diagnostic", "schema_version": L3_REPORT_SCHEMA, "scope": "Known-only parent-final OOF L3 calibration with V26j causal-GRU L1 epoch-32 and V26c L2 epoch-15 reuse; no Unknown/validation/test", "experiment": {"direct_baseline": "V26e epoch-32 masked_mean L1", "single_variable": "current-event RD frame aggregation masked_mean -> causal_gru", "l2_changed": False, "checkpoint_selection": "fixed epoch32; no outer-fold or held-out selection"}, "contract_path": relative_path(root, root / CONTRACT), "contract_sha256": sha256_file(root / CONTRACT), "readiness_path": relative_path(root, root / READINESS), "readiness_sha256": sha256_file(root / READINESS), "implementation_sha256": _implementation_sha256(root), "rd_runtime_by_fold": runtimes, "known_parent_count": len(calibrated["parent"]), "known_parent_ids_sha256": sequence_sha256(sorted(parent_ids)), "known_label_parent_support": dict(sorted(support.items())), "unknown_training_exposure": 0, "unknown_normalization_exposure": 0, "unknown_oof_prediction_count": 0, "unknown_threshold_fit_exposure": 0, "validation_access_count": 0, "test_access_count": 0, "threshold_protocol": {"outer_cross_fit": "thresholds for fold f are fitted from other four Known parent-final OOF folds only", "pending_is_unknown_rejection_success": False, "minimum_evidence_events": MINIMUM_EVIDENCE_EVENTS, "safety_floors": {"known_target_action_recall_at_least": L1_KNOWN_TARGET_ACTION_RECALL_FLOOR, "dbb_correct_action_recall_at_least": L2_CORRECT_ACTION_RECALL_FLOOR, "clutter_precision_strictly_greater_than": CLUTTER_PRECISION_FLOOR, "clutter_recall_at_least": CLUTTER_RECALL_FLOOR}}, "raw_l1": {"aggregate": {scope: _raw_l1_metrics(all_l1[scope], scope=scope) for scope in all_l1}, "fold_metrics": fold_raw}, "v26c_l2_reuse": l2_reuse, "cross_fit": cross_fit, "aggregate_metrics": metrics, "parent_component_bootstrap": _bootstrap(calibrated["parent"]), "pooled_known_only_candidate_thresholds": pooled_thresholds, "pooled_known_only_candidate_fit": pooled_fit, "all_outer_folds_feasible_under_known_safety": all(bool(row["fit"]["feasible"]) for row in cross_fit), "registry": {"path": relative_path(root, root / L3_REGISTRY), "written": False, "reason": "five-fold diagnostic is not one deployable L1/L2 registry"}, "parent_detail_csv": relative_path(root, root / PARENT_DETAIL_CSV), "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV), "mixed_unknown_audit": "not_executed", "pending_is_unknown_rejection_success": False, "deployment_eligible": False}
    _write_report(root, report)
    return {"status": report["status"], "report": str(root / L3_REPORT), "report_sha256": sha256_file(root / L3_REPORT), "all_outer_folds_feasible": report["all_outer_folds_feasible_under_known_safety"], "mixed_unknown_audit": "not_executed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V26j causal-GRU epoch-32 Known-only OOF.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()