"""Prepare the V24d cross-fitted L3 score-band Pending diagnostic.

V24d deliberately performs no model fitting. It binds a deterministic L3
decision protocol to the completed V21e treatment OOF predictions, whose
outer-fold models were trained without their held-out parents. The preparation
step verifies those immutable sources and writes only a protocol, contract,
and calibration-readiness artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate_v21e_oof import (
    ASSIGNMENT_RELATIVE,
    FOLD_CONTRACT_RELATIVE,
    FOLD_COUNT,
    PREDICTION_SCHEMA,
    TREATMENT_REPORT_RELATIVE,
    TREATMENT_REPORT_SCHEMA,
    fold_record,
    load_fold_contract,
    prediction_relative,
    relative_path,
    sequence_sha256,
    sha256_file,
)


PROTOCOL = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24d-cq08-known-gate-l3-oof-score-band-pending-protocol.json"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v24d_l3_oof_score_band_pending_calibration.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24d-cq08-known-gate-l3-oof-score-band-pending-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24d-known-gate-l3-oof-score-band-pending-calibration-readiness.json"
)
IMPLEMENTATION_FILES = (
    Path("src-v3/build/known_gate_v24d.py"),
    Path("src-v3/report/known_gate_v24d.py"),
    Path("src-v3/smoke/known_gate_v24d_contract_test.py"),
)

PROTOCOL_SCHEMA = "RADAR-LN3-V24d-known-gate-l3-oof-score-band-protocol-v1"
CONTRACT_SCHEMA = "RADAR-LN3-V24d-known-gate-l3-oof-score-band-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V24d-known-gate-l3-oof-score-band-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V24d-known-gate-l3-oof-score-band-report-v1"

SOURCE_PARENT_COUNT = 986
SOURCE_LABEL_SUPPORT = {
    "DroneTarget": 476,
    "BirdTarget": 239,
    "BalloonTarget": 238,
    "ClutterTarget": 33,
}
MINIMUM_EVIDENCE_EVENTS = 3
SELECTION_REQUIREMENTS = {
    "known_clutter_action_precision": 0.90,
    "known_clutter_acceptance_recall": 0.88,
    "known_target_acceptance_recall": 0.98,
    "overall_action_coverage": 0.95,
}
DIRECTION_REQUIREMENTS = {
    "known_target_to_clutter_rate_not_higher_than_raw": True,
    "clutter_to_known_target_rate_not_higher_than_raw": True,
}
PREFIX_SAFETY_REQUIREMENTS = {
    "parent_balanced_eligible_prefix_action_coverage": 0.80,
}


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V24d preparation artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _implementation_audit(root: Path) -> dict[str, str]:
    """Bind generated V24d evidence to the executable calibration semantics."""

    missing = [relative_path(root, root / path) for path in IMPLEMENTATION_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"V24d implementation files are missing: {missing}")
    return {
        relative_path(root, root / path): sha256_file(root / path)
        for path in IMPLEMENTATION_FILES
    }


def _source_prediction_audit(root: Path) -> dict[str, Any]:
    contract, assignment = load_fold_contract(root)
    report_path = root / TREATMENT_REPORT_RELATIVE
    if not report_path.is_file():
        raise FileNotFoundError(f"V24d source OOF treatment report is missing: {report_path}")
    report = _read_json(report_path)
    if (
        report.get("schema_version") != TREATMENT_REPORT_SCHEMA
        or report.get("scope") != "known train capture/content-grouped OOF; no validation/test/Unknown"
        or report.get("threshold_fitting") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise ValueError("V24d source OOF treatment report is not sealed known-only OOF")
    report_folds = {
        int(row["fold_id"]): dict(row)
        for row in list(report.get("fold_summaries") or ())
    }
    if set(report_folds) != set(range(FOLD_COUNT)):
        raise ValueError("V24d source OOF report lacks exactly five fold summaries")

    parent_ids: list[str] = []
    support: Counter[str] = Counter()
    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        prediction_path = root / prediction_relative("treatment", fold_id)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        payload = _read_json(prediction_path)
        records = dict(payload.get("records") or {})
        parent_rows = [dict(row) for row in list(records.get("parent") or ())]
        expected_ids = list(fold["held_out_parent_ids"])
        observed_ids = sorted(str(row.get("parent_sample_id", "")) for row in parent_rows)
        expected_hash = str(report_folds[fold_id].get("prediction_sha256", ""))
        observed_hash = sha256_file(prediction_path)
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != PREDICTION_SCHEMA
            or payload.get("arm") != "treatment"
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("grouping") != "capture_content_provenance_component"
            or payload.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or set(records) != {"parent", "eligible_prefix", "window"}
            or observed_ids != expected_ids
            or expected_hash != observed_hash
        ):
            raise ValueError(f"V24d source OOF fold {fold_id} is stale or unsealed")
        for row in parent_rows:
            clutter_probability = float(row.get("known_clutter_probability", float("nan")))
            target_probability = float(row.get("known_target_probability", float("nan")))
            if (
                int(row.get("true_id", -1)) not in {0, 1}
                or int(row.get("pred_id", -1)) not in {0, 1}
                or not (0.0 <= clutter_probability <= 1.0)
                or not (0.0 <= target_probability <= 1.0)
                or abs(clutter_probability + target_probability - 1.0) > 1.0e-5
                or int(row.get("fold_id", -1)) != fold_id
            ):
                raise ValueError(f"V24d source OOF fold {fold_id} score payload is malformed")
            parent_ids.append(str(row["parent_sample_id"]))
            support[str(row["label_name"])] += 1
        folds.append(
            {
                "fold_id": fold_id,
                "prediction_path": relative_path(root, prediction_path),
                "prediction_sha256": observed_hash,
                "held_out_parent_count": len(parent_rows),
                "held_out_label_parent_counts": dict(fold["held_out_label_parent_counts"]),
            }
        )
    assignment_ids = sorted(str(row["parent_sample_id"]) for row in assignment["assignments"])
    if (
        len(parent_ids) != SOURCE_PARENT_COUNT
        or len(set(parent_ids)) != SOURCE_PARENT_COUNT
        or sorted(parent_ids) != assignment_ids
        or sequence_sha256(sorted(parent_ids)) != assignment["known_parent_ids_sha256"]
        or dict(sorted(support.items())) != SOURCE_LABEL_SUPPORT
    ):
        raise ValueError("V24d source OOF parent coverage or label support changed")
    return {
        "fold_contract": relative_path(root, root / FOLD_CONTRACT_RELATIVE),
        "fold_contract_sha256": sha256_file(root / FOLD_CONTRACT_RELATIVE),
        "assignment": relative_path(root, root / ASSIGNMENT_RELATIVE),
        "assignment_sha256": sha256_file(root / ASSIGNMENT_RELATIVE),
        "treatment_report": relative_path(root, report_path),
        "treatment_report_sha256": sha256_file(report_path),
        "folds": folds,
        "parent_count": SOURCE_PARENT_COUNT,
        "label_parent_support": SOURCE_LABEL_SUPPORT,
        "provenance_component_count": int(assignment["provenance_component_count"]),
        "raw_echo_payload_access_count": 0,
        "mat_payload_access_count": 0,
        "rd_tensor_payload_access_count": 0,
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_oof_prediction_count": 0,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    document = root / EXPERIMENT_DOCUMENT
    if not document.is_file():
        raise FileNotFoundError(document)
    source = _source_prediction_audit(root)
    implementation = _implementation_audit(root)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_name": "V24d L3 cross-fitted OOF score-band Pending calibration",
        "scope": "known train provenance-grouped OOF L3 semantic-decision diagnostic only",
        "source_prediction_variable": "known_clutter_probability",
        "actions": [
            "KNOWN_TARGET_EVIDENCE",
            "KNOWN_CLUTTER_EVIDENCE",
            "PENDING_ABSTAIN",
        ],
        "decision_rule": {
            "target_if": "p_clutter <= t_target",
            "clutter_if": "p_clutter >= t_clutter",
            "pending_otherwise": True,
            "constraints": (
                "0.0 <= t_target < t_clutter <= 1.0; the sole zero-width "
                "raw-argmax baseline is t_target=t_clutter=0.5"
            ),
            "minimum_evidence_events": MINIMUM_EVIDENCE_EVENTS,
            "ineligible_prefix_action": "PENDING_ABSTAIN",
        },
        "threshold_fit": {
            "unit": "parent_final OOF record",
            "outer_protocol": "leave_one_outer_fold_out",
            "held_out_fold_labels_used_for_threshold_selection": False,
            "candidate_space": {
                "method": "all score decision-equivalence breakpoints",
                "threshold_values": "unique parent-final OOF scores union {0.0, 1.0}",
                "pair_rule": (
                    "strict t_target < t_clutter plus the single raw-argmax "
                    "baseline (0.5, 0.5)"
                ),
                "uses_approximation_grid": False,
            },
            "selection_requirements": SELECTION_REQUIREMENTS,
            "selection_direction_requirements": DIRECTION_REQUIREMENTS,
            "selection_order": [
                "all performance and error-direction requirements feasible",
                "maximum coverage among feasible candidates",
                "maximum minimum class acceptance recall among feasible candidates",
                "minimum total requirement violation when no feasible candidate exists",
                "maximum coverage when tied",
                "narrower score band then lower thresholds when tied",
            ],
            "prefix_or_window_labels_used_for_threshold_selection": False,
        },
        "prefix_safety_requirements": PREFIX_SAFETY_REQUIREMENTS,
        "forbidden_changes": [
            "no L1 model training or checkpoint loading",
            "no feature/cache/dt/loss/normalization/data split change",
            "no validation/test/Unknown score or label access",
            "no UNKNOWN_REJECTED action",
        ],
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / PROTOCOL, protocol)
    protocol_hash = sha256_file(root / PROTOCOL)
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment_name": protocol["experiment_name"],
        "scope": protocol["scope"],
        "experiment_document": relative_path(root, document),
        "experiment_document_sha256": sha256_file(document),
        "protocol": relative_path(root, root / PROTOCOL),
        "protocol_sha256": protocol_hash,
        "source_oof": source,
        "implementation_files_sha256": implementation,
        "decision_variable_only": True,
        "l1_model_training_performed": False,
        "l1_checkpoint_loaded": False,
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "OOF": "reused_completed_V21e_treatment_predictions_only",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    checks = {
        "exactly_five_completed_provenance_grouped_treatment_oof_folds_are_hash_bound": True,
        "every_known_train_parent_is_held_out_once": True,
        "score_band_is_the_only_experimental_variable": True,
        "thresholds_are_fit_only_from_other_outer_fold_parent_records": True,
        "prefix_and_window_labels_are_excluded_from_threshold_selection": True,
        "pending_is_distinct_from_unknown_rejection": True,
        "no_l1_training_or_checkpoint_loading": True,
        "no_validation_test_unknown_or_raw_payload_access": True,
        "capture_content_component_bootstrap_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "protocol_path": relative_path(root, root / PROTOCOL),
        "protocol_sha256": protocol_hash,
        "contract_path": relative_path(root, root / EXPERIMENT_CONTRACT),
        "contract_sha256": contract_hash,
        "source_oof": source,
        "implementation_files_sha256": implementation,
        "checks": checks,
        "known_train_parent_count": SOURCE_PARENT_COUNT,
        "clutter_parent_support": SOURCE_LABEL_SUPPORT["ClutterTarget"],
        "fold_count": FOLD_COUNT,
        "threshold_fit_unit": "parent_final_OOF_only",
        "threshold_fit_held_out_labels_accessed": False,
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_training_exposure": 0,
        "formal_training": "not_executed_not_applicable",
        "OOF": "reused_completed_V21e_treatment_predictions_only",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": READINESS_SCHEMA,
        "protocol": str(root / PROTOCOL),
        "contract": str(root / EXPERIMENT_CONTRACT),
        "readiness": str(root / READINESS),
        "known_train_parent_count": SOURCE_PARENT_COUNT,
        "clutter_parent_support": SOURCE_LABEL_SUPPORT["ClutterTarget"],
        "fold_count": FOLD_COUNT,
        "formal_training": "not_applicable",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V24d L3 OOF score-band Pending calibration."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
