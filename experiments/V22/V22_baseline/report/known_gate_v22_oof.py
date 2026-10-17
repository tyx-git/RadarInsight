"""Compare V22 fixed-denominator OOF predictions with frozen V21e Control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES, summarize_known_gate_records
from radar_ln3.known_gate_oof import load_fold_contract, sequence_sha256, sha256_file
from radar_ln3.known_gate_v22_oof import (
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    PAIRED_REPORT_RELATIVE,
    PAIRED_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    prediction_relative,
)
from report.known_gate_oof import PRIMARY_DELTA, _base_label_outcomes, _component_bootstrap, _delta, _load_arm as load_v21e_arm


def _fold_binding(contract: dict[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [row for row in contract["folds"] if int(row["fold_id"]) == fold_id]
    if len(rows) != 1:
        raise ValueError(f"V22 fold binding is unavailable or repeated: {fold_id}")
    return dict(rows[0])


def _load_treatment(root: Path) -> dict[str, Any]:
    readiness_path = root / READINESS_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != CONTRACT_SCHEMA
    ):
        raise ValueError("V22 readiness/contract is invalid or stale")
    for source_name in (
        "source_fold_contract",
        "source_assignment",
        "source_control_report",
    ):
        source_path = root / contract[f"{source_name}_path"]
        if (
            not source_path.is_file()
            or sha256_file(source_path) != contract[f"{source_name}_sha256"]
        ):
            raise ValueError(f"V22 frozen {source_name} binding changed")
    source_contract, assignment = load_fold_contract(root)
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        binding = _fold_binding(contract, fold_id)
        control_prediction = root / dict(binding["control"])["prediction_path"]
        if (
            not control_prediction.is_file()
            or sha256_file(control_prediction)
            != dict(binding["control"])["prediction_sha256"]
        ):
            raise ValueError(f"V22 fold {fold_id} frozen Control prediction changed")
        path = root / prediction_relative(fold_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = dict(payload.get("records") or {})
        evaluation = dict(payload.get("evaluation") or {})
        history = list(payload.get("fold_training_history") or [])
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records.get("parent") or [])
        source_fold = next(
            row for row in source_contract["folds"] if int(row["fold_id"]) == fold_id
        )
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != PREDICTION_SCHEMA
            or payload.get("experiment_name") != "V22 fixed parent balanced OOF"
            or payload.get("arm") != "treatment"
            or int(payload.get("fold_id", -1)) != fold_id
            or dict(payload.get("loss_aggregation") or {}).get(
                "class_weight_normalization"
            )
            != "fixed_parent_count"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or payload.get("config_sha256")
            != dict(binding["treatment"])["config_sha256"]
            or payload.get("config_identity_sha256")
            != dict(binding["treatment"])["config_identity_sha256"]
            or set(records) != set(records_by_scope)
            or len(history) != 20
            or [int(row["epoch"]) for row in history] != list(range(1, 21))
            or sequence_sha256(parent_ids) != binding["held_out_parent_ids_sha256"]
            or sequence_sha256(parent_ids) != source_fold["held_out_parent_ids_sha256"]
        ):
            raise ValueError(f"V22 fold {fold_id} prediction is invalid")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "held_out_parent_count": len(parent_ids),
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "parent_metrics": evaluation["parent_metrics"],
                "eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "paired_baseline_reference_state_sha256": payload[
                    "paired_baseline_reference_state_sha256"
                ],
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
                "pairing": payload["paired_initialization"],
            }
        )
    parent_ids = [str(row["parent_sample_id"]) for row in records_by_scope["parent"]]
    if len(parent_ids) != 986 or len(set(parent_ids)) != 986:
        raise ValueError("V22 does not cover every known train parent exactly once")
    return {
        "contract": contract,
        "assignment": assignment,
        "records_by_scope": records_by_scope,
        "fold_summaries": fold_summaries,
        "metrics_by_scope": {
            scope: summarize_known_gate_records(rows, scope=f"oof_v22_{scope}")
            for scope, rows in records_by_scope.items()
        },
    }


def _class_metric(metrics: dict[str, Any], class_name: str, field: str) -> float:
    row = next(row for row in metrics["per_class"] if row["class"] == class_name)
    return float(row[field])


def _fold_dispersion(rows: list[dict[str, Any]]) -> dict[str, float]:
    class_name = KNOWN_GATE_CLASS_NAMES[1]
    values = {
        field: [
            _class_metric(dict(row["parent_metrics"]), class_name, field)
            for row in rows
        ]
        for field in ("precision", "recall", "f1")
    }
    return {f"known_clutter_{field}_population_std": pstdev(data) for field, data in values.items()}


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    treatment = _load_treatment(root)
    control = load_v21e_arm(root, "treatment")
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        pairing = dict(after["pairing"])
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"] != after["epoch_parent_order_hashes"]
            or before["initial_model_state_sha256"]
            != after["paired_baseline_reference_state_sha256"]
            or pairing.get("loss_only_pair") is not True
            or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
            or int(pairing.get("new_fusion_columns", -1)) != 0
            or pairing.get("initial_logits_exactly_equal") is not True
        ):
            raise ValueError(f"V22 fold {fold_id} is not strictly paired to V21e")
    deltas = {
        scope: _delta(
            control["metrics_by_scope"][scope],
            treatment["metrics_by_scope"][scope],
        )
        for scope in ("parent", "window", "eligible_prefix")
    }
    parent_outcomes = _base_label_outcomes(
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
        scope="parent",
    )
    prefix_outcomes = _base_label_outcomes(
        control["records_by_scope"]["eligible_prefix"],
        treatment["records_by_scope"]["eligible_prefix"],
        scope="eligible_prefix",
    )
    bootstrap = _component_bootstrap(
        treatment["assignment"],
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
    )
    control_dispersion = _fold_dispersion(control["fold_summaries"])
    treatment_dispersion = _fold_dispersion(treatment["fold_summaries"])
    dispersion_delta = {
        key: treatment_dispersion[key] - control_dispersion[key]
        for key in control_dispersion
    }
    parent_delta = deltas["parent"]
    base_safe = all(
        int(parent_outcomes[label]["net_rescue"]) >= 0
        for label in ("DroneTarget", "BirdTarget", "BalloonTarget")
    )
    clutter_all_non_decreasing = all(
        parent_delta[f"known_clutter_{field}"] >= 0.0
        for field in ("precision", "recall", "f1")
    )
    clutter_all_strictly_improved = all(
        parent_delta[f"known_clutter_{field}"] > 0.0
        for field in ("precision", "recall", "f1")
    )
    known_target_safe = parent_delta["known_target_recall"] >= 0.0 and base_safe
    prefix_safe = (
        deltas["eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["eligible_prefix"]["known_clutter_recall"] >= 0.0
        and deltas["eligible_prefix"]["macro_f1"] >= 0.0
    )
    variance_non_increasing = all(
        treatment_dispersion[key] <= control_dispersion[key]
        for key in control_dispersion
    )
    variance_reduced = variance_non_increasing and any(
        treatment_dispersion[key] < control_dispersion[key]
        for key in control_dispersion
    )
    robust_macro = (
        float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"])
        > 0.0
    )
    robust_clutter_precision_recall = all(
        float(bootstrap["treatment_minus_control"][metric]["lower_95"]) > 0.0
        for metric in ("known_clutter_precision", "known_clutter_recall")
    )
    hypothesis_supported = (
        clutter_all_strictly_improved
        and known_target_safe
        and prefix_safe
        and variance_reduced
        and robust_clutter_precision_recall
        and robust_macro
    )
    adoption_signal = hypothesis_supported and parent_delta["macro_f1"] > PRIMARY_DELTA
    if adoption_signal:
        status = "adoption_candidate_signal"
    elif hypothesis_supported:
        status = "mechanism_supported_below_2pp_adoption_threshold"
    else:
        status = "hypothesis_not_supported"
    clutter_support_by_fold = [
        int(
            next(
                row
                for row in fold["parent_metrics"]["per_class"]
                if row["class"] == KNOWN_GATE_CLASS_NAMES[1]
            )["support"]
        )
        for fold in control["fold_summaries"]
    ]
    report = {
        "status": status,
        "schema_version": PAIRED_REPORT_SCHEMA,
        "experiment": "V22 fixed-parent-denominator loss-only paired OOF",
        "scope": "known train capture/content-grouped OOF only; Unknown/validation/test excluded",
        "control": "frozen V21e OOF Treatment with batch_weight_sum",
        "treatment": "V22 OOF with fixed_parent_count",
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "paired_component_bootstrap": bootstrap,
        "fold_clutter_dispersion": {
            "control": control_dispersion,
            "treatment": treatment_dispersion,
            "treatment_minus_control": dispersion_delta,
            "support_per_fold": clutter_support_by_fold,
            "warning": "five-fold dispersion is descriptive with only 6-7 Clutter parents per fold",
        },
        "decision": {
            "clutter_parent_precision_recall_f1_all_non_decreasing": clutter_all_non_decreasing,
            "clutter_parent_precision_recall_f1_all_strictly_improved": clutter_all_strictly_improved,
            "known_target_and_drone_bird_balloon_recall_safe": known_target_safe,
            "eligible_prefix_guard_passed": prefix_safe,
            "clutter_fold_precision_recall_f1_dispersion_all_non_increasing": variance_non_increasing,
            "clutter_fold_dispersion_at_least_one_strictly_reduced": variance_reduced,
            "component_bootstrap_macro_f1_lower_95_above_zero": robust_macro,
            "component_bootstrap_clutter_precision_recall_lower_95_above_zero": robust_clutter_precision_recall,
            "fixed_denominator_hypothesis_supported": hypothesis_supported,
            "parent_macro_f1_delta_strictly_above_0_02": (
                parent_delta["macro_f1"] > PRIMARY_DELTA
            ),
            "adoption_candidate_signal": adoption_signal,
            "advance_to_l3_calibration": False,
            "reason_l3_remains_blocked": (
                "KnownGate loss OOF cannot substitute for Unknown open-set and semantic-decision-layer calibration"
            ),
            "online_known_gate_adoption_allowed": False,
        },
        "interpretation_contract": {
            "loss_bug_mechanism_fixed": True,
            "performance_hypothesis_requires_this_completed_report": True,
            "unknown_is_not_a_training_class": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "pending_abstain_is_not_directly_supervised": True,
        },
        "test": "未执行",
        "threshold_fitting": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / PAIRED_REPORT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V22 KnownGate OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()