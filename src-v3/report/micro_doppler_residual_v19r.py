from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_NAMES
from radar_ln3.micro_doppler_residual_v19r import (
    MicroDopplerLateResidual,
    V19R_CONFIG_NAME,
    V19R_CONFIG_RELATIVE_PATH,
    V19R_REPORT_RELATIVE_PATH,
    V19R_SCHEMA_VERSION,
    V19_CLASS_NAMES,
    V19_TRAINABLE_PARAMETER_COUNT,
    V19rResidualCacheDataset,
    evaluate_residual,
    load_v19r_residual_checkpoint,
    sha256_file,
    strip_evaluation_records,
)
from train.micro_doppler_residual_v19r import (
    _device,
    _load_runtime,
    _provenance,
    _resolve,
)


STRICT_EPSILON = 1.0e-12


def _metric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"V19r metric is non-finite: {name}={value}")
    return result


def _parent_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(evaluation.get("parent_metrics") or {})
    rows = list(source.get("per_class") or [])
    if tuple(str(row.get("class")) for row in rows) != V19_CLASS_NAMES:
        raise ValueError("V19r parent metrics use a different class order")
    matrix = np.asarray(
        source.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64
    )
    if matrix.shape != (3, 3):
        raise ValueError("V19r parent confusion matrix must be 3x3")
    per_class: dict[str, Any] = {}
    for class_id, row in enumerate(rows):
        class_name = V19_CLASS_NAMES[class_id]
        support = int(row.get("support", 0))
        if support <= 0 or int(matrix[class_id].sum()) != support:
            raise ValueError(f"V19r {class_name} support is invalid")
        per_class[class_name] = {
            "precision": _metric(row.get("precision"), f"{class_name}.precision"),
            "recall": _metric(row.get("recall"), f"{class_name}.recall"),
            "f1": _metric(row.get("f1"), f"{class_name}.f1"),
            "support": support,
            "correct_parent_count": int(matrix[class_id, class_id]),
        }
    return {
        "support": int(matrix.sum()),
        "accuracy": _metric(source.get("accuracy"), "accuracy"),
        "macro_precision": _metric(source.get("macro_precision"), "macro_precision"),
        "macro_recall": _metric(source.get("macro_recall"), "macro_recall"),
        "macro_f1": _metric(source.get("macro_f1"), "macro_f1"),
        "balanced_accuracy": _metric(
            source.get("balanced_accuracy"), "balanced_accuracy"
        ),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.astype(int).tolist(),
        "directional_errors": {
            "DroneTarget_to_BirdTarget": int(matrix[0, 1]),
            "BirdTarget_to_DroneTarget": int(matrix[1, 0]),
            "Drone_Bird_bidirectional_total": int(matrix[0, 1] + matrix[1, 0]),
            "DroneTarget_to_BalloonTarget": int(matrix[0, 2]),
            "BirdTarget_to_BalloonTarget": int(matrix[1, 2]),
            "BalloonTarget_to_DroneTarget": int(matrix[2, 0]),
            "BalloonTarget_to_BirdTarget": int(matrix[2, 1]),
        },
    }


def _class_floor(metrics: Mapping[str, Any], class_name: str) -> float:
    row = metrics["per_class"][class_name]
    return min(float(row["precision"]), float(row["recall"]), float(row["f1"]))


def _drone_bird_floor(metrics: Mapping[str, Any]) -> float:
    return min(
        _class_floor(metrics, "DroneTarget"),
        _class_floor(metrics, "BirdTarget"),
    )


def _strictly_exceeds(value: float, threshold: float) -> bool:
    return float(value) > float(threshold) + STRICT_EPSILON


def _comparison(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_errors = baseline["directional_errors"]
    treatment_errors = treatment["directional_errors"]
    delta = {
        "accuracy": float(treatment["accuracy"] - baseline["accuracy"]),
        "macro_f1": float(treatment["macro_f1"] - baseline["macro_f1"]),
        "drone_bird_weak_floor": float(
            _drone_bird_floor(treatment) - _drone_bird_floor(baseline)
        ),
        "DroneTarget_floor": float(
            _class_floor(treatment, "DroneTarget")
            - _class_floor(baseline, "DroneTarget")
        ),
        "BirdTarget_floor": float(
            _class_floor(treatment, "BirdTarget")
            - _class_floor(baseline, "BirdTarget")
        ),
        "BalloonTarget_floor": float(
            _class_floor(treatment, "BalloonTarget")
            - _class_floor(baseline, "BalloonTarget")
        ),
        "Drone_to_Bird_error_reduction": int(
            baseline_errors["DroneTarget_to_BirdTarget"]
            - treatment_errors["DroneTarget_to_BirdTarget"]
        ),
        "Bird_to_Drone_error_reduction": int(
            baseline_errors["BirdTarget_to_DroneTarget"]
            - treatment_errors["BirdTarget_to_DroneTarget"]
        ),
        "Drone_Bird_bidirectional_error_reduction": int(
            baseline_errors["Drone_Bird_bidirectional_total"]
            - treatment_errors["Drone_Bird_bidirectional_total"]
        ),
    }
    return {
        "frozen_v18r": baseline,
        "v19r_late_residual": treatment,
        "v19r_minus_frozen_v18r": delta,
    }


def _prediction_changes(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_rows = {
        str(row["parent_sample_id"]): dict(row)
        for row in dict(baseline.get("records") or {}).get("parent") or []
    }
    treatment_rows = {
        str(row["parent_sample_id"]): dict(row)
        for row in dict(treatment.get("records") or {}).get("parent") or []
    }
    if set(baseline_rows) != set(treatment_rows):
        raise ValueError("V19r baseline/treatment validation parents differ")
    changed: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for parent_id in sorted(baseline_rows):
        base = baseline_rows[parent_id]
        value = treatment_rows[parent_id]
        if int(base["true_id"]) != int(value["true_id"]):
            raise ValueError("V19r baseline/treatment truth differs")
        if int(base["pred_id"]) == int(value["pred_id"]):
            continue
        true_id = int(base["true_id"])
        baseline_correct = int(base["pred_id"]) == true_id
        treatment_correct = int(value["pred_id"]) == true_id
        category = (
            "corrected"
            if not baseline_correct and treatment_correct
            else "harmed"
            if baseline_correct and not treatment_correct
            else "wrong_class_changed"
        )
        categories[category] += 1
        changed.append(
            {
                "parent_sample_id": parent_id,
                "track_id": str(value["track_id"]),
                "true_class": V19_CLASS_NAMES[true_id],
                "baseline_prediction": V19_CLASS_NAMES[int(base["pred_id"])],
                "v19r_prediction": V19_CLASS_NAMES[int(value["pred_id"])],
                "category": category,
                "baseline_confidence": float(base["confidence"]),
                "v19r_confidence": float(value["confidence"]),
            }
        )
    return {
        "changed_parent_count": len(changed),
        "category_counts": dict(sorted(categories.items())),
        "records": changed,
    }


def run(
    project_root: Path,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    runtime = _load_runtime(root, config_path.resolve())
    config = runtime["config"]
    train_config = dict(config["train"])
    artifact = _resolve(root, train_config["artifact_dir"])
    summary_path = artifact / "training-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("config_name") != V19R_CONFIG_NAME
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("epochs", -1)) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or int(summary.get("trainable_parameter_count", -1))
        != V19_TRAINABLE_PARAMETER_COUNT
    ):
        raise ValueError("V19r training summary violates completion/scope discipline")
    history = list(summary.get("history") or [])
    if [int(row.get("epoch", -1)) for row in history] != list(range(1, 21)):
        raise ValueError("V19r training summary must contain exactly epochs 1..20")
    device = _device(train_config["device"])
    validation_dataset = V19rResidualCacheDataset(
        runtime["resolved"]["validation_cache_path"]
    )
    baseline_evaluation = evaluate_residual(
        None,
        validation_dataset,
        device=device,
        batch_size=int(train_config["batch_size"]),
        prefix_auxiliary_min_events=int(train_config["prefix_auxiliary_min_events"]),
    )
    contract = json.loads(
        runtime["resolved"]["contract_path"].read_text(encoding="utf-8")
    )
    if _parent_metrics(baseline_evaluation) != _parent_metrics(
        {"parent_metrics": contract["baseline_validation_parent_metrics"]}
    ):
        raise ValueError("V19r independent frozen baseline differs from prepare contract")
    provenance = _provenance(runtime)
    model = MicroDopplerLateResidual().to(device)
    epoch_results: dict[str, Any] = {}
    baseline_metrics = _parent_metrics(baseline_evaluation)
    for epoch in range(1, 21):
        checkpoint_path = _resolve(
            root, train_config["epoch_checkpoint_dir"]
        ) / f"epoch-{epoch:03d}.pt"
        payload = load_v19r_residual_checkpoint(
            checkpoint_path,
            model,
            expected_config_name=V19R_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        if int(payload.get("epoch", -1)) != epoch:
            raise ValueError(f"V19r epoch checkpoint metadata differs: {checkpoint_path}")
        evaluation = evaluate_residual(
            model,
            validation_dataset,
            device=device,
            batch_size=int(train_config["batch_size"]),
            prefix_auxiliary_min_events=int(
                train_config["prefix_auxiliary_min_events"]
            ),
        )
        if strip_evaluation_records(evaluation) != history[epoch - 1]["validation"]:
            raise ValueError(f"V19r epoch {epoch} independent validation differs")
        epoch_metrics = _parent_metrics(evaluation)
        epoch_results[str(epoch)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parent_metrics": epoch_metrics,
            "comparison_to_frozen_v18r": _comparison(
                baseline_metrics, epoch_metrics
            )["v19r_minus_frozen_v18r"],
            "residual_contract": evaluation["residual_contract"],
        }
    best_epoch = int(summary["best_epoch"])
    best_path = _resolve(root, train_config["checkpoint_path"])
    best_payload = load_v19r_residual_checkpoint(
        best_path,
        model,
        expected_config_name=V19R_CONFIG_NAME,
        expected_config_sha256=runtime["config_sha256"],
        expected_provenance=provenance,
        map_location=device,
    )
    if int(best_payload.get("epoch", -1)) != best_epoch:
        raise ValueError("V19r best checkpoint epoch differs from summary")
    best_evaluation = evaluate_residual(
        model,
        validation_dataset,
        device=device,
        batch_size=int(train_config["batch_size"]),
        prefix_auxiliary_min_events=int(train_config["prefix_auxiliary_min_events"]),
    )
    if strip_evaluation_records(best_evaluation) != summary["best_validation_metrics"]:
        raise ValueError("V19r best checkpoint independent validation differs")
    treatment_metrics = _parent_metrics(best_evaluation)
    comparison = _comparison(baseline_metrics, treatment_metrics)
    delta = comparison["v19r_minus_frozen_v18r"]
    adoption = dict(config["adoption"])
    class_checks = {
        class_name: {
            "floor": _class_floor(treatment_metrics, class_name),
            "target": float(adoption["minimum_per_class_precision_recall_f1"]),
            "passed": _class_floor(treatment_metrics, class_name)
            >= float(adoption["minimum_per_class_precision_recall_f1"]),
        }
        for class_name in V19_CLASS_NAMES
    }
    checks = {
        "all_v19r_classes_precision_recall_f1_at_least_0_95": all(
            row["passed"] for row in class_checks.values()
        ),
        "parent_macro_f1_improves_by_more_than_0_02": _strictly_exceeds(
            delta["macro_f1"],
            float(adoption["strict_macro_f1_improvement_over_frozen_v18r"]),
        ),
        "drone_bird_weak_floor_improves_by_more_than_0_02": _strictly_exceeds(
            delta["drone_bird_weak_floor"],
            float(
                adoption[
                    "strict_drone_bird_weak_floor_improvement_over_frozen_v18r"
                ]
            ),
        ),
        "drone_to_bird_not_increased": delta["Drone_to_Bird_error_reduction"] >= 0,
        "bird_to_drone_not_increased": delta["Bird_to_Drone_error_reduction"] >= 0,
        "bidirectional_drone_bird_errors_reduced": (
            delta["Drone_Bird_bidirectional_error_reduction"]
            >= int(adoption["minimum_bidirectional_error_reduction"])
        ),
        "balloon_floor_not_decreased": delta["BalloonTarget_floor"] >= 0.0,
        "frozen_v18r_checkpoint_file_hash_unchanged": (
            sha256_file(runtime["resolved"]["baseline_checkpoint_path"])
            == config["expected_baseline_checkpoint_sha256"]
        ),
        "all_20_parent_orders_match_frozen_v18r": all(
            str(history[epoch - 1]["parent_order_sha256"])
            == runtime["source_orders"][str(epoch)]
            for epoch in range(1, 21)
        ),
        "residual_parameter_count_is_24": (
            model.trainable_parameter_count == V19_TRAINABLE_PARAMETER_COUNT
        ),
        "validation_threshold_tuning_not_executed": (
            summary.get("validation_threshold_tuning") == "not_executed"
        ),
    }
    passed = all(checks.values())
    weight = model.linear.weight.detach().cpu().numpy()
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": (
            "RADAR-LN3-V19r-v18r-micro-doppler-late-residual-validation-v1"
        ),
        "experiment_schema_version": V19R_SCHEMA_VERSION,
        "scope": "DBB specialist validation parent-final local task only",
        "single_factor": (
            "add the V19 zero-initialized bounded zero-sum 24-weight stable8 "
            "late residual to the frozen V18r best checkpoint"
        ),
        "fixed_baseline": {
            "name": "V18r density+stable8 best",
            "epoch": int(contract["source_v18r"]["checkpoint_epoch"]),
            "checkpoint": str(runtime["resolved"]["baseline_checkpoint_path"]),
            "checkpoint_sha256": config["expected_baseline_checkpoint_sha256"],
            "backbone_state_sha256": config[
                "expected_baseline_backbone_state_sha256"
            ],
            "retrained_for_v19r": False,
        },
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "checks": checks,
        "class_threshold_checks": class_checks,
        "primary_comparison": comparison,
        "epoch_comparisons_to_fixed_baseline": epoch_results,
        "prediction_changes": _prediction_changes(
            baseline_evaluation, best_evaluation
        ),
        "residual_weights_rows_classes_columns_features": {
            class_name: {
                feature_name: float(weight[class_id, feature_id])
                for feature_id, feature_name in enumerate(
                    BROADBAND_MICRO_DOPPLER_NAMES
                )
            }
            for class_id, class_name in enumerate(V19_CLASS_NAMES)
        },
        "residual_contract": best_evaluation["residual_contract"],
        "trainable_parameter_count": model.trainable_parameter_count,
        "training_summary": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config": str(config_path),
        "config_sha256": runtime["config_sha256"],
        "conclusion": (
            "V19r passes the frozen single-seed local DBB adoption gate; multi-seed confirmation is required before any cascade or open-set claim."
            if passed
            else "V19r does not pass the frozen local DBB adoption gate; retain V18r as the fixed baseline and do not advance this residual to cascade/open-set claims."
        ),
        "not_final_five_class_performance": True,
        "formal_training": "已执行（仅当V19r完成摘要通过完整性校验时）",
        "validation_threshold_tuning": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / (output_path or V19R_REPORT_RELATIVE_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the completed V19r late-residual DBB experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", type=Path, default=V19R_CONFIG_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output
    if output is not None and not output.is_absolute():
        output = root / output
    run(root, config_path, output)


if __name__ == "__main__":
    main()
