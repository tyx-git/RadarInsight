from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.micro_doppler_joint_residual_v19r import _build_initial_model
from radar_ln3.micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_NAMES
from radar_ln3.micro_doppler_joint_residual_v19r import (
    V19R_JOINT_CONFIG_NAME,
    V19R_JOINT_CONFIG_RELATIVE_PATH,
    V19R_JOINT_REPORT_RELATIVE_PATH,
    V19R_JOINT_SCHEMA_VERSION,
    load_v19r_joint_checkpoint,
    sha256_file,
)
from radar_ln3.micro_doppler_residual import V19_CLASS_NAMES
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
)
from report.micro_doppler_residual_v19r import (
    _class_floor,
    _drone_bird_floor,
    _parent_metrics,
    _prediction_changes,
    _strictly_exceeds,
)
from train.micro_doppler_joint_residual_v19r import (
    _device,
    _load_runtime,
    _provenance,
    _resolve,
    _strip_records,
)


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
        "v18r_control_reference": baseline,
        "v19r_full_joint_treatment": treatment,
        "treatment_minus_control": delta,
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
    treatment_config = runtime["v18r"]["config"]
    artifact = _resolve(root, train_config["artifact_dir"])
    summary_path = artifact / "training-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_backbone_count = int(config["model"]["backbone_trainable_parameter_count"])
    expected_total_count = int(config["model"]["total_trainable_parameter_count"])
    if (
        summary.get("status") != "completed"
        or summary.get("config_name") != V19R_JOINT_CONFIG_NAME
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("epochs", -1)) != 20
        or int(summary.get("backbone_trainable_parameter_count", -1))
        != expected_backbone_count
        or int(summary.get("residual_trainable_parameter_count", -1)) != 24
        or int(summary.get("total_trainable_parameter_count", -1))
        != expected_total_count
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V19r joint summary violates full-training or sealed scope")
    history = list(summary.get("history") or [])
    if [int(row.get("epoch", -1)) for row in history] != list(range(1, 21)):
        raise ValueError("V19r joint summary must contain epochs 1..20")
    device = _device(train_config["device"])
    validation_dataset = build_task_dataset(
        root,
        treatment_config,
        split_name=treatment_config.data.validation_split_name,
    )

    baseline_model = LiquidTaskClassifier(treatment_config).to(device)
    baseline_payload = load_task_checkpoint(
        runtime["v18r"]["checkpoint_path"],
        baseline_model,
        expected_schema=treatment_config.train.checkpoint_schema,
        map_location=device,
    )
    if int(baseline_payload.get("epoch", -1)) != 15:
        raise ValueError("V19r joint control reference is not V18r best epoch 15")
    baseline_evaluation = evaluate_task(
        baseline_model,
        validation_dataset,
        config=treatment_config,
        device=device,
    )
    expected_baseline = dict(
        dict(runtime["v18r"]["summary"]["best_validation_metrics"])
    )
    if _strip_records(baseline_evaluation) != expected_baseline:
        raise ValueError("V19r joint independent V18r control evaluation differs")

    model, _, _ = _build_initial_model(root, runtime["v18r"], runtime["v19"])
    model.to(device)
    provenance = _provenance(runtime)
    baseline_metrics = _parent_metrics(baseline_evaluation)
    epoch_results: dict[str, Any] = {}
    for epoch in range(1, 21):
        checkpoint_path = _resolve(
            root, train_config["epoch_checkpoint_dir"]
        ) / f"epoch-{epoch:03d}.pt"
        payload = load_v19r_joint_checkpoint(
            checkpoint_path,
            model,
            expected_config_name=V19R_JOINT_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        if int(payload.get("epoch", -1)) != epoch:
            raise ValueError(f"V19r joint epoch metadata differs: {checkpoint_path}")
        evaluation = evaluate_task(
            model,
            validation_dataset,
            config=treatment_config,
            device=device,
        )
        if _strip_records(evaluation) != history[epoch - 1]["validation"]:
            raise ValueError(f"V19r joint epoch {epoch} validation differs")
        metrics = _parent_metrics(evaluation)
        epoch_results[str(epoch)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parent_metrics": metrics,
            "comparison_to_v18r": _comparison(
                baseline_metrics, metrics
            )["treatment_minus_control"],
            "backbone_gradient_absolute_max": history[epoch - 1][
                "backbone_gradient_absolute_max"
            ],
            "residual_gradient_absolute_max": history[epoch - 1][
                "residual_gradient_absolute_max"
            ],
        }
    best_epoch = int(summary["best_epoch"])
    best_path = _resolve(root, train_config["checkpoint_path"])
    best_payload = load_v19r_joint_checkpoint(
        best_path,
        model,
        expected_config_name=V19R_JOINT_CONFIG_NAME,
        expected_config_sha256=runtime["config_sha256"],
        expected_provenance=provenance,
        map_location=device,
    )
    if int(best_payload.get("epoch", -1)) != best_epoch:
        raise ValueError("V19r joint best checkpoint epoch differs")
    best_evaluation = evaluate_task(
        model,
        validation_dataset,
        config=treatment_config,
        device=device,
    )
    if _strip_records(best_evaluation) != summary["best_validation_metrics"]:
        raise ValueError("V19r joint best checkpoint independent validation differs")
    treatment_metrics = _parent_metrics(best_evaluation)
    comparison = _comparison(baseline_metrics, treatment_metrics)
    delta = comparison["treatment_minus_control"]
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
        "all_classes_precision_recall_f1_at_least_0_95": all(
            row["passed"] for row in class_checks.values()
        ),
        "parent_macro_f1_improves_by_more_than_0_02": _strictly_exceeds(
            delta["macro_f1"],
            float(adoption["strict_macro_f1_improvement_over_v18r"]),
        ),
        "drone_bird_weak_floor_improves_by_more_than_0_02": _strictly_exceeds(
            delta["drone_bird_weak_floor"],
            float(adoption["strict_drone_bird_weak_floor_improvement_over_v18r"]),
        ),
        "drone_to_bird_not_increased": delta["Drone_to_Bird_error_reduction"] >= 0,
        "bird_to_drone_not_increased": delta["Bird_to_Drone_error_reduction"] >= 0,
        "bidirectional_drone_bird_errors_reduced": (
            delta["Drone_Bird_bidirectional_error_reduction"]
            >= int(adoption["minimum_bidirectional_error_reduction"])
        ),
        "balloon_floor_not_decreased": delta["BalloonTarget_floor"] >= 0.0,
        "complete_backbone_was_trainable": expected_backbone_count > 24,
        "total_trainable_parameter_count_matches": (
            model.trainable_parameter_count == expected_total_count
        ),
        "all_epochs_have_nonzero_backbone_gradient": all(
            float(row["backbone_gradient_absolute_max"]) > 0.0 for row in history
        ),
        "all_epochs_have_nonzero_residual_gradient": all(
            float(row["residual_gradient_absolute_max"]) > 0.0 for row in history
        ),
        "v18r_control_checkpoint_hash_unchanged": (
            sha256_file(runtime["v18r"]["checkpoint_path"])
            == config["expected_v18r_best_checkpoint_sha256"]
        ),
        "all_20_parent_orders_match_v18r": all(
            str(history[epoch - 1]["parent_order_sha256"])
            == runtime["source_orders"][str(epoch)]
            for epoch in range(1, 21)
        ),
        "validation_threshold_tuning_not_executed": (
            summary.get("validation_threshold_tuning") == "not_executed"
        ),
    }
    passed = all(checks.values())
    weights = model.residual.linear.weight.detach().cpu().numpy()
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": (
            "RADAR-LN3-V19r-stable8-density-joint-late-residual-validation-v1"
        ),
        "experiment_schema_version": V19R_JOINT_SCHEMA_VERSION,
        "scope": "DBB specialist full from-scratch joint training validation",
        "single_factor": (
            "V18r full backbone and zero-init V19 late residual trained jointly from "
            "the same V18r initial state and parent order"
        ),
        "control": {
            "name": "V18r density+stable8 best",
            "retrained": False,
            "best_epoch": 15,
            "checkpoint": str(runtime["v18r"]["checkpoint_path"]),
            "checkpoint_sha256": config["expected_v18r_best_checkpoint_sha256"],
        },
        "treatment": {
            "name": "V19r full joint stable8+density+late-residual",
            "trained_from_scratch": True,
            "backbone_trainable_parameter_count": expected_backbone_count,
            "residual_trainable_parameter_count": 24,
            "total_trainable_parameter_count": expected_total_count,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
        },
        "checks": checks,
        "class_threshold_checks": class_checks,
        "primary_comparison": comparison,
        "epoch_comparisons_to_v18r": epoch_results,
        "prediction_changes": _prediction_changes(
            baseline_evaluation, best_evaluation
        ),
        "residual_weights_rows_classes_columns_features": {
            class_name: {
                feature_name: float(weights[class_id, feature_id])
                for feature_id, feature_name in enumerate(
                    BROADBAND_MICRO_DOPPLER_NAMES
                )
            }
            for class_id, class_name in enumerate(V19_CLASS_NAMES)
        },
        "training_summary": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config": str(config_path),
        "config_sha256": runtime["config_sha256"],
        "conclusion": (
            "V19r full joint treatment passes the single-seed local DBB gate; multi-seed confirmation is required."
            if passed
            else "V19r full joint treatment does not pass every frozen local DBB gate; do not advance it to cascade/open-set claims."
        ),
        "not_final_five_class_performance": True,
        "formal_training": "已执行（仅当完整20 epoch摘要通过复核时）",
        "validation_threshold_tuning": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / (output_path or V19R_JOINT_REPORT_RELATIVE_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate corrected V19r full joint late-residual training."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", type=Path, default=V19R_JOINT_CONFIG_RELATIVE_PATH)
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
