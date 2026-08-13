from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
    verify_task_data_contract,
)


ARMS = ("control", "treatment")
EXPECTED_EPOCHS = 20
CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
MIN_ABSOLUTE_IMPROVEMENT = 0.02
STRICT_IMPROVEMENT_EPSILON = 1.0e-12
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler-paired-validation.json"
)


def config_path(arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
        f"rd-physical4-broadband-multiscale-micro-doppler39-{arm}-seed42-config.json"
    )


def _metric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"V18 paired metric is not finite: {name}={value}")
    return result


def _strictly_exceeds_adoption_gain(value: float) -> bool:
    return float(value) > MIN_ABSOLUTE_IMPROVEMENT + STRICT_IMPROVEMENT_EPSILON


def _parent_metrics(validation: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(validation.get("parent_metrics") or {})
    source_rows = list(source.get("per_class") or [])
    if tuple(str(row.get("class")) for row in source_rows) != CLASS_NAMES:
        raise ValueError(
            f"V18 paired confusion matrix requires class order {CLASS_NAMES}"
        )
    rows = {str(row["class"]): dict(row) for row in source_rows}
    if set(rows) != set(CLASS_NAMES):
        raise ValueError(f"V18 paired metrics require {CLASS_NAMES}")
    matrix = np.asarray(
        source.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64
    )
    if matrix.shape != (3, 3):
        raise ValueError("V18 paired parent confusion matrix must be 3x3")
    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = rows[class_name]
        support = int(row.get("support", 0))
        if support <= 0 or int(matrix[class_id].sum()) != support:
            raise ValueError(f"V18 {class_name} support/confusion matrix is invalid")
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


def _epoch_metrics(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for record in summary.get("history") or []:
        epoch = int(record.get("epoch", -1))
        if epoch in rows:
            raise ValueError(f"V18 training summary repeats epoch {epoch}")
        rows[epoch] = {
            "epoch": epoch,
            "global_step": int(record.get("global_step", -1)),
            "train_loss": _metric(record.get("train_loss"), f"epoch{epoch}.train_loss"),
            "train_main_loss": _metric(
                record.get("train_main_loss"), f"epoch{epoch}.train_main_loss"
            ),
            "train_prefix_auxiliary_loss": _metric(
                record.get("train_prefix_auxiliary_loss"),
                f"epoch{epoch}.train_prefix_auxiliary_loss",
            ),
            "parent_order_sha256": str(record.get("parent_order_sha256")),
            "v18_training_readiness_sha256": str(
                record.get("v18_training_readiness_sha256")
            ),
            "parent_metrics": _parent_metrics(dict(record.get("validation") or {})),
        }
    if set(rows) != set(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError("V18 training summary must contain exactly epochs 1..20")
    return rows


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    path = root / config_path(arm)
    config = load_task_config(path)
    if (
        config.schema_version
        != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or config.train.paired_protocol != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        or config.train.paired_arm != arm
        or config.train.seed != 42
        or config.train.epochs != EXPECTED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.checkpoint_selection_metric
        != BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
        or config.task.name != "dbb_specialist"
    ):
        raise ValueError(f"V18 {arm} config violates the frozen paired protocol")
    contract = verify_task_data_contract(root, config)
    artifact = root / config.train.formal_artifact_dir
    summary_path = artifact / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"V18 {arm} training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("config_name") != config.config_name
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("global_step", 0)) <= 0
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError(f"V18 {arm} summary violates completion/scope discipline")
    epochs = _epoch_metrics(summary)
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError(f"V18 {arm} best epoch is outside 1..20")
    best_metrics = _parent_metrics(dict(summary.get("best_validation_metrics") or {}))
    if best_metrics != epochs[best_epoch]["parent_metrics"]:
        raise ValueError(f"V18 {arm} best metrics differ from its history epoch")
    best_path = root / config.train.checkpoint_path
    last_path = root / config.train.last_checkpoint_path
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(f"V18 {arm} best/last checkpoint is incomplete")
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    missing = [
        str(epoch_dir / f"epoch-{epoch:03d}.pt")
        for epoch in range(1, 21)
        if not (epoch_dir / f"epoch-{epoch:03d}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"V18 {arm} epoch checkpoints are incomplete: {missing[:3]}")
    initialization = dict(summary.get("paired_initialization") or {})
    if (
        initialization.get("shared_parameters_and_buffers_exactly_equal") is not True
        or initialization.get("treatment_new_fusion_columns_exactly_zero") is not True
        or initialization.get("initial_cpu_reference_logits_exactly_equal") is not True
        or int(initialization.get("shared_rd_physical_columns", -1)) != 4
        or int(initialization.get("treatment_new_fusion_columns", -1)) != 39
    ):
        raise ValueError(f"V18 {arm} lacks valid paired-initialization evidence")
    readiness = dict(summary.get("v18_training_readiness") or {})
    readiness_path = Path(str(readiness.get("path", "")))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V18-broadband-multiscale-micro-doppler-training-readiness-v1"
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != readiness.get("sha256")
    ):
        raise ValueError(f"V18 {arm} summary lacks an immutable readiness binding")
    if any(
        row["v18_training_readiness_sha256"] != readiness["sha256"]
        for row in epochs.values()
    ):
        raise ValueError(f"V18 {arm} history is not bound to one readiness report")

    device = task_device(config)
    dataset = build_task_dataset(
        root, config, split_name=config.data.validation_split_name
    )
    model = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(
        best_path,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise ValueError(f"V18 {arm} checkpoint epoch differs from summary")
    independent = evaluate_task(model, dataset, config=config, device=device)
    independent_metrics = _parent_metrics(independent)
    if independent_metrics != best_metrics:
        raise ValueError(f"V18 {arm} independent checkpoint evaluation differs")
    return {
        "arm": arm,
        "config": str(path),
        "config_sha256": sha256_file(path),
        "artifact": str(artifact),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "best_epoch": best_epoch,
        "performance_status": summary.get("performance_status"),
        "best_metrics": best_metrics,
        "epochs": epochs,
        "paired_initialization": initialization,
        "v18_training_readiness": readiness,
        "independent_best_checkpoint_validation_matches": True,
        "contract_hashes": contract["hashes"],
        "micro_doppler_dim": config.model.micro_doppler_dim,
    }


def _class_floor(metrics: Mapping[str, Any], class_name: str) -> float:
    row = metrics["per_class"][class_name]
    return min(float(row["precision"]), float(row["recall"]), float(row["f1"]))


def _drone_bird_floor(metrics: Mapping[str, Any]) -> float:
    return min(
        _class_floor(metrics, "DroneTarget"), _class_floor(metrics, "BirdTarget")
    )


def _comparison(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    c_errors = control["directional_errors"]
    t_errors = treatment["directional_errors"]
    delta = {
        "macro_f1": float(treatment["macro_f1"] - control["macro_f1"]),
        "accuracy": float(treatment["accuracy"] - control["accuracy"]),
        "drone_bird_weak_floor": float(
            _drone_bird_floor(treatment) - _drone_bird_floor(control)
        ),
        "DroneTarget_floor": _class_floor(treatment, "DroneTarget")
        - _class_floor(control, "DroneTarget"),
        "BirdTarget_floor": _class_floor(treatment, "BirdTarget")
        - _class_floor(control, "BirdTarget"),
        "BalloonTarget_floor": _class_floor(treatment, "BalloonTarget")
        - _class_floor(control, "BalloonTarget"),
        "Drone_to_Bird_error_reduction": int(
            c_errors["DroneTarget_to_BirdTarget"]
            - t_errors["DroneTarget_to_BirdTarget"]
        ),
        "Bird_to_Drone_error_reduction": int(
            c_errors["BirdTarget_to_DroneTarget"]
            - t_errors["BirdTarget_to_DroneTarget"]
        ),
        "Drone_Bird_bidirectional_error_reduction": int(
            c_errors["Drone_Bird_bidirectional_total"]
            - t_errors["Drone_Bird_bidirectional_total"]
        ),
    }
    return {
        "control": control,
        "treatment": treatment,
        "treatment_minus_control": delta,
        "passes_two_percentage_point_information_gain": (
            _strictly_exceeds_adoption_gain(delta["macro_f1"])
            and _strictly_exceeds_adoption_gain(delta["drone_bird_weak_floor"])
            and delta["Drone_to_Bird_error_reduction"] >= 0
            and delta["Bird_to_Drone_error_reduction"] >= 0
            and delta["Drone_Bird_bidirectional_error_reduction"] >= 1
        ),
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    runs = {arm: _load_arm(root, arm) for arm in ARMS}
    control = runs["control"]
    treatment = runs["treatment"]
    order_mismatch = [
        epoch
        for epoch in range(1, 21)
        if control["epochs"][epoch]["parent_order_sha256"]
        != treatment["epochs"][epoch]["parent_order_sha256"]
    ]
    if order_mismatch:
        raise ValueError(f"V18 parent order differs at epochs {order_mismatch}")
    if (
        control["paired_initialization"].get("initial_logits_sha256")
        != treatment["paired_initialization"].get("initial_logits_sha256")
    ):
        raise ValueError("V18 Control/Treatment initial CPU-logit hash differs")

    own_best = _comparison(control["best_metrics"], treatment["best_metrics"])
    guard_epochs = sorted(
        {int(control["best_epoch"]), int(treatment["best_epoch"]), 20}
    )
    same_epoch = {
        str(epoch): {
            "epoch": epoch,
            **_comparison(
                control["epochs"][epoch]["parent_metrics"],
                treatment["epochs"][epoch]["parent_metrics"],
            ),
        }
        for epoch in guard_epochs
    }
    same_epoch_pass_count = sum(
        bool(row["passes_two_percentage_point_information_gain"])
        for row in same_epoch.values()
    )
    treatment_best = treatment["best_metrics"]
    class_thresholds = {
        class_name: {
            "floor": _class_floor(treatment_best, class_name),
            "target": 0.95,
            "passed": _class_floor(treatment_best, class_name) >= 0.95,
        }
        for class_name in CLASS_NAMES
    }
    own_delta = own_best["treatment_minus_control"]
    checks = {
        "all_treatment_classes_precision_recall_f1_at_least_0_95": all(
            row["passed"] for row in class_thresholds.values()
        ),
        "own_best_parent_macro_f1_improves_by_more_than_0_02": (
            _strictly_exceeds_adoption_gain(own_delta["macro_f1"])
        ),
        "own_best_drone_bird_weak_floor_improves_by_more_than_0_02": (
            _strictly_exceeds_adoption_gain(own_delta["drone_bird_weak_floor"])
        ),
        "own_best_drone_to_bird_not_increased": (
            own_delta["Drone_to_Bird_error_reduction"] >= 0
        ),
        "own_best_bird_to_drone_not_increased": (
            own_delta["Bird_to_Drone_error_reduction"] >= 0
        ),
        "own_best_bidirectional_drone_bird_errors_reduced": (
            own_delta["Drone_Bird_bidirectional_error_reduction"] >= 1
        ),
        "at_least_one_frozen_same_epoch_guard_also_passes_0_02_gate": (
            same_epoch_pass_count >= 1
        ),
        "all_20_epoch_parent_orders_identical": True,
        "initial_cpu_logits_identical": True,
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": (
            "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
            "paired-validation-v1"
        ),
        "paired_protocol": BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
        "scope": "DBB specialist validation parent-final local task only",
        "strict_lower_bound_absolute_improvement_for_adoption": (
            MIN_ABSOLUTE_IMPROVEMENT
        ),
        "strict_improvement_comparison_epsilon": STRICT_IMPROVEMENT_EPSILON,
        "minimum_improvement_role": (
            "frozen post-training paired acceptance only; never used to choose an epoch"
        ),
        "checks": checks,
        "class_threshold_checks": class_thresholds,
        "own_best_primary_comparison": own_best,
        "same_epoch_attribution_guards": same_epoch,
        "same_epoch_passing_count": same_epoch_pass_count,
        "runs": {
            arm: {key: value for key, value in run.items() if key != "epochs"}
            for arm, run in runs.items()
        },
        "conclusion": (
            "V18 Treatment passes the frozen single-seed local DBB >2 percentage-point adoption gate; multi-seed confirmation remains required before retaining it."
            if passed
            else "V18 Treatment does not pass the frozen local DBB >2 percentage-point adoption gate; do not adopt the feature set or advance it to cascade/open-set claims."
        ),
        "not_final_five_class_performance": True,
        "formal_training": "已执行（仅在两份完成摘要通过完整性校验时）",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / (output_path or OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the completed V18 DBB broadband micro-Doppler pair."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output)


if __name__ == "__main__":
    main()
