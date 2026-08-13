from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.task_models import (
    MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    MICRO_DOPPLER_PAIRED_PROTOCOL,
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
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V17-target-anchored-v3-micro-doppler-paired-validation.json"
)


def config_path(arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/RADAR-LN3-V17-cq08-liquid-dbb-specialist-"
        f"rd-physical4-micro-doppler-target-anchored-v3-{arm}-seed42-config.json"
    )


def _metric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"V17 paired metric is not finite: {name}={value}")
    return result


def _parent_metrics(validation: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(validation.get("parent_metrics") or {})
    rows = {str(row["class"]): dict(row) for row in source.get("per_class") or []}
    if set(rows) != set(CLASS_NAMES):
        raise ValueError(f"V17 paired metrics require {CLASS_NAMES}; actual={sorted(rows)}")
    matrix = np.asarray(
        source.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64
    )
    if matrix.shape != (3, 3):
        raise ValueError("V17 paired parent confusion matrix must be 3x3")
    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = rows[class_name]
        support = int(row.get("support", 0))
        if support <= 0 or int(matrix[class_id].sum()) != support:
            raise ValueError(f"V17 {class_name} support/confusion matrix is invalid")
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
            raise ValueError(f"V17 training summary repeats epoch {epoch}")
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
            "v17_training_readiness_sha256": str(
                record.get("v17_training_readiness_sha256")
            ),
            "parent_metrics": _parent_metrics(dict(record.get("validation") or {})),
        }
    if set(rows) != set(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError("V17 training summary must contain exactly epochs 1..20")
    return rows


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    path = root / config_path(arm)
    config = load_task_config(path)
    if (
        config.schema_version != MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or config.train.paired_protocol != MICRO_DOPPLER_PAIRED_PROTOCOL
        or config.train.paired_arm != arm
        or config.train.seed != 42
        or config.train.epochs != EXPECTED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.task.name != "dbb_specialist"
    ):
        raise ValueError(f"V17 {arm} config violates the frozen paired protocol")
    contract = verify_task_data_contract(root, config)
    artifact = root / config.train.formal_artifact_dir
    summary_path = artifact / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"V17 {arm} training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("config_name") != config.config_name
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("global_step", 0)) <= 0
        or summary.get("formal_pretraining") != "not_required_and_not_used"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError(f"V17 {arm} summary violates completion/scope discipline")
    epochs = _epoch_metrics(summary)
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError(f"V17 {arm} best epoch is outside 1..20")
    best_metrics = _parent_metrics(dict(summary.get("best_validation_metrics") or {}))
    if best_metrics != epochs[best_epoch]["parent_metrics"]:
        raise ValueError(f"V17 {arm} best metrics differ from its same history epoch")
    expected_best = root / config.train.checkpoint_path
    expected_last = root / config.train.last_checkpoint_path
    if not expected_best.is_file() or not expected_last.is_file():
        raise FileNotFoundError(f"V17 {arm} best/last checkpoint is incomplete")
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    epoch_paths = [epoch_dir / f"epoch-{epoch:03d}.pt" for epoch in range(1, 21)]
    missing = [str(value) for value in epoch_paths if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"V17 {arm} epoch checkpoints are incomplete: {missing[:3]}")
    initialization = dict(summary.get("paired_initialization") or {})
    if (
        initialization.get("shared_parameters_and_buffers_exactly_equal") is not True
        or initialization.get("treatment_new_fusion_columns_exactly_zero") is not True
        or initialization.get("initial_cpu_reference_logits_exactly_equal") is not True
        or int(initialization.get("shared_rd_physical_columns", -1)) != 4
        or int(initialization.get("treatment_new_fusion_columns", -1)) != 3
    ):
        raise ValueError(f"V17 {arm} lacks valid paired-initialization evidence")
    loader = dict(summary.get("dataloader_contract") or {})
    if (
        int(loader.get("seed", -1)) != 42
        or loader.get("independent_generator") is not True
        or loader.get("cuda_deterministic_algorithms") is not False
        or loader.get("determinism_claim")
        != "fixed_shared_initialization_and_sampling_order_not_bitwise_cuda_determinism"
    ):
        raise ValueError(f"V17 {arm} DataLoader/determinism evidence is invalid")
    readiness = dict(summary.get("v17_training_readiness") or {})
    readiness_path = Path(str(readiness.get("path", "")))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V17-micro-doppler-training-readiness-v4"
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != readiness.get("sha256")
    ):
        raise ValueError(f"V17 {arm} summary lacks a valid immutable readiness binding")
    if any(
        value["v17_training_readiness_sha256"] != readiness["sha256"]
        for value in epochs.values()
    ):
        raise ValueError(f"V17 {arm} epoch history is not bound to one readiness report")

    device = task_device(config)
    dataset = build_task_dataset(
        root, config, split_name=config.data.validation_split_name
    )
    model = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(
        expected_best,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise ValueError(f"V17 {arm} best checkpoint epoch differs from summary")
    if dict(checkpoint.get("metrics") or {}).get("v17_training_readiness_sha256") != readiness[
        "sha256"
    ]:
        raise ValueError(f"V17 {arm} best checkpoint readiness binding differs")
    independent = evaluate_task(model, dataset, config=config, device=device)
    independent_metrics = _parent_metrics(independent)
    if independent_metrics != best_metrics:
        raise ValueError(f"V17 {arm} independent best-checkpoint evaluation differs")
    return {
        "arm": arm,
        "config": str(path),
        "config_sha256": sha256_file(path),
        "artifact": str(artifact),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint": str(expected_best),
        "best_checkpoint_sha256": sha256_file(expected_best),
        "last_checkpoint": str(expected_last),
        "last_checkpoint_sha256": sha256_file(expected_last),
        "best_epoch": best_epoch,
        "performance_status": summary.get("performance_status"),
        "best_metrics": best_metrics,
        "epochs": epochs,
        "paired_initialization": initialization,
        "dataloader_contract": loader,
        "v17_training_readiness": readiness,
        "independent_best_checkpoint_validation_matches": True,
        "contract_hashes": contract["hashes"],
        "micro_doppler_dim": config.model.micro_doppler_dim,
    }


def _class_floor(metrics: Mapping[str, Any], class_name: str) -> float:
    row = metrics["per_class"][class_name]
    return min(float(row["precision"]), float(row["recall"]), float(row["f1"]))


def _comparison(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    c_errors = control["directional_errors"]
    t_errors = treatment["directional_errors"]
    delta = {
        "macro_f1": float(treatment["macro_f1"] - control["macro_f1"]),
        "accuracy": float(treatment["accuracy"] - control["accuracy"]),
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
        "Balloon_correct_parent_delta": int(
            treatment["per_class"]["BalloonTarget"]["correct_parent_count"]
            - control["per_class"]["BalloonTarget"]["correct_parent_count"]
        ),
    }
    improves = (
        delta["macro_f1"] > 0.0
        and delta["Drone_Bird_bidirectional_error_reduction"] >= 1
        and delta["Drone_to_Bird_error_reduction"] >= 0
        and delta["Bird_to_Drone_error_reduction"] >= 0
    )
    return {
        "control": control,
        "treatment": treatment,
        "treatment_minus_control": delta,
        "drone_bird_direction_consistent_improvement": improves,
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    runs = {arm: _load_arm(root, arm) for arm in ARMS}
    control = runs["control"]
    treatment = runs["treatment"]
    order_mismatch_epochs = [
        epoch
        for epoch in range(1, 21)
        if control["epochs"][epoch]["parent_order_sha256"]
        != treatment["epochs"][epoch]["parent_order_sha256"]
    ]
    if order_mismatch_epochs:
        raise ValueError(f"V17 parent order differs at epochs {order_mismatch_epochs}")
    if (
        control["paired_initialization"].get("initial_logits_sha256")
        != treatment["paired_initialization"].get("initial_logits_sha256")
    ):
        raise ValueError("V17 Control/Treatment initial CPU-logit hash differs")
    own_best = _comparison(control["best_metrics"], treatment["best_metrics"])
    guard_epochs = {
        "control_best_epoch": int(control["best_epoch"]),
        "treatment_best_epoch": int(treatment["best_epoch"]),
        "epoch20": 20,
    }
    same_epoch_guards = {
        name: {
            "epoch": epoch,
            **_comparison(
                control["epochs"][epoch]["parent_metrics"],
                treatment["epochs"][epoch]["parent_metrics"],
            ),
        }
        for name, epoch in guard_epochs.items()
    }
    same_epoch_passing_count = sum(
        bool(value["drone_bird_direction_consistent_improvement"])
        for value in same_epoch_guards.values()
    )
    unique_guard_epochs = sorted(set(guard_epochs.values()))
    unique_same_epoch_guards = {
        str(epoch): _comparison(
            control["epochs"][epoch]["parent_metrics"],
            treatment["epochs"][epoch]["parent_metrics"],
        )
        for epoch in unique_guard_epochs
    }
    unique_same_epoch_passing_count = sum(
        bool(value["drone_bird_direction_consistent_improvement"])
        for value in unique_same_epoch_guards.values()
    )
    treatment_best = treatment["best_metrics"]
    class_threshold_checks = {
        class_name: {
            "floor": _class_floor(treatment_best, class_name),
            "target": 0.95,
            "passed": _class_floor(treatment_best, class_name) >= 0.95,
        }
        for class_name in CLASS_NAMES
    }
    c_errors = control["best_metrics"]["directional_errors"]
    t_errors = treatment_best["directional_errors"]
    checks = {
        "all_treatment_classes_precision_recall_f1_at_least_0_95": all(
            value["passed"] for value in class_threshold_checks.values()
        ),
        "own_best_macro_f1_strictly_improved": (
            treatment_best["macro_f1"] > control["best_metrics"]["macro_f1"]
        ),
        "own_best_drone_to_bird_not_increased": (
            t_errors["DroneTarget_to_BirdTarget"]
            <= c_errors["DroneTarget_to_BirdTarget"]
        ),
        "own_best_bird_to_drone_not_increased": (
            t_errors["BirdTarget_to_DroneTarget"]
            <= c_errors["BirdTarget_to_DroneTarget"]
        ),
        "own_best_bidirectional_drone_bird_error_reduced_by_at_least_one_parent": (
            t_errors["Drone_Bird_bidirectional_total"]
            <= c_errors["Drone_Bird_bidirectional_total"] - 1
        ),
        "balloon_correct_parent_loss_no_more_than_one": (
            treatment_best["per_class"]["BalloonTarget"]["correct_parent_count"]
            >= control["best_metrics"]["per_class"]["BalloonTarget"][
                "correct_parent_count"
            ]
            - 1
        ),
        "at_least_two_distinct_same_epoch_guards_improve_or_all_when_only_one_available": (
            unique_same_epoch_passing_count
            >= min(2, len(unique_guard_epochs))
        ),
        "own_best_drone_bird_minimum_floor_strictly_improved": (
            min(
                _class_floor(treatment_best, "DroneTarget"),
                _class_floor(treatment_best, "BirdTarget"),
            )
            > min(
                _class_floor(control["best_metrics"], "DroneTarget"),
                _class_floor(control["best_metrics"], "BirdTarget"),
            )
        ),
        "all_20_epoch_parent_orders_identical": True,
        "initial_cpu_logits_identical": True,
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": "RADAR-LN3-V17-micro-doppler-paired-validation-v3",
        "paired_protocol": MICRO_DOPPLER_PAIRED_PROTOCOL,
        "scope": "DBB specialist validation parent-final local task only",
        "checks": checks,
        "class_threshold_checks": class_threshold_checks,
        "own_best_primary_comparison": own_best,
        "same_epoch_attribution_guards": same_epoch_guards,
        "same_epoch_passing_count": same_epoch_passing_count,
        "unique_same_epoch_attribution_guards": unique_same_epoch_guards,
        "unique_same_epoch_guard_count": len(unique_guard_epochs),
        "unique_same_epoch_passing_count": unique_same_epoch_passing_count,
        "runs": {
            arm: {key: value for key, value in run.items() if key != "epochs"}
            for arm, run in runs.items()
        },
        "conclusion": (
            "V17 Treatment passes the frozen single-seed local DBB screening gate; multi-seed confirmation remains required before retaining it."
            if passed
            else "V17 Treatment does not pass the frozen local DBB screening gate; do not advance it to multi-seed, cascade or open-set claims."
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
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the completed V17 DBB micro-Doppler pair.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output)


if __name__ == "__main__":
    main()
