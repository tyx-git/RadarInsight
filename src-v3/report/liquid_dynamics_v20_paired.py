"""Evaluate a completed V20/V20r DBB dynamics pair on development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.task_models import (
    LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    V20_DENSITY_PAIRED_PROTOCOL,
    V20_SKIP_PAIRED_PROTOCOL,
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
    verify_task_data_contract,
)
from report.liquid_dynamics_v20_readiness import VARIANTS, _config_paths, _training_core


CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
EXPECTED_EPOCHS = 20
MIN_ABSOLUTE_IMPROVEMENT = 0.02
MULTISEED_READINESS_SCHEMA = (
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1"
)


def _metric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"V20 metric is not finite: {name}={value}")
    return result


def _parent_metrics(validation: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(validation.get("parent_metrics") or {})
    rows = list(source.get("per_class") or [])
    if tuple(str(row.get("class")) for row in rows) != CLASS_NAMES:
        raise ValueError("V20 parent metrics have an invalid class order")
    matrix = np.asarray(source.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("V20 parent confusion matrix must be 3x3")
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        row = rows[index]
        support = int(row.get("support", 0))
        if support <= 0 or int(matrix[index].sum()) != support:
            raise ValueError(f"V20 {name} support is invalid")
        per_class[name] = {
            "precision": _metric(row.get("precision"), f"{name}.precision"),
            "recall": _metric(row.get("recall"), f"{name}.recall"),
            "f1": _metric(row.get("f1"), f"{name}.f1"),
            "support": support,
            "correct_parent_count": int(matrix[index, index]),
        }
    return {
        "support": int(matrix.sum()),
        "accuracy": _metric(source.get("accuracy"), "accuracy"),
        "macro_precision": _metric(source.get("macro_precision"), "macro_precision"),
        "macro_recall": _metric(source.get("macro_recall"), "macro_recall"),
        "macro_f1": _metric(source.get("macro_f1"), "macro_f1"),
        "balanced_accuracy": _metric(source.get("balanced_accuracy"), "balanced_accuracy"),
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


def _class_floor(metrics: Mapping[str, Any], name: str) -> float:
    row = metrics["per_class"][name]
    return min(float(row["precision"]), float(row["recall"]), float(row["f1"]))


def _weak_floor(metrics: Mapping[str, Any]) -> float:
    return min(_class_floor(metrics, "DroneTarget"), _class_floor(metrics, "BirdTarget"))


def _comparison(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    c = control["directional_errors"]
    t = treatment["directional_errors"]
    delta = {
        "accuracy": float(treatment["accuracy"] - control["accuracy"]),
        "macro_f1": float(treatment["macro_f1"] - control["macro_f1"]),
        "drone_bird_weak_floor": float(_weak_floor(treatment) - _weak_floor(control)),
        "DroneTarget_floor": _class_floor(treatment, "DroneTarget") - _class_floor(control, "DroneTarget"),
        "BirdTarget_floor": _class_floor(treatment, "BirdTarget") - _class_floor(control, "BirdTarget"),
        "BalloonTarget_floor": _class_floor(treatment, "BalloonTarget") - _class_floor(control, "BalloonTarget"),
        "Drone_to_Bird_error_reduction": int(c["DroneTarget_to_BirdTarget"] - t["DroneTarget_to_BirdTarget"]),
        "Bird_to_Drone_error_reduction": int(c["BirdTarget_to_DroneTarget"] - t["BirdTarget_to_DroneTarget"]),
        "Drone_Bird_bidirectional_error_reduction": int(
            c["Drone_Bird_bidirectional_total"] - t["Drone_Bird_bidirectional_total"]
        ),
    }
    return {"control": control, "treatment": treatment, "treatment_minus_control": delta}


def _density_shared_control_contract(root: Path) -> dict[str, Any]:
    shared_path = root / _config_paths("skip")["control"]
    density_path = root / _config_paths("density")["control"]
    shared = load_task_config(shared_path)
    density = load_task_config(density_path)
    if (
        shared.task != density.task
        or shared.data != density.data
        or shared.model != density.model
        or _training_core(shared) != _training_core(density)
        or shared.train.paired_protocol != V20_SKIP_PAIRED_PROTOCOL
        or density.train.paired_protocol != V20_DENSITY_PAIRED_PROTOCOL
        or shared.train.paired_arm != "control"
        or density.train.paired_arm != "control"
    ):
        raise ValueError(
            "V20r density report cannot reuse a non-equivalent V20 skip Control"
        )
    return {
        "source_experiment": "V20-skip",
        "source_config": str(shared_path),
        "source_config_sha256": sha256_file(shared_path),
        "density_control_reference_config": str(density_path),
        "density_control_reference_config_sha256": sha256_file(density_path),
        "model_data_and_training_core_exactly_equal": True,
        "shared_control_retrained": False,
    }


def _initialization_contract_is_valid(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("shared_parameters_and_buffers_exactly_equal") is True
        and (
            value.get("initial_pre_treatment_exposure_logits_exactly_equal") is True
            or value.get("initial_cpu_reference_logits_exactly_equal") is True
        )
        and isinstance(value.get("initial_logits_sha256"), str)
        and len(str(value.get("initial_logits_sha256"))) == 64
    )


def _load_arm(
    root: Path,
    variant: str,
    arm: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if seed == 42:
        relative_path = _config_paths(variant)[arm]
    elif variant == "density" and seed in {43, 44}:
        relative_path = Path(
            "config/RADAR-LN3/"
            "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-"
            f"{arm}-seed{seed}-config.json"
        )
    else:
        raise ValueError(
            f"unsupported V20 arm request: variant={variant}, arm={arm}, seed={seed}"
        )
    path = root / relative_path
    config = load_task_config(path)
    if (
        config.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
        or config.train.paired_protocol != VARIANTS[variant]["protocol"]
        or config.train.paired_arm != arm
        or config.train.seed != seed
        or config.train.epochs != EXPECTED_EPOCHS
        or config.task.name != "dbb_specialist"
    ):
        raise ValueError(
            f"V20 {variant} seed{seed} {arm} config violates the paired protocol"
        )
    contract = verify_task_data_contract(root, config)
    artifact = root / config.train.formal_artifact_dir
    summary_path = artifact / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readiness = dict(summary.get("v20_training_readiness") or {})
    expected_readiness_schema = (
        MULTISEED_READINESS_SCHEMA
        if variant == "density" and seed in {43, 44}
        else f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-training-readiness-v1"
    )
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != expected_readiness_schema
        or readiness.get("paired_protocol") != VARIANTS[variant]["protocol"]
    ):
        raise ValueError(
            f"V20 {variant} seed{seed} {arm} summary violates local-only scope"
        )
    readiness_path = Path(str(readiness.get("path", "")))
    if not readiness_path.is_file() or sha256_file(readiness_path) != readiness.get("sha256"):
        raise ValueError(
            f"V20 {variant} seed{seed} {arm} readiness binding is invalid"
        )
    epochs: dict[int, dict[str, Any]] = {}
    for record in summary.get("history") or []:
        epoch = int(record.get("epoch", -1))
        if epoch in epochs:
            raise ValueError(f"V20 seed{seed} {arm} repeats epoch {epoch}")
        epochs[epoch] = {
            "epoch": epoch,
            "parent_order_sha256": str(record.get("parent_order_sha256")),
            "parent_metrics": _parent_metrics(dict(record.get("validation") or {})),
            "train_loss": _metric(record.get("train_loss"), f"epoch{epoch}.train_loss"),
            "readiness_sha256": str(record.get("v20_training_readiness_sha256")),
        }
    if set(epochs) != set(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError(f"V20 seed{seed} {arm} history must contain epochs 1..20")
    if any(row["readiness_sha256"] != readiness["sha256"] for row in epochs.values()):
        raise ValueError(f"V20 seed{seed} {arm} history is not readiness-bound")
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError(f"V20 seed{seed} {arm} best epoch is invalid")
    best_path = root / config.train.checkpoint_path
    last_path = root / config.train.last_checkpoint_path
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(
            f"V20 seed{seed} {arm} best/last checkpoint is incomplete"
        )
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    if any(not (epoch_dir / f"epoch-{epoch:03d}.pt").is_file() for epoch in range(1, 21)):
        raise FileNotFoundError(
            f"V20 seed{seed} {arm} epoch checkpoints are incomplete"
        )
    device = task_device(config)
    dataset = build_task_dataset(root, config, split_name=config.data.validation_split_name)
    model = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(
        best_path,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise ValueError(
            f"V20 seed{seed} {arm} best checkpoint epoch differs from summary"
        )
    independent = _parent_metrics(evaluate_task(model, dataset, config=config, device=device))
    summary_best = _parent_metrics(dict(summary.get("best_validation_metrics") or {}))
    if independent != summary_best or independent != epochs[best_epoch]["parent_metrics"]:
        raise ValueError(
            f"V20 seed{seed} {arm} independent evaluation differs from summary"
        )
    return {
        "seed": seed,
        "arm": arm,
        "config": str(path),
        "config_sha256": sha256_file(path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "readiness": readiness,
        "contract_hashes": contract["hashes"],
        "best_epoch": best_epoch,
        "best_metrics": independent,
        "epochs": epochs,
        "paired_initialization": dict(summary.get("paired_initialization") or {}),
        "performance_status": summary.get("performance_status"),
    }


def run(project_root: Path, variant: str = "skip", output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    shared_control = None
    if variant == "density":
        shared_control = _density_shared_control_contract(root)
        runs = {
            "control": _load_arm(root, "skip", "control"),
            "treatment": _load_arm(root, "density", "treatment"),
        }
    else:
        runs = {arm: _load_arm(root, variant, arm) for arm in ("control", "treatment")}
    if runs["control"]["contract_hashes"] != runs["treatment"]["contract_hashes"]:
        raise ValueError("V20 pair data contracts differ")
    for epoch in range(1, EXPECTED_EPOCHS + 1):
        if runs["control"]["epochs"][epoch]["parent_order_sha256"] != runs["treatment"]["epochs"][epoch]["parent_order_sha256"]:
            raise ValueError(f"V20 parent order differs at epoch {epoch}")
    init_control = runs["control"]["paired_initialization"]
    init_treatment = runs["treatment"]["paired_initialization"]
    if not (
        _initialization_contract_is_valid(init_control)
        and _initialization_contract_is_valid(init_treatment)
        and init_control.get("initial_logits_sha256") == init_treatment.get("initial_logits_sha256")
    ):
        raise ValueError("V20 paired initialization or pre-exposure logits are invalid")
    own_best = _comparison(runs["control"]["best_metrics"], runs["treatment"]["best_metrics"])
    guard_epochs = sorted({runs["control"]["best_epoch"], runs["treatment"]["best_epoch"], 20})
    same_epoch = {
        str(epoch): _comparison(
            runs["control"]["epochs"][epoch]["parent_metrics"],
            runs["treatment"]["epochs"][epoch]["parent_metrics"],
        )
        for epoch in guard_epochs
    }
    delta = own_best["treatment_minus_control"]
    class_thresholds = {
        name: {
            "floor": _class_floor(runs["treatment"]["best_metrics"], name),
            "target": 0.95,
            "passed": _class_floor(runs["treatment"]["best_metrics"], name) >= 0.95,
        }
        for name in CLASS_NAMES
    }
    checks = {
        "all_treatment_classes_precision_recall_f1_at_least_0_95": all(
            row["passed"] for row in class_thresholds.values()
        ),
        "own_best_macro_f1_improves_by_more_than_0_02": delta["macro_f1"] > MIN_ABSOLUTE_IMPROVEMENT,
        "own_best_drone_bird_weak_floor_improves_by_more_than_0_02": delta["drone_bird_weak_floor"] > MIN_ABSOLUTE_IMPROVEMENT,
        "own_best_drone_to_bird_not_increased": delta["Drone_to_Bird_error_reduction"] >= 0,
        "own_best_bird_to_drone_not_increased": delta["Bird_to_Drone_error_reduction"] >= 0,
        "own_best_bidirectional_errors_reduced": delta["Drone_Bird_bidirectional_error_reduction"] >= 1,
        "all_20_parent_orders_identical": True,
        "shared_parameters_and_pre_exposure_logits_identical": True,
        "density_shared_v20_control_reused_without_retraining": (
            shared_control is not None if variant == "density" else True
        ),
    }
    readiness_run = runs["treatment"] if variant == "density" else runs["control"]
    zero_stats = json.loads(
        Path(readiness_run["readiness"]["path"]).read_text(encoding="utf-8")
    )["split_reports"]
    if variant == "skip":
        if abs(delta["macro_f1"]) < 0.01 and max(
            abs(delta[key]) for key in ("DroneTarget_floor", "BirdTarget_floor", "BalloonTarget_floor")
        ) < 0.02:
            diagnostic_conclusion = "H0_fixed_zero_delta_gain_not_materially_used"
        elif delta["macro_f1"] < -0.01:
            diagnostic_conclusion = "H1_zero_delta_events_carry_usable_temporal_information"
        else:
            diagnostic_conclusion = "small_or_mixed_effect_inconclusive"
    else:
        diagnostic_conclusion = "density_variant_requires_frozen_positive_gain_gate"
    passed = all(checks.values())
    output = root / (
        output_path
        or Path(
            "src-v3/processed/report/"
            f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-paired-validation.json"
        )
    )
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-paired-validation-v1",
        "experiment": VARIANTS[variant]["label"],
        "paired_protocol": VARIANTS[variant]["protocol"],
        "scope": "DBB specialist validation parent-final local task only",
        "shared_control_reuse": shared_control,
        "checks": checks,
        "class_threshold_checks": class_thresholds,
        "own_best_primary_comparison": own_best,
        "same_epoch_attribution_guards": same_epoch,
        "zero_delta_diagnostic": {
            "conclusion": diagnostic_conclusion,
            "train": zero_stats["train"],
            "validation_report_only": zero_stats["val"],
        },
        "runs": {
            arm: {key: value for key, value in run.items() if key != "epochs"}
            for arm, run in runs.items()
        },
        "conclusion": (
            "V20/V20r Treatment passes the frozen local DBB adoption gate."
            if passed
            else "V20/V20r Treatment does not pass the frozen local DBB adoption gate; retain as a dynamics diagnosis."
        ),
        "not_final_five_class_performance": True,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed V20/V20r DBB dynamics pair.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="skip")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root, args.variant, args.output)


if __name__ == "__main__":
    main()
