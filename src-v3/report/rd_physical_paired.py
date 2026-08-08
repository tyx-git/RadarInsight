from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
    RD_PHYSICAL_PAIRED_PROTOCOL,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
)


SEEDS = (42, 43, 44)
ARMS = ("control", "treatment")
EXPECTED_EPOCHS = 20


def config_path(seed: int, arm: str) -> Path:
    return Path(
        f"config/RADAR-LN3/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}-config.json"
    )


def _require_metric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"paired report metric is not finite: {name}={value}")
    return result


def _class_metrics(validation: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    metrics = dict(validation.get("parent_metrics") or {})
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class") or []}
    if set(rows) != {"DBB", "CU"}:
        raise ValueError(f"paired report requires DBB/CU parent metrics; actual={sorted(rows)}")
    output: dict[str, dict[str, float | int]] = {}
    for name in ("DBB", "CU"):
        row = rows[name]
        output[name] = {
            "precision": _require_metric(row.get("precision"), f"{name}.precision"),
            "recall": _require_metric(row.get("recall"), f"{name}.recall"),
            "f1": _require_metric(row.get("f1"), f"{name}.f1"),
            "support": int(row.get("support", 0)),
        }
        if int(output[name]["support"]) <= 0:
            raise ValueError(f"paired report {name} support is zero")
    matrix = np.asarray(metrics.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64)
    if matrix.shape != (2, 2) or int(matrix.sum()) != int(output["DBB"]["support"]) + int(output["CU"]["support"]):
        raise ValueError("paired report parent confusion matrix is invalid")
    output["errors"] = {
        "DBB_to_CU": int(matrix[0, 1]),
        "CU_to_DBB": int(matrix[1, 0]),
    }
    output["overall"] = {
        "accuracy": _require_metric(metrics.get("accuracy"), "parent.accuracy"),
        "macro_precision": _require_metric(metrics.get("macro_precision"), "parent.macro_precision"),
        "macro_recall": _require_metric(metrics.get("macro_recall"), "parent.macro_recall"),
        "macro_f1": _require_metric(metrics.get("macro_f1"), "parent.macro_f1"),
        "balanced_accuracy": _require_metric(metrics.get("balanced_accuracy"), "parent.balanced_accuracy"),
    }
    return output


def _epoch_metrics(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for record in summary.get("history") or []:
        epoch = int(record["epoch"])
        if epoch in rows:
            raise ValueError(f"paired training summary duplicates epoch {epoch}")
        rows[epoch] = {
            "epoch": epoch,
            "global_step": int(record["global_step"]),
            "train_loss": _require_metric(record["train_loss"], f"epoch{epoch}.train_loss"),
            "parent_order_sha256": str(record["parent_order_sha256"]),
            "parent_metrics": _class_metrics(dict(record.get("validation") or {})),
            "checkpoint_selection": dict(record.get("checkpoint_selection") or {}),
        }
    if set(rows) != set(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError(f"paired training summary does not contain exactly epochs 1..{EXPECTED_EPOCHS}")
    return rows


def _load_arm(root: Path, seed: int, arm: str) -> dict[str, Any]:
    path = root / config_path(seed, arm)
    config = load_task_config(path)
    if (
        config.schema_version != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION
        or config.train.paired_protocol != RD_PHYSICAL_PAIRED_PROTOCOL
        or config.train.paired_arm != arm
        or config.train.seed != seed
        or config.train.epochs != EXPECTED_EPOCHS
        or config.train.early_stopping_patience != 0
    ):
        raise ValueError(f"paired report config violates frozen protocol: {path}")
    artifact_dir = root / config.train.formal_artifact_dir
    summary_path = artifact_dir / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"paired formal artifact is incomplete; training summary does not exist: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "passed"
        or summary.get("config_name") != config.config_name
        or summary.get("task_name") != "dbb_cu_gate"
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("global_step", 0)) <= 0
        or summary.get("formal_pretraining") != "not_required_and_not_used"
        or summary.get("test_evaluation") != "not_executed"
    ):
        raise ValueError(f"paired training summary violates completion/test/pretraining contract: {summary_path}")
    epochs = _epoch_metrics(summary)
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError(f"paired best epoch is outside 1..20: {summary_path}: {best_epoch}")
    best_from_summary = _class_metrics(dict(summary.get("best_validation_metrics") or {}))
    if best_from_summary != epochs[best_epoch]["parent_metrics"]:
        raise ValueError(f"paired best metrics differ from the same history epoch: {summary_path}")
    best_checkpoint = Path(str(summary.get("best_checkpoint", "")))
    last_checkpoint = Path(str(summary.get("last_checkpoint", "")))
    expected_best = root / config.train.checkpoint_path
    expected_last = root / config.train.last_checkpoint_path
    if best_checkpoint.resolve() != expected_best.resolve() or last_checkpoint.resolve() != expected_last.resolve():
        raise ValueError(f"paired summary checkpoint paths differ from config: {summary_path}")
    if not expected_best.is_file() or not expected_last.is_file():
        raise FileNotFoundError(f"paired best/last checkpoint is missing: {artifact_dir}")
    device = task_device(config)
    validation_dataset = build_task_dataset(
        root,
        config,
        split_name=config.data.validation_split_name,
    )
    independent_model = LiquidTaskClassifier(config).to(device)
    checkpoint_payload = load_task_checkpoint(
        expected_best,
        independent_model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint_payload.get("epoch", -1)) != best_epoch:
        raise ValueError(f"paired best checkpoint epoch differs from training summary: {expected_best}")
    independent_validation = evaluate_task(
        independent_model,
        validation_dataset,
        config=config,
        device=device,
    )
    independent_best_metrics = _class_metrics(independent_validation)
    if independent_best_metrics != best_from_summary:
        raise ValueError(
            "paired independent best-checkpoint validation differs from the training summary: "
            f"{expected_best}"
        )
    epoch_checkpoint_dir = root / str(config.train.epoch_checkpoint_dir)
    expected_epoch_paths = [epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt" for epoch in range(1, 21)]
    missing_epoch_paths = [str(path) for path in expected_epoch_paths if not path.is_file()]
    if missing_epoch_paths:
        raise FileNotFoundError(f"paired epoch checkpoints are incomplete: {missing_epoch_paths[:3]}")
    paired_initialization = dict(summary.get("paired_initialization") or {})
    if (
        paired_initialization.get("shared_parameters_and_buffers_exactly_equal") is not True
        or paired_initialization.get("treatment_new_fusion_columns_exactly_zero") is not True
        or paired_initialization.get("initial_cpu_reference_logits_exactly_equal") is not True
    ):
        raise ValueError(f"paired summary lacks successful shared initialization evidence: {summary_path}")
    loader_contract = dict(summary.get("dataloader_contract") or {})
    if (
        int(loader_contract.get("seed", -1)) != seed
        or loader_contract.get("independent_generator") is not True
        or loader_contract.get("cuda_deterministic_algorithms") is not False
        or loader_contract.get("determinism_claim")
        != "fixed_shared_initialization_and_sampling_order_not_bitwise_cuda_determinism"
    ):
        raise ValueError(f"paired DataLoader/determinism summary is invalid: {summary_path}")
    return {
        "arm": arm,
        "seed": seed,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "artifact_dir": str(artifact_dir),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint": str(expected_best),
        "best_checkpoint_sha256": sha256_file(expected_best),
        "last_checkpoint": str(expected_last),
        "last_checkpoint_sha256": sha256_file(expected_last),
        "best_epoch": best_epoch,
        "best_metrics": best_from_summary,
        "independent_best_checkpoint_validation": {
            "device": str(device),
            "parent_count": int(independent_validation["parent_count"]),
            "window_count": int(independent_validation["window_count"]),
            "metrics": independent_best_metrics,
            "matches_training_summary": True,
        },
        "epoch20_metrics": epochs[20]["parent_metrics"],
        "epochs": epochs,
        "paired_initialization": paired_initialization,
        "loader_contract": loader_contract,
        "epoch_checkpoint_count": len(expected_epoch_paths),
    }


def _minimum_class_metric(metrics: Mapping[str, Any], class_name: str) -> float:
    row = dict(metrics[class_name])
    return min(float(row["precision"]), float(row["recall"]), float(row["f1"]))


def _seed_gate(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_metrics = dict(control["best_metrics"])
    treatment_metrics = dict(treatment["best_metrics"])
    dbb_floor = _minimum_class_metric(treatment_metrics, "DBB")
    cu_floor = _minimum_class_metric(treatment_metrics, "CU")
    control_errors = dict(control_metrics["errors"])
    treatment_errors = dict(treatment_metrics["errors"])
    dbb_reduction = int(control_errors["DBB_to_CU"]) - int(treatment_errors["DBB_to_CU"])
    cu_reduction = int(control_errors["CU_to_DBB"]) - int(treatment_errors["CU_to_DBB"])
    checks = {
        "treatment_DBB_precision_recall_f1_at_least_0_95": dbb_floor >= 0.95,
        "treatment_CU_precision_recall_f1_at_least_0_90": cu_floor >= 0.90,
        "DBB_to_CU_not_higher_than_control": dbb_reduction >= 0,
        "CU_to_DBB_not_higher_than_control": cu_reduction >= 0,
        "at_least_one_error_direction_reduced_by_parent": max(dbb_reduction, cu_reduction) >= 1,
        "CU_min_precision_recall_f1_strictly_improved": (
            _minimum_class_metric(treatment_metrics, "CU")
            > _minimum_class_metric(control_metrics, "CU")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "control_best_epoch": int(control["best_epoch"]),
        "treatment_best_epoch": int(treatment["best_epoch"]),
        "control_best": control_metrics,
        "treatment_best": treatment_metrics,
        "error_reduction_parent_counts": {
            "DBB_to_CU": dbb_reduction,
            "CU_to_DBB": cu_reduction,
        },
        "same_epoch_attribution_guards": {
            "epoch20": {
                "control": control["epochs"][20]["parent_metrics"],
                "treatment": treatment["epochs"][20]["parent_metrics"],
            },
            "control_best_epoch": {
                "epoch": int(control["best_epoch"]),
                "control": control_metrics,
                "treatment": treatment["epochs"][int(control["best_epoch"])]["parent_metrics"],
            },
            "treatment_best_epoch": {
                "epoch": int(treatment["best_epoch"]),
                "control": control["epochs"][int(treatment["best_epoch"])]["parent_metrics"],
                "treatment": treatment_metrics,
            },
        },
    }


def _median(values: Iterable[float | int]) -> float:
    return float(statistics.median(float(value) for value in values))


def _median_metrics(arms: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_name in ("DBB", "CU"):
        output[class_name] = {
            metric: _median(arm["best_metrics"][class_name][metric] for arm in arms)
            for metric in ("precision", "recall", "f1", "support")
        }
    output["errors"] = {
        direction: _median(arm["best_metrics"]["errors"][direction] for arm in arms)
        for direction in ("DBB_to_CU", "CU_to_DBB")
    }
    output["overall"] = {
        metric: _median(arm["best_metrics"]["overall"][metric] for arm in arms)
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }
    return output


def _median_gate(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_errors = dict(control["errors"])
    treatment_errors = dict(treatment["errors"])
    dbb_reduction = float(control_errors["DBB_to_CU"]) - float(treatment_errors["DBB_to_CU"])
    cu_reduction = float(control_errors["CU_to_DBB"]) - float(treatment_errors["CU_to_DBB"])
    checks = {
        "median_treatment_DBB_precision_recall_f1_at_least_0_95": _minimum_class_metric(treatment, "DBB") >= 0.95,
        "median_treatment_CU_precision_recall_f1_at_least_0_90": _minimum_class_metric(treatment, "CU") >= 0.90,
        "median_DBB_to_CU_not_higher_than_control": dbb_reduction >= 0.0,
        "median_CU_to_DBB_not_higher_than_control": cu_reduction >= 0.0,
        "at_least_one_median_error_direction_reduced_by_one_parent": max(dbb_reduction, cu_reduction) >= 1.0,
        "median_CU_min_precision_recall_f1_strictly_improved": (
            _minimum_class_metric(treatment, "CU") > _minimum_class_metric(control, "CU")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "control": control,
        "treatment": treatment,
        "error_reduction_parent_counts": {"DBB_to_CU": dbb_reduction, "CU_to_DBB": cu_reduction},
    }


def evaluate_completed_runs(
    runs: Mapping[int, Mapping[str, Any]],
    *,
    paired_protocol: str = RD_PHYSICAL_PAIRED_PROTOCOL,
) -> dict[str, Any]:
    if set(runs) != set(SEEDS) or any(set(value) != set(ARMS) for value in runs.values()):
        raise ValueError("paired report requires Control/Treatment for seeds 42,43,44")
    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        control = runs[seed]["control"]
        treatment = runs[seed]["treatment"]
        order_mismatches = [
            epoch
            for epoch in range(1, EXPECTED_EPOCHS + 1)
            if control["epochs"][epoch]["parent_order_sha256"]
            != treatment["epochs"][epoch]["parent_order_sha256"]
        ]
        if order_mismatches:
            raise ValueError(f"paired parent order differs for seed={seed}, epochs={order_mismatches}")
        initialization_hash_equal = (
            control["paired_initialization"].get("initial_logits_sha256")
            == treatment["paired_initialization"].get("initial_logits_sha256")
        )
        if not initialization_hash_equal:
            raise ValueError(f"paired initial CPU-logit hash differs for seed={seed}")
        seed_results[str(seed)] = {
            "seed": seed,
            "parent_order_hashes_equal_all_20_epochs": True,
            "initial_cpu_logit_hash_equal": True,
            "gate": _seed_gate(control, treatment),
            "runs": {arm: {key: value for key, value in runs[seed][arm].items() if key != "epochs"} for arm in ARMS},
        }
    control_median = _median_metrics([runs[seed]["control"] for seed in SEEDS])
    treatment_median = _median_metrics([runs[seed]["treatment"] for seed in SEEDS])
    median_gate = _median_gate(control_median, treatment_median)
    passing_seed_count = sum(bool(seed_results[str(seed)]["gate"]["passed"]) for seed in SEEDS)
    stage_one_passed = passing_seed_count >= 2 and bool(median_gate["passed"])
    return {
        "status": "passed" if stage_one_passed else "completed_but_gate_failed",
        "paired_protocol": paired_protocol,
        "seed_results": seed_results,
        "passing_seed_count": passing_seed_count,
        "required_passing_seed_count": 2,
        "median_gate": median_gate,
        "stage_one_passed": stage_one_passed,
        "stage_two_status": (
            "eligible_for_separate_design_authorization_not_automatically_started"
            if stage_one_passed
            else "blocked_by_stage_one"
        ),
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "P95": "未执行",
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    runs = {
        seed: {arm: _load_arm(root, seed, arm) for arm in ARMS}
        for seed in SEEDS
    }
    report = evaluate_completed_runs(runs)
    report.update(
        {
            "schema_version": "RADAR-LN3-V11-rd-physical-paired-report-v1",
            "scope": "CQ08 V3 validation only; three paired seeds; test sealed",
            "formal_training": "completed_by_user_manual_commands",
        }
    )
    path = output_path or (
        root / "src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json"
    )
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed V11 Control/Treatment paired validation runs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
