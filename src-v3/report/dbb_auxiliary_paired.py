"""Evaluate completed V12 DBB-auxiliary paired formal runs on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.task_models import (
    DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION,
    DBB_AUXILIARY_PAIRED_PROTOCOL,
    DBB_AUXILIARY_TREATMENT_WEIGHT,
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
)
from report.rd_physical_paired import (
    ARMS,
    EXPECTED_EPOCHS,
    SEEDS,
    _class_metrics,
    _epoch_metrics,
    evaluate_completed_runs,
)


def config_path(seed: int, arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V12-cq08-liquid-dbb-cu-gate-dbb-auxiliary-{arm}-seed{seed}-config.json"
    )


def _semantic_errors(records: list[dict[str, Any]]) -> dict[str, Any]:
    directions = {
        "BirdTarget_to_CU": [],
        "UnknownTarget_to_DBB": [],
        "ClutterTarget_to_DBB": [],
        "DroneTarget_to_CU": [],
        "BalloonTarget_to_CU": [],
    }
    for row in records:
        label = str(row["label_name"])
        pred_id = int(row["pred_id"])
        key = None
        if label == "BirdTarget" and pred_id == 1:
            key = "BirdTarget_to_CU"
        elif label == "UnknownTarget" and pred_id == 0:
            key = "UnknownTarget_to_DBB"
        elif label == "ClutterTarget" and pred_id == 0:
            key = "ClutterTarget_to_DBB"
        elif label == "DroneTarget" and pred_id == 1:
            key = "DroneTarget_to_CU"
        elif label == "BalloonTarget" and pred_id == 1:
            key = "BalloonTarget_to_CU"
        if key is not None:
            directions[key].append(str(row["parent_sample_id"]))
    return {
        name: {"count": len(values), "parent_sample_ids": sorted(values)}
        for name, values in directions.items()
    }


def _load_arm(root: Path, seed: int, arm: str) -> dict[str, Any]:
    path = root / config_path(seed, arm)
    config = load_task_config(path)
    expected_weight = 0.0 if arm == "control" else DBB_AUXILIARY_TREATMENT_WEIGHT
    if (
        config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
        or config.train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL
        or config.train.paired_arm != arm
        or config.train.seed != seed
        or config.train.epochs != EXPECTED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.dbb_auxiliary_loss_weight != expected_weight
    ):
        raise ValueError(f"V12 paired report config violates the frozen protocol: {path}")
    artifact_dir = root / config.train.formal_artifact_dir
    summary_path = artifact_dir / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"V12 formal training is incomplete: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "passed"
        or summary.get("config_name") != config.config_name
        or summary.get("task_name") != "dbb_cu_gate"
        or summary.get("stop_reason") != "epoch_limit"
        or summary.get("formal_pretraining") != "not_required_and_not_used"
        or summary.get("test_evaluation") != "not_executed"
    ):
        raise ValueError(f"V12 summary violates completion/pretraining/test contract: {summary_path}")
    epochs = _epoch_metrics(summary)
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError("V12 best epoch is outside 1..20")
    best_from_summary = _class_metrics(dict(summary.get("best_validation_metrics") or {}))
    if best_from_summary != epochs[best_epoch]["parent_metrics"]:
        raise ValueError("V12 best metrics differ from the same history epoch")
    for epoch, record in epochs.items():
        expected_aux_weight = float(config.train.dbb_auxiliary_loss_weight)
        raw_record = summary["history"][epoch - 1]
        if float(raw_record.get("dbb_auxiliary_loss_weight", -1.0)) != expected_aux_weight:
            raise ValueError(f"V12 epoch {epoch} auxiliary loss weight drifted")
        if int(raw_record.get("dbb_auxiliary_selected_final_state_count", -1)) != 976:
            raise ValueError(f"V12 epoch {epoch} did not select all 976 DBB train parents exactly once")
        if int(raw_record.get("dbb_auxiliary_excluded_cu_parent_count", -1)) != 105:
            raise ValueError(f"V12 epoch {epoch} did not exclude all 105 CU train parents exactly once")
    expected_best = root / config.train.checkpoint_path
    expected_last = root / config.train.last_checkpoint_path
    if not expected_best.is_file() or not expected_last.is_file():
        raise FileNotFoundError("V12 best or last checkpoint is missing")
    if Path(str(summary.get("best_checkpoint", ""))).resolve() != expected_best.resolve():
        raise ValueError("V12 summary best checkpoint path differs from config")
    initialization = dict(summary.get("paired_initialization") or {})
    if (
        initialization.get("shared_parameters_and_buffers_exactly_equal") is not True
        or initialization.get("initial_cpu_reference_logits_exactly_equal") is not True
        or initialization.get("initial_auxiliary_logits_exactly_equal") is not True
    ):
        raise ValueError("V12 summary lacks exact paired initialization evidence")
    loader_contract = dict(summary.get("dataloader_contract") or {})
    if (
        int(loader_contract.get("seed", -1)) != seed
        or loader_contract.get("independent_generator") is not True
        or loader_contract.get("cuda_deterministic_algorithms") is not False
    ):
        raise ValueError("V12 DataLoader/determinism contract is invalid")
    validation_dataset = build_task_dataset(root, config, split_name=config.data.validation_split_name)
    device = task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    payload = load_task_checkpoint(
        expected_best,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(payload.get("epoch", -1)) != best_epoch:
        raise ValueError("V12 best checkpoint epoch differs from summary")
    independent = evaluate_task(model, validation_dataset, config=config, device=device)
    independent_metrics = _class_metrics(independent)
    if independent_metrics != best_from_summary:
        raise ValueError("V12 independent best-checkpoint validation differs from summary")
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    epoch_paths = [epoch_dir / f"epoch-{epoch:03d}.pt" for epoch in range(1, 21)]
    if any(not checkpoint.is_file() for checkpoint in epoch_paths):
        raise FileNotFoundError("V12 epoch checkpoints are incomplete")
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
            "parent_count": int(independent["parent_count"]),
            "window_count": int(independent["window_count"]),
            "metrics": independent_metrics,
            "semantic_errors": _semantic_errors(independent["records"]["parent"]),
            "matches_training_summary": True,
        },
        "epoch20_metrics": epochs[20]["parent_metrics"],
        "epochs": epochs,
        "paired_initialization": initialization,
        "loader_contract": loader_contract,
        "epoch_checkpoint_count": len(epoch_paths),
    }


def _targeted_mechanism_gate(runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seed_results: dict[str, Any] = {}
    bird_reductions: list[int] = []
    unknown_reductions: list[int] = []
    for seed in SEEDS:
        control_errors = runs[seed]["control"]["independent_best_checkpoint_validation"]["semantic_errors"]
        treatment_errors = runs[seed]["treatment"]["independent_best_checkpoint_validation"]["semantic_errors"]
        bird_reduction = int(control_errors["BirdTarget_to_CU"]["count"]) - int(
            treatment_errors["BirdTarget_to_CU"]["count"]
        )
        unknown_reduction = int(control_errors["UnknownTarget_to_DBB"]["count"]) - int(
            treatment_errors["UnknownTarget_to_DBB"]["count"]
        )
        passed = bird_reduction >= 1 and unknown_reduction >= 0
        bird_reductions.append(bird_reduction)
        unknown_reductions.append(unknown_reduction)
        seed_results[str(seed)] = {
            "passed": passed,
            "control": control_errors,
            "treatment": treatment_errors,
            "BirdTarget_to_CU_reduction": bird_reduction,
            "UnknownTarget_to_DBB_reduction": unknown_reduction,
        }
    sorted_bird = sorted(bird_reductions)
    sorted_unknown = sorted(unknown_reductions)
    median_bird_reduction = int(sorted_bird[1])
    median_unknown_reduction = int(sorted_unknown[1])
    passing_seed_count = sum(bool(row["passed"]) for row in seed_results.values())
    median_passed = median_bird_reduction >= 1 and median_unknown_reduction >= 0
    return {
        "passed": passing_seed_count >= 2 and median_passed,
        "required_passing_seed_count": 2,
        "passing_seed_count": passing_seed_count,
        "seed_results": seed_results,
        "median": {
            "BirdTarget_to_CU_reduction": median_bird_reduction,
            "UnknownTarget_to_DBB_reduction": median_unknown_reduction,
            "passed": median_passed,
        },
        "interpretation": (
            "mechanism guard only: protect Bird routing without increasing dangerous Unknown acceptance"
        ),
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    runs = {seed: {arm: _load_arm(root, seed, arm) for arm in ARMS} for seed in SEEDS}
    report = evaluate_completed_runs(runs, paired_protocol=DBB_AUXILIARY_PAIRED_PROTOCOL)
    mechanism_gate = _targeted_mechanism_gate(runs)
    generic_stage_one_passed = bool(report["stage_one_passed"])
    retention_passed = generic_stage_one_passed and bool(mechanism_gate["passed"])
    report.update(
        {
            "generic_stage_one_passed": generic_stage_one_passed,
            "targeted_mechanism_gate": mechanism_gate,
            "v12_retention_passed": retention_passed,
            "status": "passed" if retention_passed else "completed_but_gate_failed",
            "stage_two_status": (
                "eligible_for_separate_design_authorization_not_automatically_started"
                if retention_passed
                else "blocked_by_v12_retention_gate"
            ),
        }
    )
    report.update(
        {
            "schema_version": "RADAR-LN3-V12-dbb-auxiliary-paired-report-v1",
            "scope": "CQ08 V3 validation only; three paired seeds; test sealed",
            "formal_training": "completed_by_user_manual_commands",
            "intervention": {
                "control_dbb_auxiliary_loss_weight": 0.0,
                "treatment_dbb_auxiliary_loss_weight": DBB_AUXILIARY_TREATMENT_WEIGHT,
                "supervision": "DBB parent final causal state only; CU excluded",
                "rd_physical_v11_treatment": "disabled",
            },
        }
    )
    path = output_path or (
        root / "src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json"
    )
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed V12 DBB-auxiliary paired validation runs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
