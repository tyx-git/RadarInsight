"""Evaluate V18r density Treatment against the frozen V18 stable8 Control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.stable8_density_v18r import config_relative_path
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
    V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
    verify_task_data_contract,
)
from report.micro_doppler_v18_paired import (
    CLASS_NAMES,
    EXPECTED_EPOCHS,
    MIN_ABSOLUTE_IMPROVEMENT,
    STRICT_IMPROVEMENT_EPSILON,
    _class_floor,
    _comparison,
    _load_arm as _load_frozen_v18_arm,
    _parent_metrics,
    _strictly_exceeds_adoption_gain,
)
from report.stable8_density_v18r_readiness import (
    OUTPUT as READINESS_OUTPUT,
    SCHEMA_VERSION as READINESS_SCHEMA,
    SOURCE_BEST_CHECKPOINT_SHA256,
    SOURCE_SUMMARY_SHA256,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18r-stable8-density-weighted-paired-validation.json"
)


def _epoch_metrics(
    summary: Mapping[str, Any], readiness_sha256: str
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for record_value in summary.get("history") or []:
        record = dict(record_value)
        epoch = int(record.get("epoch", -1))
        if epoch in rows:
            raise ValueError(f"V18r training summary repeats epoch {epoch}")
        if record.get("v18r_training_readiness_sha256") != readiness_sha256:
            raise ValueError(f"V18r epoch {epoch} has a stale readiness binding")
        rows[epoch] = {
            "epoch": epoch,
            "global_step": int(record.get("global_step", -1)),
            "parent_order_sha256": str(record.get("parent_order_sha256", "")),
            "parent_metrics": _parent_metrics(dict(record.get("validation") or {})),
        }
    if set(rows) != set(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError("V18r summary must contain exactly epochs 1..20")
    return rows


def _load_treatment(root: Path) -> dict[str, Any]:
    config_path = root / config_relative_path("treatment")
    config = load_task_config(config_path)
    if (
        config.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or config.train.paired_protocol
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or config.train.paired_arm != "treatment"
        or config.model.liquid_invalid_dt_update_mode != "density_weighted"
        or config.model.rd_physical_dim != 4
        or config.model.micro_doppler_dim != 8
        or config.train.seed != 42
        or config.train.epochs != EXPECTED_EPOCHS
        or config.train.checkpoint_selection_metric
        != BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
    ):
        raise ValueError("V18r Treatment config violates the frozen protocol")
    contract = verify_task_data_contract(root, config)
    artifact = root / config.train.formal_artifact_dir
    summary_path = artifact / "training-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readiness = dict(summary.get("v18r_training_readiness") or {})
    readiness_path = Path(str(readiness.get("path", "")))
    if (
        summary.get("status") != "completed"
        or summary.get("config_name") != config.config_name
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("paired_protocol")
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != readiness.get("sha256")
    ):
        raise ValueError("V18r Treatment summary/readiness violates sealed scope")
    epochs = _epoch_metrics(summary, str(readiness["sha256"]))
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch not in epochs:
        raise ValueError("V18r Treatment best epoch is outside 1..20")
    best_metrics = _parent_metrics(dict(summary.get("best_validation_metrics") or {}))
    if best_metrics != epochs[best_epoch]["parent_metrics"]:
        raise ValueError("V18r Treatment best metrics differ from its history")
    best_path = root / config.train.checkpoint_path
    last_path = root / config.train.last_checkpoint_path
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    if (
        not best_path.is_file()
        or not last_path.is_file()
        or any(
            not (epoch_dir / f"epoch-{epoch:03d}.pt").is_file()
            for epoch in range(1, 21)
        )
    ):
        raise FileNotFoundError("V18r Treatment checkpoints are incomplete")
    initialization = dict(summary.get("paired_initialization") or {})
    if (
        initialization.get("shared_parameters_and_buffers_exactly_equal") is not True
        or initialization.get("stable8_fusion_columns_exactly_zero") is not True
        or initialization.get("initial_pre_treatment_exposure_logits_exactly_equal")
        is not True
    ):
        raise ValueError("V18r Treatment lacks valid paired initialization evidence")
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
        raise ValueError("V18r Treatment best checkpoint epoch differs from summary")
    independent = evaluate_task(model, dataset, config=config, device=device)
    independent_metrics = _parent_metrics(independent)
    if independent_metrics != best_metrics:
        raise ValueError("V18r independent best checkpoint evaluation differs")
    return {
        "arm": "treatment",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
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
        "v18r_training_readiness": readiness,
        "independent_best_checkpoint_validation_matches": True,
        "contract_hashes": contract["hashes"],
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    readiness_path = root / READINESS_OUTPUT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V18r paired report requires the passed readiness")
    control = _load_frozen_v18_arm(root, "treatment")
    treatment = _load_treatment(root)
    if (
        control["summary_sha256"] != SOURCE_SUMMARY_SHA256
        or control["best_checkpoint_sha256"] != SOURCE_BEST_CHECKPOINT_SHA256
        or control["contract_hashes"] != treatment["contract_hashes"]
    ):
        raise ValueError("V18r frozen Control binding or data contract changed")
    order_mismatch = [
        epoch
        for epoch in range(1, 21)
        if control["epochs"][epoch]["parent_order_sha256"]
        != treatment["epochs"][epoch]["parent_order_sha256"]
    ]
    if order_mismatch:
        raise ValueError(f"V18r parent order differs at epochs {order_mismatch}")
    if (
        control["paired_initialization"].get("initial_logits_sha256")
        != treatment["paired_initialization"].get("initial_logits_sha256")
    ):
        raise ValueError("V18r does not reproduce the frozen V18 initial logits")

    own_best = _comparison(control["best_metrics"], treatment["best_metrics"])
    guard_epochs = sorted({control["best_epoch"], treatment["best_epoch"], 20})
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
    class_thresholds = {
        class_name: {
            "floor": _class_floor(treatment["best_metrics"], class_name),
            "target": 0.95,
            "passed": _class_floor(treatment["best_metrics"], class_name) >= 0.95,
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
        "frozen_v18_initial_logits_reproduced": True,
        "frozen_v18_control_not_retrained": True,
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "completed_but_gate_failed",
        "schema_version": "RADAR-LN3-V18r-stable8-density-paired-validation-v1",
        "paired_protocol": V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
        "scope": "DBB specialist development validation parent-final only",
        "control_definition": "frozen completed V18 stable8 Treatment",
        "treatment_definition": "same stable8 initialization with density-weighted zero-delta dynamics",
        "strict_lower_bound_absolute_improvement_for_adoption": (
            MIN_ABSOLUTE_IMPROVEMENT
        ),
        "strict_improvement_comparison_epsilon": STRICT_IMPROVEMENT_EPSILON,
        "checks": checks,
        "class_threshold_checks": class_thresholds,
        "own_best_primary_comparison": own_best,
        "same_epoch_attribution_guards": same_epoch,
        "same_epoch_passing_count": same_epoch_pass_count,
        "runs": {
            "control": {key: value for key, value in control.items() if key != "epochs"},
            "treatment": {
                key: value for key, value in treatment.items() if key != "epochs"
            },
        },
        "conclusion": (
            "V18r passes the frozen single-seed local DBB adoption gate; multi-seed confirmation is required before replacing the V18 stable8 baseline."
            if passed
            else "V18r does not pass the frozen local DBB adoption gate; retain V18 stable8 as the feature baseline."
        ),
        "not_final_five_class_performance": True,
        "formal_training": "已执行（仅V18r Treatment）",
        "control_retrained": False,
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
    print(json.dumps({**report, "report_path": str(output)}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V18r density against frozen V18 stable8."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root, args.output)


if __name__ == "__main__":
    main()
