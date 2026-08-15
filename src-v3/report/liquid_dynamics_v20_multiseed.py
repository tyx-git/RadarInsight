"""Aggregate the frozen V20r seed42/43/44 paired confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.task_models import V20_DENSITY_PAIRED_PROTOCOL, sha256_file
from report.liquid_dynamics_v20_paired import (
    CLASS_NAMES,
    EXPECTED_EPOCHS,
    MIN_ABSOLUTE_IMPROVEMENT,
    _class_floor,
    _comparison,
    _density_shared_control_contract,
    _initialization_contract_is_valid,
    _load_arm,
)


SEEDS = (42, 43, 44)
READINESS_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness.json"
)
READINESS_SCHEMA = (
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1"
)
OUTPUT_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-paired-validation.json"
)


def _seed_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    treatment = comparison["treatment"]
    delta = comparison["treatment_minus_control"]
    class_thresholds = {
        name: {
            "floor": _class_floor(treatment, name),
            "target": 0.95,
            "passed": _class_floor(treatment, name) >= 0.95,
        }
        for name in CLASS_NAMES
    }
    checks = {
        "all_treatment_classes_precision_recall_f1_at_least_0_95": all(
            row["passed"] for row in class_thresholds.values()
        ),
        "macro_f1_improves_by_more_than_0_02": (
            float(delta["macro_f1"]) > MIN_ABSOLUTE_IMPROVEMENT
        ),
        "drone_bird_weak_floor_improves_by_more_than_0_02": (
            float(delta["drone_bird_weak_floor"]) > MIN_ABSOLUTE_IMPROVEMENT
        ),
        "drone_to_bird_not_increased": (
            int(delta["Drone_to_Bird_error_reduction"]) >= 0
        ),
        "bird_to_drone_not_increased": (
            int(delta["Bird_to_Drone_error_reduction"]) >= 0
        ),
        "bidirectional_drone_bird_errors_reduced": (
            int(delta["Drone_Bird_bidirectional_error_reduction"]) >= 1
        ),
    }
    return {
        "full_gate_passed": all(checks.values()),
        "primary_macro_signal_passed": checks[
            "macro_f1_improves_by_more_than_0_02"
        ],
        "checks": checks,
        "class_threshold_checks": class_thresholds,
    }


def _median(values: list[float]) -> float:
    result = float(np.median(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise ValueError("V20r multiseed median is not finite")
    return result


def _aggregate_deltas(seed_results: Mapping[str, Any]) -> dict[str, Any]:
    names = tuple(
        seed_results[str(SEEDS[0])]["own_best_primary_comparison"][
            "treatment_minus_control"
        ]
    )
    aggregate: dict[str, Any] = {}
    for name in names:
        values = [
            float(
                seed_results[str(seed)]["own_best_primary_comparison"][
                    "treatment_minus_control"
                ][name]
            )
            for seed in SEEDS
        ]
        aggregate[name] = {
            "values_by_seed": {
                str(seed): values[index] for index, seed in enumerate(SEEDS)
            },
            "mean": float(np.mean(values)),
            "median": _median(values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return aggregate


def _median_gate(seed_results: Mapping[str, Any], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    treatment_metric_medians: dict[str, Any] = {}
    class_passes: dict[str, bool] = {}
    for name in CLASS_NAMES:
        rows = [
            seed_results[str(seed)]["own_best_primary_comparison"]["treatment"][
                "per_class"
            ][name]
            for seed in SEEDS
        ]
        medians = {
            metric: _median([float(row[metric]) for row in rows])
            for metric in ("precision", "recall", "f1")
        }
        treatment_metric_medians[name] = medians
        class_passes[name] = min(medians.values()) >= 0.95

    checks = {
        "median_treatment_classes_precision_recall_f1_at_least_0_95": all(
            class_passes.values()
        ),
        "median_macro_f1_improvement_more_than_0_02": (
            float(aggregate["macro_f1"]["median"]) > MIN_ABSOLUTE_IMPROVEMENT
        ),
        "median_drone_bird_weak_floor_improvement_more_than_0_02": (
            float(aggregate["drone_bird_weak_floor"]["median"])
            > MIN_ABSOLUTE_IMPROVEMENT
        ),
        "median_drone_to_bird_not_increased": (
            float(aggregate["Drone_to_Bird_error_reduction"]["median"]) >= 0.0
        ),
        "median_bird_to_drone_not_increased": (
            float(aggregate["Bird_to_Drone_error_reduction"]["median"]) >= 0.0
        ),
        "median_bidirectional_drone_bird_errors_reduced": (
            float(
                aggregate["Drone_Bird_bidirectional_error_reduction"]["median"]
            )
            >= 1.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "treatment_per_class_metric_medians": treatment_metric_medians,
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    readiness_path = root / READINESS_PATH
    if not readiness_path.is_file():
        raise FileNotFoundError(
            f"run multiseed prepare before evaluation: {readiness_path}"
        )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("paired_protocol") != V20_DENSITY_PAIRED_PROTOCOL
        or tuple(readiness.get("seed_set") or ()) != SEEDS
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or readiness.get("test_payload_access_count") != 0
    ):
        raise ValueError("V20r multiseed readiness is stale, failed, or unsealed")

    shared_seed42 = _density_shared_control_contract(root)
    runs: dict[int, dict[str, Any]] = {
        42: {
            "control": _load_arm(root, "skip", "control", seed=42),
            "treatment": _load_arm(root, "density", "treatment", seed=42),
        }
    }
    for seed in (43, 44):
        runs[seed] = {
            arm: _load_arm(root, "density", arm, seed=seed)
            for arm in ("control", "treatment")
        }

    contract_hashes = runs[42]["control"]["contract_hashes"]
    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        control = runs[seed]["control"]
        treatment = runs[seed]["treatment"]
        if (
            control["contract_hashes"] != contract_hashes
            or treatment["contract_hashes"] != contract_hashes
        ):
            raise ValueError(f"seed{seed} does not share the frozen V16 data contract")
        order_mismatches = [
            epoch
            for epoch in range(1, EXPECTED_EPOCHS + 1)
            if control["epochs"][epoch]["parent_order_sha256"]
            != treatment["epochs"][epoch]["parent_order_sha256"]
        ]
        if order_mismatches:
            raise ValueError(
                f"seed{seed} Control/Treatment parent order differs: {order_mismatches}"
            )
        control_init = control["paired_initialization"]
        treatment_init = treatment["paired_initialization"]
        if not (
            _initialization_contract_is_valid(control_init)
            and _initialization_contract_is_valid(treatment_init)
            and control_init.get("initial_logits_sha256")
            == treatment_init.get("initial_logits_sha256")
        ):
            raise ValueError(f"seed{seed} paired initialization evidence is invalid")

        comparison = _comparison(control["best_metrics"], treatment["best_metrics"])
        gate = _seed_gate(comparison)
        guard_epochs = sorted(
            {control["best_epoch"], treatment["best_epoch"], EXPECTED_EPOCHS}
        )
        same_epoch = {
            str(epoch): _comparison(
                control["epochs"][epoch]["parent_metrics"],
                treatment["epochs"][epoch]["parent_metrics"],
            )
            for epoch in guard_epochs
        }
        seed_results[str(seed)] = {
            "seed": seed,
            "parent_order_hashes_equal_all_20_epochs": True,
            "initial_cpu_logit_hash_equal": True,
            "gate": gate,
            "own_best_primary_comparison": comparison,
            "same_epoch_attribution_guards": same_epoch,
            "runs": {
                arm: {
                    key: value
                    for key, value in runs[seed][arm].items()
                    if key != "epochs"
                }
                for arm in ("control", "treatment")
            },
        }

    aggregate = _aggregate_deltas(seed_results)
    median_gate = _median_gate(seed_results, aggregate)
    full_gate_passing_seed_count = sum(
        bool(seed_results[str(seed)]["gate"]["full_gate_passed"])
        for seed in SEEDS
    )
    primary_signal_passing_seed_count = sum(
        bool(seed_results[str(seed)]["gate"]["primary_macro_signal_passed"])
        for seed in SEEDS
    )
    primary_signal_confirmed = bool(
        primary_signal_passing_seed_count >= 2
        and aggregate["macro_f1"]["median"] > MIN_ABSOLUTE_IMPROVEMENT
    )
    full_adoption_confirmed = bool(
        full_gate_passing_seed_count >= 2 and median_gate["passed"]
    )
    status = (
        "passed"
        if full_adoption_confirmed
        else (
            "primary_signal_confirmed_but_full_gate_failed"
            if primary_signal_confirmed
            else "completed_but_replication_failed"
        )
    )
    report = {
        "status": status,
        "schema_version": (
            "RADAR-LN3-V20-density-weighted-multiseed-paired-validation-v1"
        ),
        "experiment": "V20r-density-multiseed-confirmation",
        "paired_protocol": V20_DENSITY_PAIRED_PROTOCOL,
        "scope": "DBB specialist validation parent-final local task only",
        "seed_set": list(SEEDS),
        "seed_results": seed_results,
        "aggregate_treatment_minus_control": aggregate,
        "primary_signal_passing_seed_count": primary_signal_passing_seed_count,
        "required_primary_signal_passing_seed_count": 2,
        "primary_signal_confirmed": primary_signal_confirmed,
        "full_gate_passing_seed_count": full_gate_passing_seed_count,
        "required_full_gate_passing_seed_count": 2,
        "median_gate": median_gate,
        "full_adoption_confirmed": full_adoption_confirmed,
        "shared_seed42_control_reuse": shared_seed42,
        "readiness": {
            "path": str(readiness_path),
            "sha256": sha256_file(readiness_path),
            "schema_version": READINESS_SCHEMA,
        },
        "interpretation_contract": {
            "seed_role": (
                "42/43/44 are optimization replicates over the same validation parents; "
                "they are not independent data samples"
            ),
            "no_three_seed_p_value": True,
            "primary_signal_does_not_override_full_safety_gate": True,
            "test_or_oof_not_used_for_seed_confirmation": True,
        },
        "conclusion": (
            "V20r passes the frozen three-seed full adoption contract."
            if full_adoption_confirmed
            else (
                "V20r replicates the >2pp parent Macro-F1 signal, but does not pass "
                "the frozen class-floor/directional full adoption contract."
                if primary_signal_confirmed
                else "V20r does not replicate the >2pp parent Macro-F1 signal across seeds."
            )
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
    output = output_path or root / OUTPUT_PATH
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate completed V20r seed42/43/44 paired runs."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root, args.output)


if __name__ == "__main__":
    main()
