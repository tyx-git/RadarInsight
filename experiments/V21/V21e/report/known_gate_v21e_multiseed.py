"""Aggregate the frozen V21e seed42/43/44 paired confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build.known_gate_v21_multiseed import (
    artifact_relative as baseline_artifact_relative,
    config_relative as baseline_config_relative,
)
from build.known_gate_v21e import (
    CONFIG as SEED43_TREATMENT_CONFIG,
    READINESS as SEED43_TREATMENT_READINESS,
)
from build.known_gate_v21e_multiseed import (
    NEW_TRAINING_SEEDS,
    READINESS,
    SEEDS,
    SEED43_TREATMENT_ARTIFACT,
    treatment_artifact_relative,
    treatment_config_relative,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
    sha256_file,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_features import V21E_COURSE_PREFIX_VARIANT
from report.known_gate_v21e import (
    LABELS,
    PRIMARY_DELTA,
    _base_label_outcomes,
    _bootstrap,
    _metric_vector,
    _probability_attribution,
    _same_epoch_comparison,
    _scope_changes,
    _trained_feature_column,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
    "multiseed-paired-validation.json"
)


def _load_run(
    root: Path,
    *,
    artifact_relative: Path,
    variant: str,
    seed: int,
    expected_config_path: Path,
    expected_readiness_sha256: str | None,
) -> dict[str, Any]:
    artifact = root / artifact_relative
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError(f"V21e seed{seed} run is incomplete: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    inferred_seed = int(
        summary.get(
            "seed",
            dict(summary.get("dataloader_contract") or {}).get("seed", -1),
        )
    )
    detail_seed = detail.get("seed")
    records_by_scope = {
        scope: list(dict(detail.get("records") or {}).get(scope) or [])
        for scope in ("parent", "window", "eligible_prefix")
    }
    readiness = dict(summary.get("training_readiness") or {})
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != variant
        or inferred_seed != seed
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or len(list(summary.get("history") or [])) != 20
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or detail.get("variant") != variant
        or (detail_seed is not None and int(detail_seed) != seed)
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -1))
        or len(records_by_scope["parent"]) != 232
        or len(records_by_scope["window"]) != 4452
        or len(records_by_scope["eligible_prefix"]) != 3756
        or Path(str(summary.get("config_path", ""))).resolve()
        != expected_config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or (
            expected_readiness_sha256 is not None
            and readiness.get("sha256") != expected_readiness_sha256
        )
    ):
        raise ValueError(f"V21e seed{seed} {variant} run violates the full contract")
    metrics_by_scope = {
        scope: summarize_known_gate_records(
            records,
            scope=f"seed{seed}_{variant}_{scope}_recomputed",
        )
        for scope, records in records_by_scope.items()
    }
    recorded = dict(dict(summary["best_validation_metrics"])["parent_metrics"])
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(metrics_by_scope["parent"][field]) - float(recorded[field])) > 1.0e-12:
            raise ValueError(f"V21e seed{seed} recomputed metric differs at {field}")
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "summary": summary,
        "records_by_scope": records_by_scope,
        "metrics_by_scope": metrics_by_scope,
    }


def _delta_vector(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left = _metric_vector(before)
    right = _metric_vector(after)
    return {key: float(right[key] - left[key]) for key in left}


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (3,) or not np.isfinite(data).all():
        raise ValueError("V21e multiseed aggregate requires three finite values")
    return {
        "values_by_seed": {
            str(seed): float(data[index]) for index, seed in enumerate(SEEDS)
        },
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
    }


def aggregate_decision(seed_results: Mapping[str, Any]) -> dict[str, Any]:
    if set(seed_results) != {str(seed) for seed in SEEDS}:
        raise ValueError("V21e multiseed decision requires seeds 42/43/44")
    parent_keys = tuple(
        dict(seed_results[str(SEEDS[0])]["scope_deltas"]["parent"])
    )
    aggregate_by_scope: dict[str, Any] = {}
    for scope in ("parent", "window", "eligible_prefix"):
        aggregate_by_scope[scope] = {
            key: _numeric_summary(
                [
                    float(seed_results[str(seed)]["scope_deltas"][scope][key])
                    for seed in SEEDS
                ]
            )
            for key in parent_keys
        }
    label_net_rescue = {
        label: _numeric_summary(
            [
                float(
                    seed_results[str(seed)]["parent_base_label_outcomes"][label][
                        "net_rescue"
                    ]
                )
                for seed in SEEDS
            ]
        )
        for label in LABELS
    }
    combined_net_rescue = _numeric_summary(
        [
            float(seed_results[str(seed)]["combined_parent_net_rescue"])
            for seed in SEEDS
        ]
    )
    primary_count = sum(
        bool(seed_results[str(seed)]["gate"]["primary_macro_signal_passed"])
        for seed in SEEDS
    )
    robust_count = sum(
        bool(seed_results[str(seed)]["gate"]["paired_bootstrap_robust"])
        for seed in SEEDS
    )
    direction_count = sum(
        bool(seed_results[str(seed)]["gate"]["parent_direction_safety_passed"])
        for seed in SEEDS
    )
    prefix_count = sum(
        bool(seed_results[str(seed)]["gate"]["eligible_prefix_guard_passed"])
        for seed in SEEDS
    )
    parent_final_replication = bool(
        primary_count >= 2
        and aggregate_by_scope["parent"]["macro_f1"]["median"] > PRIMARY_DELTA
    )
    direction_replication = bool(
        direction_count >= 2
        and combined_net_rescue["median"] > 0.0
        and all(row["median"] >= 0.0 for row in label_net_rescue.values())
        and aggregate_by_scope["parent"]["known_target_recall"]["median"] >= 0.0
        and aggregate_by_scope["parent"]["known_clutter_recall"]["median"] >= 0.0
    )
    bootstrap_replication = robust_count >= 2
    parent_final_confirmation = bool(
        parent_final_replication and direction_replication and bootstrap_replication
    )
    online_prefix_guard = bool(
        prefix_count >= 2
        and aggregate_by_scope["eligible_prefix"]["macro_f1"]["median"] >= 0.0
        and aggregate_by_scope["eligible_prefix"]["known_target_recall"]["median"]
        >= 0.0
        and aggregate_by_scope["eligible_prefix"]["known_clutter_recall"]["median"]
        >= 0.0
    )
    return {
        "aggregate_treatment_minus_baseline_by_scope": aggregate_by_scope,
        "aggregate_parent_base_label_net_rescue": label_net_rescue,
        "aggregate_combined_parent_net_rescue": combined_net_rescue,
        "primary_macro_signal_passing_seed_count": primary_count,
        "paired_bootstrap_robust_seed_count": robust_count,
        "parent_direction_safety_passing_seed_count": direction_count,
        "eligible_prefix_guard_passing_seed_count": prefix_count,
        "required_passing_seed_count": 2,
        "parent_final_replication_confirmed": parent_final_replication,
        "direction_replication_confirmed": direction_replication,
        "bootstrap_replication_confirmed": bootstrap_replication,
        "parent_final_confirmation_passed": parent_final_confirmation,
        "online_eligible_prefix_guard_passed": online_prefix_guard,
        "advance_to_train_parent_grouped_oof": parent_final_confirmation,
        "online_known_gate_adoption_allowed": False,
    }


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    readiness_path = root / READINESS
    seed43_readiness_path = root / SEED43_TREATMENT_READINESS
    if not readiness_path.is_file() or not seed43_readiness_path.is_file():
        raise FileNotFoundError("run V21e multiseed prepare before evaluation")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA
        or tuple(readiness.get("seed_set") or ()) != SEEDS
        or tuple(readiness.get("new_training_seeds") or ()) != NEW_TRAINING_SEEDS
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V21e multiseed readiness is stale, failed or unsealed")
    readiness_hash = sha256_file(readiness_path)
    seed43_readiness_hash = sha256_file(seed43_readiness_path)

    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        baseline_config = root / baseline_config_relative(seed)
        baseline = _load_run(
            root,
            artifact_relative=baseline_artifact_relative(seed),
            variant="baseline",
            seed=seed,
            expected_config_path=baseline_config,
            expected_readiness_sha256=None,
        )
        treatment_config = (
            root / SEED43_TREATMENT_CONFIG
            if seed == 43
            else root / treatment_config_relative(seed)
        )
        treatment_artifact = (
            SEED43_TREATMENT_ARTIFACT
            if seed == 43
            else treatment_artifact_relative(seed)
        )
        treatment = _load_run(
            root,
            artifact_relative=treatment_artifact,
            variant=V21E_COURSE_PREFIX_VARIANT,
            seed=seed,
            expected_config_path=treatment_config,
            expected_readiness_sha256=(
                seed43_readiness_hash if seed == 43 else readiness_hash
            ),
        )
        pair = dict(treatment["summary"].get("paired_initialization") or {})
        if (
            pair.get("paired_baseline") is not True
            or pair.get("shared_parameters_and_buffers_exactly_equal") is not True
            or pair.get("new_fusion_columns_exactly_zero") is not True
            or pair.get("initial_logits_exactly_equal") is not True
            or int(pair.get("new_fusion_columns", -1)) != 1
            or int(pair.get("seed", -1)) != seed
            or Path(str(pair.get("baseline_config_path", ""))).resolve()
            != baseline_config.resolve()
        ):
            raise ValueError(f"V21e seed{seed} paired initialization evidence is invalid")
        baseline_orders = list(
            dict(baseline["summary"].get("dataloader_contract") or {}).get(
                "epoch_parent_order_hashes"
            )
            or []
        )
        treatment_orders = list(
            dict(treatment["summary"].get("dataloader_contract") or {}).get(
                "epoch_parent_order_hashes"
            )
            or []
        )
        if baseline_orders != treatment_orders or len(baseline_orders) != 20:
            raise ValueError(f"V21e seed{seed} Baseline/Treatment parent orders differ")

        scope_deltas = {
            scope: _delta_vector(
                baseline["metrics_by_scope"][scope],
                treatment["metrics_by_scope"][scope],
            )
            for scope in ("parent", "window", "eligible_prefix")
        }
        parent_outcomes = _base_label_outcomes(
            baseline["records_by_scope"]["parent"],
            treatment["records_by_scope"]["parent"],
        )
        prefix_outcomes = _base_label_outcomes(
            baseline["records_by_scope"]["eligible_prefix"],
            treatment["records_by_scope"]["eligible_prefix"],
            scope="eligible_prefix",
        )
        combined = int(sum(parent_outcomes[label]["net_rescue"] for label in LABELS))
        bootstrap = _bootstrap(
            baseline["records_by_scope"]["parent"],
            treatment["records_by_scope"]["parent"],
        )
        primary = float(scope_deltas["parent"]["macro_f1"]) > PRIMARY_DELTA
        each_label_safe = all(
            int(parent_outcomes[label]["net_rescue"]) >= 0 for label in LABELS
        )
        parent_direction = bool(
            combined > 0
            and each_label_safe
            and float(scope_deltas["parent"]["known_target_recall"]) >= 0.0
            and float(scope_deltas["parent"]["known_clutter_recall"]) >= 0.0
        )
        robust = bool(
            float(
                bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"]
            )
            > 0.0
        )
        prefix_guard = bool(
            float(scope_deltas["eligible_prefix"]["macro_f1"]) >= 0.0
            and float(scope_deltas["eligible_prefix"]["known_target_recall"]) >= 0.0
            and float(scope_deltas["eligible_prefix"]["known_clutter_recall"]) >= 0.0
        )
        seed_results[str(seed)] = {
            "seed": seed,
            "best_epochs": {
                "baseline": int(baseline["summary"]["best_epoch"]),
                "treatment": int(treatment["summary"]["best_epoch"]),
            },
            "parent_order_hashes_equal_all_20_epochs": True,
            "initial_logits_exactly_equal": True,
            "baseline_metrics_by_scope": baseline["metrics_by_scope"],
            "treatment_metrics_by_scope": treatment["metrics_by_scope"],
            "scope_deltas": scope_deltas,
            "parent_base_label_outcomes": parent_outcomes,
            "eligible_prefix_base_label_outcomes": prefix_outcomes,
            "combined_parent_net_rescue": combined,
            "scope_prediction_changes": _scope_changes(
                baseline["records_by_scope"], treatment["records_by_scope"]
            ),
            "probability_attribution": _probability_attribution(
                baseline["records_by_scope"]["parent"],
                treatment["records_by_scope"]["parent"],
            ),
            "paired_parent_bootstrap": bootstrap,
            "same_epoch_comparison": _same_epoch_comparison(
                baseline["summary"], treatment["summary"]
            ),
            "trained_course_fusion_column": _trained_feature_column(
                Path(treatment["checkpoint_path"])
            ),
            "gate": {
                "primary_macro_signal_passed": primary,
                "parent_direction_safety_passed": parent_direction,
                "paired_bootstrap_robust": robust,
                "eligible_prefix_guard_passed": prefix_guard,
                "full_parent_seed_gate_passed": primary and parent_direction and robust,
            },
            "runs": {
                "baseline": {
                    key: baseline[key]
                    for key in (
                        "summary_path",
                        "summary_sha256",
                        "checkpoint_path",
                        "checkpoint_sha256",
                    )
                },
                "treatment": {
                    key: treatment[key]
                    for key in (
                        "summary_path",
                        "summary_sha256",
                        "checkpoint_path",
                        "checkpoint_sha256",
                    )
                },
            },
        }

    decision = aggregate_decision(seed_results)
    if decision["parent_final_confirmation_passed"]:
        status = (
            "passed"
            if decision["online_eligible_prefix_guard_passed"]
            else "parent_final_confirmed_online_prefix_guard_failed"
        )
    else:
        status = "completed_but_replication_failed"
    report = {
        "status": status,
        "schema_version": (
            "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
            "multiseed-paired-validation-v1"
        ),
        "experiment": "V21e-causal-course-multiseed-confirmation",
        "scope": "development validation KnownGate local task only",
        "seed_set": list(SEEDS),
        "seed_results": seed_results,
        "decision": decision,
        "readiness": {
            "path": str(readiness_path.resolve()),
            "sha256": readiness_hash,
            "schema_version": KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
        },
        "interpretation_contract": {
            "seed_role": (
                "seeds 42/43/44 are optimization replicates over the same 232 "
                "development parents, not independent data samples"
            ),
            "clutter_support_per_seed": 9,
            "no_three_seed_p_value": True,
            "post_hoc_feature_discovery": True,
            "same_development_validation_used_for_feature_discovery": True,
            "parent_final_confirmation_does_not_replace_prefix_stable_or_open_set_validation": True,
            "new_independent_holdout_required": True,
        },
        "conclusion": (
            "V21e replicates its frozen parent-final signal and may advance to train "
            "parent-grouped OOF; online use remains blocked until prefix/stable and new-holdout checks."
            if decision["parent_final_confirmation_passed"]
            else "V21e does not replicate the frozen parent-final signal across seeds."
        ),
        "unknown_is_model_class": False,
        "local_argmax_is_product_action": False,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = output_path or root / OUTPUT
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate completed V21e seed42/43/44 paired runs."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root, args.output)


if __name__ == "__main__":
    main()
