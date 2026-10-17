from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build.known_gate_v21a import (
    ARTIFACT,
    BASELINE_ARTIFACT,
    BASELINE_CONFIG,
    BASELINE_SELECTION,
    CONFIG,
    EXPERIMENT_CONTRACT,
    FEATURE_MANIFEST,
    READINESS,
)
from radar_ln3.known_gate import sha256_file, summarize_known_gate_records
from radar_ln3.known_gate_features import (
    INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21A_INTERFRAME_VARIANT,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21a-known-gate-interframe-spectral-paired-validation.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 21143
PRIMARY_DELTA = 0.02
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
EXPECTED_DIRECTION_LABELS = ("BirdTarget", "BalloonTarget", "ClutterTarget")


def _load_run(
    root: Path,
    artifact_relative: Path,
    *,
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
        raise FileNotFoundError(f"V21a paired run is incomplete: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    records = list(dict(detail.get("records") or {}).get("parent") or [])
    inferred_seed = int(
        summary.get("seed", dict(summary.get("dataloader_contract") or {}).get("seed", -1))
    )
    training_readiness = dict(summary.get("training_readiness") or {})
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
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -1))
        or len(records) != 232
        or Path(str(summary.get("config_path", ""))).resolve()
        != expected_config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or Path(str(summary.get("sample_detail_information", ""))).resolve()
        != detail_path.resolve()
        or (
            expected_readiness_sha256 is not None
            and training_readiness.get("sha256") != expected_readiness_sha256
        )
    ):
        raise ValueError(f"V21a {variant} run violates the full local contract")
    parent_ids = [str(row["parent_sample_id"]) for row in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"V21a {variant} sample detail repeats a parent")
    metrics = summarize_known_gate_records(records, scope=f"{variant}_parent_final_recomputed")
    recorded = dict(dict(summary["best_validation_metrics"])["parent_metrics"])
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(metrics[field]) - float(recorded[field])) > 1.0e-12:
            raise ValueError(f"V21a {variant} recomputed metric differs at {field}")
    return {
        "artifact": str(artifact),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "summary": summary,
        "records": records,
        "metrics": metrics,
    }


def _ordered_pair(
    baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {str(row["parent_sample_id"]): dict(row) for row in baseline}
    after = {str(row["parent_sample_id"]): dict(row) for row in treatment}
    if set(before) != set(after):
        raise ValueError("V21a paired validation parent sets differ")
    ordered_ids = sorted(before)
    for parent_id in ordered_ids:
        if (
            before[parent_id]["label_name"] != after[parent_id]["label_name"]
            or int(before[parent_id]["true_id"]) != int(after[parent_id]["true_id"])
        ):
            raise ValueError("V21a paired parent label/true-id differs")
    return [before[value] for value in ordered_ids], [after[value] for value in ordered_ids]


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    target = rows["KNOWN_TARGET_EVIDENCE"]
    clutter = rows["KNOWN_CLUTTER_EVIDENCE"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_precision": float(target["precision"]),
        "known_target_recall": float(target["recall"]),
        "known_target_f1": float(target["f1"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
        "weak_class_precision_recall_f1_floor": min(
            float(row[field])
            for row in metrics["per_class"]
            for field in ("precision", "recall", "f1")
        ),
    }


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(
    baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment)
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(before) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in LABELS
    }
    expected = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }
    if {label: len(indices) for label, indices in strata.items()} != expected:
        raise ValueError("V21a bootstrap support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    before_values: dict[str, list[float]] = {}
    after_values: dict[str, list[float]] = {}
    delta_values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        before_metrics = summarize_known_gate_records(
            [before[int(index)] for index in sampled], scope="v21a_parent_bootstrap"
        )
        after_metrics = summarize_known_gate_records(
            [after[int(index)] for index in sampled], scope="v21a_parent_bootstrap"
        )
        before_vector = _metric_vector(before_metrics)
        after_vector = _metric_vector(after_metrics)
        for key in before_vector:
            before_values.setdefault(key, []).append(before_vector[key])
            after_values.setdefault(key, []).append(after_vector[key])
            delta_values.setdefault(key, []).append(after_vector[key] - before_vector[key])
    return {
        "method": "paired parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": expected,
        "baseline": {key: _ci(values) for key, values in before_values.items()},
        "treatment": {key: _ci(values) for key, values in after_values.items()},
        "treatment_minus_baseline": {
            key: _ci(values) for key, values in delta_values.items()
        },
    }


def _base_label_outcomes(
    baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment)
    result: dict[str, Any] = {}
    for label in LABELS:
        indices = [index for index, row in enumerate(before) if row["label_name"] == label]
        rescued = harmed = changed = 0
        for index in indices:
            before_correct = int(before[index]["pred_id"]) == int(before[index]["true_id"])
            after_correct = int(after[index]["pred_id"]) == int(after[index]["true_id"])
            changed += int(int(before[index]["pred_id"]) != int(after[index]["pred_id"]))
            rescued += int((not before_correct) and after_correct)
            harmed += int(before_correct and (not after_correct))
        result[label] = {
            "support": len(indices),
            "changed": int(changed),
            "rescued": int(rescued),
            "harmed": int(harmed),
            "net_rescue": int(rescued - harmed),
        }
    return result


def _same_epoch_comparison(
    baseline_summary: Mapping[str, Any], treatment_summary: Mapping[str, Any]
) -> dict[str, Any]:
    before = {
        int(row["epoch"]): dict(row)
        for row in list(baseline_summary.get("history") or [])
    }
    after = {
        int(row["epoch"]): dict(row)
        for row in list(treatment_summary.get("history") or [])
    }
    if set(before) != set(range(1, 21)) or set(after) != set(before):
        raise ValueError("V21a same-epoch comparison requires complete epochs 1..20")
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 21):
        before_validation = dict(before[epoch]["validation"])
        after_validation = dict(after[epoch]["validation"])
        before_metrics = dict(before_validation["parent_metrics"])
        after_metrics = dict(after_validation["parent_metrics"])
        before_vector = _metric_vector(before_metrics)
        after_vector = _metric_vector(after_metrics)
        before_labels = dict(before_validation["base_label_parent_metrics"])
        after_labels = dict(after_validation["base_label_parent_metrics"])
        expected_direction_delta = sum(
            float(after_labels[label]["acceptance_recall"])
            - float(before_labels[label]["acceptance_recall"])
            for label in EXPECTED_DIRECTION_LABELS
        )
        rows.append(
            {
                "epoch": epoch,
                "baseline_macro_f1": before_vector["macro_f1"],
                "treatment_macro_f1": after_vector["macro_f1"],
                "macro_f1_delta": after_vector["macro_f1"] - before_vector["macro_f1"],
                "weak_floor_delta": (
                    after_vector["weak_class_precision_recall_f1_floor"]
                    - before_vector["weak_class_precision_recall_f1_floor"]
                ),
                "expected_direction_acceptance_recall_delta_sum": expected_direction_delta,
                "baseline_parent_confusion_matrix": before_metrics[
                    "confusion_matrix_rows_true_columns_pred"
                ],
                "treatment_parent_confusion_matrix": after_metrics[
                    "confusion_matrix_rows_true_columns_pred"
                ],
            }
        )
    macro_deltas = np.asarray([row["macro_f1_delta"] for row in rows], dtype=np.float64)
    best_row = max(rows, key=lambda row: float(row["macro_f1_delta"]))
    return {
        "role": "paired training-trajectory attribution; epochs are correlated and not independent replicates",
        "rows": rows,
        "treatment_macro_f1_higher_epoch_count": int((macro_deltas > 0.0).sum()),
        "macro_f1_delta_strictly_greater_than_0_02_epoch_count": int(
            (macro_deltas > PRIMARY_DELTA).sum()
        ),
        "maximum_macro_f1_delta": float(best_row["macro_f1_delta"]),
        "maximum_macro_f1_delta_epoch": int(best_row["epoch"]),
    }


def build_report(root: Path) -> dict[str, Any]:
    required = {
        "readiness": root / READINESS,
        "experiment_contract": root / EXPERIMENT_CONTRACT,
        "feature_manifest": root / FEATURE_MANIFEST,
        "treatment_config": root / CONFIG,
        "baseline_config": root / BASELINE_CONFIG,
        "selection": root / BASELINE_SELECTION,
    }
    if not all(path.is_file() for path in required.values()):
        raise FileNotFoundError("V21a evaluation contract/config/readiness is incomplete")
    readiness = json.loads(required["readiness"].read_text(encoding="utf-8"))
    selected = dict(readiness.get("selected_baseline") or {})
    if (
        readiness.get("status") != "passed"
        or int(selected.get("seed", -1)) != 43
        or dict(readiness.get("config") or {}).get("sha256")
        != sha256_file(required["treatment_config"])
        or selected.get("config_sha256") != sha256_file(required["baseline_config"])
        or selected.get("selection_sha256") != sha256_file(required["selection"])
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V21a evaluation readiness is stale")
    readiness_hash = sha256_file(required["readiness"])
    baseline = _load_run(
        root,
        BASELINE_ARTIFACT,
        variant="baseline",
        seed=43,
        expected_config_path=required["baseline_config"],
        expected_readiness_sha256=None,
    )
    treatment = _load_run(
        root,
        ARTIFACT,
        variant=V21A_INTERFRAME_VARIANT,
        seed=43,
        expected_config_path=required["treatment_config"],
        expected_readiness_sha256=readiness_hash,
    )
    if (
        selected.get("summary_sha256") != baseline["summary_sha256"]
        or selected.get("checkpoint_sha256") != baseline["checkpoint_sha256"]
    ):
        raise ValueError("V21a Baseline summary/checkpoint differs from selected binding")
    pair = dict(treatment["summary"].get("paired_initialization") or {})
    if (
        pair.get("paired_baseline") is not True
        or pair.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pair.get("new_fusion_columns_exactly_zero") is not True
        or pair.get("initial_logits_exactly_equal") is not True
        or int(pair.get("seed", -1)) != 43
    ):
        raise ValueError("V21a Treatment does not prove same-seed paired initialization")
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
        raise ValueError("V21a Baseline/Treatment parent order hashes differ")

    before_vector = _metric_vector(baseline["metrics"])
    after_vector = _metric_vector(treatment["metrics"])
    delta = {key: after_vector[key] - before_vector[key] for key in before_vector}
    outcomes = _base_label_outcomes(baseline["records"], treatment["records"])
    expected_net = sum(
        int(outcomes[label]["net_rescue"]) for label in EXPECTED_DIRECTION_LABELS
    )
    bootstrap = _bootstrap(baseline["records"], treatment["records"])
    same_epoch = _same_epoch_comparison(
        baseline["summary"], treatment["summary"]
    )
    macro_ci = bootstrap["treatment_minus_baseline"]["macro_f1"]
    primary = float(delta["macro_f1"]) > PRIMARY_DELTA
    expected_labels_not_harmed = all(
        int(outcomes[label]["net_rescue"]) >= 0
        for label in EXPECTED_DIRECTION_LABELS
    )
    expected_direction = expected_net > 0 and expected_labels_not_harmed
    drone_not_harmed = int(outcomes["DroneTarget"]["net_rescue"]) >= 0
    known_target_recall_not_lower = float(delta["known_target_recall"]) >= 0.0
    direction_safety = drone_not_harmed and known_target_recall_not_lower
    robust = float(macro_ci["lower_95"]) > 0.0
    report = {
        "status": "completed",
        "schema_version": (
            "RADAR-LN3-V21a-known-gate-interframe-spectral-paired-validation-v1"
        ),
        "scope": "development validation parent-final KnownGate local task only",
        "feature_semantics": {
            "feature_names": list(INTERFRAME_SPECTRAL_FEATURE_NAMES),
            "unit": "current event adjacent canonical RD frames",
            "old_intra_frame_stft_candidate_used": False,
            "experiment_contract_path": str(required["experiment_contract"]),
            "experiment_contract_sha256": sha256_file(required["experiment_contract"]),
        },
        "paired_protocol": {
            "seed": 43,
            "selected_baseline": True,
            "initial_shared_state_exact": True,
            "new_fusion_columns_initially_zero": True,
            "initial_logits_exactly_equal": True,
            "all_20_epoch_parent_order_hashes_equal": True,
            "full_parameter_training": True,
            "baseline_checkpoint_is_not_a_finetune_source": True,
        },
        "baseline": {
            "metrics": baseline["metrics"],
            "summary_path": baseline["summary_path"],
            "summary_sha256": baseline["summary_sha256"],
            "checkpoint_path": baseline["checkpoint_path"],
            "checkpoint_sha256": baseline["checkpoint_sha256"],
        },
        "treatment": {
            "metrics": treatment["metrics"],
            "summary_path": treatment["summary_path"],
            "summary_sha256": treatment["summary_sha256"],
            "checkpoint_path": treatment["checkpoint_path"],
            "checkpoint_sha256": treatment["checkpoint_sha256"],
        },
        "treatment_minus_baseline": delta,
        "base_label_paired_outcomes": outcomes,
        "expected_direction_labels": list(EXPECTED_DIRECTION_LABELS),
        "expected_direction_net_rescue": int(expected_net),
        "bootstrap": bootstrap,
        "same_epoch_comparison": same_epoch,
        "frozen_decision": {
            "primary_macro_f1_delta_strictly_greater_than_0_02": primary,
            "expected_direction_labels_net_rescue_positive": expected_direction,
            "every_expected_direction_label_net_rescue_non_negative": (
                expected_labels_not_harmed
            ),
            "drone_target_net_rescue_non_negative": drone_not_harmed,
            "known_target_recall_not_lower": known_target_recall_not_lower,
            "unexpected_direction_safety_passed": direction_safety,
            "paired_bootstrap_macro_f1_lower_95_greater_than_0": robust,
            "positive_signal": primary and expected_direction and direction_safety,
            "robust_positive_signal": (
                primary and expected_direction and direction_safety and robust
            ),
            "advance_only_to_multiseed_or_train_parent_grouped_oof": (
                primary and expected_direction and direction_safety
            ),
            "deployment_claim_allowed": False,
        },
        "clutter_support_warning": {
            "validation_parent_support": 9,
            "parent_stratified_bootstrap_executed": True,
            "single_point_metric_is_not_sufficient_for_release": True,
        },
        "historical_feature_conception_exposure": readiness.get(
            "historical_feature_conception_exposure"
        ),
        "holdout_policy": {
            "current_prepare_test_payload_access_count": 0,
            "v16_test_is_unbiased_for_v21a": False,
            "reason": (
                "the original feature-conception probe sampled 14 V16 test tracks before this "
                "formal protocol; V16 test remains unexecuted but cannot support an unbiased "
                "final V21a generalization claim"
            ),
            "required_future_confirmation": (
                "train parent-grouped OOF, multi-seed, and a new independent holdout/data version"
            ),
        },
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
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**report, "report_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate corrected V21a against the selected seed43 KnownGate Baseline."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
