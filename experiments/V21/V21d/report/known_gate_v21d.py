from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v21d import (
    ARTIFACT,
    BASELINE_ARTIFACT,
    BASELINE_CONFIG,
    BASELINE_SELECTION,
    CONFIG,
    EXPERIMENT_CONTRACT,
    READINESS,
)
from radar_ln3.known_gate import sha256_file, summarize_known_gate_records
from radar_ln3.known_gate_features import (
    D_RANGE_PREFIX_FEATURE_NAMES,
    V21D_PREFIX_VARIANT,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-paired-validation.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 21443
PRIMARY_DELTA = 0.02
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
PRIMARY_HYPOTHESIS_LABELS = ("BirdTarget", "ClutterTarget")
SAFETY_LABELS = ("DroneTarget", "BalloonTarget")


def _load_run(
    root: Path,
    artifact_relative: Path,
    *,
    variant: str,
    expected_config_path: Path,
    expected_readiness_sha256: str | None,
) -> dict[str, Any]:
    artifact = root / artifact_relative
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError(f"V21d paired run is incomplete: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    records_by_scope = {
        scope: list(dict(detail.get("records") or {}).get(scope) or [])
        for scope in ("parent", "window", "eligible_prefix")
    }
    training_readiness = dict(summary.get("training_readiness") or {})
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != variant
        or int(summary.get("seed", -1)) != 43
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
        or len(records_by_scope["parent"]) != 232
        or len(records_by_scope["window"]) != 4452
        or len(records_by_scope["eligible_prefix"]) != 3756
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
        raise ValueError(f"V21d {variant} run violates the full local contract")
    metrics = summarize_known_gate_records(
        records_by_scope["parent"], scope=f"{variant}_parent_final_recomputed"
    )
    recorded = dict(dict(summary["best_validation_metrics"])["parent_metrics"])
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(metrics[field]) - float(recorded[field])) > 1.0e-12:
            raise ValueError(f"V21d {variant} recomputed metric differs at {field}")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "summary": summary,
        "records_by_scope": records_by_scope,
        "records": records_by_scope["parent"],
        "metrics": metrics,
    }


def _record_key(scope: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    parent_id = str(row["parent_sample_id"])
    return (parent_id,) if scope == "parent" else (parent_id, int(row["event_index"]))


def _ordered_pair(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {_record_key(scope, row): dict(row) for row in baseline}
    after = {_record_key(scope, row): dict(row) for row in treatment}
    if len(before) != len(baseline) or len(after) != len(treatment) or set(before) != set(after):
        raise ValueError(f"V21d paired {scope} sets differ or repeat keys")
    ordered = sorted(before)
    for key in ordered:
        if (
            before[key]["label_name"] != after[key]["label_name"]
            or int(before[key]["true_id"]) != int(after[key]["true_id"])
        ):
            raise ValueError(f"V21d paired {scope} labels differ")
    return [before[key] for key in ordered], [after[key] for key in ordered]


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


def _base_label_recall_vector(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for label in LABELS:
        selected = [row for row in records if str(row["label_name"]) == label]
        if not selected:
            raise ValueError(f"V21d bootstrap has no {label} parent")
        result[f"{label}_acceptance_recall"] = float(
            np.mean([int(row["pred_id"]) == int(row["true_id"]) for row in selected])
        )
    result["clutter_bird_balanced_acceptance_recall"] = float(
        np.mean(
            [
                result["BirdTarget_acceptance_recall"],
                result["ClutterTarget_acceptance_recall"],
            ]
        )
    )
    return result


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope="parent")
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
        raise ValueError("V21d bootstrap support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    before_values: dict[str, list[float]] = {}
    after_values: dict[str, list[float]] = {}
    delta_values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        before_sample = [before[int(index)] for index in sampled]
        after_sample = [after[int(index)] for index in sampled]
        before_vector = _metric_vector(
            summarize_known_gate_records(before_sample, scope="v21d_parent_bootstrap")
        )
        after_vector = _metric_vector(
            summarize_known_gate_records(after_sample, scope="v21d_parent_bootstrap")
        )
        before_vector.update(_base_label_recall_vector(before_sample))
        after_vector.update(_base_label_recall_vector(after_sample))
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
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope="parent")
    result: dict[str, Any] = {}
    for label in LABELS:
        indices = [index for index, row in enumerate(before) if row["label_name"] == label]
        changed = rescued = harmed = 0
        for index in indices:
            before_correct = int(before[index]["pred_id"]) == int(before[index]["true_id"])
            after_correct = int(after[index]["pred_id"]) == int(after[index]["true_id"])
            changed += int(int(before[index]["pred_id"]) != int(after[index]["pred_id"]))
            rescued += int((not before_correct) and after_correct)
            harmed += int(before_correct and (not after_correct))
        result[label] = {
            "support": len(indices),
            "changed": changed,
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
        }
    return result


def _scope_changes(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    treatment: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("parent", "window", "eligible_prefix"):
        before, after = _ordered_pair(baseline[scope], treatment[scope], scope=scope)
        changed = sum(
            int(int(left["pred_id"]) != int(right["pred_id"]))
            for left, right in zip(before, after, strict=True)
        )
        result[scope] = {
            "support": len(before),
            "changed_prediction_count": changed,
            "changed_prediction_fraction": float(changed / max(1, len(before))),
        }
    return result


def _probability_attribution(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope="parent")
    rows: dict[str, Any] = {}
    for label in LABELS:
        selected = [index for index, row in enumerate(before) if row["label_name"] == label]
        delta = np.asarray(
            [
                float(after[index]["known_clutter_probability"])
                - float(before[index]["known_clutter_probability"])
                for index in selected
            ],
            dtype=np.float64,
        )
        rows[label] = {
            "support": len(selected),
            "known_clutter_probability_delta_mean": float(delta.mean()),
            "known_clutter_probability_delta_median": float(np.median(delta)),
            "known_clutter_probability_delta_minimum": float(delta.min()),
            "known_clutter_probability_delta_maximum": float(delta.max()),
        }

    def scores(records: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
        true_probability = np.asarray(
            [
                float(row["known_target_probability"])
                if int(row["true_id"]) == 0
                else float(row["known_clutter_probability"])
                for row in records
            ]
        )
        clutter_probability = np.asarray(
            [float(row["known_clutter_probability"]) for row in records]
        )
        true_clutter = np.asarray([int(row["true_id"]) == 1 for row in records])
        return (
            float(-np.log(np.clip(true_probability, 1.0e-12, 1.0)).mean()),
            float(np.square(clutter_probability - true_clutter).mean()),
        )

    before_nll, before_brier = scores(before)
    after_nll, after_brier = scores(after)
    primary_indices = [
        index for index, row in enumerate(before) if row["label_name"] in PRIMARY_HYPOTHESIS_LABELS
    ]
    primary_before = [before[index] for index in primary_indices]
    primary_after = [after[index] for index in primary_indices]
    primary_before_nll, primary_before_brier = scores(primary_before)
    primary_after_nll, primary_after_brier = scores(primary_after)
    return {
        "by_base_label": rows,
        "overall": {
            "baseline_nll": before_nll,
            "treatment_nll": after_nll,
            "nll_delta": after_nll - before_nll,
            "baseline_brier": before_brier,
            "treatment_brier": after_brier,
            "brier_delta": after_brier - before_brier,
        },
        "clutter_bird_primary_subset": {
            "support": len(primary_before),
            "baseline_nll": primary_before_nll,
            "treatment_nll": primary_after_nll,
            "nll_delta": primary_after_nll - primary_before_nll,
            "baseline_brier": primary_before_brier,
            "treatment_brier": primary_after_brier,
            "brier_delta": primary_after_brier - primary_before_brier,
        },
        "post_hoc_probability_diagnostic_only": True,
    }


def _same_epoch_comparison(
    baseline_summary: Mapping[str, Any],
    treatment_summary: Mapping[str, Any],
) -> dict[str, Any]:
    before = {int(row["epoch"]): dict(row) for row in baseline_summary["history"]}
    after = {int(row["epoch"]): dict(row) for row in treatment_summary["history"]}
    if set(before) != set(range(1, 21)) or set(after) != set(before):
        raise ValueError("V21d same-epoch comparison requires epochs 1..20")
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
        primary_delta = sum(
            float(after_labels[label]["acceptance_recall"])
            - float(before_labels[label]["acceptance_recall"])
            for label in PRIMARY_HYPOTHESIS_LABELS
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
                "clutter_bird_acceptance_recall_delta_sum": primary_delta,
                "baseline_parent_confusion_matrix": before_metrics[
                    "confusion_matrix_rows_true_columns_pred"
                ],
                "treatment_parent_confusion_matrix": after_metrics[
                    "confusion_matrix_rows_true_columns_pred"
                ],
            }
        )
    deltas = np.asarray([row["macro_f1_delta"] for row in rows])
    best = max(rows, key=lambda row: float(row["macro_f1_delta"]))
    return {
        "role": "paired training trajectory; epochs are correlated, not independent replicates",
        "rows": rows,
        "treatment_macro_f1_higher_epoch_count": int((deltas > 0.0).sum()),
        "macro_f1_delta_strictly_greater_than_0_02_epoch_count": int(
            (deltas > PRIMARY_DELTA).sum()
        ),
        "maximum_macro_f1_delta": float(best["macro_f1_delta"]),
        "maximum_macro_f1_delta_epoch": int(best["epoch"]),
    }


def _trained_feature_column(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weight = torch.as_tensor(payload["model_state"]["fusion.0.weight"])
    column = weight[:, -1].detach().cpu().to(torch.float64)
    if column.numel() != 48 or not torch.isfinite(column).all() or float(column.abs().max()) <= 0.0:
        raise ValueError("V21d trained feature column is missing, zero or non-finite")
    return {
        "column_index": int(weight.shape[1] - 1),
        "element_count": int(column.numel()),
        "nonzero_element_count": int(torch.count_nonzero(column).item()),
        "l1_norm": float(column.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(column)),
        "maximum_absolute_weight": float(column.abs().max()),
    }


def build_report(root: Path) -> dict[str, Any]:
    required = {
        "readiness": root / READINESS,
        "experiment_contract": root / EXPERIMENT_CONTRACT,
        "treatment_config": root / CONFIG,
        "baseline_config": root / BASELINE_CONFIG,
        "selection": root / BASELINE_SELECTION,
    }
    if not all(path.is_file() for path in required.values()):
        raise FileNotFoundError("V21d evaluation contract/config/readiness is incomplete")
    readiness = json.loads(required["readiness"].read_text(encoding="utf-8"))
    selected = dict(readiness.get("selected_baseline") or {})
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-training-readiness-v1"
        or int(selected.get("seed", -1)) != 43
        or dict(readiness.get("config") or {}).get("sha256")
        != sha256_file(required["treatment_config"])
        or selected.get("config_sha256") != sha256_file(required["baseline_config"])
        or selected.get("selection_sha256") != sha256_file(required["selection"])
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("validation_feature_values_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V21d evaluation readiness is stale")
    readiness_hash = sha256_file(required["readiness"])
    baseline = _load_run(
        root,
        BASELINE_ARTIFACT,
        variant="baseline",
        expected_config_path=required["baseline_config"],
        expected_readiness_sha256=None,
    )
    treatment = _load_run(
        root,
        ARTIFACT,
        variant=V21D_PREFIX_VARIANT,
        expected_config_path=required["treatment_config"],
        expected_readiness_sha256=readiness_hash,
    )
    if (
        selected.get("summary_sha256") != baseline["summary_sha256"]
        or selected.get("checkpoint_sha256") != baseline["checkpoint_sha256"]
    ):
        raise ValueError("V21d Baseline differs from selected binding")
    pair = dict(treatment["summary"].get("paired_initialization") or {})
    if (
        pair.get("paired_baseline") is not True
        or pair.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pair.get("new_fusion_columns_exactly_zero") is not True
        or pair.get("initial_logits_exactly_equal") is not True
        or int(pair.get("new_fusion_columns", -1)) != 1
        or int(pair.get("seed", -1)) != 43
    ):
        raise ValueError("V21d Treatment does not prove same-seed initialization")
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
        raise ValueError("V21d Baseline/Treatment parent orders differ")

    before_vector = _metric_vector(baseline["metrics"])
    after_vector = _metric_vector(treatment["metrics"])
    delta = {key: after_vector[key] - before_vector[key] for key in before_vector}
    outcomes = _base_label_outcomes(baseline["records"], treatment["records"])
    primary_pair_net = sum(outcomes[label]["net_rescue"] for label in PRIMARY_HYPOTHESIS_LABELS)
    bootstrap = _bootstrap(baseline["records"], treatment["records"])
    same_epoch = _same_epoch_comparison(baseline["summary"], treatment["summary"])
    scope_changes = _scope_changes(
        baseline["records_by_scope"], treatment["records_by_scope"]
    )
    probability = _probability_attribution(baseline["records"], treatment["records"])
    trained = _trained_feature_column(Path(treatment["checkpoint_path"]))
    macro_ci = bootstrap["treatment_minus_baseline"]["macro_f1"]
    primary_ci = bootstrap["treatment_minus_baseline"][
        "clutter_bird_balanced_acceptance_recall"
    ]
    primary = float(delta["macro_f1"]) > PRIMARY_DELTA
    primary_not_harmed = all(outcomes[label]["net_rescue"] >= 0 for label in PRIMARY_HYPOTHESIS_LABELS)
    primary_improved = primary_pair_net > 0 and primary_not_harmed
    safety_not_harmed = all(outcomes[label]["net_rescue"] >= 0 for label in SAFETY_LABELS)
    known_target_not_lower = float(delta["known_target_recall"]) >= 0.0
    clutter_not_lower = float(delta["known_clutter_recall"]) >= 0.0
    direction_safety = (
        primary_not_harmed
        and safety_not_harmed
        and known_target_not_lower
        and clutter_not_lower
    )
    robust = float(macro_ci["lower_95"]) > 0.0
    primary_robust = float(primary_ci["lower_95"]) > 0.0
    report = {
        "status": "completed",
        "schema_version": (
            "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-"
            "clutter-bird-paired-validation-v1"
        ),
        "scope": "development validation parent-final KnownGate local task only",
        "feature_semantics": {
            "feature_names": list(D_RANGE_PREFIX_FEATURE_NAMES),
            "unit": "causal-prefix frame-level MTD-minus-TR range sample standard deviation / 75 m",
            "core_current_event_proxy_superseded": True,
            "experiment_contract_path": str(required["experiment_contract"]),
            "experiment_contract_sha256": sha256_file(required["experiment_contract"]),
        },
        "train_only_prior": readiness.get("train_only_feature_diagnostic"),
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
        "scope_prediction_changes": scope_changes,
        "probability_attribution": probability,
        "trained_d_range_fusion_column": trained,
        "primary_hypothesis": {
            "research_question": (
                "does causal-prefix MTD-minus-TR range sample standard deviation improve "
                "the ClutterTarget versus BirdTarget KnownGate boundary?"
            ),
            "labels": list(PRIMARY_HYPOTHESIS_LABELS),
            "combined_net_rescue": int(primary_pair_net),
        },
        "safety_labels": list(SAFETY_LABELS),
        "bootstrap": bootstrap,
        "same_epoch_comparison": same_epoch,
        "frozen_decision": {
            "primary_macro_f1_delta_strictly_greater_than_0_02": primary,
            "clutter_bird_combined_net_rescue_positive": primary_improved,
            "each_clutter_bird_label_net_rescue_non_negative": primary_not_harmed,
            "each_drone_balloon_safety_label_net_rescue_non_negative": safety_not_harmed,
            "known_target_recall_not_lower": known_target_not_lower,
            "known_clutter_recall_not_lower": clutter_not_lower,
            "direction_safety_passed": direction_safety,
            "paired_bootstrap_macro_f1_lower_95_greater_than_0": robust,
            "paired_bootstrap_clutter_bird_balanced_recall_lower_95_greater_than_0": primary_robust,
            "positive_signal": primary and primary_improved and direction_safety,
            "robust_positive_signal": (
                primary and primary_improved and direction_safety and robust and primary_robust
            ),
            "advance_only_to_multiseed_or_train_parent_grouped_oof": (
                primary and primary_improved and direction_safety
            ),
            "deployment_claim_allowed": False,
        },
        "clutter_support_warning": {
            "validation_parent_support": 9,
            "parent_stratified_bootstrap_executed": True,
            "single_point_metric_is_not_sufficient_for_release": True,
        },
        "holdout_policy": {
            "current_prepare_test_payload_access_count": 0,
            "test_used_for_formula_or_model_selection": False,
            "original_v16_validation_is_pristine_for_feature_conception": False,
            "original_v16_test_is_pristine_for_feature_conception": False,
            "reason": (
                "historical representative-track probe used Bird 333 from validation and "
                "Clutter 1868 from test"
            ),
            "required_future_confirmation": (
                "multi-seed, train parent-grouped OOF and a new independent holdout"
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
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "report_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V21d against selected seed43 KnownGate Baseline."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
