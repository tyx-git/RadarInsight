"""Evaluate V22a against the frozen V21e seed43 local KnownGate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_features import V22A_RD_TEMPORAL_RESIDUAL_VARIANT


V21E_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal"
)
V22A_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-"
    "seed43-formal"
)
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22a-known-gate-rd-temporal-residual-paired-validation.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 22042


def _load_run(root: Path, artifact: Path, expected_variant: str) -> dict[str, Any]:
    summary_path = root / artifact / "training-summary.json"
    detail_path = root / artifact / "sample_detail_information.json"
    if not summary_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError(f"V22a paired validation input is missing: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    record_sets = dict(detail.get("records") or {})
    records = {
        scope: list(record_sets.get(scope) or [])
        for scope in ("window", "eligible_prefix", "parent")
    }
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != expected_variant
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or len(records["parent"]) != 232
        or not records["window"]
        or not records["eligible_prefix"]
    ):
        raise ValueError(f"{expected_variant} summary/detail violates the local contract")
    if expected_variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
        residual_audit = dict(
            summary.get("rd_temporal_residual_training_audit") or {}
        )
        if (
            residual_audit.get("feature_used_by_best_checkpoint") is not True
            or int(residual_audit.get("best_checkpoint_nonzero_fusion_weight_count", 0))
            <= 0
        ):
            raise ValueError("V22a training summary does not prove residual use")
    return {"summary": summary, "detail": detail, "records": records}


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    result = {
        field: float(metrics[field])
        for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    for class_name in KNOWN_GATE_CLASS_NAMES:
        row = rows[class_name]
        slug = "known_target" if class_name == KNOWN_GATE_CLASS_NAMES[0] else "known_clutter"
        for field in ("precision", "recall", "f1"):
            result[f"{slug}_{field}"] = float(row[field])
    return result


def _weak_floor(metrics: Mapping[str, Any]) -> float:
    return min(float(row[field]) for row in metrics["per_class"] for field in ("precision", "recall", "f1"))


def _record_key(row: Mapping[str, Any], *, scope: str) -> str:
    parent_id = str(row["parent_sample_id"])
    if scope == "parent":
        return parent_id
    return f"{parent_id}#event{int(row['event_index']):04d}"


def _ordered_pair(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {_record_key(row, scope=scope): dict(row) for row in control}
    after = {_record_key(row, scope=scope): dict(row) for row in treatment}
    if len(before) != len(control) or len(after) != len(treatment) or set(before) != set(after):
        raise ValueError(f"V22a paired {scope} sets differ or repeat keys")
    ordered = sorted(before)
    before_rows = [before[key] for key in ordered]
    after_rows = [after[key] for key in ordered]
    for before_row, after_row in zip(before_rows, after_rows):
        if (
            before_row["label_name"] != after_row["label_name"]
            or int(before_row["true_id"]) != int(after_row["true_id"])
        ):
            raise ValueError(f"V22a paired {scope} labels differ")
        for field in (
            "track_id",
            "source_name",
            "window_length",
            "parent_length",
            "rd_frame_count",
            "rd_quality_missing",
            "rd_quality_frame_count_scaled",
        ):
            if field in before_row and field in after_row:
                before_value = before_row[field]
                after_value = after_row[field]
                if isinstance(before_value, float) or isinstance(after_value, float):
                    if not np.isclose(
                        float(before_value), float(after_value), atol=1.0e-7, rtol=0.0
                    ):
                        raise ValueError(
                            f"V22a paired {scope} immutable field differs: {field}"
                        )
                elif before_value != after_value:
                    raise ValueError(
                        f"V22a paired {scope} immutable field differs: {field}"
                    )
    return before_rows, after_rows


def _base_label_recall(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        selected = [row for row in records if str(row["label_name"]) == label]
        correct = sum(
            int(row["pred_id"]) == int(row["true_id"]) for row in selected
        )
        result[label] = {
            "support": len(selected),
            "correct_count": int(correct),
            "recall": float(correct / len(selected)) if selected else None,
        }
    return result


def _metric_delta(
    control_metrics: Mapping[str, Any], treatment_metrics: Mapping[str, Any]
) -> dict[str, float]:
    control_vector = _metric_vector(control_metrics)
    treatment_vector = _metric_vector(treatment_metrics)
    return {
        **{
            key: float(treatment_vector[key] - control_vector[key])
            for key in control_vector
        },
        "weak_floor": float(
            _weak_floor(treatment_metrics) - _weak_floor(control_metrics)
        ),
    }


def _paired_scope_metrics(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    before_rows, after_rows = _ordered_pair(control, treatment, scope=scope)
    control_metrics = summarize_known_gate_records(
        before_rows, scope=f"v22a_{scope}_control"
    )
    treatment_metrics = summarize_known_gate_records(
        after_rows, scope=f"v22a_{scope}_treatment"
    )
    control_base = _base_label_recall(before_rows)
    treatment_base = _base_label_recall(after_rows)
    return {
        "support": len(before_rows),
        "control": control_metrics,
        "treatment": treatment_metrics,
        "treatment_minus_control": _metric_delta(
            control_metrics, treatment_metrics
        ),
        "base_label_recall": {
            "control": control_base,
            "treatment": treatment_base,
            "treatment_minus_control": {
                label: float(
                    treatment_base[label]["recall"] - control_base[label]["recall"]
                )
                for label in KNOWN_LABELS
            },
        },
    }


def _bootstrap(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before_rows, after_rows = _ordered_pair(control, treatment, scope="parent")
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(before_rows) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in KNOWN_LABELS
    }
    support = {label: int(len(values)) for label, values in strata.items()}
    if support != {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9}:
        raise ValueError(f"V22a validation support changed: {support}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values_before: dict[str, list[float]] = {}
    values_after: dict[str, list[float]] = {}
    values_delta: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in strata.values()]
        )
        metric_before = _metric_vector(summarize_known_gate_records([before_rows[int(i)] for i in indices], scope="v22a_bootstrap"))
        metric_after = _metric_vector(summarize_known_gate_records([after_rows[int(i)] for i in indices], scope="v22a_bootstrap"))
        for key, value in metric_before.items():
            values_before.setdefault(key, []).append(value)
        for key, value in metric_after.items():
            values_after.setdefault(key, []).append(value)
            values_delta.setdefault(key, []).append(value - metric_before[key])

    def ci(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(array)),
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
        }

    return {
        "method": "parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": support,
        "control": {key: ci(value) for key, value in values_before.items()},
        "treatment": {key: ci(value) for key, value in values_after.items()},
        "treatment_minus_control": {key: ci(value) for key, value in values_delta.items()},
    }


def _paired_label_outcomes(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str = "parent",
) -> dict[str, Any]:
    before_rows, after_rows = _ordered_pair(control, treatment, scope=scope)
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        selected = [
            (before, after)
            for before, after in zip(before_rows, after_rows)
            if before["label_name"] == label
        ]
        rescued = harmed = changed = 0
        for before, after in selected:
            before_correct = int(before["pred_id"]) == int(before["true_id"])
            after_correct = int(after["pred_id"]) == int(after["true_id"])
            changed += int(int(before["pred_id"]) != int(after["pred_id"]))
            rescued += int((not before_correct) and after_correct)
            harmed += int(before_correct and (not after_correct))
        result[label] = {
            "support": len(selected),
            "changed": changed,
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
        }
    return result


def _paired_strata(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    groups: Mapping[str, Sequence[int]],
    scope: str,
) -> dict[str, Any]:
    before_rows, after_rows = _ordered_pair(control, treatment, scope="parent")
    result: dict[str, Any] = {}
    covered: set[int] = set()
    for name, raw_indices in groups.items():
        indices = [int(index) for index in raw_indices]
        covered.update(indices)
        selected_before = [before_rows[index] for index in indices]
        selected_after = [after_rows[index] for index in indices]
        block: dict[str, Any] = {
            "support": len(indices),
            "base_label_support": {
                label: sum(row["label_name"] == label for row in selected_after)
                for label in KNOWN_LABELS
            },
        }
        if indices:
            block["control"] = summarize_known_gate_records(
                selected_before, scope=f"{scope}_{name}_control"
            )
            block["treatment"] = summarize_known_gate_records(
                selected_after, scope=f"{scope}_{name}_treatment"
            )
            block["treatment_minus_control"] = _metric_delta(
                block["control"], block["treatment"]
            )
        else:
            block["metrics"] = "not_estimated_no_support"
        result[name] = block
    return {
        "scope": scope,
        "strata": result,
        "covered_parent_count": len(covered),
        "total_parent_count": len(after_rows),
        "all_parents_covered_exactly_once": (
            len(covered) == len(after_rows)
            and sum(len(indices) for indices in groups.values()) == len(after_rows)
        ),
    }


def _stratification(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _, after_rows = _ordered_pair(control, treatment, scope="parent")
    frame_groups = {
        "K=0_no_rd": [],
        "K=1": [],
        "K=2-4": [],
        "K=5": [],
        "K>=6": [],
    }
    quality_groups = {
        "rd_missing": [],
        "rd_available_low_frame_support": [],
        "rd_available_medium_frame_support": [],
        "rd_available_high_frame_support": [],
    }
    length_groups = {"length_1-8": [], "length_9-16": [], "length_17+": []}
    source_groups: dict[str, list[int]] = {}
    for index, row in enumerate(after_rows):
        count = int(row["rd_frame_count"])
        if count == 0:
            frame_groups["K=0_no_rd"].append(index)
        elif count == 1:
            frame_groups["K=1"].append(index)
        elif 2 <= count <= 4:
            frame_groups["K=2-4"].append(index)
        elif count == 5:
            frame_groups["K=5"].append(index)
        else:
            frame_groups["K>=6"].append(index)

        if bool(row["rd_quality_missing"]):
            quality_groups["rd_missing"].append(index)
        else:
            quality = float(row["rd_quality_frame_count_scaled"])
            if quality <= 0.25:
                quality_groups["rd_available_low_frame_support"].append(index)
            elif quality <= 0.5:
                quality_groups["rd_available_medium_frame_support"].append(index)
            else:
                quality_groups["rd_available_high_frame_support"].append(index)

        parent_length = int(row["parent_length"])
        if parent_length <= 8:
            length_groups["length_1-8"].append(index)
        elif parent_length <= 16:
            length_groups["length_9-16"].append(index)
        else:
            length_groups["length_17+"].append(index)
        source_groups.setdefault(str(row["source_name"]), []).append(index)

    return {
        "rd_frame_count": _paired_strata(
            control,
            treatment,
            groups=frame_groups,
            scope="v22a_rd_frame_count",
        ),
        "rd_quality": _paired_strata(
            control,
            treatment,
            groups=quality_groups,
            scope="v22a_rd_quality",
        ),
        "source": _paired_strata(
            control,
            treatment,
            groups=source_groups,
            scope="v22a_source",
        ),
        "parent_length": _paired_strata(
            control,
            treatment,
            groups=length_groups,
            scope="v22a_parent_length",
        ),
    }


def _residual_activity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    multiframe = [row for row in rows if int(row["rd_frame_count"]) >= 2]
    single_or_missing = [row for row in rows if int(row["rd_frame_count"]) < 2]
    nonzero_multiframe = sum(
        bool(row["rd_temporal_residual_nonzero"]) for row in multiframe
    )
    invalid_nonzero = sum(
        bool(row["rd_temporal_residual_nonzero"]) for row in single_or_missing
    )
    return {
        "event_count": len(rows),
        "multiframe_event_count": len(multiframe),
        "multiframe_nonzero_residual_event_count": int(nonzero_multiframe),
        "multiframe_nonzero_residual_fraction": (
            float(nonzero_multiframe / len(multiframe)) if multiframe else None
        ),
        "single_or_missing_event_count": len(single_or_missing),
        "single_or_missing_nonzero_residual_event_count": int(invalid_nonzero),
        "single_or_missing_residual_is_exact_zero": invalid_nonzero == 0,
        "residual_l2_max": max(
            (float(row["rd_temporal_residual_l2"]) for row in rows),
            default=0.0,
        ),
    }


def run(root: Path) -> dict[str, Any]:
    control = _load_run(root, V21E_ARTIFACT, "v21e_course_prefix")
    treatment = _load_run(root, V22A_ARTIFACT, V22A_RD_TEMPORAL_RESIDUAL_VARIANT)
    parent_scope = _paired_scope_metrics(
        control["records"]["parent"],
        treatment["records"]["parent"],
        scope="parent",
    )
    eligible_prefix_scope = _paired_scope_metrics(
        control["records"]["eligible_prefix"],
        treatment["records"]["eligible_prefix"],
        scope="eligible_prefix",
    )
    window_scope = _paired_scope_metrics(
        control["records"]["window"],
        treatment["records"]["window"],
        scope="window",
    )
    control_metrics = parent_scope["control"]
    treatment_metrics = parent_scope["treatment"]
    delta = parent_scope["treatment_minus_control"]
    bootstrap = _bootstrap(
        control["records"]["parent"], treatment["records"]["parent"]
    )
    control_orders = list(
        dict(control["summary"].get("dataloader_contract") or {}).get(
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
    control_initial_logits = dict(
        control["summary"].get("paired_initialization") or {}
    ).get("initial_logits_sha256")
    treatment_pairing = dict(
        treatment["summary"].get("paired_initialization") or {}
    )
    if (
        len(control_orders) != 20
        or treatment_orders != control_orders
        or treatment_pairing.get("initial_logits_sha256")
        != control_initial_logits
        or treatment_pairing.get(
            "frozen_v21e_initial_model_state_reproduced"
        )
        is not True
        or treatment_pairing.get(
            "real_training_probe_residual_columns_receive_gradient"
        )
        is not True
    ):
        raise ValueError("V22a historical Control/Treatment pairing is not reproducible")

    parent_vector_before = _metric_vector(control_metrics)
    parent_vector_after = _metric_vector(treatment_metrics)
    eligible_vector_before = _metric_vector(eligible_prefix_scope["control"])
    eligible_vector_after = _metric_vector(eligible_prefix_scope["treatment"])
    window_vector_before = _metric_vector(window_scope["control"])
    window_vector_after = _metric_vector(window_scope["treatment"])
    parent_base_before = parent_scope["base_label_recall"]["control"]
    parent_base_after = parent_scope["base_label_recall"]["treatment"]
    target_labels = ("DroneTarget", "BirdTarget", "BalloonTarget")
    decision = {
        "primary_macro_f1_delta_strictly_greater_than_0_02": delta["macro_f1"]
        > 0.02,
        "known_target_precision_recall_f1_non_decrease": all(
            parent_vector_after[f"known_target_{field}"]
            >= parent_vector_before[f"known_target_{field}"]
            for field in ("precision", "recall", "f1")
        ),
        "clutter_precision_recall_f1_non_decrease": all(
            parent_vector_after[f"known_clutter_{field}"]
            >= parent_vector_before[f"known_clutter_{field}"]
            for field in ("precision", "recall", "f1")
        ),
        "drone_bird_balloon_parent_recall_non_decrease": all(
            float(parent_base_after[label]["recall"])
            >= float(parent_base_before[label]["recall"])
            for label in target_labels
        ),
        "eligible_prefix_gate_metrics_non_decrease": all(
            eligible_vector_after[key] >= eligible_vector_before[key]
            for key in eligible_vector_before
        ),
        "window_gate_metrics_non_decrease": all(
            window_vector_after[key] >= window_vector_before[key]
            for key in window_vector_before
        ),
        "residual_feature_used_by_best_checkpoint": dict(
            treatment["summary"].get("rd_temporal_residual_training_audit") or {}
        ).get("feature_used_by_best_checkpoint")
        is True,
    }
    decision["all_frozen_local_adoption_gates_passed"] = all(decision.values())
    decision["deployment_claim_allowed"] = False
    report = {
        "status": "completed",
        "schema_version": "RADAR-LN3-V22a-known-gate-rd-temporal-residual-paired-validation-v1",
        "scope": "development validation parent-final KnownGate local task only",
        "control": {
            "source": "frozen V21e seed43 Treatment best validation result",
            "metrics": control_metrics,
            "summary_path": str(root / V21E_ARTIFACT / "training-summary.json"),
        },
        "treatment": {
            "source": "V22a full 20-epoch Treatment",
            "metrics": treatment_metrics,
            "summary_path": str(root / V22A_ARTIFACT / "training-summary.json"),
        },
        "treatment_minus_control": delta,
        "paired_scopes": {
            "parent_final": parent_scope,
            "eligible_prefix_excluding_final": eligible_prefix_scope,
            "all_causal_windows": window_scope,
        },
        "base_label_paired_outcomes": {
            "parent_final": _paired_label_outcomes(
                control["records"]["parent"],
                treatment["records"]["parent"],
                scope="parent",
            ),
            "eligible_prefix_excluding_final": _paired_label_outcomes(
                control["records"]["eligible_prefix"],
                treatment["records"]["eligible_prefix"],
                scope="eligible_prefix",
            ),
        },
        "paired_parent_bootstrap": bootstrap,
        "paired_training_contract": {
            "epoch_parent_order_hashes_exactly_equal": True,
            "epoch_count": len(control_orders),
            "initial_logits_sha256_exactly_equal": True,
            "initial_logits_sha256": control_initial_logits,
            "frozen_v21e_initial_model_state_reproduced": True,
            "real_training_probe_residual_columns_receive_gradient": True,
        },
        "rd_temporal_residual_training_audit": treatment["summary"][
            "rd_temporal_residual_training_audit"
        ],
        "rd_temporal_residual_validation_activity": _residual_activity(
            treatment["records"]["window"]
        ),
        "paired_parent_stratification": _stratification(
            control["records"]["parent"], treatment["records"]["parent"]
        ),
        "clutter_validation_support": 9,
        "decision": decision,
        "unknown_is_model_class": False,
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
    parser = argparse.ArgumentParser(description="Evaluate V22a paired local validation.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
