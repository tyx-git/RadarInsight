"""Paired development report for one completed V23 external-adapter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    summarize_parent_balanced_eligible_prefix_records,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_v23_adapters import (
    V23_ADAPTER_VARIANTS,
    V23A_VARIANT,
    V23B_VARIANT,
    V23BR_VARIANT,
    V23D_VARIANT,
    V23GR_VARIANT,
    V23H_VARIANT,
    adapter_state_names,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_adapter_config,
    readiness_relative,
    report_relative,
    report_schema,
    source_v23_parent_order_hashes,
    source_v23_training,
    source_v23a_control,
    source_v23b_control,
    source_v23c_control,
)
from radar_ln3.known_gate_v23 import sha256_file
from report.known_gate_v21e import (
    _base_label_outcomes,
    _bootstrap,
    _metric_vector,
    _ordered_pair,
    _probability_attribution,
    _scope_changes,
)


SCOPES = ("parent", "eligible_prefix", "window")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _same_metrics(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], *, scope: str
) -> bool:
    """Verify evaluation records, not summary prose, are the report source."""

    for field in (
        "support",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "confusion_matrix_rows_true_columns_pred",
    ):
        if recorded.get(field) != recomputed.get(field):
            return False
    recorded_rows = list(recorded.get("per_class") or [])
    recomputed_rows = list(recomputed.get("per_class") or [])
    if len(recorded_rows) != len(recomputed_rows):
        return False
    for recorded_row, recomputed_row in zip(recorded_rows, recomputed_rows, strict=True):
        for field in (
            "class",
            "precision",
            "recall",
            "f1",
            "support",
            "major_confusion_to",
            "major_confusion_count",
        ):
            if dict(recorded_row).get(field) != dict(recomputed_row).get(field):
                return False
    return True


def _load_v23(root: Path) -> dict[str, Any]:
    artifact = root / "artifacts/RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-formal"
    summary_path, detail_path, checkpoint_path = artifact / "training-summary.json", artifact / "sample_detail_information.json", artifact / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError("frozen V23 artifact is incomplete")
    summary, detail = _read_json(summary_path), _read_json(detail_path)
    records = {scope: list(dict(detail.get("records") or {}).get(scope) or []) for scope in SCOPES}
    if (
        summary.get("status") != "completed" or summary.get("variant") != "official_cfc_baseline"
        or summary.get("architecture") != "official_ncps_cfc" or int(summary.get("seed", -1)) != 43
        or int(summary.get("validation_parent_count", -1)) != 232 or int(summary.get("validation_clutter_support", -1)) != 9
        or detail.get("variant") != "official_cfc_baseline" or len(records["parent"]) != 232
        or len(records["eligible_prefix"]) != 3756 or len(records["window"]) != 4452
    ):
        raise ValueError("frozen V23 artifact violates the expected baseline contract")
    return {
        "summary": summary, "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "records": records,
    }


def _load_treatment(root: Path, variant: str) -> dict[str, Any]:
    config_path = root / config_relative(variant)
    readiness_path = root / readiness_relative(variant)
    config, _source, _source_path = load_adapter_config(root, variant)
    artifact = root / artifact_relative(variant)
    summary_path, detail_path, checkpoint_path = artifact / "training-summary.json", artifact / "sample_detail_information.json", artifact / "best_model.pt"
    if not all(path.is_file() for path in (config_path, readiness_path, summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError(f"{variant} artifact is incomplete")
    readiness, summary, detail = _read_json(readiness_path), _read_json(summary_path), _read_json(detail_path)
    records = {scope: list(dict(detail.get("records") or {}).get(scope) or []) for scope in SCOPES}
    _v23_summary_path, v23_summary, _v23_checkpoint_path, _v23_checkpoint_sha = source_v23_training(root)
    v23_parent_order_hashes = source_v23_parent_order_hashes(v23_summary)
    pairing = dict(summary.get("v23_initialization_pairing") or {})
    history = list(summary.get("history") or [])
    recomputed_metrics = {
        "parent_metrics": summarize_known_gate_records(
            records["parent"], scope="parent_final"
        ),
        "eligible_prefix_metrics": summarize_known_gate_records(
            records["eligible_prefix"],
            scope="eligible_causal_prefix_auxiliary_excluding_final",
        ),
        "window_metrics": summarize_known_gate_records(
            records["window"], scope="causal_prefix_windows"
        ),
    }
    recorded_metrics = dict(summary.get("best_validation_metrics") or {})
    recorded_metrics_match = all(
        _same_metrics(
            dict(recorded_metrics.get(key) or {}), value, scope=key
        )
        for key, value in recomputed_metrics.items()
    )
    parent_order_ok = len(history) == 20 and all(
        int(dict(row).get("epoch", -1)) == epoch
        and dict(row).get("parent_order_sha256") == v23_parent_order_hashes[epoch - 1]
        and dict(row).get("v23_control_parent_order_sha256")
        == v23_parent_order_hashes[epoch - 1]
        and dict(row).get("parent_order_exactly_matches_v23_control") is True
        for epoch, row in enumerate(history, start=1)
    )
    expected_initial = str(v23_summary["initial_model_state_sha256"])
    expected_adapter_count = config.adapter.trainable_parameter_count
    recorded_initial_hash = str(summary.get("initial_model_state_sha256") or "")
    if expected_adapter_count == 0:
        initialization_ok = (
            pairing.get("full_initial_state_exactly_matches_v23") is True
            and pairing.get("core_initial_state_exactly_matches_v23") is True
            and pairing.get("treatment_full_initial_model_state_sha256") == expected_initial
            and pairing.get("treatment_core_initial_model_state_sha256") == expected_initial
            and recorded_initial_hash == expected_initial
            and pairing.get("adapter_state_names") == []
        )
    else:
        initialization_ok = (
            pairing.get("full_initial_state_exactly_matches_v23") is False
            and pairing.get("core_initial_state_exactly_matches_v23") is True
            and pairing.get("source_v23_initial_model_state_sha256") == expected_initial
            and pairing.get("treatment_full_initial_model_state_sha256") == recorded_initial_hash
            and pairing.get("treatment_core_initial_model_state_sha256") == expected_initial
            and pairing.get("adapter_state_names")
            == list(adapter_state_names(config.adapter))
        )
    direct_control_ok = (
        True
        if variant not in (V23D_VARIANT, V23BR_VARIANT, V23GR_VARIANT, V23H_VARIANT)
        else (
            (
                config.direct_control == source_v23a_control(root)
                and pairing.get("v23d_direct_v23a_replay", {}).get(
                    "before_optimizer_step_replays_v23a_within_float32_tolerance"
                ) is True
            )
            if variant == V23D_VARIANT
            else (
                (
                    config.direct_control == source_v23b_control(root)
                    and pairing.get("v23br_direct_v23b_scale_only_audit", {}).get(
                        "before_optimizer_step_replays_v23b_within_float32_tolerance"
                    ) is True
                )
                if variant == V23BR_VARIANT
                else (
                    (
                        config.direct_control == source_v23c_control(root)
                        and pairing.get("v23gr_direct_v23c_multiplicity_audit", {}).get(
                            "pre_zero_delta_replays_v23c_within_float32_tolerance"
                        ) is True
                        and pairing.get("v23gr_direct_v23c_multiplicity_audit", {}).get(
                            "first_k2_zero_delta_differs_from_v23c"
                        ) is True
                    )
                    if variant == V23GR_VARIANT
                    else (
                        config.direct_control == source_v23a_control(root)
                        and pairing.get("v23h_direct_v23a_cap_audit", {}).get(
                            "pre_above_cap_positive_dt_replays_v23a_within_float32_tolerance"
                        ) is True
                        and pairing.get("v23h_direct_v23a_cap_audit", {}).get(
                            "first_real_above_cap_positive_dt_diverges_from_v23a"
                        ) is True
                    )
                )
            )
        )
    )
    direct_control_summary_ok = (
        True
        if variant not in (V23D_VARIANT, V23BR_VARIANT, V23GR_VARIANT, V23H_VARIANT)
        else summary.get("direct_control") == config.direct_control
    )
    tau_audit_ok = (
        True
        if variant != V23D_VARIANT
        else (
            isinstance(summary.get("v23d_per_dimension_tau_audit"), dict)
            and int(dict(summary["v23d_per_dimension_tau_audit"]).get("dimension_count", -1)) == 48
            and dict(summary["v23d_per_dimension_tau_audit"]).get("all_tau_positive_and_finite") is True
        )
    )
    if (
        readiness.get("status") != "passed" or summary.get("status") != "completed"
        or summary.get("variant") != variant or summary.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or int(summary.get("seed", -1)) != 43 or len(history) != 20
        or int(summary.get("train_parent_count", -1)) != 986 or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9 or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False or detail.get("variant") != variant
        or detail.get("task_name") != "known_gate"
        or detail.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or detail.get("adapter") != config.adapter.as_dict() or int(detail.get("seed", -1)) != 43
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -1))
        or Path(str(detail.get("checkpoint", ""))).resolve() != checkpoint_path.resolve()
        or detail.get("scope") != "development validation KnownGate only; not OOF or test"
        or {key: int(value) for key, value in dict(detail.get("counts") or {}).items()}
        != {"parent": 232, "eligible_prefix": 3756, "window": 4452}
        or len(records["parent"]) != 232
        or len(records["eligible_prefix"]) != 3756 or len(records["window"]) != 4452
        or Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve()
        or Path(str(summary.get("sample_detail_information", ""))).resolve()
        != detail_path.resolve()
        or Path(str(summary.get("last_checkpoint", ""))).resolve()
        != (artifact / "last_model.pt").resolve()
        or summary.get("external_state_adapter") != config.adapter.as_dict()
        or not recorded_metrics_match
        or int(pairing.get("adapter_trainable_parameter_count", -1)) != expected_adapter_count
        or not initialization_ok or not direct_control_ok or not direct_control_summary_ok or not tau_audit_ok or not parent_order_ok
        or dict(summary.get("dataloader_contract") or {}).get(
            "all_epoch_parent_orders_exactly_match_v23_control"
        ) is not True
        or dict(summary.get("training_readiness") or {}).get("sha256")
        != sha256_file(readiness_path)
    ):
        raise ValueError(f"{variant} completed artifact violates its contract")
    if variant == V23H_VARIANT and (
        summary.get("time_policy") != config.time_policy.as_dict()
        or detail.get("time_policy") != config.time_policy.as_dict()
    ):
        raise ValueError("V23h completed artifact time policy differs from its config")
    return {
        "config": config, "summary": summary, "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "records": records, "readiness_path": str(readiness_path), "readiness_sha256": sha256_file(readiness_path),
        "v23_pairing": pairing,
    }


def _scope_metrics(records: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    return summarize_known_gate_records(records, scope=f"{scope}_recomputed")


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {name: float(right[name] - left[name]) for name in left}


def _dt_parent_groups(root: Path, source_config: Any) -> dict[str, Any]:
    """Derive immutable development dt strata from the validation Dataset.

    The evaluation records deliberately contain no raw timing payload.  Use the
    already frozen V21 validation Dataset only to map every parent to one of
    two causal groups: all positive-dt after the first observation, or at
    least one same-second follow-up.  This does not construct Unknown/test,
    and it makes V23c's potential timing-frequency shortcut visible.
    """

    dataset = build_known_gate_dataset(root, source_config, split_name="val")
    groups: dict[str, set[str]] = {
        "all_positive_dt_after_first": set(),
        "contains_zero_delta": set(),
    }
    details: dict[str, dict[str, int | str]] = {}
    for item in dataset:
        parent_id = str(item.base.parent_sample_id)
        reasons = item.base.tr_dt_reason
        zero_count = int((reasons == 2).sum().item())
        positive_count = int((reasons == 3).sum().item())
        if zero_count:
            group = "contains_zero_delta"
        else:
            group = "all_positive_dt_after_first"
        groups[group].add(parent_id)
        details[parent_id] = {
            "group": group,
            "event_count": int(item.base.event_count),
            "zero_delta_event_count": zero_count,
            "positive_delta_event_count": positive_count,
            "label_name": str(item.base.label_name),
        }
    if len(groups["all_positive_dt_after_first"] | groups["contains_zero_delta"]) != len(dataset):
        raise ValueError("V23 adapter dt strata do not cover validation parents exactly once")
    if groups["all_positive_dt_after_first"] & groups["contains_zero_delta"]:
        raise ValueError("V23 adapter dt strata overlap")
    return {
        "groups": groups,
        "details": details,
        "validation_dataset_parent_count": len(dataset),
        "unknown_dataset_access": 0,
        "test_dataset_access": 0,
    }


def _positive_dt_cap_validation_exposure(root: Path, source_config: Any) -> dict[str, Any]:
    """Describe V23h validation cap exposure only during development evaluation."""

    dataset = build_known_gate_dataset(root, source_config, split_name="val")
    parent_ids: list[str] = []
    labels: dict[str, int] = {}
    event_count = 0
    maximum_seconds = 0.0
    for item in dataset:
        positive = item.base.tr_dt_reason == 3
        durations = item.base.tr_dt_seconds
        if bool(positive.any()):
            maximum_seconds = max(
                maximum_seconds,
                float(durations.masked_select(positive).max().item()),
            )
        exposed = positive & (durations > 120.0)
        count = int(exposed.sum().item())
        event_count += count
        if count:
            parent_ids.append(str(item.base.parent_sample_id))
            label = str(item.base.label_name)
            labels[label] = labels.get(label, 0) + 1
    return {
        "validation_parent_count": len(dataset),
        "validation_positive_dt_maximum_seconds": maximum_seconds,
        "validation_positive_dt_above_120_event_count": event_count,
        "validation_positive_dt_above_120_parent_count": len(parent_ids),
        "validation_positive_dt_above_120_parent_ids": parent_ids,
        "validation_positive_dt_above_120_parent_label_support": labels,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
    }


def _parent_records_for_ids(
    records: Sequence[Mapping[str, Any]], *, parent_ids: set[str]
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in records if str(row["parent_sample_id"]) in parent_ids]
    if len(selected) != len(parent_ids):
        raise ValueError("V23 adapter dt-stratum parent records are incomplete")
    return selected


def _dt_stratified_metrics(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    dt_groups: Mapping[str, set[str]],
    *,
    delta_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parent_ids in dt_groups.items():
        before = _parent_records_for_ids(baseline, parent_ids=parent_ids)
        after = _parent_records_for_ids(treatment, parent_ids=parent_ids)
        ordered_before, ordered_after = _ordered_pair(before, after, scope="parent")
        before_metrics = _scope_metrics(ordered_before, f"v23_{name}_parent")
        after_metrics = _scope_metrics(ordered_after, f"treatment_{name}_parent")
        labels = {
            label: sum(int(str(row["label_name"]) == label) for row in ordered_before)
            for label in ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
        }
        result[name] = {
            "support": len(ordered_before),
            "base_label_support": labels,
            "baseline": before_metrics,
            "treatment": after_metrics,
            delta_name: _delta(before_metrics, after_metrics),
        }
    return result


def _reference_metadata(treatment: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable fields needed to name a completed direct control."""

    summary = dict(treatment["summary"])
    config = treatment["config"]
    return {
        "variant": summary["variant"],
        "adapter": config.adapter.as_dict(),
        "summary_path": treatment["summary_path"],
        "summary_sha256": treatment["summary_sha256"],
        "checkpoint_path": treatment["checkpoint_path"],
        "checkpoint_sha256": treatment["checkpoint_sha256"],
        "readiness_path": treatment["readiness_path"],
        "readiness_sha256": treatment["readiness_sha256"],
        "best_epoch": summary["best_epoch"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
    }


def _v23d_direct_gate(
    direct_deltas: Mapping[str, Mapping[str, float]],
    direct_bootstrap: Mapping[str, Any],
    tau_audit: Mapping[str, Any],
) -> dict[str, bool]:
    """Freeze the decision rule for V23d versus its V23a fixed-tau control.

    V23d is not a second test of whether the official CfC is useful: V23a has
    already established that.  It can advance only if its extra 48 degrees of
    freedom improve parent-final evidence over *that* direct control without
    worsening the already weak causal-prefix evidence.  Prefix/window rows
    remain development diagnostics rather than independent parent samples.
    """

    parent = direct_deltas["parent"]
    eligible_prefix = direct_deltas["eligible_prefix"]
    window = direct_deltas["window"]
    parent_signal = (
        parent["macro_f1"] > 0.02
        and parent["known_target_recall"] >= 0.0
        and parent["known_clutter_f1"] >= 0.0
    )
    prefix_window_non_degrading = all(
        delta >= 0.0
        for delta in (
            eligible_prefix["macro_f1"],
            eligible_prefix["known_target_recall"],
            eligible_prefix["known_clutter_f1"],
            window["macro_f1"],
            window["known_target_recall"],
            window["known_clutter_f1"],
        )
    )
    bootstrap_positive = (
        float(
            dict(direct_bootstrap["treatment_minus_baseline"])["macro_f1"]
            ["lower_95"]
        )
        > 0.0
    )
    learned_from_fixed = bool(
        tau_audit["learned_values_differ_from_fixed_initialization"]
    )
    dimension_specific = float(
        tau_audit["log_tau_ratio_standard_deviation"]
    ) > 1.0e-6
    return {
        "parent_macro_f1_strictly_greater_than_2pp_vs_v23a": parent["macro_f1"]
        > 0.02,
        "known_target_recall_non_decreasing_vs_v23a": parent[
            "known_target_recall"
        ]
        >= 0.0,
        "clutter_f1_non_decreasing_vs_v23a": parent["known_clutter_f1"]
        >= 0.0,
        "parent_direct_single_factor_signal_passed": parent_signal,
        "paired_parent_bootstrap_macro_f1_lower_95_above_zero_vs_v23a": bootstrap_positive,
        "eligible_prefix_macro_f1_non_decreasing_vs_v23a": eligible_prefix[
            "macro_f1"
        ]
        >= 0.0,
        "eligible_prefix_known_target_recall_non_decreasing_vs_v23a": eligible_prefix[
            "known_target_recall"
        ]
        >= 0.0,
        "eligible_prefix_clutter_f1_non_decreasing_vs_v23a": eligible_prefix[
            "known_clutter_f1"
        ]
        >= 0.0,
        "window_macro_f1_non_decreasing_vs_v23a": window["macro_f1"] >= 0.0,
        "window_known_target_recall_non_decreasing_vs_v23a": window[
            "known_target_recall"
        ]
        >= 0.0,
        "window_clutter_f1_non_decreasing_vs_v23a": window["known_clutter_f1"]
        >= 0.0,
        "prefix_window_safety_non_degrading_vs_v23a": prefix_window_non_degrading,
        "per_dimension_tau_learned_from_fixed_initialization": learned_from_fixed,
        "per_dimension_tau_has_nonzero_cross_dimension_spread": dimension_specific,
        "advance_to_three_seed_confirmation": (
            parent_signal
            and bootstrap_positive
            and prefix_window_non_degrading
            and learned_from_fixed
            and dimension_specific
        ),
        "advance_to_oof": False,
    }


def _v23br_direct_gate(
    direct_deltas: Mapping[str, Mapping[str, float]],
    direct_bootstrap: Mapping[str, Any],
) -> dict[str, bool]:
    """Freeze V23b-r's repair test against V23b's affine LayerNorm.

    This is deliberately a *repair* comparison, not a second demand for a
    further two-point parent-final gain.  Removing beta is admissible only if
    it preserves V23b's parent-final evidence and does not worsen its already
    weak Clutter prefix/window boundary.  The direct bootstrap is retained as
    a diagnostic; robustness of the retained LayerNorm signal is separately
    assessed against the frozen V23 baseline.
    """

    parent = direct_deltas["parent"]
    eligible_prefix = direct_deltas["eligible_prefix"]
    window = direct_deltas["window"]
    bootstrap = dict(direct_bootstrap["treatment_minus_baseline"])
    parent_non_degrading = (
        parent["macro_f1"] >= 0.0
        and parent["known_target_recall"] >= 0.0
        and parent["known_clutter_f1"] >= 0.0
    )
    bootstrap_positive = float(bootstrap["macro_f1"]["lower_95"]) > 0.0
    prefix_window_safe = all(
        delta >= 0.0
        for delta in (
            eligible_prefix["macro_f1"],
            eligible_prefix["known_clutter_recall"],
            eligible_prefix["known_clutter_f1"],
            window["macro_f1"],
            window["known_clutter_recall"],
            window["known_clutter_f1"],
        )
    )
    direct_repair_passed = parent_non_degrading and prefix_window_safe
    return {
        "parent_macro_f1_non_decreasing_vs_v23b": parent["macro_f1"] >= 0.0,
        "known_target_recall_non_decreasing_vs_v23b": parent[
            "known_target_recall"
        ]
        >= 0.0,
        "clutter_f1_non_decreasing_vs_v23b": parent["known_clutter_f1"]
        >= 0.0,
        "parent_final_repair_non_degrading_vs_v23b": parent_non_degrading,
        "paired_parent_bootstrap_macro_f1_lower_95_above_zero_vs_v23b_descriptive": bootstrap_positive,
        "eligible_prefix_macro_f1_non_decreasing_vs_v23b": eligible_prefix[
            "macro_f1"
        ]
        >= 0.0,
        "eligible_prefix_clutter_recall_non_decreasing_vs_v23b": eligible_prefix[
            "known_clutter_recall"
        ]
        >= 0.0,
        "eligible_prefix_clutter_f1_non_decreasing_vs_v23b": eligible_prefix[
            "known_clutter_f1"
        ]
        >= 0.0,
        "window_macro_f1_non_decreasing_vs_v23b": window["macro_f1"] >= 0.0,
        "window_clutter_recall_non_decreasing_vs_v23b": window[
            "known_clutter_recall"
        ]
        >= 0.0,
        "window_clutter_f1_non_decreasing_vs_v23b": window["known_clutter_f1"]
        >= 0.0,
        "prefix_window_clutter_safety_non_degrading_vs_v23b": prefix_window_safe,
        "direct_control_repair_passed": direct_repair_passed,
        "advance_to_three_seed_confirmation": direct_repair_passed,
        "advance_to_oof": False,
    }


def _v23br_v23_reference_gate(
    deltas: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Any],
) -> dict[str, bool]:
    """Assess whether V23b-r retains a robust parent-final signal over V23.

    V23b-r has two separate questions: does removing beta repair the affine
    LayerNorm failure mode, and is the remaining scale-only mechanism more than
    a single-validation-split fluctuation?  This function answers only the
    latter, using the pre-registered V23 baseline and its paired parent
    bootstrap.  It intentionally does not turn prefix/window rows into
    independent samples; their direct V23b safety check remains separate.
    """

    parent = deltas["parent"]
    parent_signal = (
        parent["macro_f1"] > 0.02
        and parent["known_target_recall"] >= 0.0
        and parent["known_clutter_f1"] >= 0.0
    )
    bootstrap_positive = (
        float(bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"])
        > 0.0
    )
    robust_signal = parent_signal and bootstrap_positive
    return {
        "parent_macro_f1_strictly_greater_than_2pp_vs_v23": parent[
            "macro_f1"
        ]
        > 0.02,
        "known_target_recall_non_decreasing_vs_v23": parent[
            "known_target_recall"
        ]
        >= 0.0,
        "clutter_f1_non_decreasing_vs_v23": parent["known_clutter_f1"] >= 0.0,
        "parent_v23_reference_signal_passed": parent_signal,
        "paired_parent_bootstrap_macro_f1_lower_95_above_zero_vs_v23": bootstrap_positive,
        "v23_reference_signal_is_bootstrap_robust": robust_signal,
        "advance_to_three_seed_confirmation": robust_signal,
        "advance_to_oof": False,
    }


def _v23gr_direct_gate(
    direct_deltas: Mapping[str, Mapping[str, float]],
    direct_bootstrap: Mapping[str, Any],
) -> dict[str, bool]:
    """Frozen V23gr interpretation against fixed-gain V23c.

    V23gr is a mechanism replication test, not an opportunity to retune the
    already failed official CfC backbone.  The development parent signal is
    reported descriptively; Clutter has zero same-second exposure, so it is
    explicitly excluded from any claim of density-mechanism benefit.
    """

    parent = direct_deltas["parent"]
    prefix = direct_deltas["eligible_prefix"]
    parent_balanced_prefix = direct_deltas["parent_balanced_eligible_prefix"]
    window = direct_deltas["window"]
    bootstrap_positive = (
        float(direct_bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"])
        > 0.0
    )
    known_target_non_degrading = parent["known_target_recall"] >= 0.0
    parent_balanced_prefix_non_degrading = all(
        value >= 0.0
        for value in (
            parent_balanced_prefix["macro_f1"],
            parent_balanced_prefix["known_target_recall"],
        )
    )
    event_pooled_prefix_window_non_degrading = all(
        value >= 0.0
        for value in (
            prefix["macro_f1"],
            prefix["known_target_recall"],
            window["macro_f1"],
            window["known_target_recall"],
        )
    )
    return {
        "parent_macro_f1_strictly_greater_than_2pp_vs_v23c": parent["macro_f1"]
        > 0.02,
        "known_target_recall_non_decreasing_vs_v23c": known_target_non_degrading,
        "parent_clutter_metrics_are_not_density_estimable": True,
        "eligible_prefix_known_target_metrics_non_degrading_vs_v23c": (
            prefix["macro_f1"] >= 0.0 and prefix["known_target_recall"] >= 0.0
        ),
        "parent_balanced_eligible_prefix_known_target_metrics_non_degrading_vs_v23c": (
            parent_balanced_prefix_non_degrading
        ),
        "window_known_target_metrics_non_degrading_vs_v23c": (
            window["macro_f1"] >= 0.0 and window["known_target_recall"] >= 0.0
        ),
        "event_pooled_prefix_window_known_target_safety_non_degrading_vs_v23c": (
            event_pooled_prefix_window_non_degrading
        ),
        "parent_balanced_prefix_and_window_known_target_safety_non_degrading_vs_v23c": (
            parent_balanced_prefix_non_degrading
            and window["macro_f1"] >= 0.0
            and window["known_target_recall"] >= 0.0
        ),
        "paired_parent_bootstrap_macro_f1_lower_95_above_zero_vs_v23c": (
            bootstrap_positive
        ),
        "density_mechanism_development_signal": (
            parent["macro_f1"] > 0.02
            and known_target_non_degrading
            and parent_balanced_prefix_non_degrading
            and event_pooled_prefix_window_non_degrading
            and bootstrap_positive
        ),
        "advance_to_three_seed_confirmation": False,
        "advance_to_oof": False,
    }


def _v23h_direct_gate(
    direct_deltas: Mapping[str, Mapping[str, float]],
    direct_bootstrap: Mapping[str, Any],
    *,
    validation_above_cap_parent_count: int,
) -> dict[str, bool]:
    """Describe V23h without claiming validation evidence that does not exist."""

    parent = direct_deltas["parent"]
    prefix = direct_deltas["eligible_prefix"]
    window = direct_deltas["window"]
    bootstrap_positive = (
        float(direct_bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"])
        > 0.0
    )
    parent_non_degrading = (
        parent["macro_f1"] >= 0.0
        and parent["known_target_recall"] >= 0.0
        and parent["known_clutter_f1"] >= 0.0
    )
    prefix_window_non_degrading = all(
        value >= 0.0
        for value in (
            prefix["macro_f1"],
            prefix["known_target_recall"],
            prefix["known_clutter_f1"],
            window["macro_f1"],
            window["known_target_recall"],
            window["known_clutter_f1"],
        )
    )
    validation_exposes_cap = validation_above_cap_parent_count > 0
    return {
        "parent_macro_f1_non_decreasing_vs_v23a": parent["macro_f1"] >= 0.0,
        "known_target_recall_non_decreasing_vs_v23a": parent[
            "known_target_recall"
        ] >= 0.0,
        "clutter_f1_non_decreasing_vs_v23a": parent["known_clutter_f1"] >= 0.0,
        "parent_final_non_degrading_vs_v23a": parent_non_degrading,
        "eligible_prefix_and_window_non_degrading_vs_v23a": prefix_window_non_degrading,
        "paired_parent_bootstrap_macro_f1_lower_95_above_zero_vs_v23a_descriptive": bootstrap_positive,
        "validation_has_above_120_second_cap_exposure": validation_exposes_cap,
        "cap_effect_is_estimable_on_development_validation": validation_exposes_cap,
        "advance_to_three_seed_confirmation": False,
        "advance_to_oof": False,
    }


def run(project_root: Path, *, variant: str) -> dict[str, Any]:
    if variant not in V23_ADAPTER_VARIANTS:
        raise ValueError(f"unsupported V23 adapter variant: {variant}")
    root = project_root.resolve()
    baseline, treatment = _load_v23(root), _load_treatment(root, variant)
    source_config = load_adapter_config(root, variant)[1]
    dt_strata = _dt_parent_groups(root, source_config)
    metrics = {scope: {"baseline": _scope_metrics(baseline["records"][scope], f"v23_{scope}"), "treatment": _scope_metrics(treatment["records"][scope], f"{variant}_{scope}")} for scope in SCOPES}
    deltas = {scope: _delta(values["baseline"], values["treatment"]) for scope, values in metrics.items()}
    parent_before, parent_after = _ordered_pair(baseline["records"]["parent"], treatment["records"]["parent"], scope="parent")
    bootstrap = _bootstrap(parent_before, parent_after)
    outcomes = _base_label_outcomes(parent_before, parent_after)
    dt_metrics = _dt_stratified_metrics(
        parent_before,
        parent_after,
        dt_strata["groups"],
        delta_name="treatment_minus_v23",
    )
    known_target_recall_non_decreasing = deltas["parent"]["known_target_recall"] >= 0.0
    clutter_f1_non_decreasing = deltas["parent"]["known_clutter_f1"] >= 0.0
    macro_two_point_signal = deltas["parent"]["macro_f1"] > 0.02
    bootstrap_positive = bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"] > 0.0
    parent_signal = macro_two_point_signal and known_target_recall_non_decreasing and clutter_f1_non_decreasing
    v21_reference_reached = (
        metrics["parent"]["treatment"]["macro_f1"] >= 0.862803075103489
        and _metric_vector(metrics["parent"]["treatment"])["known_clutter_f1"] >= 0.7368421052631577
    )
    positive_dt_delta = dt_metrics["all_positive_dt_after_first"]["treatment_minus_v23"]
    positive_dt_non_degrading = positive_dt_delta["macro_f1"] >= 0.0
    report = {
        "status": "completed_development_single_factor_only",
        "schema_version": report_schema(variant),
        "experiment": f"{variant} isolated official-CfC external-state-adapter paired development evaluation",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, stable or P95",
        "source_control": {
            "name": "frozen V23 pure official dense CfC negative baseline", **{key: baseline[key] for key in ("summary_path", "summary_sha256", "checkpoint_path", "checkpoint_sha256")},
        },
        "treatment": {
            "variant": variant, "adapter": treatment["config"].adapter.as_dict(),
            **{key: treatment[key] for key in ("summary_path", "summary_sha256", "checkpoint_path", "checkpoint_sha256", "readiness_path", "readiness_sha256")},
            "best_epoch": treatment["summary"]["best_epoch"], "trainable_parameter_count": treatment["summary"]["trainable_parameter_count"],
            "v23_initialization_pairing": treatment["v23_pairing"],
        },
        "metrics_by_scope": metrics,
        "treatment_minus_v23_by_scope": deltas,
        "parent_base_label_outcomes": outcomes,
        "scope_prediction_changes": _scope_changes(baseline["records"], treatment["records"]),
        "probability_attribution": _probability_attribution(parent_before, parent_after),
        "paired_parent_bootstrap": bootstrap,
        "dt_stratified_parent_final": {
            "definition": {
                "all_positive_dt_after_first": "contains no zero_delta_at_second_precision event after the mandatory first event",
                "contains_zero_delta": "contains at least one zero_delta_at_second_precision event",
            },
            "derivation": "frozen V21 development Dataset tr_dt_reason only; no test or Unknown Dataset constructed",
            "groups": dt_metrics,
            "parent_details": dt_strata["details"],
            "validation_dataset_parent_count": dt_strata["validation_dataset_parent_count"],
            "unknown_dataset_access": dt_strata["unknown_dataset_access"],
            "test_dataset_access": dt_strata["test_dataset_access"],
        },
        "gate": {
            "parent_macro_f1_strictly_greater_than_2pp": macro_two_point_signal,
            "known_target_recall_non_decreasing": known_target_recall_non_decreasing,
            "clutter_f1_non_decreasing": clutter_f1_non_decreasing,
            "all_positive_dt_parent_macro_f1_non_decreasing": positive_dt_non_degrading,
            "parent_single_factor_signal_passed": parent_signal and positive_dt_non_degrading,
            "bootstrap_macro_f1_lower_95_above_zero": bootstrap_positive,
            "v21_seed43_reference_reached": v21_reference_reached,
            "advance_to_three_seed_confirmation": parent_signal and positive_dt_non_degrading and v21_reference_reached,
            "advance_to_oof": False,
        },
        "statistical_boundary": {
            "development_parent_count": 232, "clutter_parent_support": 9,
            "bootstrap_is_descriptive_not_new_holdout": True,
            "single_seed_is_not_confirmation": True,
            "dt_stratum_metrics_are_development_descriptive_only": True,
        },
        "OOF": "未执行", "test": "未执行", "formal_rejection": "未执行", "full_cascade": "未执行", "stable": "未执行", "P95": "未执行", "deployment_eligible": False,
    }
    if variant == V23H_VARIANT:
        report["treatment"]["time_policy"] = treatment["config"].time_policy.as_dict()
    if variant == V23D_VARIANT:
        direct_control = _load_treatment(root, V23A_VARIANT)
        direct_metrics = {
            scope: {
                "v23a_fixed_tau_control": _scope_metrics(
                    direct_control["records"][scope], f"v23a_{scope}"
                ),
                "v23d_per_dimension_tau_treatment": _scope_metrics(
                    treatment["records"][scope], f"v23d_{scope}"
                ),
            }
            for scope in SCOPES
        }
        direct_deltas = {
            scope: _delta(
                values["v23a_fixed_tau_control"],
                values["v23d_per_dimension_tau_treatment"],
            )
            for scope, values in direct_metrics.items()
        }
        direct_parent_before, direct_parent_after = _ordered_pair(
            direct_control["records"]["parent"],
            treatment["records"]["parent"],
            scope="parent",
        )
        direct_bootstrap = _bootstrap(direct_parent_before, direct_parent_after)
        direct_outcomes = _base_label_outcomes(
            direct_parent_before, direct_parent_after
        )
        direct_dt_metrics = _dt_stratified_metrics(
            direct_parent_before,
            direct_parent_after,
            dt_strata["groups"],
            delta_name="v23d_minus_v23a",
        )
        tau_audit = dict(treatment["summary"]["v23d_per_dimension_tau_audit"])
        direct_gate = _v23d_direct_gate(
            direct_deltas,
            direct_bootstrap,
            tau_audit,
        )
        report["direct_control"] = {
            "binding": treatment["config"].direct_control,
            "completed_v23a_fixed_tau_treatment": _reference_metadata(direct_control),
        }
        report["v23d_per_dimension_tau_audit"] = tau_audit
        report["v23d_vs_v23a_direct_control"] = {
            "scope": (
                "primary V23d decision comparison: completed V23a fixed-tau "
                "treatment versus V23d per-dimension-tau treatment; development "
                "KnownGate only"
            ),
            "metrics_by_scope": direct_metrics,
            "v23d_minus_v23a_by_scope": direct_deltas,
            "parent_base_label_outcomes": direct_outcomes,
            "scope_prediction_changes": _scope_changes(
                direct_control["records"], treatment["records"]
            ),
            "probability_attribution": _probability_attribution(
                direct_parent_before, direct_parent_after
            ),
            "paired_parent_bootstrap": direct_bootstrap,
            "dt_stratified_parent_final": direct_dt_metrics,
            "gate": direct_gate,
        }
        report["gate"] = {
            "v23_reference_diagnostic": report["gate"],
            "v23a_direct_control_primary": direct_gate,
            "advance_to_three_seed_confirmation": direct_gate[
                "advance_to_three_seed_confirmation"
            ],
            "advance_to_oof": False,
        }
    elif variant == V23BR_VARIANT:
        direct_control = _load_treatment(root, V23B_VARIANT)
        direct_metrics = {
            scope: {
                "v23b_affine_layer_norm_control": _scope_metrics(
                    direct_control["records"][scope], f"v23b_{scope}"
                ),
                "v23br_scale_only_layer_norm_treatment": _scope_metrics(
                    treatment["records"][scope], f"v23br_{scope}"
                ),
            }
            for scope in SCOPES
        }
        direct_deltas = {
            scope: _delta(
                values["v23b_affine_layer_norm_control"],
                values["v23br_scale_only_layer_norm_treatment"],
            )
            for scope, values in direct_metrics.items()
        }
        direct_parent_before, direct_parent_after = _ordered_pair(
            direct_control["records"]["parent"],
            treatment["records"]["parent"],
            scope="parent",
        )
        direct_bootstrap = _bootstrap(direct_parent_before, direct_parent_after)
        direct_outcomes = _base_label_outcomes(
            direct_parent_before, direct_parent_after
        )
        direct_dt_metrics = _dt_stratified_metrics(
            direct_parent_before,
            direct_parent_after,
            dt_strata["groups"],
            delta_name="v23br_minus_v23b",
        )
        direct_gate = _v23br_direct_gate(direct_deltas, direct_bootstrap)
        reference_gate = _v23br_v23_reference_gate(deltas, bootstrap)
        advance_to_three_seed_confirmation = (
            reference_gate["advance_to_three_seed_confirmation"]
            and direct_gate["direct_control_repair_passed"]
        )
        report["direct_control"] = {
            "binding": treatment["config"].direct_control,
            "completed_v23b_affine_layer_norm_treatment": _reference_metadata(
                direct_control
            ),
        }
        report["v23br_vs_v23b_direct_control"] = {
            "scope": (
                "primary V23b-r repair comparison: completed V23b all-event "
                "affine LayerNorm treatment versus V23b-r scale-only all-event "
                "LayerNorm treatment; development KnownGate only"
            ),
            "metrics_by_scope": direct_metrics,
            "v23br_minus_v23b_by_scope": direct_deltas,
            "parent_base_label_outcomes": direct_outcomes,
            "scope_prediction_changes": _scope_changes(
                direct_control["records"], treatment["records"]
            ),
            "probability_attribution": _probability_attribution(
                direct_parent_before, direct_parent_after
            ),
            "paired_parent_bootstrap": direct_bootstrap,
            "dt_stratified_parent_final": direct_dt_metrics,
            "gate": direct_gate,
        }
        report["gate"] = {
            "v23_reference_diagnostic": report["gate"],
            "v23_reference_primary": reference_gate,
            "v23b_direct_control_primary": direct_gate,
            "v23br_scale_only_signal_is_robust_and_repairs_v23b": advance_to_three_seed_confirmation,
            "advance_to_three_seed_confirmation": advance_to_three_seed_confirmation,
            "advance_to_oof": False,
        }
    elif variant == V23GR_VARIANT:
        direct_control = _load_treatment(
            root, "official_cfc_zero_delta_bounded_observation"
        )
        direct_metrics = {
            scope: {
                "v23c_fixed_zero_delta_gain_control": _scope_metrics(
                    direct_control["records"][scope], f"v23c_{scope}"
                ),
                "v23gr_density_weighted_multiplicity_treatment": _scope_metrics(
                    treatment["records"][scope], f"v23gr_{scope}"
                ),
            }
            for scope in SCOPES
        }
        direct_metrics["parent_balanced_eligible_prefix"] = {
            "v23c_fixed_zero_delta_gain_control": (
                summarize_parent_balanced_eligible_prefix_records(
                    direct_control["records"]["eligible_prefix"],
                    scope="v23c_parent_balanced_eligible_prefix",
                )
            ),
            "v23gr_density_weighted_multiplicity_treatment": (
                summarize_parent_balanced_eligible_prefix_records(
                    treatment["records"]["eligible_prefix"],
                    scope="v23gr_parent_balanced_eligible_prefix",
                )
            ),
        }
        direct_deltas = {
            scope: _delta(
                values["v23c_fixed_zero_delta_gain_control"],
                values["v23gr_density_weighted_multiplicity_treatment"],
            )
            for scope, values in direct_metrics.items()
        }
        direct_parent_before, direct_parent_after = _ordered_pair(
            direct_control["records"]["parent"], treatment["records"]["parent"], scope="parent"
        )
        direct_bootstrap = _bootstrap(direct_parent_before, direct_parent_after)
        direct_outcomes = _base_label_outcomes(direct_parent_before, direct_parent_after)
        direct_dt_metrics = _dt_stratified_metrics(
            direct_parent_before,
            direct_parent_after,
            dt_strata["groups"],
            delta_name="v23gr_minus_v23c",
        )
        clutter_details = {
            parent_id: values
            for parent_id, values in dt_strata["details"].items()
            if values["label_name"] == "ClutterTarget"
        }
        if any(int(values["zero_delta_event_count"]) != 0 for values in clutter_details.values()):
            raise ValueError("V23gr Clutter exposure audit differs from frozen development data")
        direct_gate = _v23gr_direct_gate(direct_deltas, direct_bootstrap)
        report["direct_control"] = {
            "binding": treatment["config"].direct_control,
            "completed_v23c_fixed_gain_treatment": _reference_metadata(direct_control),
        }
        report["v23gr_vs_v23c_direct_control"] = {
            "scope": (
                "primary V23gr mechanism comparison: completed V23c fixed 0.25 "
                "zero-delta external adapter versus V23gr causal multiplicity "
                "min(0.25*k,0.75); development KnownGate only"
            ),
            "metrics_by_scope": direct_metrics,
            "v23gr_minus_v23c_by_scope": direct_deltas,
            "parent_base_label_outcomes": direct_outcomes,
            "scope_prediction_changes": _scope_changes(
                direct_control["records"], treatment["records"]
            ),
            "probability_attribution": _probability_attribution(
                direct_parent_before, direct_parent_after
            ),
            "paired_parent_bootstrap": direct_bootstrap,
            "dt_stratified_parent_final": direct_dt_metrics,
            "clutter_density_exposure_boundary": {
                "clutter_parent_support": len(clutter_details),
                "clutter_zero_delta_event_count": 0,
                "interpretation": "No development Clutter parent has a zero-delta follow-up, so V23gr cannot identify a Clutter-specific same-second-density effect.",
            },
            "gate": direct_gate,
        }
        report["gate"] = {
            "v23_reference_diagnostic": report["gate"],
            "v23c_direct_control_primary": direct_gate,
            "advance_to_three_seed_confirmation": False,
            "advance_to_oof": False,
        }
    elif variant == V23H_VARIANT:
        direct_control = _load_treatment(root, V23A_VARIANT)
        direct_metrics = {
            scope: {
                "v23a_raw_positive_dt_control": _scope_metrics(
                    direct_control["records"][scope], f"v23a_{scope}"
                ),
                "v23h_positive_dt_capped_120_seconds_treatment": _scope_metrics(
                    treatment["records"][scope], f"v23h_{scope}"
                ),
            }
            for scope in SCOPES
        }
        direct_deltas = {
            scope: _delta(
                values["v23a_raw_positive_dt_control"],
                values["v23h_positive_dt_capped_120_seconds_treatment"],
            )
            for scope, values in direct_metrics.items()
        }
        direct_parent_before, direct_parent_after = _ordered_pair(
            direct_control["records"]["parent"],
            treatment["records"]["parent"],
            scope="parent",
        )
        direct_bootstrap = _bootstrap(direct_parent_before, direct_parent_after)
        direct_outcomes = _base_label_outcomes(direct_parent_before, direct_parent_after)
        direct_dt_metrics = _dt_stratified_metrics(
            direct_parent_before,
            direct_parent_after,
            dt_strata["groups"],
            delta_name="v23h_minus_v23a",
        )
        contract_path = root / "data/processed/RADAR-LN3-V23h-cq08-known-gate-official-cfc-positive-dt-fixed-retention-capped-120-seconds-contract.json"
        contract = _read_json(contract_path)
        train_exposure = dict(contract.get("positive_dt_cap_exposure_audit") or {})
        validation_exposure = _positive_dt_cap_validation_exposure(root, source_config)
        if (
            int(train_exposure.get("train_positive_dt_above_120_parent_count", 0)) <= 0
            or int(train_exposure.get("validation_dataset_access_count", -1)) != 0
            or int(train_exposure.get("unknown_dataset_access_count", -1)) != 0
            or int(train_exposure.get("test_dataset_access_count", -1)) != 0
            or int(validation_exposure.get("unknown_dataset_access_count", -1)) != 0
            or int(validation_exposure.get("test_dataset_access_count", -1)) != 0
        ):
            raise ValueError("V23h cap exposure audit differs from its sealed scope")
        direct_gate = _v23h_direct_gate(
            direct_deltas,
            direct_bootstrap,
            validation_above_cap_parent_count=int(
                validation_exposure["validation_positive_dt_above_120_parent_count"]
            ),
        )
        report["direct_control"] = {
            "binding": treatment["config"].direct_control,
            "completed_v23a_raw_positive_dt_fixed_retention_treatment": _reference_metadata(
                direct_control
            ),
        }
        report["v23h_vs_v23a_direct_control"] = {
            "scope": (
                "primary V23h mechanism comparison: completed V23a raw-positive-dt "
                "fixed-retention treatment versus V23h effective positive-dt "
                "min(raw_dt,120s) for both official CfC and fixed-retention alpha; "
                "development KnownGate only"
            ),
            "metrics_by_scope": direct_metrics,
            "v23h_minus_v23a_by_scope": direct_deltas,
            "parent_base_label_outcomes": direct_outcomes,
            "scope_prediction_changes": _scope_changes(
                direct_control["records"], treatment["records"]
            ),
            "probability_attribution": _probability_attribution(
                direct_parent_before, direct_parent_after
            ),
            "paired_parent_bootstrap": direct_bootstrap,
            "dt_stratified_parent_final": direct_dt_metrics,
            "train_cap_exposure_audit": train_exposure,
            "validation_cap_exposure_audit": validation_exposure,
            "development_cap_exposure_boundary": {
                "validation_above_120_second_parent_count": int(
                    validation_exposure["validation_positive_dt_above_120_parent_count"]
                ),
                "interpretation": "The cap has train exposure but no development-validation parent with dt>120s; development metrics cannot identify a long-positive-dt cap effect.",
            },
            "gate": direct_gate,
        }
        report["gate"] = {
            "v23_reference_diagnostic": report["gate"],
            "v23a_direct_control_primary": direct_gate,
            "advance_to_three_seed_confirmation": False,
            "advance_to_oof": False,
        }
    output = root / report_relative(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report one V23 external-state-adapter experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=V23_ADAPTER_VARIANTS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, variant=args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
