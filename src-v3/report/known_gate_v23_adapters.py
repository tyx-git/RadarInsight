"""Paired development report for one completed V23a/V23b/V23c experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_v23_adapters import (
    V23_ADAPTER_VARIANTS,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_adapter_config,
    readiness_relative,
    report_relative,
    report_schema,
    source_v23_parent_order_hashes,
    source_v23_training,
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
        "summary": summary, "summary_path": summary_path, "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": checkpoint_path, "checkpoint_sha256": sha256_file(checkpoint_path), "records": records,
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
            == ["state_adapter.recurrent_norm.bias", "state_adapter.recurrent_norm.weight"]
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
        or not initialization_ok or not parent_order_ok
        or dict(summary.get("dataloader_contract") or {}).get(
            "all_epoch_parent_orders_exactly_match_v23_control"
        ) is not True
        or dict(summary.get("training_readiness") or {}).get("sha256")
        != sha256_file(readiness_path)
    ):
        raise ValueError(f"{variant} completed artifact violates its contract")
    return {
        "config": config, "summary": summary, "summary_path": summary_path, "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": checkpoint_path, "checkpoint_sha256": sha256_file(checkpoint_path),
        "records": records, "readiness_path": readiness_path, "readiness_sha256": sha256_file(readiness_path),
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
            "treatment_minus_v23": _delta(before_metrics, after_metrics),
        }
    return result


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
