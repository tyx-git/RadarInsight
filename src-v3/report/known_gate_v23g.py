"""Development-only paired report for V23g self-developed density weighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import (
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v23g import (
    V23G_REPORT_SCHEMA,
    V23G_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_config,
    readiness_relative,
    report_relative,
    sha256_file,
)
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any], *, name: str) -> dict[str, list[dict[str, Any]]]:
    records = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in SCOPES
    }
    if (
        len(records["parent"]) != 232
        or len(records["eligible_prefix"]) != 3756
        or len(records["window"]) != 4452
    ):
        raise ValueError(f"{name} record support differs from frozen V21e scope")
    return records


def _metrics(records: Mapping[str, Sequence[Mapping[str, Any]]], *, prefix: str) -> dict[str, Any]:
    return {
        "parent": summarize_known_gate_records(records["parent"], scope=f"{prefix}_parent_final"),
        "eligible_prefix": summarize_known_gate_records(
            records["eligible_prefix"], scope=f"{prefix}_eligible_prefix"
        ),
        "parent_balanced_eligible_prefix": summarize_parent_balanced_eligible_prefix_records(
            records["eligible_prefix"], scope=f"{prefix}_parent_balanced_eligible_prefix"
        ),
        "window": summarize_known_gate_records(records["window"], scope=f"{prefix}_window"),
    }


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {key: float(right[key] - left[key]) for key in left}


def _density_groups(root: Path, source_config: Any) -> dict[str, Any]:
    """Derive causal multiplicity groups from development timing metadata only."""

    from radar_ln3.data import DT_REASON_ZERO_DELTA
    from radar_ln3.known_gate import build_known_gate_dataset

    dataset = build_known_gate_dataset(root, source_config, split_name="val")
    groups = {
        "no_zero_delta": set(),
        "zero_delta_k2_only": set(),
        "zero_delta_k3_or_later": set(),
    }
    details: dict[str, dict[str, Any]] = {}
    for item in dataset:
        parent_id = str(item.base.parent_sample_id)
        ordinal = 0
        maximum = 0
        zero_count = 0
        for reason in item.base.tr_dt_reason.tolist():
            if reason in (1, 3):
                ordinal = 1
            elif reason == DT_REASON_ZERO_DELTA:
                zero_count += 1
                ordinal += 1
                maximum = max(maximum, ordinal)
        group = (
            "no_zero_delta"
            if zero_count == 0
            else "zero_delta_k2_only"
            if maximum == 2
            else "zero_delta_k3_or_later"
        )
        groups[group].add(parent_id)
        details[parent_id] = {
            "group": group,
            "label_name": str(item.base.label_name),
            "zero_delta_event_count": zero_count,
            "maximum_causal_zero_ordinal": maximum,
        }
    if sum(len(group) for group in groups.values()) != len(dataset):
        raise ValueError("V23g density groups do not partition development parents")
    clutter = [item for item in details.values() if item["label_name"] == "ClutterTarget"]
    if len(clutter) != 9 or any(item["zero_delta_event_count"] != 0 for item in clutter):
        raise ValueError("V23g development Clutter timing exposure changed")
    return {"groups": groups, "details": details, "clutter": clutter}


def _stratified_metrics(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    groups: Mapping[str, set[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, ids in groups.items():
        before = [dict(row) for row in baseline if str(row["parent_sample_id"]) in ids]
        after = [dict(row) for row in treatment if str(row["parent_sample_id"]) in ids]
        paired_before, paired_after = _ordered_pair(before, after, scope="parent")
        before_metrics = summarize_known_gate_records(
            paired_before, scope=f"v23g_control_{name}"
        )
        after_metrics = summarize_known_gate_records(
            paired_after, scope=f"v23g_treatment_{name}"
        )
        result[name] = {
            "support": len(ids),
            "base_label_support": {
                label: sum(
                    str(row["label_name"]) == label for row in paired_before
                )
                for label in (
                    "DroneTarget",
                    "BirdTarget",
                    "BalloonTarget",
                    "ClutterTarget",
                )
            },
            "control": before_metrics,
            "treatment": after_metrics,
            "treatment_minus_control": _delta(before_metrics, after_metrics),
        }
    return result


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    wrapper, source_config, _source_path = load_config(root)
    config_path = root / config_relative()
    readiness_path = root / readiness_relative()
    artifact = root / artifact_relative()
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    control_detail_path = root / str(wrapper.source_control["detail_path"])
    control_summary_path = root / str(wrapper.source_control["summary_path"])
    if not all(
        path.is_file()
        for path in (
            config_path,
            readiness_path,
            summary_path,
            detail_path,
            checkpoint_path,
            control_detail_path,
            control_summary_path,
        )
    ):
        raise FileNotFoundError("V23g control/treatment artifacts are incomplete")
    readiness = _read_json(readiness_path)
    summary = _read_json(summary_path)
    detail = _read_json(detail_path)
    control_summary = _read_json(control_summary_path)
    control_detail = _read_json(control_detail_path)
    if (
        readiness.get("status") != "passed"
        or summary.get("status") != "completed"
        or summary.get("variant") != V23G_VARIANT
        or summary.get("architecture")
        != "self_developed_cfc_liquid_causal_encoder_density_weighted"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(wrapper)
        or summary.get("source_control") != wrapper.source_control
        or int(summary.get("best_epoch", -1)) < 1
        or len(list(summary.get("history") or ())) != 20
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or detail.get("variant") != V23G_VARIANT
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
        or str(detail.get("checkpoint", "")) != str(checkpoint_path)
        or control_summary.get("status") != "completed"
        or control_summary.get("variant") != "v21e_course_prefix"
    ):
        raise ValueError("V23g paired artifacts violate the frozen local contract")
    control_records = _records(control_detail, name="V21e control")
    treatment_records = _records(detail, name="V23g treatment")
    metrics = {
        "control": _metrics(control_records, prefix="v21e_control"),
        "treatment": _metrics(treatment_records, prefix="v23g_treatment"),
    }
    deltas = {
        scope: _delta(metrics["control"][scope], metrics["treatment"][scope])
        for scope in ("parent", "eligible_prefix", "parent_balanced_eligible_prefix", "window")
    }
    parent_before, parent_after = _ordered_pair(
        control_records["parent"], treatment_records["parent"], scope="parent"
    )
    groups = _density_groups(root, source_config)
    bootstrap = _bootstrap(parent_before, parent_after)
    outcomes = _base_label_outcomes(parent_before, parent_after)
    parent_macro_ci = float(
        bootstrap["treatment_minus_baseline"]["macro_f1"]["lower_95"]
    )
    parent_no_regression = (
        deltas["parent"]["known_target_recall"] >= 0.0
        and deltas["parent"]["known_clutter_f1"] >= 0.0
    )
    prefix_safety = (
        deltas["parent_balanced_eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["parent_balanced_eligible_prefix"]["known_clutter_f1"] >= 0.0
    )
    density_groups = _stratified_metrics(parent_before, parent_after, groups["groups"])
    density_exposed_known_target_parent_count = sum(
        int(values["base_label_support"]["DroneTarget"])
        + int(values["base_label_support"]["BirdTarget"])
        + int(values["base_label_support"]["BalloonTarget"])
        for name, values in density_groups.items()
        if name != "no_zero_delta"
    )
    report = {
        "status": "completed_development_single_factor_only",
        "schema_version": V23G_REPORT_SCHEMA,
        "experiment": "V23g self-developed CfC density-weighted versus V21e fixed 0.25 bounded observation",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "source_control": wrapper.source_control,
        "treatment": {
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": int(summary["best_epoch"]),
            "initialization_audit": summary["initialization_audit"],
        },
        "mechanism": {
            "control": "bounded_observation zero-delta gain=0.25",
            "treatment": "density_weighted zero-delta gain=min(0.25*k,0.75)",
            "causal_ordinal": "first/positive sets k=1; zero_delta increments k; first zero-delta follow-up is k=2",
        },
        "metrics_by_scope": metrics,
        "treatment_minus_control_by_scope": deltas,
        "parent_base_label_outcomes": outcomes,
        "scope_prediction_changes": _scope_changes(control_records, treatment_records),
        "probability_attribution": _probability_attribution(parent_before, parent_after),
        "paired_parent_bootstrap": bootstrap,
        "density_exposure_parent_final": {
            "groups": density_groups,
            "parent_details": groups["details"],
            "clutter_density_exposure_boundary": {
                "clutter_parent_support": 9,
                "clutter_zero_delta_event_count": 0,
                "interpretation": "V23g cannot estimate a Clutter-specific same-second-density effect on this development split.",
            },
        },
        "gate": {
            "parent_final_is_not_a_2pp_endpoint_because_v21e_control_is_100_percent": True,
            "parent_final_known_target_recall_non_decreasing": (
                deltas["parent"]["known_target_recall"] >= 0.0
            ),
            "parent_final_clutter_f1_non_decreasing": (
                deltas["parent"]["known_clutter_f1"] >= 0.0
            ),
            "parent_balanced_eligible_prefix_known_target_recall_non_decreasing": (
                deltas["parent_balanced_eligible_prefix"]["known_target_recall"] >= 0.0
            ),
            "parent_balanced_eligible_prefix_clutter_f1_non_decreasing": (
                deltas["parent_balanced_eligible_prefix"]["known_clutter_f1"] >= 0.0
            ),
            "paired_parent_bootstrap_macro_f1_lower_95_above_zero_descriptive": (
                parent_macro_ci > 0.0
            ),
            "density_exposed_known_target_parent_count": density_exposed_known_target_parent_count,
            "parent_and_prefix_safety_non_degrading": parent_no_regression and prefix_safety,
            "advance_to_three_seed_confirmation": False,
            "advance_to_oof": False,
        },
        "statistical_boundary": {
            "development_parent_count": 232,
            "clutter_parent_support": 9,
            "bootstrap_is_descriptive_not_new_holdout": True,
            "single_seed_is_not_confirmation": True,
            "clutter_same_second_density_effect_is_not_estimable": True,
        },
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / report_relative()
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report V23g density-weighted KnownGate.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
