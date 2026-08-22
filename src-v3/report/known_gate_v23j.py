"""Development-only report for the completed V23j prefix-ramp experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate import (
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23e_selection import (
    V23E_SELECTOR_SCHEMA,
    clutter_prefix_action_trajectories,
    frozen_v23a_legacy_reference,
)
from radar_ln3.known_gate_v23j import (
    V23J_REPORT_SCHEMA,
    V23J_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_config,
    prefix_auxiliary_weight_for_epoch,
    readiness_relative,
    report_relative,
)
from report.known_gate_v23e_selection import _selection_bootstrap
from report.known_gate_v23_adapters import _bootstrap, _delta, _ordered_pair, _scope_metrics


V23E_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection.json"
)
V23E_REPORT_SCHEMA = (
    "RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-report-v1"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [
            dict(row)
            for row in list(dict(detail.get("records") or {}).get(scope) or ())
        ]
        for scope in ("parent", "eligible_prefix", "window")
    }
    if (
        len(rows["parent"]) != 232
        or len(rows["eligible_prefix"]) != 3756
        or len(rows["window"]) != 4452
    ):
        raise ValueError("V23j detail record counts differ from frozen V21 scope")
    return rows


def _metrics(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "parent_metrics": summarize_known_gate_records(
            rows["parent"], scope="parent_final"
        ),
        "eligible_prefix_metrics": summarize_known_gate_records(
            rows["eligible_prefix"],
            scope="eligible_causal_prefix_auxiliary_excluding_final",
        ),
        "parent_balanced_eligible_prefix_metrics": (
            summarize_parent_balanced_eligible_prefix_records(
                rows["eligible_prefix"],
                scope="parent_balanced_eligible_causal_prefix_auxiliary_excluding_final",
            )
        ),
        "window_metrics": summarize_known_gate_records(
            rows["window"], scope="causal_prefix_windows"
        ),
    }


def _v23a_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load V23a's frozen hard actions for the direct paired comparison."""

    artifact = root / (
        "artifacts/"
        "RADAR-LN3-V23a-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "seed43-formal"
    )
    detail_path = artifact / "sample_detail_information.json"
    if not detail_path.is_file():
        raise FileNotFoundError("V23j requires V23a sample detail: " + str(detail_path))
    detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
    rows = _records(detail)
    if detail.get("variant") != "official_cfc_positive_dt_fixed_retention":
        raise ValueError("V23j direct comparator is not V23a fixed retention")
    return rows


def _frozen_v23e_selector_control(
    root: Path, *, source_v23a: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind report interpretation to V23e's already-frozen control result.

    V23a's stored ``best_model.pt`` was selected by the older
    ``parent_min_precision_recall`` rule.  V23j intentionally uses the later
    V23e rule.  Therefore a selected-checkpoint delta against V23a's legacy
    artifact is descriptive, not an isolated ramp effect.  The read-only V23e
    scan supplies the same-selector control result needed to state that fact
    explicitly.
    """

    path = (root / V23E_REPORT_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError("V23j requires the frozen V23e selector report: " + str(path))
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    source = dict(payload.get("source_v23a") or {})
    if (
        payload.get("status") != "completed_development_selector_only"
        or payload.get("schema_version") != V23E_REPORT_SCHEMA
        or payload.get("selector_schema_version") != V23E_SELECTOR_SCHEMA
        or payload.get("scanned_all_frozen_epochs") is not True
        or int(payload.get("feasible_epoch_count", -1)) != 0
        or payload.get("selected_checkpoint") is not None
        or source.get("config_sha256") != source_v23a.get("config_sha256")
        or source.get("config_identity_sha256")
        != source_v23a.get("config_identity_sha256")
        or source.get("training_summary_sha256") != source_v23a.get("summary_sha256")
        or int(source.get("legacy_best_epoch", -1))
        != int(source_v23a.get("legacy_best_epoch", -2))
    ):
        raise ValueError("V23j frozen V23e same-selector control is stale or incompatible")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload["status"],
        "selector_schema_version": payload["selector_schema_version"],
        "scanned_all_frozen_epochs": True,
        "feasible_epoch_count": 0,
        "selected_checkpoint": None,
        "conclusion": payload.get("conclusion"),
    }


def _direct_v23a_comparison(
    root: Path, rows: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Produce diagnostics against V23a without changing V23e selection.

    V23d's weak prefix/window boundary motivated this experiment but does not
    enter the comparison: V23j has one direct paired control, V23a.
    """

    control_rows = _v23a_records(root)
    metrics_by_scope: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, Any]] = {}
    for scope, title in (
        ("parent", "parent_final"),
        ("eligible_prefix", "eligible_prefix_event_pooled"),
        ("window", "causal_prefix_windows"),
    ):
        before, after = _ordered_pair(control_rows[scope], rows[scope], scope=scope)
        control = _scope_metrics(before, f"v23a_{title}")
        treatment = _scope_metrics(after, f"v23j_{title}")
        metrics_by_scope[title] = {
            "v23a_fixed_prefix_weight_0_1": control,
            "v23j_linear_prefix_ramp_treatment": treatment,
        }
        deltas[title] = _delta(control, treatment)
    before_prefix, after_prefix = _ordered_pair(
        control_rows["eligible_prefix"],
        rows["eligible_prefix"],
        scope="eligible_prefix",
    )
    control_balanced = summarize_parent_balanced_eligible_prefix_records(
        before_prefix,
        scope="v23a_parent_balanced_eligible_prefix",
    )
    treatment_balanced = summarize_parent_balanced_eligible_prefix_records(
        after_prefix,
        scope="v23j_parent_balanced_eligible_prefix",
    )
    metrics_by_scope["parent_balanced_eligible_prefix"] = {
        "v23a_fixed_prefix_weight_0_1": control_balanced,
        "v23j_linear_prefix_ramp_treatment": treatment_balanced,
    }
    deltas["parent_balanced_eligible_prefix"] = _delta(
        control_balanced, treatment_balanced
    )
    parent_before, parent_after = _ordered_pair(
        control_rows["parent"], rows["parent"], scope="parent"
    )
    return {
        "comparison_kind": "legacy-selected-state descriptive comparison, not an isolated prefix-ramp delta",
        "scope": "V23a legacy fixed-prefix checkpoint versus V23j V23e-selected checkpoint; development KnownGate only",
        "not_pure_ramp_effect_because_checkpoint_rules_differ": True,
        "metrics_by_scope": metrics_by_scope,
        "v23j_minus_v23a_by_scope": deltas,
        "paired_parent_bootstrap": _bootstrap(parent_before, parent_after),
        "causal_prefix_hypothesis": {
            "motivating_observation": "V23a and V23d both exposed weak early Clutter actions despite parent-final evidence; V23d is independent context only, not a V23j comparator.",
            "primary_endpoint": "parent_balanced_eligible_prefix Clutter Recall then F1, subject to frozen V23e parent-final and KnownTarget constraints",
            "parent_final_safety_endpoint": "V23e feasibility: parent-final Clutter 8/9 Recall and Precision >= 0.90 plus frozen KnownTarget and parent-balanced prefix safety",
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _source_config, _source_path = load_config(root)
    config_path = root / config_relative()
    readiness_path = root / readiness_relative()
    artifact = root / artifact_relative()
    summary_path = artifact / "training-summary.json"
    if not all(path.is_file() for path in (config_path, readiness_path, summary_path)):
        raise FileNotFoundError("V23j training output is incomplete")
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    if (
        summary.get("variant") != V23J_VARIANT
        or summary.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or dict(summary.get("selector") or {}) != config.selector
        or dict(summary.get("prefix_ramp") or {}) != config.prefix_ramp.as_dict()
        or dict(summary.get("epoch_one_v23a_pairing_audit") or {}).get(
            "structural_pairing_exactly_matches_frozen_v23a_epoch_one"
        )
        is not True
        or dict(summary.get("training_readiness") or {}).get("sha256")
        != sha256_file(readiness_path)
        or readiness.get("status") != "passed"
        or readiness.get("variant") != V23J_VARIANT
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V23j training summary violates the prepared contract")
    reference = frozen_v23a_legacy_reference(root)
    v23e_control = _frozen_v23e_selector_control(
        root, source_v23a=config.source_v23a
    )
    base = {
        "status": "completed_development_prefix_ramp_only",
        "schema_version": V23J_REPORT_SCHEMA,
        "experiment": "V23j V23a official-CfC epoch-linear prefix auxiliary weight ramp 0.1 to 1.0 only",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "source_v23a": config.source_v23a,
        "selector": config.selector,
        "prefix_ramp": config.prefix_ramp.as_dict(),
        "legacy_reference": reference,
        "same_selector_v23a_control": v23e_control,
        "comparison_interpretation": {
            "training_variable": "epoch-level causal-prefix auxiliary weight only: 0.1 at epoch 1 linearly to 1.0 at epoch 20",
            "v23a_training_control": "fixed 0.1 prefix weight with the same official CfC fixed-retention path",
            "v23j_checkpoint_rule": "V23e parent-final plus parent-balanced eligible-prefix selector",
            "v23a_legacy_best_checkpoint_rule": "parent_min_precision_recall",
            "same_selector_v23a_result": "V23e found zero feasible V23a epochs",
            "selected_checkpoint_delta_vs_v23a_legacy_is_not_pure_ramp_effect": True,
            "primary_admissibility_comparison": "V23j feasible-epoch count under V23e versus the frozen V23a V23e feasible-epoch count of zero",
        },
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    if summary.get("status") == "completed_no_feasible_checkpoint":
        report = {
            **base,
            "selection_status": "no_feasible_checkpoint",
            "best_epoch": None,
            "best_epoch_prefix_auxiliary_weight": None,
            "metrics": None,
            "v23j_vs_v23a_direct_control": None,
            "clutter_parent_bootstrap": None,
            "clutter_prefix_action_trajectories": None,
            "conclusion": "V23j completed all 20 epochs but no checkpoint met the frozen V23e feasibility constraints; the prefix ramp did not create an admissible development checkpoint.",
            "not_final_product_performance": True,
        }
    elif summary.get("status") == "completed":
        checkpoint_path = artifact / "best_model.pt"
        detail_path = artifact / "sample_detail_information.json"
        if not all(path.is_file() for path in (checkpoint_path, detail_path)):
            raise FileNotFoundError("V23j selected checkpoint/detail is missing")
        detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
        best_epoch = int(summary.get("best_epoch", -1))
        best_weight = prefix_auxiliary_weight_for_epoch(best_epoch, config.prefix_ramp)
        if (
            int(detail.get("epoch", -1)) != best_epoch
            or Path(str(detail.get("checkpoint", ""))).resolve()
            != checkpoint_path.resolve()
            or detail.get("variant") != V23J_VARIANT
            or dict(detail.get("prefix_ramp") or {}) != config.prefix_ramp.as_dict()
            or float(detail.get("prefix_auxiliary_weight_for_epoch", -1.0))
            != best_weight
        ):
            raise ValueError("V23j selected detail binding is stale")
        rows = _records(detail)
        metrics = _metrics(rows)
        selector = dict(summary.get("best_checkpoint_selection") or {})
        if (
            selector.get("schema_version") != V23E_SELECTOR_SCHEMA
            or selector.get("feasible") is not True
        ):
            raise ValueError("V23j best checkpoint is not feasible under V23e selector")
        report = {
            **base,
            "selection_status": "feasible_checkpoint_selected",
            "best_epoch": best_epoch,
            "best_epoch_prefix_auxiliary_weight": best_weight,
            "best_checkpoint_path": str(checkpoint_path),
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics": metrics,
            "v23j_vs_v23a_direct_control": _direct_v23a_comparison(root, rows),
            "selector_assessment": selector,
            "clutter_parent_bootstrap": _selection_bootstrap(rows),
            "clutter_prefix_action_trajectories": clutter_prefix_action_trajectories(rows),
            "conclusion": "V23j found a V23e-feasible checkpoint. This is development-only local KnownGate evidence and remains ineligible for OOF/test/product claims.",
            "not_final_product_performance": True,
        }
    else:
        raise ValueError(f"V23j training did not complete: {summary.get('status')}")
    output = root / report_relative()
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report completed V23j prefix-ramp training.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
