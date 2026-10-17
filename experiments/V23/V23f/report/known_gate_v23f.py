"""Development-only report for the completed V23f prefix-weight experiment."""

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
from radar_ln3.known_gate_v23f import (
    V23F_REPORT_SCHEMA,
    V23F_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_config,
    readiness_relative,
    report_relative,
)
from report.known_gate_v23e_selection import _selection_bootstrap


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
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    if len(rows["parent"]) != 232 or len(rows["eligible_prefix"]) != 3756 or len(rows["window"]) != 4452:
        raise ValueError("V23f detail record counts differ from frozen V21 scope")
    return rows


def _metrics(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "parent_metrics": summarize_known_gate_records(rows["parent"], scope="parent_final"),
        "eligible_prefix_metrics": summarize_known_gate_records(rows["eligible_prefix"], scope="eligible_causal_prefix_auxiliary_excluding_final"),
        "parent_balanced_eligible_prefix_metrics": summarize_parent_balanced_eligible_prefix_records(rows["eligible_prefix"], scope="parent_balanced_eligible_causal_prefix_auxiliary_excluding_final"),
        "window_metrics": summarize_known_gate_records(rows["window"], scope="causal_prefix_windows"),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _source_config, _source_path = load_config(root)
    config_path = root / config_relative()
    readiness_path = root / readiness_relative()
    artifact = root / artifact_relative()
    summary_path = artifact / "training-summary.json"
    if not all(path.is_file() for path in (config_path, readiness_path, summary_path)):
        raise FileNotFoundError("V23f training output is incomplete")
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    if (
        summary.get("variant") != V23F_VARIANT
        or summary.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or dict(summary.get("selector") or {}) != config.selector
        or dict(summary.get("training_readiness") or {}).get("sha256") != sha256_file(readiness_path)
    ):
        raise ValueError("V23f training summary violates the prepared contract")
    reference = frozen_v23a_legacy_reference(root)
    base = {
        "status": "completed_development_prefix_weight_only",
        "schema_version": V23F_REPORT_SCHEMA,
        "experiment": "V23f V23a official-CfC prefix auxiliary weight 1.0 only",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "source_v23a": config.source_v23a,
        "selector": config.selector,
        "legacy_reference": reference,
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "prefix_auxiliary_weight": config.train.prefix_auxiliary_weight,
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
            "metrics": None,
            "clutter_parent_bootstrap": None,
            "clutter_prefix_action_trajectories": None,
            "conclusion": "V23f completed all 20 epochs but no checkpoint met the frozen V23e feasibility constraints; prefix-weight increase did not create an admissible development checkpoint.",
            "not_final_product_performance": True,
        }
    elif summary.get("status") == "completed":
        checkpoint_path = artifact / "best_model.pt"
        detail_path = artifact / "sample_detail_information.json"
        if not all(path.is_file() for path in (checkpoint_path, detail_path)):
            raise FileNotFoundError("V23f selected checkpoint/detail is missing")
        detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
        if (
            int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
            or Path(str(detail.get("checkpoint", ""))).resolve() != checkpoint_path.resolve()
            or detail.get("variant") != V23F_VARIANT
        ):
            raise ValueError("V23f selected detail binding is stale")
        rows = _records(detail)
        metrics = _metrics(rows)
        selector = dict(summary.get("best_checkpoint_selection") or {})
        if selector.get("schema_version") != V23E_SELECTOR_SCHEMA or selector.get("feasible") is not True:
            raise ValueError("V23f best checkpoint is not feasible under V23e selector")
        report = {
            **base,
            "selection_status": "feasible_checkpoint_selected",
            "best_epoch": int(summary["best_epoch"]),
            "best_checkpoint_path": str(checkpoint_path),
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics": metrics,
            "selector_assessment": selector,
            "clutter_parent_bootstrap": _selection_bootstrap(rows),
            "clutter_prefix_action_trajectories": clutter_prefix_action_trajectories(rows),
            "conclusion": "V23f found a V23e-feasible checkpoint. This is development-only local KnownGate evidence and remains ineligible for OOF/test/product claims.",
            "not_final_product_performance": True,
        }
    else:
        raise ValueError(f"V23f training did not complete: {summary.get('status')}")
    output = root / report_relative()
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report completed V23f prefix-weight-only training.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
