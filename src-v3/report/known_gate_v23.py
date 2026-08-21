"""Report the completed V23 pure official-CfC baseline on development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES
from radar_ln3.known_gate_v23 import (
    ARTIFACT_RELATIVE,
    CONFIG_RELATIVE,
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FIXED_EPOCHS,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    REPORT_RELATIVE,
    REPORT_SCHEMA,
    config_identity_sha256,
    load_v23_config,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _class_row(metrics: dict[str, Any], class_name: str) -> dict[str, Any]:
    rows = list(metrics.get("per_class") or [])
    matches = [dict(row) for row in rows if row.get("class") == class_name]
    if len(matches) != 1:
        raise ValueError(f"V23 validation metrics lack exactly one {class_name} row")
    return matches[0]


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    readiness_path = root / READINESS_RELATIVE
    artifact = root / ARTIFACT_RELATIVE
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    for path in (config_path, contract_path, readiness_path, summary_path, detail_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config, source_config, _source_path = load_v23_config(root, config_path)
    contract = _read_json(contract_path)
    readiness = _read_json(readiness_path)
    summary = _read_json(summary_path)
    detail = _read_json(detail_path)
    parent = dict(summary.get("best_validation_metrics", {}).get("parent_metrics") or {})
    eligible = dict(summary.get("best_validation_metrics", {}).get("eligible_prefix_metrics") or {})
    window = dict(summary.get("best_validation_metrics", {}).get("window_metrics") or {})
    target = _class_row(parent, KNOWN_GATE_CLASS_NAMES[0])
    clutter = _class_row(parent, KNOWN_GATE_CLASS_NAMES[1])
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("oof") != "未执行（先完成V23基座与多seed）"
        or summary.get("status") != "completed"
        or summary.get("variant") != "official_cfc_baseline"
        or summary.get("architecture") != "official_ncps_cfc"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or int(summary.get("best_epoch", -1)) not in range(1, FIXED_EPOCHS + 1)
        or len(list(summary.get("history") or [])) != FIXED_EPOCHS
        or list(row.get("epoch") for row in summary["history"]) != list(range(1, FIXED_EPOCHS + 1))
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("OOF") != "未执行（延后至基座和多seed确认后）"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or detail.get("variant") != "official_cfc_baseline"
        or detail.get("scope") != "development validation KnownGate only; not OOF or test"
        or int(detail.get("counts", {}).get("parent", -1)) != 232
    ):
        raise ValueError("V23 completed baseline summary/detail violates its frozen contract")
    records = dict(detail.get("records") or {})
    if set(records) != {"window", "eligible_prefix", "parent"}:
        raise ValueError("V23 sample detail scopes are invalid")
    report = {
        "status": "completed_development_baseline_only",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V23 pure official dense CfC KnownGate baseline development validation",
        "scope": "V21 development validation KnownGate only; no OOF, test, strict Unknown calibration, L3, full cascade, stable or P95",
        "source_control": {
            "name": "V21 seed43 formal baseline",
            "architecture": "project_local_cfc_inspired_encoder",
            "config_path": config.source_config_path,
            "config_sha256": config.source_config_sha256,
        },
        "v23": {
            "architecture": "official_ncps_cfc",
            "runtime": summary["official_cfc"],
            "best_checkpoint": summary["best_checkpoint"],
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": summary["best_epoch"],
            "trainable_parameter_count": summary["trainable_parameter_count"],
            "initialization_audit": summary["initialization_audit"],
        },
        "development_metrics": {
            "window": window,
            "eligible_prefix": eligible,
            "parent_final": parent,
            "known_target_parent": target,
            "known_clutter_parent": clutter,
            "clutter_support_warning": "development Clutter support is 9 parents; this report includes no OOF or cross-source conclusion",
        },
        "performance": {
            "status": summary["performance_status"],
            "guardrails": summary["guardrails"],
            "target": summary["target_min_per_class_precision_recall_f1"],
        },
        "next_stage": {
            "multiseed_required_before_oof": True,
            "oof_status": "未执行（延后至基座和多seed确认后）",
            "comparison_to_v21e_or_v22": "not_claimed; V23 is a new no-evidence formal baseline",
            "formal_rejection": "未执行",
            "full_cascade": "未执行",
            "test": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "deployment_eligible": False,
        },
    }
    output = root / REPORT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report V23 pure official-CfC development baseline.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
