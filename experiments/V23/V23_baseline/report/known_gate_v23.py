"""Report the completed V23 pure official-CfC baseline on development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES
from radar_ln3.known_gate_v23 import (
    ARTIFACT_RELATIVE,
    CHECKPOINT_SCHEMA,
    CONFIG_RELATIVE,
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FIXED_EPOCHS,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    REPORT_RELATIVE,
    REPORT_SCHEMA,
    RUNTIME_COMPATIBILITY_REPORT_RELATIVE,
    RUNTIME_COMPATIBILITY_REPORT_SCHEMA,
    RUNTIME_COMPATIBILITY_REQUIRED_CHECKS,
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


def _core_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the report compact while retaining every decision-relevant metric."""

    scalar_fields = (
        "scope",
        "support",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    )
    missing = [field for field in scalar_fields if field not in metrics]
    if missing:
        raise ValueError(f"V23 validation metrics lack fields: {missing}")
    classes = []
    for class_name in KNOWN_GATE_CLASS_NAMES:
        row = _class_row(metrics, class_name)
        classes.append(
            {
                field: row[field]
                for field in (
                    "class",
                    "precision",
                    "recall",
                    "f1",
                    "support",
                    "major_confusion_to",
                    "major_confusion_count",
                )
            }
        )
    matrix = metrics.get("confusion_matrix_rows_true_columns_pred")
    if not isinstance(matrix, list) or len(matrix) != len(KNOWN_GATE_CLASS_NAMES):
        raise ValueError("V23 validation metrics lack the two-class confusion matrix")
    return {
        **{field: metrics[field] for field in scalar_fields},
        "per_class": classes,
        "confusion_matrix_rows_true_columns_pred": matrix,
    }


def _runtime_compatibility(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = (root / RUNTIME_COMPATIBILITY_REPORT_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V23 official-CfC runtime compatibility report is missing: {path}")
    report = _read_json(path)
    checks = dict(report.get("checks") or {})
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != RUNTIME_COMPATIBILITY_REPORT_SCHEMA
        or report.get("dataset_access_count") != 0
        or report.get("checkpoint_access_count") != 0
        or report.get("validation_access_count") != 0
        or report.get("test_access_count") != 0
        or set(checks) != set(RUNTIME_COMPATIBILITY_REQUIRED_CHECKS)
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("V23 official-CfC runtime compatibility report is invalid")
    binding = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": RUNTIME_COMPATIBILITY_REPORT_SCHEMA,
        "checks": checks,
    }
    return path, report, binding


def _runtime_report_summary(report: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    constraints = dict(report.get("core_constraints") or {})
    zero_dt = dict(constraints.get("cfc_zero_dt") or {})
    heterogeneous = dict(constraints.get("cfc_batch_two_heterogeneous_timespans") or {})
    padding = dict(constraints.get("cfc_padding_mask_keyword") or {})
    ltc_zero_dt = dict(constraints.get("ltc_zero_dt") or {})
    return {
        **binding,
        "runtime": dict(report["runtime"]),
        "facts": {
            "cfc_zero_dt_is_finite": zero_dt.get("output_finite") is True and zero_dt.get("state_finite") is True,
            "cfc_zero_dt_is_state_identity": zero_dt.get("state_equal_to_input_state") is True,
            "heterogeneous_batch_two_timespans_supported": heterogeneous.get("accepted") is True,
            "padding_mask_keyword_supported": padding.get("accepted") is True,
            "ltc_zero_dt_is_finite": ltc_zero_dt.get("output_finite") is True and ltc_zero_dt.get("state_finite") is True,
        },
        "adapter_decision": dict(report["adapter_decision"]),
        "performance_boundary": dict(report["performance_boundary"]),
    }


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
    config, _source_config, _source_path = load_v23_config(root, config_path)
    contract = _read_json(contract_path)
    readiness = _read_json(readiness_path)
    _runtime_path, runtime_compatibility, runtime_binding = _runtime_compatibility(root)
    summary = _read_json(summary_path)
    detail = _read_json(detail_path)
    checkpoint = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
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
        or dict(summary.get("official_runtime_compatibility") or {}) != runtime_binding
        or summary.get("OOF") != "未执行（延后至基座和多seed确认后）"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or detail.get("variant") != "official_cfc_baseline"
        or detail.get("scope") != "development validation KnownGate only; not OOF or test"
        or int(detail.get("counts", {}).get("parent", -1)) != 232
        or checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA
        or dict(checkpoint.get("runtime_compatibility") or {}) != {
            key: runtime_binding[key]
            for key in ("path", "sha256", "schema_version")
        }
    ):
        raise ValueError("V23 completed baseline summary/detail violates its frozen contract")
    records = dict(detail.get("records") or {})
    if set(records) != {"window", "eligible_prefix", "parent"}:
        raise ValueError("V23 sample detail scopes are invalid")
    parent_core = _core_metrics(parent)
    eligible_core = _core_metrics(eligible)
    window_core = _core_metrics(window)
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
        "training": {
            "global_step": summary["global_step"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "checkpoint_selection": summary["best_checkpoint_selection"],
            "trainable_parameter_count": summary["trainable_parameter_count"],
        },
        "development_metrics_core": {
            "parent_final": parent_core,
            "eligible_prefix": eligible_core,
            "window": window_core,
            "parent_final_focus": {
                "KnownTarget": target,
                "Clutter": clutter,
            },
            "clutter_support_warning": "development Clutter support is 9 parents; this report includes no OOF or cross-source conclusion",
        },
        "official_runtime_compatibility": _runtime_report_summary(runtime_compatibility, runtime_binding),
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
