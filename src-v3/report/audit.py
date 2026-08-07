from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from radar_ln3.config import load_config, resolve_project_path


LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _split_label_stats(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split_name, label_name), group in frame.groupby(["split_name", "label_name"], sort=True):
        valid = group["tr_dt_valid_b"]
        rows.append(
            {
                "split": str(split_name),
                "label": str(label_name),
                "parent_count": int(group["parent_sample_id"].nunique()),
                "point_count": int(len(group)),
                "valid_dt_count": int(valid.sum()),
                "invalid_dt_count": int((~valid).sum()),
                "zero_dt_count": int(group["tr_dt_zero_b"].sum()),
                "negative_dt_count": int(group["tr_dt_negative_b"].sum()),
                "rd_missing_count": int((~group["rd_present_b"]).sum()),
                "rd_quality_missing_count": int(group["rd_quality_missing_b"].sum()),
            }
        )
    return rows


def audit(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.schema_version != "RADAR-LN3-V3":
        raise ValueError("report.audit requires a RADAR-LN3-V3 configuration")

    index_path = resolve_project_path(project_root, config.data.index_path)
    contract_path = resolve_project_path(project_root, config.data.contract_path)
    stats_path = resolve_project_path(project_root, config.data.normalization_stats_path or "")
    for path in (index_path, contract_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_csv(index_path, low_memory=False)
    required = {
        "sample_id",
        "parent_sample_id",
        "track_id",
        "source_name",
        "split_group_id",
        "split_name",
        "label_name",
        "label_id",
        "causal_event_index",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "rd_present",
        "rd_quality_missing",
        "rd_quality_partial_parent",
        "rd_quality_parent_order_anomaly",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V3 index is missing audit columns: {missing}")

    frame["tr_dt_valid_b"] = _bool_series(frame["tr_dt_valid"])
    frame["rd_present_b"] = _bool_series(frame["rd_present"])
    frame["rd_quality_missing_b"] = _bool_series(frame["rd_quality_missing"])
    frame["tr_dt_zero_b"] = pd.to_numeric(frame["tr_dt_seconds"], errors="raise").abs() <= 1.0e-12
    frame["tr_dt_negative_b"] = pd.to_numeric(frame["tr_dt_seconds"], errors="raise") < 0.0

    parent_lengths = frame.groupby("parent_sample_id", sort=True).size()
    cross_split = {
        field: int(frame.groupby(field)["split_name"].nunique().gt(1).sum())
        for field in ("split_group_id", "track_id", "rd_tensor_path")
        if field in frame.columns
    }
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    labels = sorted(str(value) for value in frame["label_name"].unique())
    audit_result: dict[str, Any] = {
        "status": "passed",
        "scope": "read_only_cq08_v3_index_contract_audit",
        "test_evaluation": "not_executed",
        "checkpoint_read": False,
        "data": {
            "config_path": str(config_path),
            "index_path": str(index_path),
            "contract_path": str(contract_path),
            "normalization_stats_path": str(stats_path),
            "sha256": {
                "config": _sha256(config_path),
                "index": _sha256(index_path),
                "contract": _sha256(contract_path),
                "normalization_stats": _sha256(stats_path),
            },
            "schema_version": config.schema_version,
            "source_values": sorted(str(value) for value in frame["source_name"].unique()),
            "source_purity_passed": set(frame["source_name"].astype(str)) == {"CQ08"},
            "row_count": int(len(frame)),
            "parent_count": int(frame["parent_sample_id"].nunique()),
            "sample_id_duplicate_count": int(frame["sample_id"].astype(str).duplicated().sum()),
            "labels": labels,
            "label_parent_counts": {
                str(key): int(value)
                for key, value in frame.groupby("label_name")["parent_sample_id"].nunique().sort_index().items()
            },
            "split_parent_counts": {
                str(key): int(value)
                for key, value in frame.groupby("split_name")["parent_sample_id"].nunique().sort_index().items()
            },
            "split_point_counts": _counts(frame["split_name"]),
            "parent_label_conflict_count": int(frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).sum()),
            "parent_split_conflict_count": int(frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).sum()),
            "cross_split_group_counts": cross_split,
            "sequence_length": {
                "min": int(parent_lengths.min()),
                "median": float(parent_lengths.median()),
                "max": int(parent_lengths.max()),
                "configured_max": int(config.data.max_sequence_length),
                "parents_over_configured_max": int((parent_lengths > config.data.max_sequence_length).sum()),
            },
        },
        "time": {
            "raw_precision_seconds": contract.get("time_contract", {}).get("raw_precision_seconds"),
            "valid_definition": contract.get("time_contract", {}).get("valid_definition"),
            "row_count": int(len(frame)),
            "valid_dt_count": int(frame["tr_dt_valid_b"].sum()),
            "invalid_dt_count": int((~frame["tr_dt_valid_b"]).sum()),
            "zero_dt_count": int(frame["tr_dt_zero_b"].sum()),
            "negative_dt_count": int(frame["tr_dt_negative_b"].sum()),
            "first_event_count": int(frame["causal_event_index"].astype(int).eq(0).sum()),
            "reason_counts": _counts(frame["tr_dt_reason"]),
            "split_label_stats": _split_label_stats(frame),
            "rd_time_replaces_tr_wall_clock": contract.get("time_contract", {}).get("rd_time_is_not_tr_wall_clock") is False,
        },
        "rd_quality": {
            "rd_present_count": int(frame["rd_present_b"].sum()),
            "rd_missing_count": int((~frame["rd_present_b"]).sum()),
            "rd_quality_missing_count": int(frame["rd_quality_missing_b"].sum()),
            "partial_parent_quality_row_count": int(_bool_series(frame["rd_quality_partial_parent"]).sum()),
            "parent_order_anomaly_row_count": int(_bool_series(frame["rd_quality_parent_order_anomaly"]).sum()),
        },
        "window_contract": {
            "training_loss_mode": config.train.loss_mode,
            "training_supervision_unit": "all_nonpadding_causal_prefix_logits",
            "training_parent_reduction": "mean_valid_prefix_loss_per_parent_then_mean_parent_loss",
            "evaluation_window_unit": "all_nonpadding_sequence_logits",
            "evaluation_parent_unit": "final_causal_prefix_per_parent",
            "stable_unit": "external_causal_target_state_machine",
            "prefix_label_scope": "parent_label_repeated_for_each_prefix",
            "prefix_label_business_validation": "pending",
            "padding_excluded_from_loss": True,
            "future_event_access": False,
        },
        "rejection": {
            "unknown_target_is_ordinary_class": True,
            "interface_available": True,
            "reject_or_abstain_output": False,
            "decision_state_when_uncalibrated": "not_calibrated",
            "supports_validated_rejection": False,
            "required_calibration_source": "train_parent_grouped_oof",
            "threshold_calibration": "not_executed",
            "coverage": "not_evaluated",
            "accepted_macro_f1": "not_evaluated",
            "unknown_clutter_reject_recall": "not_evaluated",
            "false_acceptance_rate": "not_evaluated",
            "false_rejection_rate": "not_evaluated",
            "selective_risk": "not_evaluated",
        },
        "contract_assertions": {
            "labels_exactly_five": tuple(labels) == tuple(sorted(LABELS)),
            "contract_loss_mode_causal_prefix": contract.get("training_contract", {}).get("loss_mode") == "causal_prefix",
            "contract_dt_valid_field": contract.get("time_contract", {}).get("dt_valid_field") == "tr_dt_valid",
            "normalization_is_train_only": stats.get("source_split") == "train" and "no validation or test" in str(stats.get("scope", "")),
        },
    }
    assertions = audit_result["contract_assertions"]
    if not all(bool(value) for value in assertions.values()):
        raise ValueError(f"V3 audit contract assertions failed: {assertions}")
    if not audit_result["data"]["source_purity_passed"]:
        raise ValueError("V3 audit found a non-CQ08 source")
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a read-only CQ08 RADAR-LN3-V3 data and contract audit.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    result = audit(project_root, config_path)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        result["output_path"] = str(output_path.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
