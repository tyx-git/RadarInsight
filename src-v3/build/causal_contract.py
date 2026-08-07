from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "RADAR-LN3-V3"
V1_INDEX = "data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv"
OUTPUT_BASE = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt"
INDEX_NAME = f"{OUTPUT_BASE}-index.csv"
MANIFEST_NAME = f"{OUTPUT_BASE}-manifest.json"
SPLIT_NAME = f"{OUTPUT_BASE}-split.json"
CONTRACT_NAME = f"{OUTPUT_BASE}-data-contract.json"
SUMMARY_NAME = f"{OUTPUT_BASE}-summary.json"
STATS_PATH = "data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    processed = project_root / "data" / "processed"
    source_path = project_root / V1_INDEX
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = pd.read_csv(source_path, low_memory=False)
    required = {
        "schema_version", "sample_id", "parent_sample_id", "track_id", "source_name",
        "split_name", "label_name", "label_id", "causal_event_index", "tr_time_seconds",
        "tr_dt_seconds", "tr_dt_is_negative", "tr_feature_values_json", "rd_present",
        "rd_frame_count", "rd_tensor_path", "rd_tensor_frame_indices_json",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"V1 source index is missing columns: {missing}")
    if source["schema_version"].astype(str).ne("RADAR-LN3-V1").any():
        raise ValueError("V3 source must contain only RADAR-LN3-V1 rows")
    if source["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V3 source is not CQ08-pure")
    if source["sample_id"].astype(str).duplicated().any():
        raise ValueError("V3 source contains duplicate sample_id")
    if source.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("V3 source contains a parent with multiple labels")
    if source.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("V3 source contains a parent crossing splits")

    frame = source.copy()
    frame["schema_version"] = VERSION
    frame["tr_time_precision_seconds"] = 1.0
    frame["tr_dt_observation_unit"] = "track_relative_wall_clock_seconds_coarse"
    frame["tr_dt_valid"] = False
    frame["tr_dt_reason"] = ""
    for parent_id, positions in frame.groupby("parent_sample_id", sort=False).groups.items():
        group = frame.loc[positions].sort_values("causal_event_index", kind="stable")
        for row_index, row in group.iterrows():
            event_index = int(row["causal_event_index"])
            raw_dt = float(row["tr_dt_seconds"])
            if event_index == 0:
                valid = False
                reason = "first_event_no_predecessor"
            elif not np.isfinite(raw_dt):
                valid = False
                reason = "nonfinite_timestamp_delta"
            elif raw_dt < 0.0:
                valid = False
                reason = "negative_time_order"
            elif abs(raw_dt) <= 1.0e-9:
                valid = False
                reason = "zero_delta_at_second_precision"
            else:
                valid = True
                reason = "positive_delta_at_second_precision"
            frame.at[row_index, "tr_dt_valid"] = bool(valid)
            frame.at[row_index, "tr_dt_reason"] = reason

    frame = frame.sort_values(
        ["track_id", "causal_event_index"],
        key=lambda column: column.astype(int) if column.name in {"track_id", "causal_event_index"} else column,
        kind="stable",
    ).reset_index(drop=True)
    output_index = project_root / INDEX_NAME
    output_index.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_index, index=False, encoding="utf-8", lineterminator="\n")
    index_hash = sha256_file(output_index)
    source_hash = sha256_file(source_path)
    stats_path = project_root / STATS_PATH
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    stats_hash = sha256_file(stats_path)

    group_ids = {
        split: sorted(frame.loc[frame["split_name"].astype(str).eq(split), "track_id"].astype(str).unique(), key=int)
        for split in ("train", "val", "test")
    }
    split_payload = {
        "schema_version": VERSION,
        "source_name": "CQ08",
        "group_unit": "track_id",
        "frame_level_split_forbidden": True,
        "parent_level_split_forbidden": True,
        "index_path": INDEX_NAME,
        "index_sha256": index_hash,
        "source_v1_index": V1_INDEX,
        "source_v1_index_sha256": source_hash,
        "group_ids": group_ids,
        "parent_counts": {
            split: int(frame.loc[frame["split_name"].astype(str).eq(split), "parent_sample_id"].nunique())
            for split in ("train", "val", "test")
        },
        "point_counts": {
            split: int(frame["split_name"].astype(str).eq(split).sum())
            for split in ("train", "val", "test")
        },
        "policy": "Reuse V1 main parent/track split exactly; only time-validity fields and schema version change.",
    }
    output_split = project_root / SPLIT_NAME
    output_split.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_hash = sha256_file(output_split)

    dt_valid = frame["tr_dt_valid"].astype(bool)
    dt_reason_counts = {str(key): int(value) for key, value in frame["tr_dt_reason"].value_counts().sort_index().items()}
    contract_payload = {
        "schema_version": VERSION,
        "contract_name": "cq08_point_level_tr_rd_causal_prefix_dt_v1",
        "source_contract": "data/processed/RADAR-LN3-V2-cq08-tr-rd-formal-data-contract.json",
        "source_index": INDEX_NAME,
        "source_index_sha256": index_hash,
        "source_v1_index": V1_INDEX,
        "source_v1_index_sha256": source_hash,
        "source_purity": {"allowed_sources": ["CQ08"], "excluded_sources": ["raw", "Excel", "RD-only"]},
        "data_scope": {
            "index_path": INDEX_NAME,
            "parent_count": int(frame["parent_sample_id"].nunique()),
            "point_count": int(len(frame)),
            "split_parent_counts": split_payload["parent_counts"],
            "split_point_counts": split_payload["point_counts"],
            "label_parent_counts": {
                str(key): int(value)
                for key, value in frame.drop_duplicates("parent_sample_id").groupby("label_name").size().sort_index().items()
            },
        },
        "time_contract": {
            "raw_field": "tr_time_seconds",
            "raw_semantics": "track_relative_wall_clock_seconds_coarse",
            "raw_precision_seconds": 1.0,
            "dt_field": "tr_dt_seconds",
            "dt_valid_field": "tr_dt_valid",
            "dt_reason_field": "tr_dt_reason",
            "valid_definition": "event_index>0 and finite positive delta at source second precision",
            "invalid_reasons": ["first_event_no_predecessor", "zero_delta_at_second_precision", "negative_time_order", "nonfinite_timestamp_delta"],
            "zero_dt_policy": "retain_raw_zero; expose tr_dt_valid=false; numerical liquid floor is stability-only and is not a recovered physical interval",
            "rd_time_is_not_tr_wall_clock": True,
        },
        "feature_contract": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "rd_frame_shape": [2, 21, 384],
            "rd_normalization_stats": STATS_PATH,
            "rd_normalization_stats_sha256": stats_hash,
            "missing_policy": "explicit RD frame mask and quality fields; TR missing values retain missing mask",
        },
        "model_io": {
            "tr_point_features": "[B,L,14]",
            "tr_missing_mask": "[B,L,11]",
            "tr_dt_seconds": "[B,L]",
            "tr_dt_valid": "[B,L] bool",
            "rd_frames": "[B,L,K,2,21,384]",
            "rd_frame_mask": "[B,L,K]",
            "rd_quality": "[B,L,7]",
            "tr_padding_mask": "[B,L]",
            "sequence_logits": "[B,L,5]",
            "window_logits": "[B,5]",
            "stable_state": "external causal state machine",
        },
        "training_contract": {
            "task": "five_class_parent_label_with_causal_prefix_supervision",
            "initialization": "random_model_initialization",
            "pretraining": "not_required_and_not_used",
            "loss_mode": "causal_prefix",
            "loss_reduction": "mean_valid_prefix_loss_per_parent_then_mean_parent_loss",
            "checkpoint_selection": "validation parent_macro_f1",
            "validation_test_separation": True,
        },
        "leakage_policy": [
            "no label/path/session field enters tensors",
            "parent/track split grouping is inherited from V1 main view",
            "no future TR points are used by the causal model",
            "RD CPI elapsed time is not used as TR wall clock",
            "normalization statistics reuse the train-only V2 statistics without content change",
        ],
        "v2_reuse": {
            "v2_index": "data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv",
            "v2_formal_contract": "data/processed/RADAR-LN3-V2-cq08-tr-rd-formal-data-contract.json",
            "train_normalization_stats": STATS_PATH,
            "train_normalization_stats_sha256": stats_hash,
            "reuse_reason": "No RD value, range, amplitude normalization, parent range, label mapping or split change; only dt validity and training loss semantics changed.",
        },
        "version_reason": "Added explicit coarse-time dt validity/reason fields and causal-prefix training semantics; V1 parent/split/data range and V2 RD normalization are reused.",
    }
    output_contract = project_root / CONTRACT_NAME
    output_contract.write_text(json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract_hash = sha256_file(output_contract)

    manifest_payload = {
        "schema_version": VERSION,
        "manifest_type": "cq08_point_level_tr_rd_prefix_dt_contract",
        "source_name": "CQ08",
        "index_path": INDEX_NAME,
        "split_path": SPLIT_NAME,
        "contract_path": CONTRACT_NAME,
        "index_sha256": index_hash,
        "split_sha256": split_hash,
        "contract_sha256": contract_hash,
        "stats_path": STATS_PATH,
        "stats_sha256": stats_hash,
        "parent_count": int(frame["parent_sample_id"].nunique()),
        "point_count": int(len(frame)),
        "label_parent_counts": contract_payload["data_scope"]["label_parent_counts"],
        "split_parent_counts": split_payload["parent_counts"],
        "dt_valid_count": int(dt_valid.sum()),
        "dt_invalid_count": int((~dt_valid).sum()),
        "dt_reason_counts": dt_reason_counts,
        "old_files_reused_read_only": [V1_INDEX, STATS_PATH, "data/processed/cq08_rd_physical_canonical_v3_1/*.npz"],
        "version_reason": contract_payload["version_reason"],
    }
    output_manifest = project_root / MANIFEST_NAME
    output_manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_payload = {
        "schema_version": VERSION,
        "status": "generated",
        "outputs": {
            "index": INDEX_NAME,
            "manifest": MANIFEST_NAME,
            "split": SPLIT_NAME,
            "contract": CONTRACT_NAME,
        },
        "hashes": {
            "source_v1_index_sha256": source_hash,
            "index_sha256": index_hash,
            "split_sha256": split_hash,
            "contract_sha256": contract_hash,
            "stats_sha256": stats_hash,
        },
        "counts": {
            "parent_count": int(frame["parent_sample_id"].nunique()),
            "point_count": int(len(frame)),
            "dt_valid_count": int(dt_valid.sum()),
            "dt_invalid_count": int((~dt_valid).sum()),
            "dt_reason_counts": dt_reason_counts,
            "label_parent_counts": contract_payload["data_scope"]["label_parent_counts"],
            "split_parent_counts": split_payload["parent_counts"],
        },
        "version_reason": contract_payload["version_reason"],
    }
    output_summary = project_root / SUMMARY_NAME
    output_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated RADAR-LN3-V3 CQ08 prefix/dt contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
