from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "RADAR-LN3-V1"
SOURCE_VERSION = "RADAR-LN3-V0"
ANOMALY_PARENT_IDS = (
    "cq08|track|634",
    "cq08|track|785",
    "cq08|track|3338",
)
LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
MAIN_INDEX_NAME = f"{VERSION}-cq08-tr-rd-main-index.csv"
AUDIT_INDEX_NAME = f"{VERSION}-cq08-tr-rd-anomaly-audit-index.csv"
MAIN_SPLIT_NAME = f"{VERSION}-cq08-tr-rd-main-split.json"
AUDIT_SPLIT_NAME = f"{VERSION}-cq08-tr-rd-anomaly-audit-split.json"
MANIFEST_NAME = f"{VERSION}-cq08-tr-rd-view-manifest.json"
CONTRACT_NAME = f"{VERSION}-cq08-tr-rd-data-contract.json"
SUMMARY_NAME = f"{VERSION}-cq08-tr-rd-view-summary.json"
SOURCE_INDEX_NAME = f"{SOURCE_VERSION}-cq08-tr-rd-point-index.csv"
SOURCE_CONTRACT_NAME = f"{SOURCE_VERSION}-cq08-tr-rd-data-contract.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _load_source(project_root: Path) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    processed = project_root / "data" / "processed"
    index_path = processed / SOURCE_INDEX_NAME
    contract_path = processed / SOURCE_CONTRACT_NAME
    if not index_path.exists() or not contract_path.exists():
        raise FileNotFoundError("RADAR-LN3-V0 source index/contract is required before V1 generation")
    frame = pd.read_csv(index_path, low_memory=False)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "sample_id", "parent_sample_id", "track_id", "source_name", "split_name",
        "label_name", "label_id", "rd_frame_count", "rd_parent_order_status", "tr_time_order_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V0 index is missing columns needed for V1: {missing}")
    if frame["schema_version"].astype(str).ne(SOURCE_VERSION).any():
        raise ValueError("V1 source index contains a non-V0 row")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V1 source index is not CQ08-pure")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V0 source index contains duplicate sample IDs")
    if set(frame["label_name"].astype(str)) != set(LABELS):
        raise ValueError("V0 source index does not contain the five CQ08 labels")
    return index_path, frame, contract


def _parent_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for parent_id, group in frame.groupby("parent_sample_id", sort=True):
        labels = set(group["label_name"].astype(str))
        split_column = "original_split_name" if "original_split_name" in group.columns else "split_name"
        splits = set(group[split_column].astype(str))
        if len(labels) != 1 or len(splits) != 1:
            raise ValueError(f"V0 parent has inconsistent label/split: {parent_id}")
        records[str(parent_id)] = {
            "parent_sample_id": str(parent_id),
            "track_id": str(group["track_id"].iloc[0]),
            "label_name": next(iter(labels)),
            "original_split_name": next(iter(splits)),
            "point_count": int(len(group)),
            "rd_frame_count": int(pd.to_numeric(group["rd_frame_count"], errors="raise").sum()),
            "has_tr_time_anomaly": bool(group["tr_time_order_status"].astype(str).eq("non_monotonic").any()),
            "has_rd_order_anomaly": bool(group["rd_parent_order_status"].astype(str).eq("non_monotonic").any()),
        }
    return records


def _make_split_payload(
    *,
    schema_version: str,
    group_ids: dict[str, list[str]],
    parent_count: int,
    point_count: int,
    source_index_hash: str,
    audit_parent_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "source_name": "CQ08",
        "group_unit": "track_id",
        "frame_level_split_forbidden": True,
        "parent_level_split_forbidden": True,
        "source_index": f"data/processed/{SOURCE_INDEX_NAME}",
        "source_index_sha256": source_index_hash,
        "parent_count": parent_count,
        "point_count": point_count,
        "group_ids": group_ids,
        "audit_parent_ids": audit_parent_ids,
        "policy": "V1 main excludes explicitly listed anomalous parents; audit view contains only those parents and is never a training split.",
    }


def build(project_root: Path) -> dict[str, Any]:
    processed = project_root / "data" / "processed"
    config_root = project_root / "config" / "RADAR-LN3"
    source_path, source_frame, source_contract = _load_source(project_root)
    source_hash = _sha256(source_path)
    parent_records = _parent_summary(source_frame)
    all_parent_ids = set(parent_records)
    anomaly_ids = set(ANOMALY_PARENT_IDS)
    if anomaly_ids - all_parent_ids:
        raise ValueError(f"configured anomaly parent is missing from V0: {sorted(anomaly_ids - all_parent_ids)}")
    detected_anomaly_ids = {
        parent_id
        for parent_id, record in parent_records.items()
        if record["has_tr_time_anomaly"] or record["has_rd_order_anomaly"]
    }
    if detected_anomaly_ids != anomaly_ids:
        raise ValueError(f"V0 anomaly set changed: detected={sorted(detected_anomaly_ids)}, configured={sorted(anomaly_ids)}")

    main_frame = source_frame[~source_frame["parent_sample_id"].astype(str).isin(anomaly_ids)].copy()
    audit_frame = source_frame[source_frame["parent_sample_id"].astype(str).isin(anomaly_ids)].copy()
    main_frame["schema_version"] = VERSION
    audit_frame["schema_version"] = VERSION
    main_frame["view_name"] = "main"
    audit_frame["view_name"] = "anomaly_audit"
    main_frame["original_split_name"] = main_frame["split_name"].astype(str)
    audit_frame["original_split_name"] = audit_frame["split_name"].astype(str)
    audit_frame["split_name"] = "audit"
    main_frame["excluded_from_main_reason"] = ""
    audit_frame["excluded_from_main_reason"] = "tr_or_rd_temporal_order_anomaly"

    # Preserve source column order and append only view provenance columns.
    source_columns = list(source_frame.columns)
    view_columns = ["view_name", "original_split_name", "excluded_from_main_reason"]
    output_columns = source_columns + view_columns
    main_frame = main_frame[output_columns].sort_values(["track_id", "causal_event_index"], key=lambda column: column.astype(int) if column.name in {"track_id", "causal_event_index"} else column, kind="stable")
    audit_frame = audit_frame[output_columns].sort_values(["track_id", "causal_event_index"], key=lambda column: column.astype(int) if column.name in {"track_id", "causal_event_index"} else column, kind="stable")

    main_path = processed / MAIN_INDEX_NAME
    audit_path = processed / AUDIT_INDEX_NAME
    main_frame.to_csv(main_path, index=False, encoding="utf-8", lineterminator="\n")
    audit_frame.to_csv(audit_path, index=False, encoding="utf-8", lineterminator="\n")

    main_parent_records = _parent_summary(main_frame)
    audit_parent_records = _parent_summary(audit_frame)
    main_groups: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    for record in main_parent_records.values():
        main_groups[record["original_split_name"]].append(record["track_id"])
    main_groups = {key: sorted(value, key=int) for key, value in main_groups.items()}
    audit_groups = {"audit": sorted([record["track_id"] for record in audit_parent_records.values()], key=int)}
    main_split = _make_split_payload(
        schema_version=VERSION,
        group_ids=main_groups,
        parent_count=len(main_parent_records),
        point_count=len(main_frame),
        source_index_hash=source_hash,
        audit_parent_ids=[],
    )
    audit_split = _make_split_payload(
        schema_version=VERSION,
        group_ids=audit_groups,
        parent_count=len(audit_parent_records),
        point_count=len(audit_frame),
        source_index_hash=source_hash,
        audit_parent_ids=sorted(anomaly_ids),
    )
    main_split_path = processed / MAIN_SPLIT_NAME
    audit_split_path = processed / AUDIT_SPLIT_NAME
    main_split_path.write_text(json.dumps(main_split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_split_path.write_text(json.dumps(audit_split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    labels_main = {str(key): int(value) for key, value in main_frame.groupby("label_name")["parent_sample_id"].nunique().items()}
    labels_audit = {str(key): int(value) for key, value in audit_frame.groupby("label_name")["parent_sample_id"].nunique().items()}
    split_main = {str(key): int(value) for key, value in main_frame.groupby("split_name")["parent_sample_id"].nunique().items()}
    contract_payload = {
        "schema_version": VERSION,
        "contract_name": "cq08_v1_main_and_anomaly_audit_views",
        "source_contract": f"data/processed/{SOURCE_CONTRACT_NAME}",
        "source_index": f"data/processed/{SOURCE_INDEX_NAME}",
        "source_index_sha256": source_hash,
        "views": {
            "main": {
                "index_path": f"data/processed/{MAIN_INDEX_NAME}",
                "split_path": f"data/processed/{MAIN_SPLIT_NAME}",
                "parent_count": len(main_parent_records),
                "point_count": int(len(main_frame)),
                "label_parent_counts": labels_main,
                "split_parent_counts": split_main,
                "excluded_parent_ids": sorted(anomaly_ids),
                "training_allowed": True,
            },
            "anomaly_audit": {
                "index_path": f"data/processed/{AUDIT_INDEX_NAME}",
                "split_path": f"data/processed/{AUDIT_SPLIT_NAME}",
                "parent_count": len(audit_parent_records),
                "point_count": int(len(audit_frame)),
                "label_parent_counts": labels_audit,
                "original_split_counts": dict(Counter(record["original_split_name"] for record in audit_parent_records.values())),
                "audit_parent_ids": sorted(anomaly_ids),
                "training_allowed": False,
            },
        },
        "model_io_inherited_from_v0": source_contract.get("model_io", {}),
        "rd_tensor_policy": "reuse existing V0-referenced CQ08 canonical tensors read-only; no tensor copy or overwrite",
        "anomaly_policy": {
            "mode": "exclude_from_main_and_audit_separately",
            "parent_ids": sorted(anomaly_ids),
            "reason": "TR negative/non-monotonic wall-clock or RD point index order anomaly",
            "main_metrics_denominator": "V1 main view only",
            "audit_metrics_denominator": "V1 anomaly_audit view only",
        },
        "version_reason": "Substantive split/view/config semantics changed after user-selected anomaly exclusion; next global version after RADAR-LN3-V0.",
    }
    contract_path = processed / CONTRACT_NAME
    contract_path.write_text(json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_payload = {
        "schema_version": VERSION,
        "manifest_type": "cq08_v1_main_and_anomaly_audit_views",
        "source_name": "CQ08",
        "source_v0_index": f"data/processed/{SOURCE_INDEX_NAME}",
        "source_v0_index_sha256": source_hash,
        "main_index": f"data/processed/{MAIN_INDEX_NAME}",
        "anomaly_audit_index": f"data/processed/{AUDIT_INDEX_NAME}",
        "main_split": f"data/processed/{MAIN_SPLIT_NAME}",
        "anomaly_audit_split": f"data/processed/{AUDIT_SPLIT_NAME}",
        "contract": f"data/processed/{CONTRACT_NAME}",
        "main_parent_count": len(main_parent_records),
        "main_point_count": int(len(main_frame)),
        "main_rd_frame_count": int(pd.to_numeric(main_frame["rd_frame_count"], errors="raise").sum()),
        "audit_parent_count": len(audit_parent_records),
        "audit_point_count": int(len(audit_frame)),
        "audit_rd_frame_count": int(pd.to_numeric(audit_frame["rd_frame_count"], errors="raise").sum()),
        "anomaly_parent_ids": sorted(anomaly_ids),
        "source_tensor_policy": "read_only_reuse",
        "v0_files_unchanged": True,
    }
    manifest_path = processed / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "schema_version": VERSION,
        "status": "generated",
        "source_v0_index_sha256": source_hash,
        "anomaly_parent_ids": sorted(anomaly_ids),
        "anomaly_parent_details": {key: parent_records[key] for key in sorted(anomaly_ids)},
        "main": {
            "parent_count": len(main_parent_records),
            "point_count": int(len(main_frame)),
            "rd_frame_count": int(pd.to_numeric(main_frame["rd_frame_count"], errors="raise").sum()),
            "label_parent_counts": labels_main,
            "split_parent_counts": split_main,
            "index_sha256": _sha256(main_path),
            "split_sha256": _sha256(main_split_path),
        },
        "anomaly_audit": {
            "parent_count": len(audit_parent_records),
            "point_count": int(len(audit_frame)),
            "rd_frame_count": int(pd.to_numeric(audit_frame["rd_frame_count"], errors="raise").sum()),
            "label_parent_counts": labels_audit,
            "original_split_counts": dict(Counter(record["original_split_name"] for record in audit_parent_records.values())),
            "index_sha256": _sha256(audit_path),
            "split_sha256": _sha256(audit_split_path),
        },
        "outputs": {
            "main_index": f"data/processed/{MAIN_INDEX_NAME}",
            "audit_index": f"data/processed/{AUDIT_INDEX_NAME}",
            "main_split": f"data/processed/{MAIN_SPLIT_NAME}",
            "audit_split": f"data/processed/{AUDIT_SPLIT_NAME}",
            "manifest": f"data/processed/{MANIFEST_NAME}",
            "contract": f"data/processed/{CONTRACT_NAME}",
        },
    }
    summary_path = processed / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RADAR-LN3-V1 CQ08 main and anomaly-audit views.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
