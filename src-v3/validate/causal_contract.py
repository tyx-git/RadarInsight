from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "RADAR-LN3-V3"
BASE = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    processed = project_root / "data" / "processed"
    paths = {
        "index": processed / f"{BASE.split('/')[-1]}-index.csv",
        "manifest": processed / f"{BASE.split('/')[-1]}-manifest.json",
        "split": processed / f"{BASE.split('/')[-1]}-split.json",
        "contract": processed / f"{BASE.split('/')[-1]}-data-contract.json",
        "summary": processed / f"{BASE.split('/')[-1]}-summary.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_csv(paths["index"], low_memory=False)
    source_path = processed / "RADAR-LN3-V1-cq08-tr-rd-main-index.csv"
    source = pd.read_csv(source_path, low_memory=False)
    if len(frame) != 29663:
        raise ValueError(f"unexpected V3 point count: {len(frame)}")
    if frame["schema_version"].astype(str).ne(VERSION).any():
        raise ValueError("V3 index contains a wrong schema version")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V3 index is not CQ08-pure")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V3 index contains duplicate sample IDs")
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("V3 parent appears in multiple splits")
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("V3 parent has multiple labels")
    if set(frame["label_name"].astype(str)) != {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}:
        raise ValueError("V3 labels are not exactly the five CQ08 labels")
    if len(source) != len(frame):
        raise ValueError("V3 point count differs from the V1 source view")
    if set(source["sample_id"].astype(str)) != set(frame["sample_id"].astype(str)):
        raise ValueError("V3 sample IDs differ from the V1 source view")
    source_by_id = source.set_index(source["sample_id"].astype(str))
    frame_by_id = frame.set_index(frame["sample_id"].astype(str))
    for column in ("parent_sample_id", "track_id", "split_name", "label_name", "label_id", "tr_time_seconds", "tr_dt_seconds"):
        if not source_by_id[column].astype(str).equals(frame_by_id[column].astype(str)):
            raise ValueError(f"V3 source field changed unexpectedly: {column}")
    valid = frame["tr_dt_valid"].astype(str).str.lower().isin({"true", "1"})
    event_index = pd.to_numeric(frame["causal_event_index"], errors="raise")
    dt = pd.to_numeric(frame["tr_dt_seconds"], errors="raise")
    source_event_index = pd.to_numeric(source["causal_event_index"], errors="raise")
    source_dt = pd.to_numeric(source["tr_dt_seconds"], errors="raise")
    expected_valid_count = int(((source_event_index > 0) & (source_dt > 0) & source_dt.notna()).sum())
    if int(valid.sum()) != expected_valid_count:
        raise ValueError("V3 valid dt count does not match the V1 positive non-first source deltas")
    if valid.any() and ((event_index[valid] <= 0) | (dt[valid] <= 0)).any():
        raise ValueError("V3 marks an invalid dt as valid")
    if ((event_index == 0) & valid).any():
        raise ValueError("V3 first events must have invalid dt")
    if ((dt == 0) & valid).any():
        raise ValueError("V3 zero dt must have invalid validity")
    split_payload = json.loads(paths["split"].read_text(encoding="utf-8"))
    groups = []
    for split in ("train", "val", "test"):
        values = {str(value) for value in split_payload["group_ids"][split]}
        if set(groups) & values:
            raise ValueError("V3 split groups overlap")
        groups.extend(values)
    if set(groups) != set(frame["track_id"].astype(str)):
        raise ValueError("V3 split groups do not cover index tracks")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    if contract.get("schema_version") != VERSION or contract.get("source_purity", {}).get("allowed_sources") != ["CQ08"]:
        raise ValueError("V3 contract schema or purity is invalid")
    if contract.get("source_index") != f"data/processed/{paths['index'].name}":
        raise ValueError("V3 contract source_index does not identify the consumed V3 index")
    if contract.get("source_index_sha256", "").lower() != sha256_file(paths["index"]).lower():
        raise ValueError("V3 contract source_index_sha256 does not match the consumed V3 index")
    source_v1_path = processed / "RADAR-LN3-V1-cq08-tr-rd-main-index.csv"
    if contract.get("source_v1_index_sha256", "").lower() != sha256_file(source_v1_path).lower():
        raise ValueError("V3 contract source_v1_index_sha256 does not match the V1 source index")
    if contract.get("time_contract", {}).get("dt_valid_field") != "tr_dt_valid":
        raise ValueError("V3 time contract does not expose tr_dt_valid")
    if contract.get("training_contract", {}).get("loss_mode") != "causal_prefix":
        raise ValueError("V3 training contract does not expose causal_prefix")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("hashes", {}).get("index_sha256") != sha256_file(paths["index"]):
        raise ValueError("V3 summary index hash mismatch")
    result = {
        "status": "passed",
        "schema_version": VERSION,
        "index_path": str(paths["index"].relative_to(project_root)).replace("\\", "/"),
        "index_sha256": sha256_file(paths["index"]),
        "parent_count": int(frame["parent_sample_id"].nunique()),
        "point_count": int(len(frame)),
        "source_values": sorted(set(frame["source_name"].astype(str))),
        "label_counts": {str(k): int(v) for k, v in frame["label_name"].value_counts().sort_index().items()},
        "split_parent_counts": {str(k): int(v) for k, v in frame.groupby("split_name")["parent_sample_id"].nunique().items()},
        "dt_valid_count": int(valid.sum()),
        "dt_invalid_count": int((~valid).sum()),
        "dt_reason_counts": {str(k): int(v) for k, v in frame["tr_dt_reason"].value_counts().sort_index().items()},
        "split_is_grouped": True,
        "normalization_stats": "RADAR-LN3-V2 train-only statistics reused; no new stats generated",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the RADAR-LN3-V3 CQ08 prefix/dt contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    validate(args.project_root)


if __name__ == "__main__":
    main()
