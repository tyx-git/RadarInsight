from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "RADAR-LN3-V0"
LABELS = {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate(project_root: Path) -> dict[str, Any]:
    processed = project_root / "data" / "processed"
    index_path = processed / f"{VERSION}-cq08-tr-rd-point-index.csv"
    manifest_path = processed / f"{VERSION}-cq08-tr-rd-point-manifest.json"
    split_path = processed / f"{VERSION}-cq08-tr-rd-split.json"
    contract_path = processed / f"{VERSION}-cq08-tr-rd-data-contract.json"
    summary_path = processed / f"{VERSION}-cq08-tr-rd-summary.json"
    for path in (index_path, manifest_path, split_path, contract_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"V0 contract file missing: {path}")
    frame = pd.read_csv(index_path, low_memory=False)
    if len(frame) != 29724:
        raise ValueError(f"unexpected V0 point count: {len(frame)}")
    if frame["schema_version"].astype(str).ne(VERSION).any():
        raise ValueError("V0 index contains a wrong schema version")
    if set(frame["source_name"].astype(str)) != {"CQ08"}:
        raise ValueError("V0 index is not CQ08-pure")
    if set(frame["label_name"].astype(str)) != LABELS:
        raise ValueError("V0 index does not contain exactly the five CQ08 labels")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V0 index contains duplicate sample IDs")
    if frame.groupby("parent_sample_id")["split_name"].nunique().max() != 1:
        raise ValueError("V0 parent appears in more than one split")
    if frame.groupby("parent_sample_id")["label_name"].nunique().max() != 1:
        raise ValueError("V0 parent has more than one label")
    rd_present = frame["rd_present"].map(_as_bool)
    partial_parent = frame["rd_quality_partial_parent"].map(_as_bool)
    negative_dt = frame["tr_dt_is_negative"].map(_as_bool)
    if int(rd_present.sum()) != 29413:
        raise ValueError("V0 RD-present point count changed")
    if int(pd.to_numeric(frame["rd_frame_count"], errors="raise").sum()) != 106026:
        raise ValueError("V0 RD frame count changed")
    if int(partial_parent.groupby(frame["parent_sample_id"]).any().sum()) != 44:
        raise ValueError("V0 partial parent quality flags are missing")
    if int(negative_dt.sum()) != 2:
        raise ValueError("V0 negative TR dt count changed")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_ids = set()
    for split_name in ("train", "val", "test"):
        values = {str(value) for value in split_payload["group_ids"][split_name]}
        if split_ids & values:
            raise ValueError("V0 split groups overlap")
        split_ids |= values
    if split_ids != set(frame["track_id"].astype(str)):
        raise ValueError("V0 split groups do not cover the index")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("rd_aggregation", {}).get("rd_time_semantics") != "cpi_relative_elapsed_not_tr_wall_clock":
        raise ValueError("V0 contract incorrectly treats CPI time as TR wall clock")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_hash = str(summary.get("index_sha256", ""))
    actual_hash = _sha256(index_path)
    if expected_hash != actual_hash:
        raise ValueError("V0 summary index hash does not match the actual index")
    tensor_check: dict[str, Any] = {}
    present = frame[rd_present].iloc[0]
    tensor_path = project_root / str(present["rd_tensor_path"])
    with np.load(tensor_path, allow_pickle=False) as payload:
        tensor_check = {name: list(payload[name].shape) for name in ("amplitude", "phase", "time_seconds", "velocity_mps", "range_offset_m")}
    result = {
        "status": "passed",
        "schema_version": VERSION,
        "index_path": str(index_path.relative_to(project_root)).replace("\\", "/"),
        "index_sha256": actual_hash,
        "parent_count": int(frame["parent_sample_id"].nunique()),
        "point_count": int(len(frame)),
        "rd_present_points": int(rd_present.sum()),
        "rd_missing_points": int((~rd_present).sum()),
        "rd_frame_count": int(pd.to_numeric(frame["rd_frame_count"], errors="raise").sum()),
        "source_values": sorted(set(frame["source_name"].astype(str))),
        "label_counts": {str(k): int(v) for k, v in frame["label_name"].value_counts().sort_index().items()},
        "split_parent_counts": {str(k): int(v) for k, v in frame.groupby("split_name")["parent_sample_id"].nunique().items()},
        "tensor_shape_sample": tensor_check,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the RADAR-LN3-V0 CQ08 point contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    validate(args.project_root.resolve())


if __name__ == "__main__":
    main()
