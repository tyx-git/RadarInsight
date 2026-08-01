from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .rd_drone_bird_trainer import RD_ONLY_LABELS, _select_rd_only_ids
from .rd_parent_stability import build_parent_predecessor_lookup


def _parent_shape_by_id(frame_index: pd.DataFrame) -> dict[str, str]:
    return {
        str(parent_id): "+".join(sorted(set(group["rd_shape"].astype(str))))
        for parent_id, group in frame_index.groupby("parent_sample_id", sort=False)
    }


def _group_summary(frame: pd.DataFrame, predecessor_by_sample_id: dict[str, str], columns: list[str]) -> dict[str, object]:
    rows: dict[str, object] = {}
    for values, group in frame.groupby(columns, dropna=False, sort=True):
        key_values = values if isinstance(values, tuple) else (values,)
        key = "|".join(str(value) for value in key_values)
        sample_ids = group["sample_id"].astype(str)
        pair_count = int(sample_ids.isin(predecessor_by_sample_id).sum())
        rows[key] = {
            "window_count": int(len(group)),
            "parent_count": int(group["parent_sample_id"].nunique()),
            "pair_count": pair_count,
            "first_window_count": int(len(group) - pair_count),
            "pair_coverage": float(pair_count / len(group)) if len(group) else 0.0,
        }
    return rows


def audit_parent_stability(
    project_root: Path,
    *,
    index_name: str,
    window_map_name: str,
    frame_index_name: str,
    output_name: str,
) -> dict[str, object]:
    processed = project_root / "data" / "processed"
    index = pd.read_csv(processed / index_name, low_memory=False)
    window_map = pd.read_csv(processed / window_map_name, low_memory=False)
    frame_index = pd.read_csv(processed / frame_index_name, low_memory=False)
    anchor_ids = set(window_map["sample_id"].astype(str))
    eligible = index[
        index["sample_id"].astype(str).isin(anchor_ids)
        & index["label_name"].astype(str).isin(RD_ONLY_LABELS)
    ].copy()
    parent_shapes = _parent_shape_by_id(frame_index)
    eligible["rd_shape_provenance"] = eligible["parent_sample_id"].astype(str).map(parent_shapes).fillna("missing")
    split_reports: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        split = eligible[eligible["split_name"].astype(str).eq(split_name)].copy()
        lookup = build_parent_predecessor_lookup(split, split["sample_id"].astype(str).tolist())
        split_reports[split_name] = {
            "contract": lookup.report,
            "by_label": _group_summary(split, dict(lookup.predecessor_by_sample_id), ["label_name"]),
            "by_source": _group_summary(split, dict(lookup.predecessor_by_sample_id), ["source_name"]),
            "by_shape_provenance": _group_summary(split, dict(lookup.predecessor_by_sample_id), ["label_name", "rd_shape_provenance"]),
        }
    payload = {
        "scope": "read-only RD-only parent predecessor contract audit; no rebuild, training, fusion, threshold tuning, or test evaluation",
        "inputs": {
            "index": index_name,
            "window_map": window_map_name,
            "frame_index": frame_index_name,
        },
        "eligible_rd_only_window_count": int(len(eligible)),
        "split_reports": split_reports,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path = project_root / "tmp" / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit causal RD-only parent predecessor coverage.")
    parser.add_argument("--index-name", default="expert1_realtime_causal_window_index_v7.csv")
    parser.add_argument("--window-map-name", default="expert1_realtime_causal_window_cq08_physical_anchor_map_v4.csv")
    parser.add_argument("--frame-index-name", default="cq08_rd_physical_canonical_frame_index_v2.csv")
    parser.add_argument("--output-name", default="20260727-stage009j-parent-stability-audit.json")
    args = parser.parse_args()
    payload = audit_parent_stability(
        Path.cwd(),
        index_name=args.index_name,
        window_map_name=args.window_map_name,
        frame_index_name=args.frame_index_name,
        output_name=args.output_name,
    )
    print(json.dumps({"output": args.output_name, "split_reports": payload["split_reports"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
