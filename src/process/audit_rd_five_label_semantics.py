"""Audit five-class RD label provenance and train/validation metadata.

This audit deliberately does not load RD tensors, model predictions, or test
rows.  It checks only the CQ-08 manifest, the grouped split membership, and the
train/validation portions of the physical frame/window indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
MAT_RE = re.compile(r"_DAUR_RD_([^_]+)_\d+_(\d+)\.mat$", re.IGNORECASE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stat(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)

    def quantile(frac: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(frac * len(ordered))) - 1))
        return ordered[index]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "q01": quantile(0.01),
        "median": statistics.median(ordered),
        "q99": quantile(0.99),
        "max": ordered[-1],
    }


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def run_audit(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "data/manifests/cq08_unified_manifest.json"
    split_path = project_root / "data/manifests/cq08_grouped_split.json"
    frame_path = project_root / "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
    window_path = project_root / "data/processed/expert1_rd_only_causal_window_index_v1_physical_v3_1.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = {str(item) for item in split["train_group_ids"]}
    val_ids = {str(item) for item in split["val_group_ids"]}
    by_track = {str(item["track_id"]): item for item in manifest}

    frames = _read_csv(frame_path)
    windows = _read_csv(window_path)
    train_val_frames = [
        row for row in frames if str(row["track_id"]) in train_ids | val_ids
    ]
    train_val_windows = [row for row in windows if row["split_name"] in {"train", "val"}]

    mat_text_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    path_issues: list[dict[str, Any]] = []
    for item in manifest:
        track_id = str(item["track_id"])
        label_id = int(item["label_id"])
        for field, pattern in (
            ("track_file", re.compile(r"Tracks_(\d+)_(\d+)_(\d+)\.txt$", re.IGNORECASE)),
            ("point_file", re.compile(r"PointTracks_(\d+)_(\d+)_(\d+)\.txt$", re.IGNORECASE)),
            ("raw_echo_file", re.compile(r"(\d+)_Label_(\d+)\.dat$", re.IGNORECASE)),
        ):
            name = Path(str(item.get(field, ""))).name
            if not name:
                continue
            match = pattern.search(name)
            if match and (match.group(1) != track_id or int(match.group(2)) != label_id):
                path_issues.append({"track_id": track_id, "field": field, "name": name})
        for path_text in item.get("rd_mat_files", []):
            match = MAT_RE.search(Path(str(path_text)).name)
            if match:
                mat_text_by_label[str(item["label_name"])][match.group(1)] += 1
            else:
                path_issues.append(
                    {"track_id": track_id, "field": "rd_mat_file", "name": Path(str(path_text)).name}
                )

    parent_meta: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "labels": set(),
            "tracks": set(),
            "sources": set(),
            "range_coverage": [],
            "velocity_coverage": [],
            "range_reference": [],
            "observation_duration": [],
            "frame_time": [],
        }
    )
    frame_label_issues: list[dict[str, Any]] = []
    for row in train_val_frames:
        track_id = str(row["track_id"])
        meta = parent_meta[row["parent_sample_id"]]
        meta["labels"].add(row["label_name"])
        meta["tracks"].add(track_id)
        meta["sources"].add(row["source_name"])
        for destination, source in (
            ("range_coverage", "source_range_coverage_ratio"),
            ("velocity_coverage", "source_velocity_coverage_ratio"),
            ("range_reference", "target_range_reference_m"),
            ("observation_duration", "rd_observation_duration_seconds"),
            ("frame_time", "rd_frame_time_seconds"),
        ):
            try:
                value = float(row[source])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                meta[destination].append(value)
        manifest_item = by_track.get(track_id)
        if manifest_item is not None and row["label_name"] != manifest_item["label_name"]:
            frame_label_issues.append(
                {
                    "parent_sample_id": row["parent_sample_id"],
                    "track_id": track_id,
                    "frame_label": row["label_name"],
                    "manifest_label": manifest_item["label_name"],
                }
            )

    parent_conflicts = [
        {
            "parent_sample_id": parent_id,
            "labels": sorted(meta["labels"]),
            "tracks": sorted(meta["tracks"]),
            "sources": sorted(meta["sources"]),
        }
        for parent_id, meta in parent_meta.items()
        if len(meta["labels"]) != 1 or len(meta["tracks"]) != 1 or len(meta["sources"]) != 1
    ]

    physical_stats: dict[str, Any] = {}
    for label in LABELS:
        label_meta = [meta for meta in parent_meta.values() if meta["labels"] == {label}]
        physical_stats[label] = {
            "parents": len(label_meta),
            "frames": sum(len(meta["range_coverage"]) for meta in label_meta),
            "range_reference_m": _stat([v for meta in label_meta for v in meta["range_reference"]]),
            "observation_duration_s": _stat([v for meta in label_meta for v in meta["observation_duration"]]),
            "frame_time_s": _stat([v for meta in label_meta for v in meta["frame_time"]]),
            "range_coverage": _stat([v for meta in label_meta for v in meta["range_coverage"]]),
            "velocity_coverage": _stat([v for meta in label_meta for v in meta["velocity_coverage"]]),
        }

    window_parent: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"labels": set(), "splits": set(), "groups": set()}
    )
    for row in train_val_windows:
        meta = window_parent[row["parent_sample_id"]]
        meta["labels"].add(row["label_name"])
        meta["splits"].add(row["split_name"])
        meta["groups"].add(row["parent_split_group_id"])
    window_conflicts = [
        {
            "parent_sample_id": parent_id,
            "labels": sorted(meta["labels"]),
            "splits": sorted(meta["splits"]),
            "groups": sorted(meta["groups"]),
        }
        for parent_id, meta in window_parent.items()
        if len(meta["labels"]) != 1 or len(meta["splits"]) != 1 or len(meta["groups"]) != 1
    ]

    split_windows: dict[str, Any] = {}
    for split_name in ("train", "val"):
        rows = [row for row in train_val_windows if row["split_name"] == split_name]
        trainable = [row for row in rows if str(row["is_trainable"]).lower() == "true"]
        split_windows[split_name] = {
            "rows": len(rows),
            "trainable": len(trainable),
            "excluded": len(rows) - len(trainable),
            "labels_all": dict(Counter(row["label_name"] for row in rows)),
            "labels_trainable": dict(Counter(row["label_name"] for row in trainable)),
            "exclude_by_label": {
                label: dict(
                    Counter(
                        row["exclude_reason"]
                        for row in rows
                        if row["label_name"] == label and str(row["is_trainable"]).lower() != "true"
                    )
                )
                for label in LABELS
            },
        }

    return {
        "schema": "rd_five_label_semantics_audit_v1",
        "scope": "manifest plus train/val metadata only; no RD tensors, test rows, predictions, or metrics",
        "labels": list(LABELS),
        "checks": {
            "manifest": {
                "records": len(manifest),
                "label_counts": dict(Counter(item["label_name"] for item in manifest)),
                "label_ids_by_name": {
                    label: sorted({int(item["label_id"]) for item in manifest if item["label_name"] == label})
                    for label in LABELS
                },
                "usable_by_label": {
                    label: sum(bool(item["is_usable"]) for item in manifest if item["label_name"] == label)
                    for label in LABELS
                },
                "path_label_or_id_mismatch_count": len(path_issues),
                "mat_label_text_by_label": {
                    label: dict(mat_text_by_label.get(label, {})) for label in LABELS
                },
                "issues": path_issues[:20],
            },
            "train_val_frame_parent": {
                "frames": len(train_val_frames),
                "parents": len(parent_meta),
                "label_counts": dict(
                    Counter(
                        next(iter(meta["labels"]))
                        for meta in parent_meta.values()
                        if len(meta["labels"]) == 1
                    )
                ),
                "multi_label_track_or_source_parent_count": len(parent_conflicts),
                "manifest_frame_label_mismatch_count": len(frame_label_issues),
                "issues": (parent_conflicts + frame_label_issues)[:20],
            },
            "train_val_physical_stats_by_label": physical_stats,
            "train_val_windows": {
                "by_split": split_windows,
                "parent_label_split_group_conflict_count": len(window_conflicts),
                "issues": window_conflicts[:20],
            },
            "split_group_sizes_train_val_only": {"train": len(train_ids), "val": len(val_ids)},
            "train_val_group_intersection": len(train_ids & val_ids),
            "train_val_content_group_intersection": len(
                {
                    row["parent_split_group_id"]
                    for row in train_val_windows
                    if row["split_name"] == "train"
                }
                & {
                    row["parent_split_group_id"]
                    for row in train_val_windows
                    if row["split_name"] == "val"
                }
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.project_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema": report["schema"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
