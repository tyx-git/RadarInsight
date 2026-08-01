from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = "rd_only_causal_window_v1"
TIME_SEMANTICS = "canonical_sensor_relative_elapsed"
ANCHOR_POLICY = "parent_label_association_only"
RD_ONLY_WINDOW_COLUMNS = [
    "schema_version",
    "sample_id",
    "parent_sample_id",
    "source_name",
    "label_name",
    "split_name",
    "parent_split_group_id",
    "physical_frame_index_name",
    "rd_tensor_path",
    "rd_tensor_frame_start_index",
    "rd_tensor_frame_end_index",
    "rd_tensor_frame_indices_json",
    "rd_time_start_seconds",
    "rd_time_end_seconds",
    "rd_time_requested_end_seconds",
    "rd_window_duration_seconds",
    "rd_window_frame_count",
    "rd_window_mode",
    "is_prefix_window",
    "rd_present",
    "rd_is_multimodal",
    "rd_phase_missing",
    "rd_coverage_ratio",
    "rd_time_sync_status",
    "is_trainable",
    "exclude_reason",
    "rd_time_semantics",
    "rd_time_monotonic",
    "rd_fusion_eligible",
    "rd_anchor_policy",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _window_starts(start_seconds: float, end_seconds: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if end_seconds <= start_seconds + window_seconds:
        return [start_seconds]
    starts: list[float] = []
    current = start_seconds
    epsilon = 1.0e-9
    while current + window_seconds <= end_seconds + epsilon:
        starts.append(float(current))
        current += stride_seconds
    tail_start = end_seconds - window_seconds
    if not starts or tail_start - starts[-1] > stride_seconds * 0.5:
        starts.append(float(tail_start))
    return sorted({round(value, 9) for value in starts})


def _parent_split_records(split_index: pd.DataFrame) -> dict[str, dict[str, str]]:
    required = {"parent_sample_id", "label_name", "split_name", "split_group_id"}
    missing = sorted(required - set(split_index.columns))
    if missing:
        raise ValueError(f"split index is missing required columns: {missing}")
    records: dict[str, dict[str, str]] = {}
    for parent_sample_id, group in split_index.groupby("parent_sample_id", sort=True):
        labels = sorted(group["label_name"].astype(str).unique().tolist())
        splits = sorted(group["split_name"].astype(str).unique().tolist())
        split_groups = sorted(group["split_group_id"].astype(str).unique().tolist())
        if len(labels) != 1 or len(splits) != 1 or len(split_groups) != 1:
            raise ValueError(
                "split index parent contract failed: "
                f"parent={parent_sample_id}, labels={labels}, splits={splits}, groups={split_groups}"
            )
        records[str(parent_sample_id)] = {
            "label_name": labels[0],
            "split_name": splits[0],
            "split_group_id": split_groups[0],
        }
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records.values():
        group_splits[record["split_group_id"]].add(record["split_name"])
    conflicts = {group_id: sorted(splits) for group_id, splits in group_splits.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"split index has cross-split group leakage: {list(sorted(conflicts.items()))[:5]}")
    return records


def _window_candidates(
    frame_times: np.ndarray,
    window_seconds: float,
    stride_seconds: float,
    include_prefix_windows: bool,
) -> list[tuple[float, float, str, bool]]:
    start_seconds = float(frame_times[0])
    end_seconds = float(frame_times[-1])
    candidates: list[tuple[float, float, str, bool]] = []
    if include_prefix_windows:
        prefix_end = start_seconds + stride_seconds
        prefix_limit = min(end_seconds, start_seconds + window_seconds)
        while prefix_end < prefix_limit - 1.0e-9:
            candidates.append((start_seconds, float(prefix_end), "prefix", True))
            prefix_end += stride_seconds
    if end_seconds <= start_seconds + window_seconds:
        candidates.append((start_seconds, end_seconds, "short_full", False))
    else:
        for window_start in _window_starts(start_seconds, end_seconds, window_seconds, stride_seconds):
            candidates.append(
                (
                    float(window_start),
                    float(min(window_start + window_seconds, end_seconds)),
                    "sliding",
                    False,
                )
            )
    deduplicated: dict[tuple[float, float, str, bool], tuple[float, float, str, bool]] = {}
    for candidate in candidates:
        deduplicated.setdefault(candidate, candidate)
    return sorted(deduplicated.values(), key=lambda value: (value[1], value[0], value[2]))


def _trainable_reason(frame_times: np.ndarray, min_frames: int) -> tuple[bool, str]:
    reasons: list[str] = []
    if int(frame_times.size) < int(min_frames):
        reasons.append("too_few_rd_frames")
    if int(np.unique(frame_times).size) < 2:
        reasons.append("too_few_unique_rd_times")
    if frame_times.size and float(frame_times[-1] - frame_times[0]) <= 0.0:
        reasons.append("zero_rd_duration")
    return not reasons, ";".join(reasons)


def _summary(window_df: pd.DataFrame) -> dict[str, object]:
    split_group_splits = window_df.groupby("parent_split_group_id", sort=True)["split_name"].nunique()
    return {
        "window_count": int(len(window_df)),
        "trainable_window_count": int(window_df["is_trainable"].sum()),
        "diagnostic_window_count": int((~window_df["is_trainable"]).sum()),
        "parent_count": int(window_df["parent_sample_id"].nunique()),
        "split_counts": window_df["split_name"].value_counts().to_dict(),
        "split_label_counts": {
            split_name: split_df["label_name"].value_counts().to_dict()
            for split_name, split_df in window_df.groupby("split_name", sort=True)
        },
        "window_mode_counts": window_df["rd_window_mode"].value_counts().to_dict(),
        "exclude_reason_counts": window_df.loc[~window_df["is_trainable"], "exclude_reason"].value_counts().to_dict(),
        "frame_count": {
            "min": int(window_df["rd_window_frame_count"].min()),
            "p50": float(window_df["rd_window_frame_count"].quantile(0.5)),
            "p95": float(window_df["rd_window_frame_count"].quantile(0.95)),
            "max": int(window_df["rd_window_frame_count"].max()),
        },
        "duration_seconds": {
            "min": float(window_df["rd_window_duration_seconds"].min()),
            "p50": float(window_df["rd_window_duration_seconds"].quantile(0.5)),
            "p95": float(window_df["rd_window_duration_seconds"].quantile(0.95)),
            "max": float(window_df["rd_window_duration_seconds"].max()),
        },
        "cross_split_parent_group_count": int((split_group_splits > 1).sum()),
        "rd_fusion_eligible_window_count": int(window_df["rd_fusion_eligible"].sum()),
    }


def build_rd_only_causal_window_index(
    project_root: Path,
    *,
    physical_frame_index_name: str = "cq08_rd_physical_canonical_frame_index_v3_1.csv",
    split_index_name: str = "expert1_realtime_causal_window_index_v8_rd_content.csv",
    output_name: str = "expert1_rd_only_causal_window_index_v1.csv",
    manifest_name: str = "expert1_rd_only_causal_window_manifest_v1.json",
    summary_name: str = "expert1_rd_only_causal_window_summary_v1.json",
    window_seconds: float = 0.5,
    stride_seconds: float = 0.1,
    min_frames: int = 12,
    include_prefix_windows: bool = True,
) -> dict[str, object]:
    if window_seconds <= 0.0 or stride_seconds <= 0.0:
        raise ValueError("window_seconds and stride_seconds must be positive")
    if min_frames < 2:
        raise ValueError("min_frames must be at least two")
    processed_root = project_root / "data" / "processed"
    frame_index_path = processed_root / physical_frame_index_name
    split_index_path = processed_root / split_index_name
    frame_index = pd.read_csv(frame_index_path, low_memory=False)
    split_index = pd.read_csv(split_index_path, low_memory=False)
    required_frame_columns = {
        "parent_sample_id",
        "source_name",
        "label_name",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "rd_frame_time_seconds",
        "rd_tensor_path",
    }
    missing_columns = sorted(required_frame_columns - set(frame_index.columns))
    if missing_columns:
        raise ValueError(f"physical frame index is missing required columns: {missing_columns}")
    frame_index = frame_index[frame_index["is_tensor_frame_kept"].map(_as_bool)].copy()
    split_records = _parent_split_records(split_index)
    rows: list[dict[str, object]] = []
    for parent_sample_id, parent_df in frame_index.groupby("parent_sample_id", sort=True):
        parent_id = str(parent_sample_id)
        split_record = split_records.get(parent_id)
        if split_record is None:
            raise ValueError(f"physical RD parent is missing from split index: {parent_id}")
        source_names = sorted(parent_df["source_name"].astype(str).unique().tolist())
        labels = sorted(parent_df["label_name"].astype(str).unique().tolist())
        tensor_paths = sorted(parent_df["rd_tensor_path"].astype(str).unique().tolist())
        if len(source_names) != 1 or len(labels) != 1 or len(tensor_paths) != 1:
            raise ValueError(
                "physical RD parent contract failed: "
                f"parent={parent_id}, sources={source_names}, labels={labels}, tensors={tensor_paths}"
            )
        if labels[0] != split_record["label_name"]:
            raise ValueError(f"physical/split label mismatch for parent: {parent_id}")
        resolved_tensor_path = _resolve_project_path(project_root, tensor_paths[0])
        if not resolved_tensor_path.exists():
            raise FileNotFoundError(f"physical RD tensor does not exist for parent: {parent_id}")
        ordered = parent_df.sort_values(["rd_frame_time_seconds", "tensor_frame_index"], kind="stable").reset_index(drop=True)
        frame_times = pd.to_numeric(ordered["rd_frame_time_seconds"], errors="coerce").to_numpy(dtype=np.float64)
        tensor_indices = pd.to_numeric(ordered["tensor_frame_index"], errors="coerce").to_numpy(dtype=np.int64)
        if not np.isfinite(frame_times).all() or (np.diff(frame_times) < -1.0e-9).any():
            raise ValueError(f"physical RD time is not monotonic for parent: {parent_id}")
        if len(np.unique(tensor_indices)) != len(tensor_indices) or (tensor_indices < 0).any():
            raise ValueError(f"physical RD tensor indices are invalid for parent: {parent_id}")
        seen_windows: set[tuple[tuple[int, ...], str]] = set()
        emitted = 0
        for requested_start, requested_end, mode, is_prefix in _window_candidates(
            frame_times,
            window_seconds,
            stride_seconds,
            include_prefix_windows,
        ):
            selection = (frame_times >= requested_start - 1.0e-9) & (frame_times <= requested_end + 1.0e-9)
            selected_times = frame_times[selection]
            selected_indices = tensor_indices[selection]
            if not selected_indices.size:
                continue
            deduplication_key = (tuple(int(value) for value in selected_indices.tolist()), mode)
            if deduplication_key in seen_windows:
                continue
            seen_windows.add(deduplication_key)
            is_trainable, exclude_reason = _trainable_reason(selected_times, min_frames)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": f"{parent_id}|rdwin{emitted:04d}",
                    "parent_sample_id": parent_id,
                    "source_name": source_names[0],
                    "label_name": labels[0],
                    "split_name": split_record["split_name"],
                    "parent_split_group_id": split_record["split_group_id"],
                    "physical_frame_index_name": physical_frame_index_name,
                    "rd_tensor_path": tensor_paths[0],
                    "rd_tensor_frame_start_index": int(selected_indices[0]),
                    "rd_tensor_frame_end_index": int(selected_indices[-1]),
                    "rd_tensor_frame_indices_json": json.dumps([int(value) for value in selected_indices.tolist()], separators=(",", ":")),
                    "rd_time_start_seconds": float(selected_times[0]),
                    "rd_time_end_seconds": float(selected_times[-1]),
                    "rd_time_requested_end_seconds": float(requested_end),
                    "rd_window_duration_seconds": float(selected_times[-1] - selected_times[0]),
                    "rd_window_frame_count": int(selected_indices.size),
                    "rd_window_mode": mode,
                    "is_prefix_window": bool(is_prefix),
                    "rd_present": True,
                    "rd_is_multimodal": True,
                    "rd_phase_missing": False,
                    "rd_coverage_ratio": 1.0,
                    "rd_time_sync_status": "canonical_parent_association",
                    "is_trainable": bool(is_trainable),
                    "exclude_reason": exclude_reason,
                    "rd_time_semantics": TIME_SEMANTICS,
                    "rd_time_monotonic": True,
                    "rd_fusion_eligible": False,
                    "rd_anchor_policy": ANCHOR_POLICY,
                }
            )
            emitted += 1
        if emitted == 0:
            raise ValueError(f"physical RD parent emitted no causal windows: {parent_id}")
    window_df = pd.DataFrame(rows, columns=RD_ONLY_WINDOW_COLUMNS)
    if window_df.empty or window_df["sample_id"].duplicated().any():
        raise ValueError("RD-only causal window index is empty or contains duplicate sample IDs")
    if (window_df["rd_time_end_seconds"] > window_df["rd_time_requested_end_seconds"] + 1.0e-9).any():
        raise ValueError("RD-only causal window contains frame after requested decision boundary")
    output_path = processed_root / output_name
    window_df.to_csv(output_path, index=False, encoding="utf-8")
    summary = _summary(window_df)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "RD-only causal window index from canonical RD time; no track-point frame slicing, training, fusion, threshold tuning, or test evaluation",
        "inputs": {
            "physical_frame_index": physical_frame_index_name,
            "physical_frame_index_sha256": _sha256_file(frame_index_path),
            "split_index": split_index_name,
            "split_index_sha256": _sha256_file(split_index_path),
        },
        "window_contract": {
            "time_semantics": TIME_SEMANTICS,
            "anchor_policy": ANCHOR_POLICY,
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "min_frames": int(min_frames),
            "include_prefix_windows": bool(include_prefix_windows),
            "fusion_eligible": False,
        },
        "output": output_name,
        "summary": summary,
        "test_evaluation_executed": False,
    }
    manifest_path = project_root / "data" / "manifests" / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.update(
        {
            "scope": manifest["scope"],
            "output_path": output_path.relative_to(project_root).as_posix(),
            "manifest_path": manifest_path.relative_to(project_root).as_posix(),
            "test_evaluation_executed": False,
        }
    )
    summary_path = processed_root / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal RD-only windows from canonical RD frame time.")
    parser.add_argument("--physical-frame-index-name", default="cq08_rd_physical_canonical_frame_index_v3_1.csv")
    parser.add_argument("--split-index-name", default="expert1_realtime_causal_window_index_v8_rd_content.csv")
    parser.add_argument("--output-name", default="expert1_rd_only_causal_window_index_v1.csv")
    parser.add_argument("--manifest-name", default="expert1_rd_only_causal_window_manifest_v1.json")
    parser.add_argument("--summary-name", default="expert1_rd_only_causal_window_summary_v1.json")
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.1)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--exclude-prefix-windows", action="store_true")
    args = parser.parse_args()
    result = build_rd_only_causal_window_index(
        Path.cwd(),
        physical_frame_index_name=args.physical_frame_index_name,
        split_index_name=args.split_index_name,
        output_name=args.output_name,
        manifest_name=args.manifest_name,
        summary_name=args.summary_name,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_frames=args.min_frames,
        include_prefix_windows=not args.exclude_prefix_windows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
