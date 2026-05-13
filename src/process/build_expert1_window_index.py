from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.core.data.labels import two_layer_label_fields
from src.process.schema import TRACK_INDEX_COLUMNS, TRACK_SCHEMA_COLUMNS


EXPERT1_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}

WINDOW_EXTRA_COLUMNS = [
    "is_trainable",
    "exclude_reason",
    "parent_sample_id",
    "window_id",
    "window_start_seconds",
    "window_end_seconds",
    "window_duration_seconds",
    "window_point_count",
    "window_mode",
    "split_group_id",
]

WINDOW_INDEX_COLUMNS = TRACK_INDEX_COLUMNS + WINDOW_EXTRA_COLUMNS


def _safe_name(value: str) -> str:
    safe = str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe.replace(" ", "_")


def _is_present(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none"}


def _majority(values: Iterable[object], fallback: str = "UnknownTarget") -> str:
    cleaned = [str(value) for value in values if str(value).strip() and str(value).lower() != "nan"]
    if not cleaned:
        return fallback
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _window_starts(span_start: float, span_end: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if span_end < span_start:
        return []
    span = span_end - span_start
    if span <= window_seconds:
        return [span_start]
    starts: list[float] = []
    current = span_start
    epsilon = 1e-6
    while current + window_seconds <= span_end + epsilon:
        starts.append(float(current))
        current += stride_seconds
    tail_start = span_end - window_seconds
    if starts and tail_start - starts[-1] > stride_seconds * 0.5:
        starts.append(float(tail_start))
    elif not starts:
        starts.append(float(tail_start))
    return sorted({round(value, 6) for value in starts})


def _assign_group_splits(
    window_df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    group_label_rows = (
        window_df.groupby("split_group_id", sort=True)
        .agg(
            label_name=("label_name", lambda values: _majority(values, fallback="UnknownTarget")),
            source_name=("source_name", lambda values: _majority(values, fallback="")),
            window_count=("sample_id", "count"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, label_group in group_label_rows.groupby(["source_name", "label_name"], sort=True):
        group_ids = label_group["split_group_id"].astype(str).to_numpy(dtype=object)
        rng.shuffle(group_ids)
        n = len(group_ids)
        if n == 1:
            train_n, val_n, test_n = 1, 0, 0
        elif n == 2:
            train_n, val_n, test_n = 1, 0, 1
        else:
            train_n = int(round(n * train_ratio))
            val_n = int(round(n * val_ratio))
            train_n = max(1, train_n)
            val_n = max(1, val_n)
            if train_n + val_n >= n:
                if train_n >= val_n and train_n > 1:
                    train_n -= 1
                else:
                    val_n = max(1, val_n - 1)
            test_n = n - train_n - val_n
            if test_n < 1:
                test_n = 1
                if train_n >= val_n and train_n > 1:
                    train_n -= 1
                elif val_n > 1:
                    val_n -= 1
        for group_id in group_ids[:train_n]:
            assignments[str(group_id)] = "train"
        for group_id in group_ids[train_n : train_n + val_n]:
            assignments[str(group_id)] = "val"
        for group_id in group_ids[train_n + val_n : train_n + val_n + test_n]:
            assignments[str(group_id)] = "test"
    return assignments


def _summary(window_df: pd.DataFrame, assignments: dict[str, str]) -> dict[str, object]:
    split_by_sample = window_df["split_group_id"].astype(str).map(assignments).fillna("unassigned")
    trainable_mask = (
        window_df["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        if "is_trainable" in window_df.columns
        else pd.Series(True, index=window_df.index)
    )
    payload: dict[str, object] = {
        "total_windows": int(len(window_df)),
        "trainable_windows": int(trainable_mask.sum()),
        "diagnostic_windows": int((~trainable_mask).sum()),
        "parent_tracks": int(window_df["parent_sample_id"].nunique()),
        "source_counts": window_df["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": window_df["label_name"].value_counts(dropna=False).to_dict(),
        "window_mode_counts": window_df["window_mode"].value_counts(dropna=False).to_dict(),
        "rd_coverage": {
            "with_rd": int(window_df["rd_tensor_path"].map(_is_present).sum()),
            "without_rd": int((~window_df["rd_tensor_path"].map(_is_present)).sum()),
        },
        "split_counts": split_by_sample.value_counts(dropna=False).to_dict(),
        "trainable_split_counts": split_by_sample[trainable_mask].value_counts(dropna=False).to_dict(),
        "split_source_counts": {},
        "split_label_counts": {},
        "exclude_reason_counts": window_df.loc[~trainable_mask, "exclude_reason"].value_counts(dropna=False).to_dict()
        if "exclude_reason" in window_df.columns
        else {},
        "window_duration_seconds": {
            "min": float(window_df["window_duration_seconds"].min()) if len(window_df) else 0.0,
            "p50": float(window_df["window_duration_seconds"].quantile(0.50)) if len(window_df) else 0.0,
            "p90": float(window_df["window_duration_seconds"].quantile(0.90)) if len(window_df) else 0.0,
            "max": float(window_df["window_duration_seconds"].max()) if len(window_df) else 0.0,
        },
        "window_point_count": {
            "min": int(window_df["window_point_count"].min()) if len(window_df) else 0,
            "p50": float(window_df["window_point_count"].quantile(0.50)) if len(window_df) else 0.0,
            "p90": float(window_df["window_point_count"].quantile(0.90)) if len(window_df) else 0.0,
            "max": int(window_df["window_point_count"].max()) if len(window_df) else 0,
        },
    }
    for split_name in ["train", "val", "test", "unassigned"]:
        split_df = window_df[split_by_sample == split_name]
        if split_df.empty:
            continue
        payload["split_source_counts"][split_name] = split_df["source_name"].value_counts(dropna=False).to_dict()
        payload["split_label_counts"][split_name] = split_df["label_name"].value_counts(dropna=False).to_dict()
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for group_id, split_name in assignments.items():
        parent_splits[group_id].add(split_name)
    payload["parent_leakage_count"] = int(sum(1 for values in parent_splits.values() if len(values) > 1))
    return payload


def build_expert1_window_index(
    project_root: Path,
    window_seconds: float = 10.0,
    stride_seconds: float = 2.0,
    min_points: int = 5,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    processed_root = project_root / "data" / "processed"
    output_root = processed_root / "expert1_windows"
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = processed_root / "expert1_track_index.csv"
    index_df = pd.read_csv(index_path)
    index_df = index_df[index_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()
    rows: list[dict[str, object]] = []
    manifest_windows: list[dict[str, object]] = []

    for row in index_df.to_dict(orient="records"):
        parent_sample_id = str(row["sample_id"])
        source_name = str(row["source_name"])
        csv_path = project_root / str(row["csv_path"])
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty or "time_seconds" not in frame.columns:
            continue
        frame = frame.sort_values("time_seconds").reset_index(drop=True)
        time_values = pd.to_numeric(frame["time_seconds"], errors="coerce")
        frame = frame[time_values.notna()].copy()
        if frame.empty:
            continue
        frame["time_seconds"] = pd.to_numeric(frame["time_seconds"], errors="coerce").astype(float)
        start_time = float(frame["time_seconds"].min())
        end_time = float(frame["time_seconds"].max())
        parent_duration = float(end_time - start_time)
        starts = _window_starts(start_time, end_time, window_seconds, stride_seconds)
        parent_safe = _safe_name(parent_sample_id)

        def emit_window(
            window_index: int,
            window_df: pd.DataFrame,
            window_mode: str,
            effective_start: float,
            effective_end: float,
            force_exclude_reasons: list[str] | None = None,
        ) -> None:
            label_name = _majority(window_df.get("label_name", pd.Series(dtype=str)), fallback=str(row["label_name"]))
            raw_label_name = _majority(
                window_df.get("label_name_raw", pd.Series(dtype=str)),
                fallback=str(row.get("label_name_raw", label_name)),
            )
            label_fields = two_layer_label_fields(
                label_name,
                source_name=source_name,
                batch_id=row.get("batch_id", ""),
                label_name_raw=raw_label_name,
            )
            window_id = f"{parent_sample_id}|win{window_index:04d}"
            window_df = window_df.copy()
            window_df["record_id"] = [f"{window_id}#{idx:04d}" for idx in range(len(window_df))]
            window_df["pair_id"] = window_id
            window_df["label_name"] = label_name
            window_df["label_name_raw"] = raw_label_name
            window_df["coarse_label"] = label_fields["coarse_label"]
            window_df["fine_label"] = label_fields["fine_label"]
            for column in TRACK_SCHEMA_COLUMNS:
                if column not in window_df.columns:
                    window_df[column] = np.nan if column not in {"note_text"} else ""
            window_df = window_df[TRACK_SCHEMA_COLUMNS]
            source_dir = output_root / _safe_name(source_name)
            source_dir.mkdir(parents=True, exist_ok=True)
            window_csv_path = source_dir / f"{parent_safe}__win_{window_index:04d}.csv"
            window_df.to_csv(window_csv_path, index=False, encoding="utf-8")
            split_group_id = f"{source_name}|{parent_sample_id}"
            effective_duration = float(effective_end - effective_start)
            exclude_reasons: list[str] = []
            if len(window_df) < min_points:
                exclude_reasons.append("too_few_points_for_window_training")
            if effective_duration > window_seconds + 1e-6:
                exclude_reasons.append("window_duration_exceeds_target")
            if label_name not in EXPERT1_LABELS:
                exclude_reasons.append(f"unsupported_label:{label_name}")
            if force_exclude_reasons:
                exclude_reasons.extend(force_exclude_reasons)
            exclude_reasons = sorted(set(exclude_reasons))
            is_trainable = len(exclude_reasons) == 0
            row_payload = {
                "dataset_name": str(row.get("dataset_name", "")),
                "source_name": source_name,
                "sample_id": window_id,
                "sample_kind": "expert1_window",
                "timestamp": str(row.get("timestamp", "")),
                "batch_id": str(row.get("batch_id", "")),
                "label_name_raw": raw_label_name,
                "label_name": label_name,
                "coarse_label": label_fields["coarse_label"],
                "fine_label": label_fields["fine_label"],
                "record_count": int(len(window_df)),
                "time_seconds_min": float(window_df["time_seconds"].min()),
                "time_seconds_max": float(window_df["time_seconds"].max()),
                "csv_path": window_csv_path.relative_to(project_root).as_posix(),
                "rd_path": str(row.get("rd_path", "")),
                "rd_summary_csv_path": str(row.get("rd_summary_csv_path", "")),
                "rd_tensor_path": str(row.get("rd_tensor_path", "")),
                "rd_phase_missing": bool(row.get("rd_phase_missing", False)),
                "height_missing": bool(row.get("height_missing", False)),
                "is_trainable": is_trainable,
                "exclude_reason": ";".join(exclude_reasons),
                "parent_sample_id": parent_sample_id,
                "window_id": f"win{window_index:04d}",
                "window_start_seconds": float(effective_start),
                "window_end_seconds": float(effective_end),
                "window_duration_seconds": effective_duration,
                "window_point_count": int(len(window_df)),
                "window_mode": window_mode,
                "split_group_id": split_group_id,
            }
            rows.append(row_payload)
            manifest_windows.append(
                {
                    "sample_id": window_id,
                    "parent_sample_id": parent_sample_id,
                    "split_group_id": split_group_id,
                    "source_name": source_name,
                    "label_name": label_name,
                    "window_start_seconds": float(effective_start),
                    "window_end_seconds": float(effective_end),
                    "window_duration_seconds": effective_duration,
                    "window_point_count": int(len(window_df)),
                    "window_mode": window_mode,
                    "is_trainable": is_trainable,
                    "exclude_reason": ";".join(exclude_reasons),
                }
            )

        emitted_window = False
        for window_index, window_start in enumerate(starts):
            window_end = min(float(window_start + window_seconds), end_time)
            if parent_duration <= window_seconds:
                mask = np.ones((len(frame),), dtype=bool)
                window_mode = "short_full" if parent_duration < window_seconds else "exact_full"
                effective_start = start_time
                effective_end = end_time
            else:
                mask = (frame["time_seconds"] >= window_start) & (frame["time_seconds"] < window_end)
                if window_index == len(starts) - 1:
                    mask = (frame["time_seconds"] >= window_start) & (frame["time_seconds"] <= window_end)
                window_mode = "sliding"
                effective_start = float(window_start)
                effective_end = float(window_end)
            window_df = frame[mask].copy()
            if len(window_df) < min_points:
                if parent_duration <= window_seconds and not emitted_window:
                    emit_window(0, frame, "short_full_below_min_points", start_time, end_time)
                    emitted_window = True
                continue
            emit_window(
                window_index,
                window_df,
                window_mode,
                effective_start,
                effective_end,
            )
            emitted_window = True
        if not emitted_window:
            emit_window(
                0,
                frame,
                "diagnostic_sparse_full",
                start_time,
                end_time,
                force_exclude_reasons=["no_window_meets_min_points"],
            )

    window_df = pd.DataFrame(rows)
    if window_df.empty:
        raise RuntimeError("no expert1 windows were generated")
    window_df = window_df[WINDOW_INDEX_COLUMNS]
    assignments = _assign_group_splits(window_df, seed, train_ratio, val_ratio, test_ratio)
    window_df["split_name"] = window_df["split_group_id"].astype(str).map(assignments).fillna("unassigned")
    final_columns = WINDOW_INDEX_COLUMNS + ["split_name"]
    window_index_path = processed_root / "expert1_window_index.csv"
    window_df[final_columns].to_csv(window_index_path, index=False, encoding="utf-8")

    split_manifest = {
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "min_points": int(min_points),
        "split_group_key": "source_name|parent_sample_id",
        "train_ids": sorted(window_df[(window_df["split_name"] == "train") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "val_ids": sorted(window_df[(window_df["split_name"] == "val") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "test_ids": sorted(window_df[(window_df["split_name"] == "test") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "diagnostic_ids": sorted(window_df[~window_df["is_trainable"]]["sample_id"].astype(str).tolist()),
        "group_assignments": assignments,
    }
    manifest_path = project_root / "data" / "manifests" / "expert1_window_split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _summary(window_df, assignments)
    summary.update(
        {
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "min_points": int(min_points),
            "window_index": window_index_path.relative_to(project_root).as_posix(),
            "split_manifest": manifest_path.relative_to(project_root).as_posix(),
        }
    )
    summary_path = processed_root / "expert1_window_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deployment-style sliding window index for Expert1.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = build_expert1_window_index(
        project_root,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_points=args.min_points,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
