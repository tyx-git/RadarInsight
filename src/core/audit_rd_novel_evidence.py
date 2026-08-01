from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rd_drone_bird_trainer import RD_ONLY_LABELS


WINDOW_INDEX_COLUMNS = {
    "sample_id",
    "parent_sample_id",
    "split_name",
    "label_name",
    "window_start_seconds",
    "window_end_seconds",
}
WINDOW_MAP_COLUMNS = {
    "sample_id",
    "parent_sample_id",
    "label_name",
    "segment_ids_json",
}
MANIFEST_COLUMNS = {
    "segment_id",
    "parent_sample_id",
    "label_name",
    "segment_end_seconds",
    "track_point_time_seconds",
    "rd_source_frame_indices_json",
    "rd_tensor_frame_indices_json",
}
FRAME_INDEX_COLUMNS = {
    "parent_sample_id",
    "label_name",
    "raw_frame_index",
    "tensor_frame_index",
    "is_tensor_frame_kept",
}
MODEL_SELECTION_SPLITS = {"train", "val"}


@dataclass(frozen=True)
class FrameReference:
    raw_frame_index: int
    tensor_frame_index: int
    anchor_time_seconds: float


@dataclass(frozen=True)
class SegmentReference:
    parent_sample_id: str
    label_name: str
    end_seconds: float
    frame_references: tuple[FrameReference, ...]


@dataclass(frozen=True)
class WindowEvidence:
    sample_id: str
    parent_sample_id: str
    split_name: str
    label_name: str
    window_start_seconds: float
    window_end_seconds: float
    frame_references: tuple[FrameReference, ...]
    segment_count: int


def _require_columns(frame: pd.DataFrame, required_columns: set[str], name: str) -> None:
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{name} missing columns: {missing_columns}")


def _parse_json_list(value: object, *, field_name: str, record_id: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} for {record_id} is not JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {field_name} for {record_id}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} for {record_id} is not a JSON list")
    return parsed


def _finite_float(value: object, *, field_name: str, record_id: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} for {record_id} is not finite")
    return result


def _build_frame_lookup(frame_index: pd.DataFrame) -> dict[tuple[str, int], tuple[int, str]]:
    _require_columns(frame_index, FRAME_INDEX_COLUMNS, "physical canonical frame index")
    kept = frame_index[frame_index["is_tensor_frame_kept"].astype(bool)].copy()
    kept["raw_frame_index"] = pd.to_numeric(kept["raw_frame_index"], errors="raise").astype(int)
    kept["tensor_frame_index"] = pd.to_numeric(kept["tensor_frame_index"], errors="raise").astype(int)
    keys = list(zip(kept["parent_sample_id"].astype(str), kept["raw_frame_index"].tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("physical canonical frame index has duplicate kept parent/raw references")
    return {
        (str(row.parent_sample_id), int(row.raw_frame_index)): (int(row.tensor_frame_index), str(row.label_name))
        for row in kept.itertuples(index=False)
    }


def _build_segment_lookup(manifest: pd.DataFrame, frame_lookup: dict[tuple[str, int], tuple[int, str]]) -> dict[str, SegmentReference]:
    _require_columns(manifest, MANIFEST_COLUMNS, "physical anchor manifest")
    if manifest["segment_id"].astype(str).duplicated().any():
        raise ValueError("physical anchor manifest has duplicate segment_id")
    segment_lookup: dict[str, SegmentReference] = {}
    for row in manifest.itertuples(index=False):
        segment_id = str(row.segment_id)
        parent_sample_id = str(row.parent_sample_id)
        label_name = str(row.label_name)
        raw_indices = _parse_json_list(
            row.rd_source_frame_indices_json,
            field_name="rd_source_frame_indices_json",
            record_id=segment_id,
        )
        tensor_indices = _parse_json_list(
            row.rd_tensor_frame_indices_json,
            field_name="rd_tensor_frame_indices_json",
            record_id=segment_id,
        )
        if len(raw_indices) != len(tensor_indices):
            raise ValueError(f"raw/tensor frame count mismatch for {segment_id}")
        anchor_time_seconds = _finite_float(
            row.track_point_time_seconds,
            field_name="track_point_time_seconds",
            record_id=segment_id,
        )
        frame_references: list[FrameReference] = []
        for raw_value, tensor_value in zip(raw_indices, tensor_indices):
            raw_frame_index = int(raw_value)
            tensor_frame_index = int(tensor_value)
            indexed = frame_lookup.get((parent_sample_id, raw_frame_index))
            if indexed is None:
                raise ValueError(f"manifest frame missing from physical index: {segment_id}:{raw_frame_index}")
            indexed_tensor_index, indexed_label = indexed
            if indexed_tensor_index != tensor_frame_index:
                raise ValueError(f"manifest tensor index mismatch: {segment_id}:{raw_frame_index}")
            if indexed_label != label_name:
                raise ValueError(f"manifest/frame label mismatch: {segment_id}:{raw_frame_index}")
            frame_references.append(
                FrameReference(
                    raw_frame_index=raw_frame_index,
                    tensor_frame_index=tensor_frame_index,
                    anchor_time_seconds=anchor_time_seconds,
                )
            )
        segment_lookup[segment_id] = SegmentReference(
            parent_sample_id=parent_sample_id,
            label_name=label_name,
            end_seconds=_finite_float(row.segment_end_seconds, field_name="segment_end_seconds", record_id=segment_id),
            frame_references=tuple(frame_references),
        )
    return segment_lookup


def _build_window_evidence(index: pd.DataFrame, window_map: pd.DataFrame, segment_lookup: dict[str, SegmentReference]) -> list[WindowEvidence]:
    _require_columns(index, WINDOW_INDEX_COLUMNS, "causal window index")
    _require_columns(window_map, WINDOW_MAP_COLUMNS, "physical anchor window map")
    if window_map["sample_id"].astype(str).duplicated().any():
        raise ValueError("physical anchor window map has duplicate sample_id")
    selected_index = index[
        index["label_name"].astype(str).isin(RD_ONLY_LABELS)
        & index["split_name"].astype(str).isin(MODEL_SELECTION_SPLITS)
    ].copy()
    selected_index["sample_id"] = selected_index["sample_id"].astype(str)
    selected_map = window_map.copy()
    selected_map["sample_id"] = selected_map["sample_id"].astype(str)
    joined = selected_index.merge(
        selected_map[["sample_id", "parent_sample_id", "label_name", "segment_ids_json"]],
        on="sample_id",
        how="inner",
        suffixes=("_index", "_map"),
        validate="one_to_one",
    )
    evidence_rows: list[WindowEvidence] = []
    for row in joined.itertuples(index=False):
        sample_id = str(row.sample_id)
        parent_sample_id = str(row.parent_sample_id_index)
        label_name = str(row.label_name_index)
        if parent_sample_id != str(row.parent_sample_id_map) or label_name != str(row.label_name_map):
            raise ValueError(f"window index/map identity mismatch for {sample_id}")
        window_end_seconds = _finite_float(row.window_end_seconds, field_name="window_end_seconds", record_id=sample_id)
        frame_references: list[FrameReference] = []
        seen_tensor_indices: set[int] = set()
        segment_ids = _parse_json_list(row.segment_ids_json, field_name="segment_ids_json", record_id=sample_id)
        for segment_value in segment_ids:
            segment_id = str(segment_value)
            segment = segment_lookup.get(segment_id)
            if segment is None:
                raise ValueError(f"window references missing segment: {sample_id}:{segment_id}")
            if segment.parent_sample_id != parent_sample_id or segment.label_name != label_name:
                raise ValueError(f"window/segment identity mismatch: {sample_id}:{segment_id}")
            if segment.end_seconds > window_end_seconds + 1.0e-9:
                raise ValueError(f"window references future segment: {sample_id}:{segment_id}")
            for frame_reference in segment.frame_references:
                if frame_reference.tensor_frame_index not in seen_tensor_indices:
                    seen_tensor_indices.add(frame_reference.tensor_frame_index)
                    frame_references.append(frame_reference)
        evidence_rows.append(
            WindowEvidence(
                sample_id=sample_id,
                parent_sample_id=parent_sample_id,
                split_name=str(row.split_name),
                label_name=label_name,
                window_start_seconds=_finite_float(row.window_start_seconds, field_name="window_start_seconds", record_id=sample_id),
                window_end_seconds=window_end_seconds,
                frame_references=tuple(sorted(frame_references, key=lambda item: (item.anchor_time_seconds, item.raw_frame_index))),
                segment_count=len(segment_ids),
            )
        )
    return evidence_rows


def _distribution(values: list[int] | list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    array = np.asarray(values, dtype=np.float64)
    integers_only = bool(np.all(np.equal(array, np.floor(array))))
    return {
        "count": int(array.size),
        "min": int(array.min()) if integers_only else float(array.min()),
        "max": int(array.max()) if integers_only else float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _summarize_split(rows: list[WindowEvidence]) -> dict[str, object]:
    parent_rows: dict[str, list[WindowEvidence]] = {}
    for row in rows:
        parent_rows.setdefault(row.parent_sample_id, []).append(row)
    overlap_ratios: list[float] = []
    novel_counts: list[int] = []
    update_intervals: list[float] = []
    zero_novel_update_count = 0
    predecessor_pair_count = 0
    nested_prefix_pair_count = 0
    historical_frame_reference_count = 0
    novel_duplicate_emission_count = 0
    union_reconstruction_failure_count = 0
    parent_label_conflict_count = 0
    parent_split_conflict_count = 0
    for parent_sample_id, parent_windows in parent_rows.items():
        labels = {window.label_name for window in parent_windows}
        splits = {window.split_name for window in parent_windows}
        parent_label_conflict_count += int(len(labels) != 1)
        parent_split_conflict_count += int(len(splits) != 1)
        ordered = sorted(
            parent_windows,
            key=lambda window: (window.window_end_seconds, window.window_start_seconds, window.sample_id),
        )
        ordered_ends = [window.window_end_seconds for window in ordered]
        if any(next_end <= previous_end for previous_end, next_end in zip(ordered_ends, ordered_ends[1:])):
            raise ValueError(f"non-strict window end order for {parent_sample_id}")
        seen_frames: set[int] = set()
        all_current_frames: set[int] = set()
        previous_frames: set[int] | None = None
        previous_end_seconds: float | None = None
        for window in ordered:
            current_frames = {reference.tensor_frame_index for reference in window.frame_references}
            all_current_frames.update(current_frames)
            novel_frames = current_frames.difference(seen_frames)
            historical_frame_reference_count += len(current_frames.intersection(seen_frames))
            novel_duplicate_emission_count += len(novel_frames.intersection(seen_frames))
            seen_frames.update(novel_frames)
            novel_counts.append(len(novel_frames))
            if previous_frames is not None:
                predecessor_pair_count += 1
                overlap_ratios.append(float(len(current_frames.intersection(previous_frames)) / len(current_frames)) if current_frames else 0.0)
                nested_prefix_pair_count += int(previous_frames.issubset(current_frames))
                if previous_end_seconds is not None:
                    update_intervals.append(window.window_end_seconds - previous_end_seconds)
                if not novel_frames:
                    zero_novel_update_count += 1
            previous_frames = current_frames
            previous_end_seconds = window.window_end_seconds
        union_reconstruction_failure_count += int(seen_frames != all_current_frames)
    return {
        "window_count": int(len(rows)),
        "parent_count": int(len(parent_rows)),
        "label_counts": dict(sorted(Counter(row.label_name for row in rows).items())),
        "predecessor_pair_count": int(predecessor_pair_count),
        "nested_prefix_pair_count": int(nested_prefix_pair_count),
        "nested_prefix_pair_ratio": float(nested_prefix_pair_count / predecessor_pair_count) if predecessor_pair_count else 0.0,
        "adjacent_overlap_ratio": _distribution(overlap_ratios),
        "novel_frame_count": _distribution(novel_counts),
        "update_interval_seconds": _distribution(update_intervals),
        "zero_novel_update_count": int(zero_novel_update_count),
        "historical_frame_reference_count": int(historical_frame_reference_count),
        "novel_duplicate_emission_count": int(novel_duplicate_emission_count),
        "union_reconstruction_failure_count": int(union_reconstruction_failure_count),
        "parent_label_conflict_count": int(parent_label_conflict_count),
        "parent_split_conflict_count": int(parent_split_conflict_count),
    }


def audit_novel_evidence_contract(
    project_root: Path,
    *,
    index_name: str,
    window_map_name: str,
    manifest_name: str,
    frame_index_name: str,
    output_name: str,
) -> dict[str, object]:
    evidence_rows, segment_count, physical_frame_reference_count = load_window_evidence(
        project_root,
        index_name=index_name,
        window_map_name=window_map_name,
        manifest_name=manifest_name,
        frame_index_name=frame_index_name,
    )
    split_reports = {
        split_name: _summarize_split([row for row in evidence_rows if row.split_name == split_name])
        for split_name in ("train", "val")
    }
    payload = {
        "scope": "read-only physical-v2 novel-evidence contract audit; no index rebuild, tensor rebuild, training, fusion, threshold tuning, or test evaluation",
        "inputs": {
            "index": index_name,
            "window_map": window_map_name,
            "manifest": manifest_name,
            "frame_index": frame_index_name,
        },
        "eligible_rd_only_window_count": int(len(evidence_rows)),
        "segment_count": segment_count,
        "physical_frame_reference_count": physical_frame_reference_count,
        "split_reports": split_reports,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path = project_root / "tmp" / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_window_evidence(
    project_root: Path,
    *,
    index_name: str,
    window_map_name: str,
    manifest_name: str,
    frame_index_name: str,
) -> tuple[list[WindowEvidence], int, int]:
    processed_root = project_root / "data" / "processed"
    index = pd.read_csv(processed_root / index_name, low_memory=False)
    window_map = pd.read_csv(processed_root / window_map_name, low_memory=False)
    manifest = pd.read_csv(processed_root / manifest_name, low_memory=False)
    frame_index = pd.read_csv(processed_root / frame_index_name, low_memory=False)
    frame_lookup = _build_frame_lookup(frame_index)
    segment_lookup = _build_segment_lookup(manifest, frame_lookup)
    evidence_rows = _build_window_evidence(index, window_map, segment_lookup)
    return evidence_rows, int(len(segment_lookup)), int(len(frame_lookup))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-overlapping physical-v2 RD evidence updates.")
    parser.add_argument("--index-name", default="expert1_realtime_causal_window_index_v7.csv")
    parser.add_argument("--window-map-name", default="expert1_realtime_causal_window_cq08_physical_anchor_map_v4.csv")
    parser.add_argument("--manifest-name", default="expert1_cq08_rd_physical_anchor_manifest_v4.csv")
    parser.add_argument("--frame-index-name", default="cq08_rd_physical_canonical_frame_index_v2.csv")
    parser.add_argument("--output-name", default="20260727-stage009p-novel-evidence-contract-audit.json")
    args = parser.parse_args()
    payload = audit_novel_evidence_contract(
        Path.cwd(),
        index_name=args.index_name,
        window_map_name=args.window_map_name,
        manifest_name=args.manifest_name,
        frame_index_name=args.frame_index_name,
        output_name=args.output_name,
    )
    print(json.dumps({"output": args.output_name, "split_reports": payload["split_reports"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
