from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from src.core.data.rd import EncodedRDSequence, encode_multimodal_rd
from src.process.build_expert1_rd_segment_store import MANIFEST_COLUMNS


ANCHOR_PREPROCESS_VERSION = "expert1_cq08_point_anchor_rd_v3_1_common_range_time"
ANCHOR_SYNC_STATUS = "track_point_anchor"


def _safe_name(value: object) -> str:
    return str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _track_id(sample_id: object) -> int:
    return int(str(sample_id).rsplit("|", 1)[-1])


def _unwrap_scalar(value: object) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError("expected scalar MAT metadata")
    return int(array[0])


def _mat_point_index(path: Path) -> int:
    payload = loadmat(path, variable_names=["index_trackPointNo"], squeeze_me=True, struct_as_record=False)
    return _unwrap_scalar(payload["index_trackPointNo"])


def _mat_observation_seconds(path: Path) -> float:
    payload = loadmat(path, variable_names=["debug_info"], squeeze_me=True, struct_as_record=False)
    debug = payload.get("debug_info")
    if debug is None:
        return 0.0
    try:
        return float(getattr(debug, "cpi_count")) * float(getattr(debug, "prt_num")) * float(getattr(debug, "prt_s"))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _mat_rd_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(
        path,
        variable_names=["data_proc_MTD_result_db", "data_proc_MTD_result"],
        squeeze_me=True,
        struct_as_record=False,
    )
    amplitude = np.asarray(payload["data_proc_MTD_result_db"], dtype=np.float32)
    amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
    if "data_proc_MTD_result" in payload:
        phase = np.angle(np.asarray(payload["data_proc_MTD_result"], dtype=np.complex128)).astype(np.float32)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        phase = np.zeros_like(amplitude, dtype=np.float32)
    return amplitude, phase


def _anchor_uncertainty_seconds(track_times: np.ndarray, point_index: int) -> float:
    position = point_index - 1
    if position < 0 or position >= len(track_times):
        return 0.0
    current = float(track_times[position])
    gaps: list[float] = []
    if position > 0:
        gaps.append(max(0.0, current - float(track_times[position - 1])))
    if position + 1 < len(track_times):
        gaps.append(max(0.0, float(track_times[position + 1]) - current))
    return 0.5 * max(gaps) if gaps else 0.0


def _encoded_segment(amplitudes: list[np.ndarray], phases: list[np.ndarray]) -> EncodedRDSequence:
    if not amplitudes:
        raise ValueError("anchor segment has no RD frames")
    target_shape = tuple(int(value) for value in amplitudes[0].shape)
    selected = [(amp, phase) for amp, phase in zip(amplitudes, phases) if tuple(amp.shape) == target_shape and tuple(phase.shape) == target_shape]
    if not selected:
        raise ValueError("anchor segment has no shape-compatible RD frames")
    amplitude = np.stack([item[0] for item in selected], axis=0)
    phase = np.stack([item[1] for item in selected], axis=0)
    return encode_multimodal_rd({"amplitude": amplitude, "phase": phase, "time_raw": np.arange(len(selected), dtype=np.float32)}, legacy=True)


def _as_bool(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes"}


def _group_kept_tensor_frames(
    frame_rows: pd.DataFrame,
    track_point_count: int,
) -> tuple[dict[int, list[tuple[int, int, float]]], int]:
    grouped: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    dropped = 0
    for row in frame_rows.to_dict(orient="records"):
        if not _as_bool(row.get("is_tensor_frame_kept")):
            dropped += 1
            continue
        point_index = int(row.get("track_point_index", 0))
        tensor_frame_index = int(row.get("tensor_frame_index", -1))
        if point_index < 1 or point_index > track_point_count or tensor_frame_index < 0:
            continue
        raw_frame_index = int(row.get("raw_frame_index", -1))
        observation = float(row.get("rd_observation_duration_seconds", 0.0) or 0.0)
        grouped[point_index].append((raw_frame_index, tensor_frame_index, observation))
    return grouped, dropped


def build_expert1_cq08_anchor_segment_store(
    project_root: Path,
    *,
    index_name: str = "expert1_track_index.csv",
    frame_index_name: str = "cq08_rd_frame_index_v1.csv",
    output_name: str = "expert1_cq08_rd_anchor_store_v2",
    manifest_name: str = "expert1_cq08_rd_anchor_manifest_v2.csv",
    preprocess_version: str = ANCHOR_PREPROCESS_VERSION,
    max_parents: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build a CQ08 anchor manifest using the explicit frame-index tensor mapping."""
    processed_root = project_root / "data" / "processed"
    index = pd.read_csv(processed_root / index_name, low_memory=False)
    parents = index[index["source_name"].astype(str).eq("cq08_track")].copy()
    frame_index_path = processed_root / frame_index_name
    frame_index = pd.read_csv(frame_index_path, low_memory=False)
    required_columns = {
        "parent_sample_id", "is_tensor_frame_kept", "tensor_frame_index",
        "track_point_index", "rd_observation_duration_seconds",
    }
    missing_columns = required_columns - set(frame_index.columns)
    if missing_columns:
        raise ValueError(f"frame index missing required columns: {sorted(missing_columns)}")
    indexed_parent_ids = set(frame_index["parent_sample_id"].astype(str))
    parents = parents[parents["sample_id"].astype(str).isin(indexed_parent_ids)]
    parents = parents.drop_duplicates(subset=["sample_id"])
    parents["_track_id_sort"] = parents["sample_id"].map(_track_id)
    parents = parents.sort_values(["_track_id_sort", "sample_id"]).drop(columns=["_track_id_sort"])
    if max_parents > 0:
        parents = parents.head(int(max_parents))
    frame_rows_by_parent = {
        str(parent_id): group.sort_values("raw_frame_index").reset_index(drop=True)
        for parent_id, group in frame_index.groupby("parent_sample_id", sort=False)
    }
    manifest_path = processed_root / manifest_name
    summary_path = manifest_path.with_suffix(".summary.json")
    if (manifest_path.exists() or summary_path.exists()) and not overwrite:
        raise FileExistsError(f"anchor outputs already exist: {manifest_path}; pass overwrite=True")

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    full_anchor_parent_count = 0
    partial_anchor_parent_count = 0
    source_frame_count = 0
    dropped_shape_frame_count = 0

    for _, parent in parents.iterrows():
        sample_id = str(parent["sample_id"])
        parent_frame_rows = frame_rows_by_parent.get(sample_id)
        if parent_frame_rows is None or parent_frame_rows.empty:
            errors.append({"parent_sample_id": sample_id, "error": "missing_frame_index_rows"})
            continue
        track = pd.read_csv(project_root / str(parent["csv_path"]))
        track_times = pd.to_numeric(track["time_seconds"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(track_times).all() or len(track_times) == 0:
            errors.append({"parent_sample_id": sample_id, "error": "invalid_track_times"})
            continue
        grouped_frames, parent_dropped = _group_kept_tensor_frames(parent_frame_rows, len(track_times))
        dropped_shape_frame_count += parent_dropped
        if not grouped_frames:
            errors.append({"parent_sample_id": sample_id, "error": "no_kept_tensor_frames"})
            continue
        if len(grouped_frames) == len(track_times) and min(grouped_frames) == 1 and max(grouped_frames) == len(track_times):
            full_anchor_parent_count += 1
        else:
            partial_anchor_parent_count += 1
        source_name = str(parent["source_name"])
        label_name = str(parent["label_name"])
        tensor_paths = [
            str(value)
            for value in parent_frame_rows.get("rd_tensor_path", pd.Series(dtype=object)).dropna().unique()
            if str(value).strip()
        ]
        if len(tensor_paths) > 1:
            errors.append({"parent_sample_id": sample_id, "error": "multiple_frame_index_tensor_paths"})
            continue
        rd_tensor_path = tensor_paths[0] if tensor_paths else str(parent["rd_tensor_path"])
        for segment_index, point_index in enumerate(sorted(grouped_frames)):
            frames = grouped_frames[point_index]
            raw_mat_indices = sorted(set(int(item[0]) for item in frames))
            tensor_indices = sorted(set(int(item[1]) for item in frames))
            anchor_time = float(track_times[point_index - 1])
            observation = float(np.median([item[2] for item in frames])) if frames else 0.0
            uncertainty = _anchor_uncertainty_seconds(track_times, point_index)
            flags = [ANCHOR_SYNC_STATUS, "point_association_verified", "tensor_reference_only", "tensor_index_verified"]
            if uncertainty > 0.0:
                flags.append("time_anchor_uncertain")
            if len(frames) <= 1:
                flags.append("single_rd_frame")
            quality_flags = ";".join(flags)
            rows.append(
                {
                    "segment_id": f"{sample_id}|anchor{point_index:04d}",
                    "parent_sample_id": sample_id,
                    "track_id": sample_id,
                    "source_name": source_name,
                    "label_name": label_name,
                    "segment_start_seconds": anchor_time,
                    "segment_end_seconds": anchor_time,
                    "segment_duration_seconds": observation,
                    "segment_frame_count": len(frames),
                    "rd_feature_path": "",
                    "rd_patch_path": "",
                    "rd_deep_patch_path": "",
                    "shard_key": "",
                    "segment_index": segment_index,
                    "rd_feature_key": "rd_features",
                    "rd_patch_key": "rd_patch_features",
                    "rd_deep_patch_key": "rd_deep_patch_features",
                    "rd_coverage_ratio": 0.0,
                    "rd_frame_start_index": min(tensor_indices),
                    "rd_frame_end_index": max(tensor_indices),
                    "rd_time_start_seconds": anchor_time,
                    "rd_time_end_seconds": anchor_time,
                    "rd_time_coverage_ratio": 0.0,
                    "rd_quality_flags": quality_flags,
                    "rd_time_sync_status": ANCHOR_SYNC_STATUS,
                    "rd_preprocess_version": preprocess_version,
                    "rd_is_multimodal": True,
                    "rd_target_association_id": sample_id,
                    "tr_target_association_id": sample_id,
                    "target_association_verified": True,
                    "track_point_index": point_index,
                    "track_point_time_seconds": anchor_time,
                    "rd_anchor_uncertainty_seconds": uncertainty,
                    "rd_observation_duration_seconds": observation,
                    "rd_diagnostic_eligible": True,
                    "rd_tensor_path": rd_tensor_path,
                    "rd_source_frame_indices_json": json.dumps(raw_mat_indices),
                    "rd_tensor_frame_indices_json": json.dumps(tensor_indices),
                }
            )
            source_frame_count += len(tensor_indices)
            status_counts[ANCHOR_SYNC_STATUS] += 1
            quality_counts.update(flags)
    ordered_columns = list(MANIFEST_COLUMNS) + [
        "rd_target_association_id", "tr_target_association_id", "target_association_verified",
        "track_point_index", "track_point_time_seconds", "rd_anchor_uncertainty_seconds",
        "rd_observation_duration_seconds", "rd_diagnostic_eligible", "rd_tensor_path",
        "rd_source_frame_indices_json", "rd_tensor_frame_indices_json",
    ]
    manifest = pd.DataFrame(rows).reindex(columns=ordered_columns)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    summary = {
        "schema_version": preprocess_version,
        "index_name": index_name,
        "frame_index_name": frame_index_name,
        "output_name": output_name,
        "manifest_name": manifest_name,
        "candidate_parent_count": int(len(parents)),
        "written_parent_count": int(manifest["parent_sample_id"].nunique()) if not manifest.empty else 0,
        "segment_count": int(len(manifest)),
        "source_tensor_frame_reference_count": int(source_frame_count),
        "dropped_shape_frame_count": int(dropped_shape_frame_count),
        "full_anchor_parent_count": int(full_anchor_parent_count),
        "partial_anchor_parent_count": int(partial_anchor_parent_count),
        "rd_time_sync_status_counts": dict(status_counts),
        "rd_quality_flag_counts": dict(quality_counts),
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "errors": errors[:100],
        "fusion_policy": "track_point_anchor is diagnostic-only and must not be passed to fusion without independent timing evidence.",
        "storage_policy": "manifest references existing processed CQ08 tensors with verified compact tensor indices; no duplicate encoded RD shards are written.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build CQ08 point-anchored RD segment store without claiming exact time synchronization.")
    parser.add_argument("--index-name", default="expert1_track_index.csv")
    parser.add_argument("--frame-index-name", default="cq08_rd_frame_index_v1.csv")
    parser.add_argument("--output-name", default="expert1_cq08_rd_anchor_store_v2")
    parser.add_argument("--manifest-name", default="expert1_cq08_rd_anchor_manifest_v2.csv")
    parser.add_argument("--preprocess-version", default=ANCHOR_PREPROCESS_VERSION)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(__file__).resolve().parents[2]
    print(json.dumps(build_expert1_cq08_anchor_segment_store(
        project_root,
        index_name=args.index_name,
        frame_index_name=args.frame_index_name,
        output_name=args.output_name,
        manifest_name=args.manifest_name,
        preprocess_version=args.preprocess_version,
        max_parents=args.max_parents,
        overwrite=args.overwrite,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
