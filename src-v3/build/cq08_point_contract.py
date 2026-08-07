from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "RADAR-LN3-V0"
LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
TR_COLUMNS = (
    "azimuth_deg",
    "range_m",
    "radial_speed_mps",
    "elevation_deg",
    "height_m",
    "course_deg",
    "speed_mps",
    "amplitude_db",
    "snr_db",
    "latitude",
    "longitude",
)
OUTPUT_STEM = f"{VERSION}-cq08-tr-rd"
INDEX_NAME = f"{OUTPUT_STEM}-point-index.csv"
MANIFEST_NAME = f"{OUTPUT_STEM}-point-manifest.json"
SPLIT_NAME = f"{OUTPUT_STEM}-split.json"
CONTRACT_NAME = f"{OUTPUT_STEM}-data-contract.json"
SUMMARY_NAME = f"{OUTPUT_STEM}-summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _path_from_project(project_root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute():
        return path
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("data/"):
        return project_root / path
    return project_root / path


def _relative_project_path(project_root: Path, path: str | Path) -> str:
    candidate = _path_from_project(project_root, path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _split_map(project_root: Path) -> dict[str, str]:
    split_path = project_root / "data" / "manifests" / "cq08_grouped_split.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for group_id in payload.get(f"{split_name}_group_ids", []):
            key = str(group_id).strip()
            if key in result:
                raise ValueError(f"CQ08 split group appears more than once: {key}")
            result[key] = split_name
    return result


def _load_manifest(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not bool(record.get("is_usable")):
            continue
        track_id = str(record["track_id"])
        if track_id in result:
            raise ValueError(f"duplicate usable CQ08 track_id in manifest: {track_id}")
        result[track_id] = record
    return result


def _load_track_index(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "cq08_track_index.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "sample_id",
        "source_name",
        "label_name",
        "record_count",
        "csv_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CQ08 track index missing columns: {missing}")
    frame["track_id"] = frame["sample_id"].astype(str).str.extract(r"(?:^|\|)(\d+)$", expand=False)
    if frame["track_id"].isna().any():
        raise ValueError("CQ08 track index contains an unparseable sample_id")
    frame["track_id"] = frame["track_id"].astype(str)
    frame = frame[frame["source_name"].astype(str).eq("cq08_track")].copy()
    if frame.empty:
        raise ValueError("CQ08 track index has no cq08_track rows")
    if frame["track_id"].duplicated().any():
        raise ValueError("CQ08 track index has duplicate track_id rows")
    return frame


def _load_rd_index(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "cq08_rd_physical_canonical_frame_index_v3_1.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "parent_sample_id",
        "track_id",
        "source_name",
        "label_name",
        "raw_frame_index",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "canonical_rd_shape",
        "track_point_index",
        "rd_frame_time_seconds",
        "rd_tensor_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CQ08 RD index missing columns: {missing}")
    frame = frame[frame["source_name"].astype(str).eq("cq08_track")].copy()
    frame = frame[frame["is_tensor_frame_kept"].astype(str).str.lower().isin({"true", "1"})].copy()
    if frame.empty:
        raise ValueError("CQ08 RD index has no kept CQ08 rows")
    frame["track_id"] = frame["track_id"].astype(str)
    frame["parent_sample_id"] = frame["parent_sample_id"].astype(str)
    for column in ("raw_frame_index", "tensor_frame_index", "track_point_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["rd_frame_time_seconds"] = pd.to_numeric(frame["rd_frame_time_seconds"], errors="raise").astype(float)
    if frame["rd_tensor_path"].isna().any() or frame["rd_tensor_path"].astype(str).str.strip().eq("").any():
        raise ValueError("CQ08 RD index contains an empty tensor path")
    if frame["canonical_rd_shape"].astype(str).ne("21x384").any():
        raise ValueError("V0 requires the verified CQ08 canonical RD shape 21x384")
    return frame.sort_values(["track_id", "raw_frame_index"], kind="stable").reset_index(drop=True)


def _bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _track_features(frame: pd.DataFrame, row: pd.Series) -> tuple[dict[str, Any], list[str]]:
    output: dict[str, Any] = {}
    missing: list[str] = []
    for column in TR_COLUMNS:
        value = row[column] if column in row else np.nan
        if not _finite(value):
            output[column] = None
            missing.append(column)
        else:
            output[column] = float(value)
    return output, missing


def _group_rd(rd_frame: pd.DataFrame) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[str, str], dict[str, int]]:
    grouped: dict[tuple[str, int], pd.DataFrame] = {}
    parent_order: dict[str, str] = {}
    parent_inversion_count: dict[str, int] = {}
    for track_id, parent_frame in rd_frame.groupby("track_id", sort=False):
        parent = str(track_id)
        point_values = parent_frame["track_point_index"].to_numpy(dtype=np.int64)
        inversion_count = int(np.sum(np.diff(point_values) < 0))
        parent_inversion_count[parent] = inversion_count
        parent_order[parent] = "non_monotonic" if inversion_count else "monotonic"
        for point_index, point_frame in parent_frame.groupby("track_point_index", sort=True):
            grouped[(parent, int(point_index))] = point_frame.sort_values("raw_frame_index", kind="stable").reset_index(drop=True)
    return grouped, parent_order, parent_inversion_count


def _validate_tensor_reference(
    project_root: Path,
    path_value: str,
    indices: list[int],
    cache: dict[str, tuple[tuple[int, ...], np.ndarray, str]],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    path = _path_from_project(project_root, path_value)
    if not path.exists():
        raise FileNotFoundError(f"CQ08 RD tensor is missing: {path}")
    cached = cache.get(str(path))
    if cached is None:
        with np.load(path, allow_pickle=False) as payload:
            required = {"amplitude", "phase", "time_seconds", "velocity_mps", "range_offset_m"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"CQ08 RD tensor missing keys: {path}: {missing}")
            amplitude = np.asarray(payload["amplitude"])
            phase = np.asarray(payload["phase"])
            times = np.asarray(payload["time_seconds"])
            if amplitude.ndim != 3 or tuple(amplitude.shape[1:]) != (21, 384):
                raise ValueError(f"CQ08 RD tensor has invalid shape: {path}: {amplitude.shape}")
            if phase.shape != amplitude.shape or times.shape != (amplitude.shape[0],):
                raise ValueError(f"CQ08 RD tensor arrays have inconsistent shapes: {path}")
            if not np.isfinite(times).all():
                raise ValueError(f"CQ08 RD tensor local times are non-finite: {path}")
            time_status = "non_monotonic" if times.size > 1 and bool((np.diff(times) < -1.0e-8).any()) else "monotonic"
            cached = (tuple(int(value) for value in amplitude.shape), times.astype(np.float32, copy=False), time_status)
            cache[str(path)] = cached
    tensor_shape, times, time_status = cached
    if indices and (min(indices) < 0 or max(indices) >= tensor_shape[0]):
        raise ValueError(f"CQ08 tensor_frame_index exceeds tensor bounds: {path}")
    selected_times = times[np.asarray(indices, dtype=np.int64)] if indices else np.empty((0,), dtype=np.float32)
    return tensor_shape, (time_status,)


def build(project_root: Path) -> dict[str, Any]:
    data_root = project_root / "data" / "processed"
    output_index = data_root / INDEX_NAME
    output_manifest = data_root / MANIFEST_NAME
    output_split = data_root / SPLIT_NAME
    output_contract = data_root / CONTRACT_NAME
    output_summary = data_root / SUMMARY_NAME

    track_index = _load_track_index(project_root)
    rd_index = _load_rd_index(project_root)
    split_by_track = _split_map(project_root)
    source_manifest = _load_manifest(project_root)
    grouped, parent_order, parent_inversion_count = _group_rd(rd_index)

    rd_parent_ids = set(rd_index["parent_sample_id"].astype(str))
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    rd_status_counts: Counter[str] = Counter()
    parent_coverage_counts: Counter[str] = Counter()
    parent_rows: list[dict[str, Any]] = []
    tensor_reference_cache: dict[str, tuple[tuple[int, ...], np.ndarray, str]] = {}
    validation = {
        "track_rows": int(len(track_index)),
        "point_rows": 0,
        "rd_index_rows": int(len(rd_index)),
        "missing_track_files": [],
        "missing_rd_tensors": [],
        "track_record_count_mismatch": [],
        "label_mismatch": [],
        "split_missing": [],
        "point_index_mismatch": [],
        "tensor_shape_mismatch": [],
        "tr_time_anomaly_parents": [],
    }

    for track_record in track_index.sort_values("track_id", key=lambda s: s.astype(int)).itertuples(index=False):
        track_id = str(track_record.track_id)
        parent_sample_id = str(track_record.sample_id)
        split_name = split_by_track.get(track_id)
        if split_name is None:
            validation["split_missing"].append(track_id)
            continue
        track_path = _path_from_project(project_root, str(track_record.csv_path))
        if not track_path.exists():
            validation["missing_track_files"].append(str(track_path))
            continue
        track_frame = pd.read_csv(track_path, low_memory=False)
        expected_count = int(track_record.record_count)
        if len(track_frame) != expected_count:
            validation["track_record_count_mismatch"].append({"track_id": track_id, "expected": expected_count, "actual": int(len(track_frame))})
        required_track_columns = {"frame_index", "time_seconds", "label_name", *TR_COLUMNS}
        missing_columns = sorted(required_track_columns - set(track_frame.columns))
        if missing_columns:
            raise ValueError(f"CQ08 track file missing required columns: {track_path}: {missing_columns}")
        track_frame = track_frame.sort_values("frame_index", kind="stable").reset_index(drop=True)
        if not np.array_equal(track_frame["frame_index"].to_numpy(dtype=np.int64), np.arange(len(track_frame), dtype=np.int64)):
            raise ValueError(f"CQ08 track frame_index is not contiguous: {track_path}")
        times = pd.to_numeric(track_frame["time_seconds"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError(f"CQ08 track time is non-finite: {track_path}")
        tr_time_inversion_count = int(np.sum(np.diff(times) < -1.0e-8))
        tr_time_order_status = "non_monotonic" if tr_time_inversion_count else "monotonic"
        if tr_time_inversion_count:
            validation["tr_time_anomaly_parents"].append({"track_id": track_id, "inversion_count": tr_time_inversion_count})
        label_values = set(track_frame["label_name"].astype(str))
        if label_values != {str(track_record.label_name)}:
            validation["label_mismatch"].append({"track_id": track_id, "index_label": str(track_record.label_name), "file_labels": sorted(label_values)})
        manifest_record = source_manifest.get(track_id)
        if manifest_record is None:
            raise ValueError(f"usable CQ08 unified manifest missing track_id: {track_id}")
        rd_parent_key = parent_sample_id
        parent_point_indices = {
            int(point_index)
            for (group_track_id, point_index) in grouped
            if str(group_track_id) == track_id
        }
        expected_point_indices = set(range(1, len(track_frame) + 1))
        coverage_status = "full" if parent_point_indices == expected_point_indices else "partial"
        parent_coverage_counts[coverage_status] += 1
        source_counts["CQ08"] += 1
        label_counts[str(track_record.label_name)] += 1
        split_counts[split_name] += 1
        parent_rows.append({
            "parent_sample_id": parent_sample_id,
            "track_id": track_id,
            "label_name": str(track_record.label_name),
            "split_name": split_name,
            "tr_point_count": int(len(track_frame)),
            "rd_point_count": int(len(parent_point_indices)),
            "rd_parent_coverage_status": coverage_status,
            "rd_parent_order_status": parent_order.get(track_id, "no_rd"),
            "rd_parent_inversion_count": int(parent_inversion_count.get(track_id, 0)),
            "tr_time_order_status": tr_time_order_status,
            "tr_time_inversion_count": tr_time_inversion_count,
        })
        for row_position, (_, tr_row) in enumerate(track_frame.iterrows()):
            point_index = int(tr_row["frame_index"]) + 1
            tr_time = float(times[row_position])
            tr_dt = 0.0 if row_position == 0 else float(times[row_position] - times[row_position - 1])
            tr_values, tr_missing = _track_features(track_frame, tr_row)
            rd_rows = grouped.get((track_id, point_index))
            rd_present = rd_rows is not None and not rd_rows.empty
            rd_frame_indices: list[int] = []
            rd_raw_indices: list[int] = []
            rd_local_times: list[float] = []
            rd_tensor_path = ""
            rd_shape = ""
            rd_local_status = "missing"
            if rd_present:
                rd_frame_indices = [int(value) for value in rd_rows["tensor_frame_index"].tolist()]
                rd_raw_indices = [int(value) for value in rd_rows["raw_frame_index"].tolist()]
                rd_local_times = [float(value) for value in rd_rows["rd_frame_time_seconds"].tolist()]
                rd_tensor_path = _relative_project_path(project_root, str(rd_rows.iloc[0]["rd_tensor_path"]))
                if rd_rows["rd_tensor_path"].astype(str).nunique() != 1:
                    raise ValueError(f"CQ08 point references multiple tensor paths: {parent_sample_id} point {point_index}")
                rd_shape = str(rd_rows.iloc[0]["canonical_rd_shape"])
                try:
                    tensor_shape, (rd_local_status,) = _validate_tensor_reference(project_root, rd_tensor_path, rd_frame_indices, tensor_reference_cache)
                    if tensor_shape[1:] != (21, 384):
                        validation["tensor_shape_mismatch"].append({"track_id": track_id, "point_index": point_index, "shape": tensor_shape})
                except (FileNotFoundError, ValueError) as error:
                    validation["missing_rd_tensors"].append(str(error))
                    raise
                if len(rd_local_times) > 1 and bool((np.diff(np.asarray(rd_local_times)) < -1.0e-8).any()):
                    rd_local_status = "non_monotonic"
                rd_status = "present"
            else:
                rd_status = "missing"
            rd_status_counts[rd_status] += 1
            sample_id = f"{parent_sample_id}#point{point_index:04d}"
            row: dict[str, Any] = {
                "schema_version": VERSION,
                "sample_id": sample_id,
                "parent_sample_id": parent_sample_id,
                "track_id": track_id,
                "source_name": "CQ08",
                "split_group_id": track_id,
                "split_name": split_name,
                "label_name": str(track_record.label_name),
                "label_id": int(LABELS.index(str(track_record.label_name))),
                "causal_event_index": row_position,
                "track_point_index": point_index,
                "tr_frame_index": int(tr_row["frame_index"]),
                "tr_time_seconds": tr_time,
                "tr_dt_seconds": tr_dt,
                "tr_dt_is_zero": bool(abs(tr_dt) <= 1.0e-9),
                "tr_dt_is_negative": bool(tr_dt < -1.0e-9),
                "tr_time_order_status": tr_time_order_status,
                "tr_time_inversion_count": tr_time_inversion_count,
                "tr_time_semantics": "track_relative_wall_clock_seconds",
                "tr_missing_count": int(len(tr_missing)),
                "tr_missing_fields_json": _json(tr_missing),
                "tr_feature_values_json": _json(tr_values),
                "rd_present": bool(rd_present),
                "rd_frame_count": int(len(rd_frame_indices)),
                "rd_tensor_path": rd_tensor_path,
                "rd_tensor_frame_indices_json": _json(rd_frame_indices),
                "rd_raw_frame_indices_json": _json(rd_raw_indices),
                "rd_canonical_shape": rd_shape or "21x384",
                "rd_local_time_start_seconds": float(min(rd_local_times)) if rd_local_times else 0.0,
                "rd_local_time_end_seconds": float(max(rd_local_times)) if rd_local_times else 0.0,
                "rd_local_duration_seconds": float(max(rd_local_times) - min(rd_local_times)) if rd_local_times else 0.0,
                "rd_time_semantics": "cpi_relative_elapsed_not_tr_wall_clock",
                "rd_local_time_order_status": rd_local_status,
                "rd_point_coverage_status": "present" if rd_present else "missing",
                "rd_parent_coverage_status": coverage_status,
                "rd_parent_order_status": parent_order.get(track_id, "no_rd"),
                "rd_parent_inversion_count": int(parent_inversion_count.get(track_id, 0)),
                "rd_quality_missing": not rd_present,
                "rd_quality_partial_parent": coverage_status == "partial",
                "rd_quality_parent_order_anomaly": parent_order.get(track_id) == "non_monotonic",
                "track_csv_path": _relative_project_path(project_root, str(track_record.csv_path)),
                "cq08_track_file": str(manifest_record.get("track_file", "")),
                "cq08_point_file": str(manifest_record.get("point_file", "")),
                "cq08_raw_echo_file": str(manifest_record.get("raw_echo_file", "")),
                "cq08_rd_mat_count": int(len(manifest_record.get("rd_mat_files", []))),
            }
            rows.append(row)
            validation["point_rows"] += 1

    if validation["missing_track_files"] or validation["track_record_count_mismatch"] or validation["label_mismatch"] or validation["split_missing"] or validation["tensor_shape_mismatch"]:
        raise ValueError(f"CQ08 point contract validation failed before write: {validation}")
    if set(source_counts) != {"CQ08"} or set(label_counts) != set(LABELS):
        raise ValueError(f"CQ08 V0 source/label purity failed: sources={source_counts}, labels={label_counts}")
    if len(parent_rows) != 1549:
        raise ValueError(f"CQ08 V0 expected 1549 usable parents, got {len(parent_rows)}")
    if len(rows) != 29724:
        raise ValueError(f"CQ08 V0 expected 29724 TR points, got {len(rows)}")
    if len(rd_index) != 106026:
        raise ValueError(f"CQ08 V0 expected 106026 kept RD frames, got {len(rd_index)}")

    index_frame = pd.DataFrame(rows)
    index_frame = index_frame.sort_values(["track_id", "causal_event_index"], key=lambda column: column.astype(int) if column.name in {"track_id", "causal_event_index"} else column, kind="stable").reset_index(drop=True)
    index_frame.to_csv(output_index, index=False, encoding="utf-8", lineterminator="\n")

    split_payload = {
        "schema_version": VERSION,
        "source_name": "CQ08",
        "group_unit": "track_id",
        "frame_level_split_forbidden": True,
        "source_split_policy": "data/manifests/cq08_grouped_split.json",
        "seed": 42,
        "group_counts": {name: int(sum(1 for row in parent_rows if row["split_name"] == name)) for name in ("train", "val", "test")},
        "group_ids": {name: sorted([row["track_id"] for row in parent_rows if row["split_name"] == name], key=int) for name in ("train", "val", "test")},
    }
    output_split.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_payload = {
        "schema_version": VERSION,
        "manifest_type": "cq08_point_level_tr_rd_contract",
        "source_name": "CQ08",
        "source_roots": [
            "OriData/23/CQ-08/航迹",
            "OriData/23/CQ-08/点迹",
            "OriData/23/CQ-08/MAT",
            "OriData/23/CQ-08/原始回波",
        ],
        "index_path": f"data/processed/{INDEX_NAME}",
        "split_path": f"data/processed/{SPLIT_NAME}",
        "canonical_rd_index_path": "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv",
        "canonical_rd_tensor_root": "data/processed/cq08_rd_physical_canonical_v3_1",
        "parent_count": len(parent_rows),
        "point_count": len(rows),
        "rd_frame_count": len(rd_index),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "rd_point_status_counts": dict(sorted(rd_status_counts.items())),
        "rd_parent_coverage_counts": dict(sorted(parent_coverage_counts.items())),
        "rd_parent_order_counts": dict(sorted(Counter(row["rd_parent_order_status"] for row in parent_rows).items())),
        "rd_parent_inversion_parent_count": int(sum(value > 0 for value in parent_inversion_count.values())),
        "rd_parent_inversion_event_count": int(sum(parent_inversion_count.values())),
        "tr_dt_zero_count": int(index_frame["tr_dt_is_zero"].sum()),
        "tr_dt_min_seconds": float(index_frame["tr_dt_seconds"].min()),
        "tr_dt_max_seconds": float(index_frame["tr_dt_seconds"].max()),
        "tr_dt_nonzero_count": int((~index_frame["tr_dt_is_zero"]).sum()),
        "tr_dt_negative_count": int(index_frame["tr_dt_is_negative"].sum()),
        "tr_time_anomaly_parent_count": int(len(validation["tr_time_anomaly_parents"])),
        "index_sha256": _sha256(output_index),
        "split_sha256": _sha256(output_split),
        "old_files_reused_read_only": [
            "data/processed/cq08_track_index.csv",
            "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv",
            "data/processed/cq08_rd_physical_canonical_v3_1/*.npz",
            "data/manifests/cq08_unified_manifest.json",
            "data/manifests/cq08_grouped_split.json",
        ],
        "version_reason": "Initial RADAR-LN3 point-level TR/RD contract; no prior RADAR-LN3 version existed.",
    }
    output_manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    contract_payload = {
        "schema_version": VERSION,
        "contract_name": "cq08_point_level_tr_rd_causal_v1",
        "source_purity": {"allowed_sources": ["CQ08"], "excluded_sources": ["raw", "Excel", "RD-only"]},
        "event_unit": "one_TR_point",
        "sequence_axis": "causal_event_index_in_parent",
        "tr_wall_clock": {"field": "tr_time_seconds", "dt_field": "tr_dt_seconds", "semantics": "track_relative_wall_clock_seconds", "first_dt": "0.0", "zero_dt_is_valid": True},
        "rd_aggregation": {"group_key": ["parent_sample_id", "track_point_index"], "token_shape": "[rd_frame_count,21,384]", "rd_time_field": "rd_local_time_start_seconds/rd_local_time_end_seconds", "rd_time_semantics": "cpi_relative_elapsed_not_tr_wall_clock", "missing_policy": "explicit_mask_and_zero_frame_count"},
        "tr_features": {"raw_columns": list(TR_COLUMNS), "encoding": "implemented_in_src_v3.features; angles use sin/cos", "missing_policy": "value_impute_zero_after_normalization_plus_feature_missing_mask"},
        "rd_features": {"amplitude_key": "amplitude", "phase_key": "phase", "range_axis_key": "range_offset_m", "velocity_axis_key": "velocity_mps", "canonical_shape": [21,384], "normalization": "smoke config declares per-tensor finite standardization; train statistics not generated in this smoke"},
        "model_io": {"tr_point_features": "[B,L,Dtr]", "rd_frames": "[B,L,K,2,21,384]", "rd_frame_mask": "[B,L,K]", "tr_padding_mask": "[B,L]", "tr_dt_seconds": "[B,L]", "window_logits": "[B,5]", "stable_state": "external causal state machine"},
        "leakage_policy": ["no label/path/session field enters tensors", "split is parent/track grouped", "no future TR points are used", "RD CPI elapsed time is not used as TR wall clock"],
        "known_quality_flags": {"partial_parent_count": int(parent_coverage_counts.get("partial", 0)), "non_monotonic_rd_parent_count": int(sum(value > 0 for value in parent_inversion_count.values())), "non_monotonic_rd_event_count": int(sum(parent_inversion_count.values())), "non_monotonic_tr_parent_count": int(len(validation["tr_time_anomaly_parents"])), "negative_tr_dt_policy": "retain_raw_dt; model clamps and exposes dt_clamped_mask; default training/smoke excludes anomalous parents"},
        "index_path": f"data/processed/{INDEX_NAME}",
        "manifest_path": f"data/processed/{MANIFEST_NAME}",
        "split_path": f"data/processed/{SPLIT_NAME}",
        "version_reason": "Initial RADAR-LN3 contract version V0.",
    }
    output_contract.write_text(json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "schema_version": VERSION,
        "status": "generated",
        "validation": validation,
        "counts": {
            "parent_count": len(parent_rows),
            "point_count": len(rows),
            "rd_frame_count": len(rd_index),
            "source_counts": dict(source_counts),
            "label_counts": dict(label_counts),
            "split_counts": dict(split_counts),
            "rd_status_counts": dict(rd_status_counts),
            "parent_coverage_counts": dict(parent_coverage_counts),
            "parent_order_counts": dict(Counter(row["rd_parent_order_status"] for row in parent_rows)),
        },
        "outputs": {
            "index": f"data/processed/{INDEX_NAME}",
            "manifest": f"data/processed/{MANIFEST_NAME}",
            "split": f"data/processed/{SPLIT_NAME}",
            "contract": f"data/processed/{CONTRACT_NAME}",
        },
        "index_sha256": _sha256(output_index),
    }
    output_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated RADAR-LN3-V0 CQ08 point-level TR/RD contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = build(args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
