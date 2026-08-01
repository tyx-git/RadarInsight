from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


PHYSICAL_STORE_VERSION = "cq08_rd_physical_canonical_v3_1"
CANONICAL_RANGE_BINS = 21
CANONICAL_DOPPLER_BINS = 384
RANGE_OFFSET_START_M = -75.0
RANGE_OFFSET_END_M = 75.0
VELOCITY_START_MPS = -85.0
VELOCITY_END_MPS = 85.0


def _safe_name(value: object) -> str:
    return str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")


def _scalar(debug: object, name: str) -> float:
    values = np.asarray(getattr(debug, name)).reshape(-1)
    if values.size != 1:
        raise ValueError(f"debug_info.{name} is not scalar")
    return float(values[0])


def _unwrap_scalar(value: object) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError("expected scalar MAT metadata")
    return int(values[0])


def _observation_seconds(debug: object) -> float:
    try:
        return _scalar(debug, "cpi_count") * _scalar(debug, "prt_num") * _scalar(debug, "prt_s")
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _target_velocity_axis() -> np.ndarray:
    return np.linspace(VELOCITY_START_MPS, VELOCITY_END_MPS, CANONICAL_DOPPLER_BINS, dtype=np.float32)


def _target_range_offset_axis() -> np.ndarray:
    return np.linspace(RANGE_OFFSET_START_M, RANGE_OFFSET_END_M, CANONICAL_RANGE_BINS, dtype=np.float32)


def _source_velocity_axis(debug: object, doppler_bins: int) -> tuple[np.ndarray, float, float]:
    vr_start = _scalar(debug, "vr_start")
    vr_end = _scalar(debug, "vr_end")
    if doppler_bins <= 1 or not np.isfinite(vr_start) or not np.isfinite(vr_end) or vr_end <= vr_start:
        raise ValueError("invalid debug_info physical velocity grid")
    return np.linspace(vr_start, vr_end, doppler_bins, dtype=np.float32), vr_start, vr_end


def _source_range_offset_axis(range_plot: np.ndarray, debug: object, range_bins: int) -> tuple[np.ndarray, float]:
    axis = np.asarray(range_plot, dtype=np.float32).reshape(-1)
    if axis.size != range_bins or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0.0):
        raise ValueError("invalid Range_plot physical range grid")
    target_range = _scalar(debug, "detected_range_m")
    if not np.isfinite(target_range):
        raise ValueError("invalid debug_info.detected_range_m target range reference")
    return axis - target_range, target_range


def _anti_alias(values: np.ndarray, source_step: float, target_step: float) -> np.ndarray:
    if target_step <= source_step:
        return values
    sigma = 0.5 * max(0.0, target_step / source_step - 1.0)
    if sigma <= 1.0e-6:
        return values
    return gaussian_filter1d(values, sigma=sigma, axis=1, mode="nearest").astype(np.float32)


def _interpolate_velocity(values: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_axis, source_axis, row).astype(np.float32) for row in values],
        axis=0,
    )


def _interpolate_range(
    values: np.ndarray,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    circular_component: bool = False,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in values.T:
        if circular_component:
            left = right = 0.0
        else:
            floor = float(np.min(column))
            left = right = floor
        columns.append(np.interp(target_axis, source_axis, column, left=left, right=right).astype(np.float32))
    return np.stack(columns, axis=1)


def canonicalize_physical_frame(
    amplitude: np.ndarray,
    phase: np.ndarray,
    range_plot: np.ndarray,
    debug: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    amplitude = np.asarray(amplitude, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    if amplitude.ndim != 2 or phase.shape != amplitude.shape:
        raise ValueError(f"expected matching two-dimensional RD planes, got amplitude={amplitude.shape}, phase={phase.shape}")
    source_axis, vr_start, vr_end = _source_velocity_axis(debug, int(amplitude.shape[1]))
    target_axis = _target_velocity_axis()
    source_range_axis, target_range_m = _source_range_offset_axis(range_plot, debug, int(amplitude.shape[0]))
    target_range_axis = _target_range_offset_axis()
    source_step = float((vr_end - vr_start) / (amplitude.shape[1] - 1))
    target_step = float((VELOCITY_END_MPS - VELOCITY_START_MPS) / (CANONICAL_DOPPLER_BINS - 1))
    amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    amplitude = _interpolate_velocity(_anti_alias(amplitude, source_step, target_step), source_axis, target_axis)
    phase_sin = _interpolate_velocity(_anti_alias(np.sin(phase), source_step, target_step), source_axis, target_axis)
    phase_cos = _interpolate_velocity(_anti_alias(np.cos(phase), source_step, target_step), source_axis, target_axis)
    amplitude = _interpolate_range(amplitude, source_range_axis, target_range_axis)
    phase_sin = _interpolate_range(phase_sin, source_range_axis, target_range_axis, circular_component=True)
    phase_cos = _interpolate_range(phase_cos, source_range_axis, target_range_axis, circular_component=True)
    canonical_phase = np.arctan2(phase_sin, phase_cos).astype(np.float32)
    metadata = {
        "source_vr_start_mps": vr_start,
        "source_vr_end_mps": vr_end,
        "source_doppler_bin_width_mps": source_step,
        "canonical_doppler_bin_width_mps": target_step,
        "source_velocity_coverage_ratio": float(np.mean((target_axis >= vr_start) & (target_axis <= vr_end))),
        "anti_alias_sigma_source_bins": float(max(0.0, 0.5 * (target_step / source_step - 1.0))),
        "target_range_reference_m": target_range_m,
        "target_range_reference_source": "debug_info.detected_range_m",
        "source_range_offset_start_m": float(source_range_axis[0]),
        "source_range_offset_end_m": float(source_range_axis[-1]),
        "source_range_coverage_ratio": float(
            np.mean((target_range_axis >= source_range_axis[0]) & (target_range_axis <= source_range_axis[-1]))
        ),
    }
    return amplitude, canonical_phase, metadata


def _load_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, object]:
    payload = loadmat(
        path,
        variable_names=["data_proc_MTD_result_db", "data_proc_MTD_result", "Range_plot", "index_trackPointNo", "debug_info"],
        squeeze_me=True,
        struct_as_record=False,
    )
    amplitude = np.asarray(payload["data_proc_MTD_result_db"], dtype=np.float32)
    complex_rd = payload.get("data_proc_MTD_result")
    if complex_rd is None:
        raise ValueError("canonical physical RD requires data_proc_MTD_result for phase; missing phase cannot be zero-filled")
    phase = np.angle(np.asarray(complex_rd, dtype=np.complex64)).astype(np.float32)
    if phase.shape != amplitude.shape:
        raise ValueError(f"complex RD phase shape {phase.shape} does not match amplitude shape {amplitude.shape}")
    range_plot = np.asarray(payload["Range_plot"], dtype=np.float32).reshape(-1)
    debug = payload["debug_info"]
    observation_seconds = _observation_seconds(debug)
    if observation_seconds <= 0.0:
        observation_seconds = _scalar(debug, "prt_num") * _scalar(debug, "prt_s")
    return amplitude, phase, range_plot, _unwrap_scalar(payload["index_trackPointNo"]), observation_seconds, debug


def build_physical_canonical_store(
    project_root: Path,
    *,
    store_name: str = PHYSICAL_STORE_VERSION,
    output_name: str = "cq08_rd_physical_canonical_frame_index_v3_1.csv",
    summary_name: str = "cq08_rd_physical_canonical_frame_index_v3_1.summary.json",
    start_parent: int = 0,
    max_parents: int = 0,
    progress_interval: int = 2_000,
    overwrite: bool = False,
    allow_existing_store: bool = False,
) -> dict[str, object]:
    """Build a full-frame CQ08 physical-velocity canonical RD store."""
    manifest_path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    records = [record for record in json.loads(manifest_path.read_text(encoding="utf-8")) if bool(record.get("is_usable"))]
    records.sort(key=lambda record: int(record["track_id"]))
    start_parent = max(0, int(start_parent))
    records = records[start_parent:]
    if max_parents > 0:
        records = records[:int(max_parents)]
    processed_root = project_root / "data" / "processed"
    store_root = processed_root / store_name
    output_path = processed_root / output_name
    summary_path = processed_root / summary_name
    if output_path.exists() or summary_path.exists():
        if not overwrite:
            raise FileExistsError("physical canonical index outputs already exist; pass --overwrite to replace them")
    if store_root.exists() and not (overwrite or allow_existing_store):
        raise FileExistsError("physical canonical store already exists; pass --allow-existing-store for a disjoint parent partition")
    store_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    shape_counts: Counter[str] = Counter()
    source_bin_widths: list[float] = []
    coverage_ratios: list[float] = []
    anti_alias_count = 0
    scanned = written = parent_count_written = error_count = 0
    total_frames = sum(len(record.get("rd_mat_files", [])) for record in records)
    started = time.perf_counter()
    for record in records:
        sample_id = f"cq08|track|{int(record['track_id'])}"
        tensor_path = store_root / f"{_safe_name(sample_id)}_rd_tensor.npz"
        amplitudes: list[np.ndarray] = []
        phases: list[np.ndarray] = []
        time_raw: list[float] = []
        time_seconds: list[float] = []
        elapsed_seconds = 0.0
        parent_rows: list[dict[str, object]] = []
        for raw_frame_index, path_value in enumerate(record.get("rd_mat_files", [])):
            try:
                amplitude, phase, range_plot, point_index, observation, debug = _load_frame(Path(path_value))
                source_shape = "x".join(str(value) for value in amplitude.shape)
                canonical_amplitude, canonical_phase, metadata = canonicalize_physical_frame(
                    amplitude,
                    phase,
                    range_plot,
                    debug,
                )
                tensor_frame_index = len(amplitudes)
                amplitudes.append(canonical_amplitude.astype(np.float16))
                phases.append(canonical_phase.astype(np.float16))
                time_raw.append(float(raw_frame_index))
                time_seconds.append(elapsed_seconds)
                elapsed_seconds += max(float(observation), 1.0e-6)
                source_bin_widths.append(float(metadata["source_doppler_bin_width_mps"]))
                coverage_ratios.append(float(metadata["source_velocity_coverage_ratio"]))
                if metadata["anti_alias_sigma_source_bins"] > 1.0e-6:
                    anti_alias_count += 1
                shape_counts[source_shape] += 1
                parent_rows.append(
                    {
                        "schema_version": PHYSICAL_STORE_VERSION,
                        "parent_sample_id": sample_id,
                        "track_id": int(record["track_id"]),
                        "source_name": "cq08_track",
                        "label_name": str(record["label_name"]),
                        "raw_frame_index": raw_frame_index,
                        "tensor_frame_index": tensor_frame_index,
                        "is_tensor_frame_kept": True,
                        "rd_shape": source_shape,
                        "canonical_rd_shape": f"{CANONICAL_RANGE_BINS}x{CANONICAL_DOPPLER_BINS}",
                        "track_point_index": point_index,
                        "rd_observation_duration_seconds": observation,
                        "rd_frame_time_seconds": time_seconds[-1],
                        "rd_mat_path": Path(path_value).as_posix(),
                        "rd_tensor_path": tensor_path.relative_to(project_root).as_posix(),
                        **metadata,
                    }
                )
            except Exception as exc:
                error_count += 1
                if len(errors) < 100:
                    errors.append({"parent_sample_id": sample_id, "raw_frame_index": str(raw_frame_index), "error": repr(exc)})
            scanned += 1
            if progress_interval > 0 and scanned % progress_interval == 0:
                print(f"[cq08-physical-canonical] {scanned}/{total_frames} written={written} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if not amplitudes:
            continue
        temporary_path = tensor_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            amplitude=np.stack(amplitudes, axis=0),
            phase=np.stack(phases, axis=0),
            time_raw=np.asarray(time_raw, dtype=np.float32),
            time_seconds=np.asarray(time_seconds, dtype=np.float32),
            velocity_mps=_target_velocity_axis(),
            range_offset_m=_target_range_offset_axis(),
        )
        os.replace(temporary_path, tensor_path)
        rows.extend(parent_rows)
        written += len(parent_rows)
        parent_count_written += 1
    if error_count:
        raise RuntimeError(
            f"physical canonical store build failed with {error_count} frame errors; "
            f"first_errors={json.dumps(errors[:3], ensure_ascii=False)}"
        )
    frame_index = pd.DataFrame(rows)
    frame_index.to_csv(output_path, index=False, encoding="utf-8")
    summary = {
        "schema_version": PHYSICAL_STORE_VERSION,
        "scope": "CQ08 original MAT frames canonicalized onto common physical velocity and universally supported target-relative range axes; diagnostic-only and not fusion-eligible",
        "canonical_shape": [CANONICAL_RANGE_BINS, CANONICAL_DOPPLER_BINS],
        "velocity_axis_mps": [VELOCITY_START_MPS, VELOCITY_END_MPS],
        "range_offset_axis_m": [RANGE_OFFSET_START_M, RANGE_OFFSET_END_M],
        "range_axis_policy": "global intersection of all audited source Range_plot offsets; 21 native 7.5m bins; no edge padding or range upsampling",
        "target_range_reference_policy": "debug_info.detected_range_m is mandatory; track_range_m fallback is prohibited",
        "canonical_doppler_bin_width_mps": (VELOCITY_END_MPS - VELOCITY_START_MPS) / (CANONICAL_DOPPLER_BINS - 1),
        "interpolation": "per-frame separable physical velocity and target-relative range interpolation; phase remains in sin/cos form until atan2",
        "anti_alias": "Gaussian prefilter is applied only when target velocity bins are coarser than source bins",
        "time_axis": "cumulative per-frame CPI acquisition duration; uses cpi_count*prt_num*prt_s when positive, otherwise prt_num*prt_s; not wall-clock synchronization",
        "start_parent": start_parent, "parent_count_requested": len(records), "parent_count_written": parent_count_written,
        "raw_frame_count": scanned, "canonical_tensor_frame_count": written,
        "dropped_shape_frame_count": 0, "error_count": total_frames - written,
        "errors": errors, "source_shape_counts": dict(shape_counts),
        "source_doppler_bin_width_mps": {"min": min(source_bin_widths, default=0.0), "max": max(source_bin_widths, default=0.0)},
        "source_velocity_coverage_ratio": {"min": min(coverage_ratios, default=0.0), "mean": float(np.mean(coverage_ratios)) if coverage_ratios else 0.0},
        "anti_alias_frame_count": anti_alias_count,
        "elapsed_seconds": time.perf_counter() - started,
        "store_path": store_root.relative_to(project_root).as_posix(),
        "frame_index_path": output_path.relative_to(project_root).as_posix(),
        "fusion_policy": "track_point_anchor remains diagnostic-only and must not enter joint fusion without independent timing evidence.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CQ08 physical-velocity canonical RD tensors.")
    parser.add_argument("--store-name", default=PHYSICAL_STORE_VERSION)
    parser.add_argument("--output-name", default="cq08_rd_physical_canonical_frame_index_v3_1.csv")
    parser.add_argument("--summary-name", default="cq08_rd_physical_canonical_frame_index_v3_1.summary.json")
    parser.add_argument("--start-parent", type=int, default=0)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-existing-store", action="store_true")
    args = parser.parse_args()
    summary = build_physical_canonical_store(
        Path(__file__).resolve().parents[2],
        store_name=args.store_name,
        output_name=args.output_name,
        summary_name=args.summary_name,
        start_parent=args.start_parent,
        max_parents=args.max_parents,
        progress_interval=args.progress_interval,
        overwrite=args.overwrite,
        allow_existing_store=args.allow_existing_store,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
