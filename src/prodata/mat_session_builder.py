import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from src.preprocessing.slicer import DataSlicer
from src.qc.integrity_check import DataIntegrityChecker


LABEL_NORMALIZATION = {
    "bird": "BirdTarget",
    "rotary drone": "DroneTarget",
    "fixed-wing uav": "DroneTarget",
    "helicopter": "AircraftTarget",
    "passenger ship": "ShipTarget",
    "speedboat": "ShipTarget",
}

FILE_PATTERN = re.compile(
    r"(?P<timestamp>\d+)_DAUR_(?P<mode>TR|RD)_(?P<label>.+?)_(?P<segment>\d+)_(?P<token>\d+)\.mat$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RawSessionPair:
    session_id: str
    label_name: str
    tr_path: Path
    rd_path: Path


def _normalize_label(label_text: str) -> str:
    normalized = LABEL_NORMALIZATION.get(label_text.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported label in filename: {label_text}")
    return normalized


def _parse_file_name(file_path: Path) -> dict[str, str]:
    matched = FILE_PATTERN.match(file_path.name)
    if matched is None:
        raise ValueError(f"Unexpected file name format: {file_path.name}")
    payload = matched.groupdict()
    payload["label"] = _normalize_label(payload["label"])
    payload["session_id"] = (
        f"{payload['timestamp']}_{payload['label'].lower()}_{payload['segment']}_{payload['token']}"
    )
    return payload


def discover_raw_session_pairs(raw_root: str) -> list[RawSessionPair]:
    root = Path(raw_root)
    tr_dir = root / "TR"
    rd_dir = root / "RD"
    if not tr_dir.exists() or not rd_dir.exists():
        raise FileNotFoundError("Expected both pro/raw/TR and pro/raw/RD to exist.")

    tr_index: dict[str, tuple[dict[str, str], Path]] = {}
    for tr_path in sorted(tr_dir.glob("*.mat")):
        info = _parse_file_name(tr_path)
        tr_index[info["session_id"]] = (info, tr_path)

    pairs: list[RawSessionPair] = []
    for rd_path in sorted(rd_dir.glob("*.mat")):
        info = _parse_file_name(rd_path)
        tr_entry = tr_index.get(info["session_id"])
        if tr_entry is None:
            continue
        tr_info, tr_path = tr_entry
        if tr_info["label"] != info["label"]:
            raise ValueError(f"TR/RD label mismatch for session {info['session_id']}")
        pairs.append(
            RawSessionPair(
                session_id=info["session_id"],
                label_name=info["label"],
                tr_path=tr_path,
                rd_path=rd_path,
            )
        )
    return pairs


def _read_float_dataset(handle: h5py.File, key: str) -> np.ndarray:
    return np.asarray(handle[key][:], dtype=np.float64).reshape(-1)


def _read_complex_dpl(handle: h5py.File, key: str = "DPL") -> np.ndarray:
    dataset = handle[key][:]
    if dataset.dtype.fields and {"real", "imag"}.issubset(dataset.dtype.fields):
        real_part = np.asarray(dataset["real"], dtype=np.float32)
        imag_part = np.asarray(dataset["imag"], dtype=np.float32)
        return real_part + 1j * imag_part
    if np.iscomplexobj(dataset):
        return np.asarray(dataset, dtype=np.complex64)
    raise ValueError("DPL dataset is not stored as a supported complex matrix.")


def _relative_seconds(time_values: np.ndarray) -> np.ndarray:
    time_values = np.asarray(time_values, dtype=np.float64).reshape(-1)
    return (time_values - float(time_values.min())).astype(np.float32)


def build_track_dataframe(tr_path: Path, range_scale_to_meter: float) -> pd.DataFrame:
    with h5py.File(tr_path, "r") as handle:
        time_values = _relative_seconds(_read_float_dataset(handle, "DATA_time"))
        frame_ids = np.arange(len(time_values), dtype=np.int64)
        range_values = _read_float_dataset(handle, "R_m") * range_scale_to_meter
        radial_speed_values = _read_float_dataset(handle, "V_m")
        azimuth_values = _read_float_dataset(handle, "A_m")
        snr_values = _read_float_dataset(handle, "SNR")
        elevation_values = _read_float_dataset(handle, "E_m")

    return pd.DataFrame(
        {
            "t": time_values,
            "frame_id": frame_ids,
            "track_id": np.ones_like(frame_ids, dtype=np.int64),
            "range_m": range_values.astype(np.float32),
            "azimuth_deg": azimuth_values.astype(np.float32),
            "radial_speed_mps": radial_speed_values.astype(np.float32),
            "snr_db": snr_values.astype(np.float32),
            "elevation_deg": elevation_values.astype(np.float32),
        }
    )


def _build_window(length: int, kind: str) -> np.ndarray:
    normalized = kind.strip().lower()
    if normalized in {"none", "rect", "rectangular"}:
        return np.ones(length, dtype=np.float32)
    if normalized in {"hann", "hanning"}:
        return np.hanning(length).astype(np.float32)
    if normalized == "hamming":
        return np.hamming(length).astype(np.float32)
    raise ValueError(f"Unsupported FFT window kind: {kind}")


def _suppress_static_clutter(range_fft: np.ndarray, mode: str) -> np.ndarray:
    normalized = mode.strip().lower()
    if normalized in {"none", "off"}:
        return range_fft
    if normalized == "median_subtract":
        background = np.median(range_fft, axis=1, keepdims=True)
        return range_fft - background
    raise ValueError(f"Unsupported clutter suppression mode: {mode}")


def _enhance_micro_motion(range_fft: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return range_fft
    previous = np.concatenate([range_fft[:, :1], range_fft[:, :-1]], axis=1)
    temporal_residual = range_fft - previous
    return range_fft + alpha * temporal_residual


def _build_range_axis(track_frame: pd.DataFrame, num_bins: int) -> np.ndarray:
    if num_bins <= 0:
        return np.zeros(1, dtype=np.float32)
    if track_frame.empty or "range_m" not in track_frame.columns:
        return np.arange(num_bins, dtype=np.float32)

    range_values = track_frame["range_m"].to_numpy(dtype=np.float32)
    q05 = float(np.percentile(range_values, 5))
    q95 = float(np.percentile(range_values, 95))
    spread = max(q95 - q05, 1.0)
    padding = max(80.0, spread * 0.15)
    range_min = max(0.0, q05 - padding)
    range_max = max(range_min + 1.0, q95 + padding)
    step = (range_max - range_min) / max(num_bins - 1, 1)
    return (range_min + np.arange(num_bins, dtype=np.float32) * step).astype(np.float32)


def _build_calibrated_range_axis(
    num_bins: int,
    range_resolution_m: float,
    track_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    resolution = max(float(range_resolution_m), 1e-3)
    if track_frame is None or track_frame.empty or "range_m" not in track_frame.columns:
        return (np.arange(num_bins, dtype=np.float32) * resolution).astype(np.float32)

    range_values = track_frame["range_m"].to_numpy(dtype=np.float32)
    q05 = float(np.percentile(range_values, 5))
    q95 = float(np.percentile(range_values, 95))
    spread = max(q95 - q05, resolution)
    padding = max(64.0, spread * 0.2)
    range_start = max(0.0, q05 - padding)
    range_start = float(np.floor(range_start / resolution) * resolution)
    return (range_start + np.arange(num_bins, dtype=np.float32) * resolution).astype(np.float32)


def _build_doppler_axis(track_frame: pd.DataFrame, num_bins: int) -> np.ndarray:
    if num_bins <= 0:
        return np.zeros(1, dtype=np.float32)
    if track_frame.empty or "radial_speed_mps" not in track_frame.columns:
        zero_index = num_bins // 2
        return (np.arange(num_bins, dtype=np.float32) - zero_index).astype(np.float32)

    speed_values = track_frame["radial_speed_mps"].to_numpy(dtype=np.float32)
    robust_extent = float(np.percentile(np.abs(speed_values), 98))
    extent = max(robust_extent * 1.25, 3.0)
    zero_index = num_bins // 2
    step = extent / max(zero_index, 1)
    return ((np.arange(num_bins, dtype=np.float32) - zero_index) * step).astype(np.float32)


def _build_calibrated_doppler_axis(
    num_bins: int,
    doppler_resolution_mps: float,
    zero_doppler_idx: int,
) -> np.ndarray:
    resolution = max(float(doppler_resolution_mps), 1e-3)
    safe_zero_index = int(zero_doppler_idx)
    if safe_zero_index < 0 or safe_zero_index >= num_bins:
        safe_zero_index = num_bins // 2
    return ((np.arange(num_bins, dtype=np.float32) - safe_zero_index) * resolution).astype(np.float32)


def _coverage_ratio(axis_values: np.ndarray, target_values: np.ndarray, tolerance: float = 0.0) -> float:
    if axis_values.size == 0 or target_values.size == 0:
        return 0.0
    lower_bound = float(axis_values.min()) - float(tolerance)
    upper_bound = float(axis_values.max()) + float(tolerance)
    in_range = (target_values >= lower_bound) & (target_values <= upper_bound)
    return float(np.mean(in_range))


def _select_range_axis(
    track_frame: pd.DataFrame,
    range_bins: int,
    range_resolution_m: float,
    axis_mapping_mode: str,
) -> tuple[np.ndarray, str]:
    normalized_mode = axis_mapping_mode.strip().lower()
    if normalized_mode not in {"robust", "calibrated", "calibrated_preferred"}:
        raise ValueError(f"Unsupported axis_mapping_mode: {axis_mapping_mode}")

    calibrated_range_axis = _build_calibrated_range_axis(range_bins, range_resolution_m, track_frame=track_frame)
    if normalized_mode == "calibrated":
        return calibrated_range_axis, "calibrated_track_offset"

    robust_range_axis = _build_range_axis(track_frame, range_bins)
    if normalized_mode == "robust":
        return robust_range_axis, "robust_track_aligned"

    range_values = track_frame["range_m"].to_numpy(dtype=np.float32)
    for multiplier in (1.0, 2.0, 4.0):
        candidate_resolution = float(range_resolution_m) * multiplier
        candidate_axis = _build_calibrated_range_axis(range_bins, candidate_resolution, track_frame=track_frame)
        candidate_coverage = _coverage_ratio(
            candidate_axis,
            range_values,
            tolerance=max(candidate_resolution, 1e-3),
        )
        if candidate_coverage >= 0.95:
            if multiplier == 1.0:
                return candidate_axis, "calibrated_track_offset"
            return candidate_axis, f"adaptive_calibrated_range_x{multiplier:g}"
    return robust_range_axis, "robust_track_aligned"


def _select_doppler_axis(
    track_frame: pd.DataFrame,
    doppler_bins: int,
    doppler_resolution_mps: float,
    zero_doppler_idx: int,
    axis_mapping_mode: str,
) -> tuple[np.ndarray, str]:
    normalized_mode = axis_mapping_mode.strip().lower()
    if normalized_mode not in {"robust", "calibrated", "calibrated_preferred"}:
        raise ValueError(f"Unsupported axis_mapping_mode: {axis_mapping_mode}")

    calibrated_doppler_axis = _build_calibrated_doppler_axis(
        doppler_bins,
        doppler_resolution_mps,
        zero_doppler_idx,
    )
    if normalized_mode == "calibrated":
        return calibrated_doppler_axis, "calibrated_zero_centered"

    robust_doppler_axis = _build_doppler_axis(track_frame, doppler_bins)
    if normalized_mode == "robust":
        return robust_doppler_axis, "robust_track_aligned"

    speed_values = track_frame["radial_speed_mps"].to_numpy(dtype=np.float32)
    for multiplier in (1.0, 1.5, 2.0):
        candidate_resolution = float(doppler_resolution_mps) * multiplier
        candidate_axis = _build_calibrated_doppler_axis(
            doppler_bins,
            candidate_resolution,
            zero_doppler_idx,
        )
        candidate_coverage = _coverage_ratio(
            candidate_axis,
            speed_values,
            tolerance=max(candidate_resolution, 1e-3),
        )
        if candidate_coverage >= 0.95:
            if multiplier == 1.0:
                return candidate_axis, "calibrated_zero_centered"
            return candidate_axis, f"adaptive_calibrated_doppler_x{multiplier:g}"
    return robust_doppler_axis, "robust_track_aligned"


def _select_physical_axes(
    track_frame: pd.DataFrame,
    range_bins: int,
    doppler_bins: int,
    range_resolution_m: float,
    doppler_resolution_mps: float,
    zero_doppler_idx: int,
    axis_mapping_mode: str,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    selected_range_axis, range_axis_mode = _select_range_axis(
        track_frame=track_frame,
        range_bins=range_bins,
        range_resolution_m=range_resolution_m,
        axis_mapping_mode=axis_mapping_mode,
    )
    selected_doppler_axis, doppler_axis_mode = _select_doppler_axis(
        track_frame=track_frame,
        doppler_bins=doppler_bins,
        doppler_resolution_mps=doppler_resolution_mps,
        zero_doppler_idx=zero_doppler_idx,
        axis_mapping_mode=axis_mapping_mode,
    )
    resolved_mode = f"range:{range_axis_mode}|doppler:{doppler_axis_mode}"
    return selected_range_axis, selected_doppler_axis, resolved_mode, range_axis_mode, doppler_axis_mode


def generate_multiscale_rd_frames(
    dpl_matrix: np.ndarray,
    doppler_windows: list[int],
    doppler_fft_size: int,
    range_window_kind: str,
    doppler_window_kind: str,
    clutter_suppression_mode: str,
    micro_motion_alpha: float,
) -> np.ndarray:
    num_range_samples, num_time_steps = dpl_matrix.shape
    if not doppler_windows:
        raise ValueError("doppler_windows must not be empty.")

    range_window = _build_window(num_range_samples, range_window_kind).reshape(-1, 1)
    range_fft = np.fft.fft(dpl_matrix * range_window, axis=0)
    keep_range_bins = range_fft.shape[0] // 2
    range_fft = range_fft[:keep_range_bins, :]
    range_fft = _suppress_static_clutter(range_fft, clutter_suppression_mode)
    range_fft = _enhance_micro_motion(range_fft, micro_motion_alpha)

    scale_frames: list[np.ndarray] = []
    for doppler_window in doppler_windows:
        if doppler_window <= 1:
            raise ValueError("Each doppler window must be greater than 1.")
        half_window = doppler_window // 2
        padded = np.pad(range_fft, ((0, 0), (half_window, half_window)), mode="edge")
        rd_frames = np.empty((num_time_steps, keep_range_bins, doppler_fft_size), dtype=np.complex64)
        doppler_window_values = _build_window(doppler_window, doppler_window_kind).reshape(1, -1)
        for time_index in range(num_time_steps):
            window = padded[:, time_index : time_index + doppler_window] * doppler_window_values
            doppler_map = np.fft.fftshift(
                np.fft.fft(window, n=doppler_fft_size, axis=1),
                axes=1,
            )
            rd_frames[time_index] = doppler_map.astype(np.complex64)
        scale_frames.append(rd_frames)
    return np.stack(scale_frames, axis=0)


def write_rd_data_h5(
    output_path: Path,
    rd_frames: np.ndarray,
    track_frame: pd.DataFrame,
    doppler_windows: list[int],
    use_multires_rd: bool,
    range_window_kind: str,
    doppler_window_kind: str,
    clutter_suppression_mode: str,
    micro_motion_alpha: float,
    axis_mapping_mode: str,
    range_resolution_m: float,
    doppler_resolution_mps: float,
    zero_doppler_idx: int,
) -> None:
    if use_multires_rd:
        primary_scale_index = len(doppler_windows) // 2
        primary_rd_frames = rd_frames[primary_scale_index]
    else:
        primary_rd_frames = rd_frames
    range_axis_m, doppler_axis_mps, resolved_axis_mapping_mode, range_axis_mode, doppler_axis_mode = _select_physical_axes(
        track_frame=track_frame,
        range_bins=int(primary_rd_frames.shape[1]),
        doppler_bins=int(primary_rd_frames.shape[2]),
        range_resolution_m=range_resolution_m,
        doppler_resolution_mps=doppler_resolution_mps,
        zero_doppler_idx=zero_doppler_idx,
        axis_mapping_mode=axis_mapping_mode,
    )
    range_res_m = float(np.median(np.diff(range_axis_m))) if len(range_axis_m) > 1 else 1.0
    doppler_res_mps = float(np.median(np.diff(doppler_axis_mps))) if len(doppler_axis_mps) > 1 else 0.1
    zero_doppler_idx = int(np.argmin(np.abs(doppler_axis_mps))) if len(doppler_axis_mps) else primary_rd_frames.shape[2] // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("rd_frames", data=primary_rd_frames, compression="gzip")
        if use_multires_rd:
            handle.create_dataset("rd_frames_multiscale", data=rd_frames, compression="gzip")
        handle.attrs["range_res_m"] = range_res_m
        handle.attrs["doppler_res_mps"] = doppler_res_mps
        handle.attrs["zero_doppler_idx"] = zero_doppler_idx
        handle.create_dataset("range_axis_m", data=range_axis_m, compression="gzip")
        handle.create_dataset("doppler_axis_mps", data=doppler_axis_mps, compression="gzip")
        handle.attrs["doppler_windows"] = np.asarray(doppler_windows, dtype=np.int32)
        handle.attrs["use_multires_rd"] = int(use_multires_rd)
        handle.attrs["range_window_kind"] = range_window_kind
        handle.attrs["doppler_window_kind"] = doppler_window_kind
        handle.attrs["clutter_suppression_mode"] = clutter_suppression_mode
        handle.attrs["micro_motion_alpha"] = float(micro_motion_alpha)
        handle.attrs["axis_mapping_mode"] = resolved_axis_mapping_mode
        handle.attrs["range_axis_mode"] = range_axis_mode
        handle.attrs["doppler_axis_mode"] = doppler_axis_mode


def write_labels_matched(output_path: Path, label_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track_id": 1,
        "label_main": label_name,
        "confidence": "gold",
        "evidence": "filename_label",
    }
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_meta_session(
    output_path: Path,
    pair: RawSessionPair,
    track_frame: pd.DataFrame,
    rd_frames: np.ndarray,
    use_multires_rd: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": pair.session_id,
        "source": {
            "tr_file": pair.tr_path.name,
            "rd_file": pair.rd_path.name,
        },
        "label": pair.label_name,
        "track": {
            "num_rows": int(len(track_frame)),
            "duration_s": float(track_frame["t"].max()) if not track_frame.empty else 0.0,
        },
        "rd": {
            "num_scales": int(rd_frames.shape[0]) if use_multires_rd else 1,
            "num_frames": int(rd_frames.shape[1]) if use_multires_rd else int(rd_frames.shape[0]),
            "shape": [int(rd_frames.shape[2]), int(rd_frames.shape[3])] if use_multires_rd else [int(rd_frames.shape[1]), int(rd_frames.shape[2])],
            "pipeline_mode": "toTest_multiresolution" if use_multires_rd else "default_single_scale",
        },
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_rd_clips(
    session_dir: Path,
    track_clips: list[dict[str, Any]],
    rd_frames: np.ndarray,
    track_frame: pd.DataFrame,
    use_multires_rd: bool,
) -> Path:
    rd_clip_path = session_dir / "radar/rd/rd_clips.h5"
    rd_clip_path.parent.mkdir(parents=True, exist_ok=True)
    times = track_frame["t"].to_numpy(dtype=np.float32)

    with h5py.File(rd_clip_path, "w") as handle:
        for clip in track_clips:
            t0 = float(clip["t0"])
            t1 = float(clip["t1"])
            clip_indices = np.where((times >= t0) & (times < t1))[0]
            if len(clip_indices) == 0:
                center_index = int(np.argmin(np.abs(times - t0))) if len(times) else 0
                clip_indices = np.asarray([center_index], dtype=np.int64)
            rd_cube = rd_frames[:, clip_indices] if use_multires_rd else rd_frames[clip_indices]
            handle.create_dataset(f"rd_clip_{clip['sample_id']}", data=rd_cube, compression="gzip")
    return rd_clip_path


def prepare_real_sessions(
    raw_root: str,
    output_root: str,
    range_scale_to_meter: float,
    track_window_size_s: float,
    track_overlap: float,
    rd_crop_range_win: int,
    rd_crop_doppler_win: int,
    rd_align_smoothing_window: int,
    rd_align_refine_range_radius: int,
    rd_align_refine_doppler_radius: int,
    doppler_window: int,
    doppler_windows_to_test: list[int],
    use_multires_rd: bool,
    doppler_fft_size: int,
    range_window_kind: str,
    doppler_window_kind: str,
    clutter_suppression_mode: str,
    micro_motion_alpha: float,
    axis_mapping_mode: str,
    range_resolution_m: float,
    doppler_resolution_mps: float,
    zero_doppler_idx: int,
    rd_align_clip_range_margin: int,
    rd_align_clip_doppler_margin: int,
    rd_align_peak_guard_ratio: float,
    rd_align_max_range_shift: int,
    rd_align_max_doppler_shift: int,
    session_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    output_root_path = Path(output_root)

    for pair in discover_raw_session_pairs(raw_root):
        if session_ids is not None and pair.session_id not in session_ids:
            continue

        session_dir = output_root_path / pair.session_id
        track_frame = build_track_dataframe(pair.tr_path, range_scale_to_meter=range_scale_to_meter)
        with h5py.File(pair.rd_path, "r") as handle:
            dpl_matrix = _read_complex_dpl(handle)
        if dpl_matrix.shape[1] != len(track_frame):
            min_length = min(dpl_matrix.shape[1], len(track_frame))
            dpl_matrix = dpl_matrix[:, :min_length]
            track_frame = track_frame.iloc[:min_length].copy()
            track_frame["frame_id"] = np.arange(min_length, dtype=np.int64)

        resolved_windows = list(doppler_windows_to_test) if use_multires_rd else [doppler_window]
        rd_frames = generate_multiscale_rd_frames(
            dpl_matrix=dpl_matrix,
            doppler_windows=resolved_windows,
            doppler_fft_size=doppler_fft_size,
            range_window_kind=range_window_kind,
            doppler_window_kind=doppler_window_kind,
            clutter_suppression_mode=clutter_suppression_mode,
            micro_motion_alpha=micro_motion_alpha,
        )
        if not use_multires_rd:
            rd_frames = rd_frames[0]

        track_csv = session_dir / "radar/track/track_data.csv"
        track_csv.parent.mkdir(parents=True, exist_ok=True)
        track_frame.to_csv(track_csv, index=False)
        write_rd_data_h5(
            session_dir / "radar/rd/rd_data.h5",
            rd_frames,
            track_frame,
            resolved_windows,
            use_multires_rd,
            range_window_kind,
            doppler_window_kind,
            clutter_suppression_mode,
            micro_motion_alpha,
            axis_mapping_mode,
            range_resolution_m,
            doppler_resolution_mps,
            zero_doppler_idx,
        )
        write_meta_session(session_dir / "meta_session.json", pair, track_frame, rd_frames, use_multires_rd)

        checker = DataIntegrityChecker(session_dir)
        qc_result = checker.check_integrity()

        slicer = DataSlicer(session_dir)
        track_clips = slicer.slice_tracks(window_size_s=track_window_size_s, overlap=track_overlap)
        write_labels_matched(session_dir / "labels/labels_matched.jsonl", pair.label_name)
        slicer.slice_rd_clips(
            track_clips,
            range_win=rd_crop_range_win,
            doppler_win=rd_crop_doppler_win,
            smoothing_window=rd_align_smoothing_window,
            refine_range_radius=rd_align_refine_range_radius,
            refine_doppler_radius=rd_align_refine_doppler_radius,
            clip_range_margin=rd_align_clip_range_margin,
            clip_doppler_margin=rd_align_clip_doppler_margin,
            peak_guard_ratio=rd_align_peak_guard_ratio,
            max_range_shift=rd_align_max_range_shift,
            max_doppler_shift=rd_align_max_doppler_shift,
        )

        summaries.append(
            {
                "session_id": pair.session_id,
                "label": pair.label_name,
                "num_track_rows": int(len(track_frame)),
                "num_track_clips": len(track_clips),
                "rd_shape": [int(rd_frames.shape[2]), int(rd_frames.shape[3])] if use_multires_rd else [int(rd_frames.shape[1]), int(rd_frames.shape[2])],
                "rd_num_scales": int(rd_frames.shape[0]) if use_multires_rd else 1,
                "rd_pipeline_mode": "toTest_multiresolution" if use_multires_rd else "default_single_scale",
                "qc_status": qc_result.get("status", "Unknown"),
            }
        )

    return summaries


def sync_prepared_sessions(processed_root: str, dataset_root: str, session_ids: list[str] | None = None) -> list[str]:
    processed_root_path = Path(processed_root)
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    copied_sessions: list[str] = []
    for session_dir in sorted(path for path in processed_root_path.iterdir() if path.is_dir()):
        if session_ids is not None and session_dir.name not in session_ids:
            continue

        destination = dataset_root_path / session_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(session_dir, destination)
        copied_sessions.append(session_dir.name)
    return copied_sessions
