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
    "bird": "Bird",
    "rotary drone": "RotaryDrone",
    "fixed-wing uav": "FixedWing",
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


def generate_multiscale_rd_frames(
    dpl_matrix: np.ndarray,
    doppler_windows: list[int],
    doppler_fft_size: int,
) -> np.ndarray:
    num_range_samples, num_time_steps = dpl_matrix.shape
    if not doppler_windows:
        raise ValueError("doppler_windows must not be empty.")

    range_fft = np.fft.fft(dpl_matrix, axis=0)
    keep_range_bins = range_fft.shape[0] // 2
    range_fft = range_fft[:keep_range_bins, :]

    scale_frames: list[np.ndarray] = []
    for doppler_window in doppler_windows:
        if doppler_window <= 1:
            raise ValueError("Each doppler window must be greater than 1.")
        half_window = doppler_window // 2
        padded = np.pad(range_fft, ((0, 0), (half_window, half_window)), mode="edge")
        rd_frames = np.empty((num_time_steps, keep_range_bins, doppler_fft_size), dtype=np.complex64)
        for time_index in range(num_time_steps):
            window = padded[:, time_index : time_index + doppler_window]
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
) -> None:
    if use_multires_rd:
        primary_scale_index = len(doppler_windows) // 2
        primary_rd_frames = rd_frames[primary_scale_index]
    else:
        primary_rd_frames = rd_frames
    max_range_m = float(track_frame["range_m"].max()) if not track_frame.empty else float(primary_rd_frames.shape[1])
    max_speed_mps = float(np.max(np.abs(track_frame["radial_speed_mps"].to_numpy()))) if not track_frame.empty else 0.0
    range_res_m = max(max_range_m / max(primary_rd_frames.shape[1], 1), 1.0)
    doppler_res_mps = max((2 * max_speed_mps) / max(primary_rd_frames.shape[2], 1), 0.1)
    zero_doppler_idx = primary_rd_frames.shape[2] // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("rd_frames", data=primary_rd_frames, compression="gzip")
        if use_multires_rd:
            handle.create_dataset("rd_frames_multiscale", data=rd_frames, compression="gzip")
        handle.attrs["range_res_m"] = range_res_m
        handle.attrs["doppler_res_mps"] = doppler_res_mps
        handle.attrs["zero_doppler_idx"] = zero_doppler_idx
        handle.attrs["doppler_windows"] = np.asarray(doppler_windows, dtype=np.int32)
        handle.attrs["use_multires_rd"] = int(use_multires_rd)


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
    doppler_window: int,
    doppler_windows_to_test: list[int],
    use_multires_rd: bool,
    doppler_fft_size: int,
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
        )
        write_meta_session(session_dir / "meta_session.json", pair, track_frame, rd_frames, use_multires_rd)

        checker = DataIntegrityChecker(session_dir)
        qc_result = checker.check_integrity()

        slicer = DataSlicer(session_dir)
        track_clips = slicer.slice_tracks(window_size_s=track_window_size_s, overlap=track_overlap)
        write_labels_matched(session_dir / "labels/labels_matched.jsonl", pair.label_name)
        write_rd_clips(session_dir, track_clips, rd_frames, track_frame, use_multires_rd)

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
