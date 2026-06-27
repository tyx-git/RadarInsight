from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .schema import TRACK_INDEX_COLUMNS, TRACK_SCHEMA_COLUMNS
from src.core.data.labels import two_layer_label_fields


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_time_to_seconds(text: str) -> float:
    value = str(text).strip()
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"unexpected CQ-08 time format: {value}")
    hour, minute, second = parts
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def _load_track_df(track_path: Path, label_name: str, track_id: int, label_id: int, capture_id: str) -> pd.DataFrame:
    frame = pd.read_csv(track_path, encoding="gbk")
    point_time = frame["点时间"].map(_parse_time_to_seconds).astype(np.float64)
    point_time = point_time - float(point_time.iloc[0])
    label_fields = two_layer_label_fields(
        label_name,
        source_name="cq08_track",
        label_id=label_id,
        batch_id=label_id,
        label_name_raw=label_name,
    )
    standardized = pd.DataFrame(
        {
            "source_name": "cq08_track",
            "record_id": [f"cq08|{track_id}#{idx:04d}" for idx in range(len(frame))],
            "pair_id": f"cq08|track|{track_id}",
            "timestamp": capture_id,
            "frame_index": np.arange(len(frame), dtype=int),
            "time_raw": frame["点时间"].astype(str),
            "time_seconds": point_time.astype(float),
            "azimuth_deg": frame["滤波方位"].astype(float),
            "range_m": frame["滤波距离"].astype(float),
            "radial_speed_mps": frame["全速度"].astype(float),
            "elevation_deg": frame["滤波俯仰"].astype(float),
            "height_m": np.nan,
            "amplitude_db": np.nan,
            "snr_db": np.nan,
            "course_deg": frame["航向"].astype(float),
            "speed_mps": frame["全速度"].astype(float),
            "latitude": np.nan,
            "longitude": np.nan,
            "batch_id": str(label_id),
            "label_name_raw": label_name,
            "label_name": label_name,
            "coarse_label": label_fields["coarse_label"],
            "fine_label": label_fields["fine_label"],
            "label_id": label_id,
            "note_text": "",
        }
    )
    return standardized[TRACK_SCHEMA_COLUMNS]


def _load_point_df(point_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(point_path, encoding="gbk")
    point_time = frame["点时间"].map(_parse_time_to_seconds).astype(np.float64)
    point_time = point_time - float(point_time.iloc[0])
    frame = frame.copy()
    frame["time_seconds"] = point_time.astype(float)
    frame["point_radial_speed_mps"] = pd.to_numeric(frame["多普勒速度"], errors="coerce").astype(float)
    frame["point_amplitude_db"] = pd.to_numeric(frame["和幅度"], errors="coerce").astype(float)
    point_snr = pd.to_numeric(frame["信噪比"], errors="coerce").fillna(0.0).astype(float)
    point_snr = np.maximum(point_snr.to_numpy(dtype=float), 0.0)
    frame["point_snr_db"] = 10.0 * np.log10(point_snr + 1.0)
    return frame


def _load_rd_stack(rd_mat_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    amplitude_frames: list[np.ndarray] = []
    phase_frames: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for mat_file in rd_mat_files:
        payload = loadmat(mat_file)
        # Amplitude: use the pre-computed dB version (unchanged behavior).
        amplitude = np.asarray(payload["data_proc_MTD_result_db"], dtype=np.float32)
        amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
        # Phase: extract from complex data when available.
        if "data_proc_MTD_result" in payload:
            complex_rd = np.asarray(payload["data_proc_MTD_result"], dtype=np.complex128)
            phase = np.angle(complex_rd).astype(np.float32)
            phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            phase = np.zeros_like(amplitude, dtype=np.float32)
        if target_shape is None:
            target_shape = tuple(int(x) for x in amplitude.shape)
        if tuple(amplitude.shape) != target_shape:
            continue
        amplitude_frames.append(amplitude.astype(np.float32))
        phase_frames.append(phase.astype(np.float32))
    if not amplitude_frames:
        raise ValueError("no valid CQ-08 RD frames after shape filtering")
    return np.stack(amplitude_frames, axis=0), np.stack(phase_frames, axis=0)


def export_cq08_tracks(project_root: Path, limit: int | None = None) -> list[Dict[str, object]]:
    manifest_path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    output_dir = project_root / "data" / "processed" / "cq08_tracks"
    _ensure_dir(output_dir)
    records: List[Dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    usable_records = [item for item in records if bool(item.get("is_usable"))]
    if limit is not None:
        usable_records = usable_records[: int(limit)]
    index_rows: list[Dict[str, object]] = []
    for item in usable_records:
        track_path = project_root / Path(item["track_file"])
        point_path = project_root / Path(item["point_file"])
        df = _load_track_df(
            track_path=track_path,
            label_name=str(item["label_name"]),
            track_id=int(item["track_id"]),
            label_id=int(item["label_id"]),
            capture_id=str(item["capture_id"]),
        )
        point_df = _load_point_df(point_path)
        point_agg = (
            point_df.groupby("time_seconds", as_index=False)[["point_radial_speed_mps", "point_amplitude_db", "point_snr_db"]]
            .mean()
            .sort_values("time_seconds")
        )
        df = pd.merge_asof(
            df.sort_values("time_seconds"),
            point_agg,
            on="time_seconds",
            direction="nearest",
            tolerance=1.0,
        )
        if "point_radial_speed_mps" in df:
            df["radial_speed_mps"] = pd.to_numeric(df["point_radial_speed_mps"], errors="coerce").fillna(df["radial_speed_mps"]).astype(float)
            df = df.drop(columns=["point_radial_speed_mps"])
        if "point_amplitude_db" in df:
            df["amplitude_db"] = pd.to_numeric(df["point_amplitude_db"], errors="coerce").fillna(df["amplitude_db"]).astype(float)
            df = df.drop(columns=["point_amplitude_db"])
        if "point_snr_db" in df:
            df["snr_db"] = pd.to_numeric(df["point_snr_db"], errors="coerce").fillna(df["snr_db"]).astype(float)
            df = df.drop(columns=["point_snr_db"])
        df = df[TRACK_SCHEMA_COLUMNS]
        sample_id = f"cq08|track|{int(item['track_id'])}"
        output_path = output_dir / f"{sample_id.replace('|', '__')}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        label_fields = two_layer_label_fields(
            str(item["label_name"]),
            source_name="cq08_track",
            label_id=int(item["label_id"]),
            batch_id=str(item["label_id"]),
            label_name_raw=str(item["label_name"]),
        )
        index_rows.append(
            {
                "dataset_name": "cq08_tracks",
                "source_name": "cq08_track",
                "sample_id": sample_id,
                "sample_kind": "cq08_track_group",
                "timestamp": str(item["capture_id"]),
                "batch_id": str(item["label_id"]),
                "label_name_raw": str(item["label_name"]),
                "label_name": str(item["label_name"]),
                "coarse_label": label_fields["coarse_label"],
                "fine_label": label_fields["fine_label"],
                "record_count": int(len(df)),
                "time_seconds_min": float(df["time_seconds"].min()) if len(df) else np.nan,
                "time_seconds_max": float(df["time_seconds"].max()) if len(df) else np.nan,
                "csv_path": output_path.relative_to(project_root).as_posix(),
                "rd_path": str(item["raw_echo_file"]),
                "rd_summary_csv_path": f"data/processed/cq08_rd/{sample_id.replace('|', '__')}_rd_summary.csv",
                "rd_tensor_path": f"data/processed/cq08_rd/{sample_id.replace('|', '__')}_rd_tensor.npz",
                "rd_phase_missing": False,
                "height_missing": True,
            }
        )
    return index_rows


def export_cq08_rd(project_root: Path, limit: int | None = None) -> list[Dict[str, object]]:
    manifest_path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    output_dir = project_root / "data" / "processed" / "cq08_rd"
    _ensure_dir(output_dir)
    records: List[Dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    usable_records = [item for item in records if bool(item.get("is_usable"))]
    if limit is not None:
        usable_records = usable_records[: int(limit)]
    index_rows: list[Dict[str, object]] = []
    for item in usable_records:
        sample_id = f"cq08|track|{int(item['track_id'])}"
        amplitude, phase = _load_rd_stack([str(project_root / Path(path)) for path in item["rd_mat_files"]])
        frame_energy = amplitude.mean(axis=(1, 2))
        summary_df = pd.DataFrame(
            {
                "pair_id": sample_id,
                "timestamp": str(item["capture_id"]),
                "frame_index": np.arange(amplitude.shape[0], dtype=int),
                "time_raw": np.arange(amplitude.shape[0], dtype=float),
                "gps_time_raw": np.nan,
                "iframecnt": np.nan,
                "rd_bin_count": int(amplitude.shape[1]),
                "frame_energy_mean": frame_energy.astype(float),
                "frame_energy_max": amplitude.max(axis=(1, 2)).astype(float),
                "frame_energy_min": amplitude.min(axis=(1, 2)).astype(float),
            }
        )
        summary_path = output_dir / f"{sample_id.replace('|', '__')}_rd_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        tensor_path = output_dir / f"{sample_id.replace('|', '__')}_rd_tensor.npz"
        np.savez_compressed(
            tensor_path,
            amplitude=amplitude.astype(np.float32),
            phase=phase.astype(np.float32),
            time_raw=np.arange(amplitude.shape[0], dtype=np.float32),
        )
        index_rows.append(
            {
                "dataset_name": "cq08_rd",
                "source_name": "cq08_track",
                "sample_id": sample_id,
                "sample_kind": "cq08_rd_group",
                "timestamp": str(item["capture_id"]),
                "label_name_raw": str(item["label_name"]),
                "label_name": str(item["label_name"]),
                "frame_count": int(amplitude.shape[0]),
                "rd_bin_count": int(amplitude.shape[1]),
                "time_raw_min": 0.0,
                "time_raw_max": float(max(0, amplitude.shape[0] - 1)),
                "frame_energy_mean": float(frame_energy.mean()),
                "frame_energy_std": float(frame_energy.std()),
                "summary_csv_path": summary_path.relative_to(project_root).as_posix(),
                "rd_tensor_path": tensor_path.relative_to(project_root).as_posix(),
                "rd_mat_path": "",
            }
        )
    return index_rows


def export_all(project_root: Path, limit: int | None = None) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    track_rows = export_cq08_tracks(project_root, limit=limit)
    rd_rows = export_cq08_rd(project_root, limit=limit)

    track_index = pd.DataFrame(track_rows)[TRACK_INDEX_COLUMNS]
    track_index.to_csv(processed_root / "cq08_track_index.csv", index=False, encoding="utf-8")
    rd_index = pd.DataFrame(rd_rows)
    rd_index.to_csv(processed_root / "cq08_rd_index.csv", index=False, encoding="utf-8")

    summary = {
        "cq08_track_exports": int(len(track_rows)),
        "cq08_rd_exports": int(len(rd_rows)),
        "source_counts": track_index["source_name"].value_counts(dropna=False).to_dict() if not track_index.empty else {},
        "label_counts": track_index["label_name"].value_counts(dropna=False).to_dict() if not track_index.empty else {},
    }
    (processed_root / "cq08_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    import argparse

    parser = argparse.ArgumentParser(description="Standardize CQ-08 track and RD into processed outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke export")
    args = parser.parse_args()
    summary = export_all(project_root, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
