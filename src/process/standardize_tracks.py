from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

from .schema import TRACK_INDEX_COLUMNS, TRACK_SCHEMA_COLUMNS
from src.core.data.labels import two_layer_label_fields


RAW_LABEL_MAP = {
    "Bird": "BirdTarget",
    "Rotary drone": "DroneTarget",
    "Fixed-wing UAV": "AircraftTarget",
    "Helicopter": "AircraftTarget",
    "Passenger ship": "ShipTarget",
    "Speedboat": "ShipTarget",
}


EXCEL_MARKTYPE_MAP = {
    "无人机": "DroneTarget",
    "小型旋翼无人机": "DroneTarget",
    "轻型旋翼无人机": "DroneTarget",
    "飞鸟": "BirdTarget",
    "鸟": "BirdTarget",
    "疑似气球": "BalloonTarget",
    "杂波": "ClutterTarget",
    "直升机": "AircraftTarget",
    "固定翼": "AircraftTarget",
    "有人机": "AircraftTarget",
    "大型机": "AircraftTarget",
    "小型机": "AircraftTarget",
    "微型机": "DroneTarget",
    "船": "ShipTarget",
    "船只": "ShipTarget",
    "快艇": "ShipTarget",
    "车": "CarTarget",
    "人": "PeopleTarget",
    "待定": "UnknownTarget",
}


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _versioned_output_name(prefix: str, version: str) -> str:
    normalized = str(version).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError("version must contain only lowercase letters, digits, '_' or '-'")
    return f"{prefix}_tr_only_{normalized}"


def _processed_output_dir(project_root: Path, output_dir_name: str) -> Path:
    relative = Path(str(output_dir_name))
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError("processed output directory must be one relative directory name")
    processed_root = (project_root / "data" / "processed").resolve()
    output_dir = (processed_root / relative).resolve()
    if output_dir.parent != processed_root:
        raise ValueError("processed output directory escapes data/processed")
    return output_dir


def _normalize_elapsed_time_seconds(
    values: np.ndarray,
    *,
    seconds_per_input_unit: float,
) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    base = float(values[0])
    return (values.astype(float) - base) * seconds_per_input_unit


def _normalize_raw_data_time_seconds(values: np.ndarray) -> np.ndarray:
    # Raw DATA_time is seconds since midnight with millisecond precision.
    return _normalize_elapsed_time_seconds(values, seconds_per_input_unit=1.0)


def _normalize_excel_time_seconds(values: np.ndarray) -> np.ndarray:
    return _normalize_elapsed_time_seconds(values, seconds_per_input_unit=1.0 / 1000.0)


def _normalize_raw_range_m(values: np.ndarray) -> np.ndarray:
    # DAUR raw R is stored in kilometres; the canonical track schema uses metres.
    return values.astype(float) * 1000.0


def _normalize_excel_radial_speed_mps(values: np.ndarray) -> np.ndarray:
    # Excel Speed_R is positive for approaching targets; canonical range rate is
    # positive when the target moves away from the radar.
    return -values.astype(float)


def _derive_raw_radial_speed_mps(
    time_seconds: np.ndarray,
    range_m: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Derive positive-receding range rate using only the current and prior TR points."""
    time = np.asarray(time_seconds, dtype=float).reshape(-1)
    distance = np.asarray(range_m, dtype=float).reshape(-1)
    source = np.asarray(source_values, dtype=float).reshape(-1)
    if not (time.size == distance.size == source.size):
        raise ValueError("raw time, range, and V arrays must have equal length")
    if bool((np.isfinite(source) & ~np.isclose(source, 0.0, atol=1.0e-12)).any()):
        raise ValueError("raw V contract changed: non-zero source values require review")

    speed = np.full(time.shape, np.nan, dtype=float)
    previous_index: int | None = None
    for index in range(time.size):
        if not (np.isfinite(time[index]) and np.isfinite(distance[index])):
            continue
        if previous_index is None:
            previous_index = index
            continue
        delta_time = float(time[index] - time[previous_index])
        if delta_time < 0.0:
            raise ValueError("raw DATA_time must be non-decreasing")
        if delta_time == 0.0:
            if not np.isclose(
                distance[index],
                distance[previous_index],
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError("duplicate raw DATA_time has inconsistent range")
            speed[index] = speed[previous_index]
        else:
            speed[index] = (
                distance[index] - distance[previous_index]
            ) / delta_time
        previous_index = index
    return speed


def export_raw_tracks(
    project_root: Path,
    *,
    output_dir_name: str = "raw_tracks",
    include_rd_metadata: bool = True,
) -> List[Dict[str, object]]:
    manifest_path = project_root / "data" / "manifests" / "raw_tr_rd_manifest.json"
    rd_index_path = project_root / "data" / "processed" / "raw_rd_index.csv"
    output_dir = _processed_output_dir(project_root, output_dir_name)
    _ensure_clean_dir(output_dir)
    records: List[Dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    rd_summary_map: Dict[str, str] = {}
    if include_rd_metadata and rd_index_path.exists():
        rd_index_df = pd.read_csv(rd_index_path)
        rd_summary_map = {
            str(row["sample_id"]): {
                "summary_csv_path": str(row["summary_csv_path"]),
                "rd_tensor_path": str(row["rd_tensor_path"]) if "rd_tensor_path" in row and not pd.isna(row["rd_tensor_path"]) else "",
            }
            for row in rd_index_df.to_dict(orient="records")
            if row.get("summary_csv_path")
        }
    index_rows: List[Dict[str, object]] = []
    for item in records:
        tr_path = project_root / Path(item["tr_file"])
        label_name = RAW_LABEL_MAP.get(item["label"], item["label"])
        label_fields = two_layer_label_fields(
            label_name,
            source_name="raw_tr_rd",
            batch_id=item["segment"],
            label_name_raw=item["label"],
        )
        with h5py.File(tr_path, "r") as handle:
            time_raw = np.array(handle["DATA_time"]).reshape(-1)
            azimuth = np.array(handle["A"]).reshape(-1)
            distance = np.array(handle["R"]).reshape(-1)
            radial_speed = np.array(handle["V"]).reshape(-1)
            elevation = np.array(handle["E"]).reshape(-1)
            snr = np.array(handle["SNR"]).reshape(-1)
        frame_count = len(time_raw)
        time_seconds = _normalize_raw_data_time_seconds(time_raw)
        range_m = _normalize_raw_range_m(distance)
        radial_speed_mps = _derive_raw_radial_speed_mps(
            time_seconds,
            range_m,
            radial_speed,
        )
        df = pd.DataFrame(
            {
                "source_name": "raw_tr_rd",
                "record_id": [f"{item['pair_id']}#{idx:04d}" for idx in range(frame_count)],
                "pair_id": item["pair_id"],
                "timestamp": item["timestamp"],
                "frame_index": np.arange(frame_count, dtype=int),
                "time_raw": time_raw.astype(float),
                "time_seconds": time_seconds,
                "azimuth_deg": azimuth.astype(float),
                "range_m": range_m,
                "radial_speed_mps": radial_speed_mps,
                "elevation_deg": elevation.astype(float),
                "height_m": np.nan,
                "amplitude_db": np.nan,
                "snr_db": snr.astype(float),
                "course_deg": np.nan,
                "speed_mps": np.nan,
                "latitude": np.nan,
                "longitude": np.nan,
                "batch_id": item["segment"],
                "label_name_raw": item["label"],
                "label_name": label_name,
                "coarse_label": label_fields["coarse_label"],
                "fine_label": label_fields["fine_label"],
                "label_id": np.nan,
                "note_text": "",
            }
        )
        df = df[TRACK_SCHEMA_COLUMNS]
        output_path = output_dir / f"{item['pair_id'].replace('|', '__')}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        index_rows.append(
            {
                "dataset_name": "raw_tracks",
                "source_name": "raw_tr_rd",
                "sample_id": item["pair_id"],
                "sample_kind": "raw_pair",
                "timestamp": item["timestamp"],
                "batch_id": item["segment"],
                "label_name_raw": item["label"],
                "label_name": label_name,
                "coarse_label": label_fields["coarse_label"],
                "fine_label": label_fields["fine_label"],
                "record_count": frame_count,
                "time_seconds_min": float(df["time_seconds"].min()) if frame_count else np.nan,
                "time_seconds_max": float(df["time_seconds"].max()) if frame_count else np.nan,
                "csv_path": output_path.relative_to(project_root).as_posix(),
                "rd_path": (
                    Path(item["rd_file"]).relative_to(project_root).as_posix()
                    if include_rd_metadata
                    else ""
                ),
                "rd_summary_csv_path": rd_summary_map.get(item["pair_id"], {}).get("summary_csv_path", ""),
                "rd_tensor_path": rd_summary_map.get(item["pair_id"], {}).get("rd_tensor_path", ""),
                "rd_phase_missing": False,
                "height_missing": True,
            }
        )
    return index_rows


def export_excel_tracks(
    project_root: Path,
    *,
    output_dir_name: str = "excel_tracks",
) -> List[Dict[str, object]]:
    manifest_path = project_root / "data" / "manifests" / "excel_track_manifest.json"
    output_dir = _processed_output_dir(project_root, output_dir_name)
    _ensure_clean_dir(output_dir)
    records: List[Dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_rows: List[Dict[str, object]] = []
    for item in records:
        track_path = project_root / Path(item["track_file"])
        mark_path = project_root / Path(item["mark_file"])
        track_df = pd.read_excel(track_path)
        mark_df = pd.read_excel(mark_path)
        merged = track_df.merge(
            mark_df[["StartTimer", "Batch", "MarkTypeID", "MarkType", "Notes"]],
            on=["StartTimer", "Batch"],
            how="left",
        )
        merged["time_seconds"] = _normalize_excel_time_seconds(
            merged["Time"].to_numpy(dtype=float)
        )
        merged["label_name_raw"] = merged["MarkType"].fillna("UnknownTarget").astype(str)
        merged["label_name"] = merged["label_name_raw"].map(EXCEL_MARKTYPE_MAP).fillna("UnknownTarget")
        label_fields = merged.apply(
            lambda row: two_layer_label_fields(
                str(row["label_name"]),
                source_name="excel_track",
                label_id=row.get("MarkTypeID"),
                batch_id=row.get("Batch"),
                label_name_raw=str(row["label_name_raw"]),
            ),
            axis=1,
        )
        merged["coarse_label"] = label_fields.map(lambda item: item["coarse_label"])
        merged["fine_label"] = label_fields.map(lambda item: item["fine_label"])
        merged["record_id"] = [
            f"{item['pair_id']}#{idx:04d}" for idx in range(len(merged))
        ]
        standardized = pd.DataFrame(
            {
                "source_name": "excel_track",
                "record_id": merged["record_id"],
                "pair_id": item["pair_id"],
                "timestamp": item["timestamp"],
                "frame_index": np.arange(len(merged), dtype=int),
                "time_raw": merged["Time"].astype(float),
                "time_seconds": merged["time_seconds"].astype(float),
                "azimuth_deg": merged["Azimuth"].astype(float),
                "range_m": merged["Distance"].astype(float),
                "radial_speed_mps": _normalize_excel_radial_speed_mps(
                    merged["Speed_R"].to_numpy(dtype=float)
                ),
                "elevation_deg": merged["Elevation"].astype(float),
                "height_m": merged["Height"].astype(float),
                "amplitude_db": merged["Amplitude"].astype(float),
                "snr_db": np.nan,
                "course_deg": merged["Course"].astype(float),
                "speed_mps": merged["Speed"].astype(float),
                "latitude": merged["Latitude"].astype(float),
                "longitude": merged["longitude"].astype(float),
                "batch_id": merged["Batch"].astype(str),
                "label_name_raw": merged["label_name_raw"].astype(str),
                "label_name": merged["label_name"].astype(str),
                "coarse_label": merged["coarse_label"].astype(str),
                "fine_label": merged["fine_label"].astype(str),
                "label_id": merged["MarkTypeID"],
                "note_text": merged["Notes"].fillna("").astype(str),
            }
        )
        standardized["batch_id"] = standardized["batch_id"].astype(str)
        standardized = standardized[TRACK_SCHEMA_COLUMNS]
        for batch_id, batch_df in standardized.groupby("batch_id", sort=True):
            batch_df = batch_df.reset_index(drop=True).copy()
            batch_df["record_id"] = [
                f"{item['pair_id']}|batch{batch_id}#{idx:04d}" for idx in range(len(batch_df))
            ]
            label_values = sorted(
                {
                    str(value)
                    for value in batch_df["label_name"].dropna().astype(str)
                    if str(value).strip()
                }
            )
            raw_label_values = sorted(
                {
                    str(value)
                    for value in batch_df["label_name_raw"].dropna().astype(str)
                    if str(value).strip()
                }
            )
            if not label_values:
                label_name = "UnknownTarget"
            elif len(label_values) == 1:
                label_name = label_values[0]
            else:
                label_name = "MixedLabel"
            if not raw_label_values:
                raw_label_name = "UnknownTarget"
            elif len(raw_label_values) == 1:
                raw_label_name = raw_label_values[0]
            else:
                raw_label_name = "MixedLabel"
            fine_label_values = sorted(
                {
                    str(value)
                    for value in batch_df["fine_label"].dropna().astype(str)
                    if str(value).strip()
                }
            )
            if not fine_label_values:
                fine_label = label_name
            elif len(fine_label_values) == 1:
                fine_label = fine_label_values[0]
            else:
                fine_label = "MixedLabel"
            batch_df["pair_id"] = f"{item['pair_id']}|batch{batch_id}"
            output_path = output_dir / f"{item['pair_id']}__batch_{batch_id}.csv"
            batch_df.to_csv(output_path, index=False, encoding="utf-8")
            index_rows.append(
                {
                    "dataset_name": "excel_tracks",
                    "source_name": "excel_track",
                    "sample_id": f"{item['pair_id']}|batch{batch_id}",
                    "sample_kind": "excel_batch",
                    "timestamp": item["timestamp"],
                    "batch_id": batch_id,
                    "label_name_raw": raw_label_name,
                    "label_name": label_name,
                    "coarse_label": label_name,
                    "fine_label": fine_label,
                    "record_count": len(batch_df),
                    "time_seconds_min": float(batch_df["time_seconds"].min()) if len(batch_df) else np.nan,
                    "time_seconds_max": float(batch_df["time_seconds"].max()) if len(batch_df) else np.nan,
                    "csv_path": output_path.relative_to(project_root).as_posix(),
                    "rd_path": "",
                    "rd_summary_csv_path": "",
                    "rd_tensor_path": "",
                    "rd_phase_missing": True,
                    "height_missing": False,
                }
            )
    return index_rows


def export_all(project_root: Path) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    raw_index_rows = export_raw_tracks(project_root)
    excel_index_rows = export_excel_tracks(project_root)
    raw_index = pd.DataFrame(raw_index_rows)[TRACK_INDEX_COLUMNS]
    excel_index = pd.DataFrame(excel_index_rows)[TRACK_INDEX_COLUMNS]
    full_index = pd.concat([raw_index, excel_index], ignore_index=True)[TRACK_INDEX_COLUMNS]
    raw_index.to_csv(processed_root / "raw_track_index.csv", index=False, encoding="utf-8")
    excel_index.to_csv(processed_root / "excel_track_index.csv", index=False, encoding="utf-8")
    full_index.to_csv(processed_root / "track_index.csv", index=False, encoding="utf-8")
    summary = {
        "raw_track_exports": len(raw_index_rows),
        "excel_track_exports": len(excel_index_rows),
        "total_track_exports": len(full_index),
        "source_counts": full_index["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": full_index["label_name"].value_counts(dropna=False).to_dict(),
    }
    (processed_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_tr_only_versioned(project_root: Path, version: str) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    normalized_version = str(version).strip().lower()
    raw_dir_name = _versioned_output_name("raw_tracks", normalized_version)
    excel_dir_name = _versioned_output_name("excel_tracks", normalized_version)
    raw_index_rows = export_raw_tracks(
        project_root,
        output_dir_name=raw_dir_name,
        include_rd_metadata=False,
    )
    excel_index_rows = export_excel_tracks(
        project_root,
        output_dir_name=excel_dir_name,
    )
    raw_index = pd.DataFrame(raw_index_rows)[TRACK_INDEX_COLUMNS]
    excel_index = pd.DataFrame(excel_index_rows)[TRACK_INDEX_COLUMNS]
    full_index = pd.concat([raw_index, excel_index], ignore_index=True)[TRACK_INDEX_COLUMNS]
    raw_index_name = _versioned_output_name("raw_track_index", normalized_version) + ".csv"
    excel_index_name = _versioned_output_name("excel_track_index", normalized_version) + ".csv"
    full_index_name = _versioned_output_name("track_index", normalized_version) + ".csv"
    summary_name = _versioned_output_name("track_standardization_summary", normalized_version) + ".json"
    raw_index_path = processed_root / raw_index_name
    excel_index_path = processed_root / excel_index_name
    full_index_path = processed_root / full_index_name
    summary_path = processed_root / summary_name
    raw_index.to_csv(raw_index_path, index=False, encoding="utf-8")
    excel_index.to_csv(excel_index_path, index=False, encoding="utf-8")
    full_index.to_csv(full_index_path, index=False, encoding="utf-8")
    summary: Dict[str, object] = {
        "contract": "tr_only_track_standardization_v1",
        "version": normalized_version,
        "raw_time_contract": "seconds_since_midnight_relative_to_track_start",
        "raw_range_contract": "source_kilometres_to_metres",
        "radial_speed_contract": "meters_per_second_positive_receding",
        "raw_radial_speed_contract": "causal_backward_range_rate",
        "rd_metadata": "not_applicable_tr_only",
        "raw_track_exports": len(raw_index_rows),
        "excel_track_exports": len(excel_index_rows),
        "total_track_exports": len(full_index),
        "source_counts": full_index["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": full_index["label_name"].value_counts(dropna=False).to_dict(),
        "raw_output_dir": (processed_root / raw_dir_name).relative_to(project_root).as_posix(),
        "excel_output_dir": (processed_root / excel_dir_name).relative_to(project_root).as_posix(),
        "raw_index": raw_index_path.relative_to(project_root).as_posix(),
        "excel_index": excel_index_path.relative_to(project_root).as_posix(),
        "track_index": full_index_path.relative_to(project_root).as_posix(),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary"] = summary_path.relative_to(project_root).as_posix()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize raw and Excel trajectory tracks.")
    parser.add_argument(
        "--tr-only-version",
        default="",
        help="Write a versioned TR-only dataset without reading or carrying RD metadata.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = (
        export_tr_only_versioned(project_root, args.tr_only_version)
        if args.tr_only_version
        else export_all(project_root)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
