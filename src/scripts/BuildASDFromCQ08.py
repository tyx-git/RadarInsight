"""Build a self-contained standard ``.asd`` package from one CQ08 parent.

The package keeps CQ08 raw physical RD frames.  The ASD adapter performs the
same physical-axis canonicalization later, so this artifact is useful both as
an integration fixture and as a frontend handoff reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from radar_ln3.data.cq08 import CQ08Adapter, resolve_frame_source_path  # noqa: E402
from radar_ln3.data.raw_reader import read_mat  # noqa: E402
from radar_ln3.data.rd_sanitizer import sanitize_rd_amplitude_db  # noqa: E402


ASD_PACKAGE_SCHEMA = "RADAR-LN3-ASD-PACKAGE-V1"
ASD_TR_SCHEMA = "RADAR-LN3-ASD-TR-V1"
ASD_RD_SCHEMA = "RADAR-LN3-ASD-RD-RAW-PHYSICAL-V1"
ASD_TR_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "amplitude_db",
    "snr_db",
    "speed_mps",
    "course_deg",
)


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _debug_number(debug: Any, name: str, *, path: Path) -> float:
    value = getattr(debug, name, None)
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"MAT debug_info.{name} is not scalar: {path}")
    return _finite(array[0], name=f"{path}:debug_info.{name}")


def _read_raw_frame(
    path: Path,
    *,
    frame_time_seconds: float,
    frame_index: int,
    track_point_index: int,
    source_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = read_mat(
        path,
        variable_names=(
            "data_proc_MTD_result_db",
            "data_proc_MTD_result",
            "Range_plot",
            "Vr",
            "debug_info",
        ),
    )
    if "data_proc_MTD_result_db" not in payload:
        raise ValueError(f"MAT lacks data_proc_MTD_result_db: {path}")
    amplitude_source = np.asarray(payload["data_proc_MTD_result_db"])
    complex_values = payload.get("data_proc_MTD_result")
    sanitized = sanitize_rd_amplitude_db(
        amplitude_source,
        complex_values,
        name=str(path),
    )
    amplitude = np.asarray(sanitized.values, dtype=np.float32)
    if complex_values is None:
        raise ValueError(f"MAT lacks data_proc_MTD_result for phase: {path}")
    complex_array = np.asarray(complex_values)
    if complex_array.shape != amplitude.shape:
        raise ValueError(f"MAT amplitude/complex shape mismatch: {path}")
    phase = np.asarray(np.angle(complex_array), dtype=np.float32)
    if not np.isfinite(phase).all():
        raise ValueError(f"MAT phase contains non-finite values: {path}")

    range_axis = np.asarray(payload.get("Range_plot"), dtype=np.float64).reshape(-1)
    velocity_axis = np.asarray(payload.get("Vr"), dtype=np.float64).reshape(-1)
    if amplitude.ndim != 2 or range_axis.shape != (amplitude.shape[0],):
        raise ValueError(f"MAT range axis does not match amplitude: {path}")
    if velocity_axis.shape != (amplitude.shape[1],):
        raise ValueError(f"MAT velocity axis does not match amplitude: {path}")
    if not np.isfinite(range_axis).all() or not np.isfinite(velocity_axis).all():
        raise ValueError(f"MAT physical axes contain non-finite values: {path}")
    debug = payload.get("debug_info")
    if debug is None:
        raise ValueError(f"MAT lacks debug_info: {path}")
    target_range_m = _debug_number(debug, "detected_range_m", path=path)
    metadata = {
        "schema_version": ASD_RD_SCHEMA,
        "source_path": source_path,
        "track_point_index": int(track_point_index),
        "raw_frame_index": int(frame_index),
        "mat_frame_number": int(frame_index) + 1,
        "time_seconds": _finite(frame_time_seconds, name="RD time_seconds"),
        "time_semantics": "parent_relative_cpi_elapsed_seconds",
        "shape": [int(value) for value in amplitude.shape],
        "dtype": "float32",
        "byte_order": "little-endian",
        "layout": "C-order",
        "amplitude_unit": "dB",
        "amplitude_mode": "db",
        "phase_unit": "rad",
        "target_range_m": target_range_m,
        "target_range_source": "MAT.debug_info.detected_range_m",
        "range_axis_m": range_axis.tolist(),
        "velocity_axis_mps": velocity_axis.tolist(),
        "amplitude_sanitization": sanitized.audit,
    }
    return amplitude, phase, metadata


def _build_parent_package(
    adapter: CQ08Adapter,
    parent: Any,
    root: Path,
) -> dict[str, Any]:
    events = adapter.discover_events(parent)
    if not events:
        raise ValueError(f"CQ08 parent has no events: {parent.parent_sample_id}")
    event_root = root / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    cumulative_rd_time = 0.0
    frame_count = 0
    rd_event_count = 0
    rd_shape_counts: dict[str, int] = {}
    for event in events:
        event_index = int(event.causal_event_index)
        event_path = event_root / f"{event_index:04d}"
        event_path.mkdir(parents=True, exist_ok=True)
        raw_values = dict(event.metadata.get("tr_values") or {})
        tr_values: dict[str, float] = {}
        for name in ASD_TR_FIELDS:
            if raw_values.get(name) is None:
                raise ValueError(
                    f"CQ08 event has no finite {name}: {event.sample_id}"
                )
            tr_values[name] = _finite(raw_values[name], name=f"{event.sample_id}:{name}")
        timestamp_seconds = _finite(
            raw_values.get("time_seconds"),
            name=f"{event.sample_id}:timestamp_seconds",
        )
        tr_payload = {
            "schema_version": ASD_TR_SCHEMA,
            "event_index": event_index,
            "track_point_index": int(event.metadata.get("track_point_index", event_index + 1)),
            "timestamp_seconds": timestamp_seconds,
            "tr": tr_values,
            "source": {
                "sample_id": str(event.sample_id),
                "source_name": "CQ08",
                "track_time_raw": event.metadata.get("track_time_raw"),
                "field_semantics": {
                    "range_m": "CQ08 track filtered range",
                    "azimuth_deg": "CQ08 track filtered azimuth",
                    "elevation_deg": "CQ08 track filtered elevation",
                    "radial_speed_mps": "CQ08 point Doppler speed",
                    "amplitude_db": "CQ08 point linear magnitude retained under legacy field name",
                    "snr_db": "10*log10(CQ08 point linear SNR)",
                    "speed_mps": "CQ08 track full speed",
                    "course_deg": "CQ08 track course",
                },
            },
        }
        _write_json(event_path / "tr.json", tr_payload)

        raw_paths = list(event.metadata.get("rd_frame_paths") or [])
        frame_metadata = list(event.metadata.get("rd_frame_metadata") or [])
        if not raw_paths:
            continue
        rd_event_count += 1
        rd_root = event_path / "rd"
        rd_root.mkdir(parents=True, exist_ok=True)
        for position, raw_path in enumerate(raw_paths):
            path = resolve_frame_source_path(
                adapter.project_root,
                str(raw_path),
                source_root=adapter.source_root,
                source_config=adapter.source_config,
            )
            source_meta = (
                dict(frame_metadata[position])
                if position < len(frame_metadata)
                and isinstance(frame_metadata[position], Mapping)
                else {}
            )
            duration = max(0.0, float(source_meta.get("duration_seconds") or 0.0))
            frame_number = int(source_meta.get("frame_index", position + 1))
            amplitude, phase, metadata = _read_raw_frame(
                path,
                frame_time_seconds=cumulative_rd_time,
                frame_index=frame_number - 1,
                track_point_index=int(event.metadata.get("track_point_index", event_index + 1)),
                source_path=str(raw_path),
            )
            frame_dir = rd_root / f"{position:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            amplitude.astype("<f4", copy=False).tofile(frame_dir / "amplitude.bin")
            phase.astype("<f4", copy=False).tofile(frame_dir / "phase.bin")
            metadata.update(
                {
                    "event_index": event_index,
                    "frame_position": position,
                    "rd_observation_duration_seconds": duration,
                }
            )
            _write_json(frame_dir / "meta.json", metadata)
            cumulative_rd_time += duration
            frame_count += 1
            shape_name = "x".join(str(value) for value in amplitude.shape)
            rd_shape_counts[shape_name] = rd_shape_counts.get(shape_name, 0) + 1

    raw_label_id = parent.metadata.get("raw_label_id")
    source_class = {
        "DroneTarget": "Drone",
        "BirdTarget": "Bird",
        "BalloonTarget": "Balloon",
        "ClutterTarget": "Clutter",
        "UnknownTarget": "Unknown",
    }.get(str(parent.label_name), str(parent.label_name))
    manifest = {
        "schema_version": ASD_PACKAGE_SCHEMA,
        "package_type": "AeroSight ASD raw physical RD",
        "parent_sample_id": str(parent.parent_sample_id),
        "label_id": int(parent.label_id) if parent.label_id is not None else None,
        "label_name": str(parent.label_name),
        "timestamp_unit": "s",
        "event_index_basis": "zero_based_causal_order",
        "source": {
            "dataset": "CQ08",
            "session": "cq08",
            "track_id": str(parent.track_id),
            "raw_label_id": int(raw_label_id) if raw_label_id is not None else None,
            "mat_file_class": source_class,
            "source_parent_sample_id": str(parent.parent_sample_id),
        },
        "tr": {
            "schema_version": ASD_TR_SCHEMA,
            "fields": list(ASD_TR_FIELDS),
            "numeric_only": True,
            "dt_fields_forbidden": ["dt_seconds", "dt_valid", "dt_reason"],
        },
        "rd": {
            "schema_version": ASD_RD_SCHEMA,
            "representation": "raw_physical_frame",
            "pairing": "CQ08 MAT.debug_info.index_trackPointNo",
            "frame_order": "mat_frame_index_ascending",
            "time_semantics": "parent_relative_cpi_elapsed_seconds",
            "max_frames_per_event": 12,
            "amplitude_mode": "db",
            "amplitude_unit": "dB",
            "phase_unit": "rad",
            "dtype": "float32",
            "byte_order": "little-endian",
            "layout": "C-order",
            "canonicalization_version": "RADAR-LN3-CANONICAL-RD-INTERP-V1",
            "canonical_shape": [21, 384],
        },
        "counts": {
            "event_count": len(events),
            "rd_event_count": rd_event_count,
            "rd_frame_count": frame_count,
            "rd_frame_shape_counts": rd_shape_counts,
        },
        "provenance": {
            "source_track_path": str(parent.source_path),
            "source_point_path": str(parent.metadata.get("point_source_path", "")),
            "source_root": "CQ08",
            "generated_by": "BuildASDFromCQ08.py",
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _zip_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source).as_posix())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def build(
    *,
    project_root: Path,
    track_id: str,
    output_path: Path,
    source_root: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    target = output_path if output_path.is_absolute() else project_root / output_path
    target = target.resolve()
    if target.suffix.lower() != ".asd":
        raise ValueError("output_path must use the .asd extension")
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    adapter = CQ08Adapter(
        project_root,
        source_root=source_root,
        parent_limit=None,
        verify_files=False,
    )
    wanted = f"cq08|track|{str(track_id)}"
    parents = [parent for parent in adapter.discover_parents() if parent.parent_sample_id == wanted]
    if len(parents) != 1:
        raise ValueError(f"unable to resolve one CQ08 parent: {wanted}")
    temporary_root = Path(tempfile.mkdtemp(prefix="asd-build-", dir=target.parent))
    try:
        manifest = _build_parent_package(adapter, parents[0], temporary_root)
        _zip_directory(temporary_root, target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return {
        "path": str(target),
        "schema_version": ASD_PACKAGE_SCHEMA,
        "parent_sample_id": wanted,
        "event_count": int(manifest["counts"]["event_count"]),
        "rd_event_count": int(manifest["counts"]["rd_event_count"]),
        "rd_frame_count": int(manifest["counts"]["rd_frame_count"]),
        "amplitude_mode": "db",
        "canonical_shape": [21, 384],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a standard ASD package from CQ08")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/asd-reference/cq08_track.asd"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build(
            project_root=args.project_root.resolve(),
            track_id=str(args.track_id),
            output_path=args.output,
            source_root=args.source_root,
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ASD build failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
