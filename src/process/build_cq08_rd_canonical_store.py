from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import zoom


CANONICAL_STORE_VERSION = "cq08_rd_canonical_v1"
CANONICAL_SHAPE = (31, 64)


def _safe_name(value: object) -> str:
    return str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")


def _unwrap_scalar(value: object) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError("expected scalar MAT metadata")
    return int(array[0])


def _observation_seconds(debug: object) -> float:
    try:
        return float(getattr(debug, "cpi_count")) * float(getattr(debug, "prt_num")) * float(getattr(debug, "prt_s"))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _resize_plane(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected two-dimensional RD frame, got shape={values.shape}")
    if values.shape == CANONICAL_SHAPE:
        return values
    factors = (CANONICAL_SHAPE[0] / values.shape[0], CANONICAL_SHAPE[1] / values.shape[1])
    return zoom(values, factors, order=1, mode="nearest", prefilter=False).astype(np.float32)


def _canonicalize(amplitude: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amplitude = _resize_plane(np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0))
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    phase_sin = _resize_plane(np.sin(phase))
    phase_cos = _resize_plane(np.cos(phase))
    return amplitude, np.arctan2(phase_sin, phase_cos).astype(np.float32)


def _load_frame(path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    payload = loadmat(
        path,
        variable_names=["data_proc_MTD_result_db", "data_proc_MTD_result", "index_trackPointNo", "debug_info"],
        squeeze_me=True,
        struct_as_record=False,
    )
    amplitude = np.asarray(payload["data_proc_MTD_result_db"], dtype=np.float32)
    complex_rd = payload.get("data_proc_MTD_result")
    if complex_rd is None:
        phase = np.zeros_like(amplitude, dtype=np.float32)
    else:
        phase = np.angle(np.asarray(complex_rd, dtype=np.complex64)).astype(np.float32)
    return amplitude, phase, _unwrap_scalar(payload["index_trackPointNo"]), _observation_seconds(payload.get("debug_info"))


def _load_frame_metadata(path: Path) -> tuple[tuple[int, ...], int, float]:
    payload = loadmat(
        path,
        variable_names=["data_proc_MTD_result_db", "index_trackPointNo", "debug_info"],
        squeeze_me=True,
        struct_as_record=False,
    )
    shape = tuple(int(value) for value in np.asarray(payload["data_proc_MTD_result_db"]).shape)
    return shape, _unwrap_scalar(payload["index_trackPointNo"]), _observation_seconds(payload.get("debug_info"))


def build_cq08_rd_canonical_store(
    project_root: Path,
    *,
    store_name: str = CANONICAL_STORE_VERSION,
    output_name: str = "cq08_rd_canonical_frame_index_v1.csv",
    summary_name: str = "cq08_rd_canonical_frame_index_v1.summary.json",
    max_parents: int = 0,
    track_ids: set[int] | None = None,
    progress_interval: int = 5000,
    overwrite: bool = False,
    resume: bool = False,
    reference_frame_index_name: str = "cq08_rd_frame_index_v1.csv",
) -> dict[str, object]:
    """Write one full-frame, fixed-grid CQ08 tensor per parent trajectory."""
    manifest_path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    records = [record for record in json.loads(manifest_path.read_text(encoding="utf-8")) if bool(record.get("is_usable"))]
    records.sort(key=lambda record: int(record["track_id"]))
    if track_ids:
        records = [record for record in records if int(record["track_id"]) in track_ids]
    if max_parents > 0:
        records = records[: int(max_parents)]
    processed_root = project_root / "data" / "processed"
    store_root = processed_root / store_name
    output_path = processed_root / output_name
    summary_path = processed_root / summary_name
    if output_path.exists() or summary_path.exists():
        if not overwrite:
            raise FileExistsError("canonical index outputs already exist; pass overwrite=True")
    elif store_root.exists() and not (overwrite or resume):
        raise FileExistsError("canonical outputs already exist; pass overwrite=True")
    store_root.mkdir(parents=True, exist_ok=True)
    reference_rows_by_parent: dict[str, pd.DataFrame] = {}
    reference_path = processed_root / reference_frame_index_name
    if resume and reference_path.exists():
        reference_index = pd.read_csv(reference_path, low_memory=False)
        required_reference_columns = {
            "parent_sample_id", "raw_frame_index", "rd_shape", "track_point_index",
            "rd_observation_duration_seconds", "rd_mat_path",
        }
        missing_reference_columns = required_reference_columns - set(reference_index.columns)
        if missing_reference_columns:
            raise ValueError(f"reference frame index missing columns: {sorted(missing_reference_columns)}")
        reference_rows_by_parent = {
            str(parent_id): group.sort_values("raw_frame_index").reset_index(drop=True)
            for parent_id, group in reference_index.groupby("parent_sample_id", sort=False)
        }

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    shape_counts: Counter[str] = Counter()
    total_frames = sum(len(record.get("rd_mat_files", [])) for record in records)
    scanned = written = parent_count_written = 0
    started = time.perf_counter()
    for record in records:
        sample_id = f"cq08|track|{int(record['track_id'])}"
        tensor_path = store_root / f"{_safe_name(sample_id)}_rd_tensor.npz"
        reuse_existing_tensor = resume and tensor_path.exists()
        amplitudes: list[np.ndarray] = []
        phases: list[np.ndarray] = []
        time_raw: list[float] = []
        parent_rows: list[dict[str, object]] = []
        reference_rows = reference_rows_by_parent.get(sample_id) if reuse_existing_tensor else None
        if reference_rows is not None:
            for reference_row in reference_rows.to_dict(orient="records"):
                raw_frame_index = int(reference_row["raw_frame_index"])
                source_shape = str(reference_row["rd_shape"])
                parent_rows.append(
                    {
                        "schema_version": CANONICAL_STORE_VERSION,
                        "parent_sample_id": sample_id,
                        "track_id": int(record["track_id"]),
                        "source_name": "cq08_track",
                        "label_name": str(record["label_name"]),
                        "raw_frame_index": raw_frame_index,
                        "tensor_frame_index": raw_frame_index,
                        "is_tensor_frame_kept": True,
                        "rd_shape": source_shape,
                        "canonical_rd_shape": "x".join(str(value) for value in CANONICAL_SHAPE),
                        "track_point_index": int(reference_row["track_point_index"]),
                        "rd_observation_duration_seconds": float(reference_row["rd_observation_duration_seconds"]),
                        "rd_mat_path": str(reference_row["rd_mat_path"]),
                        "rd_tensor_path": tensor_path.relative_to(project_root).as_posix(),
                    }
                )
                shape_counts[source_shape] += 1
            scanned += len(parent_rows)
        for raw_frame_index, path_value in ([] if reference_rows is not None else enumerate(record.get("rd_mat_files", []))):
            try:
                if reuse_existing_tensor:
                    shape, point_index, observation = _load_frame_metadata(Path(path_value))
                    tensor_frame_index = raw_frame_index
                else:
                    amplitude, phase, point_index, observation = _load_frame(Path(path_value))
                    shape = amplitude.shape
                    canonical_amplitude, canonical_phase = _canonicalize(amplitude, phase)
                    tensor_frame_index = len(amplitudes)
                    amplitudes.append(canonical_amplitude.astype(np.float16))
                    phases.append(canonical_phase.astype(np.float16))
                    time_raw.append(float(raw_frame_index))
                source_shape = "x".join(str(value) for value in shape)
                shape_counts[source_shape] += 1
                parent_rows.append(
                    {
                        "schema_version": CANONICAL_STORE_VERSION,
                        "parent_sample_id": sample_id,
                        "track_id": int(record["track_id"]),
                        "source_name": "cq08_track",
                        "label_name": str(record["label_name"]),
                        "raw_frame_index": raw_frame_index,
                        "tensor_frame_index": tensor_frame_index,
                        "is_tensor_frame_kept": True,
                        "rd_shape": source_shape,
                        "canonical_rd_shape": "x".join(str(value) for value in CANONICAL_SHAPE),
                        "track_point_index": point_index,
                        "rd_observation_duration_seconds": observation,
                        "rd_mat_path": Path(path_value).as_posix(),
                        "rd_tensor_path": tensor_path.relative_to(project_root).as_posix(),
                    }
                )
            except Exception as exc:
                errors.append({"parent_sample_id": sample_id, "raw_frame_index": str(raw_frame_index), "error": repr(exc)})
            scanned += 1
            if progress_interval > 0 and scanned % progress_interval == 0:
                elapsed = time.perf_counter() - started
                print(f"[cq08-canonical-store] {scanned}/{total_frames} written={written} elapsed={elapsed:.1f}s", flush=True)
        if reuse_existing_tensor:
            with np.load(tensor_path) as handle:
                existing_count = int(handle["amplitude"].shape[0])
            if existing_count != len(parent_rows):
                errors.append({"parent_sample_id": sample_id, "raw_frame_index": "", "error": f"resume_tensor_count_mismatch:{existing_count}!={len(parent_rows)}"})
                continue
        elif not amplitudes:
            errors.append({"parent_sample_id": sample_id, "raw_frame_index": "", "error": "no_valid_frames"})
            continue
        if not reuse_existing_tensor:
            temporary_path = tensor_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary_path,
                amplitude=np.stack(amplitudes, axis=0),
                phase=np.stack(phases, axis=0),
                time_raw=np.asarray(time_raw, dtype=np.float32),
            )
            os.replace(temporary_path, tensor_path)
        rows.extend(parent_rows)
        written += len(parent_rows)
        parent_count_written += 1

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    summary = {
        "schema_version": CANONICAL_STORE_VERSION,
        "scope": "CQ08 original MAT frames canonicalized to a common range-Doppler grid; no source-shape frame is intentionally dropped.",
        "canonical_shape": list(CANONICAL_SHAPE),
        "amplitude_dtype": "float16",
        "phase_dtype": "float16",
        "parent_count_requested": len(records),
        "parent_count_written": parent_count_written,
        "raw_frame_count": scanned,
        "canonical_tensor_frame_count": written,
        "dropped_shape_frame_count": 0,
        "error_count": len(errors),
        "errors": errors[:100],
        "source_shape_counts": dict(shape_counts),
        "elapsed_seconds": time.perf_counter() - started,
        "store_path": store_root.relative_to(project_root).as_posix(),
        "frame_index_path": output_path.relative_to(project_root).as_posix(),
        "fusion_policy": "track_point_anchor remains diagnostic-only and must not enter joint fusion without independent timing evidence.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build full-frame CQ08 canonical RD tensors without shape filtering.")
    parser.add_argument("--store-name", default=CANONICAL_STORE_VERSION)
    parser.add_argument("--output-name", default="cq08_rd_canonical_frame_index_v1.csv")
    parser.add_argument("--summary-name", default="cq08_rd_canonical_frame_index_v1.summary.json")
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--track-ids", default="", help="Comma-separated CQ08 track IDs for a focused smoke build.")
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed parent tensors when final frame index is absent.")
    parser.add_argument("--reference-frame-index-name", default="cq08_rd_frame_index_v1.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(__file__).resolve().parents[2]
    track_ids = {int(value.strip()) for value in args.track_ids.split(",") if value.strip()}
    summary = build_cq08_rd_canonical_store(
        project_root,
        store_name=args.store_name,
        output_name=args.output_name,
        summary_name=args.summary_name,
        max_parents=args.max_parents,
        track_ids=track_ids or None,
        progress_interval=args.progress_interval,
        overwrite=args.overwrite,
        resume=args.resume,
        reference_frame_index_name=args.reference_frame_index_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
