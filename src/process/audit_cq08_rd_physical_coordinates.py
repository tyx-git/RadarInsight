from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


SPEED_OF_LIGHT_MPS = 300_000_000.0
REQUIRED_DEBUG_FIELDS = {"freq_hz", "prt_num", "prt_s", "rd_result_size", "vr_start", "vr_end"}


def _scalar(debug: object, name: str) -> float:
    values = np.asarray(getattr(debug, name)).reshape(-1)
    if values.size != 1:
        raise ValueError(f"debug_info.{name} is not scalar")
    return float(values[0])


def _shape(debug: object) -> tuple[int, int]:
    values = np.asarray(getattr(debug, "rd_result_size")).reshape(-1)
    if values.size != 2:
        raise ValueError("debug_info.rd_result_size is not two-dimensional")
    return int(values[0]), int(values[1])


def _range(values: list[float]) -> dict[str, float]:
    return {"min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values))}


def audit(project_root: Path, frame_index_name: str, output_name: str, start_frame: int, max_frames: int, progress_interval: int) -> dict[str, object]:
    """Audit whether original CQ08 MAT metadata defines per-frame velocity axes."""
    index_path = project_root / "data" / "processed" / frame_index_name
    frame_index = pd.read_csv(index_path)
    start_frame = max(0, int(start_frame))
    frame_index = frame_index.iloc[start_frame:].copy()
    if max_frames > 0:
        frame_index = frame_index.iloc[:max_frames].copy()
    counters: Counter[str] = Counter()
    field_sets: Counter[str] = Counter()
    metrics: dict[str, dict[str, list[float]]] = {}
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for position, row in enumerate(frame_index.itertuples(index=False), start=1):
        relative_path = Path(str(row.rd_mat_path))
        path = relative_path if relative_path.is_absolute() else project_root / relative_path
        try:
            payload = loadmat(path, variable_names=["debug_info"], squeeze_me=True, struct_as_record=False)
            debug = payload["debug_info"]
            names = set(getattr(debug, "_fieldnames", []) or [])
            field_sets["|".join(sorted(names))] += 1
            missing = REQUIRED_DEBUG_FIELDS - names
            if missing:
                raise ValueError(f"missing:{','.join(sorted(missing))}")
            range_bins, doppler_bins = _shape(debug)
            frequency_hz = _scalar(debug, "freq_hz")
            prt_num = _scalar(debug, "prt_num")
            prt_seconds = _scalar(debug, "prt_s")
            vr_start = _scalar(debug, "vr_start")
            vr_end = _scalar(debug, "vr_end")
            bin_width = (vr_end - vr_start) / max(1, doppler_bins - 1)
            unambiguous_velocity = SPEED_OF_LIGHT_MPS / (4.0 * frequency_hz * prt_seconds)
            theoretical_bin_width = 2.0 * unambiguous_velocity / prt_num
            shape_metrics = metrics.setdefault(str(row.rd_shape), {key: [] for key in ("bin_width_mps", "theoretical_bin_width_mps", "relative_bin_error", "vr_start_mps", "vr_end_mps", "frequency_hz", "prt_seconds")})
            shape_metrics["bin_width_mps"].append(bin_width)
            shape_metrics["theoretical_bin_width_mps"].append(theoretical_bin_width)
            shape_metrics["relative_bin_error"].append(abs(bin_width - theoretical_bin_width) / max(abs(theoretical_bin_width), 1.0e-12))
            shape_metrics["vr_start_mps"].append(vr_start)
            shape_metrics["vr_end_mps"].append(vr_end)
            shape_metrics["frequency_hz"].append(frequency_hz)
            shape_metrics["prt_seconds"].append(prt_seconds)
            counters["frames_with_required_fields"] += 1
            if f"{range_bins}x{doppler_bins}" != str(row.rd_shape):
                counters["metadata_shape_mismatch"] += 1
            if int(round(prt_num)) != doppler_bins:
                counters["prt_num_doppler_width_mismatch"] += 1
            if not math.isfinite(bin_width) or bin_width <= 0.0:
                counters["invalid_velocity_grid"] += 1
            if abs(vr_start + unambiguous_velocity) > max(1.0e-4, abs(unambiguous_velocity) * 1.0e-4):
                counters["unambiguous_velocity_start_mismatch"] += 1
        except Exception as exc:
            counters["read_or_parse_error"] += 1
            if len(failures) < 100:
                failures.append({"path": str(relative_path), "error": repr(exc)})
        if progress_interval and position % progress_interval == 0:
            print(f"[cq08-rd-physical-audit] {position}/{len(frame_index)} valid={counters['frames_with_required_fields']} errors={counters['read_or_parse_error']} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    valid = int(counters["frames_with_required_fields"])
    summary = {
        "scope": "read-only original-MAT physical-coordinate audit; no rebuild, training, fusion, threshold tuning, or test evaluation",
        "frame_index": index_path.relative_to(project_root).as_posix(), "start_frame": start_frame,
        "frames_requested": len(frame_index), "frames_with_required_fields": valid,
        "required_field_coverage": valid / len(frame_index) if len(frame_index) else 0.0,
        "required_debug_fields": sorted(REQUIRED_DEBUG_FIELDS), "distinct_debug_field_sets": dict(field_sets),
        "counters": dict(counters),
        "by_source_shape": {shape: {name: _range(values) for name, values in values_by_name.items()} for shape, values_by_name in sorted(metrics.items())},
        "velocity_grid_contract": {"metadata_grid": "linspace(vr_start, vr_end, rd_result_size[1])", "theoretical_unambiguous_velocity_mps": "c / (4 * freq_hz * prt_s)", "theoretical_bin_width_mps": "2 * unambiguous_velocity / prt_num"},
        "failures": failures, "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = project_root / "tmp" / output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CQ08 original-MAT physical Doppler coordinates.")
    parser.add_argument("--frame-index-name", default="cq08_rd_frame_index_v1.csv")
    parser.add_argument("--output-name", default="20260727-stage009i-cq08-rd-physical-coordinate-audit.json")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=2_000)
    args = parser.parse_args()
    summary = audit(Path.cwd(), args.frame_index_name, args.output_name, args.start_frame, args.max_frames, args.progress_interval)
    print(json.dumps({key: summary[key] for key in ("frames_requested", "frames_with_required_fields", "counters", "elapsed_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
