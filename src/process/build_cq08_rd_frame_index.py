from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


FRAME_INDEX_VERSION = "cq08_rd_frame_index_v1"


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


def build_cq08_rd_frame_index(
    project_root: Path,
    *,
    output_name: str = "cq08_rd_frame_index_v1.csv",
    summary_name: str = "cq08_rd_frame_index_v1.summary.json",
    max_parents: int = 0,
    progress_interval: int = 5000,
) -> dict[str, object]:
    manifest_records = json.loads((project_root / "data" / "manifests" / "cq08_unified_manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest_records if bool(record.get("is_usable"))]
    records.sort(key=lambda item: int(item["track_id"]))
    if max_parents > 0:
        records = records[: int(max_parents)]
    total_frames = sum(len(record.get("rd_mat_files", [])) for record in records)
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    shape_counts: Counter[str] = Counter()
    dropped_shape_counts: Counter[str] = Counter()
    parent_stats: list[dict[str, object]] = []
    scanned = 0
    kept_total = 0
    dropped_total = 0
    for record in records:
        track_id = int(record["track_id"])
        sample_id = f"cq08|track|{track_id}"
        target_shape: tuple[int, ...] | None = None
        kept_index = 0
        parent_kept = 0
        parent_dropped = 0
        for raw_frame_index, relative_path in enumerate(record.get("rd_mat_files", [])):
            path = project_root / Path(relative_path)
            try:
                payload = loadmat(
                    path,
                    variable_names=["data_proc_MTD_result_db", "index_trackPointNo", "debug_info"],
                    squeeze_me=True,
                    struct_as_record=False,
                )
                amplitude = np.asarray(payload["data_proc_MTD_result_db"])
                shape = tuple(int(value) for value in amplitude.shape)
                if target_shape is None:
                    target_shape = shape
                is_kept = shape == target_shape
                point_index = _unwrap_scalar(payload["index_trackPointNo"])
                tensor_frame_index = kept_index if is_kept else -1
                if is_kept:
                    kept_index += 1
                    parent_kept += 1
                    kept_total += 1
                else:
                    parent_dropped += 1
                    dropped_total += 1
                    dropped_shape_counts["x".join(map(str, shape))] += 1
                shape_counts["x".join(map(str, shape))] += 1
                rows.append(
                    {
                        "schema_version": FRAME_INDEX_VERSION,
                        "parent_sample_id": sample_id,
                        "track_id": track_id,
                        "source_name": "cq08_track",
                        "label_name": str(record["label_name"]),
                        "raw_frame_index": raw_frame_index,
                        "tensor_frame_index": tensor_frame_index,
                        "is_tensor_frame_kept": bool(is_kept),
                        "rd_shape": "x".join(map(str, shape)),
                        "target_rd_shape": "x".join(map(str, target_shape)),
                        "track_point_index": point_index,
                        "rd_observation_duration_seconds": _observation_seconds(payload.get("debug_info")),
                        "rd_mat_path": Path(relative_path).as_posix(),
                    }
                )
            except Exception as exc:
                errors.append({"parent_sample_id": sample_id, "error": f"{Path(relative_path).name}: {repr(exc)}"})
            scanned += 1
            if progress_interval > 0 and scanned % progress_interval == 0:
                elapsed = time.perf_counter() - start
                print(f"[cq08-frame-index] {scanned}/{total_frames} elapsed={elapsed:.1f}s rate={scanned / max(elapsed, 1.0e-6):.1f}/s")
        parent_stats.append(
            {
                "parent_sample_id": sample_id,
                "raw_frame_count": len(record.get("rd_mat_files", [])),
                "tensor_frame_count": parent_kept,
                "dropped_shape_frame_count": parent_dropped,
                "target_rd_shape": "x".join(map(str, target_shape or ())),
            }
        )
    output_path = project_root / "data" / "processed" / output_name
    summary_path = project_root / "data" / "processed" / summary_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    parents = pd.DataFrame(parent_stats)
    drift_parents = parents[parents["dropped_shape_frame_count"] > 0] if not parents.empty else parents
    summary = {
        "schema_version": FRAME_INDEX_VERSION,
        "scope": "CQ08 original MAT order to existing standardized tensor frame index; no RD tensor rewrite",
        "parent_count": int(len(parents)),
        "raw_frame_count": int(scanned),
        "tensor_frame_count": int(kept_total),
        "dropped_shape_frame_count": int(dropped_total),
        "parents_with_dropped_shape_frames": int(len(drift_parents)),
        "parent_dropped_shape_frame_count": {
            str(row["parent_sample_id"]): int(row["dropped_shape_frame_count"])
            for _, row in drift_parents.iterrows()
        },
        "shape_counts": dict(shape_counts),
        "dropped_shape_counts": dict(dropped_shape_counts),
        "error_count": int(len(errors)),
        "errors": errors[:100],
        "elapsed_seconds": time.perf_counter() - start,
        "frame_index_path": output_path.relative_to(project_root).as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build CQ08 MAT-to-standardized-tensor frame index without rewriting tensors.")
    parser.add_argument("--output-name", default="cq08_rd_frame_index_v1.csv")
    parser.add_argument("--summary-name", default="cq08_rd_frame_index_v1.summary.json")
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(__file__).resolve().parents[2]
    print(json.dumps(build_cq08_rd_frame_index(
        project_root,
        output_name=args.output_name,
        summary_name=args.summary_name,
        max_parents=args.max_parents,
        progress_interval=args.progress_interval,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
