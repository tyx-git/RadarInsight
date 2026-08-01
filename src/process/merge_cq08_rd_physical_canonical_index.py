from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def merge_and_validate(
    project_root: Path,
    *,
    output_name: str,
    part_pattern: str,
    reference_frame_index_name: str,
) -> dict[str, object]:
    processed_root = project_root / "data" / "processed"
    parts = sorted(processed_root.glob(part_pattern))
    if not parts:
        raise FileNotFoundError(f"no index parts found for {part_pattern}")
    frames = [pd.read_csv(path, low_memory=False) for path in parts]
    frame_index = pd.concat(frames, ignore_index=True)
    required = {
        "parent_sample_id", "raw_frame_index", "tensor_frame_index", "is_tensor_frame_kept",
        "rd_tensor_path", "canonical_rd_shape", "source_vr_start_mps", "source_vr_end_mps",
        "source_velocity_coverage_ratio", "anti_alias_sigma_source_bins",
        "source_range_offset_start_m", "source_range_offset_end_m", "source_range_coverage_ratio",
    }
    missing = required - set(frame_index.columns)
    if missing:
        raise ValueError(f"merged index missing columns: {sorted(missing)}")
    frame_index = frame_index.sort_values(["parent_sample_id", "raw_frame_index"]).reset_index(drop=True)
    reference = pd.read_csv(processed_root / reference_frame_index_name, low_memory=False)
    expected_pairs = set(zip(reference["parent_sample_id"].astype(str), reference["raw_frame_index"].astype(int)))
    actual_pairs = set(zip(frame_index["parent_sample_id"].astype(str), frame_index["raw_frame_index"].astype(int)))
    missing_reference_pairs = expected_pairs - actual_pairs
    unexpected_pairs = actual_pairs - expected_pairs
    duplicates = int(frame_index.duplicated(["parent_sample_id", "raw_frame_index"]).sum())
    expected_tensor = frame_index.groupby("parent_sample_id").cumcount().to_numpy()
    tensor_values = frame_index["tensor_frame_index"].to_numpy(dtype=np.int64)
    tensor_index_mismatch = int(np.count_nonzero(expected_tensor != tensor_values))
    tensor_paths = [project_root / Path(path) for path in frame_index["rd_tensor_path"].drop_duplicates().astype(str)]
    missing_tensors = [path.relative_to(project_root).as_posix() for path in tensor_paths if not path.exists()]
    invalid_shapes = int((frame_index["canonical_rd_shape"].astype(str) != "21x384").sum())
    invalid_coverage = int((frame_index["source_velocity_coverage_ratio"].astype(float) < 0.999999).sum())
    partial_range_coverage = int((frame_index["source_range_coverage_ratio"].astype(float) < 0.999999).sum())
    summary = {
        "scope": "physical canonical v3.1 common-support velocity/range part-index merge and structural validation; no training, fusion, threshold tuning, or test evaluation",
        "parts": [path.name for path in parts],
        "frame_count": int(len(frame_index)),
        "parent_count": int(frame_index["parent_sample_id"].nunique()),
        "reference_frame_count": int(len(reference)),
        "missing_reference_frame_count": len(missing_reference_pairs),
        "unexpected_frame_count": len(unexpected_pairs),
        "missing_reference_frames": [f"{parent}|{frame}" for parent, frame in sorted(missing_reference_pairs)[:100]],
        "unexpected_frames": [f"{parent}|{frame}" for parent, frame in sorted(unexpected_pairs)[:100]],
        "duplicate_parent_raw_frame_count": duplicates,
        "tensor_index_mismatch_count": tensor_index_mismatch,
        "missing_tensor_count": len(missing_tensors),
        "missing_tensors": missing_tensors[:100],
        "invalid_canonical_shape_count": invalid_shapes,
        "partial_velocity_coverage_count": invalid_coverage,
        "partial_range_coverage_count": partial_range_coverage,
        "anti_alias_frame_count": int((frame_index["anti_alias_sigma_source_bins"].astype(float) > 1.0e-6).sum()),
        "source_velocity_bounds_mps": {
            "min_start": float(frame_index["source_vr_start_mps"].min()),
            "max_end": float(frame_index["source_vr_end_mps"].max()),
        },
        "source_range_offset_bounds_m": {
            "min_start": float(frame_index["source_range_offset_start_m"].min()),
            "max_end": float(frame_index["source_range_offset_end_m"].max()),
        },
        "output_path": (processed_root / output_name).relative_to(project_root).as_posix(),
        "fusion_policy": "track_point_anchor remains diagnostic-only; physical coordinate recovery does not establish frame-level wall-clock synchronization.",
    }
    if any((duplicates, tensor_index_mismatch, len(missing_tensors), invalid_shapes, invalid_coverage, partial_range_coverage, len(missing_reference_pairs), len(unexpected_pairs))):
        raise ValueError(json.dumps(summary, ensure_ascii=False))
    output_path = processed_root / output_name
    frame_index.to_csv(output_path, index=False, encoding="utf-8")
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate physical canonical CQ08 frame-index partitions.")
    parser.add_argument("--output-name", default="cq08_rd_physical_canonical_frame_index_v3_1.csv")
    parser.add_argument("--part-pattern", default="cq08_rd_physical_canonical_frame_index_v3_1.part*.csv")
    parser.add_argument("--reference-frame-index-name", default="cq08_rd_frame_index_v1.csv")
    args = parser.parse_args()
    summary = merge_and_validate(
        Path(__file__).resolve().parents[2],
        output_name=args.output_name,
        part_pattern=args.part_pattern,
        reference_frame_index_name=args.reference_frame_index_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
