from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path(
    "src-v3/processed/report/RADAR-LN3-track-0333-original-echo-audit/matlab-v2"
)
DEFAULT_RAW = Path("OriData/23/CQ-08/原始回波/333_Label_3.dat")
EXPECTED_RAW_SHA256 = "70bee1da3bf665d8436cd855d5d32ea97f9737fa05457388643b70ff60ef27a0"
EXPECTED_COUNTS = {16: 4, 17: 4, 18: 4, 19: 2, 20: 2, 21: 2, 22: 4}


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate(project_root: Path, report_dir: Path, raw_path: Path) -> dict[str, Any]:
    summary_path = report_dir / "summary.json"
    frame_path = report_dir / "frame-comparison.csv"
    coverage_path = report_dir / "point-coverage.csv"
    target_path = report_dir / "raw-target-records.csv"
    rd_plot = report_dir / "rd-comparison.png"
    coverage_plot = report_dir / "point-coverage.png"
    required = (
        summary_path,
        frame_path,
        coverage_path,
        target_path,
        rd_plot,
        coverage_plot,
        raw_path,
    )
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Track 333 MATLAB-v2 audit output missing/empty: {missing}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "RADAR-LN3-track-0333-original-echo-matlab-audit-v2":
        raise ValueError("unexpected Track 333 MATLAB audit schema")
    if summary.get("status") != "passed":
        raise ValueError(f"Track 333 MATLAB-v2 audit did not pass: {summary.get('status')}")
    if int(summary.get("raw_frame_count", -1)) != 22 or int(summary.get("mat_file_count", -1)) != 22:
        raise ValueError("Track 333 MATLAB-v2 raw/MAT frame count changed")
    if summary.get("raw_point_indices") != list(range(16, 23)):
        raise ValueError("Track 333 MATLAB-v2 raw point values changed")
    if summary.get("raw_points_1_to_15_present") is not False:
        raise ValueError("Track 333 MATLAB-v2 unexpectedly found points 1-15")
    if summary.get("raw_points_16_to_22_present") is not True:
        raise ValueError("Track 333 MATLAB-v2 lacks a point in 16-22")
    required_true = (
        "raw_contiguous_from_zero_to_eof",
        "all_raw_target_records_match_mat_debug",
        "all_point_indices_match",
        "all_metadata_match",
        "all_iq_shapes_match",
        "all_iq_numeric_match",
        "all_rd_shapes_match",
        "all_rd_numeric_match",
        "all_point_coverage_raw_matches_mat",
    )
    failed = [field for field in required_true if summary.get(field) is not True]
    if failed:
        raise ValueError(f"Track 333 MATLAB-v2 audit has failed checks: {failed}")
    for field in (
        "maximum_iq_absolute_error",
        "maximum_iq_relative_error",
        "maximum_rd_absolute_error",
        "maximum_rd_relative_error",
    ):
        if float(summary.get(field, -1.0)) != 0.0:
            raise ValueError(f"Track 333 MATLAB-v2 nonzero numeric error: {field}")
    recoverability = summary.get("recoverability", {})
    if (
        recoverability.get("conclusion")
        != "source_raw_has_no_frames_for_track_points_1_to_15"
        or recoverability.get("points_1_to_15_recoverable_from_this_dat") is not False
    ):
        raise ValueError("Track 333 MATLAB-v2 recoverability conclusion changed")
    if summary.get("data_rebuild") != "未执行" or summary.get("training") != "未执行":
        raise ValueError("Track 333 MATLAB-v2 claims an out-of-scope mutation")

    with frame_path.open("r", encoding="utf-8-sig", newline="") as stream:
        frames = list(csv.DictReader(stream))
    if len(frames) != 22 or not all(row.get("status") == "matched" for row in frames):
        raise ValueError("Track 333 MATLAB-v2 does not contain 22 matched frames")
    point_counts = Counter(int(row["raw_track_point_index"]) for row in frames)
    if dict(sorted(point_counts.items())) != EXPECTED_COUNTS:
        raise ValueError(f"Track 333 MATLAB-v2 frame point counts changed: {point_counts}")
    if not all(
        _as_bool(row[field])
        for row in frames
        for field in (
            "raw_target_match",
            "point_index_match",
            "metadata_match",
            "iq_numeric_match",
            "rd_numeric_match",
        )
    ):
        raise ValueError("Track 333 MATLAB-v2 frame-level match flag failed")

    with coverage_path.open("r", encoding="utf-8-sig", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    if len(coverage) != 22 or not all(_as_bool(row["raw_matches_mat"]) for row in coverage):
        raise ValueError("Track 333 MATLAB-v2 point coverage does not match")
    for row in coverage:
        point = int(row["track_point_index"])
        expected = EXPECTED_COUNTS.get(point, 0)
        if int(row["raw_frame_count"]) != expected or int(row["mat_frame_count"]) != expected:
            raise ValueError(f"Track 333 MATLAB-v2 coverage count mismatch at point {point}")

    raw_hash = _sha256(raw_path)
    if raw_hash != EXPECTED_RAW_SHA256 or summary.get("raw_echo_sha256") != raw_hash:
        raise ValueError("Track 333 raw echo SHA-256 changed")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-track-0333-matlab-v2-validation-v1",
        "project_root": str(project_root),
        "report_dir": str(report_dir),
        "summary_sha256": _sha256(summary_path),
        "raw_echo_sha256": raw_hash,
        "raw_frame_count": 22,
        "mat_frame_count": 22,
        "raw_point_indices": list(range(16, 23)),
        "raw_frame_counts_by_point": {str(key): value for key, value in EXPECTED_COUNTS.items()},
        "all_iq_numeric_match": True,
        "all_rd_numeric_match": True,
        "all_point_coverage_raw_matches_mat": True,
        "points_1_to_15_recoverable_from_this_dat": False,
        "data_rebuild": "未执行",
        "training": "未执行",
        "test": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the corrected Track 333 MATLAB-v2 audit.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    root = args.project_root.resolve()
    validate(root, _resolve(root, args.report_dir), _resolve(root, args.raw))


if __name__ == "__main__":
    main()
