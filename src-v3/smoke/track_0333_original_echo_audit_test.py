from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from report.track_0333_original_echo_audit import (
    DEFAULT_MAT_DIR,
    DEFAULT_PHYSICAL_INDEX,
    DEFAULT_RAW,
    DEFAULT_V3_INDEX,
    _resolve,
    run,
)
from validate.track_0333_matlab_audit import (
    DEFAULT_REPORT_DIR as DEFAULT_MATLAB_REPORT_DIR,
    validate as validate_matlab_audit,
)


DEFAULT_REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-track-0333-original-echo-audit/summary.json"
)
DEFAULT_OUTPUT = Path(
    "src-v3/smoke/RADAR-LN3-track-0333-original-echo-audit-contract-result.json"
)
EXPECTED_POINT_COUNTS = {
    "16": 4,
    "17": 4,
    "18": 4,
    "19": 2,
    "20": 2,
    "21": 2,
    "22": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hashes(
    raw_path: Path,
    mat_dir: Path,
    v3_index: Path,
    physical_index: Path,
) -> dict[str, str]:
    paths = [raw_path, v3_index, physical_index]
    paths.extend(sorted(mat_dir.glob("*_333.mat")))
    return {str(path.resolve()): _sha256(path) for path in paths}


def _assert_contract(summary: dict[str, Any]) -> None:
    if summary.get("status") != "passed":
        raise AssertionError(f"Track 333 audit did not pass: {summary.get('status')}")
    parser = summary.get("raw_parser", {})
    if (
        int(parser.get("file_bytes", -1)) != 2_694_512
        or int(parser.get("parsed_bytes", -1)) != 2_694_512
        or int(parser.get("frame_count", -1)) != 22
        or parser.get("continuous_from_zero_to_eof") is not True
        or int(parser.get("unparsed_payload_bytes_total", -1)) != 0
    ):
        raise AssertionError("Track 333 raw frame coverage changed")
    if summary.get("raw_point_indices") != list(range(16, 23)):
        raise AssertionError("Track 333 raw point indices changed")
    if summary.get("raw_frame_counts_by_point") != EXPECTED_POINT_COUNTS:
        raise AssertionError("Track 333 raw frame counts changed")
    if summary.get("raw_points_1_to_15_present") is not False:
        raise AssertionError("Track 333 unexpectedly contains raw frames for points 1-15")
    if summary.get("raw_points_16_to_22_present") is not True:
        raise AssertionError("Track 333 lacks a raw point in 16-22")
    comparison = summary.get("raw_mat_comparison", {})
    for field in (
        "all_point_indices_match",
        "all_target_records_match_mat_debug",
        "all_metadata_match",
        "all_iq_shapes_match",
        "all_iq_numeric_match",
    ):
        if comparison.get(field) is not True:
            raise AssertionError(f"Track 333 raw/MAT comparison failed: {field}")
    if (
        float(comparison.get("maximum_iq_absolute_error", -1.0)) != 0.0
        or float(comparison.get("maximum_iq_relative_error", -1.0)) != 0.0
    ):
        raise AssertionError("Track 333 raw/MAT I/Q is no longer exact")
    mapping = summary.get("mapping_comparison", {})
    if not all(
        mapping.get(field) is True
        for field in (
            "all_points_raw_match_mat",
            "all_points_raw_match_physical_index",
            "all_points_raw_match_v3",
        )
    ):
        raise AssertionError("Track 333 raw/MAT/V3 mapping changed")
    recoverability = summary.get("recoverability", {})
    if (
        recoverability.get("conclusion")
        != "source_raw_has_no_frames_for_track_points_1_to_15"
        or recoverability.get("points_1_to_15_recoverable_from_this_dat") is not False
    ):
        raise AssertionError("Track 333 recoverability conclusion changed")
    if summary.get("training") != "未执行" or summary.get("test") != "未执行":
        raise AssertionError("Track 333 audit claims an out-of-scope experiment")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused immutable-input contract test for Track 333 raw echo audit."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw_path = _resolve(root, DEFAULT_RAW)
    mat_dir = _resolve(root, DEFAULT_MAT_DIR)
    v3_index = _resolve(root, DEFAULT_V3_INDEX)
    physical_index = _resolve(root, DEFAULT_PHYSICAL_INDEX)
    actual_report = _resolve(root, args.report)
    recorded = json.loads(actual_report.read_text(encoding="utf-8"))
    _assert_contract(recorded)
    matlab_validation = validate_matlab_audit(
        root,
        _resolve(root, DEFAULT_MATLAB_REPORT_DIR),
        raw_path,
    )

    before = _input_hashes(raw_path, mat_dir, v3_index, physical_index)
    smoke_root = root / "src-v3" / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="track-0333-audit-", dir=smoke_root) as temporary:
        rerun = run(
            root,
            raw_path,
            mat_dir,
            v3_index,
            physical_index,
            Path(temporary) / "report",
        )
        _assert_contract(rerun)
    after = _input_hashes(raw_path, mat_dir, v3_index, physical_index)
    if before != after:
        changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
        raise AssertionError(f"Track 333 audit modified immutable inputs: {changed}")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-track-0333-original-echo-audit-contract-test-v1",
        "actual_report": str(actual_report),
        "actual_report_sha256": _sha256(actual_report),
        "raw_frame_count": 22,
        "raw_point_indices": list(range(16, 23)),
        "raw_frame_counts_by_point": EXPECTED_POINT_COUNTS,
        "raw_points_1_to_15_present": False,
        "raw_mat_iq_exact": True,
        "raw_mat_physical_v3_mapping_exact": True,
        "points_1_to_15_recoverable_from_this_dat": False,
        "immutable_input_file_count": len(before),
        "immutable_inputs_unchanged": True,
        "matlab_independent_audit": "passed",
        "matlab_summary_sha256": matlab_validation["summary_sha256"],
        "matlab_all_iq_numeric_match": matlab_validation["all_iq_numeric_match"],
        "matlab_all_rd_numeric_match": matlab_validation["all_rd_numeric_match"],
        "matlab_all_point_coverage_raw_matches_mat": matlab_validation[
            "all_point_coverage_raw_matches_mat"
        ],
        "data_rebuild": "未执行",
        "training": "未执行",
        "test": "未执行",
    }
    output_path = _resolve(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
