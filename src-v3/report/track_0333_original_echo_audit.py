from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


FRAME_HEAD = 0xFA55FA55
FRAME_END = 0x55FA55FA
RANGE_BIN_COUNT = 31
TARGET_TRACK_ID = 333
TARGET_POINT_COUNT = 22

DEFAULT_RAW = Path("OriData/23/CQ-08/原始回波/333_Label_3.dat")
DEFAULT_MAT_DIR = Path("OriData/23/CQ-08/MAT")
DEFAULT_V3_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
DEFAULT_PHYSICAL_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
DEFAULT_OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-track-0333-original-echo-audit"
)


@dataclass(frozen=True)
class RawTargetRecord:
    frame_sequence: int
    target_record_index: int
    raw_target_identifier: int
    track_point_index_raw: int
    range_bin_raw: int
    doppler_bin_raw: int


@dataclass(frozen=True)
class RawFrame:
    frame_sequence: int
    frame_offset_bytes: int
    frame_length_words: int
    frame_length_bytes: int
    scan_angle_raw: int
    point_count: int
    frequency_mhz_raw: int
    frequency_hz: float
    cpi_count: int
    prt_num: int
    prt_raw: int
    prt_seconds: float
    data_length: int
    iq_rows: int
    iq_columns: int
    iq_sample_count_float32: int
    iq_finite_fraction: float
    unparsed_payload_bytes: int
    footer_ok: bool
    target_records: tuple[RawTargetRecord, ...]
    iq: np.ndarray


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_number(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_number(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _unpack_u32(buffer: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}I", buffer, offset)


def parse_raw_echo(path: Path) -> tuple[list[RawFrame], dict[str, Any]]:
    """Parse every contiguous frame in the CQ08 little-endian raw echo file."""

    payload = path.read_bytes()
    frames: list[RawFrame] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 16:
            raise ValueError(f"truncated frame header at byte offset {offset}")
        head, length_words, scan_angle, point_count = _unpack_u32(payload, offset, 4)
        if head != FRAME_HEAD:
            raise ValueError(
                f"non-contiguous or invalid frame head at offset {offset}: 0x{head:08X}"
            )
        frame_length = int(length_words) * 4
        if frame_length < 36 or offset + frame_length > len(payload):
            raise ValueError(
                f"invalid frame length at offset {offset}: {frame_length} bytes"
            )
        footer = _unpack_u32(payload, offset + frame_length - 4, 1)[0]
        if footer != FRAME_END:
            raise ValueError(
                f"invalid frame footer at offset {offset + frame_length - 4}: "
                f"0x{footer:08X}"
            )
        if point_count > 100_000:
            raise ValueError(f"implausible point count at offset {offset}: {point_count}")

        target_offset = offset + 16
        target_word_count = int(point_count) * 4
        target_words = _unpack_u32(payload, target_offset, target_word_count)
        metadata_offset = target_offset + target_word_count * 4
        frequency_mhz, cpi_count, prt_num, prt_raw, data_length = _unpack_u32(
            payload, metadata_offset, 5
        )
        if prt_num <= 0 or prt_num > 1_000_000:
            raise ValueError(f"invalid PRT count at offset {offset}: {prt_num}")
        sample_count = int(prt_num) * RANGE_BIN_COUNT * 2
        sample_offset = metadata_offset + 20
        sample_bytes = sample_count * 4
        footer_offset = offset + frame_length - 4
        if sample_offset + sample_bytes > footer_offset:
            raise ValueError(f"I/Q payload exceeds frame boundary at offset {offset}")
        raw_samples = np.frombuffer(
            payload,
            dtype="<f4",
            count=sample_count,
            offset=sample_offset,
        )
        iq = (raw_samples[0::2] + 1j * raw_samples[1::2]).reshape(
            (RANGE_BIN_COUNT, int(prt_num)), order="F"
        )
        frame_sequence = len(frames) + 1
        records = tuple(
            RawTargetRecord(
                frame_sequence=frame_sequence,
                target_record_index=index + 1,
                raw_target_identifier=int(target_words[index * 4]),
                track_point_index_raw=int(target_words[index * 4 + 1]),
                range_bin_raw=int(target_words[index * 4 + 2]),
                doppler_bin_raw=int(target_words[index * 4 + 3]),
            )
            for index in range(int(point_count))
        )
        frames.append(
            RawFrame(
                frame_sequence=frame_sequence,
                frame_offset_bytes=offset,
                frame_length_words=int(length_words),
                frame_length_bytes=frame_length,
                scan_angle_raw=int(scan_angle),
                point_count=int(point_count),
                frequency_mhz_raw=int(frequency_mhz),
                frequency_hz=float(frequency_mhz) * 1e6,
                cpi_count=int(cpi_count),
                prt_num=int(prt_num),
                prt_raw=int(prt_raw),
                prt_seconds=float(prt_raw) * 0.0125e-6,
                data_length=int(data_length),
                iq_rows=RANGE_BIN_COUNT,
                iq_columns=int(prt_num),
                iq_sample_count_float32=sample_count,
                iq_finite_fraction=float(np.isfinite(iq).mean()),
                unparsed_payload_bytes=footer_offset - (sample_offset + sample_bytes),
                footer_ok=True,
                target_records=records,
                iq=iq,
            )
        )
        offset += frame_length

    metadata = {
        "file_bytes": len(payload),
        "parsed_bytes": offset,
        "continuous_from_zero_to_eof": offset == len(payload),
        "frame_count": len(frames),
        "all_headers_valid": True,
        "all_footers_valid": all(frame.footer_ok for frame in frames),
        "all_iq_finite": all(frame.iq_finite_fraction == 1.0 for frame in frames),
        "unparsed_payload_bytes_total": sum(
            frame.unparsed_payload_bytes for frame in frames
        ),
    }
    return frames, metadata


def _mat_sequence(path: Path) -> int:
    match = re.search(r"_(\d{4})_333\.mat$", path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unexpected Track 333 MAT filename: {path.name}")
    return int(match.group(1))


def _debug_value(debug: Any, name: str, default: Any = None) -> Any:
    return getattr(debug, name, default) if debug is not None else default


def compare_mat_frames(frames: list[RawFrame], mat_dir: Path) -> list[dict[str, Any]]:
    mat_paths = sorted(mat_dir.glob("*_333.mat"), key=_mat_sequence)
    rows: list[dict[str, Any]] = []
    for index in range(max(len(frames), len(mat_paths))):
        frame = frames[index] if index < len(frames) else None
        mat_path = mat_paths[index] if index < len(mat_paths) else None
        row: dict[str, Any] = {
            "frame_sequence": index + 1,
            "raw_frame_available": frame is not None,
            "mat_available": mat_path is not None,
            "mat_path": str(mat_path) if mat_path is not None else "",
            "mat_filename_sequence": _mat_sequence(mat_path) if mat_path else None,
            "raw_track_point_indices_json": json.dumps(
                [record.track_point_index_raw for record in frame.target_records]
                if frame
                else []
            ),
            "raw_target_identifiers_json": json.dumps(
                [record.raw_target_identifier for record in frame.target_records]
                if frame
                else []
            ),
            "mat_track_point_index": None,
            "mat_track_no_info_head_json": "",
            "point_index_match": False,
            "target_record_match": False,
            "metadata_match": False,
            "iq_shape_match": False,
            "iq_max_absolute_error": None,
            "iq_max_relative_error": None,
            "iq_numeric_match": False,
            "rd_shape": "",
            "python_rd_numeric_comparison": "not_performed_matlab_taylorwin_is_authoritative",
            "status": "comparison_unavailable",
        }
        if frame is None or mat_path is None:
            rows.append(row)
            continue
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        mat_point = int(np.asarray(mat["index_trackPointNo"]).reshape(-1)[0])
        debug = mat.get("debug_info")
        debug_head = np.asarray(
            _debug_value(debug, "track_no_info_head", []), dtype=np.int64
        ).reshape(-1)
        mat_iq = np.asarray(mat["data_out"])
        mat_rd = np.asarray(mat.get("data_proc_MTD_result", []))
        raw_record_words = [
            value
            for record in frame.target_records
            for value in (
                record.raw_target_identifier,
                record.track_point_index_raw,
                record.range_bin_raw,
                record.doppler_bin_raw,
            )
        ]
        point_match = mat_point in {
            record.track_point_index_raw for record in frame.target_records
        }
        target_match = debug_head.tolist() in (
            [raw_record_words[position : position + 4]
             for position in range(0, len(raw_record_words), 4)]
        )
        prt_seconds = float(_debug_value(debug, "prt_s", math.nan))
        metadata_match = (
            math.isclose(
                float(_debug_value(debug, "freq_hz", math.nan)),
                frame.frequency_hz,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and int(_debug_value(debug, "cpi_count", -1)) == frame.cpi_count
            and int(_debug_value(debug, "prt_num", -1)) == frame.prt_num
            and math.isclose(prt_seconds, frame.prt_seconds, rel_tol=0.0, abs_tol=1e-15)
            and int(_debug_value(debug, "data_length", -1)) == frame.data_length
        )
        shape_match = frame.iq.shape == mat_iq.shape
        if shape_match:
            absolute_error = float(np.max(np.abs(frame.iq - mat_iq), initial=0.0))
            denominator = max(float(np.max(np.abs(mat_iq), initial=0.0)), 1e-12)
            relative_error = absolute_error / denominator
        else:
            absolute_error = math.nan
            relative_error = math.nan
        numeric_match = shape_match and relative_error <= 1e-7
        row.update(
            {
                "mat_track_point_index": mat_point,
                "mat_track_no_info_head_json": json.dumps(debug_head.tolist()),
                "point_index_match": point_match,
                "target_record_match": target_match,
                "metadata_match": metadata_match,
                "iq_shape_match": shape_match,
                "iq_max_absolute_error": absolute_error,
                "iq_max_relative_error": relative_error,
                "iq_numeric_match": numeric_match,
                "rd_shape": "x".join(str(item) for item in mat_rd.shape),
                "status": "matched" if all(
                    (point_match, target_match, metadata_match, numeric_match)
                ) else "mismatch",
            }
        )
        rows.append(row)
    return rows


def read_v3_points(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("parent_sample_id") != "cq08|track|333":
                continue
            point = int(row["track_point_index"])
            rows[point] = {
                "v3_rd_present": row["rd_present"].strip().lower() == "true",
                "v3_rd_frame_count": int(row["rd_frame_count"]),
                "v3_raw_frame_indices": [
                    int(value) for value in json.loads(row["rd_raw_frame_indices_json"])
                ],
                "v3_rd_tensor_frame_indices": [
                    int(value) for value in json.loads(row["rd_tensor_frame_indices_json"])
                ],
                "v3_rd_coverage": row["rd_point_coverage_status"],
                "v3_label_name": row["label_name"],
                "v3_split_name": row["split_name"],
            }
    if sorted(rows) != list(range(1, TARGET_POINT_COUNT + 1)):
        raise ValueError("V3 Track 333 point coverage is not exactly 1..22")
    return rows


def read_physical_frames(path: Path) -> dict[int, list[int]]:
    by_point: dict[int, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("parent_sample_id") != "cq08|track|333":
                continue
            by_point[int(row["track_point_index"])].append(int(row["raw_frame_index"]))
    return {point: sorted(indices) for point, indices in by_point.items()}


def build_point_coverage(
    frames: list[RawFrame],
    comparisons: list[dict[str, Any]],
    v3_points: dict[int, dict[str, Any]],
    physical_frames: dict[int, list[int]],
) -> list[dict[str, Any]]:
    raw_by_point: dict[int, list[int]] = defaultdict(list)
    mat_by_point: dict[int, list[int]] = defaultdict(list)
    for frame in frames:
        for record in frame.target_records:
            raw_by_point[record.track_point_index_raw].append(frame.frame_sequence - 1)
    for row in comparisons:
        if row.get("mat_track_point_index") is not None:
            mat_by_point[int(row["mat_track_point_index"])].append(
                int(row["frame_sequence"]) - 1
            )

    rows: list[dict[str, Any]] = []
    for point in range(1, TARGET_POINT_COUNT + 1):
        v3 = v3_points[point]
        raw_indices = sorted(raw_by_point.get(point, []))
        mat_indices = sorted(mat_by_point.get(point, []))
        physical_indices = sorted(physical_frames.get(point, []))
        v3_indices = sorted(v3["v3_raw_frame_indices"])
        rows.append(
            {
                "track_point_index": point,
                "raw_frame_count": len(raw_indices),
                "raw_frame_indices_zero_based_json": json.dumps(raw_indices),
                "mat_frame_count": len(mat_indices),
                "mat_frame_indices_zero_based_json": json.dumps(mat_indices),
                "physical_index_frame_count": len(physical_indices),
                "physical_index_frames_zero_based_json": json.dumps(physical_indices),
                "v3_rd_present": v3["v3_rd_present"],
                "v3_rd_frame_count": v3["v3_rd_frame_count"],
                "v3_raw_frame_indices_zero_based_json": json.dumps(v3_indices),
                "raw_matches_mat": raw_indices == mat_indices,
                "raw_matches_physical_index": raw_indices == physical_indices,
                "raw_matches_v3": raw_indices == v3_indices,
            }
        )
    return rows


def run(
    project_root: Path,
    raw_path: Path,
    mat_dir: Path,
    v3_index_path: Path,
    physical_index_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (raw_path, mat_dir, v3_index_path, physical_index_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, raw_metadata = parse_raw_echo(raw_path)
    comparisons = compare_mat_frames(frames, mat_dir)
    v3_points = read_v3_points(v3_index_path)
    physical_frames = read_physical_frames(physical_index_path)
    coverage = build_point_coverage(frames, comparisons, v3_points, physical_frames)

    frame_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_row = asdict(frame)
        frame_row.pop("iq")
        frame_row.pop("target_records")
        frame_row["target_point_indices_json"] = json.dumps(
            [record.track_point_index_raw for record in frame.target_records]
        )
        frame_row["raw_target_identifiers_json"] = json.dumps(
            [record.raw_target_identifier for record in frame.target_records]
        )
        frame_rows.append(frame_row)
        target_rows.extend(asdict(record) for record in frame.target_records)

    raw_point_counts = Counter(
        record.track_point_index_raw
        for frame in frames
        for record in frame.target_records
    )
    points_1_to_15_present = any(point in raw_point_counts for point in range(1, 16))
    parsed_point_values = sorted(raw_point_counts)
    frame_columns = list(frame_rows[0]) if frame_rows else []
    target_columns = list(target_rows[0]) if target_rows else []
    comparison_columns = list(comparisons[0]) if comparisons else []
    coverage_columns = list(coverage[0]) if coverage else []
    frames_csv = output_dir / "raw-frames.csv"
    targets_csv = output_dir / "raw-target-records.csv"
    comparison_csv = output_dir / "raw-mat-comparison.csv"
    coverage_csv = output_dir / "point-coverage.csv"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    _write_csv(frames_csv, frame_rows, frame_columns)
    _write_csv(targets_csv, target_rows, target_columns)
    _write_csv(comparison_csv, comparisons, comparison_columns)
    _write_csv(coverage_csv, coverage, coverage_columns)

    raw_complete = (
        raw_metadata["continuous_from_zero_to_eof"]
        and raw_metadata["all_headers_valid"]
        and raw_metadata["all_footers_valid"]
        and raw_metadata["unparsed_payload_bytes_total"] == 0
    )
    all_mat_match = (
        len(comparisons) == len(frames) == 22
        and all(row["status"] == "matched" for row in comparisons)
    )
    all_mapping_match = all(
        row["raw_matches_mat"]
        and row["raw_matches_physical_index"]
        and row["raw_matches_v3"]
        for row in coverage
    )
    if raw_complete and all_mat_match and all_mapping_match and not points_1_to_15_present:
        conclusion = "source_raw_has_no_frames_for_track_points_1_to_15"
        recoverable = False
        action = (
            "Keep explicit RD-missing masks for Track 333 points 1-15; this .dat "
            "contains no frame record from which those RD observations can be reconstructed."
        )
        status = "passed"
    else:
        conclusion = "inconclusive_or_source_mapping_conflict"
        recoverable = None
        action = (
            "Do not rebuild data. Resolve the reported raw/MAT/index discrepancy and "
            "complete the independent MATLAB audit first."
        )
        status = "needs_review"

    readme_path.write_text(
        "# Track 333 CQ08 original-echo audit\n\n"
        "This is a read-only audit. It does not create training data or modify MAT files.\n\n"
        f"- Parsed raw frames: {len(frames)}\n"
        f"- Parsed raw point indices: {parsed_point_values}\n"
        f"- Raw points 1-15 present: {points_1_to_15_present}\n"
        f"- Raw/MAT I/Q comparisons all matched: {all_mat_match}\n"
        f"- Raw/MAT/physical-index/V3 mapping all matched: {all_mapping_match}\n"
        f"- Conclusion: `{conclusion}`\n\n"
        "The first uint32 in each raw target record is retained as `raw_target_identifier`; "
        "it is not assumed to equal CQ08 parent track ID 333. The independent MATLAB script "
        "is authoritative for the `taylorwin` RD recomputation comparison.\n",
        encoding="utf-8",
    )
    output_files = [frames_csv, targets_csv, comparison_csv, coverage_csv, readme_path]
    summary: dict[str, Any] = {
        "status": status,
        "schema_version": "RADAR-LN3-track-0333-original-echo-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CQ08 Track 333 only; read-only raw/MAT/V3 mapping audit; test excluded",
        "project_root": str(project_root),
        "track_id": TARGET_TRACK_ID,
        "current_label": "BirdTarget",
        "label_policy": (
            "Keep the current label unless multiple independent evidence sources support "
            "a review request; motion strength and model confidence are not label evidence."
        ),
        "inputs": {
            "raw_echo": {
                "path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256(raw_path),
            },
            "mat_directory": str(mat_dir),
            "mat_file_count": len(list(mat_dir.glob("*_333.mat"))),
            "v3_index": {"path": str(v3_index_path), "sha256": _sha256(v3_index_path)},
            "physical_frame_index": {
                "path": str(physical_index_path),
                "sha256": _sha256(physical_index_path),
            },
        },
        "raw_parser": raw_metadata,
        "raw_point_indices": parsed_point_values,
        "raw_frame_counts_by_point": {
            str(point): int(raw_point_counts[point]) for point in parsed_point_values
        },
        "raw_points_1_to_15_present": points_1_to_15_present,
        "raw_points_16_to_22_present": all(point in raw_point_counts for point in range(16, 23)),
        "raw_mat_comparison": {
            "comparison_count": len(comparisons),
            "all_point_indices_match": all(row["point_index_match"] for row in comparisons),
            "all_target_records_match_mat_debug": all(
                row["target_record_match"] for row in comparisons
            ),
            "all_metadata_match": all(row["metadata_match"] for row in comparisons),
            "all_iq_shapes_match": all(row["iq_shape_match"] for row in comparisons),
            "all_iq_numeric_match": all(row["iq_numeric_match"] for row in comparisons),
            "maximum_iq_absolute_error": max(
                float(row["iq_max_absolute_error"]) for row in comparisons
            ),
            "maximum_iq_relative_error": max(
                float(row["iq_max_relative_error"]) for row in comparisons
            ),
            "python_rd_numeric_comparison": (
                "not performed; MATLAB taylorwin recomputation is the authoritative path"
            ),
        },
        "mapping_comparison": {
            "all_points_raw_match_mat": all(row["raw_matches_mat"] for row in coverage),
            "all_points_raw_match_physical_index": all(
                row["raw_matches_physical_index"] for row in coverage
            ),
            "all_points_raw_match_v3": all(row["raw_matches_v3"] for row in coverage),
        },
        "recoverability": {
            "conclusion": conclusion,
            "points_1_to_15_recoverable_from_this_dat": recoverable,
            "action": action,
            "matlab_independent_audit": "未执行",
        },
        "interpretation_limits": [
            "The raw target identifier is not assumed to be CQ08 parent track ID 333.",
            "This audit proves what is present in the specified .dat file; it cannot recover a lost acquisition record.",
            "No label, index, MAT, data contract, config, checkpoint or training artifact was modified.",
            "No training, test evaluation, threshold fitting or deployment switch was executed.",
        ],
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in output_files
        ],
        "test": "未执行",
        "training": "未执行",
        "data_rebuild": "未执行",
        "output_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Python struct audit for CQ08 Track 333 raw echo."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--mat-dir", type=Path, default=DEFAULT_MAT_DIR)
    parser.add_argument("--v3-index", type=Path, default=DEFAULT_V3_INDEX)
    parser.add_argument("--physical-index", type=Path, default=DEFAULT_PHYSICAL_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        _resolve(root, args.raw),
        _resolve(root, args.mat_dir),
        _resolve(root, args.v3_index),
        _resolve(root, args.physical_index),
        _resolve(root, args.output_dir),
    )
    print(json.dumps(_json_number(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
