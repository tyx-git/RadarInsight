"""只读校验 X5A 点级标签 sidecar 的覆盖率和质量。

sidecar 采用 JSONL，一行对应一个 X5A 点，字段至少包括：
file_id、record_id、point_id、label。该脚本不会修改原始 .dat 或 sidecar。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}
REQUIRED_FIELDS = {"file_id", "record_id", "point_id", "label"}


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"sidecar JSON 无效: line={line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"sidecar 行必须是 object: line={line_number}")
            value["_line_number"] = line_number
            rows.append(value)
    return rows


def _load_x5a_index(path: Path) -> dict[str, dict[str, Any]]:
    """从 per-record index 读取精确 record/point 边界。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    raw_files = payload.get("files", {})
    if isinstance(raw_files, list):
        for row in raw_files:
            result[str(row["file_name"])] = {
                "point_count": int(row["point_count"]),
                "record_count": int(row["record_count"]),
                "records": {},
            }
        return result
    for file_id, meta in raw_files.items():
        records = {
            int(row["record_id"]): int(row["point_count"])
            for row in meta.get("records", [])
        }
        result[str(file_id)] = {
            "point_count": int(meta["point_count"]),
            "record_count": int(meta["record_count"]),
            "records": records,
        }
    return result


def validate_sidecar(
    sidecar_path: str | Path,
    *,
    exploration_path: str | Path,
    record_index_path: str | Path | None = None,
    minimum_coverage: float = 0.995,
) -> dict[str, Any]:
    sidecar = Path(sidecar_path).resolve()
    exploration = Path(exploration_path).resolve()
    rows = _load_records(sidecar)
    index_path = Path(record_index_path).resolve() if record_index_path is not None else exploration
    expected_by_file = _load_x5a_index(index_path)
    precise_record_index = record_index_path is not None
    seen: set[tuple[str, int, int]] = set()
    errors: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    records_by_file: dict[str, set[int]] = defaultdict(set)
    points_by_file: dict[str, set[tuple[int, int]]] = defaultdict(set)
    confidence_values: list[float] = []
    for row in rows:
        line = int(row["_line_number"])
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append({"line": line, "type": "missing_fields", "fields": missing})
            continue
        file_id = str(row["file_id"])
        try:
            record_id = int(row["record_id"])
            point_id = int(row["point_id"])
        except (TypeError, ValueError):
            errors.append({"line": line, "type": "non_integer_key"})
            continue
        label = str(row["label"])
        key = (file_id, record_id, point_id)
        file_meta = expected_by_file.get(file_id)
        if file_meta is None:
            errors.append({"line": line, "type": "unknown_file_id", "file_id": file_id})
        if record_id < 0 or point_id < 0:
            errors.append({"line": line, "type": "negative_key", "key": key})
        elif file_meta is not None:
            if record_id >= int(file_meta["record_count"]):
                errors.append({"line": line, "type": "record_id_out_of_range", "record_id": record_id, "record_count": file_meta["record_count"]})
            if precise_record_index:
                point_count_for_record = file_meta["records"].get(record_id)
                if point_count_for_record is None:
                    errors.append({"line": line, "type": "record_id_missing_from_index", "record_id": record_id})
                elif point_id >= point_count_for_record:
                    errors.append({"line": line, "type": "point_id_out_of_range_record", "point_id": point_id, "record_id": record_id, "record_point_count": point_count_for_record})
            elif point_id >= int(file_meta["point_count"]):
                errors.append({"line": line, "type": "point_id_out_of_range_global", "point_id": point_id, "point_count": file_meta["point_count"]})
        if label not in ALLOWED_LABELS:
            errors.append({"line": line, "type": "invalid_label", "label": label})
        if key in seen:
            errors.append({"line": line, "type": "duplicate_key", "key": key})
        seen.add(key)
        label_counts[label] += 1
        file_counts[file_id] += 1
        records_by_file[file_id].add(record_id)
        points_by_file[file_id].add((record_id, point_id))
        if "label_confidence" in row:
            try:
                confidence = float(row["label_confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError
                confidence_values.append(confidence)
            except (TypeError, ValueError):
                errors.append({"line": line, "type": "invalid_label_confidence"})
    expected_points = sum(int(meta["point_count"]) for meta in expected_by_file.values())
    observed_points = len(seen)
    coverage = observed_points / max(1, expected_points)
    per_file = {
        file_id: {
            "expected_points": int(meta["point_count"]),
            "expected_records": int(meta["record_count"]),
            "sidecar_points": int(file_counts.get(file_id, 0)),
            "coverage": float(file_counts.get(file_id, 0) / max(1, int(meta["point_count"]))),
            "record_count": len(records_by_file.get(file_id, set())),
        }
        for file_id, meta in sorted(expected_by_file.items())
    }
    coverage_failures = [
        file_id
        for file_id, value in per_file.items()
        if value["coverage"] < float(minimum_coverage)
    ]
    result = {
        "schema_version": "RADAR-LN3-X5A-label-sidecar-validation-v2",
        "sidecar_path": sidecar.as_posix(),
        "exploration_path": exploration.as_posix(),
        "record_index_path": index_path.as_posix(),
        "precise_record_index": precise_record_index,
        "minimum_coverage": float(minimum_coverage),
        "status": "passed" if not errors and coverage >= float(minimum_coverage) and not coverage_failures else "blocked",
        "blocking_reasons": [
            *(["sidecar_parse_or_schema_errors"] if errors else []),
            *(["global_coverage_below_minimum"] if coverage < float(minimum_coverage) else []),
            *(["per_file_coverage_below_minimum"] if coverage_failures else []),
        ],
        "expected_file_count": len(expected_by_file),
        "expected_point_count": expected_points,
        "sidecar_point_count": observed_points,
        "global_coverage": float(coverage),
        "label_counts": dict(label_counts),
        "label_confidence": {
            "count": len(confidence_values),
            "mean": sum(confidence_values) / max(1, len(confidence_values)),
            "minimum": min(confidence_values) if confidence_values else None,
        },
        "per_file": per_file,
        "coverage_failures": coverage_failures,
        "error_count": len(errors),
        "errors": errors[:200],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 X5A 标签 sidecar")
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--exploration", type=Path, default=Path("tmp/x5a_exploration.json"))
    parser.add_argument("--record-index", type=Path, default=Path("src-v3/v27/V27d/r20/manifests/V27d-r20-x5a-record-index.json"))
    parser.add_argument("--minimum-coverage", type=float, default=0.995)
    parser.add_argument("--output", type=Path, default=Path("tmp/x5a_label_sidecar_validation.json"))
    args = parser.parse_args()
    result = validate_sidecar(
        args.sidecar,
        exploration_path=args.exploration,
        record_index_path=args.record_index,
        minimum_coverage=args.minimum_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "blocking_reasons", "global_coverage", "error_count", "label_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
