"""校验 X5A 原始 .dat 与机器可读数据契约的一致性。

该脚本只读原始文件，不修改数据。它将“结构上可验证的事实”和“仍需
设备协议/标签 sidecar 确认的事项”分开输出，避免把候选字段误当物理字段。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = bytes.fromhex("bbbbbbbb")
AA_MARK = bytes.fromhex("aaaaaaaa")
POINT_WIDTH = 12
POINT_BYTES = POINT_WIDTH * 2
POINT_PAYLOAD_OFFSET = 50


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("X5A contract must be a JSON object")
    return value


def _check_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    offset = 0
    record_count = 0
    point_count = 0
    lengths: list[int] = []
    points_per_record: list[int] = []
    marker_counts: Counter[str] = Counter()
    ticks: list[int] = []
    field_min = np.full(POINT_WIDTH, np.inf, dtype=np.float64)
    field_max = np.full(POINT_WIDTH, -np.inf, dtype=np.float64)
    field_sum = np.zeros(POINT_WIDTH, dtype=np.float64)
    field_sum_sq = np.zeros(POINT_WIDTH, dtype=np.float64)
    field_zero = np.zeros(POINT_WIDTH, dtype=np.int64)
    field_saturated = np.zeros(POINT_WIDTH, dtype=np.int64)
    errors: list[str] = []
    tick_resets = 0
    while offset < len(payload):
        if payload[offset : offset + 4] != MAGIC:
            errors.append(f"record_magic_mismatch@{offset}")
            break
        if offset + 6 > len(payload):
            errors.append(f"record_header_truncated@{offset}")
            break
        record_bytes = int.from_bytes(payload[offset + 4 : offset + 6], "little") + 18
        end = offset + record_bytes
        if end > len(payload):
            errors.append(f"record_exceeds_file@{offset}")
            break
        record = payload[offset:end]
        if record[18:22] != AA_MARK:
            errors.append(f"secondary_magic_mismatch@{offset}")
            break
        points = int.from_bytes(record[22:26], "little") + 1
        expected = POINT_PAYLOAD_OFFSET + POINT_BYTES * points
        if expected != record_bytes:
            errors.append(f"length_point_count_mismatch@{offset}")
            break
        marker_counts[record[30:34].hex()] += 1
        tick = int.from_bytes(record[26:30], "little")
        if ticks and tick < ticks[-1]:
            tick_resets += 1
        ticks.append(tick)
        values = np.frombuffer(
            record,
            dtype="<u2",
            count=points * POINT_WIDTH,
            offset=POINT_PAYLOAD_OFFSET,
        ).reshape(points, POINT_WIDTH)
        as_float = values.astype(np.float64, copy=False)
        field_min = np.minimum(field_min, as_float.min(axis=0))
        field_max = np.maximum(field_max, as_float.max(axis=0))
        field_sum += as_float.sum(axis=0)
        field_sum_sq += np.square(as_float).sum(axis=0)
        field_zero += np.sum(values == 0, axis=0)
        field_saturated += np.sum(values >= 65535, axis=0)
        record_count += 1
        point_count += points
        lengths.append(record_bytes)
        points_per_record.append(points)
        offset = end
    if not record_count:
        raise ValueError(f"X5A file has no valid records: {path}")
    count = float(point_count)
    means = field_sum / count
    variances = np.maximum(0.0, field_sum_sq / count - np.square(means))
    return {
        "file_name": path.name,
        "file_size_bytes": len(payload),
        "parsed_bytes": offset,
        "remaining_bytes": len(payload) - offset,
        "record_count": record_count,
        "point_count": point_count,
        "record_count_matches_contract": not errors and offset == len(payload),
        "point_count_matches_record_lengths": not errors,
        "points_per_record": {
            "min": int(min(points_per_record)),
            "mean": float(np.mean(points_per_record)),
            "p95": float(np.percentile(points_per_record, 95)),
            "max": int(max(points_per_record)),
        },
        "record_bytes": {
            "min": int(min(lengths)),
            "mean": float(np.mean(lengths)),
            "max": int(max(lengths)),
        },
        "marker_counts": dict(marker_counts),
        "tick_candidate": {
            "min": int(min(ticks)),
            "max": int(max(ticks)),
            "reset_or_wrap_count": tick_resets,
            "monotonic_non_decreasing": tick_resets == 0,
        },
        "fields": {
            f"u16_{index:02d}": {
                "dtype_observed": "uint16",
                "min": float(field_min[index]),
                "max": float(field_max[index]),
                "mean": float(means[index]),
                "variance_population": float(variances[index]),
                "zero_fraction": float(field_zero[index] / count),
                "saturated_65535_fraction": float(field_saturated[index] / count),
            }
            for index in range(POINT_WIDTH)
        },
        "errors": errors,
    }


def validate(project_root: str | Path, contract_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = root / contract_file
    contract = _load_contract(contract_file)
    raw_root = root / str(contract["dataset"]["root"])
    files = sorted(raw_root.glob(str(contract["dataset"]["file_glob"])))
    if not files:
        raise FileNotFoundError(raw_root)
    reports = [_check_file(path) for path in files]
    checks = {
        "file_count_matches_expected": len(files) == int(contract["dataset"]["expected_file_count"]),
        "all_files_fully_parsed": all(row["record_count_matches_contract"] for row in reports),
        "all_records_length_consistent": all(row["point_count_matches_record_lengths"] for row in reports),
        "all_observed_point_dtypes_uint16": all(
            field["dtype_observed"] == "uint16"
            for row in reports
            for field in row["fields"].values()
        ),
        "label_sidecar_present": bool(contract["dataset"]["label_sidecar"].get("path")),
        "vendor_schema_present": not any(
            item["status"] == "pending_vendor_schema" for item in contract.get("fields", [])
        ),
        "tick_semantics_confirmed": contract["record_layout"]["frame_or_tick_candidate"]["status"] == "verified",
    }
    blocking = [
        "label_sidecar_present",
        "vendor_schema_present",
        "tick_semantics_confirmed",
    ]
    result = {
        "schema_version": "RADAR-LN3-X5A-contract-validation-v1",
        "contract_path": contract_file.as_posix(),
        "contract_status": contract.get("contract_status"),
        "checks": checks,
        "verified_pass_count": int(sum(checks.values())),
        "blocking_unresolved_items": [name for name in blocking if not checks[name]],
        "files": reports,
        "aggregate": {
            "file_count": len(files),
            "record_count": int(sum(row["record_count"] for row in reports)),
            "point_count": int(sum(row["point_count"] for row in reports)),
            "remaining_bytes": int(sum(row["remaining_bytes"] for row in reports)),
            "tick_reset_or_wrap_count": int(sum(row["tick_candidate"]["reset_or_wrap_count"] for row in reports)),
            "marker_counts": dict(
                Counter(
                    marker
                    for row in reports
                    for marker, count in row["marker_counts"].items()
                    for _ in range(count)
                )
            ),
        },
        "next_actions": [
            "获取设备协议和固件版本，关闭 u16 字段语义、单位、缩放及无效码阻断项",
            "提供 file_id/record_id/point_id 标签 sidecar，并验证 join coverage >= 99.5%",
            "确认 tick 回绕/重置和 scan window 后，再启用时间连续性特征",
            "在契约关闭后，以文件级 source-disjoint 方式校准 Clutter 与 Drone/Bird 保留率",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="校验梁子湖 X5A 数据契约")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path("src-v3/v27/V27d/r20/x5a_contract_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("tmp/x5a_contract_validation.json"))
    args = parser.parse_args()
    result = validate(args.project_root, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("checks", "verified_pass_count", "blocking_unresolved_items", "aggregate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
