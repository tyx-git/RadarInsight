"""X5A 逐点输入适配器与格式诊断。

X5A 字段语义未知，因此只输出 ``uint16`` 原始列、零值掩码和协议候选布局，
不把任何列命名为距离、速度或坐标。r01 默认采用 summary.json 已使用的
``payload_offset=74, point_count=q`` 口径，并把其它口径保留为显式诊断候选。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .protocol import R01_LAYOUT_CANDIDATES, R01_X5A_FIELD_NAMES


MAGIC = b"\xbb\xbb\xbb\xbb"
AA_MARK = b"\xaa\xaa\xaa\xaa"


@dataclass(frozen=True)
class X5ALayout:
    payload_offset: int = 74
    count_mode: str = "q"
    tail_bytes: int = 0

    def expected_record_bytes(self, q: int) -> int:
        count = q if self.count_mode == "q" else q + 1
        return self.payload_offset + 24 * count + self.tail_bytes


@dataclass(frozen=True)
class X5ARecord:
    file_name: str
    record_index: int
    frame_candidate: int
    variant_marker: str
    point_values: np.ndarray
    zero_mask: np.ndarray
    raw_record_bytes: int

    @property
    def point_count(self) -> int:
        return int(self.point_values.shape[0])

    def point_rows(self) -> Iterator[dict[str, object]]:
        for index, values in enumerate(self.point_values):
            yield {
                "record_index": self.record_index,
                "point_index": index,
                "field_names": R01_X5A_FIELD_NAMES,
                "values": values,
                "zero_mask": values == 0,
            }


def _resolve_layout(layout: X5ALayout | None) -> X5ALayout:
    value = layout or X5ALayout()
    if value.payload_offset not in {50, 62, 74}:
        raise ValueError("X5A payload_offset must be one of 50, 62, 74")
    if value.count_mode not in {"q", "q_plus_1"}:
        raise ValueError("X5A count_mode must be q or q_plus_1")
    if value.tail_bytes not in {0, 12}:
        raise ValueError("X5A tail_bytes must be 0 or 12")
    return value


def iter_x5a_records(path: str | Path, *, layout: X5ALayout | None = None) -> Iterator[X5ARecord]:
    selected = _resolve_layout(layout)
    target = Path(path)
    blob = target.read_bytes()
    offset = 0
    record_index = 0
    while offset < len(blob):
        if blob[offset:offset + 4] != MAGIC:
            raise ValueError(f"X5A magic mismatch: {target}:{offset}")
        length_field = int.from_bytes(blob[offset + 4:offset + 6], "little")
        q = int.from_bytes(blob[offset + 22:offset + 26], "little")
        record_bytes = length_field + 18
        if blob[offset + 18:offset + 22] != AA_MARK:
            raise ValueError(f"X5A AA marker mismatch: {target}:{offset}")
        if selected.expected_record_bytes(q) != record_bytes:
            raise ValueError(
                f"X5A layout mismatch: {target}:{offset}; "
                f"expected={selected.expected_record_bytes(q)} actual={record_bytes}"
            )
        end = offset + record_bytes
        if end > len(blob):
            raise ValueError(f"X5A record exceeds file: {target}:{offset}")
        point_count = q if selected.count_mode == "q" else q + 1
        values = np.frombuffer(
            blob,
            dtype="<u2",
            count=point_count * 12,
            offset=offset + selected.payload_offset,
        ).reshape(point_count, 12).copy()
        yield X5ARecord(
            file_name=target.name,
            record_index=record_index,
            frame_candidate=int.from_bytes(blob[offset + 26:offset + 30], "little"),
            variant_marker=blob[offset + 30:offset + 34].hex(),
            point_values=values,
            zero_mask=values == 0,
            raw_record_bytes=record_bytes,
        )
        offset = end
        record_index += 1
    if offset != len(blob):
        raise ValueError(f"X5A parser did not consume file: {target}")


def summarize_x5a_file(path: str | Path, *, layout: X5ALayout | None = None) -> dict[str, object]:
    records = list(iter_x5a_records(path, layout=layout))
    point_count = sum(record.point_count for record in records)
    empty_count = sum(record.point_count == 0 for record in records)
    if point_count:
        values = np.concatenate([record.point_values for record in records if record.point_count], axis=0)
        zero_fraction = np.mean(values == 0, axis=0).astype(float).tolist()
        p50 = np.percentile(values, 50, axis=0).astype(float).tolist()
        p95 = np.percentile(values, 95, axis=0).astype(float).tolist()
    else:
        zero_fraction = [1.0] * 12
        p50 = [0.0] * 12
        p95 = [0.0] * 12
    return {
        "file": str(Path(path)),
        "layout": {
            "payload_offset": (layout or X5ALayout()).payload_offset,
            "count_mode": (layout or X5ALayout()).count_mode,
            "tail_bytes": (layout or X5ALayout()).tail_bytes,
        },
        "record_count": len(records),
        "point_count": point_count,
        "empty_record_count": empty_count,
        "empty_record_fraction": empty_count / max(1, len(records)),
        "point_zero_fraction": dict(zip(R01_X5A_FIELD_NAMES, zero_fraction, strict=True)),
        "point_p50": dict(zip(R01_X5A_FIELD_NAMES, p50, strict=True)),
        "point_p95": dict(zip(R01_X5A_FIELD_NAMES, p95, strict=True)),
    }


def diagnose_layout(path: str | Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for payload_offset, count_mode in R01_LAYOUT_CANDIDATES:
        candidate = X5ALayout(payload_offset=payload_offset, count_mode=count_mode)
        try:
            report = summarize_x5a_file(path, layout=candidate)
            report["status"] = "parseable"
        except (ValueError, OverflowError) as exc:
            report = {
                "layout": {
                    "payload_offset": payload_offset,
                    "count_mode": count_mode,
                    "tail_bytes": 0,
                },
                "status": "rejected",
                "error": str(exc),
            }
        reports.append(report)
    return reports


__all__ = ["AA_MARK", "MAGIC", "X5ALayout", "X5ARecord", "diagnose_layout", "iter_x5a_records", "summarize_x5a_file"]