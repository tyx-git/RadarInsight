"""生成 X5A 变长记录索引。

索引只记录结构信息，不复制点载荷、不修改原始 .dat。每个 record_id
从 0 开始，point_count 用于精确校验标签 sidecar 的 point_id 边界。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAGIC = bytes.fromhex("bbbbbbbb")
AA_MARK = bytes.fromhex("aaaaaaaa")
POINT_PAYLOAD_OFFSET = 50
POINT_WIDTH = 12
POINT_BYTES = POINT_WIDTH * 2


def build_index(root: str | Path) -> dict[str, Any]:
    project_root = Path(root).resolve()
    data_root = project_root / "OriData" / "梁子湖X5A"
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(data_root.glob("*.dat")):
        payload = path.read_bytes()
        offset = 0
        records: list[dict[str, Any]] = []
        while offset < len(payload):
            if payload[offset : offset + 4] != MAGIC:
                raise ValueError(f"record magic mismatch: {path}:{offset}")
            length = int.from_bytes(payload[offset + 4 : offset + 6], "little") + 18
            end = offset + length
            if end > len(payload):
                raise ValueError(f"record exceeds file: {path}:{offset}")
            record = payload[offset:end]
            if record[18:22] != AA_MARK:
                raise ValueError(f"secondary magic mismatch: {path}:{offset}")
            point_count = int.from_bytes(record[22:26], "little") + 1
            expected = POINT_PAYLOAD_OFFSET + POINT_BYTES * point_count
            if expected != length:
                raise ValueError(f"length/point mismatch: {path}:{offset}")
            records.append(
                {
                    "record_id": len(records),
                    "byte_offset": offset,
                    "byte_length": length,
                    "point_count": point_count,
                    "payload_offset": offset + POINT_PAYLOAD_OFFSET,
                    "marker_hex": record[30:34].hex(),
                    "tick_candidate": int.from_bytes(record[26:30], "little"),
                }
            )
            offset = end
        files[path.name] = {
            "file_size_bytes": len(payload),
            "record_count": len(records),
            "point_count": int(sum(item["point_count"] for item in records)),
            "records": records,
        }
    if not files:
        raise FileNotFoundError(data_root)
    return {
        "schema_version": "RADAR-LN3-X5A-record-index-v1",
        "raw_root": data_root.as_posix(),
        "file_count": len(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 X5A 变长记录索引")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("tmp/x5a_record_index.json"))
    args = parser.parse_args()
    result = build_index(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": result["file_count"], "files": {k: {"record_count": v["record_count"], "point_count": v["point_count"]} for k, v in result["files"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
