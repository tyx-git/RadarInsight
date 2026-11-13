"""原子写：字节与 JSON 的统一实现。

语义（所有调用方一致）：

- 已存在且内容相同 → 直接复用，返回写入字节的 sha256；
- 已存在且内容不同、``overwrite=False`` → ``FileExistsError``（不静默覆盖）；
- ``overwrite=True`` → 写入同目录临时文件后原子替换；
- ``overwrite=False`` 走 ``os.link``，并发写入时复检内容，避免竞态覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
    label: str = "文件",
) -> str:
    """原子写入字节串，返回写入内容的 sha256。"""

    payload_hash = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"目标路径不是文件：{path}")
        if path.read_bytes() == payload:
            return payload_hash
        if not overwrite:
            raise FileExistsError(f"拒绝覆盖内容不同的{label}：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_file() and path.read_bytes() == payload:
                    return payload_hash
                raise FileExistsError(f"拒绝覆盖静态生成的不同{label}：{path}")
    finally:
        temporary.unlink(missing_ok=True)
    return payload_hash


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
    sort_keys: bool = False,
    label: str = "JSON 文件",
) -> str:
    """以统一格式（UTF-8、两空格缩进、末尾换行）原子写入 JSON。"""

    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=sort_keys)
        + "\n"
    ).encode("utf-8")
    return write_bytes(path, encoded, overwrite=overwrite, label=label)


__all__ = ["write_bytes", "write_json"]
