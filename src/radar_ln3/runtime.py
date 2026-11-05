"""radar_ln3 公共运行时工具。

 产品需要的路径、哈希和原子写入能力统一复用现有 checkpoint/data 工具，
不再在各模块中复制一套实现。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch

from .checkpoint import canonical_hash, sha256_file
from .data.paths import relative_project_path, resolve_project_path


canonical_sha256 = canonical_hash


def sequence_sha256(values: Iterable[Any]) -> str:
    """对有序值计算长度分隔的 SHA-256。"""

    import hashlib

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def json_text(value: Any, *, indent: int = 2) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str) + "\n"


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_payload_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def write_torch_if_absent_or_identical(path: str | Path, value: Any) -> str:
    """只创建或复用内容相同的 torch artifact，拒绝静默覆盖。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        if target.exists():
            existing = torch.load(target, map_location="cpu", weights_only=False)
            candidate = torch.load(temporary, map_location="cpu", weights_only=False)
            if _payload_equal(existing, candidate):
                return "reused_content_identical"
            raise FileExistsError(f"拒绝覆盖不同的 torch artifact：{target}")
        os.replace(temporary, target)
        return "created"
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "atomic_torch_save",
    "atomic_write_text",
    "canonical_sha256",
    "canonical_hash",
    "json_text",
    "relative_project_path",
    "resolve_project_path",
    "sequence_sha256",
    "sha256_file",
    "write_torch_if_absent_or_identical",
]
