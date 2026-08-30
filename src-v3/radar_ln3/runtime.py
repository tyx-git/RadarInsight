"""Shared runtime primitives for the isolated ``src-v3`` implementation.

This module intentionally contains no model, dataset, or experiment policy.
It centralizes the small pieces of infrastructure that used to be copied
through the V21--V26 modules: strict boolean parsing, project-relative path
resolution, content hashing, and bounded atomic artifact writes.

Historical callers may still import compatibility aliases from their original
modules.  New product code should import these functions from here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


TRUE_TEXT = frozenset({"1", "true", "yes", "y", "on"})
FALSE_TEXT = frozenset({"0", "false", "no", "n", "off"})


def parse_bool(value: Any, *, field: str = "value") -> bool:
    """Parse a configuration boolean without Python's ``bool("false")`` trap.

    Unknown textual values are rejected instead of silently becoming false.
    This is important for immutable experiment contracts: a typo must stop
    loading rather than change a training or evaluation mode.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_TEXT:
            return True
        if normalized in FALSE_TEXT:
            return False
    raise ValueError(f"{field} must be a boolean, 0/1, or a recognized boolean string")


def resolve_project_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
    allow_external: bool = False,
) -> Path:
    """Resolve a path and enforce the project-root boundary by default.

    Windows and POSIX separators are both accepted because experiment JSON
    files are exchanged across Windows PowerShell and Python.  An external
    path requires an explicit ``allow_external=True`` at the call site.
    """

    root = Path(project_root).resolve()
    raw = str(value).replace("\\", "/").strip()
    if not raw:
        raise ValueError("project-relative path must not be empty")
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not allow_external:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"path escapes project root: root={root}, path={resolved}"
            ) from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def relative_project_path(project_root: str | Path, path: str | Path) -> str:
    """Return a normalized project-relative path after boundary validation."""

    root = Path(project_root).resolve()
    resolved = resolve_project_path(root, path)
    return resolved.relative_to(root).as_posix()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest for one regular file."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    """Hash an ordered sequence with length-delimited UTF-8 elements."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON data independently of key order and formatting."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def json_text(value: Any, *, indent: int = 2) -> str:
    """Serialize an artifact payload with deterministic Path handling."""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        default=_json_default,
    ) + "\n"


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write a file through a same-directory temporary and ``os.replace``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
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


def write_json_atomic(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Atomically write one JSON artifact."""

    return atomic_write_text(path, json_text(value, indent=indent))


def write_json_if_absent_or_identical(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
) -> str:
    """Create an immutable JSON artifact or reuse an byte-identical copy."""

    target = Path(path)
    text = json_text(value, indent=indent)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different artifact: {target}")
        return "reused_identical"
    atomic_write_text(target, text)
    return "created"


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    """Atomically serialize a PyTorch payload without changing its schema."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def verify_file_hash(path: str | Path, expected_sha256: str, *, field: str = "file") -> str:
    """Recompute and verify one case-insensitive hexadecimal file digest."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{field} expected SHA-256 is malformed")
    actual = sha256_file(path).lower()
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch: expected={expected}, actual={actual}")
    return actual


__all__ = [
    "FALSE_TEXT",
    "TRUE_TEXT",
    "atomic_write_text",
    "atomic_torch_save",
    "canonical_sha256",
    "json_text",
    "parse_bool",
    "relative_project_path",
    "resolve_project_path",
    "sequence_sha256",
    "sha256_file",
    "verify_file_hash",
    "write_json_atomic",
    "write_json_if_absent_or_identical",
]
