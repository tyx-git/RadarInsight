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
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any

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








def _torch_payload_equal(left: Any, right: Any) -> bool:
    """Compare the subset of values used by local Torch artifacts exactly.

    ``torch.save`` archives contain implementation-specific storage identifiers,
    so serializing the same payload twice is not a reliable byte comparison.
    Artifact reuse therefore compares decoded values instead: tensors retain
    exact dtype, shape, and values; containers retain their structure.
    """

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, MappingABC) or isinstance(right, MappingABC):
        if not isinstance(left, MappingABC) or not isinstance(right, MappingABC):
            return False
        if set(left) != set(right):
            return False
        return all(_torch_payload_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or type(left) is not type(right)
            or len(left) != len(right)
        ):
            return False
        return all(_torch_payload_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Path) or isinstance(right, Path):
        return isinstance(left, Path) and isinstance(right, Path) and left == right
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def write_torch_if_absent_or_identical(path: str | Path, value: Any) -> str:
    """Create a Torch artifact or reuse a semantically identical payload.

    Checkpoints and derived score caches are provenance-bearing artifacts.  A
    rerun must not silently replace one with a different result at the same
    path: doing so would leave downstream registries referring to an
    unobservable history.  Serialization is performed in the destination
    directory so the comparison and final replace remain on one filesystem.
    """

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
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"refusing to replace non-file artifact: {target}")
            existing = torch.load(target, map_location="cpu", weights_only=False)
            candidate = torch.load(temporary, map_location="cpu", weights_only=False)
            if _torch_payload_equal(existing, candidate):
                return "reused_content_identical"
            raise FileExistsError(
                f"refusing to overwrite a different torch artifact: {target}"
            )
        os.replace(temporary, target)
        return "created"
    finally:
        if temporary.exists():
            temporary.unlink()






__all__ = ["FALSE_TEXT", "TRUE_TEXT", "atomic_write_text", "canonical_sha256", "json_text", "parse_bool", "relative_project_path", "resolve_project_path", "sha256_file", "write_torch_if_absent_or_identical"]