"""项目路径和 CQ08 数据产物、索引的边界。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .contracts import DataPathError, MissingDataFileError, PathBoundaryError


DATA_ROOT = Path("data/cq08")
TR_ROOT = DATA_ROOT / "tr"
RD_ROOT = DATA_ROOT / "rd"
PROJECT_MANIFEST_PATH = Path("src/config/manifests.json")
TR_INDEX_PATH = TR_ROOT / "cq08-tr-event-index.jsonl"
RD_INDEX_PATH = RD_ROOT / "cq08-rd-frame-index.jsonl"
FORBIDDEN_PROJECT_RELATIVE_ROOTS = (Path("src-v3"), Path("experiments"))


def data_root_name() -> str:
    """返回总 manifest 使用的数据根相对路径。"""

    return DATA_ROOT.as_posix()


def tr_root_name() -> str:
    """返回 TR 数据根相对路径。"""

    return TR_ROOT.as_posix()


def rd_root_name() -> str:
    """返回 RD 数据根相对路径。"""

    return RD_ROOT.as_posix()


def _normalise(value: str | os.PathLike[str]) -> str:
    return str(value).replace("\\", "/").strip()


def ensure_path_inside_root(path: str | Path, root: str | Path) -> Path:
    """确保 path 位于 root 内。"""

    candidate = Path(path).resolve()
    boundary = Path(root).resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise PathBoundaryError(
            f"路径越过边界：path={candidate}，root={boundary}"
        ) from exc
    return candidate


def _reject_forbidden_project_subtree(path: Path, project_root: Path) -> None:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return
    if any(
        relative == forbidden or forbidden in relative.parents
        for forbidden in FORBIDDEN_PROJECT_RELATIVE_ROOTS
    ):
        raise PathBoundaryError(f"数据路径指向禁止的项目目录：{relative}")


def resolve_project_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
    allow_external: bool = False,
) -> Path:
    """解析项目内路径，并拒绝越界路径和实验目录。"""

    root = Path(project_root).resolve()
    raw = _normalise(value)
    if not raw:
        raise DataPathError("数据路径不能为空")
    candidate = Path(raw)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if not allow_external:
        ensure_path_inside_root(resolved, root)
    _reject_forbidden_project_subtree(resolved, root)
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def relative_project_path(project_root: str | Path, path: str | Path) -> str:
    """将项目内路径转换为稳定的 POSIX 相对路径。"""

    root = Path(project_root).resolve()
    resolved = ensure_path_inside_root(Path(path), root)
    _reject_forbidden_project_subtree(resolved, root)
    return resolved.relative_to(root).as_posix()


def data_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    path = (root / DATA_ROOT).resolve()
    ensure_path_inside_root(path, root)
    return path


def tr_root(project_root: str | Path) -> Path:
    path = (data_root(project_root) / "tr").resolve()
    ensure_path_inside_root(path, data_root(project_root))
    return path


def rd_root(project_root: str | Path) -> Path:
    path = (data_root(project_root) / "rd").resolve()
    ensure_path_inside_root(path, data_root(project_root))
    return path


def resolve_data_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """解析 data/cq08 内的产物路径。"""

    root = Path(project_root).resolve()
    raw = _normalise(value)
    if not raw:
        raise DataPathError("数据产物路径不能为空")
    candidate = Path(raw)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    ensure_path_inside_root(resolved, data_root(root))
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def resolve_tr_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """解析 data/cq08/tr 内的路径。"""

    resolved = resolve_data_path(project_root, value, must_exist=False)
    ensure_path_inside_root(resolved, tr_root(project_root))
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def resolve_rd_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """解析 data/cq08/rd 内的路径。"""

    resolved = resolve_data_path(project_root, value, must_exist=False)
    ensure_path_inside_root(resolved, rd_root(project_root))
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def resolve_index_path(
    project_root: str | Path,
    value: str | Path,
    *,
    section: str,
    must_exist: bool = False,
) -> Path:
    """解析 TR 或 RD sidecar 索引路径。"""

    if section not in {"TR", "RD"}:
        raise DataPathError(f"不支持的索引 section：{section!r}")
    resolved = resolve_data_path(project_root, value, must_exist=False)
    boundary = tr_root(project_root) if section == "TR" else rd_root(project_root)
    ensure_path_inside_root(resolved, boundary)
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def resolve_source_path(
    project_root: str | Path,
    value: str | Path,
    *,
    source_roots: Mapping[str, str | Path] | None = None,
    must_exist: bool = False,
    allow_external: bool = False,
) -> Path:
    """解析项目相对路径或显式配置的外部原始数据来源。"""

    raw = _normalise(value)
    roots = source_roots or {}
    if raw.startswith("external/"):
        parts = raw.split("/", 2)
        logical_root = parts[1] if len(parts) > 1 else ""
        if logical_root not in roots:
            raise DataPathError(f"未配置 external source root：{logical_root!r}")
        suffix = parts[2] if len(parts) > 2 else ""
        configured_root = Path(roots[logical_root]).expanduser().resolve()
        resolved = (configured_root / suffix).resolve()
        ensure_path_inside_root(resolved, configured_root)
        _reject_forbidden_project_subtree(resolved, Path(project_root).resolve())
        if not allow_external:
            raise PathBoundaryError("解析 external source 必须显式允许外部路径")
    else:
        resolved = resolve_project_path(
            project_root,
            raw,
            must_exist=False,
            allow_external=allow_external,
        )
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


__all__ = [
    "DATA_ROOT",
    "TR_ROOT",
    "RD_ROOT",
    "PROJECT_MANIFEST_PATH",
    "TR_INDEX_PATH",
    "RD_INDEX_PATH",
    "ensure_path_inside_root",
    "resolve_project_path",
    "relative_project_path",
    "data_root",
    "data_root_name",
    "tr_root",
    "tr_root_name",
    "rd_root",
    "rd_root_name",
    "resolve_data_path",
    "resolve_tr_path",
    "resolve_rd_path",
    "resolve_index_path",
    "resolve_source_path",
]
