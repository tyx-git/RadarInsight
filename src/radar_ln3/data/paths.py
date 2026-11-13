"""项目路径和 CQ08 数据产物、索引的边界。"""

from __future__ import annotations

import os
import stat as stat_module
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Mapping

from .contracts import DataPathError, MissingDataFileError, PathBoundaryError


DATA_ROOT = Path("data/cq08")
TR_ROOT = DATA_ROOT / "tr"
RD_ROOT = DATA_ROOT / "rd"
PROJECT_MANIFEST_PATH = Path("src/config/manifests.json")
TR_INDEX_PATH = TR_ROOT / "cq08-tr-event-index.jsonl"
RD_INDEX_PATH = RD_ROOT / "cq08-rd-frame-index.jsonl"
FORBIDDEN_PROJECT_RELATIVE_ROOTS = (Path("src-v3"), Path("experiments"))
_FORBIDDEN_PROJECT_ROOT_NAMES = frozenset(
    item.as_posix() for item in FORBIDDEN_PROJECT_RELATIVE_ROOTS
)

_RESOLUTION_CACHE: ContextVar[dict[Path, Path] | None] = ContextVar(
    "radar_ln3_path_resolution_cache",
    default=None,
)


@contextmanager
def path_resolution_cache() -> Iterator[None]:
    """在一次只读批量校验内复用 ``Path.resolve()`` 结果。

    作用域内不得创建或删除文件；退出后缓存立即释放，作用域外行为不变。
    仅用于 manifest 等纯读取校验，避免同一路径被反复解析。
    """

    token = _RESOLUTION_CACHE.set({})
    try:
        yield
    finally:
        _RESOLUTION_CACHE.reset(token)


def resolve_path(path: str | Path) -> Path:
    """解析路径；``path_resolution_cache`` 生效时同一路径只解析一次。

    缓存生效时先解析父目录再拼接叶节点，避免同目录下大量文件重复执行完整
    realpath；叶节点是符号链接或 junction 时仍回退到完整解析，越界校验
    语义不变。
    """

    candidate = path if isinstance(path, Path) else Path(path)
    cache = _RESOLUTION_CACHE.get()
    if cache is None:
        return candidate.resolve()
    resolved = cache.get(candidate)
    if resolved is not None:
        return resolved
    if candidate.name in {"", ".", ".."} or candidate.parent == candidate:
        resolved = candidate.resolve()
    else:
        parent = resolve_path(candidate.parent)
        resolved = parent / candidate.name
        try:
            info = os.lstat(resolved)
        except OSError:
            needs_full_resolve = True
        else:
            # 符号链接或 Windows junction 的叶节点必须完整解析，防止逃逸。
            needs_full_resolve = stat_module.S_ISLNK(info.st_mode) or (
                getattr(info, "st_reparse_tag", 0) == 0xA0000003
            )
        if needs_full_resolve:
            resolved = candidate.resolve()
    cache[candidate] = resolved
    # 已解析路径常被再次传入（如 ensure_path_inside_root），直接复用结果。
    cache[resolved] = resolved
    return resolved


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


def _normalised_text(path: Path) -> str:
    """返回用于边界比较的规范化路径文本（Windows 大小写不敏感）。"""

    return os.path.normcase(os.fspath(path))


def _is_within_normalised(candidate: str, boundary: str) -> bool:
    if candidate == boundary:
        return True
    prefix = boundary if boundary.endswith(os.sep) else boundary + os.sep
    return candidate.startswith(prefix)


def _is_within(path: Path, root: Path) -> bool:
    return _is_within_normalised(_normalised_text(path), _normalised_text(root))


def ensure_path_inside_root(path: str | Path, root: str | Path) -> Path:
    """确保 path 位于 root 内。"""

    candidate = resolve_path(path)
    boundary = resolve_path(root)
    if not _is_within(candidate, boundary):
        raise PathBoundaryError(
            f"路径越过边界：path={candidate}，root={boundary}"
        )
    return candidate


def _reject_forbidden_project_subtree(path: Path, project_root: Path) -> None:
    candidate = _normalised_text(path)
    boundary = _normalised_text(project_root)
    if not _is_within_normalised(candidate, boundary) or candidate == boundary:
        return
    prefix = boundary if boundary.endswith(os.sep) else boundary + os.sep
    relative = candidate[len(prefix):]
    first = relative.split(os.sep, 1)[0]
    if first in _FORBIDDEN_PROJECT_ROOT_NAMES:
        raise PathBoundaryError(
            f"数据路径指向禁止的项目目录：{relative.replace(os.sep, '/')}"
        )


def resolve_project_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
    allow_external: bool = False,
) -> Path:
    """解析项目内路径，并拒绝越界路径和实验目录。"""

    root = resolve_path(project_root)
    raw = _normalise(value)
    if not raw:
        raise DataPathError("数据路径不能为空")
    candidate = Path(raw)
    resolved = (
        resolve_path(candidate)
        if candidate.is_absolute()
        else resolve_path(root / candidate)
    )
    if not allow_external:
        ensure_path_inside_root(resolved, root)
    _reject_forbidden_project_subtree(resolved, root)
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def resolve_project_artifact_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """解析项目内模型或实验产物路径，允许 ``experiments`` 子目录。"""

    root = resolve_path(project_root)
    raw = _normalise(value)
    if not raw:
        raise DataPathError("项目产物路径不能为空")
    candidate = Path(raw)
    resolved = (
        resolve_path(candidate)
        if candidate.is_absolute()
        else resolve_path(root / candidate)
    )
    ensure_path_inside_root(resolved, root)
    relative = resolved.relative_to(root)
    archived_source = Path("src-v3")
    if relative == archived_source or archived_source in relative.parents:
        raise PathBoundaryError(f"项目产物路径指向归档源码目录：{relative}")
    if must_exist and not resolved.is_file():
        raise MissingDataFileError(str(resolved))
    return resolved


def relative_project_path(project_root: str | Path, path: str | Path) -> str:
    """将项目内路径转换为稳定的 POSIX 相对路径。"""

    root = resolve_path(project_root)
    resolved = ensure_path_inside_root(Path(path), root)
    _reject_forbidden_project_subtree(resolved, root)
    return resolved.relative_to(root).as_posix()


def data_root(project_root: str | Path) -> Path:
    root = resolve_path(project_root)
    path = resolve_path(root / DATA_ROOT)
    ensure_path_inside_root(path, root)
    return path


def tr_root(project_root: str | Path) -> Path:
    data = data_root(project_root)
    path = resolve_path(data / "tr")
    ensure_path_inside_root(path, data)
    return path


def rd_root(project_root: str | Path) -> Path:
    data = data_root(project_root)
    path = resolve_path(data / "rd")
    ensure_path_inside_root(path, data)
    return path


def resolve_data_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """解析 data/cq08 内的产物路径。"""

    root = resolve_path(project_root)
    raw = _normalise(value)
    if not raw:
        raise DataPathError("数据产物路径不能为空")
    candidate = Path(raw)
    resolved = (
        resolve_path(candidate)
        if candidate.is_absolute()
        else resolve_path(root / candidate)
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
        configured_root = resolve_path(Path(roots[logical_root]).expanduser())
        resolved = resolve_path(configured_root / suffix)
        ensure_path_inside_root(resolved, configured_root)
        _reject_forbidden_project_subtree(resolved, resolve_path(project_root))
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
    "path_resolution_cache",
    "resolve_path",
    "resolve_project_path",
    "resolve_project_artifact_path",
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
