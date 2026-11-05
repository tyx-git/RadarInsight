"""供数据适配器使用的有大小边界、按格式读取的底层接口。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import (
    DataFormatError,
    DataIntegrityError,
    DataReadError,
    MissingDataFileError,
)


DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_NPZ_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 128 * 1024 * 1024
MAX_NPZ_DECLARED_BYTES = 512 * 1024 * 1024
MAX_NPZ_MEMBERS = 32


def require_file(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Path:
    target = Path(path)
    if not target.is_file():
        raise MissingDataFileError(str(target))
    size = int(target.stat().st_size)
    if size > int(max_bytes):
        raise DataReadError(
            f"文件超过大小限制：{target}（{size} > {max_bytes}）"
        )
    return target


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    target = require_file(path, max_bytes=2**63 - 1)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bytes(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> bytes:
    target = require_file(path, max_bytes=max_bytes)
    try:
        return target.read_bytes()
    except OSError as exc:
        raise DataReadError(f"无法读取文件：{target}") from exc


def read_text(
    path: str | Path,
    *,
    encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk"),
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> str:
    payload = read_bytes(path, max_bytes=max_bytes)
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataFormatError(f"不支持的文本编码：{path}") from last_error


def read_json(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Any:
    try:
        return json.loads(read_text(path, max_bytes=max_bytes))
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"JSON 格式无效：{path}") from exc


def read_delimited(
    path: str | Path,
    *,
    delimiter: str = ",",
    encodings: Iterable[str] = ("gbk", "utf-8-sig", "utf-8"),
) -> list[dict[str, str]]:
    text = read_text(path, encodings=encodings)
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataFormatError(f"分隔文件没有表头：{path}")
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise DataFormatError(f"分隔文件格式无效：{path}") from exc
    if any(None in row for row in rows):
        raise DataFormatError(f"分隔文件包含格式错误的行：{path}")
    return rows


def _preflight_npz(path: Path) -> None:
    require_file(path, max_bytes=MAX_NPZ_ARCHIVE_BYTES)
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if len(members) > MAX_NPZ_MEMBERS:
                raise DataFormatError(f"NPZ 包含过多成员：{path}")
            declared = 0
            for member in members:
                if member.is_dir() or member.filename.replace("\\", "/").endswith("/"):
                    raise DataFormatError(f"NPZ 包含目录成员：{path}")
                if int(member.file_size) > MAX_NPZ_MEMBER_BYTES:
                    raise DataFormatError(f"NPZ 成员超过大小限制：{path}")
                declared += int(member.file_size)
            if declared > MAX_NPZ_DECLARED_BYTES:
                raise DataFormatError(f"NPZ 声明大小超过限制：{path}")
    except zipfile.BadZipFile as exc:
        raise DataFormatError(f"NPZ 压缩包格式无效：{path}") from exc


def read_npz(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    allow_pickle: bool = False,
) -> dict[str, np.ndarray]:
    if allow_pickle:
        raise DataFormatError("NPZ 读取禁止启用 allow_pickle")
    target = Path(path)
    _preflight_npz(target)
    try:
        with np.load(target, allow_pickle=False) as archive:
            missing = sorted(set(required_keys) - set(archive.files))
            if missing:
                raise DataFormatError(f"NPZ 缺少必需字段：{target}: {missing}")
            return {key: np.asarray(archive[key]) for key in archive.files}
    except DataFormatError:
        raise
    except (OSError, ValueError) as exc:
        raise DataReadError(f"无法读取 NPZ：{target}") from exc


def read_mat(
    path: str | Path,
    *,
    variable_names: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    target = require_file(path, max_bytes=max_bytes)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise DataReadError("读取 MAT 文件需要 scipy") from exc
    try:
        payload = loadmat(
            target,
            variable_names=None if variable_names is None else list(variable_names),
            squeeze_me=True,
            struct_as_record=False,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise DataReadError(f"无法读取 MAT 文件：{target}") from exc
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("__")
    }


def finite_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise DataIntegrityError(f"{name} 包含非有限值")
    return array


def validate_shape(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != int(ndim):
        raise DataFormatError(f"{name} 的 ndim={array.ndim}，期望为 {ndim}")
    return array


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "require_file",
    "sha256_file",
    "read_bytes",
    "read_text",
    "read_json",
    "read_delimited",
    "read_npz",
    "read_mat",
    "finite_array",
    "validate_shape",
]
