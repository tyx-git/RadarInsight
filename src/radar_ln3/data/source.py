"""注册数据源配置和逻辑 source path 解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import DataFormatError, DataPathError, PathBoundaryError
from .paths import resolve_project_path, resolve_source_path


SOURCE_CONFIG_SCHEMA = "RADAR-LN3-CQ08-SOURCE-CONFIG-V1"


def load_source_config(
    project_root: str | Path,
    path: str | Path = "src/config/source.json",
) -> dict[str, Any]:
    """加载并校验正式 source 配置。"""

    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"无法读取 source 配置：{target}") from exc
    if not isinstance(payload, Mapping):
        raise DataFormatError("source 配置根节点必须是对象")
    value = dict(payload)
    if value.get("schema_version") != SOURCE_CONFIG_SCHEMA:
        raise DataFormatError("source 配置 schema_version 不匹配")
    if str(value.get("dataset", "")) != "CQ08":
        raise DataFormatError("source 配置 dataset 必须是 CQ08")
    roots = value.get("source_roots")
    if not isinstance(roots, Mapping) or not isinstance(roots.get("cq08"), Mapping):
        raise DataFormatError("source 配置缺少 cq08 source root")
    cq08 = dict(roots["cq08"])
    if not str(cq08.get("path", "")).strip():
        raise DataPathError("CQ08 source root 不能为空")
    prefixes = cq08.get("logical_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise DataFormatError("CQ08 logical_prefixes 必须是非空列表")
    if not isinstance(cq08.get("allow_external", False), bool):
        raise DataFormatError("CQ08 source_roots.cq08.allow_external 必须是布尔值")
    value["source_roots"] = {"cq08": cq08}
    return value


def resolve_registered_source_path(
    project_root: str | Path,
    value: str | Path,
    config: Mapping[str, Any],
    *,
    must_exist: bool = False,
) -> Path:
    """按 source.json 将 manifest 中的 logical source path 解析为文件。"""

    root = Path(project_root).resolve()
    raw = str(value).replace("\\", "/").strip()
    source_roots = config.get("source_roots")
    cq08 = source_roots.get("cq08") if isinstance(source_roots, Mapping) else None
    if not isinstance(cq08, Mapping):
        raise DataFormatError("source 配置缺少 cq08 source root")
    configured_root = str(cq08["path"])
    prefixes = [str(item).replace("\\", "/").strip("/") for item in cq08["logical_prefixes"]]
    for prefix in prefixes:
        if raw == prefix or raw.startswith(prefix + "/"):
            if not bool(cq08.get("allow_external", False)):
                raise PathBoundaryError(
                    "source.json 未允许解析 CQ08 external source path"
                )
            suffix = raw[len(prefix) :].lstrip("/")
            return resolve_source_path(
                root,
                f"external/cq08/{suffix}",
                source_roots={"cq08": (root / configured_root).resolve()
                if not Path(configured_root).is_absolute()
                else configured_root},
                must_exist=must_exist,
                allow_external=True,
            )
    return resolve_project_path(root, raw, must_exist=must_exist, allow_external=False)


__all__ = [
    "SOURCE_CONFIG_SCHEMA",
    "load_source_config",
    "resolve_registered_source_path",
]
