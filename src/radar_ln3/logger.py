"""Radar LN3 统一日志入口。

日志配置集中从 ``src/config/runtime.json`` 读取。产品代码只需要获取
logger 并按标准库的级别记录消息；控制台和文件输出由本模块统一管理。
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from pathlib import Path
import sys
from typing import Any


_LOGGER_NAME = "radar_ln3"
_DEFAULT_CONFIG_PATH = Path("src/config/runtime.json")
_DEFAULT_LOG_PATH = Path("logs/radar_ln3.log")
_DEFAULT_LEVEL = logging.INFO
_CONFIGURED_HANDLER_MARK = "_radar_ln3_configured_handler"


class LoggerError(RuntimeError):
    """日志配置或生命周期错误。"""


class LoggerConfigError(LoggerError):
    """日志配置结构或路径无效。"""


def _project_file(project_root: str | Path, value: str | Path, *, field: str) -> Path:
    """解析项目内文件并拒绝绝对路径和 ``..`` 越界。"""

    root = Path(project_root).resolve()
    try:
        raw = Path(value)
        target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise LoggerConfigError(f"日志配置 {field} 不是有效路径：{value}") from exc
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise LoggerConfigError(f"日志配置 {field} 必须位于项目根目录内：{value}") from exc
    return target


def _read_config(project_root: str | Path, config_path: str | Path) -> Mapping[str, Any]:
    target = _project_file(project_root, config_path, field="config_path")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LoggerConfigError(f"日志配置不存在：{target}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoggerConfigError(f"日志配置无法读取：{target}") from exc
    if not isinstance(payload, Mapping):
        raise LoggerConfigError("runtime 配置根节点必须是对象")
    settings = payload.get("logging", {})
    if settings is None:
        return {}
    if not isinstance(settings, Mapping):
        raise LoggerConfigError("runtime.logging 必须是对象")
    return settings


def _bool_setting(settings: Mapping[str, Any], name: str, default: bool) -> bool:
    value = settings.get(name, default)
    if not isinstance(value, bool):
        raise LoggerConfigError(f"runtime.logging.{name} 必须是布尔值")
    return value


def _level_setting(value: Any, *, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise LoggerConfigError(f"runtime.logging.{field} 不是有效日志级别")
    if isinstance(value, int):
        if value < logging.NOTSET:
            raise LoggerConfigError(f"runtime.logging.{field} 不是有效日志级别")
        return value
    name = str(value).strip().upper()
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        raise LoggerConfigError(f"runtime.logging.{field} 不是有效日志级别：{value}")
    return level


def _remove_owned_handlers(logger: logging.Logger) -> None:
    """只清理本模块创建的 handler，避免破坏调用方的其它配置。"""

    for handler in list(logger.handlers):
        if not getattr(handler, _CONFIGURED_HANDLER_MARK, False):
            continue
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def _mark_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _CONFIGURED_HANDLER_MARK, True)
    return handler


def configure_logging(
    project_root: str | Path,
    *,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    log_path: str | Path | None = None,
    level: str | int | None = None,
    console: bool | None = None,
) -> logging.Logger:
    """配置产品 logger，同时输出到控制台和日志文件。

    ``log_path``、``level`` 和 ``console`` 仅用于覆盖 runtime 配置，覆盖值
    不会写回配置文件。日志文件默认追加写入，重复配置时会先关闭本模块
    创建的旧 handler，避免同一条记录重复输出。
    """

    settings = _read_config(project_root, config_path)
    enabled = _bool_setting(settings, "enabled", True)
    console_enabled = _bool_setting(settings, "console", True)
    file_enabled = _bool_setting(settings, "file", True)
    if console is not None:
        if not isinstance(console, bool):
            raise LoggerConfigError("console 必须是布尔值")
        console_enabled = console

    logger_level = _level_setting(
        level if level is not None else settings.get("level"),
        field="level",
        default=_DEFAULT_LEVEL,
    )
    console_level = _level_setting(
        settings.get("console_level"),
        field="console_level",
        default=logger_level,
    )
    file_level = _level_setting(
        settings.get("file_level"),
        field="file_level",
        default=logger_level,
    )

    logger = logging.getLogger(_LOGGER_NAME)
    logger.propagate = False
    _remove_owned_handlers(logger)

    if not enabled:
        logger.setLevel(logger_level)
        logger.addHandler(_mark_handler(logging.NullHandler()))
        return logger

    # logger 作为全局最低门槛，console/file 再分别进行更严格的筛选。
    # 若文件需要 DEBUG 记录，应在 runtime.logging.level 中配置 DEBUG。
    logger.setLevel(logger_level)

    console_format = str(settings.get("console_format", "%(message)s"))
    file_format = str(
        settings.get(
            "file_format",
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
    )
    date_format = str(settings.get("date_format", "%Y-%m-%d %H:%M:%S"))

    if console_enabled:
        console_handler = _mark_handler(logging.StreamHandler(sys.stdout))
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(console_format, datefmt=date_format))
        logger.addHandler(console_handler)

    try:
        if file_enabled:
            configured_path = log_path
            if configured_path is None:
                configured_path = settings.get("file_path", _DEFAULT_LOG_PATH)
            target = _project_file(project_root, configured_path, field="file_path")
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = _mark_handler(logging.FileHandler(target, mode="a", encoding="utf-8"))
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
            logger.addHandler(file_handler)
    except (OSError, UnicodeError, LoggerConfigError) as exc:
        _remove_owned_handlers(logger)
        if isinstance(exc, LoggerConfigError):
            raise
        raise LoggerError(f"日志文件无法打开：{target}") from exc

    if not logger.handlers:
        logger.addHandler(_mark_handler(logging.NullHandler()))
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取挂在产品根 logger 下的标准库 logger。"""

    value = str(name or "").strip()
    if not value:
        return logging.getLogger(_LOGGER_NAME)
    if value == _LOGGER_NAME or value.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(value)
    return logging.getLogger(f"{_LOGGER_NAME}.{value}")


def log_classification_metrics(
    metrics: Mapping[str, Any],
    *,
    task: str,
    epoch: int,
    phase: str = "validation",
) -> None:
    """按评估结果中的类别顺序打印精度、召回率和 F1，不改写类别名称。"""

    rows = metrics["per_class"]
    lines = [f"[{task.upper()}] {phase} epoch={epoch} 精度/召回/F1"]
    lines.extend(
        f"{row['class']}:{float(row['precision']):.6f}/"
        f"{float(row['recall']):.6f}/{float(row['f1']):.6f}"
        for row in rows
    )
    get_logger("training").info("\n".join(lines))


def close_logging() -> None:
    """关闭本模块创建的 handler，供 CLI、测试和长进程生命周期使用。"""

    _remove_owned_handlers(logging.getLogger(_LOGGER_NAME))


__all__ = ["close_logging", "configure_logging", "get_logger", "log_classification_metrics"]
