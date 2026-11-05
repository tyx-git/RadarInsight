"""统一日志模块的最小契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar_ln3.logger import (
    LoggerConfigError,
    close_logging,
    configure_logging,
    get_logger,
)


def _write_runtime(path: Path, *, file_path: str = "logs/test.log") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "RADAR-LN3-RUNTIME-CONFIG-V1",
                "logging": {
                    "enabled": True,
                    "console": True,
                    "file": True,
                    "level": "DEBUG",
                    "console_level": "INFO",
                    "file_level": "DEBUG",
                    "file_path": file_path,
                    "console_format": "%(message)s",
                    "file_format": "%(levelname)s|%(name)s|%(message)s",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_logging_mirrors_console_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_runtime(tmp_path / "runtime.json")
    try:
        configure_logging(tmp_path, config_path="runtime.json")
        logger = get_logger("test")
        logger.info("控制台和文件")
        logger.debug("仅文件")
        captured = capsys.readouterr()
        content = (tmp_path / "logs" / "test.log").read_text(encoding="utf-8")
        assert "控制台和文件" in captured.out
        assert "仅文件" not in captured.out
        assert "控制台和文件" in content
        assert "仅文件" in content
    finally:
        close_logging()


def test_reconfigure_does_not_duplicate_records(tmp_path: Path) -> None:
    _write_runtime(tmp_path / "runtime.json")
    try:
        configure_logging(tmp_path, config_path="runtime.json")
        configure_logging(tmp_path, config_path="runtime.json")
        get_logger("test").info("只写一次")
        content = (tmp_path / "logs" / "test.log").read_text(encoding="utf-8")
        assert content.count("只写一次") == 1
    finally:
        close_logging()


def test_log_path_must_stay_inside_project_root(tmp_path: Path) -> None:
    _write_runtime(tmp_path / "runtime.json", file_path="../outside.log")
    with pytest.raises(LoggerConfigError, match="项目根目录"):
        configure_logging(tmp_path, config_path="runtime.json")
    close_logging()
