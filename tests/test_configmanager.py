from __future__ import annotations

import json
from pathlib import Path

from config.configmanager import load_expert1_config


def _write_config(path: Path, expert1: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "project": {
                    "data_root": "shared-data",
                    "device": "cpu",
                    "num_workers": 1,
                    "seed": 7,
                },
                "expert1": expert1,
            }
        ),
        encoding="utf-8",
    )


def test_expert_section_values_override_project_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "data_root": "expert1-data",
            "device": "cuda:0",
            "num_workers": 3,
            "seed": 11,
        },
    )

    config = load_expert1_config(config_path)

    assert config.data_root == "expert1-data"
    assert config.device == "cuda:0"
    assert config.num_workers == 3
    assert config.seed == 11


def test_project_values_fill_missing_expert_section_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, {})

    config = load_expert1_config(config_path)

    assert config.data_root == "shared-data"
    assert config.device == "cpu"
    assert config.num_workers == 1
    assert config.seed == 7
