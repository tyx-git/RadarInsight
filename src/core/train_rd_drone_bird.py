from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from config.configmanager import load_config_file
from .rd_drone_bird_config import RDDroneBirdDiagnosticConfig
from .rd_drone_bird_trainer import train_rd_drone_bird_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diagnostic-only RD Drone/Bird expert; test and fusion are disabled.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = load_config_file(args.config)
    values = dict(payload.get("rd_only", payload))
    project = payload.get("project", {})
    for key in ("device", "seed"):
        if key in project:
            values.setdefault(key, project[key])
    allowed = {field.name for field in fields(RDDroneBirdDiagnosticConfig)}
    config = RDDroneBirdDiagnosticConfig(**{key: value for key, value in values.items() if key in allowed})
    print(json.dumps(train_rd_drone_bird_diagnostic(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
