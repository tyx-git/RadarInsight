from __future__ import annotations

import argparse
import json
from dataclasses import fields

from config.configmanager import load_config_file

from .rd_physical_drone_bird_config import PhysicalRDDroneBirdConfig
from .rd_physical_drone_bird_trainer import train_physical_rd_drone_bird


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physical RD-only Drone/Bird classifier; test and fusion are disabled.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = load_config_file(args.config)
    values = dict(payload.get("rd_physical_only", payload))
    project = payload.get("project", {})
    for key in ("device", "seed"):
        if key in project:
            values.setdefault(key, project[key])
    allowed = {field.name for field in fields(PhysicalRDDroneBirdConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown PhysicalRDDroneBirdConfig fields: {', '.join(unknown)}")
    config = PhysicalRDDroneBirdConfig(**values)
    print(json.dumps(train_physical_rd_drone_bird(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
