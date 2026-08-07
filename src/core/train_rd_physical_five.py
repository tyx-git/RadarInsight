from __future__ import annotations

import argparse
import json
from dataclasses import fields

from config.configmanager import load_config_file

from .rd_physical_five_config import PhysicalRDFiveConfig
from .rd_physical_five_trainer import train_physical_rd_five


def main() -> None:
    parser = argparse.ArgumentParser(description="Train five-class physical RD-only classifier; TR entry is preserved but TR inputs are disabled.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = load_config_file(args.config)
    values = dict(payload.get("rd_physical_five", payload))
    project = payload.get("project", {})
    for key in ("device", "seed"):
        if key in project:
            values.setdefault(key, project[key])
    allowed = {field.name for field in fields(PhysicalRDFiveConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown PhysicalRDFiveConfig fields: {', '.join(unknown)}")
    config = PhysicalRDFiveConfig(**values)
    print(json.dumps(train_physical_rd_five(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
