from __future__ import annotations

import argparse
import json
from dataclasses import fields

from config.configmanager import load_config_file

from .rd_physical_specialist_config import PhysicalRDSpecialistConfig
from .rd_physical_specialist_trainer import train_physical_rd_specialist


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an independent RD-only ParentGate, DBB, or CU specialist; TR entry is preserved but TR inputs are disabled.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(); payload = load_config_file(args.config); values = dict(payload.get("rd_physical_specialist", payload)); project = payload.get("project", {})
    for key in ("device", "seed"):
        if key in project: values.setdefault(key, project[key])
    allowed = {field.name for field in fields(PhysicalRDSpecialistConfig)}; unknown = sorted(set(values) - allowed)
    if unknown: raise ValueError(f"unknown PhysicalRDSpecialistConfig fields: {unknown}")
    print(json.dumps(train_physical_rd_specialist(PhysicalRDSpecialistConfig(**values)), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
