from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .rd_patch_residual_config import RDPatchResidualConfig
from .rd_patch_residual_trainer import train_rd_patch_residual


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-sequence RD patch residual diagnostic.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    values = dict(payload.get("rd_patch_residual", {}))
    project = dict(payload.get("project", {}))
    for name in ("device", "seed"):
        if name in project:
            values[name] = project[name]
    allowed = {field.name for field in fields(RDPatchResidualConfig)}
    config = RDPatchResidualConfig(**{key: value for key, value in values.items() if key in allowed})
    print(json.dumps(train_rd_patch_residual(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
