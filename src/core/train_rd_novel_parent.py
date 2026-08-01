from __future__ import annotations

import argparse
import json
from dataclasses import fields

from config.configmanager import load_config_file

from .rd_novel_parent_config import RDNovelParentConfig
from .rd_novel_parent_trainer import train_rd_novel_parent_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RD-only physical-v2 novel-evidence causal parent state; fusion and test are disabled.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = load_config_file(args.config)
    values = dict(payload.get("rd_novel_parent", payload))
    project = payload.get("project", {})
    for key in ("device", "seed"):
        if key in project:
            values.setdefault(key, project[key])
    allowed = {field.name for field in fields(RDNovelParentConfig)}
    config = RDNovelParentConfig(**{key: value for key, value in values.items() if key in allowed})
    print(json.dumps(train_rd_novel_parent_state(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
