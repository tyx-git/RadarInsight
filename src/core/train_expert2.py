from __future__ import annotations

import argparse
import json

from config.configmanager import load_expert2_config
from .expert_other_trainer import train_other_expert


def main() -> None:
    parser = argparse.ArgumentParser(description="Train expert2 (conventional air expert).")
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()
    config = load_expert2_config(args.config)
    result = train_other_expert(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
