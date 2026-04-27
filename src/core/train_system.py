from __future__ import annotations

import argparse
import json

from config.configmanager import load_system_config
from .system_trainer import train_system_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first route-aware radar classification system.")
    parser.add_argument("--config", type=str, default="config/config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_system_config(args.config)
    summary = train_system_v1(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
