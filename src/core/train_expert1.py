from __future__ import annotations

import argparse
import json

from config.configmanager import load_expert1_config
from .expert1_trainer import train_expert1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train expert1 (low-altitude multimodal RD expert) only.")
    parser.add_argument("--config", type=str, default="config/config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_expert1_config(args.config)
    summary = train_expert1(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
