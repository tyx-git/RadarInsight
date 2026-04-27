from __future__ import annotations

import argparse
import json

from config.configmanager import load_router_config
from .router_trainer import train_router_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone router on physical features.")
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()

    config = load_router_config(args.config)
    result = train_router_v1(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
