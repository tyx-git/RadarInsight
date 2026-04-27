from __future__ import annotations

import argparse
import json

from config.configmanager import load_pretrain_track_config
from .pretrain_track_trainer import pretrain_track_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain track encoder with MTM + InfoNCE.")
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()

    config = load_pretrain_track_config(args.config)
    result = pretrain_track_encoder(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
