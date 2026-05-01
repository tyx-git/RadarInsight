from __future__ import annotations

import argparse
import json

from config.configmanager import load_pipeline_config
from .pipeline_trainer import run_main_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main training pipeline: pretrain -> router -> system.")
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    summary = run_main_pipeline(config, config_path=args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
