from __future__ import annotations

import argparse
import json
from pathlib import Path

from .system_config import SystemTrainConfig
from .system_trainer import train_system_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first route-aware radar classification system.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/system_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SystemTrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
    )
    summary = train_system_v1(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
