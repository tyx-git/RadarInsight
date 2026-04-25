from __future__ import annotations

import argparse
import json

from .train_config import TrainConfig
from .trainer import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU baseline on processed radar track dataset.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/baseline_gru")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    result = train_baseline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
