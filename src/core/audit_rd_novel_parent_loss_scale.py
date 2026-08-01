from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .rd_novel_parent_data import NovelParentSequenceDataset, collate_novel_parent_batch
from .rd_novel_parent_model import FrozenPhysicalV2NovelParentState
from .rd_novel_parent_trainer import _move_batch, _parent_sequence_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train-only loss scale for RD novel parent-state training.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--index-name", default="expert1_cq08_rd_physical_novel_update_index_v2.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--target-prefix-fraction", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="20260728-stage009t-novel-parent-loss-scale-train-only.json")
    args = parser.parse_args()
    if not 0.0 < args.target_prefix_fraction < 1.0:
        raise ValueError("target-prefix-fraction must be between zero and one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = NovelParentSequenceDataset(Path.cwd(), index_name=args.index_name, split_name="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_novel_parent_batch)
    model = FrozenPhysicalV2NovelParentState(args.base_checkpoint).to(device)
    model.eval()
    final_losses: list[float] = []
    prefix_losses: list[float] = []
    with torch.no_grad():
        for batch_index, source_batch in enumerate(loader, start=1):
            if batch_index > args.max_batches:
                break
            batch = _move_batch(source_batch, device)
            output = model(batch)
            final_loss, prefix_loss = _parent_sequence_losses(output.logits, output.final_logits, batch)
            final_losses.append(float(final_loss.cpu()))
            prefix_losses.append(float(prefix_loss.cpu()))
    if not final_losses:
        raise RuntimeError("loss-scale audit processed zero train batches")
    final_mean = float(np.mean(final_losses))
    prefix_mean = float(np.mean(prefix_losses))
    alpha = args.target_prefix_fraction * final_mean / ((1.0 - args.target_prefix_fraction) * prefix_mean)
    payload = {
        "scope": "train-only no-optimizer loss-scale audit; validation/test/fusion/threshold tuning disabled",
        "base_checkpoint": args.base_checkpoint,
        "index": args.index_name,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "processed_batches": len(final_losses),
        "final_parent_ce_mean": final_mean,
        "weighted_prefix_ce_mean": prefix_mean,
        "target_prefix_fraction": args.target_prefix_fraction,
        "frozen_prefix_loss_weight": alpha,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path = Path.cwd() / "tmp" / args.output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
