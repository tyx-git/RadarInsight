from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .rd_novel_parent_data import NovelParentSequenceDataset, collate_novel_parent_batch
from .rd_novel_parent_model import FrozenPhysicalV2NovelParentState
from .rd_novel_parent_trainer import _move_batch


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50_ms": float(np.quantile(array, 0.50) * 1_000.0),
        "p95_ms": float(np.quantile(array, 0.95) * 1_000.0),
        "p99_ms": float(np.quantile(array, 0.99) * 1_000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark physical-v2 novel parent-state data loading and forward latency.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--index-name", default="expert1_cq08_rd_physical_novel_update_index_v2.csv")
    parser.add_argument("--split-name", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-name", default="20260728-stage009u-novel-parent-latency-smoke.json")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = NovelParentSequenceDataset(Path.cwd(), index_name=args.index_name, split_name=args.split_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_novel_parent_batch)
    model = FrozenPhysicalV2NovelParentState(args.base_checkpoint).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loading_seconds: list[float] = []
    forward_seconds: list[float] = []
    total_seconds: list[float] = []
    iterator = iter(loader)
    with torch.no_grad():
        for batch_index in range(args.warmup_batches + args.measure_batches):
            loading_started = time.perf_counter()
            source_batch = next(iterator)
            loading_elapsed = time.perf_counter() - loading_started
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            batch = _move_batch(source_batch, device)
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_elapsed = time.perf_counter() - forward_started
            if batch_index >= args.warmup_batches:
                loading_seconds.append(loading_elapsed)
                forward_seconds.append(forward_elapsed)
                total_seconds.append(loading_elapsed + forward_elapsed)
    payload = {
        "scope": "train/validation-only latency smoke for frozen physical-v2 novel parent-state; no training, fusion, threshold tuning, or test evaluation",
        "base_checkpoint": args.base_checkpoint,
        "index": args.index_name,
        "split": args.split_name,
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "data_loading": _percentiles(loading_seconds),
        "parent_state_forward": _percentiles(forward_seconds),
        "end_to_end_batch": _percentiles(total_seconds),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path = Path.cwd() / "tmp" / args.output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
