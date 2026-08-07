from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .data.rd_physical_window_dataset import PhysicalRDWindowBatch, PhysicalRDWindowDataset, collate_physical_rd_windows
from .rd_physical_artifacts import load_physical_rd_checkpoint


def latency_summary(samples_ms: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(samples_ms), dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency summary requires at least one sample")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("latency samples must be finite non-negative milliseconds")
    return {
        "sample_count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def _to_device(batch: PhysicalRDWindowBatch, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch.features.to(device, non_blocking=True),
        batch.time_seconds.to(device, non_blocking=True),
        batch.padding_mask.to(device, non_blocking=True),
        batch.quality_features.to(device, non_blocking=True),
        batch.range_offset_m.to(device, non_blocking=True),
        batch.velocity_mps.to(device, non_blocking=True),
        batch.raw_amplitude_db.to(device, non_blocking=True),
    )


def benchmark_physical_rd_stream(
    checkpoint_path: Path,
    output_path: Path,
    *,
    device_name: str = "cuda",
    sample_count: int = 64,
    warmup_count: int = 8,
) -> dict[str, object]:
    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise RuntimeError("physical RD stream benchmark requires CUDA")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative")

    project_root = Path.cwd()
    device = torch.device(device_name)
    loaded = load_physical_rd_checkpoint(checkpoint_path, project_root, device)
    dataset = PhysicalRDWindowDataset(
        project_root / "data",
        index_name=loaded.config.index_name,
        split_name="val",
        normalization_stats_name=loaded.config.normalization_stats_name,
        include_missing_rd=loaded.config.include_missing_rd,
        quality_schema=loaded.config.quality_schema,
    )
    required_samples = warmup_count + sample_count
    if required_samples > len(dataset):
        raise ValueError(
            f"requested warmup + measured samples ({required_samples}) exceeds natural validation windows ({len(dataset)})"
        )

    def run_one(item_index: int, *, measure: bool) -> dict[str, float] | None:
        total_start = time.perf_counter()
        batch = collate_physical_rd_windows([dataset[item_index]])
        preprocess_end = time.perf_counter()
        torch.cuda.synchronize(device)
        transfer_start = time.perf_counter()
        inputs = _to_device(batch, device)
        torch.cuda.synchronize(device)
        transfer_end = time.perf_counter()
        with torch.inference_mode():
            output = loaded.model(*inputs)
        torch.cuda.synchronize(device)
        forward_end = time.perf_counter()
        if not torch.isfinite(output.logits).all():
            raise ValueError(f"non-finite physical RD logits at validation item {item_index}")
        if not measure:
            return None
        return {
            "preprocess_ms": (preprocess_end - total_start) * 1000.0,
            "h2d_ms": (transfer_end - transfer_start) * 1000.0,
            "forward_ms": (forward_end - transfer_end) * 1000.0,
            "total_ms": (forward_end - total_start) * 1000.0,
        }

    loaded.model.eval()
    for item_index in range(warmup_count):
        run_one(item_index, measure=False)
    torch.cuda.synchronize(device)
    allocated_before_bytes = int(torch.cuda.memory_allocated(device))
    reserved_before_bytes = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    rows = [run_one(warmup_count + item_index, measure=True) for item_index in range(sample_count)]
    torch.cuda.synchronize(device)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    measurements = [row for row in rows if row is not None]
    payload = {
        "scope": "physical RD-only natural-validation sequential streaming latency; closed-set Drone/Bird, no Unknown/OOD rejection, batch_size=1, no fusion or test evaluation",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_schema_version": loaded.checkpoint["checkpoint_schema_version"],
        "checkpoint_epoch": int(loaded.checkpoint["epoch"]),
        "artifact_manifest": loaded.artifacts.manifest,
        "device": {
            "requested": device_name,
            "resolved": str(device),
            "name": torch.cuda.get_device_name(device),
        },
        "stream_contract": {
            "split_name": "val",
            "batch_size": 1,
            "dataset_order": "PhysicalRDWindowDataset sorted parent_sample_id, rd_time_end_seconds, sample_id",
            "warmup_count": warmup_count,
            "measured_sample_count": sample_count,
            "includes_dataset_preprocess": True,
            "includes_host_to_device": True,
            "includes_cuda_synchronization": True,
            "test_evaluation_executed": False,
            "fusion_enabled": False,
        },
        "latency_ms": {name: latency_summary([float(row[name]) for row in measurements]) for name in measurements[0]},
        "cuda_memory_bytes": {
            "allocated_before_measurement": allocated_before_bytes,
            "reserved_before_measurement": reserved_before_bytes,
            "peak_allocated_during_measurement": peak_allocated_bytes,
            "peak_reserved_during_measurement": peak_reserved_bytes,
            "peak_allocated_increment": max(0, peak_allocated_bytes - allocated_before_bytes),
            "peak_reserved_increment": max(0, peak_reserved_bytes - reserved_before_bytes),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark physical RD-only natural-validation streaming latency.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--warmup-count", type=int, default=8)
    args = parser.parse_args()
    result = benchmark_physical_rd_stream(
        Path(args.checkpoint),
        Path(args.output),
        device_name=args.device,
        sample_count=args.sample_count,
        warmup_count=args.warmup_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
