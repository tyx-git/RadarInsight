from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import pandas as pd
import torch

from .expert1_streaming import (
    Expert1CheckpointTrajectoryPredictor,
    Expert1StreamingInferenceEngine,
    RDFrameEvent,
    StableDecisionConfig,
    StreamingInferenceResult,
    StreamingWindowConfig,
    TrajectoryEvent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Expert1 validation trajectories through the trajectory-only "
            "streaming path. It never evaluates test data or selects a model."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--index",
        default="data/processed/expert1_realtime_causal_window_index_v7.csv",
        type=Path,
    )
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--parents-per-level", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _quantiles_ns(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(len(values)),
        "mean_ms": float(tensor.mean().item() / 1_000_000.0),
        "p50_ms": float(torch.quantile(tensor, 0.50).item() / 1_000_000.0),
        "p95_ms": float(torch.quantile(tensor, 0.95).item() / 1_000_000.0),
        "p99_ms": float(torch.quantile(tensor, 0.99).item() / 1_000_000.0),
    }


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_validation_parents(index_path: Path, *, parent_limit: int, seed: int) -> list[dict[str, Any]]:
    index = pd.read_csv(index_path, low_memory=False)
    required = {"parent_sample_id", "split_name", "is_trainable"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"window index is missing columns: {sorted(missing)}")
    subset = index.loc[
        (index["split_name"] == "val") & index["is_trainable"].map(_is_true)
    ].copy()
    parent_ids = subset["parent_sample_id"].drop_duplicates().sort_values().tolist()
    if parent_limit > 0 and len(parent_ids) > parent_limit:
        parent_ids = (
            pd.Series(parent_ids)
            .sample(n=parent_limit, random_state=seed)
            .sort_values()
            .tolist()
        )
    track_index_path = index_path.parent / "expert1_track_index.csv"
    track_index = pd.read_csv(track_index_path).set_index("sample_id", drop=False)
    records: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        if parent_id not in track_index.index:
            continue
        row = track_index.loc[parent_id]
        csv_path = Path(str(row["csv_path"]))
        frame = pd.read_csv(csv_path).sort_values(["time_seconds", "record_id"], kind="stable")
        if not frame.empty:
            records.append(
                {
                    "parent_sample_id": str(parent_id),
                    "source_name": str(row["source_name"]),
                    "frame": frame,
                }
            )
    if not records:
        raise ValueError("selected validation parents have no readable trajectory samples")
    return records


def _event_time_ns(value: Any, start_seconds: float) -> int:
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError("trajectory time_seconds must be finite")
    return int(round((seconds - start_seconds) * 1_000_000_000.0))


def _payload(row: dict[str, Any], source_name: str) -> dict[str, Any]:
    normalized = {key: (None if pd.isna(value) else value) for key, value in row.items()}
    normalized["source_name"] = source_name
    return normalized


def _events(parent: dict[str, Any], target_id: str) -> list[TrajectoryEvent]:
    frame = parent["frame"]
    start_seconds = float(frame.iloc[0]["time_seconds"])
    return [
        TrajectoryEvent(
            target_id=target_id,
            event_time_ns=_event_time_ns(row["time_seconds"], start_seconds),
            arrival_time_ns=_event_time_ns(row["time_seconds"], start_seconds),
            sequence_id=sequence_id,
            payload=_payload(row, str(parent["source_name"])),
        )
        for sequence_id, row in enumerate(frame.to_dict(orient="records"))
    ]


def _synchronized_predictor(checkpoint: Path, device: str):
    predictor = Expert1CheckpointTrajectoryPredictor(checkpoint, device=device)
    cuda = predictor.device.type == "cuda" and torch.cuda.is_available()

    def predict(window):
        if cuda:
            torch.cuda.synchronize(predictor.device)
        result = predictor(window)
        if cuda:
            torch.cuda.synchronize(predictor.device)
        return result

    return predictor, predict


def _run_level(
    parents: list[dict[str, Any]], *, checkpoint: Path, device: str, concurrency: int
) -> dict[str, Any]:
    selected = [parents[index % len(parents)] for index in range(concurrency)]
    predictor, synchronized_predict = _synchronized_predictor(checkpoint, device)
    if predictor.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(predictor.device)
    engine = Expert1StreamingInferenceEngine(
        synchronized_predict,
        window_config=StreamingWindowConfig(),
        decision_config=StableDecisionConfig(),
        model_version=predictor.model_version,
        checkpoint_id=predictor.checkpoint_id,
    )
    streams = {
        f"{parent['parent_sample_id']}#{slot}": _events(parent, f"{parent['parent_sample_id']}#{slot}")
        for slot, parent in enumerate(selected)
    }
    results: list[StreamingInferenceResult] = []
    start_ns = perf_counter_ns()
    for position in range(max(len(events) for events in streams.values())):
        for events in streams.values():
            if position < len(events):
                results.extend(engine.ingest_trajectory(events[position]))
    for target_id, events in streams.items():
        results.extend(
            item
            for item in engine.end_target(target_id, events[-1].event_time_ns)
            if isinstance(item, StreamingInferenceResult)
        )
    elapsed_ns = perf_counter_ns() - start_ns
    stats = [engine.target_stats(target_id) for target_id in streams]
    gpu_peak_mib = 0.0
    if predictor.device.type == "cuda" and torch.cuda.is_available():
        gpu_peak_mib = float(torch.cuda.max_memory_allocated(predictor.device) / (1024 * 1024))
    return {
        "concurrency": concurrency,
        "parent_count": len(selected),
        "trajectory_event_count": sum(len(events) for events in streams.values()),
        "inference_window_count": len(results),
        "wall_time_ms": float(elapsed_ns / 1_000_000.0),
        "throughput_windows_per_second": float(len(results) * 1_000_000_000.0 / max(1, elapsed_ns)),
        "end_to_end": _quantiles_ns([result.total_latency_ns for result in results]),
        "preprocess": _quantiles_ns([result.preprocess_latency_ns for result in results]),
        "model": _quantiles_ns([result.model_latency_ns for result in results]),
        "postprocess": _quantiles_ns([result.postprocess_latency_ns for result in results]),
        "fallback_window_count": sum(result.decision.used_fallback for result in results),
        "safe_rd_window_count": sum(result.window.rd_fusion_eligible for result in results),
        "max_tr_buffer_size": max(stat["tr_buffer_size"] for stat in stats),
        "max_rd_buffer_size": max(stat["rd_buffer_size"] for stat in stats),
        "dropped_late_event_count": sum(stat["dropped_late_events"] for stat in stats),
        "dropped_buffer_event_count": sum(stat["dropped_buffer_events"] for stat in stats),
        "gpu_peak_memory_mib": gpu_peak_mib,
    }


def _safety_probe() -> dict[str, Any]:
    received_rd_counts: list[int] = []

    def predictor(window):
        received_rd_counts.append(len(window.rd_events))
        return {"DroneTarget": 1.0}

    engine = Expert1StreamingInferenceEngine(
        predictor,
        window_config=StreamingWindowConfig(prefix_stride_points=100),
        decision_config=StableDecisionConfig(min_confirmations=1),
    )
    engine.ingest_rd(RDFrameEvent("probe", 0, 0, 0, "proportional_fallback"))
    for sequence_id in range(5):
        engine.ingest_trajectory(
            TrajectoryEvent("probe", sequence_id, sequence_id, sequence_id, {"time_seconds": float(sequence_id)})
        )
    inference = [
        item
        for item in engine.end_target("probe", 4)
        if isinstance(item, StreamingInferenceResult)
    ]
    return {
        "unsafe_rd_passed_to_predictor": bool(any(received_rd_counts)),
        "fallback_observed": bool(any(item.decision.used_fallback for item in inference)),
        "inference_window_count": len(inference),
    }


def main() -> None:
    args = parse_args()
    if any(value < 1 for value in args.concurrency):
        raise ValueError("--concurrency values must be positive")
    if args.parents_per_level < 1:
        raise ValueError("--parents-per-level must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    parents = _load_validation_parents(args.index, parent_limit=args.parents_per_level, seed=args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _run_level(parents, checkpoint=args.checkpoint, device=args.device, concurrency=1)
    levels = [
        _run_level(parents, checkpoint=args.checkpoint, device=args.device, concurrency=value)
        for value in sorted(set(args.concurrency))
    ]
    report = {
        "report_scope": "validation_streaming_hardware_benchmark_only",
        "model_selection_used": False,
        "test_evaluation_executed": False,
        "checkpoint": args.checkpoint.as_posix(),
        "index": args.index.as_posix(),
        "split": args.split,
        "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None,
        "parents_available": len(parents),
        "source_parent_counts": {
            str(name): int(count)
            for name, count in pd.Series([parent["source_name"] for parent in parents]).value_counts().items()
        },
        "streaming_window_config": asdict(StreamingWindowConfig()),
        "stable_decision_config": asdict(StableDecisionConfig()),
        "levels": levels,
        "unsafe_rd_safety_probe": _safety_probe(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
