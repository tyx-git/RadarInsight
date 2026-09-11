"""Deployment-oriented measurements for the frozen V27 classification stages.

The benchmark deliberately measures the current model contracts rather than
rebuilding a new end-to-end data pipeline.  Raw L1/L2 inputs come from the
known validation split.  V27a/V27b/V27c inputs come from their sealed frozen
feature caches, so their stage cost is reported separately from feature-cache
materialisation cost.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SRC_V3 = ROOT / "src-v3"
if str(SRC_V3) not in sys.path:
    sys.path.insert(0, str(SRC_V3))

from radar_ln3.runtime import sha256_file  # noqa: E402
from v27.v27_baseline.base_binding import (  # noqa: E402
    load_frozen_base_model,
)
from v27.v27_baseline.data import (  # noqa: E402
    V27_L1_MODEL_INPUT_KEYS,
    V27_L2_MODEL_INPUT_KEYS,
    V27TaskDataset,
    collate_v27,
    model_input_kwargs,
)
from v27.v27_baseline.protocol import (  # noqa: E402
    V27_L1_TASK_NAME,
    V27_L2_TASK_NAME,
    load_v27_config,
)
from v27.v27a.checkpoint import tensor_state_hash as v27a_state_hash  # noqa: E402
from v27.v27a.model import build_v27a_model  # noqa: E402
from v27.v27a.protocol import load_v27a_config  # noqa: E402
from v27.v27b.model import build_v27b_model  # noqa: E402
from v27.v27b.protocol import load_v27b_config, tensor_state_hash as v27b_state_hash  # noqa: E402
from v27.v27c.model import build_v27c_model  # noqa: E402
from v27.v27c.protocol import load_v27c_config  # noqa: E402


DEFAULT_OUTPUT = ROOT / "src-v3" / "v27" / "v27-deployment-benchmark.json"
BASE_CONFIG = ROOT / "src-v3" / "v27" / "v27_baseline" / "config" / "V27-liquid-v2-dbb-baseline.json"
V27A_CONFIG = ROOT / "src-v3" / "v27" / "v27a" / "config" / "V27a-micro-doppler-evidence-decoupling.json"
V27B_CONFIG = ROOT / "src-v3" / "v27" / "v27b" / "config" / "V27b-baseline-direct-frame-gru-r5.json"
V27C_CONFIG = ROOT / "src-v3" / "v27" / "v27c" / "config" / "V27c-causal-tr-motion-late-residual.json"
V27A_CHECKPOINT = ROOT / "src-v3" / "v27" / "v27b" / "inputs" / "V27a-r3-main-seed42-epoch-020.pt"
V27B_CHECKPOINT = ROOT / "src-v3" / "v27" / "v27b" / "runs" / "V27b-r5-conditional-seed42" / "epochs" / "epoch-120.pt"
V27C_CHECKPOINT = ROOT / "src-v3" / "v27" / "v27c" / "runs" / "V27c-r21-motion-seed42" / "epochs" / "epoch-020.pt"
V27A_CACHE = ROOT / "src-v3" / "v27" / "v27b" / "inputs" / "V27a-r3-validation-cache.pt"
V27B_CACHE = ROOT / "src-v3" / "v27" / "v27b" / "calibration" / "features" / "V27b-r5-validation-direct-frame-features.pt"
V27C_CACHE = ROOT / "src-v3" / "v27" / "v27c" / "calibration" / "features" / "V27c-r21-val-frozen-v27a-main-motion-cache.pt"


@dataclass
class Stage:
    name: str
    model: nn.Module
    batches: dict[str, list[dict[str, torch.Tensor]]]
    call: Callable[[nn.Module, Mapping[str, torch.Tensor], bool], Any]
    parameter_count: int
    checkpoint_paths: tuple[Path, ...]
    checkpoint_file_bytes: int
    notes: str


class RSSSampler:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0

    def _run(self) -> None:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            while not self._stop.is_set():
                self.peak = max(self.peak, int(process.memory_info().rss))
                self._stop.wait(self.interval_seconds)
        except Exception:
            return

    def start(self) -> None:
        try:
            import psutil

            self.peak = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            self.peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return int(self.peak)


class MacCounter:
    """Count registered matrix/convolution MACs for one representative call."""

    def __init__(self) -> None:
        self.macs = 0
        self.by_module: dict[str, int] = {}
        self._handles: list[Any] = []

    def attach(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear(name, module)))
            elif isinstance(module, nn.Conv2d):
                self._handles.append(module.register_forward_hook(self._conv(name, module)))
            elif isinstance(module, nn.GRU):
                self._handles.append(module.register_forward_hook(self._gru(name, module)))
            elif isinstance(module, nn.GRUCell):
                self._handles.append(module.register_forward_hook(self._gru_cell(name, module)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _add(self, name: str, value: int) -> None:
        value = int(value)
        self.macs += value
        self.by_module[name] = self.by_module.get(name, 0) + value

    def _linear(self, name: str, module: nn.Linear) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            value = inputs[0]
            if isinstance(value, torch.Tensor):
                rows = value.numel() // module.in_features
                self._add(name, rows * module.in_features * module.out_features)

        return hook

    def _conv(self, name: str, module: nn.Conv2d) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                spatial = int(np.prod(output.shape[2:]))
                per_position = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
                self._add(name, int(output.shape[0]) * spatial * per_position * module.out_channels)

        return hook

    def _gru(self, name: str, module: nn.GRU) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            value = inputs[0]
            if isinstance(value, PackedSequence):
                rows = int(value.data.shape[0])
            elif isinstance(value, torch.Tensor):
                rows = int(value.shape[0] * value.shape[1])
            else:
                return
            layers = int(module.num_layers) * (2 if module.bidirectional else 1)
            per_row = 3 * (module.input_size * module.hidden_size + module.hidden_size * module.hidden_size)
            self._add(name, rows * layers * per_row)

        return hook

    def _gru_cell(self, name: str, module: nn.GRUCell) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            value = inputs[0]
            if isinstance(value, torch.Tensor):
                rows = int(value.numel() // module.input_size)
                per_row = 3 * (module.input_size * module.hidden_size + module.hidden_size * module.hidden_size)
                self._add(name, rows * per_row)

        return hook


def _file_bytes(paths: tuple[Path, ...]) -> int:
    return int(sum(path.stat().st_size for path in paths if path.exists()))


def _parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_state(model: nn.Module, path: Path, state_hash: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint has no model_state: {path}")
    if payload.get("model_state_sha256") != state_hash(state):
        raise ValueError(f"checkpoint state hash differs: {path}")
    if payload.get("model_contract") != model.model_contract():
        raise ValueError(f"checkpoint model contract differs: {path}")
    model.load_state_dict(state, strict=True)
    return dict(payload)


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return torch.from_numpy(np.asarray(value)).clone()


def _slice_batch(batch: Mapping[str, torch.Tensor], rows: slice) -> dict[str, torch.Tensor]:
    return {name: value[rows].contiguous() for name, value in batch.items()}


def _single_event_batches(batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    result: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        item = {name: value[:, :1].contiguous() if value.ndim >= 2 else value for name, value in batch.items()}
        if "tr_padding_mask" in item:
            item["tr_padding_mask"] = torch.zeros_like(item["tr_padding_mask"], dtype=torch.bool)
        if "padding_mask" in item:
            item["padding_mask"] = torch.zeros_like(item["padding_mask"], dtype=torch.bool)
        result.append(item)
    return result


def _split_batches(
    batches: list[dict[str, torch.Tensor]], batch_size: int
) -> list[dict[str, torch.Tensor]]:
    result: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        count = int(next(iter(batch.values())).shape[0])
        for start in range(0, count, batch_size):
            result.append(_slice_batch(batch, slice(start, min(start + batch_size, count))))
    return result


def _raw_batches(root: Path, config: Any, task: str) -> dict[str, dict[str, list[dict[str, torch.Tensor]]]]:
    dataset = V27TaskDataset(
        root,
        config,
        task=task,
        split_name=config.data.validation_split_name,
    )
    # The project training loader enables pin_memory whenever CUDA is present.
    # This preparation step is deliberately host-only; pinning here adds no
    # model signal and can retain unnecessary page-locked memory between runs.
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_v27,
    )
    normal: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        keys = V27_L1_MODEL_INPUT_KEYS if task == V27_L1_TASK_NAME else V27_L2_MODEL_INPUT_KEYS
        values = model_input_kwargs({**batch, "task": task}, torch.device("cpu"))
        normal.append({key: values[key].contiguous() for key in keys})
    result: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {}
    for requested_size in (1, 8, 32):
        normal = _split_batches(normal, requested_size)
        unavailable: list[dict[str, torch.Tensor]] = []
        for batch in normal:
            item = {name: value.clone() for name, value in batch.items()}
            item["rd_frames"].zero_()
            item["rd_frame_mask"].fill_(False)
            item["rd_quality"].zero_()
            if task == V27_L2_TASK_NAME:
                item["rd_physical_features"].zero_()
                item["rd_physical_valid"].fill_(False)
                item["micro_doppler_features"].zero_()
                item["micro_doppler_valid"].fill_(False)
            unavailable.append(item)
        result[str(requested_size)] = {
            "normal_tr_rd": normal,
            "rd_unavailable": unavailable,
            "single_event": _single_event_batches(normal),
        }
    return result


def _cached_batches(payload: Mapping[str, Any], batch_size: int, *, kind: str) -> dict[str, list[dict[str, torch.Tensor]]]:
    if kind == "v27a":
        names = {
            "baseline_logits": "event_baseline_logits",
            "summary_tokens": "event_summary_tokens",
            "summary_valid": "event_summary_valid",
            "frame_values": "event_frame_values",
            "frame_mask": "event_frame_mask",
            "padding_mask": "padding_mask",
        }
    elif kind == "v27b":
        names = {
            "parent_state": "parent_state",
            "event_representation": "event_representation",
            "frozen_l2_logits": "frozen_l2_logits",
            "v27a_residual_logits": "v27a_residual_logits",
            "event_summary_valid": "event_summary_valid",
            "padding_mask": "padding_mask",
            "event_frame_values": "event_frame_values",
            "event_frame_mask": "event_frame_mask",
        }
    elif kind == "v27c":
        tensors = payload["tensors"]
        names = {
            "frozen_endpoint_logits": "frozen_endpoint_logits",
            "motion_values": "motion_values",
            "padding_mask": "padding_mask",
        }
        payload = tensors
    else:
        raise ValueError(kind)
    tensors = {target: _to_tensor(payload[source]) for target, source in names.items()}
    count = int(next(iter(tensors.values())).shape[0])
    normal = []
    for start in range(0, count, batch_size):
        normal.append({name: value[start : start + batch_size].contiguous() for name, value in tensors.items()})
    unavailable = []
    for batch in normal:
        item = {name: value.clone() for name, value in batch.items()}
        if kind == "v27a":
            item["frame_values"].zero_()
            item["frame_mask"].fill_(False)
            item["summary_tokens"].zero_()
            item["summary_valid"].fill_(False)
        elif kind == "v27b":
            item["event_frame_values"].zero_()
            item["event_frame_mask"].fill_(False)
            item["event_summary_valid"].fill_(False)
        else:
            item["motion_values"].zero_()
        unavailable.append(item)
    return {
        "normal": normal,
        "observation_unavailable": unavailable,
        "single_event": _single_event_batches(normal),
    }


def _call_baseline(model: nn.Module, batch: Mapping[str, torch.Tensor], amp: bool) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        return model(**batch)


def _call_v27a(model: nn.Module, batch: Mapping[str, torch.Tensor], amp: bool) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        return model(**batch)


def _call_v27b(model: nn.Module, batch: Mapping[str, torch.Tensor], amp: bool) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        return model(**batch)


def _call_v27c(model: nn.Module, batch: Mapping[str, torch.Tensor], amp: bool) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        return model(**batch)


def _output_tensor(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not isinstance(output, Mapping):
        raise ValueError("benchmark model output is not a mapping")
    if isinstance(output.get("window_logits"), torch.Tensor):
        return output["window_logits"]
    sequence = output.get("sequence_logits")
    if not isinstance(sequence, torch.Tensor):
        raise ValueError("benchmark model output has no logits")
    padding = batch.get("tr_padding_mask", batch.get("padding_mask"))
    lengths = (~padding).long().sum(dim=1)
    rows = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[rows, lengths - 1]


def _device_batches(batches: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{name: value.to(device=device, non_blocking=False) for name, value in batch.items()} for batch in batches]


def _device_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device, non_blocking=False) for name, value in batch.items()}


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _run_forward(stage: Stage, batch: Mapping[str, torch.Tensor], amp: bool) -> torch.Tensor:
    return _output_tensor(stage.call(stage.model, batch, amp), batch)


def _measure(
    stage: Stage,
    *,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
    variant: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    max_latency_batches: int,
    max_throughput_batches: int,
) -> dict[str, Any]:
    model = stage.model.to(device).eval()
    amp = precision == "fp16_amp"
    if not batches:
        raise ValueError(f"no batches for {stage.name}/{variant}/{batch_size}")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def timed_forward(cpu_batch: Mapping[str, torch.Tensor]) -> float:
        device_batch = _device_batch(cpu_batch, device)
        try:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _run_forward(stage, device_batch, amp)
                end.record()
                end.synchronize()
                return float(start.elapsed_time(end))
            start_time = time.perf_counter()
            _run_forward(stage, device_batch, amp)
            return (time.perf_counter() - start_time) * 1000.0
        finally:
            del device_batch

    for _ in range(max(1, warmup)):
        timed_forward(batches[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    sampler = RSSSampler()
    sampler.start()
    latencies: list[float] = []
    measured = batches[: max(1, min(len(batches), max_latency_batches))]
    for _ in range(max(1, repeats)):
        for batch in measured:
            latencies.append(timed_forward(batch))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    throughput_batches = batches[: max(1, min(len(batches), max_throughput_batches))]
    throughput_latencies: list[float] = []
    for _ in range(max(1, repeats)):
        for batch in throughput_batches:
            throughput_latencies.append(timed_forward(batch))
    elapsed_total = sum(throughput_latencies) / 1000.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak_rss = sampler.stop()
    sample_count = int(sum(int(batch[next(iter(batch))].shape[0]) for batch in throughput_batches) * max(1, repeats))
    padding_name = "tr_padding_mask" if "tr_padding_mask" in throughput_batches[0] else "padding_mask"
    event_count = int(
        sum(int((~batch[padding_name]).sum().item()) for batch in throughput_batches) * max(1, repeats)
    )
    mac_counter = MacCounter()
    mac_counter.attach(model)
    mac_batch = _device_batch(batches[0], device)
    _run_forward(stage, mac_batch, amp=False)
    del mac_batch
    mac_counter.close()
    first_batch_count = int(batches[0][next(iter(batches[0]))].shape[0])
    macs_per_batch = int(mac_counter.macs)
    macs_per_sample = float(macs_per_batch / max(1, first_batch_count))
    allocated = reserved = None
    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "stage": stage.name,
        "variant": variant,
        "device": str(device),
        "precision": precision,
        "batch_size": int(batch_size),
        "sample_count": sample_count,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": float(max(latencies)),
            "mean": float(np.mean(latencies)),
            "calls": len(latencies),
        },
        "throughput": {
            "samples_per_second": float(sample_count / max(elapsed_total, 1.0e-12)),
            "batches_per_second": float((len(throughput_batches) * max(1, repeats)) / max(elapsed_total, 1.0e-12)),
            "events_per_second": float(event_count / max(elapsed_total, 1.0e-12)),
        },
        "macs": {
            "per_representative_batch": macs_per_batch,
            "per_sample": macs_per_sample,
            "flops_2x_macs_per_sample": 2.0 * macs_per_sample,
            "counted_modules": mac_counter.by_module,
            "method": "analytic Linear/Conv2d/GRU/GRUCell matmul MACs; elementwise and normalization operations excluded",
        },
        "memory": {
            "cpu_peak_rss_bytes": peak_rss,
            "cuda_peak_allocated_bytes": allocated,
            "cuda_peak_reserved_bytes": reserved,
        },
        "parameter_count": stage.parameter_count,
        "checkpoint_file_bytes": stage.checkpoint_file_bytes,
    }


def _load_stages(root: Path, base_config: Any) -> tuple[dict[str, Stage], dict[str, Any]]:
    torch.manual_seed(42)
    context = SimpleNamespace(project_root=root, config=base_config)
    l1, l1_payload, binding = load_frozen_base_model(context, task=V27_L1_TASK_NAME, device=torch.device("cpu"))
    torch.manual_seed(42)
    l2, l2_payload, _ = load_frozen_base_model(context, task=V27_L2_TASK_NAME, device=torch.device("cpu"))
    l1.eval()
    l2.eval()
    a_config = load_v27a_config(root, V27A_CONFIG)
    a_model = build_v27a_model(a_config, arm="main", seed=42)
    a_payload = _load_state(a_model, V27A_CHECKPOINT, v27a_state_hash)
    b_config = load_v27b_config(root, V27B_CONFIG)
    b_model = build_v27b_model(b_config, seed=42)
    b_payload = _load_state(b_model, V27B_CHECKPOINT, v27b_state_hash)
    c_config = load_v27c_config(root, V27C_CONFIG)
    c_model = build_v27c_model(c_config, seed=42)
    c_payload = torch.load(V27C_CHECKPOINT, map_location="cpu", weights_only=False)
    c_model.load_state_dict(c_payload["model_state"], strict=True)
    for model in (l1, l2, a_model, b_model, c_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    raw_l1 = _raw_batches(root, base_config, V27_L1_TASK_NAME)
    raw_l2 = _raw_batches(root, base_config, V27_L2_TASK_NAME)
    a_payload_cache = torch.load(V27A_CACHE, map_location="cpu", weights_only=False)
    b_payload_cache = torch.load(V27B_CACHE, map_location="cpu", weights_only=False)
    c_payload_cache = torch.load(V27C_CACHE, map_location="cpu", weights_only=False)
    stages: dict[str, Stage] = {}
    l1_path = root / binding["l1"]["checkpoint_path"]
    l2_path = root / binding["l2"]["checkpoint_path"]
    stages["L1_known_gate"] = Stage(
        "L1_known_gate", l1, raw_l1, _call_baseline, _parameters(l1), (l1_path,), _file_bytes((l1_path,)), "Frozen V27 L1 classifier base; trajectory heads remain present for the current forward contract.",
    )
    stages["L2_base"] = Stage(
        "L2_base", l2, raw_l2, _call_baseline, _parameters(l2), (l2_path,), _file_bytes((l2_path,)), "Frozen V27 L2 Liquid V2 classifier base.",
    )
    stages["V27a_main_residual"] = Stage(
        "V27a_main_residual", a_model, {str(size): _cached_batches(a_payload_cache, size, kind="v27a") for size in (1, 8, 32)}, _call_v27a, _parameters(a_model), (V27A_CHECKPOINT,), _file_bytes((V27A_CHECKPOINT,)), "Incremental V27a main residual stage over the frozen L2 endpoint.",
    )
    stages["V27b_r5_residual"] = Stage(
        "V27b_r5_residual", b_model, {str(size): _cached_batches(b_payload_cache, size, kind="v27b") for size in (1, 8, 32)}, _call_v27b, _parameters(b_model), (V27B_CHECKPOINT,), _file_bytes((V27B_CHECKPOINT,)), "Incremental V27b-r5 direct-frame GRU residual stage; not a standalone raw-input L2 model.",
    )
    stages["V27c_r21_motion_residual"] = Stage(
        "V27c_r21_motion_residual", c_model, {str(size): _cached_batches(c_payload_cache, size, kind="v27c") for size in (1, 8, 32)}, _call_v27c, _parameters(c_model), (V27C_CHECKPOINT,), _file_bytes((V27C_CHECKPOINT,)), "Known-only motion residual comparison; report shows zero paired classification delta and it is not the primary L2 candidate.",
    )
    provenance = {
        "binding_sha256": binding.get("binding_sha256"),
        "l1_checkpoint": str(binding["l1"]["checkpoint_path"]),
        "l2_checkpoint": str(binding["l2"]["checkpoint_path"]),
        "l1_checkpoint_sha256": binding["l1"]["checkpoint_sha256"],
        "l2_checkpoint_sha256": binding["l2"]["checkpoint_sha256"],
        "v27a_checkpoint": str(V27A_CHECKPOINT.relative_to(root)),
        "v27a_checkpoint_sha256": sha256_file(V27A_CHECKPOINT),
        "v27a_epoch": int(a_payload.get("epoch", -1)),
        "v27b_checkpoint": str(V27B_CHECKPOINT.relative_to(root)),
        "v27b_checkpoint_sha256": sha256_file(V27B_CHECKPOINT),
        "v27b_epoch": int(b_payload.get("epoch", -1)),
        "v27c_checkpoint": str(V27C_CHECKPOINT.relative_to(root)),
        "v27c_checkpoint_sha256": sha256_file(V27C_CHECKPOINT),
        "v27c_epoch": int(c_payload.get("epoch", -1)),
        "active_manifest_load": "rejected_as_stale_worktree; frozen binding and checkpoint contracts verified",
    }
    return stages, provenance


def _run_consistency(stage: Stage, device: torch.device) -> dict[str, Any]:
    cpu_model = stage.model.to("cpu").eval()
    cpu_batch = next(iter(stage.batches["1"].values()))[0]
    with torch.inference_mode():
        cpu_output = _output_tensor(stage.call(cpu_model, cpu_batch, False), cpu_batch).detach().cpu().float()
    result: dict[str, Any] = {"cpu_fp32_finite": bool(torch.isfinite(cpu_output).all())}
    if device.type != "cuda":
        return result
    gpu_model = stage.model.to(device).eval()
    gpu_batch = _device_batches([cpu_batch], device)[0]
    with torch.inference_mode():
        gpu_output = _output_tensor(stage.call(gpu_model, gpu_batch, False), gpu_batch).detach().cpu().float()
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                amp_output = _output_tensor(stage.call(gpu_model, gpu_batch, True), gpu_batch).detach().cpu().float()
        except Exception as exc:
            result.update(
                {
                    "amp_supported": False,
                    "amp_error_type": type(exc).__name__,
                    "amp_error": str(exc),
                    "cuda_fp32_finite": bool(torch.isfinite(gpu_output).all()),
                }
            )
            return result
    result.update(
        {
            "amp_supported": True,
            "cuda_fp32_finite": bool(torch.isfinite(gpu_output).all()),
            "amp_finite": bool(torch.isfinite(amp_output).all()),
            "cpu_vs_cuda_fp32_max_abs": float((cpu_output - gpu_output).abs().max()),
            "cpu_vs_cuda_fp32_max_rel": float(((cpu_output - gpu_output).abs() / cpu_output.abs().clamp_min(1.0e-6)).max()),
            "cuda_fp32_vs_amp_max_abs": float((gpu_output - amp_output).abs().max()),
            "cuda_fp32_vs_amp_max_rel": float(((gpu_output - amp_output).abs() / gpu_output.abs().clamp_min(1.0e-6)).max()),
        }
    )
    return result


def _model_spec(stages: Mapping[str, Stage]) -> dict[str, Any]:
    return {
        name: {
            "parameter_count": stage.parameter_count,
            "checkpoint_file_bytes": stage.checkpoint_file_bytes,
            "notes": stage.notes,
            "checkpoint_paths": [str(path.relative_to(ROOT)) for path in stage.checkpoint_paths],
        }
        for name, stage in stages.items()
    }


def _composed_specs(stages: Mapping[str, Stage]) -> dict[str, Any]:
    base_l2 = stages["L2_base"]
    a = stages["V27a_main_residual"]
    b = stages["V27b_r5_residual"]
    c = stages["V27c_r21_motion_residual"]
    l1 = stages["L1_known_gate"]
    return {
        "L1_plus_L2_base": {
            "stages": [l1.name, base_l2.name],
            "parameter_count": l1.parameter_count + base_l2.parameter_count,
            "checkpoint_file_bytes": l1.checkpoint_file_bytes + base_l2.checkpoint_file_bytes,
        },
        "L1_plus_L2_V27b_r5": {
            "stages": [l1.name, base_l2.name, a.name, b.name],
            "parameter_count": l1.parameter_count + base_l2.parameter_count + a.parameter_count + b.parameter_count,
            "checkpoint_file_bytes": l1.checkpoint_file_bytes + base_l2.checkpoint_file_bytes + a.checkpoint_file_bytes + b.checkpoint_file_bytes,
            "interpretation": "serial sum of independently measured raw L1/L2 and frozen feature-contract residual stages; feature extraction and L3 are excluded",
        },
        "L1_plus_L2_V27c_r21": {
            "stages": [l1.name, base_l2.name, a.name, c.name],
            "parameter_count": l1.parameter_count + base_l2.parameter_count + a.parameter_count + c.parameter_count,
            "checkpoint_file_bytes": l1.checkpoint_file_bytes + base_l2.checkpoint_file_bytes + a.checkpoint_file_bytes + c.checkpoint_file_bytes,
            "interpretation": "serial sum of independently measured raw L1/L2 and V27c motion residual stages; feature extraction and L3 are excluded",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen V27 deployment stages")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-latency-batches", type=int, default=16)
    parser.add_argument("--max-throughput-batches", type=int, default=32)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    base_config = load_v27_config(ROOT, BASE_CONFIG)
    stages, provenance = _load_stages(ROOT, base_config)
    devices: list[tuple[str, torch.device, str]] = []
    if not args.skip_cuda and torch.cuda.is_available():
        devices.extend([("cuda_fp32", torch.device("cuda:0"), "fp32"), ("cuda_fp16_amp", torch.device("cuda:0"), "fp16_amp")])
    if not args.skip_cpu:
        devices.append(("cpu_fp32", torch.device("cpu"), "fp32"))
    measurements: list[dict[str, Any]] = []
    consistency: dict[str, Any] = {}
    for stage_name, stage in stages.items():
        consistency[stage_name] = _run_consistency(stage, torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        for _device_name, device, precision in devices:
            if precision == "fp16_amp" and consistency[stage_name].get("amp_supported") is False:
                continue
            for batch_size in (1, 8, 32):
                for variant in stage.batches[str(batch_size)]:
                    measurements.append(
                        _measure(
                            stage,
                            batches=stage.batches[str(batch_size)][variant],
                            device=device,
                            precision=precision,
                            variant=variant,
                            batch_size=batch_size,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            max_latency_batches=args.max_latency_batches,
                            max_throughput_batches=args.max_throughput_batches,
                        )
                    )
    report = {
        "schema_version": "RADAR-LN3-V27-deployment-benchmark-v1",
        "status": "completed",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        },
        "protocol": {
            "data_scope": "CQ08 known validation only for raw L1/L2; sealed V27a/V27b/V27c validation feature caches for residual stages",
            "unknown_payload_access": 0,
            "test_payload_access": 0,
            "l3_status": "unfitted_and_not_benchmarked",
            "batch_sizes": [1, 8, 32],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_latency_batches": args.max_latency_batches,
            "max_throughput_batches": args.max_throughput_batches,
            "fused_io_or_preprocessing": False,
            "mac_definition": "2x MACs for FLOPs; only registered Linear/Conv2d/GRU/GRUCell matmuls counted",
        },
        "provenance": provenance,
        "models": _model_spec(stages),
        "composed_parameter_summaries": _composed_specs(stages),
        "consistency": consistency,
        "measurements": measurements,
        "limitations": [
            "The active V27 manifests are stale relative to the current worktree; immutable frozen binding/checkpoint contracts were verified instead of rewriting manifests.",
            "V27a/V27b/V27c measurements exclude raw feature-cache materialisation and L3 scoring/calibration because L3 is not fitted.",
            "Peak RSS is sampled process RSS; CUDA memory is torch peak allocated/reserved memory.",
            "The reported FLOPs are twice the counted matmul MACs and exclude elementwise, normalization, packing and memory movement costs.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "measurement_count": len(measurements)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
