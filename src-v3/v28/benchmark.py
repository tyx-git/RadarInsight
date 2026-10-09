"""V28 deployment benchmark: parameters, FLOPs, latency, throughput, memory and metrics.

The benchmark deliberately reports several *non-interchangeable* accuracy
scopes.  A single "F1" number for this system is meaningless because the
project stacks three semantic gates (knownness, conformal candidate set, and
the fail-closed L3 resolver) on top of the closed-set classifiers:

``known_only_conditional``
    L1/L2 closed-set accuracy on known classes only, computed per task on its
    own dataset.  This is the scope that the historical "Micro F1 ~0.96"
    figures refer to.  It contains no Unknown samples and no abstention.

``end_to_end_audit``
    L1 -> L2 -> L3 chain over the frozen test replay pool with
    ``audit_mode=True``.  Unknown targets, clutter and abstention are all
    present.  Per-class F1 here is expected to be far below the conditional
    scope; the gap is the cost of open-set safety, not a regression.

``end_to_end_fail_closed``
    The production path.  If the registry is not ``deployment_eligible`` the
    resolver abstains on every sample by design.

Measurement conditions (device, thread count, dtype, batch size, tensor
shapes) are recorded verbatim so that numbers cannot be quoted out of context.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

if __package__ in {None, ""}:  # allow ``python src-v3/v28/benchmark.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v28.checkpoint import load_checkpoint
    from v28.data import V28ReplayDataset, V28TaskDataset, build_dataloader, collate_v28, model_input_kwargs
    from v28.heads import V28TrajectoryHeads, deterministic_quality_target
    from v28.l3 import resolve_action_verified
    from v28.metrics import confusion_metrics
    from v28.model import V28TaskModel
    from v28.protocol import (
        V28_L1_TASK,
        V28_L2_CLASSES,
        V28_L2_TASK,
        V28_TASKS,
        artifact_path,
        load_context,
        read_sealed_json,
    )
    from radar_ln3.runtime import resolve_project_path
    from v28.registry import verify_registry
    from v28.replay import _action_metrics, _candidate_set, _probabilities
else:
    from .checkpoint import load_checkpoint
    from .data import V28ReplayDataset, V28TaskDataset, build_dataloader, collate_v28, model_input_kwargs
    from .heads import V28TrajectoryHeads, deterministic_quality_target
    from .l3 import resolve_action_verified
    from .metrics import confusion_metrics
    from .model import V28TaskModel
    from .protocol import (
        V28_L1_TASK,
        V28_L2_CLASSES,
        V28_L2_TASK,
        V28_TASKS,
        artifact_path,
        load_context,
        read_sealed_json,
    )
    from radar_ln3.runtime import resolve_project_path
    from .registry import verify_registry
    from .replay import _action_metrics, _candidate_set, _probabilities

from radar_ln3.runtime import relative_project_path, resolve_project_path, sha256_file, atomic_write_text  # noqa: E402

LOGGER = logging.getLogger("radar_ln3.v28.benchmark")

BENCHMARK_SCHEMA = "RADAR-LN3-V28-deployment-benchmark-v1"

#: Linear/Conv2d MACs only.  Element-wise ops, LayerNorm, activations and
#: memory traffic are excluded.  Declared explicitly so the number is never
#: quoted without its scope.
FLOPS_DEFINITION_NARROW = (
    "FLOPS=2*MACs, counting only Linear/Conv2d matrix multiplies; "
    "element-wise ops, LayerNorm, softmax, activations and memory traffic are excluded."
)
FLOPS_DEFINITION_TORCH = (
    "torch.utils.flop_counter.FlopCounterMode, which additionally counts "
    "element-wise arithmetic and normalization; it does not model memory traffic."
)

#: How many times the hook-free baseline forward pass is repeated before its
#: median is taken as the reference. A single repeat cannot reveal a competing
#: process; a handful can.
BASELINE_SAMPLE_COUNT = 7

#: If the slowest baseline repeat exceeds this multiple of the median, the device
#: was not idle and the run's latency figures are not quotable. Calibrated from
#: the observed contrast between an idle device (<1.1x) and one running a
#: concurrent CUDA process (>3x).
BASELINE_UNSTABLE_RATIO = 2.0


def _dispersion_ratio(values: Sequence[float]) -> float | None:
    """max/median for a set of per-call timings; ``None`` when undefined."""

    if not values:
        return None
    median = statistics.median(values)
    if median <= 0:
        return None
    return float(max(values) / median)


# --------------------------------------------------------------------------- #
# measurement conditions
# --------------------------------------------------------------------------- #


def _thread_settings() -> dict[str, Any]:
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "cpu_count_os": os.cpu_count(),
    }


def measurement_conditions(device: torch.device, *, batch_size: int, dtype: str) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "batch_size": int(batch_size),
        "parameter_dtype": str(dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": _thread_settings(),
        "timing_note": (
            "Latency is measured with per-sample cuda synchronisation; throughput is wall-clock "
            "over distinct samples and includes host-side input assembly."
        ),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        conditions["cuda"] = {
            "device_name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return conditions


# --------------------------------------------------------------------------- #
# parameter / size accounting
# --------------------------------------------------------------------------- #


def parameter_accounting(model: nn.Module, checkpoint: Path | None) -> dict[str, Any]:
    by_module: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        by_module[name] = by_module.get(name, 0) + int(parameter.numel())
    total = sum(by_module.values())
    bytes_fp32 = sum(int(p.numel()) * int(p.element_size()) for p in model.parameters())
    result: dict[str, Any] = {
        "parameter_count": int(total),
        "parameter_bytes": int(bytes_fp32),
        "parameter_bytes_definition": "sum(numel * element_size) over parameters at their live dtype",
        "by_parameter_name": dict(sorted(by_module.items())),
    }
    if checkpoint is not None and Path(checkpoint).is_file():
        result["checkpoint_path"] = Path(checkpoint).as_posix()
        result["checkpoint_file_bytes"] = int(Path(checkpoint).stat().st_size)
        result["checkpoint_sha256"] = sha256_file(Path(checkpoint))
    return result


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #


class MacCounter:
    """Counts Linear/Conv2d MACs via forward hooks (narrow scope)."""

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

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _add(self, name: str, value: int) -> None:
        self.macs += int(value)
        self.by_module[name] = self.by_module.get(name, 0) + int(value)

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
                position = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
                self._add(name, int(output.shape[0]) * spatial * position * module.out_channels)

        return hook


def _narrow_flops(model: nn.Module, forward: Callable[[], Any]) -> dict[str, Any]:
    counter = MacCounter()
    counter.attach(model)
    try:
        with torch.inference_mode():
            forward()
    finally:
        counter.close()
    return {
        "macs": int(counter.macs),
        "flops": int(counter.macs * 2),
        "definition": FLOPS_DEFINITION_NARROW,
        "by_module": dict(sorted(counter.by_module.items())),
    }


def _torch_flops(forward: Callable[[], Any]) -> dict[str, Any]:
    """Count FLOPs with torch's dispatcher-based counter.

    ``torch.inference_mode`` must NOT wrap the counted call: it disables the
    dispatch interception that ``FlopCounterMode`` relies on, and the counter
    then silently reports zero while still returning ``status="ok"``. Only
    ``no_grad`` is compatible, because it keeps dispatch active while avoiding
    graph construction.
    """

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover - torch too old
        return {"flops": None, "definition": FLOPS_DEFINITION_TORCH, "status": "flop_counter_unavailable"}
    mode = FlopCounterMode(display=False)
    try:
        with mode, torch.no_grad():
            forward()
    except Exception as exc:  # noqa: BLE001 - counter must never mask a real failure
        return {"flops": None, "definition": FLOPS_DEFINITION_TORCH, "status": f"failed: {type(exc).__name__}: {exc}"}
    counts = mode.get_flop_counts().get("Global", {})
    by_operator = {str(key): int(value) for key, value in counts.items()}
    total = int(sum(by_operator.values()))
    status = "ok" if total > 0 else "zero_ops_recorded"
    result = {
        "flops": total,
        "definition": FLOPS_DEFINITION_TORCH,
        "status": status,
        "by_operator": dict(sorted(by_operator.items())),
    }
    if total == 0:
        result["note"] = (
            "The dispatcher recorded no recognised matrix-multiply operators. Verify that the call "
            "is not wrapped in torch.inference_mode and that a real CUDA kernel was launched."
        )
    return result


# --------------------------------------------------------------------------- #
# memory sampling (CPU side, for the edge-board analogue)
# --------------------------------------------------------------------------- #


class RSSSampler:
    """Samples resident set size on a background thread to catch transient peaks."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        import threading

        self._interval = float(interval_seconds)
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _rss_bytes() -> int:
        try:
            import resource  # POSIX

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except ImportError:
            pass
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # noqa: BLE001
            pass
        return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._rss_bytes())
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._peak = self._rss_bytes()
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return max(self._peak, self._rss_bytes())


# --------------------------------------------------------------------------- #
# latency / throughput
# --------------------------------------------------------------------------- #


class TimingRecorder:
    def __init__(self) -> None:
        self.carrier: list[float] = []
        self.loader: list[float] = []
        self.transfer: list[float] = []
        self.total: list[float] = []

    def summary(self) -> dict[str, Any]:
        def block(values: Sequence[float]) -> dict[str, Any]:
            if not values:
                return {"count": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
            array = np.asarray(values, dtype=np.float64)
            return {
                "count": int(array.size),
                "mean_ms": float(array.mean()),
                "p50_ms": float(np.percentile(array, 50)),
                "p90_ms": float(np.percentile(array, 90)),
                "p95_ms": float(np.percentile(array, 95)),
                "p99_ms": float(np.percentile(array, 99)),
                "max_ms": float(array.max()),
                "stdev_ms": float(statistics.pstdev(array.tolist())) if array.size > 1 else 0.0,
            }

        return {
            "model_forward": block(self.carrier),
            "host_input_assembly": block(self.loader),
            "host_to_device_transfer": block(self.transfer),
            "end_to_end_per_sample": block(self.total),
            "leg_note": (
                "model_forward excludes host assembly and the H2D copy (fenced by cuda synchronisation "
                "before the window opens). end_to_end_per_sample is assembly + transfer + forward and is "
                "the figure comparable to a deployed single-sample latency claim."
            ),
        }


def _sample_latency(
    dataset: V28TaskDataset,
    *,
    forward: Callable[[Mapping[str, Any]], None],
    device: torch.device,
    max_samples: int | None,
    warmup: int,
    recorder: TimingRecorder,
) -> dict[str, Any]:
    """Per-sample latency with cold start excluded and CUDA sync inside the window.

    The RD cache decode is included on purpose: on an edge board the host-side
    assembly is part of the end-to-end budget, not an external cost. It is
    reported separately as ``host_input_assembly`` so the carrier-only figure
    remains comparable with a batch-1 isolated model benchmark.
    """

    total = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if total <= 0:
        raise ValueError("V28 latency benchmark has no samples")

    def timed(sample_index: int, *, record: bool) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        item = dataset[sample_index]
        batch = collate_v28([item])
        assembled = time.perf_counter()
        inputs = model_input_kwargs(batch, device)
        # The host-to-device copy is asynchronous, so it must be fenced before
        # the forward window opens; otherwise the transfer latency is charged
        # to the model and the carrier figure becomes meaningless.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        carrier_start = time.perf_counter()
        forward(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        if not record:
            return
        recorder.loader.append((assembled - start) * 1000.0)
        recorder.transfer.append((carrier_start - assembled) * 1000.0)
        recorder.carrier.append((end - carrier_start) * 1000.0)
        recorder.total.append((end - start) * 1000.0)

    # Warmup is excluded from the statistics but not from the run, so the first
    # measured sample already pays for lazy CUDA context and cache warm-up.
    for _ in range(max(0, int(warmup))):
        timed(0, record=False)
    for index in range(total):
        timed(index, record=True)
    return {
        "sample_count": int(total),
        "warmup_count": int(max(0, int(warmup))),
        "measured_sample_indices": [0, total - 1] if total > 1 else [0],
        "latency": recorder.summary(),
    }


def _module_attribution(context: Any, model: V28TaskModel, dataset: V28TaskDataset, *, device: torch.device) -> dict[str, Any]:
    """Attribute batch-1 forward time to top-level submodules.

    This exists because parameter count and latency are not correlated here: the
    Liquid V2 wrapper owns ~47% of the parameters but dominates the forward
    budget with a per-event Python loop, so a reader who assumes "few parameters
    means low latency" would draw the wrong conclusion from the size table alone.
    """

    loader = build_dataloader(dataset, batch_size=1, shuffle=False, seed=int(context.config.training["dataloader_seed"]))
    batch = next(iter(loader))
    inputs = model_input_kwargs(batch, device)

    named = [(name, module) for name, module in model.named_children()]
    spans: dict[int, float] = {}
    calls: dict[int, int] = {}

    # No cuda synchronisation happens inside the hooks: with several modules
    # instrumented at once, per-hook syncs fence each other and inflate every
    # window beyond the true value. Kernels are launched asynchronously, so the
    # host-side timestamps measure launch spacing, and the single outer sync in
    # ``timed_forward`` makes the totals comparable.
    def pre_hook(module: Any, _args: Any) -> None:
        module.__dict__["_bench_mark"] = time.perf_counter()

    def post_hook(module: Any, _args: Any, _output: Any) -> None:
        mark = module.__dict__.pop("_bench_mark", None)
        if mark is not None:
            spans[id(module)] = spans.get(id(module), 0.0) + (time.perf_counter() - mark) * 1000.0
            calls[id(module)] = calls.get(id(module), 0) + 1

    repeats = 20

    def timed_forward() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - started) / repeats * 1000.0

    # Baseline measured with NO hooks installed, so hook overhead is visible in
    # the report rather than silently folded into the denominator. Hooks are
    # registered exactly once, after this measurement.
    for _ in range(5):
        with torch.no_grad():
            model(**inputs)

    # Measure the baseline as a distribution, not a single number, so that an
    # external load on the GPU is detected instead of silently published. A
    # competing process preempts kernels and shows up as a heavy right tail on
    # the forward pass, while pure host-side work (input assembly) is far less
    # affected -- so the *ratio* between the two dispersions is the discriminator.
    clean_samples: list[float] = []
    for _ in range(BASELINE_SAMPLE_COUNT):
        clean_samples.append(timed_forward())
    clean_total_ms = float(statistics.median(clean_samples))
    clean_max_ms = float(max(clean_samples))
    dispersions: dict[str, float | None] = {}
    for key, values in (("model_forward", clean_samples),):
        dispersions[key] = _dispersion_ratio(values)

    handles = []
    for _name, module in named:
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    try:
        instrumented_total_ms = timed_forward()
    finally:
        for handle in handles:
            handle.remove()

    raw = {name: spans.get(id(module), 0.0) / repeats for name, module in named}
    call_counts = {name: calls.get(id(module), 0) // max(1, repeats) for name, module in named}
    attributed = sum(raw.values())
    rows = [
        {
            "module": name,
            "exclusive_ms": value,
            "calls_per_forward": call_counts[name],
            "share_of_forward": float(value / clean_total_ms) if clean_total_ms else None,
        }
        for name, value in raw.items()
    ]
    rows.sort(key=lambda row: -row["exclusive_ms"])
    dominant = rows[0] if rows else None
    share = None if dominant is None else dominant["share_of_forward"]
    total_share = sum(row["share_of_forward"] or 0.0 for row in rows)
    # A duplicated hook registration or a competing process both surface here.
    # Either way the table must not be quoted, so the report says so explicitly
    # rather than leaving an impossible percentage for a reader to notice.
    flags: list[str] = []
    if total_share > 1.02:
        flags.append("attributed_share_exceeds_unity")
    if any(row["calls_per_forward"] != 1 for row in rows):
        flags.append("unexpected_call_multiplicity")
    if dominant is not None and (share or 0.0) > 1.0:
        flags.append("dominant_share_exceeds_unity")
    if clean_max_ms > BASELINE_UNSTABLE_RATIO * clean_total_ms:
        flags.append("baseline_forward_time_unstable")
    return {
        "reference_total_ms": float(clean_total_ms),
        "reference_total_is_median_of": len(clean_samples),
        "reference_max_ms": clean_max_ms,
        "reference_dispersion_max_over_median": dispersions.get("model_forward"),
        "instrumented_total_ms": float(instrumented_total_ms),
        "hook_overhead_ms": float(instrumented_total_ms - clean_total_ms),
        "batch_size": 1,
        "repeats": repeats,
        "event_slots": int(inputs["tr_features"].shape[1]),
        "attributed_ms": float(attributed),
        "unattributed_ms": float(clean_total_ms - attributed),
        "attributed_share_total": float(total_share),
        "self_consistency_flags": flags,
        "self_consistency_ok": not flags,
        "self_consistency_note": (
            "These flags are the standard by which this table must be judged. "
            "attributed_share_exceeds_unity means hooks were registered more than once. "
            "baseline_forward_time_unstable means another process was competing for the GPU "
            "while these numbers were taken, and every latency figure in this report should be "
            "discarded and re-measured on an idle device."
        ),
        "timing_note": (
            "reference_total_ms is measured without hooks. Per-module windows are measured with hooks "
            "installed and normalised against that clean baseline; hooks perform no CUDA synchronisation, "
            "because per-hook syncs fence each other and inflate every window. Because kernels launch "
            "asynchronously, a submodule's window measures host-side launch time, which for a "
            "launch-bound loop is the correct attribution of the cost."
        ),
        "tensor_caveat": (
            "The Liquid V2 wrapper iterates events in Python and runs "
            "torch.nonzero per step, so on CUDA it is launch-bound rather than "
            "compute-bound. A low parameter count therefore does NOT imply low "
            "latency on this backend; the ranking must be measured, not inferred."
        ),
        "modules": rows,
        "dominant_module": None if dominant is None else dominant["module"],
        "dominant_share_of_forward": None if dominant is None else dominant["share_of_forward"],
    }


def _input_tensor_bytes(batch: Mapping[str, Any]) -> int:
    """Bytes of every host-side tensor that the carrier will consume for this batch.

    Only tensors are counted.  Metadata (ids, split names, label strings) is not
    payload and would distort a rate figure; it is also negligible here.

    ``rd_frames`` is the only padded member of the batch, so this total is an
    upper bound on the informative input volume for the same reason the FLOPs
    figures are: the tensor is allocated at ``max_sequence_length`` x
    ``max_rd_frames_per_point`` regardless of how many slots are valid.
    """

    total = 0
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.numel()
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            for item in value:
                total += item.element_size() * item.numel()
    return int(total)


def _throughput(
    dataset: V28TaskDataset,
    *,
    forward: Callable[[Mapping[str, Any]], None],
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    warmup_batches: int,
) -> dict[str, Any]:
    """Batched throughput in samples/second, valid RD frames/second and input bytes/second."""

    if max_samples is not None and int(max_samples) < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(int(max_samples))))
    loader = build_dataloader(dataset, batch_size=int(batch_size), shuffle=False, seed=0)
    # Materialize only the capped subset, so the RD cache is decoded once per
    # timed sample rather than once per sample in the whole split.
    batches = list(loader)
    if not batches:
        raise ValueError("V28 throughput benchmark has no batches")

    def run(batch: Mapping[str, Any]) -> None:
        forward(model_input_kwargs(batch, device))

    for index in range(max(0, int(warmup_batches))):
        run(batches[index % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    sampler = RSSSampler()
    sampler.start()
    started = time.perf_counter()
    sample_count = 0
    valid_frames = 0
    slot_count = 0
    padded_input_bytes = 0
    for batch in batches:
        run(batch)
        sample_count += len(batch["parent_sample_ids"])
        mask = batch["rd_frame_mask"]
        slot_count += int(mask.numel())
        valid_frames += int(mask.sum().item())
        # Count the *actual* tensor bytes handed to the carrier, not a nominal
        # per-sample size.  ``rd_frames`` is the only padded payload; the
        # remaining tensors are small fixed-width vectors, but they are still
        # counted so the bandwidth leg is a genuine measurement.
        padded_input_bytes += _input_tensor_bytes(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    peak_rss = sampler.stop()

    bytes_per_second = padded_input_bytes / elapsed
    result: dict[str, Any] = {
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "sample_count": int(sample_count),
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(sample_count / elapsed),
        "rd_frame_accounting": {
            "total_frame_slots": int(slot_count),
            "valid_frame_count": int(valid_frames),
            "skipped_frame_count": int(slot_count - valid_frames),
            "skipped_fraction": float((slot_count - valid_frames) / max(1, slot_count)),
        },
        "valid_rd_frames_per_second": float(valid_frames / elapsed),
        "bandwidth": {
            "padded_input_bytes_per_second": float(bytes_per_second),
            "padded_input_megabytes_per_second_si": float(bytes_per_second / 1_000_000.0),
            "padded_input_mebibytes_per_second": float(bytes_per_second / (1024.0**2)),
            "padded_input_bytes_per_sample": float(padded_input_bytes / max(1, sample_count)),
            "definition": (
                "Sum of the element_size*numel of every host tensor in the collated batch that is "
                "handed to the carrier, divided by the wall-clock throughput window. This includes "
                "the padded rd_frames tensor (max_sequence_length x max_rd_frames slots, of which "
                "only a fraction are valid) and every fixed-width per-event feature vector."
            ),
            "not_a_device_rate": (
                "This is a host-side INPUT CONSUMPTION rate, not DRAM or interconnect bandwidth. "
                "It is NOT comparable with figures quoted as radar sensor output rate, cache read "
                "bandwidth, or H2D transfer bandwidth (the measured H2D leg is ~2.2 ms/sample and "
                "is reported separately under latency)."
            ),
            "context_note": (
                "Because the rd_frames tensor is fully padded, this number overstates the sensor "
                "data rate actually carrying information by roughly 1/utilisation. Quote it only "
                "with samples_per_second and the utilisation ratio alongside."
            ),
        },
        "memory": {"cpu_peak_rss_bytes": int(peak_rss)},
    }
    if device.type == "cuda":
        result["memory"].update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
        result["memory"]["cuda_peak_allocated_definition"] = (
            "torch.cuda.max_memory_allocated over the benchmark window only; "
            "includes activations and workspace, excludes other processes."
        )
    return result


# --------------------------------------------------------------------------- #
# accuracy scopes
# --------------------------------------------------------------------------- #


def _known_only_scope(
    context: Any,
    models: Mapping[str, V28TaskModel],
    datasets: Mapping[str, V28TaskDataset],
    device: torch.device,
    *,
    split: str,
) -> dict[str, Any]:
    """Closed-set conditional metrics, one task at a time, on their own dataset."""

    scope: dict[str, Any] = {"scope": "known_only_conditional", "split": split, "scope_note": "Known classes only. No Unknown samples, no abstention. Comparable to historical closed-set Micro F1 figures."}
    for task in V28_TASKS:
        model = models[task]
        dataset = datasets[task]
        records = _task_records(model, dataset, device=device, config=context.config)
        scope[task] = {
            "parent_count": len(records),
            "class_names": list(dataset.class_names),
            "metrics": confusion_metrics(records, class_names=dataset.class_names, scope=f"known_only_conditional|{task}|{split}"),
        }
    return scope


def _task_records(
    model: V28TaskModel,
    dataset: V28TaskDataset,
    *,
    device: torch.device,
    config: Any,
    latencies: TimingRecorder | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    loader = build_dataloader(dataset, batch_size=int(config.training["batch_size"]), shuffle=False, seed=int(config.training["dataloader_seed"]))
    with torch.inference_mode():
        for batch in loader:
            output = model(**model_input_kwargs(batch, device))
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("V28 benchmark probabilities are non-finite")
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][row])
                final_probability = probabilities[row, length - 1]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "label_name": str(batch["task_label_names"][row]),
                        "true_id": int(batch["labels"][row]),
                        "pred_id": int(final_probability.argmax()),
                        "event_count": length,
                    }
                )
    if latencies is not None:
        latencies.carrier.append(0.0)
    return records


def _end_to_end_scope(
    context: Any,
    models: Mapping[str, V28TaskModel],
    heads: Mapping[str, V28TrajectoryHeads],
    registry: Mapping[str, Any],
    device: torch.device,
    *,
    audit_mode: bool,
) -> dict[str, Any]:
    """Full L1 -> L2 -> L3 chain over the frozen replay pool."""

    calibration = registry["calibration"]
    l1_task = calibration["tasks"][V28_L1_TASK]
    l2_task = calibration["tasks"][V28_L2_TASK]
    l1_names = list(l1_task["class_names"])
    l1_floors = l1_task["conformal"].get("candidate_probability_floor_by_class", {})
    l2_floors = l2_task["conformal"].get("candidate_probability_floor_by_class", {})
    l1_threshold = float(l1_task["knownness"].get("selected", {}).get("threshold", 1.0))
    l2_threshold = float(l2_task["knownness"].get("selected", {}).get("threshold", 1.0))
    l1_temperature = float(l1_task["temperature"].get("temperature", 1.0))
    l2_temperature = float(l2_task["temperature"].get("temperature", 1.0))
    l2_names = list(V28_L2_CLASSES)

    l1_dataset = V28ReplayDataset(context, task=V28_L1_TASK)
    l2_dataset = V28ReplayDataset(context, task=V28_L2_TASK)
    batch_size = int(context.config.training["batch_size"])
    l1_loader = build_dataloader(l1_dataset, batch_size=batch_size, shuffle=False, seed=int(context.config.training["dataloader_seed"]))
    l2_loader = build_dataloader(l2_dataset, batch_size=batch_size, shuffle=False, seed=int(context.config.training["dataloader_seed"]))

    for model in models.values():
        model.eval()
    for head in heads.values():
        head.eval()

    records: list[dict[str, Any]] = []
    route_counts = {"l1_clutter": 0, "l1_unknown": 0, "l2_evaluated": 0}
    with torch.inference_mode():
        for l1_batch, l2_batch in zip(l1_loader, l2_loader, strict=True):
            if list(l1_batch["parent_sample_ids"]) != list(l2_batch["parent_sample_ids"]):
                raise ValueError("V28 benchmark L1/L2 parent order differs")
            l1_output = models[V28_L1_TASK](**model_input_kwargs(l1_batch, device))
            l2_output = models[V28_L2_TASK](**model_input_kwargs(l2_batch, device))
            l1_aux = heads[V28_L1_TASK](l1_output)
            l2_aux = heads[V28_L2_TASK](l2_output)
            l1_lengths = l1_output["lengths"].long()
            l2_lengths = l2_output["lengths"].long()
            rows = torch.arange(l1_lengths.shape[0], device=device)
            l1_quality = deterministic_quality_target(
                context, task=V28_L1_TASK, task_facts=l1_output["task_facts_sequence"], event_facts=l1_output["event_facts_sequence"], padding_mask=l1_output["padding_mask"]
            )[rows, l1_lengths - 1]
            l2_quality = deterministic_quality_target(
                context, task=V28_L2_TASK, task_facts=l2_output["task_facts_sequence"], event_facts=l2_output["event_facts_sequence"], padding_mask=l2_output["padding_mask"]
            )[rows, l2_lengths - 1]
            l1_probs = _probabilities(l1_output["final_logits"].cpu().numpy(), l1_temperature)
            l2_probs = _probabilities(l2_output["final_logits"].cpu().numpy(), l2_temperature)
            l1_scores = torch.sigmoid(l1_aux["knownness_logits"][rows, l1_lengths - 1]).cpu().numpy()
            l2_scores = torch.sigmoid(l2_aux["knownness_logits"][rows, l2_lengths - 1]).cpu().numpy()
            for index, parent_id in enumerate(l1_batch["parent_sample_ids"]):
                l1_candidates = _candidate_set(l1_probs[index], l1_names, l1_floors)
                l1_prediction = l1_names[int(l1_probs[index].argmax())] if l1_probs.shape[1] else None
                l1_target = l1_prediction == "KNOWN_TARGET_EVIDENCE"
                l1_clutter = l1_prediction == "KNOWN_CLUTTER_EVIDENCE"
                l2_candidates = _candidate_set(l2_probs[index], l2_names, l2_floors)
                l2_prediction = l2_names[int(l2_probs[index].argmax())] if l2_probs.shape[1] else None
                if l1_clutter:
                    route_counts["l1_clutter"] += 1
                elif not l1_target:
                    route_counts["l1_unknown"] += 1
                else:
                    route_counts["l2_evaluated"] += 1
                decision = resolve_action_verified(
                    registry=registry,
                    l1_candidate_set=l1_candidates,
                    l1_prediction=l1_prediction,
                    l1_knownness=bool(l1_scores[index] >= l1_threshold),
                    l1_quality_eligible=bool(l1_quality[index]),
                    l1_target=l1_target,
                    l1_clutter=l1_clutter,
                    l2_candidate_set=l2_candidates,
                    l2_prediction=l2_prediction,
                    l2_knownness=bool(l2_scores[index] >= l2_threshold),
                    l2_quality_eligible=bool(l2_quality[index]),
                    streaming=False,
                    audit_mode=bool(audit_mode),
                )
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "label_name": str(l1_batch["base_label_names"][index]),
                        "action": decision.action,
                        "reason_code": decision.reason_code,
                        "l1_prediction": l1_prediction,
                        "l2_prediction": l2_prediction,
                        "l1_candidates": l1_candidates,
                        "l2_candidates": l2_candidates,
                        "l1_knownness": float(l1_scores[index]),
                        "l2_knownness": float(l2_scores[index]),
                        "l1_quality_eligible": bool(l1_quality[index]),
                        "l2_quality_eligible": bool(l2_quality[index]),
                        "event_count": int(l1_lengths[index].item()),
                    }
                )

    metrics = _action_metrics(records)
    accepted = [record for record in records if record["action"] != "PENDING_ABSTAIN"]
    return {
        "scope": "end_to_end_audit" if audit_mode else "end_to_end_fail_closed",
        "registry_status": registry.get("status"),
        "registry_deployment_eligible": bool(registry.get("deployment_eligible")),
        "audit_mode": bool(audit_mode),
        "parent_count": len(records),
        "route_counts": route_counts,
        "action_metrics": metrics,
        "accepted_only_closed_set": _accepted_closed_set(accepted),
        "scope_note": (
            "Full open-set chain including UnknownTarget, ClutterTarget and abstention. "
            "Per-class F1 here is conditional on the accepted subset and is deliberately "
            "lower than known_only_conditional."
        ),
        "records": records,
    }


def _accepted_closed_set(accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Closed-set metrics computed only over non-abstained samples."""

    names = ("DroneTarget", "BirdTarget", "BalloonTarget", "Clutter")
    pairs = []
    for record in accepted:
        true_name = str(record["label_name"])
        action = str(record["action"])
        if true_name == "ClutterTarget" and action == "Clutter":
            pairs.append(("Clutter", "Clutter"))
        elif true_name in V28_L2_CLASSES and action in V28_L2_CLASSES:
            pairs.append((true_name, action))
    index = {name: position for position, name in enumerate(names)}
    wanted = [pair for pair in pairs if pair[0] in index and pair[1] in index]
    if not wanted:
        return {"support": 0, "metrics": None, "note": "no accepted sample fell inside the known closed set"}
    return {
        "support": len(wanted),
        "metrics": confusion_metrics(
            [{"true_id": index[t], "pred_id": index[p]} for t, p in wanted],
            class_names=names,
            scope="accepted_only_closed_set",
        ),
        "note": "Only known-class samples that the L3 resolver accepted. Excludes abstention and Unknown rejection, so this is a conditional accuracy, not coverage.",
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def _load_registry_tolerant(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the L3 registry, degrading gracefully when its binding is stale.

    ``verify_registry`` binds the registry to ``context.manifest_sha256``.  When
    the manifest has been regenerated after the registry was written, that
    identity check fails even though every referenced checkpoint is still
    present and hash-valid.  A benchmark must still be able to report model
    size, FLOPs, latency, throughput, memory and closed-set accuracy in that
    situation, so the raw payload is used for path/seed lookup and the
    integrity failure is recorded verbatim rather than raised.
    """

    path = artifact_path(context.project_root, str(context.config.artifacts["registry_path"]))
    payload = read_sealed_json(path)
    integrity: dict[str, Any] = {
        "path": relative_project_path(context.project_root, path),
        "file_sha256": sha256_file(path),
        "declared_registry_sha256": payload.get("registry_sha256"),
        "declared_system_manifest_sha256": payload.get("system_manifest_sha256"),
        "context_manifest_sha256": context.manifest_sha256,
        "verified": False,
        "verification_error": None,
    }
    try:
        verified = verify_registry(context, payload)
    except Exception as exc:  # noqa: BLE001 - integrity failure must not abort measurement
        integrity["verification_error"] = f"{type(exc).__name__}: {exc}"
        integrity["note"] = (
            "Registry identity does not match the current manifest; the registry was written against a "
            "different manifest revision. Endpoint and head bindings below are read from the raw "
            "payload. End-to-end L3 action metrics are NOT reproducible under this condition."
        )
        LOGGER.warning("L3 registry failed verification: %s", integrity["verification_error"])
        return payload, integrity
    integrity["verified"] = True
    return verified, integrity


def run_benchmark(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    device_name: str = "cuda",
    split: str = "validation",
    batch_size: int | None = None,
    max_latency_samples: int | None = 128,
    warmup: int = 5,
    include_flops: bool = True,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V28 benchmark was asked for cuda but torch.cuda.is_available() is False")
    device = torch.device(device_name)
    context = load_context(project_root, config_path)
    effective_batch = int(batch_size if batch_size is not None else context.config.training["batch_size"])
    dtype = "float32"

    LOGGER.info("loading V28 endpoint bindings")
    registry, registry_integrity = _load_registry_tolerant(context)
    l1_binding = registry["endpoint_binding"][V28_L1_TASK]
    l2_binding = registry["endpoint_binding"][V28_L2_TASK]
    l1_head_binding = registry["head_bindings"][V28_L1_TASK]
    l2_head_binding = registry["head_bindings"][V28_L2_TASK]

    models = {V28_L1_TASK: V28TaskModel(context, task=V28_L1_TASK).to(device), V28_L2_TASK: V28TaskModel(context, task=V28_L2_TASK).to(device)}
    payloads = {
        V28_L1_TASK: load_checkpoint(context, models[V28_L1_TASK], l1_binding["path"], task=V28_L1_TASK, seed=int(l1_binding["seed"]), map_location=device),
        V28_L2_TASK: load_checkpoint(context, models[V28_L2_TASK], l2_binding["path"], task=V28_L2_TASK, seed=int(l2_binding["seed"]), map_location=device),
    }
    heads = {V28_L1_TASK: V28TrajectoryHeads(context, task=V28_L1_TASK).to(device), V28_L2_TASK: V28TrajectoryHeads(context, task=V28_L2_TASK).to(device)}

    heads[V28_L1_TASK].load_state_dict(torch.load(resolve_project_path(context.project_root, l1_head_binding["path"], must_exist=True), map_location=device, weights_only=False)["state_dict"], strict=True)
    heads[V28_L2_TASK].load_state_dict(torch.load(resolve_project_path(context.project_root, l2_head_binding["path"], must_exist=True), map_location=device, weights_only=False)["state_dict"], strict=True)
    for model in models.values():
        model.eval()
    for head in heads.values():
        head.eval()

    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA,
        "experiment_id": "V28",
        "status": "completed",
        "manifest_sha256": context.manifest_sha256,
        "config_sha256": context.config.sha256,
        "registry_sha256": registry.get("registry_sha256"),
        "registry_status": registry.get("status"),
        "registry_deployment_eligible": bool(registry.get("deployment_eligible")),
        "registry_integrity": registry_integrity,
        "checkpoint_epochs": {task: int(payloads[task]["epoch"]) for task in V28_TASKS},
        "checkpoint_paths": {task: str(binding["path"]) for task, binding in ((V28_L1_TASK, l1_binding), (V28_L2_TASK, l2_binding))},
        "measurement_conditions": measurement_conditions(device, batch_size=effective_batch, dtype=dtype),
    }

    LOGGER.info("accounting parameters and artifact sizes")
    report["model_size"] = {
        V28_L1_TASK: parameter_accounting(models[V28_L1_TASK], resolve_project_path(context.project_root, l1_binding["path"], must_exist=True)),
        V28_L2_TASK: parameter_accounting(models[V28_L2_TASK], resolve_project_path(context.project_root, l2_binding["path"], must_exist=True)),
        "L3_decision_layer": {"parameter_count": 0, "note": "l3.py / final_resolver.py / registry policy contain no trainable nn.Module parameters."},
    }
    report["model_size"]["serial_L1_plus_L2"] = {
        "parameter_count": int(report["model_size"][V28_L1_TASK]["parameter_count"] + report["model_size"][V28_L2_TASK]["parameter_count"]),
        "checkpoint_file_bytes": int(report["model_size"][V28_L1_TASK]["checkpoint_file_bytes"] + report["model_size"][V28_L2_TASK]["checkpoint_file_bytes"]),
        "l3_included": False,
        "l3_tunable_parameters": 0,
    }

    # Dataset construction loads the full index and materializes the V28 caches,
    # so each (task, split) pair is built exactly once and reused by the
    # compute, latency, throughput and accuracy stages.
    LOGGER.info("constructing benchmark datasets (one per task)")
    datasets: dict[str, V28TaskDataset] = {}
    for task in V28_TASKS:
        datasets[task] = V28TaskDataset(context, task=task, split=split)
        LOGGER.info("  %s: %d parents", task, len(datasets[task]))

    if include_flops:
        LOGGER.info("counting FLOPs on a representative batch")
        report["compute"] = _compute_block(context, models, datasets, device=device, batch_size=effective_batch)

    LOGGER.info("measuring per-sample latency")
    report["latency"] = {}
    for task in V28_TASKS:
        recorder = TimingRecorder()

        def forward(inputs: Mapping[str, Any], model: V28TaskModel = models[task]) -> None:
            model(**inputs)

        summary = _sample_latency(datasets[task], forward=forward, device=device, max_samples=max_latency_samples, warmup=warmup, recorder=recorder)
        summary["scope"] = f"{task}|{split}|batch=1|per_sample"
        report["latency"][task] = summary

    LOGGER.info("measuring per-module forward attribution")
    report["module_attribution"] = {}
    for task in V28_TASKS:
        report["module_attribution"][task] = _module_attribution(context, models[task], datasets[task], device=device)

    LOGGER.info("measuring batched throughput")
    report["throughput"] = {}
    for task in V28_TASKS:
        report["throughput"][task] = _throughput(
            datasets[task],
            forward=lambda inputs, model=models[task]: model(**inputs),
            device=device,
            batch_size=effective_batch,
            max_samples=max_latency_samples,
            warmup_batches=2,
        )
        report["throughput"][task]["scope"] = f"{task}|{split}|batch={effective_batch}"

    LOGGER.info("computing known-only conditional accuracy")
    report["accuracy"] = {
        "known_only_conditional": _known_only_scope(context, models, datasets, device, split=split),
    }
    report["accuracy"]["scope_order"] = [
        "known_only_conditional",
        "end_to_end_audit",
        "end_to_end_fail_closed",
    ]
    report["accuracy"]["reading_guide"] = {
        "known_only_conditional": "Closed-set per-task metrics on known classes. No Unknown, no abstention. This is the scope of historical 'closed-set Micro F1' claims.",
        "end_to_end_audit": "L1->L2->L3 with Unknown, Clutter and abstention present. This is the only scope that bounds field behaviour.",
        "end_to_end_fail_closed": "Production fail-closed path. Abstains on every sample when the registry is not deployment eligible.",
        "caution": "Do not quote the conditional scope as if it were end-to-end performance, and do not quote FLOPs without its definition field.",
    }

    if registry_integrity["verified"]:
        LOGGER.info("computing end-to-end open-set accuracy (audit mode)")
        report["accuracy"]["end_to_end_audit"] = _end_to_end_scope(context, models, heads, registry, device, audit_mode=True)
        LOGGER.info("computing end-to-end fail-closed behaviour")
        report["accuracy"]["end_to_end_fail_closed"] = _end_to_end_scope(context, models, heads, registry, device, audit_mode=False)
    else:
        reason = (
            "L3 registry failed identity verification against the current manifest, so registry-gated "
            "end-to-end actions are not reproducible. Every end-to-end call would return "
            "PENDING_ABSTAIN/REGISTRY_INVALID_OR_UNAVAILABLE, which carries no information about model "
            "quality. Rebuild the L3 registry against the current manifest to restore this scope."
        )
        report["accuracy"]["end_to_end_audit"] = {"scope": "end_to_end_audit", "status": "unavailable", "reason": reason, "registry_integrity_error": registry_integrity["verification_error"]}
        report["accuracy"]["end_to_end_fail_closed"] = {"scope": "end_to_end_fail_closed", "status": "unavailable", "reason": reason, "registry_integrity_error": registry_integrity["verification_error"]}
        LOGGER.warning("skipping end-to-end scopes: %s", reason)

    report["serial_chain"] = _serial_chain(report)
    return report


def _serial_chain(report: Mapping[str, Any]) -> dict[str, Any]:
    """L1 -> L2 aggregates, for the deployment claim that a sample traverses both.

    Every quantity here is *derived* from the per-task measurements by the stated
    rule, never measured directly.  The arithmetic differs by quantity and the
    rule is recorded next to each, because naive summation is wrong for rate
    quantities:

    - parameters and checkpoint bytes **add**
    - compute adds (two independent forwards, each at the configured batch size)
    - latency adds (the same sample is processed by both stages in sequence)
    - throughput and bandwidth **do not add**: they are reciprocals of the sum of
      the per-stage per-sample times

    L3 contributes no trainable parameters and no measurable arithmetic; it is a
    JSON-policy lookup over the two stages' outputs.  Its cost is host-side code
    and is not covered by these numbers.
    """

    l1, l2 = V28_L1_TASK, V28_L2_TASK
    size = report.get("model_size", {})
    compute = report.get("compute", {})
    latency = report.get("latency", {})
    throughput = report.get("throughput", {})
    conditions = report.get("measurement_conditions", {})
    batch_size = int(conditions.get("batch_size") or 1)

    if l1 not in size or l2 not in size:
        return {"status": "unavailable", "reason": "per-task size blocks missing"}

    result: dict[str, Any] = {
        "scope": f"{l1} -> {l2} serial, one sample traverses both stages",
        "derivation": "All figures are derived from the per-task blocks by the rule stated per field; none are measured directly.",
    }

    result["parameters"] = {
        "rule": "sum of the two stages' parameter_count",
        "l1_parameter_count": int(size[l1]["parameter_count"]),
        "l2_parameter_count": int(size[l2]["parameter_count"]),
        "total_parameter_count": int(size[l1]["parameter_count"] + size[l2]["parameter_count"]),
        "l3_trainable_parameters": 0,
        "parameter_bytes_fp32": int(size[l1].get("parameter_bytes", 0) + size[l2].get("parameter_bytes", 0)),
        "checkpoint_file_bytes": int(size[l1]["checkpoint_file_bytes"] + size[l2]["checkpoint_file_bytes"]),
    }

    if l1 in compute and l2 in compute:
        narrow = int(compute[l1]["narrow_scope"]["flops"]) + int(compute[l2]["narrow_scope"]["flops"])
        result["compute"] = {
            "rule": "sum of the two stages' FLOPs, each counted at the configured batch size",
            "batch_size": batch_size,
            "flops_narrow_at_batch": narrow,
            "flops_narrow_per_sample": float(narrow / max(1, batch_size)),
            "padding_caveat": compute[l1]["padding_caveat"],
            "definition": compute[l1]["narrow_scope"]["definition"],
        }

    if l1 in latency and l2 in latency:
        legs: dict[str, Any] = {"rule": "sum of the two stages' per-sample legs"}
        for leg in ("host_input_assembly", "host_to_device_transfer", "model_forward", "end_to_end_per_sample"):
            one, two = latency[l1]["latency"].get(leg), latency[l2]["latency"].get(leg)
            if not one or not two:
                continue
            legs[leg] = {
                "p50_ms": float(one["p50_ms"] + two["p50_ms"]),
                "p95_ms": float(one["p95_ms"] + two["p95_ms"]),
                "mean_ms": float(one["mean_ms"] + two["mean_ms"]),
            }
        result["latency"] = legs
        result["latency"]["note"] = (
            "end_to_end_per_sample is the figure comparable to a deployed single-sample latency claim. "
            "It is the sum of the two stages' end-to-end legs and therefore already includes both stages' "
            "host assembly and both H2D copies."
        )

    if l1 in throughput and l2 in throughput:
        s1 = float(throughput[l1]["samples_per_second"])
        s2 = float(throughput[l2]["samples_per_second"])
        serial_rate = 1.0 / ((1.0 / s1) + (1.0 / s2)) if s1 > 0 and s2 > 0 else 0.0
        b1 = float(throughput[l1]["bandwidth"]["padded_input_bytes_per_second"])
        b2 = float(throughput[l2]["bandwidth"]["padded_input_bytes_per_second"])
        serial_bw = 1.0 / ((1.0 / b1) + (1.0 / b2)) if b1 > 0 and b2 > 0 else 0.0
        result["throughput"] = {
            "rule": "harmonic sum: rate_serial = 1 / (1/rate_L1 + 1/rate_L2). Rates are NOT additive.",
            "l1_samples_per_second": s1,
            "l2_samples_per_second": s2,
            "samples_per_second": serial_rate,
            "l1_padded_input_megabytes_per_second_si": b1 / 1_000_000.0,
            "l2_padded_input_megabytes_per_second_si": b2 / 1_000_000.0,
            "padded_input_megabytes_per_second_si": serial_bw / 1_000_000.0,
            "bandwidth_note": throughput[l1]["bandwidth"]["not_a_device_rate"],
        }

    result["l3_layer"] = {
        "trainable_parameters": 0,
        "arithmetic": "not measured; the resolver is a policy lookup over both stages' outputs",
        "note": "L3 adds no parameters and no tensor arithmetic, so it is absent from every figure above.",
    }
    return result


def _compute_block(context: Any, models: Mapping[str, V28TaskModel], datasets: Mapping[str, V28TaskDataset], *, device: torch.device, batch_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in V28_TASKS:
        loader = build_dataloader(datasets[task], batch_size=int(batch_size), shuffle=False, seed=int(context.config.training["dataloader_seed"]))
        batch = next(iter(loader))
        inputs = model_input_kwargs(batch, device)
        model = models[task]

        def forward(model: V28TaskModel = model, inputs: Mapping[str, Any] = inputs) -> Any:
            return model(**inputs)

        narrow = _narrow_flops(model, forward)
        wide = _torch_flops(forward)
        slots = int(batch["rd_frame_mask"].numel())
        valid = int(batch["rd_frame_mask"].sum().item())
        result[task] = {
            "batch_size": int(batch_size),
            "event_count_max": int(batch["tr_features"].shape[1]),
            "rd_frame_slots": slots,
            "rd_frame_valid": valid,
            "rd_frame_utilization": float(valid / slots) if slots else None,
            "narrow_scope": narrow,
            "torch_scope": wide,
            "flops_per_sample_narrow": float(narrow["flops"] / max(1, len(batch["parent_sample_ids"]))),
            "flops_per_sample_torch": None if wide.get("flops") is None else float(wide["flops"] / max(1, len(batch["parent_sample_ids"]))),
            "macs_per_sample_narrow": float(narrow["macs"] / max(1, len(batch["parent_sample_ids"]))),
            "padding_caveat": (
                "Both counters measure the full padded tensor width (max_sequence_length x max_rd_frames), "
                "not the valid content. The RD encoder is a dense convolution over every frame slot, so its "
                f"cost is charged for all {slots} slots even though only {valid} are valid "
                f"({100.0 * valid / slots:.1f}% utilisation). Per-sample figures are therefore an upper bound "
                "for this batch's padding profile, and are not comparable with a hand-computed count that "
                "assumes only valid frames."
            ),
        }
    result["reading_guide"] = (
        "Two estimates are reported because they are not interchangeable. "
        "'narrow_scope' counts matrix multiplies only and is comparable to the historical "
        "V27 deployment_benchmark. 'torch_scope' additionally counts element-wise and "
        "normalization arithmetic. Neither models memory bandwidth, which dominates on edge NPUs. "
        "Both are computed on padded tensors; see padding_caveat in each task block."
    )
    serial_narrow = int(result[V28_L1_TASK]["narrow_scope"]["flops"] + result[V28_L2_TASK]["narrow_scope"]["flops"])
    result["serial_L1_plus_L2"] = {
        "narrow_scope_flops_for_one_batch": serial_narrow,
        "narrow_scope_flops_for_one_batch_definition": FLOPS_DEFINITION_NARROW,
        "logical_serial_note": "Upper bound only: L1 short-circuits to Clutter or abstain for a large fraction of samples, so the average field cost is lower.",
    }
    return result


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _default_output(project_root: Path) -> Path:
    return project_root / "src-v3" / "v28" / "reports" / "V28-deployment-benchmark-v1.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V28 deployment benchmark (params / FLOPs / latency / throughput / memory / metrics)")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="cap samples for latency/throughput (default: all)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-flops", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=str(arguments.log_level).upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")

    root = Path(arguments.project_root).resolve()
    report = run_benchmark(
        root,
        config_path=arguments.config,
        device_name=arguments.device,
        split=arguments.split,
        batch_size=arguments.batch_size,
        max_latency_samples=arguments.max_samples,
        warmup=arguments.warmup,
        include_flops=not arguments.no_flops,
    )
    output = Path(arguments.output) if arguments.output else _default_output(root)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Every run also drops a timestamped copy. Latency on a shared desktop GPU
    # varies enough between runs that a single report cannot be audited for
    # stability; keeping the history makes the spread visible instead of
    # invisible.
    report["run_history"] = _run_history(output.parent, report)
    atomic_write_text(output, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False))
    _archive_run(output, report)
    digest = sha256_file(output)

    _print_summary(report)
    LOGGER.info("wrote %s (sha256=%s)", relative_project_path(root, output), digest)
    return 0


def _run_history(reports_dir: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise prior runs in the same directory, to expose run-to-run spread."""

    previous: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("V28-deployment-benchmark-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = _run_signature(payload, label=path.name)
        if entry:
            previous.append(entry)
    this = _run_signature(current, label="current")
    if this:
        previous.append(this)
    if len(previous) < 2:
        return {"run_count": len(previous), "note": "Not enough prior runs to estimate spread."}

    metrics: dict[str, Any] = {}
    for field in ("l1_forward_p50_ms", "l2_forward_p50_ms", "l1_samples_per_second", "l2_samples_per_second", "l1_e2e_p95_ms", "l2_e2e_p95_ms"):
        values = [entry[field] for entry in previous if isinstance(entry.get(field), (int, float))]
        if len(values) < 2:
            continue
        low, high = min(values), max(values)
        metrics[field] = {
            "min": float(low),
            "max": float(high),
            "spread_ratio": float(high / low) if low else None,
        }
    unstable = [name for name, stat in metrics.items() if (stat["spread_ratio"] or 0.0) > 1.5]
    return {
        "run_count": len(previous),
        "runs": previous,
        "per_metric_spread": metrics,
        "unstable_metrics": unstable,
        "note": (
            "run-to-run spread across reports in this directory. Any metric whose max/min exceeds "
            "1.5 across runs is flagged unstable and should not be quoted to more than one significant "
            "figure, because the device is shared and other processes perturb CUDA scheduling."
        ),
    }


def _run_signature(report: Mapping[str, Any], *, label: str) -> dict[str, Any] | None:
    latency = report.get("latency") or {}
    throughput = report.get("throughput") or {}
    if "l1_known_gate" not in latency or "l1_known_gate" not in throughput:
        return None
    conditions = report.get("measurement_conditions") or {}
    entry: dict[str, Any] = {
        "label": label,
        "device": conditions.get("device"),
        "gpu": (conditions.get("cuda") or {}).get("device_name"),
        "batch_size": conditions.get("batch_size"),
    }
    for task, prefix in (("l1_known_gate", "l1"), ("l2_dbb_specialist", "l2")):
        legs = latency.get(task, {}).get("latency") or {}
        if "model_forward" in legs:
            entry[f"{prefix}_forward_p50_ms"] = float(legs["model_forward"]["p50_ms"])
        if "end_to_end_per_sample" in legs:
            entry[f"{prefix}_e2e_p95_ms"] = float(legs["end_to_end_per_sample"]["p95_ms"])
        rates = throughput.get(task) or {}
        if "samples_per_second" in rates:
            entry[f"{prefix}_samples_per_second"] = float(rates["samples_per_second"])
    attribution = (report.get("module_attribution") or {}).get("l1_known_gate") or {}
    if attribution:
        entry["l1_attribution_self_consistent"] = bool(attribution.get("self_consistency_ok", False))
        entry["l1_attribution_flags"] = list(attribution.get("self_consistency_flags") or [])
    return entry


def _archive_run(output: Path, report: Mapping[str, Any]) -> None:
    """Persist a timestamped copy so consecutive runs can be compared."""

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    archive = output.with_name(f"{output.stem}-{stamp}.json")
    try:
        atomic_write_text(archive, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False))
    except OSError as exc:  # archiving must never fail the benchmark
        LOGGER.warning("could not archive run to %s: %s", archive, exc)


def _print_summary(report: Mapping[str, Any]) -> None:
    print("=" * 78)
    print("V28 deployment benchmark")
    print("=" * 78)
    conditions = report["measurement_conditions"]
    print(f"device={conditions['device']}  batch={conditions['batch_size']}  dtype={conditions['parameter_dtype']}")
    if "cuda" in conditions:
        print(f"gpu={conditions['cuda']['device_name']}  torch_threads={conditions['threads']['torch_num_threads']}")
    print(f"registry_status={report['registry_status']}  deployment_eligible={report['registry_deployment_eligible']}")
    integrity = report.get("registry_integrity") or {}
    if integrity and not integrity.get("verified"):
        print(f"  !! registry integrity FAILED: {integrity.get('verification_error')}")
        print("  !! end-to-end L3 scopes are reported as unavailable")
    size = report["model_size"]
    for task in V28_TASKS:
        entry = size[task]
        print(f"  [size] {task:<20} params={entry['parameter_count']:>8,}  ckpt={entry['checkpoint_file_bytes']:>9,} B")
    serial = size["serial_L1_plus_L2"]
    print(f"  [size] {'serial L1+L2':<20} params={serial['parameter_count']:>8,}  L3 params={serial['l3_tunable_parameters']}")
    if "compute" in report:
        for task in V28_TASKS:
            entry = report["compute"][task]
            torch_flops = entry["torch_scope"].get("flops")
            print(
                f"  [flops] {task:<20} narrow={entry['narrow_scope']['flops']:>12,} "
                f"torch={torch_flops if torch_flops is not None else 'n/a'} "
                f"(torch_status={entry['torch_scope']['status']})"
            )
    for task in V28_TASKS:
        lat = report["latency"][task]["latency"]
        fwd = lat["model_forward"]
        asm = lat["host_input_assembly"]
        xfer = lat.get("host_to_device_transfer", {})
        e2e = lat["end_to_end_per_sample"]
        print(
            f"  [latency] {task:<20} forward_p50={fwd['p50_ms']:.3f} ms  "
            f"assembly_p50={asm['p50_ms']:.3f} ms  h2d_p50={xfer.get('p50_ms') if xfer.get('p50_ms') is None else round(xfer['p50_ms'], 3)} ms  "
            f"e2e_p95={e2e['p95_ms']:.3f} ms"
        )
    for task in V28_TASKS:
        attribution = (report.get("module_attribution") or {}).get(task)
        if not attribution:
            continue
        dominant = attribution["dominant_module"]
        share = attribution["dominant_share_of_forward"]
        print(f"  [attribution] {task:<16} forward={attribution['reference_total_ms']:.2f} ms  dominant={dominant} ({100.0 * share:.1f}%)")
        for row in attribution["modules"][:4]:
            print(f"        {row['module']:<22} {row['exclusive_ms']:>8.3f} ms  {100.0 * row['share_of_forward']:>5.1f}%")
        if not attribution.get("self_consistency_ok", True):
            print(f"        !! SELF-CONSISTENCY FAILED: {', '.join(attribution['self_consistency_flags'])}")
    for task in V28_TASKS:
        throughput = report["throughput"][task]
        memory = throughput["memory"]
        bw = throughput.get("bandwidth") or {}
        print(
            f"  [throughput] {task:<20} {throughput['samples_per_second']:.1f} samples/s  "
            f"{bw.get('padded_input_megabytes_per_second_si', float('nan')):.1f} MB/s (padded input)  "
            f"peak_cuda_alloc={memory.get('cuda_peak_allocated_bytes')}"
        )
    accuracy = report["accuracy"]
    print("-" * 78)
    print("accuracy scopes (NOT interchangeable)")
    print("-" * 78)
    for task in V28_TASKS:
        metrics = accuracy["known_only_conditional"][task]["metrics"]
        print(
            f"  [known_only_conditional] {task:<20} macro_f1={metrics['macro_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f} support={metrics['support']}"
        )
    for key in ("end_to_end_audit", "end_to_end_fail_closed"):
        entry = accuracy[key]
        if entry.get("status") == "unavailable":
            print(f"  [{key}] UNAVAILABLE - {entry['registry_integrity_error']}")
            continue
        metrics = entry["action_metrics"]
        print(f"  [{key}] pending_rate={metrics['pending_rate']['estimate']:.4f} coverage={metrics['coverage']['estimate']:.4f}")
        accepted = entry.get("accepted_only_closed_set") or {}
        if accepted.get("metrics"):
            print(f"      accepted-only macro_f1={accepted['metrics']['macro_f1']:.4f} support={accepted['support']}")
        for row in metrics["actions"]:
            print(f"      {row['action']:<18} precision={row['precision']}  recall={row['recall']}  f1={row['f1']}  n={row['predicted_support']}")
    serial = report.get("serial_chain") or {}
    if serial.get("status") != "unavailable":
        print("-" * 78)
        print("serial chain L1 -> L2 (derived, not directly measured)")
        print("-" * 78)
        params = serial.get("parameters") or {}
        if params:
            print(
                f"  [total params] {params['total_parameter_count']:,}  "
                f"(L1 {params['l1_parameter_count']:,} + L2 {params['l2_parameter_count']:,}; L3 trainable 0)"
            )
            print(f"  [pt files]     {params['checkpoint_file_bytes']:,} B combined")
        comp = serial.get("compute") or {}
        if comp:
            print(
                f"  [compute]      {comp['flops_narrow_per_sample']/1e9:.4f} GFLOPs/sample  "
                f"({comp['flops_narrow_at_batch']:,} at batch={comp['batch_size']})"
            )
        lat = serial.get("latency") or {}
        if lat.get("end_to_end_per_sample"):
            e2e = lat["end_to_end_per_sample"]
            print(f"  [latency]      e2e p50={e2e['p50_ms']:.3f} ms  p95={e2e['p95_ms']:.3f} ms")
        thr = serial.get("throughput") or {}
        if thr:
            print(
                f"  [throughput]   {thr['samples_per_second']:.2f} samples/s  "
                f"{thr['padded_input_megabytes_per_second_si']:.1f} MB/s (padded input)"
            )
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
