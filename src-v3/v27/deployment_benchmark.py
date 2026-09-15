"""V27 串行部署基准。

基准绑定固定为 V27d-r15 v3 treatment L1 与 V27c-r23 robust_control L2。
每个测试样本按 parent 串行送入 L1，再按 L1 路由送入 L2；RD 优化变体只
改变无效 RD 帧是否进入卷积，不改变参数、权重或事件级聚合语义。

L3 必须使用 fitted registry 才能报告 Unknown 接受率/拒绝率。当前仓库的
L3 registry 是 unavailable，脚本会保留该事实并 fail-closed，不把 L1
分类结果冒充 L3 校准结果。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SRC_V3 = ROOT / "src-v3"
if str(SRC_V3) not in sys.path:
    sys.path.insert(0, str(SRC_V3))

from radar_ln3.runtime import sha256_file, write_json_atomic  # noqa: E402
from v27.V27d.r15.data import (  # noqa: E402
    R15L1Dataset,
    collate_r15,
    r15_model_input_kwargs,
    r15_sidecar_kwargs,
)
from v27.V27d.r15.model import build_r15_model, model_state_hash as r15_state_hash  # noqa: E402
from v27.V27d.r15.protocol import load_context as load_r15_context  # noqa: E402
from v27.v27c.r23.data import (  # noqa: E402
    R23L2Dataset,
    collate_r23,
    r23_model_input_kwargs,
    r23_reference_model_input_kwargs,
    r23_sidecar_kwargs,
)
from v27.v27c.r23.model import R23IntegratedL2Model  # noqa: E402
from v27.v27c.r23.protocol import load_r23_context  # noqa: E402
from v27.v27c.r23.training import load_r23_checkpoint  # noqa: E402
from v27.v27_baseline.model import V27RDSpatialEncoder  # noqa: E402


LOGGER = logging.getLogger("radar_ln3.deployment_benchmark")
DEFAULT_CONFIG = Path(__file__).with_name("deployment_benchmark_config.json")
DEFAULT_OUTPUT = ROOT / "src-v3" / "v27" / "v27-deployment-benchmark.json"
CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
L1_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")


def _configure_logging() -> None:
    # TODO：后续接入产品统一日志配置和前端进度事件协议。
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[benchmark] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


class MaskedFrameRDSpatialEncoder(V27RDSpatialEncoder):
    """只编码有效 RD 帧，然后恢复到 [batch,event,embedding]。"""

    def forward(self, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor) -> torch.Tensor:
        if (
            rd_frames.ndim != 6
            or tuple(rd_frames.shape[3:]) != (2, self.rd_height, self.rd_width)
            or rd_frame_mask.shape != rd_frames.shape[:3]
            or rd_frame_mask.dtype != torch.bool
            or not rd_frames.is_floating_point()
            or not torch.isfinite(rd_frames).all()
        ):
            raise ValueError("optimized RD encoder input differs")
        batch, length, frames = rd_frames.shape[:3]
        flat_mask = rd_frame_mask.reshape(-1)
        flat_frames = rd_frames.reshape(-1, 2, self.rd_height, self.rd_width)
        output = rd_frames.new_zeros((batch * length, self.embedding_dim))
        if bool(flat_mask.any()):
            encoded = self.projection(self.network(flat_frames[flat_mask]))
            event_ids = torch.arange(batch * length, device=rd_frames.device).repeat_interleave(frames)
            output.index_add_(0, event_ids[flat_mask], encoded)
        counts = rd_frame_mask.reshape(batch * length, frames).sum(dim=1).to(dtype=rd_frames.dtype)
        output = output / counts.clamp_min(1.0).unsqueeze(-1)
        return output.reshape(batch, length, self.embedding_dim)


def _replace_rd_encoder(module: nn.Module, optimized: bool) -> None:
    """递归切换 RD 编码器，保留完全相同的 state_dict。"""

    for name, child in list(module.named_children()):
        if name != "rd_encoder":
            _replace_rd_encoder(child, optimized)
            continue
        if optimized and type(child) is V27RDSpatialEncoder:
            replacement = MaskedFrameRDSpatialEncoder(
                child.embedding_dim,
                rd_height=child.rd_height,
                rd_width=child.rd_width,
            )
            replacement.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, replacement)
        elif not optimized and isinstance(child, MaskedFrameRDSpatialEncoder):
            replacement = V27RDSpatialEncoder(
                child.embedding_dim,
                rd_height=child.rd_height,
                rd_width=child.rd_width,
            )
            replacement.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, replacement)


class RSSSampler:
    """采样当前进程 RSS，作为 CPU 峰值近似值。"""

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
    """统计代表性前向中的 Linear、Conv2d 和 GRU 矩阵乘 MAC。"""

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
                per_position = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
                self._add(name, output.shape[0] * spatial * per_position * module.out_channels)

        return hook

    def _gru(self, name: str, module: nn.GRU) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                return
            rows = value.shape[0] * value.shape[1]
            directions = 2 if module.bidirectional else 1
            per_row = 3 * (module.input_size * module.hidden_size + module.hidden_size * module.hidden_size)
            self._add(name, rows * module.num_layers * directions * per_row)

        return hook


@dataclass
class Models:
    r15: nn.Module
    r23: R23IntegratedL2Model
    r15_checkpoint: Path
    r23_checkpoint: Path
    r15_payload: Mapping[str, Any]
    r23_payload: Mapping[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark config is not an object: {path}")
    return dict(value)


def _path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    candidate = Path(value)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"benchmark path escapes project root: {value}") from exc
    if must_exist and not target.is_file():
        raise FileNotFoundError(target)
    return target


def _checkpoint_path(root: Path, config: Mapping[str, Any]) -> Path:
    l1 = config["l1"]
    candidates = [str(l1["checkpoint_template"]), *[str(x) for x in l1.get("fallback_checkpoints", [])]]
    for template in candidates:
        candidate = _path(root, template.format(arm=l1["arm"], seed=int(l1["seed"])))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "V27d-r15 checkpoint is not ready; expected global-best or declared fallback: "
        + ", ".join(candidates)
    )


def _load_models(root: Path, config: Mapping[str, Any]) -> tuple[Models, Any, Any]:
    l1_config_path = _path(root, str(config["l1"]["config_path"]), must_exist=True)
    l1_context = load_r15_context(root, l1_config_path)
    l1_path = _checkpoint_path(root, config)
    l1_payload = torch.load(l1_path, map_location="cpu", weights_only=False)
    if not isinstance(l1_payload, Mapping):
        raise ValueError("V27d-r15 checkpoint is not a mapping")
    l1_model = build_r15_model(
        l1_context,
        arm=str(config["l1"]["arm"]),
        seed=int(config["l1"]["seed"]),
        device=torch.device("cpu"),
    )
    if l1_payload.get("model_contract") != l1_model.model_contract():
        raise ValueError("V27d-r15 checkpoint model contract differs")
    l1_model.load_state_dict(l1_payload["model_state"], strict=True)
    if r15_state_hash(l1_model) != l1_payload.get("model_state_sha256"):
        raise ValueError("V27d-r15 checkpoint state hash differs")

    l2_config_path = _path(root, str(config["l2"]["config_path"]), must_exist=True)
    l2_context = load_r23_context(root, l2_config_path)
    l2_arm = str(config["l2"]["arm"])
    l2_seed = int(config["l2"]["seed"])
    template = l2_context.config.artifacts.checkpoint_template
    from v27.v27c.r23.protocol import r23_artifact_path

    l2_path = r23_artifact_path(
        root,
        template,
        arm=l2_arm,
        seed=l2_seed,
        epoch=int(l2_context.config.training.primary_epoch),
    )
    if l2_context.config.protocol.get("checkpoint_selection") == "global_validation_per_class_f1":
        l2_path = l2_path.with_name("global-best.pt")
    if not l2_path.is_file():
        raise FileNotFoundError(f"V27c-r23 checkpoint is not ready: {l2_path}; wait for training to finish")
    # 复用实验包的完整 provenance 校验，避免误读旧 epoch-20 文件。
    l2_model, l2_payload = load_r23_checkpoint(
        l2_context,
        arm=l2_arm,
        seed=l2_seed,
        device=torch.device("cpu"),
    )
    for model in (l1_model, l2_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return Models(l1_model, l2_model, l1_path, l2_path, l1_payload, l2_payload), l1_context, l2_context


def _tensor_bytes(values: Mapping[str, Any]) -> int:
    total = 0
    for value in values.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
        elif isinstance(value, Mapping):
            total += _tensor_bytes(value)
    return int(total)


def _last_logits(output: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    sequence = output.get("sequence_logits")
    if not isinstance(sequence, torch.Tensor):
        value = output.get("final_logits")
        if not isinstance(value, torch.Tensor):
            raise ValueError("model output has no sequence/final logits")
        return value
    padding = batch["tr_padding_mask"]
    lengths = (~padding).long().sum(dim=1)
    rows = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[rows, lengths - 1]


def _r15_input(dataset: R15L1Dataset, index: int, device: torch.device) -> tuple[Any, dict[str, torch.Tensor], int, str]:
    item = dataset[index]
    batch = collate_r15([item])
    inputs = r15_model_input_kwargs(batch, device)
    inputs.update(r15_sidecar_kwargs(batch, device))
    return item, inputs, _tensor_bytes(inputs), str(item.sequence.base.parent_sample_id)


def _r23_input(dataset: R23L2Dataset, index: int, device: torch.device) -> tuple[Any, dict[str, Any], int]:
    item = dataset[index]
    batch = collate_r23([item])
    inputs: dict[str, Any] = {
        "base_inputs": r23_model_input_kwargs(batch, device),
        "reference_base_inputs": r23_reference_model_input_kwargs(batch, device),
    }
    inputs.update(r23_sidecar_kwargs(batch, device))
    byte_count = (
        _tensor_bytes(inputs["base_inputs"])
        + _tensor_bytes(inputs["reference_base_inputs"])
        + _tensor_bytes(
            {
                name: value
                for name, value in inputs.items()
                if name not in {"base_inputs", "reference_base_inputs"}
            }
        )
    )
    return item, inputs, byte_count


def _forward_l1(model: nn.Module, inputs: Mapping[str, Any]) -> tuple[int, torch.Tensor]:
    base = {key: value for key, value in inputs.items() if key in {
        "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason",
        "tr_padding_mask", "rd_frames", "rd_frame_mask", "rd_quality", "course_features",
        "course_valid", "course_support", "zero_ordinal",
    }}
    output = model(base_inputs=base, observation_increment_values=inputs["observation_increment_values"])
    logits = output["final_logits"]
    return int(torch.argmax(logits, dim=-1).item()), logits


def _forward_l2(model: R23IntegratedL2Model, inputs: Mapping[str, Any]) -> tuple[int, torch.Tensor]:
    output = model(**inputs)
    logits = _last_logits(output, inputs["base_inputs"])
    return int(torch.argmax(logits, dim=-1).item()), logits


def _class_metrics(labels: Sequence[str], predictions: Sequence[str], names: Sequence[str]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("metric labels and predictions differ")
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for name in names:
        tp = sum(label == name and prediction == name for label, prediction in zip(labels, predictions))
        fp = sum(label != name and prediction == name for label, prediction in zip(labels, predictions))
        fn = sum(label == name and prediction != name for label, prediction in zip(labels, predictions))
        support = sum(label == name for label in labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support})
        f1_values.append(f1)
    return {
        "support": len(labels),
        "accuracy": sum(label == prediction for label, prediction in zip(labels, predictions)) / max(1, len(labels)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def _prefix_inputs(inputs: Mapping[str, Any], event_count: int) -> dict[str, Any]:
    """把固定宽度序列截成 prefix，并对所有时间轴字段做 suffix padding。"""

    result: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, Mapping):
            result[name] = _prefix_inputs(value, event_count)
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] <= event_count:
            result[name] = value
            continue
        copied = value.clone()
        if value.dtype == torch.bool:
            copied[:, event_count:] = name.endswith("padding_mask")
        else:
            copied[:, event_count:] = 0
        result[name] = copied
    return result


def _model_macs(model: nn.Module, forward: Callable[[], Any]) -> dict[str, Any]:
    counter = MacCounter()
    counter.attach(model)
    try:
        with torch.inference_mode():
            forward()
    finally:
        counter.close()
    return {"macs": int(counter.macs), "flops": int(counter.macs * 2), "by_module": counter.by_module}


def _measure_serial(
    models: Models,
    r15_dataset: R15L1Dataset,
    r23_dataset: R23L2Dataset,
    *,
    device: torch.device,
    optimized_rd: bool,
    max_samples: int | None,
    warmup: int,
) -> dict[str, Any]:
    _replace_rd_encoder(models.r15, optimized_rd)
    _replace_rd_encoder(models.r23, optimized_rd)
    models.r15.to(device).eval()
    models.r23.to(device).eval()
    r23_positions = {parent_id: index for index, parent_id in enumerate(r23_dataset.parent_ids)}
    total = len(r15_dataset) if max_samples is None else min(len(r15_dataset), int(max_samples))
    if total <= 0:
        raise ValueError("serial benchmark has no samples")

    _, first_l1, _, first_id = _r15_input(r15_dataset, 0, device)
    with torch.inference_mode():
        first_l1_pred, _ = _forward_l1(models.r15, first_l1)
        if first_l1_pred == 0 and first_id in r23_positions:
            _, first_l2, _ = _r23_input(r23_dataset, r23_positions[first_id], device)
            _forward_l2(models.r23, first_l2)
    for _ in range(max(0, int(warmup) - 1)):
        with torch.inference_mode():
            _forward_l1(models.r15, first_l1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sampler = RSSSampler()
    sampler.start()
    labels: list[str] = []
    predictions: list[str] = []
    l1_labels: list[str] = []
    l2_labels: list[str] = []
    l1_predictions: list[str] = []
    l2_predictions: list[str] = []
    latencies: list[float] = []
    byte_count = 0
    rd_total = rd_valid = 0
    l1_macs: dict[str, Any] | None = None
    l2_macs: dict[str, Any] | None = None
    started = time.perf_counter()
    for index in range(total):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sample_start = time.perf_counter()
        item, l1_inputs, l1_bytes, parent_id = _r15_input(r15_dataset, index, device)
        label = str(item.sequence.base.label_name)
        if label in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
            l1_labels.append("KNOWN_TARGET_EVIDENCE")
        elif label == "ClutterTarget":
            l1_labels.append("KNOWN_CLUTTER_EVIDENCE")
        else:
            raise ValueError(f"unexpected closed-set L1 label: {label}")
        rd_mask = l1_inputs["rd_frame_mask"]
        rd_total += int(rd_mask.numel())
        rd_valid += int(rd_mask.sum().item())
        with torch.inference_mode():
            l1_pred, _ = _forward_l1(models.r15, l1_inputs)
            if l1_macs is None:
                l1_macs = _model_macs(models.r15, lambda: _forward_l1(models.r15, l1_inputs))
            l1_name = L1_CLASS_NAMES[l1_pred]
            if l1_pred == 0 and parent_id in r23_positions:
                l2_item, l2_inputs, l2_bytes = _r23_input(r23_dataset, r23_positions[parent_id], device)
                rd_mask_l2 = l2_inputs["base_inputs"]["rd_frame_mask"]
                rd_total += int(rd_mask_l2.numel())
                rd_valid += int(rd_mask_l2.sum().item())
                l2_pred, _ = _forward_l2(models.r23, l2_inputs)
                if l2_macs is None:
                    l2_macs = _model_macs(models.r23, lambda: _forward_l2(models.r23, l2_inputs))
                prediction = ("DroneTarget", "BirdTarget", "BalloonTarget")[l2_pred]
                l2_name = prediction
                l2_labels.append(str(l2_item.sequence.base.label_name))
                byte_count += l2_bytes
            elif l1_pred == 1:
                prediction = "ClutterTarget"
                l2_name = "not_routed"
            else:
                prediction = "UNRESOLVED"
                l2_name = "missing_l2_input"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - sample_start) * 1000.0)
        byte_count += l1_bytes
        labels.append(label)
        predictions.append(prediction)
        l1_predictions.append(l1_name)
        l2_predictions.append(l2_name)
        if (index + 1) % 25 == 0 or index + 1 == total:
            LOGGER.info("串行推理 rd=%s %d/%d", "valid-only" if optimized_rd else "all", index + 1, total)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    peak_rss = sampler.stop()
    allocated = reserved = None
    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    serial_macs = int((l1_macs or {"macs": 0})["macs"] + (l2_macs or {"macs": 0})["macs"])
    return {
        "rd_encoder_variant": "valid_frames_only" if optimized_rd else "all_frames",
        "sample_count": total,
        "labels": labels,
        "predictions": predictions,
        "l1_predictions": l1_predictions,
        "l2_predictions": l2_predictions,
        "metrics": _class_metrics(labels, predictions, CLASS_NAMES),
        "l1_metrics": _class_metrics(l1_labels, l1_predictions, L1_CLASS_NAMES),
        "l2_routed_metrics": _class_metrics(l2_labels, [value for value in l2_predictions if value in CLASS_NAMES], ("DroneTarget", "BirdTarget", "BalloonTarget")),
        "l2_route_count": len(l2_labels),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
            "max": float(max(latencies)),
        },
        "throughput": {
            "samples_per_second": float(total / elapsed),
            "MB_per_second": float(byte_count / elapsed / 1_000_000.0),
            "input_bytes": byte_count,
            "elapsed_seconds": elapsed,
        },
        "compute": {
            "representative_l1": l1_macs,
            "representative_l2": l2_macs,
            "serial_representative_macs": serial_macs,
            "serial_representative_flops": serial_macs * 2,
            "flops_definition": "FLOPS=2*MACs；只统计 Linear/Conv2d/GRU 矩阵乘，未统计逐元素、LayerNorm 和内存搬运。",
        },
        "memory": {
            "cpu_peak_rss_bytes": peak_rss,
            "cuda_peak_allocated_bytes": allocated,
            "cuda_peak_reserved_bytes": reserved,
        },
        "rd_frame_accounting": {
            "total_frame_slots": rd_total,
            "valid_frame_count": rd_valid,
            "skipped_frame_count": rd_total - rd_valid,
            "skipped_fraction": float((rd_total - rd_valid) / max(1, rd_total)),
        },
    }


def _unknown_diagnostic(
    models: Models,
    dataset: R15L1Dataset,
    device: torch.device,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _replace_rd_encoder(models.r15, True)
    models.r15.to(device).eval()
    labels: list[str] = []
    known_route: list[bool] = []
    for index in range(len(dataset)):
        item, inputs, _, _ = _r15_input(dataset, index, device)
        with torch.inference_mode():
            prediction, _ = _forward_l1(models.r15, inputs)
        labels.append(str(item.sequence.base.label_name))
        known_route.append(prediction == 0 or prediction == 1)
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            LOGGER.info("Unknown 诊断 %d/%d", index + 1, len(dataset))
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": index + 1,
                        "total": len(dataset),
                        "unknown_labels_seen": sum(value == "UnknownTarget" for value in labels),
                    }
                )
    unknown = [value for value in labels if value == "UnknownTarget"]
    accepted = sum(route for value, route in zip(labels, known_route) if value == "UnknownTarget")
    return {
        "l3_available": False,
        "l3_status": "unavailable",
        "unknown_support": len(unknown),
        "unknown_acceptance_rate": None,
        "unknown_rejection_rate": None,
        "reason": "当前 V27d-r1 registry 未 fitted 且旧 L3 head/cache 绑定 V27 baseline，不可直接用于 r15+r23 组合。",
        "diagnostic_l1_known_route_rate": float(accepted / max(1, len(unknown))),
        "diagnostic_only_not_l3": True,
    }


def _model_stats(models: Models, l3_path: Path) -> dict[str, Any]:
    l1_params = sum(parameter.numel() for parameter in models.r15.parameters())
    l2_params = sum(parameter.numel() for parameter in models.r23.parameters())
    l1_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in models.r15.parameters()
    )
    l2_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in models.r23.parameters()
    )
    l3_bytes = l3_path.stat().st_size if l3_path.is_file() else None
    return {
        "L1_r15_v3_treatment": {
            "parameter_count": int(l1_params),
            "parameter_bytes_fp32": int(l1_parameter_bytes),
            "model_size_bytes": int(models.r15_checkpoint.stat().st_size),
            "checkpoint_file_bytes": int(models.r15_checkpoint.stat().st_size),
        },
        "L2_r23_robust_control": {
            "parameter_count": int(l2_params),
            "parameter_bytes_fp32": int(l2_parameter_bytes),
            "model_size_bytes": int(models.r23_checkpoint.stat().st_size),
            "checkpoint_file_bytes": int(models.r23_checkpoint.stat().st_size),
        },
        "L3_current_registry": {"parameter_count": None, "artifact_file_bytes": l3_bytes, "status": "unavailable"},
        "serial_L1_plus_L2": {
            "parameter_count": int(l1_params + l2_params),
            "parameter_bytes_fp32": int(l1_parameter_bytes + l2_parameter_bytes),
            "model_size_bytes": int(models.r15_checkpoint.stat().st_size + models.r23_checkpoint.stat().st_size),
            "checkpoint_file_bytes": int(models.r15_checkpoint.stat().st_size + models.r23_checkpoint.stat().st_size),
            "l3_included": False,
        },
    }


def _prefix_stability(
    models: Models,
    r15_dataset: R15L1Dataset,
    r23_dataset: R23L2Dataset,
    device: torch.device,
    max_samples: int | None,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """报告每条航迹首次达到最终分类并保持稳定所需的事件数。"""

    _replace_rd_encoder(models.r15, True)
    _replace_rd_encoder(models.r23, True)
    models.r15.to(device).eval()
    models.r23.to(device).eval()
    r23_positions = {parent_id: index for index, parent_id in enumerate(r23_dataset.parent_ids)}
    total = len(r15_dataset) if max_samples is None else min(len(r15_dataset), int(max_samples))
    stable_counts: list[int] = []
    prefix_metrics: dict[str, dict[str, Any]] = {}
    for index in range(total):
        item, full_l1, _, parent_id = _r15_input(r15_dataset, index, device)
        length = int(item.sequence.base.event_count)
        full_l2 = None
        if parent_id in r23_positions:
            _, full_l2, _ = _r23_input(r23_dataset, r23_positions[parent_id], device)
        with torch.inference_mode():
            l1_final, _ = _forward_l1(models.r15, full_l1)
            if l1_final == 0 and full_l2 is not None:
                l2_final, _ = _forward_l2(models.r23, full_l2)
                final_prediction = ("DroneTarget", "BirdTarget", "BalloonTarget")[l2_final]
            elif l1_final == 1:
                final_prediction = "ClutterTarget"
            else:
                final_prediction = "UNRESOLVED"
        predictions: list[str] = []
        for event_count in range(1, length + 1):
            with torch.inference_mode():
                l1_pred, _ = _forward_l1(models.r15, _prefix_inputs(full_l1, event_count))
                if l1_pred == 0 and full_l2 is not None:
                    l2_pred, _ = _forward_l2(models.r23, _prefix_inputs(full_l2, event_count))
                    prediction = ("DroneTarget", "BirdTarget", "BalloonTarget")[l2_pred]
                elif l1_pred == 1:
                    prediction = "ClutterTarget"
                else:
                    prediction = "UNRESOLVED"
            predictions.append(prediction)
            prefix_metrics.setdefault(str(event_count), {"labels": [], "predictions": []})
            prefix_metrics[str(event_count)]["labels"].append(str(item.sequence.base.label_name))
            prefix_metrics[str(event_count)]["predictions"].append(prediction)
        first_stable = length
        for event_count in range(length):
            if all(value == final_prediction for value in predictions[event_count:]):
                first_stable = event_count + 1
                break
        stable_counts.append(first_stable)
        if (index + 1) % 10 == 0 or index + 1 == total:
            LOGGER.info("prefix 分析 %d/%d", index + 1, total)
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": index + 1,
                        "total": total,
                        "per_parent_minimum_event_count": list(stable_counts),
                        "prefix_metrics": {
                            key: _class_metrics(value["labels"], value["predictions"], CLASS_NAMES)
                            for key, value in sorted(
                                prefix_metrics.items(), key=lambda pair: int(pair[0])
                            )
                        },
                    }
                )
    return {
        "criterion": "从该事件起直到 parent 末尾，串行输出均等于完整 parent 输出；这是稳定性定义，不等同于正确率保证。",
        "sample_count": total,
        "per_parent_minimum_event_count": stable_counts,
        "minimum_event_count_quantiles": {
            "p50": float(np.percentile(stable_counts, 50)) if stable_counts else None,
            "p90": float(np.percentile(stable_counts, 90)) if stable_counts else None,
            "max": int(max(stable_counts)) if stable_counts else None,
        },
        "prefix_metrics": {
            key: _class_metrics(value["labels"], value["predictions"], CLASS_NAMES)
            for key, value in sorted(prefix_metrics.items(), key=lambda pair: int(pair[0]))
        },
    }


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    """原子写入当前阶段结果，保证异常时保留已完成指标。"""

    write_json_atomic(output, dict(report))


def _initial_report(
    *,
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    r15_context: Any,
    r23_context: Any,
    models: Models,
    l3_path: Path,
    l3_registry: Mapping[str, Any],
    r15_dataset: R15L1Dataset,
    unknown_dataset: R15L1Dataset,
    r23_dataset: R23L2Dataset,
    output: Path,
) -> dict[str, Any]:
    """建立可增量更新的报告骨架。"""

    return {
        "schema_version": "RADAR-LN3-V27-deployment-benchmark-v2",
        "status": "running",
        "current_stage": "loaded",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(output.relative_to(root).as_posix()),
        "completed_stages": [],
        "protocol": {
            "serial_input": True,
            "closed_set_data": "R15 validation view; L1 routes target parents to R23 robust_control L2 and clutter parents remain at L1",
            "unknown_data": "R15 representation_development view; no test payload",
            "l3_required_for_unknown_rates": True,
            "minimum_event_answer": "empirical prefix stability plus metrics at every prefix length",
            "rd_optimization": config["rd_encoder"]["optimized_description"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "provenance": {
            "config_path": str(config_path.relative_to(root).as_posix()),
            "config_sha256": sha256_file(config_path),
            "l1_config": str(r15_context.config.path.relative_to(root).as_posix()),
            "l1_checkpoint": str(models.r15_checkpoint.relative_to(root).as_posix()),
            "l1_checkpoint_sha256": sha256_file(models.r15_checkpoint),
            "l2_config": str(r23_context.config.config_path.relative_to(root).as_posix()),
            "l2_checkpoint": str(models.r23_checkpoint.relative_to(root).as_posix()),
            "l2_checkpoint_sha256": sha256_file(models.r23_checkpoint),
            "l3_registry": str(l3_path.relative_to(root).as_posix()),
            "l3_registry_sha256": sha256_file(l3_path) if l3_path.is_file() else None,
            "l3_registry_status": l3_registry.get("status"),
            "validation_parent_count": len(r15_dataset),
            "unknown_diagnostic_parent_count": len(unknown_dataset),
            "l2_validation_parent_count": len(r23_dataset),
        },
        "models": _model_stats(models, l3_path),
        "closed_set": {
            "l1_and_l2_serial": None,
            "measurements": [],
        },
        "unknown": None,
        "prefix_analysis": None,
        "rd_optimization_conclusion": {
            "parameter_count_unchanged": True,
            "model_size_unchanged": True,
            "measured_variants": [],
            "valid_frame_skip_fraction": None,
            "interpretation": "只有实际运行结果中的 MAC、FLOPS、吞吐量和时延可以作为结论；跳帧比例本身不能直接等价为同倍率端到端加速。",
        },
        "limitations": [
            "当前 L3 registry status=unavailable，未报告伪造的 Unknown 接受率/拒绝率。",
            "当前 r15/r23 没有为新组合重新拟合 L3；旧 V27d head/cache 不能跨 endpoint 复用。",
            "MAC/FLOPS 不含 LayerNorm、逐元素运算、数据搬运和 Python/DataLoader 开销；吞吐量和时延是本机实测值。",
            "CPU 峰值是进程 RSS 采样近似值，CUDA 峰值是 torch allocated/reserved 统计。",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="V27d-r15 + V27c-r23 串行部署基准")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    root = args.project_root.resolve()
    output = _path(root, args.output)
    report: dict[str, Any] = {
        "schema_version": "RADAR-LN3-V27-deployment-benchmark-v2",
        "status": "starting",
        "current_stage": "startup",
        "completed_stages": [],
        "output_path": str(output.relative_to(root).as_posix()),
    }
    _write_report(output, report)

    try:
        config_path = _path(root, args.config, must_exist=True)
        config = _load_json(config_path)
        if config.get("schema_version") != "RADAR-LN3-V27-deployment-benchmark-config-v2":
            raise ValueError("deployment benchmark config schema differs")

        report["current_stage"] = "loading_models_and_data"
        _write_report(output, report)
        models, r15_context, r23_context = _load_models(root, config)
        r15_dataset = R15L1Dataset(
            r15_context, view=str(config["data"]["r15_validation_view"])
        )
        unknown_dataset = R15L1Dataset(
            r15_context, view=str(config["data"]["r15_unknown_view"])
        )
        r23_dataset = R23L2Dataset(
            r23_context,
            split=str(config["data"]["r23_split"]),
            arm=str(config["l2"]["arm"]),
        )
        l3_path = _path(root, str(config["l3"]["registry_path"]))
        l3_registry = _load_json(l3_path) if l3_path.is_file() else {"status": "missing"}
        report = _initial_report(
            root=root,
            config_path=config_path,
            config=config,
            r15_context=r15_context,
            r23_context=r23_context,
            models=models,
            l3_path=l3_path,
            l3_registry=l3_registry,
            r15_dataset=r15_dataset,
            unknown_dataset=unknown_dataset,
            r23_dataset=r23_dataset,
            output=output,
        )
        _write_report(output, report)
        LOGGER.info(
            "模型和数据加载完成，validation=%d unknown=%d",
            len(r15_dataset),
            len(unknown_dataset),
        )

        devices: list[torch.device] = []
        if torch.cuda.is_available() and not args.skip_cuda:
            devices.append(torch.device("cuda:0"))
        devices.append(torch.device("cpu"))
        measurements: list[dict[str, Any]] = []
        planned_stages = [
            (device, optimized)
            for device in devices
            for optimized in (False, True)
        ]
        for stage_index, (device, optimized) in enumerate(planned_stages, start=1):
            variant = "valid_frames_only" if optimized else "all_frames"
            stage_name = f"serial_{device}_{variant}"
            report["status"] = "running"
            report["current_stage"] = stage_name
            report["stage_progress"] = {
                "stage_index": stage_index,
                "stage_count": len(planned_stages),
                "sample_count": 0,
                "total_samples": len(r15_dataset)
                if args.max_samples is None
                else min(len(r15_dataset), int(args.max_samples)),
            }
            _write_report(output, report)
            measurement = _measure_serial(
                models,
                r15_dataset,
                r23_dataset,
                device=device,
                optimized_rd=optimized,
                max_samples=args.max_samples,
                warmup=args.warmup,
            )
            measurement["device"] = str(device)
            measurement["stage_index"] = stage_index
            measurements.append(measurement)
            report["closed_set"]["measurements"] = measurements
            report["rd_optimization_conclusion"]["measured_variants"] = [
                value["rd_encoder_variant"] + ":" + value["device"]
                for value in measurements
            ]
            report["current_stage"] = f"completed_{stage_name}"
            report["completed_stages"].append(stage_name)
            report["stage_progress"] = {
                "stage_index": stage_index,
                "stage_count": len(planned_stages),
                "sample_count": measurement["sample_count"],
                "total_samples": measurement["sample_count"],
            }
            _write_report(output, report)

        preferred_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.skip_cuda else "cpu"
        )
        preferred = next(
            value
            for value in measurements
            if value["device"] == str(preferred_device)
            and value["rd_encoder_variant"] == "valid_frames_only"
        )
        report["closed_set"]["l1_and_l2_serial"] = preferred["metrics"]
        report["rd_optimization_conclusion"]["valid_frame_skip_fraction"] = preferred[
            "rd_frame_accounting"
        ]["skipped_fraction"]
        report["current_stage"] = "closed_set_measurements_complete"
        report["completed_stages"].append("closed_set_measurements")
        _write_report(output, report)

        report["current_stage"] = "unknown_diagnostic"
        _write_report(output, report)

        def save_unknown_progress(progress: Mapping[str, Any]) -> None:
            report["unknown_progress"] = dict(progress)
            _write_report(output, report)

        report["unknown"] = _unknown_diagnostic(
            models,
            unknown_dataset,
            preferred_device,
            progress_callback=save_unknown_progress,
        )
        report.pop("unknown_progress", None)
        report["completed_stages"].append("unknown_diagnostic")
        _write_report(output, report)

        if args.skip_prefix:
            report["prefix_analysis"] = {
                "status": "skipped_by_argument",
                "criterion": "not run",
            }
        else:
            report["current_stage"] = "prefix_analysis"
            _write_report(output, report)

            def save_prefix_progress(progress: Mapping[str, Any]) -> None:
                progress_value = dict(progress)
                report["prefix_analysis"] = {
                    "status": "running",
                    "criterion": "从该事件起直到 parent 末尾，串行输出均等于完整 parent 输出；这是稳定性定义，不等同于正确率保证。",
                    "sample_count": progress_value["processed"],
                    "total_sample_count": progress_value["total"],
                    "per_parent_minimum_event_count": progress_value[
                        "per_parent_minimum_event_count"
                    ],
                    "prefix_metrics": progress_value["prefix_metrics"],
                }
                _write_report(output, report)

            report["prefix_analysis"] = _prefix_stability(
                models,
                r15_dataset,
                r23_dataset,
                preferred_device,
                args.max_samples,
                progress_callback=save_prefix_progress,
            )
        report["completed_stages"].append("prefix_analysis")
        report["status"] = "completed_with_l3_unavailable"
        report["current_stage"] = "completed"
        report.pop("stage_progress", None)
        _write_report(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(output),
                    "sample_count": preferred["sample_count"],
                    "completed_stages": report["completed_stages"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["failed_at"] = report.get("current_stage", "unknown")
        try:
            _write_report(output, report)
        except Exception:
            LOGGER.exception("无法写入阶段性失败报告")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
