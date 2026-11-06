"""L1/L2 训练共享的 batch、评估和 checkpoint 辅助逻辑。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_ln3.checkpoint import canonical_hash, sha256_file
from radar_ln3.cache import CachedParentSequenceDataset
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.features import FeatureCache, build_model_batch
from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.model.config import ModelConfig, task_class_names


ProgressCallback = Callable[[Mapping[str, Any]], None]

TASK_LABELS = {
    "l1": ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"),
    "l2": ("DroneTarget", "BirdTarget", "BalloonTarget"),
}
KNOWN_TRAINING_LABELS = tuple(
    sorted({label for labels in TASK_LABELS.values() for label in labels})
)


def _identity_batch(items: list[Any]) -> list[Any]:
    """保留 parent 序列，交由特征层按当前 batch 做 suffix padding。"""

    return items


@dataclass(frozen=True)
class ModelBatchCollator:
    """把 parent 序列转为任务专用模型输入，可在 DataLoader worker 中运行。"""

    task: str
    max_rd_frames: int
    max_sequence_length: int
    stable_indices: tuple[int, ...]
    clipping: Mapping[str, Any] | None = None
    raw_echo_provider: Any | None = None
    stable_median: tuple[float, ...] | None = None
    stable_scale: tuple[float, ...] | None = None
    feature_caches: Mapping[str, FeatureCache] | None = None
    expected_view_hash: str | None = None
    baseline_anchor: Any | None = None
    stable_input_mode: str = "normalized"

    def __call__(self, sequences: list[Any]) -> dict[str, Any]:
        batch = build_model_batch(
            sequences,
            task=self.task,
            max_rd_frames=self.max_rd_frames,
            max_sequence_length=self.max_sequence_length,
            stable_indices=self.stable_indices,
            clipping=self.clipping,
            raw_echo_provider=self.raw_echo_provider,
            stable_median=(
                None
                if self.stable_median is None
                else np.asarray(self.stable_median, dtype=np.float32)
            ),
            stable_scale=(
                None
                if self.stable_scale is None
                else np.asarray(self.stable_scale, dtype=np.float32)
            ),
            feature_caches=self.feature_caches,
            expected_view_hash=self.expected_view_hash,
            baseline_anchor=self.baseline_anchor,
            stable_input_mode=self.stable_input_mode,
        )
        if self.task == "l1":
            keep = {
                "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
                "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
                "rd_quality", "observation_increment_values", "course_features",
                "course_valid", "course_support", "labels",
            }
        else:
            keep = {
                "tr_features", "tr_missing_mask", "reference_tr_features",
                "reference_tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
                "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
                "rd_quality", "rd_physical_features", "rd_physical_valid",
                "micro_doppler_features", "micro_doppler_valid",
                "stable_summary_tokens", "stable_summary_valid",
                "stable_frame_values", "stable_frame_mask", "motion_values",
                "labels", "lengths",
            }
            if "baseline_logits" in batch:
                keep.add("baseline_logits")
        return {key: value for key, value in batch.items() if key in keep}


def build_training_dataset(
    project_root: str | Path,
    *,
    task: str,
    split_name: str,
    manifest_path: str | Path = "src/config/manifests.json",
    manifest: DatasetManifest | None = None,
    validate_event_cache: bool = True,
) -> CachedParentSequenceDataset:
    """按 task 生成统一 manifest view 的 known-only 数据集。"""

    if task not in TASK_LABELS:
        raise ValueError(f"不支持的训练 task：{task}")
    return CachedParentSequenceDataset(
        project_root,
        manifest_path=manifest_path,
        split_name=split_name,
        include_unknown=False,
        training_view=True,
        allowed_labels=TASK_LABELS[task],
        task_name=task,
        manifest=manifest,
        validate_event_cache=validate_event_cache,
    )


def build_training_loader(
    dataset: CachedParentSequenceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    collate_fn: Callable[[list[Any]], Any] = _identity_batch,
) -> DataLoader[list[Any]]:
    """构造只按 parent 采样的 DataLoader。"""

    if int(batch_size) <= 0:
        raise ValueError("batch_size 必须是正整数")
    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("num_workers 必须是非负整数")
    if isinstance(prefetch_factor, bool) or int(prefetch_factor) <= 0:
        raise ValueError("prefetch_factor 必须是正整数")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(workers),
        prefetch_factor=int(prefetch_factor) if workers else None,
        collate_fn=collate_fn,
        generator=generator,
    )


def _metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """计算不依赖第三方指标实现的 parent-final 分类指标。"""

    predicted = np.asarray(predictions, dtype=np.int64)
    actual = np.asarray(labels, dtype=np.int64)
    if predicted.shape != actual.shape or actual.ndim != 1 or actual.size == 0:
        raise ValueError("评估预测和标签未对齐")
    rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        true_positive = int(((actual == class_id) & (predicted == class_id)).sum())
        false_positive = int(((actual != class_id) & (predicted == class_id)).sum())
        false_negative = int(((actual == class_id) & (predicted != class_id)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": str(name),
            "class_id": class_id,
            "support": int((actual == class_id).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    correct = int((predicted == actual).sum())
    micro_precision = correct / int(actual.size)
    micro_recall = micro_precision
    micro_f1 = micro_precision
    return {
        "sample_count": int(actual.size),
        "accuracy": float(correct / int(actual.size)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "confusion_matrix": [
            [int(((actual == true_id) & (predicted == pred_id)).sum()) for pred_id in range(len(class_names))]
            for true_id in range(len(class_names))
        ],
        "per_class": rows,
    }


def _move_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = batch.keys() if keys is None else keys
    for key in names:
        value = batch[key]
        result[key] = (
            value.to(device=device, non_blocking=device.type == "cuda")
            if isinstance(value, torch.Tensor)
            else value
        )
    return result


def _l1_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return _move_inputs(
        batch,
        device,
        keys=(
            "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
            "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
            "rd_quality", "course_features", "course_valid", "course_support",
        ),
    )


def _l2_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return _move_inputs(
        batch,
        device,
        keys=(
            "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
            "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
            "rd_quality", "rd_physical_features", "rd_physical_valid",
            "micro_doppler_features", "micro_doppler_valid",
        ),
    )


def _l2_reference_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    base_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """构造源实验 legacy reference 输入，保留当前 batch 的其他模态。"""

    required = ("reference_tr_features", "reference_tr_missing_mask")
    if any(key not in batch for key in required):
        raise ValueError("L2 batch 缺少 legacy reference TR 输入")
    result = dict(base_inputs)
    for key in required:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"L2 reference 输入不是 tensor：{key}")
        result[key.removeprefix("reference_")] = value.to(device=device)
    return result


def _require_stable_sidecars(batch: Mapping[str, Any]) -> None:
    """确认 L2 batch 带有 stable8 侧车字段。

    validity 是逐事件标记；一个 batch 内所有事件都无可靠 stable8 证据
    是合法情况，整体覆盖率应由训练入口按整个 train view 检查。
    """

    required = (
        "stable_summary_tokens",
        "stable_summary_valid",
        "stable_frame_values",
        "stable_frame_mask",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"L2 缺少 stable8 侧车：{missing}")


def train_l1_step(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float = 100.0,
    return_loss_tensor: bool = False,
) -> float | torch.Tensor:
    """执行一个 L1 parent-final backward step。"""

    inputs = _l1_inputs(batch, device)
    observations = batch["observation_increment_values"].to(
        device=device, non_blocking=device.type == "cuda"
    )
    output = model(base_inputs=inputs, observation_increment_values=observations)
    loss = parent_final_fixed_parent_count_ce(
        output["sequence_logits"],
        batch["labels"].to(device=device, non_blocking=device.type == "cuda"),
        inputs["tr_padding_mask"],
        class_weights=class_weights,
    )["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(gradient_clip_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    detached = loss.detach()
    return detached if return_loss_tensor else float(detached.item())


def train_l2_step(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float = 100.0,
    return_loss_tensor: bool = False,
    prefix_weight: float = 0.0,
    prefix_events: int = 0,
    return_loss_components: bool = False,
    baseline_reference_verified: bool = False,
) -> float | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """执行一个 L2 parent-final backward step。"""

    _require_stable_sidecars(batch)
    inputs = _l2_inputs(batch, device)
    reference_inputs = (
        None
        if baseline_reference_verified
        else _l2_reference_inputs(batch, device, base_inputs=inputs)
    )
    anchor = batch.get("baseline_logits")
    if not isinstance(anchor, torch.Tensor):
        raise ValueError("L2 batch 缺少 event_baseline_logits anchor")
    anchor = anchor.to(device=device, non_blocking=device.type == "cuda")
    output = model(
        base_inputs=inputs,
        reference_base_inputs=reference_inputs,
        baseline_logits=anchor.detach(),
        frozen_reference_logits_override=(
            anchor.detach() if baseline_reference_verified else None
        ),
        stable_summary_tokens=batch["stable_summary_tokens"].to(
            device=device, non_blocking=device.type == "cuda"
        ),
        stable_summary_valid=batch["stable_summary_valid"].to(
            device=device, non_blocking=device.type == "cuda"
        ),
        stable_frame_values=batch["stable_frame_values"].to(
            device=device, non_blocking=device.type == "cuda"
        ),
        stable_frame_mask=batch["stable_frame_mask"].to(
            device=device, non_blocking=device.type == "cuda"
        ),
        motion_values=batch["motion_values"].to(
            device=device, non_blocking=device.type == "cuda"
        ),
    )
    loss_result = parent_final_fixed_parent_count_ce(
        output["sequence_logits"],
        batch["labels"].to(device=device, non_blocking=device.type == "cuda"),
        inputs["tr_padding_mask"],
        class_weights=class_weights,
        prefix_weight=prefix_weight,
        prefix_events=prefix_events,
    )
    loss = loss_result["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(gradient_clip_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    detached = loss.detach()
    if return_loss_components:
        return detached, {
            "final_state_loss": loss_result["final_state_loss"].detach(),
            "prefix_auxiliary_loss": loss_result["prefix_auxiliary_loss"].detach(),
            "total_loss": detached,
            "prefix_event_count": int(loss_result["prefix_event_count"]),
        }
    return detached if return_loss_tensor else float(detached.item())


def train_l2_direct_step(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float = 100.0,
    return_loss_tensor: bool = False,
) -> float | torch.Tensor:
    """执行完整 L2 主干训练 step。"""

    inputs = _l2_inputs(batch, device)
    output = model.direct_logits(base_inputs=inputs)
    loss = parent_final_fixed_parent_count_ce(
        output["sequence_logits"],
        batch["labels"].to(device=device),
        inputs["tr_padding_mask"],
        class_weights=class_weights,
    )["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(gradient_clip_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    detached = loss.detach()
    return detached if return_loss_tensor else float(detached.item())


def evaluate_l1(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """在 parent-final 级别评估 L1。"""

    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                base_inputs=_l1_inputs(batch, device),
                observation_increment_values=batch["observation_increment_values"].to(
                    device=device, non_blocking=device.type == "cuda"
                ),
            )
            predictions.extend(output["final_logits"].argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return _metrics(predictions, labels, task_class_names(model.config, "l1"))


def evaluate_l2(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    baseline_reference_verified: bool = False,
) -> dict[str, Any]:
    """在 parent-final 级别评估 L2。"""

    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            _require_stable_sidecars(batch)
            inputs = _l2_inputs(batch, device)
            anchor = batch.get("baseline_logits")
            if not isinstance(anchor, torch.Tensor):
                raise ValueError("L2 batch 缺少 event_baseline_logits anchor")
            anchor = anchor.to(device=device, non_blocking=device.type == "cuda")
            reference_inputs = (
                None
                if baseline_reference_verified
                else _l2_reference_inputs(batch, device, base_inputs=inputs)
            )
            output = model(
                base_inputs=inputs,
                reference_base_inputs=reference_inputs,
                baseline_logits=anchor,
                frozen_reference_logits_override=(
                    anchor if baseline_reference_verified else None
                ),
                stable_summary_tokens=batch["stable_summary_tokens"].to(device=device, non_blocking=device.type == "cuda"),
                stable_summary_valid=batch["stable_summary_valid"].to(device=device, non_blocking=device.type == "cuda"),
                stable_frame_values=batch["stable_frame_values"].to(device=device, non_blocking=device.type == "cuda"),
                stable_frame_mask=batch["stable_frame_mask"].to(device=device, non_blocking=device.type == "cuda"),
                motion_values=batch["motion_values"].to(device=device, non_blocking=device.type == "cuda"),
            )
            lengths = inputs["tr_padding_mask"].logical_not().sum(dim=1)
            rows = torch.arange(lengths.shape[0], device=device)
            predictions.extend(output["sequence_logits"][rows, lengths - 1].argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return _metrics(predictions, labels, task_class_names(model.config, "l2"))


def evaluate_l2_direct(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """在完整 L2 主干上评估，不注入 baseline 或 stable residual。"""

    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model.direct_logits(base_inputs=_l2_inputs(batch, device))
            predictions.extend(output["final_logits"].argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return _metrics(predictions, labels, task_class_names(model.config, "l2"))


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    config: ModelConfig,
    manifest_sha256: str,
    task: str,
    epoch: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """构造正式 checkpoint payload，不保存旧 optimizer 状态。

    manifest_sha256 绑定稳定的数据 manifest hash，不绑定会登记当前
    checkpoint 的产品总 manifest 自 hash，避免形成自引用哈希循环。
    """

    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract() if hasattr(model, "model_contract") else {}
    return {
        "checkpoint_schema": "RADAR-LN3-CHECKPOINT-V1",
        "task": str(task),
        "epoch": int(epoch),
        "config_path": config.path.as_posix(),
        "config_sha256": sha256_file(config.path),
        "manifest_sha256": str(manifest_sha256),
        "model_contract": contract,
        "model_contract_sha256": canonical_hash(contract),
        "model_state": state,
        "metrics": dict(metrics),
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "l3": "not_in_scope",
        },
    }


__all__ = [
    "ProgressCallback",
    "KNOWN_TRAINING_LABELS",
    "TASK_LABELS",
    "build_training_dataset",
    "build_training_loader",
    "checkpoint_payload",
    "evaluate_l1",
    "evaluate_l2",
    "evaluate_l2_direct",
    "train_l1_step",
    "train_l2_step",
    "train_l2_direct_step",
]
