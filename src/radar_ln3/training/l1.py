"""L1 正式训练流程。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from radar_ln3.checkpoint import Checkpoint
from radar_ln3.data import load_manifest
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.derived import register_product_artifacts
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.features import FeatureCache
from radar_ln3.model import build_l1_model, load_model_config
from radar_ln3.logger import log_classification_metrics

from .common import (
    ModelBatchCollator,
    build_training_dataset,
    build_training_loader,
    checkpoint_payload,
    evaluate_l1,
    train_l1_step,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value) % (2**32))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前环境不可用")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支持的训练 device：{requested}")


def _emit(
    progress_callback: ProgressCallback | None,
    event: Mapping[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)
    if "validation" in event:
        log_classification_metrics(
            event["validation"], task="l1", epoch=event["epoch"]
        )


def load_l1_base_model(
    project_root: str | Path,
    *,
    device: torch.device | None = None,
    arm: str = "observation_increment",
    config_path: str | Path = "src/config/model.json",
    artifact_root: str | Path | None = None,
) -> tuple[Any, Any]:
    """加载正式配置和 l1base.pt，构造可训练 L1 模型。"""

    root = Path(project_root).resolve()
    config = load_model_config(root, config_path, artifact_root=artifact_root)
    checkpoint = Checkpoint.load(
        root,
        config.artifacts.l1_base_weight_path,
    )
    model = build_l1_model(
        config,
        arm=arm,
        seed=config.training.seed,
        base_state=checkpoint.state,
        device=device,
    )
    return config, model


def train_l1(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    device: str = "cpu",
    arm: str = "observation_increment",
    manifest: DatasetManifest | None = None,
    progress_callback: ProgressCallback | None = None,
    train_feature_caches: Mapping[str, FeatureCache] | None = None,
    validation_feature_caches: Mapping[str, FeatureCache] | None = None,
    validate_caches: bool = True,
    progress_interval: int = 25,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    config_path: str | Path = "src/config/model.json",
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """执行正式 L1 训练并保存 epoch/final checkpoint。"""

    root = Path(project_root).resolve()
    if isinstance(progress_interval, bool) or int(progress_interval) <= 0:
        raise ValueError("progress_interval 必须是正整数")
    config = load_model_config(root, config_path, artifact_root=artifact_root)
    _seed(config.training.l1_seed)
    manifest = manifest or load_manifest(
        manifest_path, root, verify_files=True, allow_unmaterialized=False
    )
    source_manifest_hash = data_manifest_hash(manifest)
    train_set = build_training_dataset(
        root,
        task="l1",
        split_name="train",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_event_cache=validate_caches,
    )
    validation_set = build_training_dataset(
        root,
        task="l1",
        split_name="validation",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_event_cache=validate_caches,
    )
    effective_batch_size = int(batch_size or config.training.l1_batch_size)
    if effective_batch_size <= 0:
        raise ValueError("L1 batch_size 必须为正整数")
    target_device = _device(device)
    effective_workers = (
        (2 if target_device.type == "cuda" else 0)
        if num_workers is None
        else int(num_workers)
    )
    effective_pin_memory = (
        target_device.type == "cuda" if pin_memory is None else bool(pin_memory)
    )
    collator_args = dict(
        task="l1",
        max_rd_frames=config.data.max_rd_frames_per_point,
        max_sequence_length=config.data.max_sequence_length,
        stable_indices=config.data.l2.stable8_indices,
    )
    train_loader = build_training_loader(
        train_set,
        collate_fn=ModelBatchCollator(
            **collator_args,
            feature_caches=train_feature_caches,
            expected_view_hash=train_set.view.view_hash,
        ),
        batch_size=effective_batch_size,
        shuffle=True,
        seed=config.training.l1_dataloader_seed,
        num_workers=effective_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=effective_pin_memory,
    )
    validation_loader = build_training_loader(
        validation_set,
        collate_fn=ModelBatchCollator(
            **collator_args,
            feature_caches=validation_feature_caches,
            expected_view_hash=validation_set.view.view_hash,
        ),
        batch_size=effective_batch_size,
        shuffle=False,
        seed=config.training.l1_dataloader_seed,
        num_workers=effective_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=effective_pin_memory,
    )
    _, model = load_l1_base_model(
        root, device=target_device, arm=arm, config_path=config_path,
        artifact_root=artifact_root,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.l1_learning_rate,
        weight_decay=config.training.l1_weight_decay,
    )
    class_weights = torch.tensor(
        config.training.l1_class_weights,
        dtype=torch.float32,
        device=target_device,
    )
    # epoch 和最终 checkpoint 共用配置指定的 artifact root。
    output_root = root / config.artifacts.l1_output_path
    output_root = output_root.parent
    epoch_root = output_root / "epochs"
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, ...] | None = None
    best_payload: dict[str, Any] | None = None
    started = time.perf_counter()
    _emit(progress_callback, {
        "stage": "l1_start",
        "train_parents": len(train_set),
        "validation_parents": len(validation_set),
        "num_workers": effective_workers,
        "prefetch_factor": int(prefetch_factor) if effective_workers else 0,
        "pin_memory": effective_pin_memory,
    })
    for epoch in range(1, config.training.l1_epochs + 1):
        model.train()
        losses: list[torch.Tensor] = []
        loader_iterator = iter(train_loader)
        for step in range(1, len(train_loader) + 1):
            wait_started = time.perf_counter()
            batch = next(loader_iterator)
            batch_wait_seconds = time.perf_counter() - wait_started
            compute_started = time.perf_counter()
            loss = train_l1_step(
                model,
                batch,
                device=target_device,
                class_weights=class_weights,
                optimizer=optimizer,
                gradient_clip_norm=config.training.l1_gradient_clip_norm,
                return_loss_tensor=True,
            )
            assert isinstance(loss, torch.Tensor)
            losses.append(loss)
            if step % int(progress_interval) == 0 or step == len(train_loader):
                _emit(
                    progress_callback,
                    {
                        "stage": "l1_step",
                        "epoch": epoch,
                        "step": step,
                        "total": len(train_loader),
                        "loss": float(loss.item()),
                        "batch_wait_seconds": batch_wait_seconds,
                        "step_compute_seconds": time.perf_counter() - compute_started,
                    },
                )
        metrics = evaluate_l1(
            model,
            validation_loader,
            device=target_device,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
            "validation": metrics,
        }
        rows = {str(row["class"]): row for row in metrics["per_class"]}
        minimum_class_f1 = min(float(row["f1"]) for row in rows.values())
        rank = (
            minimum_class_f1,
            float(metrics["macro_f1"]),
            -float(epoch),
        )
        record["checkpoint_selection_metrics"] = {
            "minimum_class_f1": minimum_class_f1,
            "macro_f1": float(metrics["macro_f1"]),
            "epoch": int(epoch),
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            config=config,
            manifest_sha256=source_manifest_hash,
            task="l1",
            epoch=epoch,
            metrics=record,
        )
        Checkpoint.save(root, epoch_root / f"epoch-{epoch:03d}.pt", payload)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = payload
        _emit(progress_callback, {"stage": "l1_epoch", **record, "elapsed_seconds": time.perf_counter() - started})
    if best_payload is None:
        raise RuntimeError("L1 训练没有产生 checkpoint")
    best_payload["selection"] = config.training.l1_checkpoint_selection
    final_checkpoint, _ = Checkpoint.save(root, config.artifacts.l1_output_path, best_payload)
    registered_manifest = root / manifest_path
    if artifact_root is None or Path(artifact_root).as_posix().rstrip("/") == "artifacts/radar_ln3":
        registered_manifest = register_product_artifacts(
            root,
            manifest_path=manifest_path,
            manifest=manifest,
            overwrite=True,
            artifact_root="artifacts/radar_ln3",
        )
    summary = {
        "schema_version": "RADAR-LN3-TRAINING-SUMMARY-V1",
        "task": "l1",
        "arm": arm,
        "selected_epoch": int(best_payload["epoch"]),
        "train_parent_count": len(train_set),
        "validation_parent_count": len(validation_set),
        "batch_size": effective_batch_size,
        "data_loader": {
            "num_workers": effective_workers,
            "prefetch_factor": int(prefetch_factor) if effective_workers else 0,
            "pin_memory": effective_pin_memory,
        },
        "training": {
            "learning_rate": config.training.l1_learning_rate,
            "weight_decay": config.training.l1_weight_decay,
            "gradient_clip_norm": config.training.l1_gradient_clip_norm,
            "class_weights": list(config.training.l1_class_weights),
            "epochs": config.training.l1_epochs,
            "seed": config.training.l1_seed,
            "dataloader_seed": config.training.l1_dataloader_seed,
            "checkpoint_selection": config.training.l1_checkpoint_selection,
        },
        "history": history,
        "checkpoint": final_checkpoint.provenance(kind="final", task="l1"),
        "manifest_sha256": source_manifest_hash,
        "manifest_path": str(registered_manifest),
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _emit(progress_callback, {"stage": "l1_complete", **summary["checkpoint"], "selected_epoch": summary["selected_epoch"]})
    return summary


__all__ = ["ProgressCallback", "load_l1_base_model", "train_l1"]
