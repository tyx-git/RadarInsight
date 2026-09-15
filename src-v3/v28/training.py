"""Known-only V28 classification training and global endpoint selection."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import atomic_write_text, json_text, relative_project_path, sha256_file, canonical_sha256

from .checkpoint import checkpoint_path, save_checkpoint
from .data import V28TaskDataset, build_dataloader, model_input_kwargs
from .evaluation import evaluate_model
from .metrics import selection_key
from .model import V28TaskModel
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, artifact_path, task_classes


def _seed(seed: int, deterministic: bool = False) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)


def _device(context: V28Context) -> torch.device:
    requested = str(context.config.training["device"]).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V28 training requires CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported V28 device: {requested}")


def _log_line(handle: Any, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()


def _migration_initialization(context: V28Context, model: V28TaskModel, *, task: str) -> dict[str, Any]:
    """Copy only TR/RD encoder weights from the registered historical source."""

    source_key = "r15_l1_checkpoint" if task == V28_L1_TASK else "r23_l2_checkpoint"
    source_path = context.project_root / str(context.config.payload["migration_sources"][source_key])
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict):
        raise ValueError(f"V28 migration source is not a checkpoint object: {source_path}")
    source_state = source_payload.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError(f"V28 migration source has no model_state: {source_path}")
    source_prefix = "base." if task == V28_L1_TASK else "trainable_base_model."
    key_map: dict[str, str] = {}
    for name in model.tr_encoder.state_dict():
        key_map[f"tr_encoder.{name}"] = f"{source_prefix}tr_encoder.{name}"
    for name in model.rd_encoder.net.state_dict():
        key_map[f"rd_encoder.net.{name}"] = f"{source_prefix}rd_encoder.network.{name}"
    for name in model.rd_encoder.proj.state_dict():
        key_map[f"rd_encoder.proj.{name}"] = f"{source_prefix}rd_encoder.projection.{name}"
    target_state = model.state_dict()
    copied: list[str] = []
    missing: list[str] = []
    with torch.no_grad():
        for target_key, source_key_name in key_map.items():
            if source_key_name not in source_state:
                missing.append(source_key_name)
                continue
            source_value = source_state[source_key_name]
            if not isinstance(source_value, torch.Tensor) or tuple(source_value.shape) != tuple(target_state[target_key].shape):
                raise ValueError(f"V28 migration tensor shape differs: {source_key_name} -> {target_key}")
            target_state[target_key].copy_(source_value.to(dtype=target_state[target_key].dtype, device=target_state[target_key].device))
            copied.append(target_key)
    if missing:
        raise ValueError(f"V28 migration source is missing registered encoder keys: {missing[:3]}")
    source_state_hash = str(source_payload.get("model_state_sha256") or canonical_sha256(sorted(source_state)))
    return {
        "enabled": True,
        "source_key": source_key,
        "source_path": relative_project_path(context.project_root, source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_model_state_sha256": source_state_hash,
        "copied_component_scope": ["tr_encoder", "rd_encoder"],
        "copied_target_keys": copied,
        "forbidden_components": ["fusion", "liquid", "classifier", "classifier_norm", "observation_residual", "knownness", "quality"],
    }


def train_task(context: V28Context, *, task: str, seed: int, epochs: int | None = None, device: torch.device | None = None) -> dict[str, Any]:
    if task not in {V28_L1_TASK, V28_L2_TASK}:
        raise ValueError("V28 training task is invalid")
    if seed not in [int(value) for value in context.config.training["seeds"]]:
        raise ValueError("V28 seed is not registered")
    total_epochs = int(context.config.training["epochs"] if epochs is None else epochs)
    if total_epochs <= 0 or total_epochs > int(context.config.training["epochs"]):
        raise ValueError("V28 epochs are outside configured budget")
    target_device = _device(context) if device is None else device
    _seed(seed)
    train_set = V28TaskDataset(context, task=task, split="train")
    validation_set = V28TaskDataset(context, task=task, split="validation")
    model = V28TaskModel(context, task=task).to(target_device)
    migration = _migration_initialization(context, model, task=task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(context.config.training["learning_rate"]), weight_decay=float(context.config.training["weight_decay"]))
    key = "l1_known_gate_class_weights" if task == V28_L1_TASK else "l2_dbb_specialist_class_weights"
    weights = torch.tensor(context.config.training[key], dtype=torch.float32, device=target_device)
    log_path = artifact_path(context.project_root, str(context.config.artifacts["training_log_template"]), task=task, seed=seed)
    summary_path = artifact_path(context.project_root, str(context.config.artifacts["training_summary_template"]), task=task, seed=seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    steps = 0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        _log_line(log, f"run=initializing | task={task} | seed={seed} | epochs={total_epochs} | device={target_device} | manifest_sha256={context.manifest_sha256}")
        _log_line(log, f"dataset | train_parents={len(train_set)} | validation_parents={len(validation_set)} | unknown_classifier_exposure=0")
        for epoch in range(1, total_epochs + 1):
            model.train()
            losses: list[float] = []
            loader = build_dataloader(train_set, batch_size=int(context.config.training["batch_size"]), shuffle=True, seed=int(context.config.training["dataloader_seed"]) + epoch)
            total_batches = (len(train_set) + int(context.config.training["batch_size"]) - 1) // int(context.config.training["batch_size"])
            for batch_index, batch in enumerate(loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                output = model(**model_input_kwargs(batch, target_device))
                result = parent_final_fixed_parent_count_ce(output["sequence_logits"], batch["labels"].to(target_device), batch["tr_padding_mask"].to(target_device), class_weights=weights)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(context.config.training["gradient_clip_norm"]), error_if_nonfinite=True)
                optimizer.step(); steps += 1; losses.append(float(result["loss"].detach().cpu()))
                if batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0:
                    _log_line(log, f"epoch={epoch:03d} | batch={batch_index:04d}/{total_batches:04d} | batch_loss={losses[-1]:.8f}")
            evaluation = evaluate_model(model, validation_set, context=context, device=target_device, epoch=epoch)
            metrics = evaluation["parent_metrics"]
            assessment = {"epoch": epoch, "ranking_key": list(selection_key(metrics, task=task, epoch=epoch)), "metrics": metrics}
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_parent_metrics": metrics, "checkpoint_assessment": assessment})
            _log_line(log, f"epoch={epoch:03d} | train_loss={np.mean(losses):.8f} | accuracy={metrics['accuracy']:.8f} | macro_f1={metrics['macro_f1']:.8f}")
            for row in metrics["per_class"]:
                _log_line(log, f"class={row['class']} | precision={row['precision']:.8f} | recall={row['recall']:.8f} | f1={row['f1']:.8f} | support={row['support']}")
            epoch_path = checkpoint_path(context, task=task, seed=seed, epoch=epoch)
            save_checkpoint(context, model, task=task, seed=seed, epoch=epoch, global_step=steps, kind="epoch", metrics=metrics, optimizer=optimizer, path=epoch_path, migration_initialization=migration)
            if best_record is None or tuple(assessment["ranking_key"]) > tuple(best_record["ranking_key"]):
                best_record = assessment
                save_checkpoint(context, model, task=task, seed=seed, epoch=epoch, global_step=steps, kind="best", metrics=metrics, optimizer=None, migration_initialization=migration)
        payload = {"schema_version": "RADAR-LN3-V28-training-summary-v2", "experiment_id": "V28", "task": task, "seed": seed, "status": "completed", "elapsed_seconds": time.perf_counter() - started, "history": history, "best": best_record, "migration_initialization": migration, "unknown_classifier_training_exposure": 0, "test_payload_access_count": 0}
        atomic_write_text(summary_path, json_text(payload))
        _log_line(log, f"run=complete | best_epoch={best_record['epoch'] if best_record else None} | elapsed_seconds={payload['elapsed_seconds']:.3f}")
    return {"task": task, "seed": seed, "summary_path": str(summary_path), "log_path": str(log_path), "best": best_record}


__all__ = ["train_task"]
