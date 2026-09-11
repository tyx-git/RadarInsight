"""Fixed-epoch known-only training for the V27c motion residual."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import canonical_sha256, sha256_file, write_torch_if_absent_or_identical
from v27.v27_baseline.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    progress_interval,
)
from v27.v27_baseline.runtime import environment_snapshot, seed_v27

from .features import load_feature_cache, load_feature_manifest
from .model import V27CMotionResidual, build_v27c_model
from .protocol import (
    V27C_CHECKPOINT_SCHEMA_VERSION,
    V27C_L2_CLASSES,
    V27CContext,
    v27c_artifact_path,
    write_sealed_json_if_identical,
)


def configured_device(context: V27CContext) -> torch.device:
    requested = context.config.training.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27c configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported V27c device: {requested}")


def _deterministic_cuda_workspace(context: V27CContext) -> str | None:
    """Require the cuBLAS setting before strict deterministic CUDA starts."""

    if (
        context.config.training.device != "cuda"
        or not context.config.training.cuda_deterministic_algorithms
    ):
        return None
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "V27c strict deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
            "to be ':4096:8' or ':16:8' before Python starts"
        )
    return value


class _MotionDataset(Dataset[dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping):
            raise ValueError("V27c motion Dataset lacks tensors")
        self.tensors = {str(name): value for name, value in tensors.items()}
        self.parents = [str(value) for value in payload["parent_ids"]]
        self.labels = [str(value) for value in payload["label_names"]]
        if len(self.parents) != len(self.labels):
            raise ValueError("V27c motion Dataset metadata differs")
        self.label_ids = torch.tensor(
            [V27C_L2_CLASSES.index(value) for value in self.labels], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.parents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not 0 <= int(index) < len(self):
            raise IndexError("V27c motion Dataset index is invalid")
        row = int(index)
        return {
            "frozen_endpoint_logits": self.tensors["frozen_endpoint_logits"][row],
            "motion_values": self.tensors["motion_values"][row],
            "padding_mask": self.tensors["padding_mask"][row],
            "length": self.tensors["lengths"][row],
            "label": self.label_ids[row],
            "label_name": self.labels[row],
            "parent_id": self.parents[row],
        }


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27c cannot collate an empty batch")
    names = ("frozen_endpoint_logits", "motion_values", "padding_mask", "length", "label")
    return {
        **{name: torch.stack([item[name] for item in items]) for name in names},
        "label_names": [item["label_name"] for item in items],
        "parent_ids": [item["parent_id"] for item in items],
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("V27c state contains a non-tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_path(context: V27CContext, *, seed: int, epoch: int) -> Path:
    return v27c_artifact_path(
        context.project_root, context.config.artifacts.checkpoint_template,
        seed=seed, epoch=epoch,
    )


def _log_path(context: V27CContext, *, seed: int) -> Path:
    return v27c_artifact_path(
        context.project_root, context.config.artifacts.training_log_template, seed=seed
    )


def _summary_path(context: V27CContext, *, seed: int) -> Path:
    return v27c_artifact_path(
        context.project_root, context.config.artifacts.training_summary_template, seed=seed
    )


def _checkpoint_payload(
    context: V27CContext,
    *,
    model: V27CMotionResidual,
    seed: int,
    epoch: int,
    global_step: int,
    train_loss: float,
    train_cache: Mapping[str, Any],
    validation_cache: Mapping[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": V27C_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "manifest_sha256": context.manifest_sha256,
        "input_binding_sha256": canonical_sha256(context.input_binding),
        "feature_manifest_sha256": load_feature_manifest(context)["payload_sha256"],
        "train_cache_sha256": train_cache["cache_sha256"],
        "validation_cache_sha256": validation_cache["cache_sha256"],
        "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "frozen_v27a_main_state_sha256": context.config.base.expected_v27a_main_state_sha256,
        "class_names": list(V27C_L2_CLASSES),
        "class_weights": list(context.config.base.expected_l2_class_weights),
        "optimizer": context.config.training.optimizer,
        "loss_name": context.config.training.loss_name,
        "loss_reduction": context.config.training.loss_reduction,
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "model_state": state,
        "model_state_sha256": _state_hash(state),
        "metrics": {"train_loss": float(train_loss), "selection": "fixed_primary_epoch"},
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }


def load_v27c_checkpoint(
    context: V27CContext, *, seed: int, device: torch.device | None = None
) -> tuple[V27CMotionResidual, dict[str, Any]]:
    if seed not in context.config.training.seeds:
        raise ValueError("V27c checkpoint seed is not registered")
    epoch = context.config.training.primary_epoch
    path = _checkpoint_path(context, seed=seed, epoch=epoch)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c checkpoint is not an object")
    checked = dict(payload)
    if (
        checked.get("checkpoint_schema") != V27C_CHECKPOINT_SCHEMA_VERSION
        or checked.get("experiment_id") != context.config.experiment_id
        or int(checked.get("seed", -1)) != int(seed)
        or int(checked.get("epoch", -1)) != epoch
        or checked.get("manifest_sha256") != context.manifest_sha256
        or checked.get("input_binding_sha256") != canonical_sha256(context.input_binding)
        or checked.get("frozen_v27a_main_checkpoint_sha256") != context.config.base.expected_v27a_main_checkpoint_sha256
        or checked.get("frozen_v27a_main_state_sha256") != context.config.base.expected_v27a_main_state_sha256
        or checked.get("class_names") != list(V27C_L2_CLASSES)
        or checked.get("class_weights") != list(context.config.base.expected_l2_class_weights)
        or checked.get("optimizer") != context.config.training.optimizer
        or checked.get("loss_name") != context.config.training.loss_name
        or checked.get("loss_reduction") != context.config.training.loss_reduction
        or checked.get("unknown_training_exposure") != 0
        or checked.get("test_payload_access_count") != 0
    ):
        raise ValueError("V27c checkpoint identity differs")
    feature_manifest = load_feature_manifest(context)
    if checked.get("feature_manifest_sha256") != feature_manifest["payload_sha256"]:
        raise ValueError("V27c checkpoint feature-manifest binding differs")
    train_cache = load_feature_cache(context, split="train")
    if checked.get("train_cache_sha256") != train_cache["cache_sha256"]:
        raise ValueError("V27c checkpoint train-cache binding differs")
    validation_cache = load_feature_cache(context, split="val")
    if checked.get("validation_cache_sha256") != validation_cache["cache_sha256"]:
        raise ValueError("V27c checkpoint validation-cache binding differs")
    model = build_v27c_model(context.config, seed=seed)
    state = checked.get("model_state")
    if not isinstance(state, Mapping) or checked.get("model_state_sha256") != _state_hash(state):
        raise ValueError("V27c checkpoint state hash differs")
    if (
        checked.get("model_contract") != model.model_contract()
        or checked.get("model_contract_sha256") != canonical_sha256(model.model_contract())
    ):
        raise ValueError("V27c checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    return model.to(torch.device("cpu") if device is None else device), checked


def train_v27c_motion(
    context: V27CContext, *, seed: int, epochs_override: int | None = None
) -> dict[str, Any]:
    """Fit one registered V27c head at the preregistered fixed endpoint."""

    if seed not in context.config.training.seeds:
        raise ValueError("V27c training seed is not registered")
    epochs = context.config.training.epochs if epochs_override is None else int(epochs_override)
    if epochs != context.config.training.primary_epoch:
        raise ValueError("V27c only permits its fixed primary epoch")
    cublas_workspace_config = _deterministic_cuda_workspace(context)
    seed_v27(
        seed,
        deterministic_cuda=context.config.training.cuda_deterministic_algorithms,
    )
    train_cache = load_feature_cache(context, split="train")
    validation_cache = load_feature_cache(context, split="val")
    dataset = _MotionDataset(train_cache)
    if len(dataset) != 953 or len(validation_cache["parent_ids"]) != 223:
        raise ValueError("V27c frozen parent support differs")
    loader_seed = int(context.config.training.dataloader_seed) + int(seed)
    loader = DataLoader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator(device="cpu").manual_seed(loader_seed),
        collate_fn=_collate,
    )
    if len(loader) <= 0:
        raise ValueError("V27c training loader is empty")
    device = configured_device(context)
    model = build_v27c_model(context.config, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    weights = torch.tensor(
        context.config.base.expected_l2_class_weights, dtype=torch.float32, device=device
    )
    logger = configure_training_logger(
        _log_path(context, seed=seed), logger_name=f"radar_ln3.v27c.motion.seed_{seed}"
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    global_step = 0
    try:
        log_event(
            logger, "v27c_motion_run=start", seed=seed, epochs=epochs,
            train_parents=len(dataset), validation_parents=len(validation_cache["parent_ids"]),
            batch_size=context.config.training.batch_size, dataloader_seed=loader_seed,
            frozen_endpoint=context.config.base.expected_v27a_main_checkpoint_sha256,
            motion_schema=context.config.data.motion_schema,
            motion_mask="numeric_gating_only_not_input", unknown=0, clutter=0, test=0,
            optimizer=context.config.training.optimizer,
            loss=context.config.training.loss_name,
            loss_reduction=context.config.training.loss_reduction,
            cuda_deterministic_algorithms=context.config.training.cuda_deterministic_algorithms,
            cublas_workspace_config=cublas_workspace_config,
        )
        interval = progress_interval(len(loader))
        for epoch in range(1, epochs + 1):
            model.train()
            losses: list[float] = []
            gradient_max = 0.0
            clipped_steps = 0
            nonzero_events = 0
            active_events = 0
            for step, batch in enumerate(loader, start=1):
                logits = batch["frozen_endpoint_logits"].to(device)
                motion = batch["motion_values"].to(device)
                padding = batch["padding_mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    frozen_endpoint_logits=logits, motion_values=motion, padding_mask=padding
                )
                if not torch.equal(output["baseline_logits"], logits):
                    raise AssertionError("V27c model altered frozen endpoint logits")
                loss = parent_final_fixed_parent_count_ce(
                    output["sequence_logits"], batch["label"].to(device), padding,
                    class_weights=weights,
                )["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27c motion loss is non-finite")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), context.config.training.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item())
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().item()))
                gradient_max = max(gradient_max, norm)
                clipped_steps += int(norm > context.config.training.gradient_clip_norm)
                nonzero_events += int(output["motion_evidence_mask"].sum().item())
                active_events += int((~padding).sum().item())
                if step == 1 or step == len(loader) or step % interval == 0:
                    log_event(
                        logger, "v27c_motion_progress", epoch=f"{epoch:02d}/{epochs:02d}",
                        step=f"{step:03d}/{len(loader):03d}", global_step=global_step,
                        loss=losses[-1], gradient_norm=norm,
                    )
            record = {
                "epoch": epoch, "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm_max": gradient_max,
                "gradient_clipped_steps": clipped_steps,
                "motion_nonzero_event_fraction": float(nonzero_events / active_events),
            }
            history.append(record)
            log_event(
                logger, "v27c_motion_epoch=complete", epoch=epoch,
                train_loss=record["train_loss"], gradient_norm_max=gradient_max,
                motion_nonzero_event_fraction=record["motion_nonzero_event_fraction"],
                elapsed=format_duration(time.perf_counter() - started),
            )
        payload = _checkpoint_payload(
            context, model=model, seed=seed, epoch=epochs, global_step=global_step,
            train_loss=history[-1]["train_loss"], train_cache=train_cache,
            validation_cache=validation_cache,
        )
        checkpoint_path = _checkpoint_path(context, seed=seed, epoch=epochs)
        write_torch_if_absent_or_identical(checkpoint_path, payload)
        summary = {
            "schema_version": "RADAR-LN3-V27c-motion-training-summary-v1",
            "status": "completed_fixed_epoch_motion_training",
            "experiment_id": context.config.experiment_id,
            "seed": seed,
            "selected_epoch": epochs,
            "selection": "fixed_primary_epoch",
            "dataloader_seed": loader_seed,
            "history": history,
            "checkpoint_path": str(checkpoint_path.relative_to(context.project_root)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_state_sha256": payload["model_state_sha256"],
            "training_log_path": str(_log_path(context, seed=seed).relative_to(context.project_root)).replace("\\", "/"),
            "feature_manifest_sha256": payload["feature_manifest_sha256"],
            "validation_cache_sha256": payload["validation_cache_sha256"],
            "environment": environment_snapshot(),
            "cublas_workspace_config": cublas_workspace_config,
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
        }
        summary_path = _summary_path(context, seed=seed)
        write_sealed_json_if_identical(summary_path, summary)
        return {**summary, "summary_path": str(summary_path)}
    finally:
        close_training_logger(logger)


__all__ = ["_MotionDataset", "_collate", "configured_device", "load_v27c_checkpoint", "train_v27c_motion"]
