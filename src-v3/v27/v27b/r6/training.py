"""Fixed-epoch training for the isolated V27b-r6 knownness heads."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radar_ln3.runtime import canonical_sha256, sha256_file, write_torch_if_absent_or_identical
from v27.v27_baseline.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    progress_interval,
)

from .features import load_feature_cache, load_feature_manifest, stable8_stability_features
from .model import R6KnownnessHead, build_r6_model
from .protocol import (
    R6_ARMS,
    R6_HEAD_CHECKPOINT_SCHEMA_VERSION,
    R6_KNOWN_LABELS,
    R6_NEGATIVE_LABEL,
    R6Context,
    r6_artifact_path,
)


def configured_r6_device(context: R6Context) -> torch.device:
    """Resolve r6's declared runtime without borrowing the V27 config type."""

    requested = context.config.training.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27b-r6 configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported V27b-r6 device: {requested}")


class _FeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        tensors = payload["tensors"]
        self.tensors = {str(name): value for name, value in tensors.items()}
        self.labels = [str(value) for value in payload["label_names"]]
        self.parents = [str(value) for value in payload["parent_ids"]]
        self.stable = stable8_stability_features(payload)
        if len(self.labels) != len(self.parents) or self.stable.shape[0] != len(self.parents):
            raise ValueError("V27b-r6 feature dataset metadata differs")

    def __len__(self) -> int:
        return len(self.parents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not 0 <= int(index) < len(self):
            raise IndexError("V27b-r6 feature dataset index is invalid")
        row = int(index)
        return {
            "event_trajectory": self.tensors["event_trajectory"][row],
            "summary_features": self.tensors["summary_features"][row],
            "task_facts_sequence": self.tensors["task_facts_sequence"][row],
            "stable_features": self.stable[row],
            "padding_mask": self.tensors["padding_mask"][row],
            "fit_eligible": self.tensors["fit_eligible"][row],
            "label_name": self.labels[row],
            "parent_id": self.parents[row],
        }


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27b-r6 cannot collate an empty batch")
    tensor_names = (
        "event_trajectory",
        "summary_features",
        "task_facts_sequence",
        "stable_features",
        "padding_mask",
        "fit_eligible",
    )
    return {
        **{name: torch.stack([item[name] for item in items]) for name in tensor_names},
        "label_names": [item["label_name"] for item in items],
        "parent_ids": [item["parent_id"] for item in items],
    }


def _fit_targets(labels: list[str], fit_eligible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    known = torch.tensor(
        [label in R6_KNOWN_LABELS for label in labels], dtype=torch.bool, device=fit_eligible.device
    )
    unknown = torch.tensor(
        [label == R6_NEGATIVE_LABEL for label in labels], dtype=torch.bool, device=fit_eligible.device
    )
    mask = fit_eligible & (known | unknown)
    return known.to(dtype=torch.float32), mask


def _class_balance(target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = target.masked_select(mask)
    positives = int(selected.sum().item())
    negatives = int(selected.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("V27b-r6 fit pool needs both Known and Unknown examples")
    return float(negatives) / float(positives)


def _checkpoint_path(context: R6Context, *, arm: str, seed: int, epoch: int) -> Path:
    return r6_artifact_path(
        context.project_root,
        context.config.artifacts.checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=epoch,
    )


def _log_path(context: R6Context, *, arm: str, seed: int) -> Path:
    return r6_artifact_path(
        context.project_root,
        context.config.artifacts.training_log_template,
        arm=arm,
        seed=seed,
    )


def _summary_path(context: R6Context, *, arm: str, seed: int) -> Path:
    return r6_artifact_path(
        context.project_root,
        context.config.artifacts.training_summary_template,
        arm=arm,
        seed=seed,
    )


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    chunks: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("V27b-r6 model state has a non-tensor")
        chunks[str(name)] = value.detach().cpu()
    from .features import _tensor_digest

    return _tensor_digest(chunks)


def _checkpoint_payload(
    context: R6Context,
    *,
    model: R6KnownnessHead,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    train_loss: float,
    fit_cache: Mapping[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {
        "checkpoint_schema": R6_HEAD_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "r6_manifest_sha256": context.manifest_sha256,
        "feature_manifest_sha256": load_feature_manifest(context)["feature_manifest_sha256"],
        "fit_cache_sha256": fit_cache["cache_sha256"],
        "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "classifier_logits_must_remain_bitwise_unchanged": True,
        "classifier_logits_input_to_knownness": False,
        "model_contract": model.model_contract(),
        "model_contract_sha256": canonical_sha256(model.model_contract()),
        "model_state": state,
        "model_state_sha256": _state_hash(state),
        "metrics": {
            "train_loss": float(train_loss),
            "selection": "fixed_primary_epoch",
        },
        "unknown_is_classifier_class": False,
        "test_payload_access_count": 0,
    }


def load_r6_checkpoint(
    context: R6Context,
    *,
    arm: str,
    seed: int,
    epoch: int | None = None,
    device: torch.device | None = None,
) -> tuple[R6KnownnessHead, dict[str, Any]]:
    actual_epoch = context.config.training.primary_epoch if epoch is None else int(epoch)
    path = _checkpoint_path(context, arm=arm, seed=seed, epoch=actual_epoch)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27b-r6 checkpoint is not an object")
    checked = dict(payload)
    if (
        checked.get("checkpoint_schema") != R6_HEAD_CHECKPOINT_SCHEMA_VERSION
        or checked.get("experiment_id") != context.config.experiment_id
        or checked.get("arm") != arm
        or int(checked.get("seed", -1)) != seed
        or int(checked.get("epoch", -1)) != actual_epoch
        or checked.get("r6_manifest_sha256") != context.manifest_sha256
        or checked.get("frozen_v27a_main_checkpoint_sha256")
        != context.config.base.expected_v27a_main_checkpoint_sha256
        or checked.get("classifier_logits_input_to_knownness") is not False
    ):
        raise ValueError("V27b-r6 checkpoint identity differs")
    if checked.get("feature_manifest_sha256") != load_feature_manifest(context)["feature_manifest_sha256"]:
        raise ValueError("V27b-r6 checkpoint feature-manifest binding differs")
    fit_cache = load_feature_cache(context, role="representation_fit")
    if checked.get("fit_cache_sha256") != fit_cache["cache_sha256"]:
        raise ValueError("V27b-r6 checkpoint fit-cache binding differs")
    state = checked.get("model_state")
    if not isinstance(state, Mapping) or checked.get("model_state_sha256") != _state_hash(state):
        raise ValueError("V27b-r6 checkpoint state hash differs")
    model = build_r6_model(context.config, arm=arm, seed=seed)
    if checked.get("model_contract") != model.model_contract() or checked.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("V27b-r6 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    return model.to(torch.device("cpu") if device is None else device), checked


def train_r6_head(
    context: R6Context,
    *,
    arm: str,
    seed: int,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one arm on representation_fit; selection stays at the fixed epoch."""

    if arm not in R6_ARMS or seed not in context.config.training.seeds:
        raise ValueError("V27b-r6 training arm or seed is invalid")
    epochs = context.config.training.epochs if epochs_override is None else int(epochs_override)
    if epochs <= 0 or (epochs_override is None and epochs != context.config.training.primary_epoch):
        raise ValueError("V27b-r6 formal training uses its configured fixed primary epoch")
    if epochs_override is not None and epochs > context.config.training.epochs:
        raise ValueError("V27b-r6 epoch override exceeds configured maximum")
    if epochs != context.config.training.primary_epoch:
        raise ValueError("V27b-r6 cannot create a non-primary formal checkpoint")
    from v27.v27_baseline.runtime import seed_v27

    seed_v27(seed)
    fit_cache = load_feature_cache(context, role="representation_fit")
    dataset = _FeatureDataset(fit_cache)
    target_all, mask_all = _fit_targets(
        dataset.labels, dataset.tensors["fit_eligible"]
    )
    pos_weight = _class_balance(target_all, mask_all)
    # Keep the configured loader stream distinct from model initialization.
    # This makes data order reproducible from the declared experiment contract
    # while still varying it deterministically across independent head seeds.
    loader_seed = int(context.config.training.dataloader_seed) + int(seed)
    generator = torch.Generator(device="cpu").manual_seed(loader_seed)
    loader = DataLoader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=_collate,
    )
    device = configured_r6_device(context)
    model = build_r6_model(context.config, arm=arm, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    logger = configure_training_logger(
        _log_path(context, arm=arm, seed=seed),
        logger_name=f"radar_ln3.v27b.r6.{arm}.seed_{seed}",
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    step = 0
    try:
        log_event(
            logger,
            "r6_knownness_run=start",
            arm=arm,
            seed=seed,
            epochs=epochs,
            parents=len(dataset),
            fit_eligible=int(mask_all.sum().item()),
            known_positive=int(target_all.masked_select(mask_all).sum().item()),
            unknown_negative=int(mask_all.sum().item() - target_all.masked_select(mask_all).sum().item()),
            dataloader_seed=loader_seed,
            frozen_v27a_main=context.config.base.expected_v27a_main_checkpoint_sha256,
        )
        for epoch in range(1, epochs + 1):
            losses: list[float] = []
            skipped = 0
            model.train()
            for batch in loader:
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                target, mask = _fit_targets(batch["label_names"], batch["fit_eligible"])
                if not bool(mask.any()):
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                scores = model(
                    event_trajectory=batch["event_trajectory"],
                    summary_features=batch["summary_features"],
                    task_facts_sequence=batch["task_facts_sequence"],
                    stable_features=batch["stable_features"],
                    padding_mask=batch["padding_mask"],
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    scores[mask],
                    target[mask],
                    pos_weight=scores.new_tensor(pos_weight),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27b-r6 knownness loss is non-finite")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), context.config.training.gradient_clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                losses.append(float(loss.detach().item()))
                if step == 1 or step % progress_interval(max(1, len(loader))) == 0:
                    log_event(
                        logger,
                        "r6_knownness_progress",
                        arm=arm,
                        epoch=f"{epoch:02d}/{epochs:02d}",
                        global_step=step,
                        loss=losses[-1],
                        gradient_norm=float(gradient_norm.item()),
                    )
            if not losses:
                raise RuntimeError("V27b-r6 epoch contains no eligible knownness batch")
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "eligible_batches": len(losses),
                "skipped_batches": skipped,
                "global_step": step,
            }
            history.append(record)
            log_event(
                logger,
                "r6_knownness_epoch=complete",
                arm=arm,
                epoch=epoch,
                train_loss=record["train_loss"],
                eligible_batches=record["eligible_batches"],
                elapsed=format_duration(time.perf_counter() - started),
            )
        payload = _checkpoint_payload(
            context,
            model=model,
            arm=arm,
            seed=seed,
            epoch=epochs,
            global_step=step,
            train_loss=history[-1]["train_loss"],
            fit_cache=fit_cache,
        )
        checkpoint = _checkpoint_path(context, arm=arm, seed=seed, epoch=epochs)
        write_torch_if_absent_or_identical(checkpoint, payload)
        summary = {
            "status": "completed_fixed_epoch_knownness_training",
            "schema_version": "RADAR-LN3-V27b-r6-knownness-training-summary-v1",
            "experiment_id": context.config.experiment_id,
            "arm": arm,
            "seed": seed,
            "selected_epoch": epochs,
            "selection": "fixed_primary_epoch",
            "dataloader_seed": loader_seed,
            "history": history,
            "checkpoint_path": str(checkpoint.relative_to(context.project_root)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_state_sha256": payload["model_state_sha256"],
            "training_log_path": str(_log_path(context, arm=arm, seed=seed).relative_to(context.project_root)).replace("\\", "/"),
            "feature_manifest_sha256": payload["feature_manifest_sha256"],
            "unknown_is_classifier_class": False,
            "test_payload_access_count": 0,
        }
        from .protocol import write_sealed_json_if_identical

        path = _summary_path(context, arm=arm, seed=seed)
        write_sealed_json_if_identical(path, summary)
        return {**summary, "summary_path": str(path)}
    finally:
        close_training_logger(logger)


__all__ = ["configured_r6_device", "load_r6_checkpoint", "train_r6_head"]
