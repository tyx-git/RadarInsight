"""Train the V27d-r10 compact causal Doppler-shape branch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import relative_project_path
from v27.V27d.logging_utils import (
    close_logger,
    configure_logger,
    log_event,
    log_metric_set,
    progress_interval,
)
from v27.v27_baseline.metrics import summarize_records

from .checkpoint import save_checkpoint
from .features import load_descriptor_cache, validate_cache_binding
from .model import CompactCausalDopplerResidual, build_model
from .protocol import (
    R10_CLASS_NAMES,
    R10_SEEDS,
    artifact_path,
    load_context,
    write_sealed_json_if_identical,
)


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"]["device"]).strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r10 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r10 training.device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], split: str) -> Path:
    return artifact_path(
        root,
        config["artifacts"]["cache_template"].format(split=split),
        must_exist=True,
    )


def _inputs(
    payload: Mapping[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {
        "baseline_logits": tensors["frozen_l2_logits"][indices].to(device),
        "descriptor_values": tensors["descriptor_values"][indices].to(device),
        "descriptor_valid": tensors["descriptor_valid"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _records(payload: Mapping[str, Any], logits: torch.Tensor) -> list[dict[str, Any]]:
    tensors = payload["tensors"]
    labels = tensors["labels"].detach().cpu().numpy().astype(np.int64)
    lengths = tensors["lengths"].detach().cpu().numpy().astype(np.int64)
    values = logits.detach().cpu()
    rows = torch.arange(len(lengths), dtype=torch.long)
    finals = values[rows, torch.from_numpy(lengths - 1)]
    predictions = finals.argmax(dim=-1).tolist()
    return [
        {
            "parent_sample_id": str(parent_id),
            "true_id": int(true_id),
            "pred_id": int(pred_id),
        }
        for parent_id, true_id, pred_id in zip(
            payload["parent_sample_ids"], labels.tolist(), predictions, strict=True
        )
    ]


def _metrics(payload: Mapping[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    return summarize_records(
        _records(payload, logits),
        class_names=R10_CLASS_NAMES,
        scope="parent_final",
    )


def _evaluate(
    model: CompactCausalDopplerResidual,
    payload: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
    indices = torch.arange(len(payload["parent_sample_ids"]), dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model(**_inputs(payload, indices, device))
    return _metrics(payload, output["sequence_logits"]), output


def _seed_everything(seed: int) -> None:
    value = int(seed)
    torch.manual_seed(value)
    np.random.seed(value % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def train_seed(
    project_root: str | Path,
    *,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R10_SEEDS:
        raise ValueError("V27d-r10 seed is not registered")
    train = load_descriptor_cache(_cache_path(root, config, "train"))
    validation = load_descriptor_cache(_cache_path(root, config, "val"))
    validate_cache_binding(train, root=root, config=config, split="train")
    validate_cache_binding(validation, root=root, config=config, split="val")
    device = _device(config)
    _seed_everything(seed)
    model = build_model(config["model"], seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    class_weights = torch.tensor(
        config["training"]["class_weights"], dtype=torch.float32, device=device
    )
    indices = torch.arange(len(train["parent_sample_ids"]), dtype=torch.long)
    batch_size = int(config["training"]["batch_size"])
    steps_per_epoch = (len(indices) + batch_size - 1) // batch_size
    order_generator = torch.Generator(device="cpu").manual_seed(
        int(config["training"]["dataloader_seed"]) + int(seed)
    )
    log_path = artifact_path(
        root, config["artifacts"]["log_template"].format(seed=seed)
    )
    logger = configure_logger(log_path, name=f"v27d.r10.seed.{seed}")
    started = time.perf_counter()
    primary_state: dict[str, torch.Tensor] | None = None
    primary_epoch = int(config["training"]["primary_epoch"])
    history: list[dict[str, Any]] = []
    try:
        log_event(
            logger,
            "training_start",
            experiment_id="V27d-r10",
            seed=seed,
            device=str(device),
            train_parents=len(indices),
            epochs=int(config["training"]["epochs"]),
            loss=config["training"]["loss_name"],
            descriptor_dim=int(config["model"]["descriptor_dim"]),
            unknown_payload_access_count=0,
            test_payload_access_count=0,
        )
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            shuffled = indices[torch.randperm(len(indices), generator=order_generator)]
            losses: list[float] = []
            gradient_max = 0.0
            for batch_number, start in enumerate(
                range(0, len(shuffled), batch_size), start=1
            ):
                batch_indices = shuffled[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(train, batch_indices, device))
                loss = parent_final_fixed_parent_count_ce(
                    output["sequence_logits"],
                    train["tensors"]["labels"][batch_indices].to(device),
                    train["tensors"]["padding_mask"][batch_indices].to(device),
                    class_weights=class_weights,
                )["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r10 loss is non-finite")
                loss.backward()
                norm = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"]["gradient_clip_norm"]),
                        error_if_nonfinite=True,
                    ).item()
                )
                optimizer.step()
                losses.append(float(loss.detach().item()))
                gradient_max = max(gradient_max, norm)
                if (
                    batch_number == 1
                    or batch_number == steps_per_epoch
                    or batch_number % progress_interval(steps_per_epoch) == 0
                ):
                    log_event(
                        logger,
                        "batch",
                        epoch=epoch,
                        batch=f"{batch_number}/{steps_per_epoch}",
                        loss=losses[-1],
                        gradient_norm=norm,
                    )
            validation_metrics, _ = _evaluate(model, validation, device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "gradient_norm_max": gradient_max,
                "validation_parent_metrics": validation_metrics,
                "unknown_payload_access_count": 0,
                "test_payload_access_count": 0,
            }
            history.append(row)
            log_metric_set(logger, "validation_parent_final", validation_metrics)
            if epoch == primary_epoch:
                primary_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            log_event(
                logger,
                "epoch_complete",
                epoch=epoch,
                train_loss=row["train_loss"],
                validation_accuracy=validation_metrics["accuracy"],
                validation_macro_f1=validation_metrics["macro_f1"],
                primary_epoch=primary_epoch,
                elapsed_seconds=time.perf_counter() - started,
            )
        if primary_state is None:
            raise RuntimeError("V27d-r10 primary epoch state was not captured")
        model.load_state_dict(primary_state, strict=True)
        final_metrics, _ = _evaluate(model, validation, device)
        cache_hashes = {
            split: load_descriptor_cache(_cache_path(root, config, split))["cache_sha256"]
            for split in ("train", "val")
        }
        checkpoint_path = artifact_path(
            root, config["artifacts"]["checkpoint_template"].format(seed=seed)
        )
        save_checkpoint(
            checkpoint_path,
            root=root,
            config=config,
            manifest=manifest,
            model=model,
            seed=seed,
            epoch=primary_epoch,
            global_step=primary_epoch * steps_per_epoch,
            metrics=final_metrics,
            cache_hashes=cache_hashes,
        )
        summary = {
            "schema_version": "RADAR-LN3-V27d-r10-training-summary-v1",
            "experiment_id": "V27d-r10",
            "status": "completed",
            "seed": int(seed),
            "selected_epoch": primary_epoch,
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "log_path": relative_project_path(root, log_path),
            "validation_parent_metrics": final_metrics,
            "history": history,
            "frozen_liquid_v2": True,
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path = artifact_path(
            root,
            config["artifacts"]["training_summary_template"].format(seed=seed),
        )
        write_sealed_json_if_identical(summary_path, summary)
        log_event(
            logger,
            "training_complete",
            selected_epoch=primary_epoch,
            checkpoint_path=str(checkpoint_path),
            summary_path=str(summary_path),
            elapsed_seconds=summary["elapsed_seconds"],
        )
        return {
            "status": "completed",
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "summary_path": relative_project_path(root, summary_path),
            "log_path": relative_project_path(root, log_path),
            "selected_epoch": primary_epoch,
            "validation_parent_metrics": final_metrics,
        }
    finally:
        close_logger(logger)


__all__ = ["train_seed"]
