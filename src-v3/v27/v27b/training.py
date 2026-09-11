"""Training loop for the frozen-base V27b conditional residual head."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import relative_project_path

from .cache import V27BFeatureCacheDataset, build_feature_dataloader, load_feature_cache
from .checkpoint import save_v27b_checkpoint
from .evaluation import assess_v27b_checkpoint, evaluate_v27b_arm
from .logging_utils import (
    LOGGER_PREFIX,
    close_logger,
    configure_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from .model import V27BConditionalPairResidual, build_v27b_model
from .protocol import V27BContext, V27B_L2_CLASS_NAMES
from .runtime import (
    checkpoint_path,
    configured_device,
    environment_snapshot,
    feature_cache_path,
    seed_v27b,
    training_log_path,
    training_summary_path,
    write_json_if_absent_or_identical,
)


TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27b-conditioned-training-summary-v2"


def _batch_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = [
        "parent_state",
        "event_representation",
        "frozen_l2_logits",
        "v27a_residual_logits",
        "event_summary_valid",
        "padding_mask",
    ]
    if batch.get("event_frame_values") is not None:
        names.extend(("event_frame_values", "event_frame_mask"))
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27b batch field is not a tensor: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    return result


def _assert_trainable_boundary(model: V27BConditionalPairResidual) -> None:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise ValueError("V27b conditional head has no trainable parameters")
    if len(trainable_names) != len(list(model.named_parameters())):
        raise ValueError("V27b model unexpectedly contains frozen parameters")
    if not torch.equal(
        model.correction_projection[-1].weight,
        torch.zeros_like(model.correction_projection[-1].weight),
    ):
        raise ValueError("V27b correction projection is not zero initialised")


def _diagnostics(
    output: Mapping[str, torch.Tensor | dict[str, torch.Tensor]],
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    correction = output["correction_logits"]
    delta_increment = output["pairwise_delta_increment"]
    if not isinstance(correction, torch.Tensor) or not isinstance(delta_increment, torch.Tensor):
        raise TypeError("V27b output diagnostics fields are invalid")
    padding = inputs["padding_mask"]
    reliable = inputs["event_summary_valid"]
    active = ~padding
    no_reliable = active & ~reliable
    return {
        "correction_absolute_max": float(correction[active].abs().max().item()) if bool(active.any()) else 0.0,
        "correction_zero_sum_max": float(correction[active].sum(-1).abs().max().item()) if bool(active.any()) else 0.0,
        "balloon_correction_absolute_max": float(correction[..., 2][active].abs().max().item()) if bool(active.any()) else 0.0,
        "no_reliable_event_count": int(no_reliable.sum().item()),
        "no_reliable_increment_nonzero_count": int(
            (delta_increment.abs() > 0.0).masked_select(no_reliable).sum().item()
        ),
    }


def _initial_zero_evaluation(
    context: V27BContext,
    *,
    model: V27BConditionalPairResidual,
    device: torch.device,
) -> dict[str, Any]:
    result = evaluate_v27b_arm(
        context,
        arm="conditional",
        split_name="val",
        model=model,
        checkpoint_epoch=1,
        device=device,
    )
    # A zero-initialised treatment must be bitwise equivalent to control.
    control = evaluate_v27b_arm(context, arm="control", split_name="val")
    if result["parent_metrics"] != control["parent_metrics"]:
        raise AssertionError("V27b zero-initialised treatment differs from control")
    return result


def train_v27b(
    context: V27BContext,
    *,
    seed: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    if actual_seed not in context.config.training.seeds:
        raise ValueError("V27b requested seed is not registered")
    total_epochs = (
        context.config.training.epochs
        if epochs_override is None
        else epochs_override
    )
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise ValueError("V27b epochs_override must be an integer")
    if total_epochs <= 0 or total_epochs > context.config.training.epochs:
        raise ValueError("V27b epochs_override exceeds configured budget")
    if context.config.training.primary_epoch > total_epochs:
        raise ValueError("V27b primary epoch is outside requested range")
    log_path = training_log_path(context, seed=actual_seed)
    logger = configure_logger(
        log_path,
        name=f"{LOGGER_PREFIX}.train.seed_{actual_seed}",
    )
    started = time.perf_counter()
    log_event(
        logger,
        "run=initializing",
        experiment="V27b",
        seed=actual_seed,
        epochs=total_epochs,
        log_path=str(log_path),
    )
    try:
        return _train_v27b(
            context,
            seed=actual_seed,
            total_epochs=total_epochs,
            logger=logger,
            log_path=log_path,
            started=started,
        )
    except Exception:
        logger.exception(
            "run=failed | experiment=V27b | seed=%d | elapsed=%s",
            actual_seed,
            format_duration(time.perf_counter() - started),
        )
        raise
    finally:
        close_logger(logger)


def _train_v27b(
    context: V27BContext,
    *,
    seed: int,
    total_epochs: int,
    logger: logging.Logger,
    log_path: Path,
    started: float,
) -> dict[str, Any]:
    device = configured_device(context)
    seed_v27b(seed)
    train_path = feature_cache_path(context, "train")
    val_path = feature_cache_path(context, "val")
    train_payload = load_feature_cache(train_path, context=context, split="train")
    val_payload = load_feature_cache(val_path, context=context, split="val")
    train_digest = str(train_payload["cache_payload_sha256"])
    val_digest = str(val_payload["cache_payload_sha256"])
    train_set = V27BFeatureCacheDataset(train_path, context=context, split="train")
    val_set = V27BFeatureCacheDataset(val_path, context=context, split="val")
    generator = torch.Generator(device="cpu").manual_seed(
        context.config.training.dataloader_seed
    )
    loader = build_feature_dataloader(
        train_set,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = build_v27b_model(context.config, seed=seed).to(device)
    _assert_trainable_boundary(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    weights = torch.tensor(
        context.config.base.expected_l2_class_weights,
        dtype=torch.float32,
        device=device,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch <= 0:
        raise ValueError("V27b training cache contains no batches")
    log_event(
        logger,
        "run=start",
        experiment="V27b",
        seed=seed,
        device=str(device),
        train_parents=len(train_set),
        validation_parents=len(val_set),
        batch_size=context.config.training.batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        loss=context.config.training.loss_name,
        loss_reduction=context.config.training.loss_reduction,
        frozen_v27a_encoder=True,
        train_cache_sha256=train_digest,
        validation_cache_sha256=val_digest,
        unknown=0,
        test=0,
    )
    zero_eval = _initial_zero_evaluation(context, model=model, device=device)
    log_metric_set(logger, "zero_initialised_control_equivalence", zero_eval["parent_metrics"])
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    global_step = 0
    interval = progress_interval(steps_per_epoch)
    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        diagnostics = {
            "correction_absolute_max": 0.0,
            "correction_zero_sum_max": 0.0,
            "balloon_correction_absolute_max": 0.0,
            "no_reliable_event_count": 0,
            "no_reliable_increment_nonzero_count": 0,
        }
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            inputs = _batch_inputs(batch, device)
            output = model(**inputs)
            sequence_logits = output["sequence_logits"]
            if not isinstance(sequence_logits, torch.Tensor):
                raise TypeError("V27b model sequence logits are invalid")
            loss = parent_final_fixed_parent_count_ce(
                sequence_logits,
                batch["label"].to(device=device),
                inputs["padding_mask"],
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError("V27b loss is non-finite")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    context.config.training.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("V27b gradient is non-finite")
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, gradient_norm)
            clipped_steps += int(
                gradient_norm > context.config.training.gradient_clip_norm
            )
            current = _diagnostics(output, inputs)
            for key in diagnostics:
                if key.endswith("count"):
                    diagnostics[key] = int(diagnostics[key]) + int(current[key])
                else:
                    diagnostics[key] = max(float(diagnostics[key]), float(current[key]))
            if step == 1 or step == steps_per_epoch or step % interval == 0:
                log_event(
                    logger,
                    "progress",
                    epoch=f"{epoch:03d}/{total_epochs:03d}",
                    step=f"{step:04d}/{steps_per_epoch:04d}",
                    global_step=global_step,
                    loss=float(loss.detach().item()),
                    gradient_norm=gradient_norm,
                    elapsed=format_duration(time.perf_counter() - epoch_started),
                )
        validation = evaluate_v27b_arm(
            context,
            arm="conditional",
            split_name="val",
            model=model,
            checkpoint_epoch=epoch,
            device=device,
        )
        assessment = assess_v27b_checkpoint(validation, epoch=epoch)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": gradient_max,
            "gradient_clipped_steps": clipped_steps,
            "validation_parent_metrics": validation["parent_metrics"],
            "validation_parent_balanced_eligible_prefix_metrics": validation[
                "parent_balanced_eligible_prefix_metrics"
            ],
            "checkpoint_assessment": assessment,
            "diagnostics": dict(diagnostics),
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
        }
        history.append(record)
        epoch_checkpoint = checkpoint_path(context, seed=seed, epoch=epoch)
        save_v27b_checkpoint(
            epoch_checkpoint,
            context=context,
            model=model,
            optimizer=optimizer,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            metrics={"train_loss": record["train_loss"], "checkpoint_assessment": assessment, "diagnostics": diagnostics},
            kind="epoch",
            cache_hashes={"train": train_digest, "val": val_digest},
        )
        if best is None or tuple(assessment["ranking_key"]) > tuple(best["ranking_key"]):
            best = dict(assessment)
        log_event(
            logger,
            "epoch=complete",
            epoch=f"{epoch:03d}/{total_epochs:03d}",
            global_step=global_step,
            train_loss=record["train_loss"],
            gradient_norm_max=gradient_max,
            gradient_clipped_steps=clipped_steps,
            secondary_best_epoch=best["epoch"] if best else "n/a",
            selection_minimum=assessment["minimum_per_class_precision_recall_f1"],
            selection_macro_f1=assessment["macro_f1"],
            elapsed=format_duration(time.perf_counter() - epoch_started),
        )
        log_metric_set(logger, "validation_parent_final", validation["parent_metrics"])
        log_metric_set(
            logger,
            "validation_parent_balanced_eligible_prefix",
            validation["parent_balanced_eligible_prefix_metrics"],
        )
    if best is None:
        raise RuntimeError("V27b training completed without checkpoint assessment")
    primary_path = checkpoint_path(
        context, seed=seed, epoch=context.config.training.primary_epoch
    )
    summary = {
        "schema_version": TRAINING_SUMMARY_SCHEMA,
        "experiment_id": "V27b",
        "status": "completed_frozen_v27a_conditioned_residual_training",
        "arm": "conditional",
        "seed": seed,
        "epochs": total_epochs,
        "primary_epoch": context.config.training.primary_epoch,
        "primary_checkpoint_path": relative_project_path(context.project_root, context.project_root / primary_path),
        "secondary_own_best_assessment": best,
        "history": history,
        "source_checkpoint_file_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "source_checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "class_weights": list(context.config.base.expected_l2_class_weights),
        "loss_name": context.config.training.loss_name,
        "loss_reduction": context.config.training.loss_reduction,
        "feature_cache_payload_sha256s": {"train": train_digest, "val": val_digest},
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "environment": environment_snapshot(),
        "training_log_path": relative_project_path(context.project_root, log_path),
        "deployment_eligible": False,
        "conclusion_scope": "exploratory_multi_seed_confirmation_pending" if len(context.config.training.seeds) > 1 else "exploratory_single_seed",
    }
    summary_path = training_summary_path(context, seed=seed)
    write_json_if_absent_or_identical(summary_path, summary)
    log_event(
        logger,
        "run=complete",
        experiment="V27b",
        seed=seed,
        status=summary["status"],
        secondary_best_epoch=best["epoch"],
        elapsed=format_duration(time.perf_counter() - started),
        summary_path=str(summary_path),
    )
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["train_v27b"]
