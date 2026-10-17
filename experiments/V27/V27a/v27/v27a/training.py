"""Residual-branch training for the frozen V27a comparison arms."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import relative_project_path
from v27.v27_baseline.protocol import task_class_weights

from .cache import V27AEvidenceCacheDataset, build_cache_dataloader
from .checkpoint import save_v27a_checkpoint
from .evaluation import assess_v27a_checkpoint, evaluate_v27a_arm
from .logging_utils import (
    LOGGER_PREFIX,
    close_logger,
    configure_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from .model import V27AEvidenceResidual, build_v27a_model
from .protocol import V27A_ARMS, V27A_L2_CLASS_NAMES, v27a_artifact_path
from .runtime import (
    V27AContext,
    cache_path,
    cache_payload_digest,
    configured_device,
    environment_snapshot,
    load_verified_cache,
    seed_v27a,
    write_json_if_absent_or_identical,
)


_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27a-residual-training-summary-v2"


def _run_path(
    context: V27AContext,
    template: str,
    *,
    arm: str,
    seed: int,
    epoch: int | None = None,
) -> Path:
    return v27a_artifact_path(
        context.project_root,
        template,
        arm=arm,
        seed=seed,
        epoch=epoch,
    )


def _global_best_path(context: V27AContext, *, arm: str, seed: int) -> Path:
    epoch_path = _run_path(
        context,
        context.config.artifacts.epoch_checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=1,
    )
    return epoch_path.with_name("global-best.pt")


def _validate_class_weights(context: V27AContext) -> torch.Tensor:
    configured = tuple(float(value) for value in task_class_weights(
        context.v27_config, "dbb_specialist"
    ))
    bound = context.frozen_binding.get("l2", {}).get("class_weights")
    if not isinstance(bound, (list, tuple)) or tuple(float(value) for value in bound) != configured:
        raise ValueError(
            "V27a frozen binding class weights differ from the current V27 L2 contract"
        )
    if len(configured) != len(V27A_L2_CLASS_NAMES) or any(
        not np.isfinite(value) or value <= 0.0 for value in configured
    ):
        raise ValueError("V27a L2 class weights are malformed")
    return torch.tensor(configured, dtype=torch.float32)


def _batch_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "baseline_logits",
        "summary_tokens",
        "summary_valid",
        "frame_values",
        "frame_mask",
        "padding_mask",
    )
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27a cache batch field is not a tensor: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    return result


def _assert_trainable_boundary(model: V27AEvidenceResidual) -> None:
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("V27a residual model has no trainable parameters")
    for parameter in model.parameters():
        if not parameter.requires_grad:
            raise ValueError("V27a residual training unexpectedly froze a model parameter")


def _update_diagnostics(
    diagnostics: dict[str, float | int],
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> None:
    residual = output["residual_logits"].detach()
    increment = output["residual_increment"].detach()
    evidence = output["evidence_update"].detach()
    padding = batch["padding_mask"].to(device=residual.device)
    reliable = batch["summary_valid"].to(device=residual.device)
    active = ~padding
    diagnostics["residual_abs_max"] = max(
        float(diagnostics["residual_abs_max"]),
        float(residual.abs().masked_select(active.unsqueeze(-1)).max().item())
        if bool(active.any())
        else 0.0,
    )
    diagnostics["residual_zero_sum_max"] = max(
        float(diagnostics["residual_zero_sum_max"]),
        float(
            residual.sum(dim=-1)
            .masked_select(active)
            .abs()
            .max()
            .item()
        )
        if bool(active.any())
        else 0.0,
    )
    no_reliable = active & ~reliable
    diagnostics["no_reliable_event_count"] = int(
        diagnostics["no_reliable_event_count"]
    ) + int(no_reliable.sum().item())
    diagnostics["no_reliable_increment_nonzero_count"] = int(
        diagnostics["no_reliable_increment_nonzero_count"]
    ) + int(
        (increment.abs().amax(dim=-1) > 0.0)
        .masked_select(no_reliable)
        .sum()
        .item()
    )
    diagnostics["no_reliable_evidence_nonzero_count"] = int(
        diagnostics["no_reliable_evidence_nonzero_count"]
    ) + int(
        (evidence.abs().amax(dim=-1) > 0.0)
        .masked_select(no_reliable)
        .sum()
        .item()
    )


def _new_diagnostics() -> dict[str, float | int]:
    return {
        "residual_abs_max": 0.0,
        "residual_zero_sum_max": 0.0,
        "no_reliable_event_count": 0,
        "no_reliable_increment_nonzero_count": 0,
        "no_reliable_evidence_nonzero_count": 0,
    }


def train_v27a_arm(
    context: V27AContext,
    *,
    arm: str,
    seed: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one independent residual arm against cached frozen V27 logits."""

    if arm not in V27A_ARMS:
        raise ValueError(f"V27a training arm must be mechanism or main: {arm}")
    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    if actual_seed not in context.config.training.seeds:
        raise ValueError("V27a training seed is not registered")
    total_epochs = (
        context.config.training.epochs
        if epochs_override is None
        else epochs_override
    )
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise ValueError("V27a epochs_override must be an integer")
    if total_epochs <= 0 or total_epochs > context.config.training.epochs:
        raise ValueError("V27a epochs_override exceeds the configured budget")
    if context.config.training.primary_epoch > total_epochs:
        raise ValueError("V27a primary_epoch is outside the requested training range")

    log_path = _run_path(
        context,
        context.config.artifacts.training_log_template,
        arm=arm,
        seed=actual_seed,
    )
    logger = configure_logger(
        log_path,
        logger_name=f"{LOGGER_PREFIX}.train.{arm}.seed_{actual_seed}",
    )
    started = time.perf_counter()
    log_event(
        logger,
        "run=initializing",
        experiment="V27a",
        arm=arm,
        seed=actual_seed,
        epochs=total_epochs,
        log_path=str(log_path),
    )
    try:
        return _train_v27a_arm(
            context,
            arm=arm,
            seed=actual_seed,
            total_epochs=total_epochs,
            logger=logger,
            log_path=log_path,
            started=started,
        )
    except Exception:
        logger.exception(
            "run=failed | experiment=V27a | arm=%s | seed=%d | elapsed=%s",
            arm,
            actual_seed,
            format_duration(time.perf_counter() - started),
        )
        raise
    finally:
        close_logger(logger)


def _train_v27a_arm(
    context: V27AContext,
    *,
    arm: str,
    seed: int,
    total_epochs: int,
    logger: logging.Logger,
    log_path: Path,
    started: float,
) -> dict[str, Any]:
    device = configured_device(context)
    seed_v27a(seed)
    # Loading both roles here verifies the complete cache schema before the
    # first optimizer update and prevents a late validation-cache failure.
    train_payload = load_verified_cache(context, "train")
    validation_payload = load_verified_cache(context, "val")
    cache_payload_sha256s = {
        "train": cache_payload_digest(train_payload),
        "val": cache_payload_digest(validation_payload),
    }
    train_set = V27AEvidenceCacheDataset(cache_path(context, "train"))
    validation_set = V27AEvidenceCacheDataset(cache_path(context, "val"))
    if len(train_set) != len(train_payload["parent_sample_ids"]):
        raise ValueError("V27a training cache dataset length changed during loading")
    if len(validation_set) != len(validation_payload["parent_sample_ids"]):
        raise ValueError("V27a validation cache dataset length changed during loading")
    generator = torch.Generator(device="cpu").manual_seed(
        int(context.config.training.dataloader_seed)
    )
    loader = build_cache_dataloader(
        train_set,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = build_v27a_model(
        context.config,
        arm=arm,
        seed=seed,
    ).to(device)
    _assert_trainable_boundary(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    class_weights = _validate_class_weights(context).to(device=device)
    history: list[dict[str, Any]] = []
    best_assessment: dict[str, Any] | None = None
    global_step = 0
    steps_per_epoch = len(loader)
    if steps_per_epoch <= 0:
        raise ValueError("V27a training cache contains no batches")
    log_event(
        logger,
        "run=start",
        experiment="V27a",
        arm=arm,
        seed=seed,
        device=str(device),
        train_parents=len(train_set),
        validation_parents=len(validation_set),
        batch_size=context.config.training.batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        loss="parent_final_fixed_parent_count_ce",
        loss_reduction="fixed_parent_count",
        input_cache="materialized_frozen_logit_only",
        train_cache_sha256=cache_payload_sha256s["train"],
        validation_cache_sha256=cache_payload_sha256s["val"],
        unknown=0,
        test=0,
    )
    interval = progress_interval(steps_per_epoch)

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        diagnostics = _new_diagnostics()
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            inputs = _batch_inputs(batch, device)
            output = model(**inputs)
            if not torch.equal(output["baseline_logits"], inputs["baseline_logits"]):
                raise AssertionError("V27a residual arm altered frozen baseline logits")
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["label"].to(device=device),
                inputs["padding_mask"],
                class_weights=class_weights,
            )["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError("V27a residual loss is non-finite")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    context.config.training.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("V27a residual gradient is non-finite")
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, gradient_norm)
            clipped_steps += int(
                gradient_norm > context.config.training.gradient_clip_norm
            )
            _update_diagnostics(diagnostics, output, inputs)
            if step == 1 or step == steps_per_epoch or step % interval == 0:
                log_event(
                    logger,
                    "progress",
                    epoch=f"{epoch:02d}/{total_epochs:02d}",
                    step=f"{step:04d}/{steps_per_epoch:04d}",
                    global_step=global_step,
                    loss=float(loss.detach().item()),
                    gradient_norm=gradient_norm,
                    elapsed=format_duration(time.perf_counter() - epoch_started),
                )

        validation = evaluate_v27a_arm(
            context,
            arm=arm,
            seed=seed,
            split_name="val",
            model=model,
            checkpoint_epoch=epoch,
            bootstrap_replicates=0,
            device=device,
        )
        assessment = assess_v27a_checkpoint(validation, epoch=epoch)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": float(gradient_max),
            "gradient_clipped_steps": int(clipped_steps),
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
        metrics = {
            "train_loss": record["train_loss"],
            "checkpoint_assessment": assessment,
            "diagnostics": dict(diagnostics),
        }
        epoch_path = _run_path(
            context,
            context.config.artifacts.epoch_checkpoint_template,
            arm=arm,
            seed=seed,
            epoch=epoch,
        )
        save_v27a_checkpoint(
            epoch_path,
            context=context,
            model=model,
            optimizer=optimizer,
            arm=arm,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            kind="epoch",
            cache_payload_sha256s=cache_payload_sha256s,
        )
        log_event(logger, "checkpoint=saved", kind="epoch", epoch=epoch, path=str(epoch_path))

        if best_assessment is None or tuple(assessment["ranking_key"]) > tuple(
            best_assessment["ranking_key"]
        ):
            best_assessment = dict(assessment)

        log_event(
            logger,
            "epoch=complete",
            epoch=f"{epoch:02d}/{total_epochs:02d}",
            global_step=global_step,
            train_loss=record["train_loss"],
            gradient_norm_max=gradient_max,
            gradient_clipped_steps=clipped_steps,
            secondary_best_epoch="n/a",
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

    if best_assessment is None:
        raise RuntimeError("V27a training completed without a global validation assessment")
    selected_epoch = int(best_assessment["epoch"])
    selected_epoch_path = _run_path(
        context,
        context.config.artifacts.epoch_checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=selected_epoch,
    )
    selected_payload = torch.load(selected_epoch_path, map_location="cpu", weights_only=False)
    if not isinstance(selected_payload, Mapping) or not isinstance(selected_payload.get("model_state"), Mapping):
        raise ValueError("V27a global-best checkpoint payload is invalid")
    model.load_state_dict(selected_payload["model_state"], strict=True)
    global_best_path = _global_best_path(context, arm=arm, seed=seed)
    save_v27a_checkpoint(
        global_best_path,
        context=context,
        model=model,
        optimizer=None,
        arm=arm,
        seed=seed,
        epoch=selected_epoch,
        global_step=int(selected_payload.get("global_step", 0)),
        metrics={"checkpoint_assessment": best_assessment, "selection": "global_validation_per_class_f1"},
        kind="selected",
        cache_payload_sha256s=cache_payload_sha256s,
    )
    primary_path = _run_path(
        context,
        context.config.artifacts.epoch_checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=context.config.training.primary_epoch,
    )
    if not primary_path.is_file():
        raise FileNotFoundError(f"V27a primary checkpoint was not written: {primary_path}")
    summary = {
        "schema_version": _TRAINING_SUMMARY_SCHEMA,
        "experiment_id": "V27a",
        "status": "completed_frozen_logit_residual_training",
        "arm": arm,
        "seed": seed,
        "epochs": total_epochs,
        "primary_epoch": context.config.training.primary_epoch,
        "selected_epoch": selected_epoch,
        "selection": "global_validation_per_class_f1",
        "primary_checkpoint_path": relative_project_path(
            context.project_root, primary_path
        ),
        "global_best_assessment": best_assessment,
        "global_best_checkpoint_path": relative_project_path(context.project_root, global_best_path),
        "history": history,
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "class_weights": _validate_class_weights(context).tolist(),
        "loss_name": "parent_final_fixed_parent_count_ce",
        "loss_reduction": "fixed_parent_count",
        "input_cache_role": "materialized_frozen_logit_evidence_cache",
        "cache_payload_sha256s": cache_payload_sha256s,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "environment": environment_snapshot(),
        "training_log_path": relative_project_path(context.project_root, log_path),
        "deployment_eligible": False,
        "conclusion_scope": (
            "exploratory_single_seed"
            if len(context.config.training.seeds) == 1
            else "multi_seed_confirmation_pending"
        ),
    }
    summary_path = _run_path(
        context,
        context.config.artifacts.training_summary_template,
        arm=arm,
        seed=seed,
    )
    write_json_if_absent_or_identical(summary_path, summary)
    log_event(
        logger,
        "run=complete",
        experiment="V27a",
        arm=arm,
        seed=seed,
        status=summary["status"],
        secondary_best_epoch=best_assessment["epoch"],
        elapsed=format_duration(time.perf_counter() - started),
        summary_path=str(summary_path),
    )
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["train_v27a_arm"]