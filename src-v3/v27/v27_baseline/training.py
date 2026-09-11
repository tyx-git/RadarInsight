"""Task-local V27 known-class classification training.

Knownness and quality heads are intentionally *not* optimised here.  They are
registered in the model to make the representation boundary auditable, while
their supervision/calibration is deferred until an approved data-role and
label contract exists.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import relative_project_path

from .checkpoint import save_v27_checkpoint
from .data import V27TaskDataset, build_v27_dataloader, model_input_kwargs
from .evaluation import assess_v27_checkpoint, evaluate_v27, is_better_v27_checkpoint
from .logging_utils import (
    TRAINING_LOGGER_PREFIX,
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from .protocol import (
    V27_L1_TASK_NAME,
    V27_L2_TASK_NAME,
    task_class_weights,
    v27_artifact_path,
    v27_training_log_path,
)
from .runtime import (
    V27Context,
    build_v27_model,
    configured_device,
    environment_snapshot,
    seed_v27,
    write_v27_sealed_json,
)


def _path(context: V27Context, field: str, *, task: str, seed: int) -> Path:
    return v27_artifact_path(
        context.project_root,
        getattr(context.config.artifacts, field),
        seed,
        task=task,
    )


def train_v27_task(
    context: V27Context,
    *,
    task: str,
    seed: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
        raise ValueError("V27 training requires known_gate or dbb_specialist")
    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    if actual_seed not in context.config.training.seeds:
        raise ValueError("V27 requested seed is not registered")
    total_epochs = context.config.training.epochs if epochs_override is None else int(epochs_override)
    if total_epochs <= 0 or total_epochs > context.config.training.epochs:
        raise ValueError("V27 epochs_override must stay within configured budget")
    log_path = v27_training_log_path(
        context.project_root,
        context.config,
        task=task,
        seed=actual_seed,
    )
    logger = configure_training_logger(
        log_path,
        logger_name=f"{TRAINING_LOGGER_PREFIX}.{task}.seed_{actual_seed}",
    )
    started = time.perf_counter()
    log_event(
        logger,
        "run=initializing",
        experiment="V27",
        task=task,
        seed=actual_seed,
        epochs=total_epochs,
        log_path=str(log_path),
    )
    try:
        return _train_v27_task(
            context,
            task=task,
            actual_seed=actual_seed,
            total_epochs=total_epochs,
            logger=logger,
            log_path=log_path,
            started=started,
        )
    except Exception:
        logger.exception(
            "run=failed | experiment=V27 | task=%s | seed=%d | elapsed=%s",
            task,
            actual_seed,
            format_duration(time.perf_counter() - started),
        )
        raise
    finally:
        close_training_logger(logger)


def _train_v27_task(
    context: V27Context,
    *,
    task: str,
    actual_seed: int,
    total_epochs: int,
    logger: logging.Logger,
    log_path: Path,
    started: float,
) -> dict[str, Any]:
    """Run the logged known-class classification stage after boundary checks."""

    device = configured_device(context.config)
    seed_v27(
        actual_seed,
        deterministic_cuda=context.config.training.cuda_deterministic_algorithms,
    )
    train_set = V27TaskDataset(
        context.project_root,
        context.config,
        task=task,
        split_name=context.config.data.train_split_name,
    )
    validation_set = V27TaskDataset(
        context.project_root,
        context.config,
        task=task,
        split_name=context.config.data.validation_split_name,
    )
    expected_train = context.data_binding["task_split_details"][task]["train"][
        "parent_label_counts"
    ]
    if train_set.label_counts() != expected_train:
        raise ValueError("V27 training dataset support differs from registry")
    generator = torch.Generator(device="cpu").manual_seed(
        context.config.training.dataloader_seed
    )
    loader = build_v27_dataloader(
        train_set,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = build_v27_model(context, task=task, device=device)
    # Classification-only baseline stage: leave the independent heads at their
    # initial parameters so a later audited knownness/quality phase has not
    # already received classification gradients through a shared objective.
    for parameter in model.trajectory_heads.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.trajectory_heads.parameters()):
        raise AssertionError(
            "V27 classification stage must freeze knownness/quality heads"
        )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("V27 classification stage has no trainable classification parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    weights = torch.tensor(
        task_class_weights(context.config, task),
        dtype=torch.float32,
        device=device,
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    global_step = 0
    steps_per_epoch = len(loader)
    log_event(
        logger,
        "run=start",
        experiment="V27",
        task=task,
        seed=actual_seed,
        device=str(device),
        train_parents=len(train_set),
        validation_parents=len(validation_set),
        batch_size=context.config.training.batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        loss=context.config.training.loss_name,
        loss_reduction=context.config.training.loss_reduction,
        knownness_training="not_executed",
        quality_training="deferred_unfitted",
        unknown=0,
        test=0,
        log_path=str(log_path),
    )
    interval = progress_interval(steps_per_epoch)
    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_input_kwargs(batch, device), return_trace=False)
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["labels"].to(device),
                batch["tr_padding_mask"].to(device),
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("V27 classification loss is non-finite")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    context.config.training.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("V27 classification gradient is non-finite")
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, gradient_norm)
            clipped_steps += int(
                gradient_norm > context.config.training.gradient_clip_norm
            )
            if step == 1 or step == steps_per_epoch or step % interval == 0:
                log_event(
                    logger,
                    "progress",
                    task=task,
                    epoch=f"{epoch:02d}/{total_epochs:02d}",
                    step=f"{step:04d}/{steps_per_epoch:04d}",
                    global_step=global_step,
                    loss=float(loss.detach().item()),
                    gradient_norm=gradient_norm,
                    elapsed=format_duration(time.perf_counter() - epoch_started),
                )
        evaluation = evaluate_v27(
            model,
            validation_set,
            config=context.config,
            device=device,
            bootstrap_replicates=0,
            checkpoint_epoch=epoch,
        )
        assessment = assess_v27_checkpoint(evaluation, epoch=epoch)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": gradient_max,
            "gradient_clipped_steps": clipped_steps,
            "validation_parent_metrics": evaluation["parent_metrics"],
            "validation_parent_balanced_eligible_prefix_metrics": evaluation[
                "parent_balanced_eligible_prefix_metrics"
            ],
            "checkpoint_assessment": assessment,
            "knownness_quality_training": (
                "not_executed; heads are not included in classification objective"
            ),
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
        }
        history.append(record)
        metrics = {"train_loss": record["train_loss"], "checkpoint_assessment": assessment}
        last_checkpoint = save_v27_checkpoint(
            _path(context, "last_checkpoint_template", task=task, seed=actual_seed),
            project_root=context.project_root,
            config=context.config,
            model=model,
            optimizer=optimizer,
            manifest=context.manifest,
            data_binding=context.data_binding,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            kind="last",
            seed=actual_seed,
        )
        log_event(
            logger,
            "checkpoint=saved",
            kind="last",
            epoch=epoch,
            path=str(last_checkpoint),
        )
        if context.config.training.save_every_epoch:
            epoch_checkpoint = save_v27_checkpoint(
                _path(
                    context,
                    "epoch_checkpoint_dir_template",
                    task=task,
                    seed=actual_seed,
                )
                / f"epoch-{epoch:03d}.pt",
                project_root=context.project_root,
                config=context.config,
                model=model,
                optimizer=optimizer,
                manifest=context.manifest,
                data_binding=context.data_binding,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                kind="epoch",
                seed=actual_seed,
            )
            log_event(
                logger,
                "checkpoint=saved",
                kind="epoch",
                epoch=epoch,
                path=str(epoch_checkpoint),
            )
        if is_better_v27_checkpoint(assessment, best):
            best = assessment
            best_checkpoint = save_v27_checkpoint(
                _path(context, "best_checkpoint_template", task=task, seed=actual_seed),
                project_root=context.project_root,
                config=context.config,
                model=model,
                optimizer=optimizer,
                manifest=context.manifest,
                data_binding=context.data_binding,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                kind="best",
                seed=actual_seed,
            )
            log_event(
                logger,
                "checkpoint=saved",
                kind="best",
                epoch=epoch,
                path=str(best_checkpoint),
            )
        log_event(
            logger,
            "epoch=complete",
            task=task,
            epoch=f"{epoch:02d}/{total_epochs:02d}",
            global_step=global_step,
            train_loss=record["train_loss"],
            gradient_norm_max=gradient_max,
            gradient_clipped_steps=clipped_steps,
            best_epoch=best["epoch"] if best is not None else "n/a",
            selection_minimum=assessment[
                "minimum_per_class_precision_recall_f1"
            ],
            selection_macro_f1=assessment["macro_f1"],
            elapsed=format_duration(time.perf_counter() - epoch_started),
        )
        log_metric_set(logger, "validation_parent_final", evaluation["parent_metrics"])
        log_metric_set(
            logger,
            "validation_parent_balanced_eligible_prefix",
            evaluation["parent_balanced_eligible_prefix_metrics"],
        )
    summary = {
        "status": "completed_known_class_classification_only",
        "schema_version": "RADAR-LN3-V27-task-training-summary-v2",
        "experiment_id": "V27",
        "task": task,
        "seed": actual_seed,
        "epochs": total_epochs,
        "history": history,
        "best_assessment": best,
        "environment": environment_snapshot(),
        "knownness_training": "not_executed",
        "quality_training": "deferred_unfitted_no_pseudo_labels",
        "l3_calibration": "unfitted",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
        "training_log_path": relative_project_path(context.project_root, log_path),
    }
    path = _path(context, "training_summary_template", task=task, seed=actual_seed)
    write_v27_sealed_json(path, summary)
    log_event(
        logger,
        "run=complete",
        experiment="V27",
        task=task,
        seed=actual_seed,
        status=summary["status"],
        best_epoch=best["epoch"] if best is not None else "n/a",
        elapsed=format_duration(time.perf_counter() - started),
        summary_path=str(path),
    )
    return {**summary, "summary_path": str(path)}


def run_v27_training_step(
    context: V27Context,
    *,
    task: str,
    parent_limit: int = 2,
) -> dict[str, Any]:
    """Run one CPU classification update for a fast training-contract check.

    This is deliberately not an experiment run: it writes no checkpoint,
    summary or calibration artifact.  It exercises the same data collation,
    model, loss, frozen-head boundary and optimizer step used by ``train`` so
    a command-line smoke check can expose train-only regressions quickly.
    """

    if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
        raise ValueError("V27 training step requires known_gate or dbb_specialist")
    if int(parent_limit) < 2:
        raise ValueError("V27 training step requires at least two parents")
    seed_v27(int(context.config.training.seeds[0]))
    dataset = V27TaskDataset(
        context.project_root,
        context.config,
        task=task,
        split_name=context.config.data.train_split_name,
        parent_limit=int(parent_limit),
    )
    batch = next(
        iter(
            build_v27_dataloader(
                dataset,
                batch_size=int(parent_limit),
                shuffle=False,
            )
        )
    )
    device = torch.device("cpu")
    model = build_v27_model(context, task=task, device=device)
    for parameter in model.trajectory_heads.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )
    output = model(**model_input_kwargs(batch, device), return_trace=False)
    weights = torch.tensor(
        task_class_weights(context.config, task), dtype=torch.float32, device=device
    )
    loss = parent_final_fixed_parent_count_ce(
        output["sequence_logits"],
        batch["labels"].to(device),
        batch["tr_padding_mask"].to(device),
        class_weights=weights,
    )["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("V27 training contract loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        context.config.training.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if any(parameter.grad is not None for parameter in model.trajectory_heads.parameters()):
        raise AssertionError("V27 frozen trajectory heads received classification gradients")
    optimizer.step()
    return {
        "task": task,
        "parent_count": len(dataset),
        "loss": float(loss.detach().item()),
        "knownness_quality_heads_frozen": True,
    }


__all__ = ["run_v27_training_step", "train_v27_task"]
