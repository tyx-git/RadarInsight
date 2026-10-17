"""Training and checkpoint loading for V27d-r16."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    write_torch_if_absent_or_identical,
)
from v27.v27_baseline.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from v27.v27_baseline.protocol import V27_L2_TASK_NAME, task_class_weights
from v27.v27_baseline.runtime import environment_snapshot, seed_v27

from .data import (
    R16L2Dataset,
    build_r16_dataloader,
    r16_frame_time_kwargs,
    r16_model_input_kwargs,
)
from .model import R16IntegratedL2Model, build_r16_model, tensor_state_hash, trainable_parameter_groups
from .protocol import (
    R16_ARMS,
    R16_CHECKPOINT_SCHEMA,
    R16_SEEDS,
    R16_TRAINING_SUMMARY_SCHEMA,
    R16Context,
    artifact_path,
    read_sealed_json,
    write_sealed_json_if_identical,
)


def _run_path(
    context: R16Context,
    template_name: str,
    *,
    arm: str,
    seed: int,
    epoch: int | None = None,
) -> Path:
    return artifact_path(
        context.project_root,
        context.config.artifacts[template_name],
        arm=arm,
        seed=seed,
        epoch=epoch,
    )


def _device(context: R16Context) -> torch.device:
    if context.config.training["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r16 requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _checkpoint_payload(
    context: R16Context,
    *,
    model: R16IntegratedL2Model,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    initial_state_sha256: str,
    optimizer: torch.optim.Optimizer | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": R16_CHECKPOINT_SCHEMA,
        "experiment_id": "V27d-r16",
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "r16_manifest_sha256": context.manifest_sha256,
        "r14_source_checkpoint_sha256": context.config.source["expected_hashes"][f"r14_control_seed{seed}_checkpoint"],
        "v27_config_sha256": context.baseline_config.config_sha256,
        "data_registry_sha256": context.data_binding["data_registry_sha256"],
        "class_names": list(model.class_names),
        "loss_name": "parent_final_fixed_parent_count_ce",
        "loss_reduction": "fixed_parent_count",
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "initial_state_sha256": initial_state_sha256,
        "metrics": dict(metrics),
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
    }


def load_checkpoint(
    context: R16Context,
    *,
    arm: str,
    seed: int,
    epoch: int,
    device: torch.device,
) -> tuple[R16IntegratedL2Model, dict[str, Any]]:
    if arm not in R16_ARMS or int(seed) not in R16_SEEDS:
        raise ValueError("V27d-r16 checkpoint arm or seed is invalid")
    path = _run_path(context, "checkpoint_template", arm=arm, seed=seed, epoch=epoch)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r16 checkpoint is not a mapping")
    model = build_r16_model(context, arm=arm, seed=seed, device=device)
    contract = model.model_contract()
    if (
        payload.get("checkpoint_schema") != R16_CHECKPOINT_SCHEMA
        or payload.get("experiment_id") != "V27d-r16"
        or payload.get("arm") != arm
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", -1)) != int(epoch)
        or payload.get("r16_manifest_sha256") != context.manifest_sha256
        or payload.get("model_contract") != contract
        or payload.get("model_contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("V27d-r16 checkpoint identity/contract differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27d-r16 checkpoint state hash differs")
    model.load_state_dict(dict(state), strict=True)
    return model, dict(payload)


def _write_reused_control_checkpoint(
    context: R16Context,
    *,
    model: R16IntegratedL2Model,
    seed: int,
    logger: logging.Logger,
    log_path: Path,
) -> dict[str, Any]:
    epoch = int(context.config.training["primary_epoch"])
    initial_hash = tensor_state_hash(model.state_dict())
    payload = _checkpoint_payload(
        context,
        model=model,
        arm="control",
        seed=seed,
        epoch=epoch,
        global_step=0,
        initial_state_sha256=initial_hash,
        optimizer=None,
        metrics={"source": "V27d-r14-control-epoch020"},
    )
    checkpoint_path = _run_path(
        context,
        "checkpoint_template",
        arm="control",
        seed=seed,
        epoch=epoch,
    )
    write_torch_if_absent_or_identical(checkpoint_path, payload)
    summary = {
        "schema_version": R16_TRAINING_SUMMARY_SCHEMA,
        "status": "reused_frozen_r14_control",
        "experiment_id": "V27d-r16",
        "arm": "control",
        "seed": int(seed),
        "selected_epoch": epoch,
        "initial_state_sha256": initial_hash,
        "model_contract": model.model_contract(),
        "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
        "training_log_path": relative_project_path(context.project_root, log_path),
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
        "environment": environment_snapshot(),
    }
    summary_path = _run_path(
        context,
        "training_summary_template",
        arm="control",
        seed=seed,
    )
    write_sealed_json_if_identical(summary_path, summary)
    log_event(
        logger,
        "r16_control_reused",
        seed=seed,
        source="V27d-r14-control-epoch020",
        checkpoint_path=relative_project_path(context.project_root, checkpoint_path),
    )
    return {**summary, "summary_path": str(summary_path)}


def _train_treatment(
    context: R16Context,
    *,
    seed: int,
    logger: logging.Logger,
    log_path: Path,
) -> dict[str, Any]:
    device = _device(context)
    seed_v27(seed, deterministic_cuda=bool(context.config.training["cuda_deterministic_algorithms"]))
    train_set = R16L2Dataset(context, split="train")
    validation_set = R16L2Dataset(context, split="val")
    loader = build_r16_dataloader(
        train_set,
        batch_size=int(context.config.training["batch_size"]),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    model = build_r16_model(context, arm="treatment", seed=seed, device=device)
    groups = trainable_parameter_groups(model)
    initial_state_hash = tensor_state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(context.config.training["learning_rate"]),
        weight_decay=float(context.config.training["weight_decay"]),
    )
    weights = torch.tensor(
        task_class_weights(context.baseline_config, V27_L2_TASK_NAME),
        dtype=torch.float32,
        device=device,
    )
    epochs = int(context.config.training["epochs"])
    steps_per_epoch = len(loader)
    interval = progress_interval(steps_per_epoch)
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    log_event(
        logger,
        "r16_treatment_run=start",
        experiment="V27d-r16",
        arm="treatment",
        seed=seed,
        device=str(device),
        train_parents=len(train_set),
        validation_parents=len(validation_set),
        batch_size=int(context.config.training["batch_size"]),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        trainable_parameters=sum(parameter.numel() for parameter in groups),
        loss="parent_final_fixed_parent_count_ce",
        loss_reduction="fixed_parent_count",
        unknown=0,
        test=0,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        model.frozen_reference_base_model.eval()
        model.stable_model.eval()
        if hasattr(model.trainable_base_model.rd_encoder, "network"):
            model.trainable_base_model.rd_encoder.network.eval()
            model.trainable_base_model.rd_encoder.projection.eval()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        temporal_events = 0
        positive_deltas = 0
        zero_deltas = 0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                base_inputs=r16_model_input_kwargs(batch, device),
                **r16_frame_time_kwargs(batch, device),
                **{
                    "cached_v27_base_logits": batch["cached_v27_base_logits"].to(device=device, non_blocking=True),
                    "stable_summary_tokens": batch["stable_summary_tokens"].to(device=device, non_blocking=True),
                    "stable_summary_valid": batch["stable_summary_valid"].to(device=device, non_blocking=True),
                    "stable_frame_values": batch["stable_frame_values"].to(device=device, non_blocking=True),
                    "stable_frame_mask": batch["stable_frame_mask"].to(device=device, non_blocking=True),
                },
            )
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["labels"].to(device),
                batch["tr_padding_mask"].to(device),
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError("V27d-r16 treatment loss is non-finite")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    groups,
                    float(context.config.training["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                ).item()
            )
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, gradient_norm)
            clipped_steps += int(gradient_norm > float(context.config.training["gradient_clip_norm"]))
            diagnostics = output["rd_time_diagnostics"]
            temporal_events += int(diagnostics.get("temporal_eligible_event_count", 0))
            positive_deltas += int(diagnostics.get("positive_delta_count", 0))
            zero_deltas += int(diagnostics.get("zero_delta_count", 0))
            if step == 1 or step == steps_per_epoch or step % interval == 0:
                log_event(
                    logger,
                    "r16_treatment_progress",
                    epoch=f"{epoch:02d}/{epochs:02d}",
                    step=f"{step:04d}/{steps_per_epoch:04d}",
                    global_step=global_step,
                    loss=losses[-1],
                    gradient_norm=gradient_norm,
                    temporal_events=temporal_events,
                    positive_frame_deltas=positive_deltas,
                    elapsed=format_duration(time.perf_counter() - started),
                )
        from .evaluation import evaluate_model

        evaluation = evaluate_model(
            context,
            model=model,
            dataset=validation_set,
            device=device,
            checkpoint_epoch=epoch,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": gradient_max,
            "gradient_clipped_steps": clipped_steps,
            "temporal_eligible_event_count": temporal_events,
            "positive_frame_delta_count": positive_deltas,
            "zero_frame_delta_count": zero_deltas,
            "validation_parent_metrics": evaluation["parent_metrics"],
            "validation_parent_balanced_eligible_prefix_metrics": evaluation["parent_balanced_eligible_prefix_metrics"],
            "rd_time_usage": evaluation["rd_time_usage"],
        }
        history.append(record)
        payload = _checkpoint_payload(
            context,
            model=model,
            arm="treatment",
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            initial_state_sha256=initial_state_hash,
            optimizer=optimizer,
            metrics={"train_loss": record["train_loss"]},
        )
        checkpoint_path = _run_path(
            context,
            "checkpoint_template",
            arm="treatment",
            seed=seed,
            epoch=epoch,
        )
        write_torch_if_absent_or_identical(checkpoint_path, payload)
        log_event(
            logger,
            "r16_treatment_epoch=complete",
            epoch=f"{epoch:02d}/{epochs:02d}",
            train_loss=record["train_loss"],
            gradient_norm_max=gradient_max,
            temporal_events=temporal_events,
            checkpoint_path=relative_project_path(context.project_root, checkpoint_path),
            elapsed=format_duration(time.perf_counter() - started),
        )
        log_metric_set(logger, "validation_parent_final", evaluation["parent_metrics"])
    summary = {
        "schema_version": R16_TRAINING_SUMMARY_SCHEMA,
        "status": "completed_fixed_epoch_temporal_rd_training",
        "experiment_id": "V27d-r16",
        "arm": "treatment",
        "seed": int(seed),
        "selected_epoch": epochs,
        "initial_state_sha256": initial_state_hash,
        "history": history,
        "final_metrics": history[-1]["validation_parent_metrics"],
        "model_contract": model.model_contract(),
        "r16_manifest_sha256": context.manifest_sha256,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "training_log_path": relative_project_path(context.project_root, log_path),
        "environment": environment_snapshot(),
        "deployment_eligible": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_path = _run_path(
        context,
        "training_summary_template",
        arm="treatment",
        seed=seed,
    )
    write_sealed_json_if_identical(summary_path, summary)
    log_event(
        logger,
        "r16_treatment_run=complete",
        seed=seed,
        status=summary["status"],
        summary_path=relative_project_path(context.project_root, summary_path),
        elapsed=format_duration(time.perf_counter() - started),
    )
    return {**summary, "summary_path": str(summary_path)}


def train_r16_arm(context: R16Context, *, arm: str, seed: int) -> dict[str, Any]:
    if arm not in R16_ARMS or int(seed) not in R16_SEEDS:
        raise ValueError("V27d-r16 arm or seed is invalid")
    summary_path = _run_path(
        context,
        "training_summary_template",
        arm=arm,
        seed=seed,
    )
    if summary_path.exists():
        return read_sealed_json(summary_path, expected_schema=R16_TRAINING_SUMMARY_SCHEMA)
    log_path = _run_path(context, "training_log_template", arm=arm, seed=seed)
    logger = configure_training_logger(
        log_path,
        logger_name=f"radar_ln3.v27d.r16.{arm}.seed_{seed}",
    )
    try:
        if arm == "control":
            model = build_r16_model(
                context,
                arm="control",
                seed=seed,
                device=torch.device("cpu"),
            )
            return _write_reused_control_checkpoint(
                context,
                model=model,
                seed=seed,
                logger=logger,
                log_path=log_path,
            )
        return _train_treatment(context, seed=seed, logger=logger, log_path=log_path)
    except Exception:
        logger.exception("run=failed | experiment=V27d-r16 | arm=%s | seed=%d", arm, seed)
        raise
    finally:
        close_training_logger(logger)


__all__ = ["load_checkpoint", "train_r16_arm"]