"""Paired anchored L2 training for V27d-r14."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import canonical_sha256, relative_project_path, write_torch_if_absent_or_identical
from v27.v27_baseline.logging_utils import close_training_logger, configure_training_logger, format_duration, log_event, log_metric_set, progress_interval
from v27.v27_baseline.protocol import V27_L2_TASK_NAME, task_class_weights
from v27.v27_baseline.runtime import environment_snapshot, seed_v27

from .data import R14L2Dataset, build_r14_dataloader, r14_model_input_kwargs, r14_sidecar_kwargs
from .model import R14IntegratedL2Model, build_r14_model, trainable_parameter_groups
from .protocol import R14_ARMS, R14_MODES, R14Context, artifact_path, read_sealed_json, write_sealed_json_if_identical


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    parts: list[Any] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r14 state entry is not a tensor: {name}")
        item = value.detach().cpu().contiguous()
        parts.extend((name, str(item.dtype), repr(tuple(item.shape)), item.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def _run_path(context: R14Context, template_name: str, *, arm: str, seed: int, epoch: int | None = None) -> Path:
    return artifact_path(context.project_root, context.config.artifacts[template_name], arm=arm, seed=seed, epoch=epoch)


def _device(context: R14Context) -> torch.device:
    if context.config.training["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r14 requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _checkpoint_payload(context: R14Context, *, model: R14IntegratedL2Model, arm: str, seed: int, epoch: int, global_step: int, initial_state_sha256: str, optimizer: torch.optim.Optimizer, metrics: Mapping[str, Any]) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": "RADAR-LN3-V27d-r14-density-weighted-checkpoint-v1",
        "experiment_id": "V27d-r14",
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "mode": R14_MODES[arm],
        "r14_manifest_sha256": context.manifest_sha256,
        "v27_config_sha256": context.baseline_config.config_sha256,
        "data_registry_sha256": context.data_binding["data_registry_sha256"],
        "loss_name": "parent_final_fixed_parent_count_ce",
        "loss_reduction": "fixed_parent_count",
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "initial_state_sha256": initial_state_sha256,
        "metrics": dict(metrics),
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
        "optimizer_state": optimizer.state_dict(),
        "scope": {"unknown_training_exposure": 0, "clutter_training_exposure": 0, "knownness": "not_in_scope", "quality": "not_in_scope", "l3": "not_in_scope", "test_payload_access_count": 0},
    }


def _materialize_reference_logits(context: R14Context, *, model: R14IntegratedL2Model, dataset: R14L2Dataset, device: torch.device, split: str) -> dict[str, torch.Tensor]:
    """Compute the frozen epoch-11 reference once per parent for this split."""

    rows: list[torch.Tensor] = []
    model.frozen_reference_base_model.eval()
    with torch.inference_mode():
        for batch in build_r14_dataloader(dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False):
            base_inputs = r14_model_input_kwargs(batch, device)
            logits = model.frozen_reference_base_model(**base_inputs, return_trace=False)["sequence_logits"]
            width = int(logits.shape[1])
            if width > 29:
                raise ValueError(f"V27d-r14 {split} reference width exceeds contract")
            padded = logits.new_zeros((logits.shape[0], 29, logits.shape[-1]))
            padded[:, :width] = logits
            rows.append(padded.detach().cpu())
    result = torch.cat(rows, dim=0)
    if result.shape != (len(dataset), 29, 3) or not torch.isfinite(result).all():
        raise ValueError(f"V27d-r14 {split} reference cache shape/value differs")
    return {parent_id: result[index] for index, parent_id in enumerate(dataset.parent_ids)}


def load_checkpoint(context: R14Context, *, arm: str, seed: int, epoch: int, device: torch.device) -> tuple[R14IntegratedL2Model, dict[str, Any]]:
    if arm not in R14_ARMS or seed not in (42, 44):
        raise ValueError("V27d-r14 checkpoint arm or seed is invalid")
    path = _run_path(context, "checkpoint_template", arm=arm, seed=seed, epoch=epoch)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r14 checkpoint is not a mapping")
    model = build_r14_model(context, arm=arm, seed=seed, device=device)
    if payload.get("checkpoint_schema") != "RADAR-LN3-V27d-r14-density-weighted-checkpoint-v1" or payload.get("experiment_id") != "V27d-r14" or payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed) or int(payload.get("epoch", -1)) != int(epoch) or payload.get("mode") != R14_MODES[arm] or payload.get("r14_manifest_sha256") != context.manifest_sha256 or payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("V27d-r14 checkpoint identity/contract differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27d-r14 checkpoint state hash differs")
    model.load_state_dict(dict(state), strict=True)
    return model, dict(payload)


def _train_arm(context: R14Context, *, arm: str, seed: int, logger: logging.Logger, log_path: Path) -> dict[str, Any]:
    device = _device(context)
    seed_v27(seed, deterministic_cuda=False)
    train_set = R14L2Dataset(context, split="train")
    validation_set = R14L2Dataset(context, split="val")
    loader = build_r14_dataloader(train_set, batch_size=int(context.config.training["batch_size"]), shuffle=True, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
    model = build_r14_model(context, arm=arm, seed=seed, device=device)
    groups = trainable_parameter_groups(model)
    initial_state_sha256 = tensor_state_hash(model.state_dict())
    train_reference_logits = _materialize_reference_logits(context, model=model, dataset=train_set, device=device, split="train")
    validation_reference_logits = _materialize_reference_logits(context, model=model, dataset=validation_set, device=device, split="val")
    optimizer = torch.optim.AdamW(groups, lr=float(context.config.training["learning_rate"]), weight_decay=float(context.config.training["weight_decay"]))
    weights = torch.tensor(task_class_weights(context.baseline_config, V27_L2_TASK_NAME), dtype=torch.float32, device=device)
    epochs = int(context.config.training["epochs"])
    global_step = 0
    history: list[dict[str, Any]] = []
    interval = progress_interval(len(loader))
    started = time.perf_counter()
    log_event(logger, "r14_l2_run=start", experiment="V27d-r14", arm=arm, seed=seed, mode=R14_MODES[arm], train_parents=len(train_set), validation_parents=len(validation_set), batch_size=int(context.config.training["batch_size"]), epochs=epochs, loss="parent_final_fixed_parent_count_ce", loss_reduction="fixed_parent_count", initial_state_sha256=initial_state_sha256, unknown=0, clutter=0, test=0)
    for epoch in range(1, epochs + 1):
        model.train()
        model.frozen_reference_base_model.eval()
        model.trainable_base_model.trajectory_heads.eval()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            base_inputs = r14_model_input_kwargs(batch, device)
            sidecars = r14_sidecar_kwargs(batch, device)
            reference_logits = torch.stack(
                [train_reference_logits[str(parent_id)] for parent_id in batch["parent_sample_ids"]],
                dim=0,
            )[:, :base_inputs["tr_padding_mask"].shape[1]].to(device=device, non_blocking=True)
            output = model(base_inputs=base_inputs, reference_sequence_logits=reference_logits, **sidecars)
            loss = parent_final_fixed_parent_count_ce(output["sequence_logits"], batch["labels"].to(device), base_inputs["tr_padding_mask"], class_weights=weights)["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError("V27d-r14 loss is non-finite")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(groups, float(context.config.training["gradient_clip_norm"]), error_if_nonfinite=True).item())
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, gradient_norm)
            clipped_steps += int(gradient_norm > float(context.config.training["gradient_clip_norm"]))
            if step == 1 or step == len(loader) or step % interval == 0:
                log_event(logger, "r14_l2_progress", arm=arm, epoch=f"{epoch:02d}/{epochs:02d}", step=f"{step:04d}/{len(loader):04d}", global_step=global_step, loss=losses[-1], gradient_norm=gradient_norm, elapsed=format_duration(time.perf_counter() - epoch_started))
        from .evaluation import evaluate_model
        evaluation = evaluate_model(context, model=model, dataset=validation_set, device=device, checkpoint_epoch=epoch, reference_logits=validation_reference_logits)
        record = {"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(losses)), "gradient_norm_max": gradient_max, "gradient_clipped_steps": clipped_steps, "validation_parent_metrics": evaluation["parent_metrics"], "validation_parent_balanced_eligible_prefix_metrics": evaluation["parent_balanced_eligible_prefix_metrics"], "mode": R14_MODES[arm], "unknown_payload_access_count": 0, "test_payload_access_count": 0}
        history.append(record)
        payload = _checkpoint_payload(context, model=model, arm=arm, seed=seed, epoch=epoch, global_step=global_step, initial_state_sha256=initial_state_sha256, optimizer=optimizer, metrics={"train_loss": record["train_loss"]})
        checkpoint_path = _run_path(context, "checkpoint_template", arm=arm, seed=seed, epoch=epoch)
        write_torch_if_absent_or_identical(checkpoint_path, payload)
        log_event(logger, "r14_l2_epoch=complete", arm=arm, epoch=f"{epoch:02d}/{epochs:02d}", train_loss=record["train_loss"], gradient_norm_max=gradient_max, elapsed=format_duration(time.perf_counter() - epoch_started))
        log_metric_set(logger, "validation_parent_final", evaluation["parent_metrics"])
    summary = {"schema_version": "RADAR-LN3-V27d-r14-density-weighted-training-summary-v1", "status": "completed_fixed_epoch_anchored_l2_retraining", "experiment_id": "V27d-r14", "arm": arm, "seed": int(seed), "mode": R14_MODES[arm], "selected_epoch": epochs, "initial_state_sha256": initial_state_sha256, "history": history, "final_metrics": history[-1]["validation_parent_metrics"], "model_contract": model.model_contract(), "r14_manifest_sha256": context.manifest_sha256, "unknown_training_exposure": 0, "clutter_training_exposure": 0, "test_payload_access_count": 0, "environment": environment_snapshot(), "training_log_path": relative_project_path(context.project_root, log_path), "deployment_eligible": False, "elapsed_seconds": float(time.perf_counter() - started)}
    summary_path = _run_path(context, "training_summary_template", arm=arm, seed=seed)
    write_sealed_json_if_identical(summary_path, summary)
    log_event(logger, "r14_l2_run=complete", arm=arm, seed=seed, status=summary["status"], summary_path=relative_project_path(context.project_root, summary_path))
    return {**summary, "summary_path": str(summary_path)}


def train_v27d_r14_arm(context: R14Context, *, arm: str, seed: int) -> dict[str, Any]:
    if arm not in R14_ARMS or int(seed) not in (42, 44):
        raise ValueError("V27d-r14 arm or seed is invalid")
    summary_path = _run_path(context, "training_summary_template", arm=arm, seed=seed)
    if summary_path.exists():
        return read_sealed_json(summary_path)
    log_path = _run_path(context, "training_log_template", arm=arm, seed=seed)
    logger = configure_training_logger(log_path, logger_name=f"radar_ln3.v27d.r14.{arm}.seed_{seed}")
    try:
        return _train_arm(context, arm=arm, seed=int(seed), logger=logger, log_path=log_path)
    except Exception:
        logger.exception("run=failed | experiment=V27d-r14 | arm=%s | seed=%d", arm, seed)
        raise
    finally:
        close_training_logger(logger)


__all__ = ["load_checkpoint", "tensor_state_hash", "train_v27d_r14_arm"]
