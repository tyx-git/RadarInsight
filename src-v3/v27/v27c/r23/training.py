"""Fixed-endpoint paired L2 retraining for V27c-r23."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    sha256_file,
    write_torch_if_absent_or_identical,
)
from v27.v27_baseline.base_binding import classifier_state_hash
from v27.v27_baseline.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from v27.v27_baseline.runtime import environment_snapshot, seed_v27

from .data import (
    R23L2Dataset,
    build_r23_dataloader,
    r23_model_input_kwargs,
    r23_reference_model_input_kwargs,
    r23_sidecar_kwargs,
)
from .model import R23IntegratedL2Model, build_r23_model, trainable_parameter_groups
from .protocol import (
    R23_ARMS,
    R23_ARM_SPECS,
    R23_CHECKPOINT_SCHEMA_VERSION,
    R23_L2_CLASSES,
    R23Context,
    load_verified_v27a_cache,
    read_sealed_json,
    r23_artifact_path,
    write_sealed_json_if_identical,
)


def configured_device(context: R23Context) -> torch.device:
    if context.config.training.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27c-r23 configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    if context.config.training.device == "cpu":
        return torch.device("cpu")
    raise ValueError("V27c-r23 requested an unsupported device")


def _deterministic_cuda_workspace(context: R23Context) -> str | None:
    if (
        context.config.training.device != "cuda"
        or not context.config.training.cuda_deterministic_algorithms
    ):
        return None
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "V27c-r23 strict deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
            "to be ':4096:8' or ':16:8' before Python starts"
        )
    return value


def _path(context: R23Context, field: str, *, arm: str, seed: int, epoch: int | None = None) -> Path:
    return r23_artifact_path(
        context.project_root,
        getattr(context.config.artifacts, field),
        arm=arm,
        seed=seed,
        epoch=epoch,
    )


def _load_initial_states(context: R23Context) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    l2_path = context.project_root / context.config.base.l2_checkpoint_path
    stable_path = context.project_root / context.config.base.v27a_main_checkpoint_path
    l2 = torch.load(l2_path, map_location="cpu", weights_only=False)
    stable = torch.load(stable_path, map_location="cpu", weights_only=False)
    if not isinstance(l2, Mapping) or not isinstance(stable, Mapping):
        raise ValueError("V27c-r23 initial checkpoints are not mappings")
    l2_state = l2.get("model_state")
    stable_state = stable.get("model_state")
    if not isinstance(l2_state, Mapping) or not isinstance(stable_state, Mapping):
        raise ValueError("V27c-r23 initial checkpoints lack model states")
    if (
        l2.get("model_state_sha256") != context.config.base.expected_l2_state_sha256
        or stable.get("model_state_sha256") != context.config.base.expected_v27a_main_state_sha256
        or classifier_state_hash(l2_state)
        != context.config.base.expected_l2_classifier_state_sha256
    ):
        raise ValueError("V27c-r23 initial checkpoint state bindings differ")
    return dict(l2_state), dict(stable_state)


def build_initialized_model(
    context: R23Context, *, arm: str, seed: int, device: torch.device
) -> R23IntegratedL2Model:
    l2_state, stable_state = _load_initial_states(context)
    return build_r23_model(
        context.config,
        arm=arm,
        v27_config=context.v27_config,
        v27_l2_state=l2_state,
        v27a_config=context.v27a_config,
        v27a_main_state=stable_state,
        seed=seed,
    ).to(device)


def _checkpoint_payload(
    context: R23Context,
    *,
    model: R23IntegratedL2Model,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    train_loss: float,
    loader_seed: int,
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": R23_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "manifest_sha256": context.manifest_sha256,
        "input_binding_sha256": canonical_sha256(context.input_binding),
        "l1_checkpoint_deserialized": False,
        "l1_forward_calls": 0,
        "l1_optimizer_parameter_count": 0,
        "l2_initial_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "l2_initial_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "v27a_main_initial_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "v27a_train_cache_payload_sha256": load_verified_v27a_cache(context, split="train")["cache_payload_sha256"],
        "v27a_validation_cache_payload_sha256": load_verified_v27a_cache(context, split="val")["cache_payload_sha256"],
        "tr_clipping": context.input_binding["tr_clipping"],
        "tr_clipping_audits": context.input_binding["tr_clipping_audits"],
        "arm_spec": context.input_binding["arm_specs"][arm],
        "class_names": list(R23_L2_CLASSES),
        "class_weights": list(context.config.base.expected_l2_class_weights),
        "optimizer": context.config.training.optimizer,
        "loss_name": context.config.training.loss_name,
        "loss_reduction": context.config.training.loss_reduction,
        "dataloader_seed": int(loader_seed),
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "model_state": state,
        "model_state_sha256": _state_hash(state),
        "metrics": {"train_loss": float(train_loss), "selection": "fixed_primary_epoch"},
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    from v27.v27a.checkpoint import tensor_state_hash

    return tensor_state_hash(state)


def load_r23_checkpoint(
    context: R23Context,
    *,
    arm: str,
    seed: int,
    device: torch.device | None = None,
) -> tuple[R23IntegratedL2Model, dict[str, Any]]:
    if arm not in R23_ARMS or seed not in context.config.training.seeds:
        raise ValueError("V27c-r23 checkpoint arm or seed is invalid")
    path = _path(
        context,
        "checkpoint_template",
        arm=arm,
        seed=seed,
        epoch=context.config.training.primary_epoch,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r23 checkpoint is not an object")
    checked = dict(payload)
    expected_scope = {
        "unknown_training_exposure": 0,
        "clutter_training_exposure": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "calibration": "not_in_scope",
        "l3": "not_in_scope",
        "test_payload_access_count": 0,
    }
    if (
        checked.get("checkpoint_schema") != R23_CHECKPOINT_SCHEMA_VERSION
        or checked.get("experiment_id") != context.config.experiment_id
        or checked.get("arm") != arm
        or int(checked.get("seed", -1)) != int(seed)
        or int(checked.get("epoch", -1)) != context.config.training.primary_epoch
        or checked.get("manifest_sha256") != context.manifest_sha256
        or checked.get("input_binding_sha256") != canonical_sha256(context.input_binding)
        or checked.get("l1_checkpoint_deserialized") is not False
        or checked.get("l1_forward_calls") != 0
        or checked.get("l1_optimizer_parameter_count") != 0
        or checked.get("l2_initial_checkpoint_sha256")
        != context.config.base.expected_l2_checkpoint_sha256
        or checked.get("l2_initial_classifier_state_sha256")
        != context.config.base.expected_l2_classifier_state_sha256
        or checked.get("v27a_main_initial_checkpoint_sha256")
        != context.config.base.expected_v27a_main_checkpoint_sha256
        or checked.get("class_names") != list(R23_L2_CLASSES)
        or checked.get("class_weights") != list(context.config.base.expected_l2_class_weights)
        or checked.get("optimizer") != context.config.training.optimizer
        or checked.get("loss_name") != context.config.training.loss_name
        or checked.get("loss_reduction") != context.config.training.loss_reduction
        or checked.get("scope") != expected_scope
        or checked.get("arm_spec") != context.input_binding["arm_specs"][arm]
        or checked.get("tr_clipping") != context.input_binding["tr_clipping"]
        or checked.get("tr_clipping_audits")
        != context.input_binding["tr_clipping_audits"]
    ):
        raise ValueError("V27c-r23 checkpoint identity differs")
    l2_state, stable_state = _load_initial_states(context)
    model = build_r23_model(
        context.config,
        arm=arm,
        v27_config=context.v27_config,
        v27_l2_state=l2_state,
        v27a_config=context.v27a_config,
        v27a_main_state=stable_state,
        seed=seed,
    )
    state = checked.get("model_state")
    if not isinstance(state, Mapping) or checked.get("model_state_sha256") != _state_hash(state):
        raise ValueError("V27c-r23 checkpoint model-state hash differs")
    if (
        checked.get("model_contract") != model.model_contract()
        or checked.get("model_contract_sha256") != canonical_sha256(model.model_contract())
    ):
        raise ValueError("V27c-r23 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    summary_path = _path(
        context,
        "training_summary_template",
        arm=arm,
        seed=seed,
    )
    summary = read_sealed_json(
        summary_path,
        schema="RADAR-LN3-V27c-r23-factorial-l2-training-summary-v1",
    )
    if (
        summary.get("experiment_id") != context.config.experiment_id
        or summary.get("arm") != arm
        or int(summary.get("seed", -1)) != int(seed)
        or int(summary.get("selected_epoch", -1))
        != context.config.training.primary_epoch
        or summary.get("manifest_sha256") != context.manifest_sha256
        or summary.get("checkpoint_state_sha256")
        != checked.get("model_state_sha256")
        or summary.get("checkpoint_path")
        != relative_project_path(context.project_root, path)
        or summary.get("checkpoint_sha256") != sha256_file(path)
    ):
        raise ValueError("V27c-r23 training summary provenance differs")
    return model.to(torch.device("cpu") if device is None else device), checked


def _motion_coverage(motion: torch.Tensor, padding: torch.Tensor) -> float:
    active = ~padding
    if not bool(active.any()):
        raise ValueError("V27c-r23 training batch has no active event")
    return float(
        motion.abs().amax(dim=-1).masked_select(active).gt(0.0).float().mean().item()
    )


def train_r23_arm(
    context: R23Context,
    *,
    arm: str,
    seed: int,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Retrain one matched arm through the fixed primary epoch only."""

    if arm not in R23_ARMS or seed not in context.config.training.seeds:
        raise ValueError("V27c-r23 training arm or seed is invalid")
    epochs = context.config.training.epochs if epochs_override is None else int(epochs_override)
    if epochs != context.config.training.primary_epoch:
        raise ValueError("V27c-r23 formal runs permit only the fixed primary epoch")
    workspace = _deterministic_cuda_workspace(context)
    seed_v27(seed, deterministic_cuda=context.config.training.cuda_deterministic_algorithms)
    train_set = R23L2Dataset(context, split="train", arm=arm)
    validation_set = R23L2Dataset(context, split="val", arm=arm)
    loader_seed = int(context.config.training.dataloader_seed) + int(seed)
    loader = build_r23_dataloader(
        train_set,
        batch_size=context.config.training.batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(loader_seed),
    )
    if len(loader) <= 0:
        raise ValueError("V27c-r23 training loader is empty")
    device = configured_device(context)
    model = build_initialized_model(context, arm=arm, seed=seed, device=device)
    groups = trainable_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["base"], "lr": context.config.training.base_learning_rate},
            {"params": groups["stable"], "lr": context.config.training.stable_learning_rate},
            {"params": groups["motion"], "lr": context.config.training.motion_learning_rate},
        ],
        weight_decay=context.config.training.weight_decay,
    )
    weights = torch.tensor(
        context.config.base.expected_l2_class_weights, dtype=torch.float32, device=device
    )
    log_path = _path(context, "training_log_template", arm=arm, seed=seed)
    logger = configure_training_logger(
        log_path,
        logger_name=f"radar_ln3.v27c.r23.{arm}.seed_{seed}",
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    global_step = 0
    try:
        log_event(
            logger,
            "r23_l2_run=start",
            experiment=context.config.experiment_id,
            arm=arm,
            seed=seed,
            epochs=epochs,
            device=str(device),
            train_parents=len(train_set),
            validation_parents=len(validation_set),
            batch_size=context.config.training.batch_size,
            steps_per_epoch=len(loader),
            dataloader_seed=loader_seed,
            l1_checkpoint_deserialized=False,
            l1_forward_calls=0,
            l1_optimizer_parameter_count=0,
            l2_initial_epoch=11,
            stable8_initial_epoch=20,
            input_mode=context.input_binding["arm_specs"][arm]["input_mode"],
            motion_input=context.input_binding["arm_specs"][arm]["motion_mode"],
            tr_clipping=context.input_binding["tr_clipping"],
            tr_clipping_audits=context.input_binding["tr_clipping_audits"],
            unknown=0,
            clutter=0,
            test=0,
            cublas_workspace_config=workspace,
        )
        interval = progress_interval(len(loader))
        for epoch in range(1, epochs + 1):
            model.train()
            # These modules must remain deterministic/frozen even when the
            # container switches to train mode for its trainable components.
            model.frozen_reference_base_model.eval()
            model.trainable_base_model.trajectory_heads.eval()
            losses: list[float] = []
            gradient_max = 0.0
            clipped_steps = 0
            source_coverage: list[float] = []
            effective_coverage: list[float] = []
            epoch_started = time.perf_counter()
            for step, batch in enumerate(loader, start=1):
                base_inputs = r23_model_input_kwargs(batch, device)
                reference_inputs = r23_reference_model_input_kwargs(batch, device)
                sidecars = r23_sidecar_kwargs(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    base_inputs=base_inputs,
                    reference_base_inputs=reference_inputs,
                    **sidecars,
                )
                loss = parent_final_fixed_parent_count_ce(
                    output["sequence_logits"],
                    batch["labels"].to(device=device, non_blocking=True),
                    base_inputs["tr_padding_mask"],
                    class_weights=weights,
                )["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27c-r23 L2 loss is non-finite")
                loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in groups.values()
                            for parameter in group
                        ],
                        context.config.training.gradient_clip_norm,
                        error_if_nonfinite=True,
                    ).item()
                )
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().item()))
                gradient_max = max(gradient_max, gradient_norm)
                clipped_steps += int(
                    gradient_norm > context.config.training.gradient_clip_norm
                )
                source_coverage.append(
                    _motion_coverage(
                        sidecars["motion_values"], base_inputs["tr_padding_mask"]
                    )
                )
                effective_motion = (
                    sidecars["motion_values"]
                    if R23_ARM_SPECS[arm]["motion_mode"] == "causal"
                    else torch.zeros_like(sidecars["motion_values"])
                )
                effective_coverage.append(
                    _motion_coverage(effective_motion, base_inputs["tr_padding_mask"])
                )
                if step == 1 or step == len(loader) or step % interval == 0:
                    log_event(
                        logger,
                        "r23_l2_progress",
                        arm=arm,
                        epoch=f"{epoch:02d}/{epochs:02d}",
                        step=f"{step:04d}/{len(loader):04d}",
                        global_step=global_step,
                        loss=losses[-1],
                        gradient_norm=gradient_norm,
                        elapsed=format_duration(time.perf_counter() - epoch_started),
                    )
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm_max": gradient_max,
                "gradient_clipped_steps": clipped_steps,
                "source_motion_nonzero_event_fraction_observed": float(
                    np.mean(source_coverage)
                ),
                "effective_motion_nonzero_event_fraction_observed": float(
                    np.mean(effective_coverage)
                ),
            }
            history.append(record)
            log_event(
                logger,
                "r23_l2_epoch=complete",
                arm=arm,
                epoch=f"{epoch:02d}/{epochs:02d}",
                train_loss=record["train_loss"],
                gradient_norm_max=gradient_max,
                source_motion_nonzero_event_fraction=record[
                    "source_motion_nonzero_event_fraction_observed"
                ],
                effective_motion_nonzero_event_fraction=record[
                    "effective_motion_nonzero_event_fraction_observed"
                ],
                elapsed=format_duration(time.perf_counter() - epoch_started),
            )

        payload = _checkpoint_payload(
            context,
            model=model,
            arm=arm,
            seed=seed,
            epoch=epochs,
            global_step=global_step,
            train_loss=history[-1]["train_loss"],
            loader_seed=loader_seed,
        )
        checkpoint_path = _path(
            context,
            "checkpoint_template",
            arm=arm,
            seed=seed,
            epoch=epochs,
        )
        write_torch_if_absent_or_identical(checkpoint_path, payload)
        summary = {
            "schema_version": "RADAR-LN3-V27c-r23-factorial-l2-training-summary-v1",
            "status": "completed_fixed_epoch_matched_l2_retraining",
            "experiment_id": context.config.experiment_id,
            "arm": arm,
            "seed": int(seed),
            "selected_epoch": int(epochs),
            "selection": "fixed_primary_epoch",
            "manifest_sha256": context.manifest_sha256,
            "input_binding_sha256": canonical_sha256(context.input_binding),
            "arm_spec": context.input_binding["arm_specs"][arm],
            "tr_clipping": context.input_binding["tr_clipping"],
            "tr_clipping_audits": context.input_binding["tr_clipping_audits"],
            "dataloader_seed": loader_seed,
            "history": history,
            "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_state_sha256": payload["model_state_sha256"],
            "training_log_path": relative_project_path(context.project_root, log_path),
            "l1_checkpoint_deserialized": False,
            "l1_forward_calls": 0,
            "l1_optimizer_parameter_count": 0,
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "test_payload_access_count": 0,
            "environment": environment_snapshot(),
            "cublas_workspace_config": workspace,
        }
        summary_path = _path(context, "training_summary_template", arm=arm, seed=seed)
        write_sealed_json_if_identical(summary_path, summary)
        log_event(
            logger,
            "r23_l2_run=complete",
            arm=arm,
            seed=seed,
            checkpoint_path=str(checkpoint_path),
            summary_path=str(summary_path),
            elapsed=format_duration(time.perf_counter() - started),
        )
        return {**summary, "summary_path": str(summary_path)}
    except Exception:
        logger.exception(
            "r23_l2_run=failed | arm=%s | seed=%d | elapsed=%s",
            arm,
            seed,
            format_duration(time.perf_counter() - started),
        )
        raise
    finally:
        close_training_logger(logger)


def run_r23_training_step(context: R23Context, *, arm: str) -> dict[str, Any]:
    """Run one CPU optimizer update without writing a formal artifact.

    This is intentionally the smallest execution check that covers the real
    L2 forward, the loss reduction, each optimizer group, and the frozen-L1/
    trajectory-head boundaries.  It does not create a partial r23 checkpoint.
    """

    if arm not in R23_ARMS:
        raise ValueError("V27c-r23 training-step arm is invalid")
    seed = int(context.config.training.seeds[0])
    seed_v27(seed, deterministic_cuda=False)
    dataset = R23L2Dataset(context, split="train", arm=arm)
    batch = next(
        iter(build_r23_dataloader(dataset, batch_size=2, shuffle=False))
    )
    device = torch.device("cpu")
    model = build_initialized_model(context, arm=arm, seed=seed, device=device)
    groups = trainable_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["base"], "lr": context.config.training.base_learning_rate},
            {"params": groups["stable"], "lr": context.config.training.stable_learning_rate},
            {"params": groups["motion"], "lr": context.config.training.motion_learning_rate},
        ],
        weight_decay=context.config.training.weight_decay,
    )
    base_inputs = r23_model_input_kwargs(batch, device)
    reference_inputs = r23_reference_model_input_kwargs(batch, device)
    sidecars = r23_sidecar_kwargs(batch, device)
    output = model(
        base_inputs=base_inputs,
        reference_base_inputs=reference_inputs,
        **sidecars,
    )
    weights = torch.tensor(
        context.config.base.expected_l2_class_weights,
        dtype=torch.float32,
        device=device,
    )
    loss = parent_final_fixed_parent_count_ce(
        output["sequence_logits"],
        batch["labels"].to(device),
        base_inputs["tr_padding_mask"],
        class_weights=weights,
    )["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("V27c-r23 training-step loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for values in groups.values() for parameter in values],
            context.config.training.gradient_clip_norm,
            error_if_nonfinite=True,
        ).item()
    )
    if any(
        parameter.grad is not None
        for parameter in model.frozen_reference_base_model.parameters()
    ):
        raise AssertionError("V27c-r23 frozen reference received a gradient")
    if any(
        parameter.grad is not None
        for parameter in model.trainable_base_model.trajectory_heads.parameters()
    ):
        raise AssertionError("V27c-r23 trajectory heads received a gradient")
    optimizer.step()
    return {
        "arm": arm,
        "parent_count": len(batch["parent_sample_ids"]),
        "loss": float(loss.detach().item()),
        "gradient_norm": gradient_norm,
        "l1_checkpoint_deserialized": False,
        "frozen_reference_gradients": False,
        "trajectory_head_gradients": False,
        "formal_artifact_written": False,
    }


__all__ = [
    "build_initialized_model",
    "configured_device",
    "load_r23_checkpoint",
    "run_r23_training_step",
    "train_r23_arm",
]
