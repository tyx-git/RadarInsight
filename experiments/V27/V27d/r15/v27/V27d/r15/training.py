"""训练 V27d-r15 的 matched control 与 V10 treatment。"""

from __future__ import annotations

import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import canonical_sha256, write_torch_if_absent_or_identical
from v27.v27_baseline.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)

from .data import R15L1Dataset, build_r15_dataloader, r15_model_input_kwargs, r15_sidecar_kwargs
from .model import R15L1Model, build_r15_model, model_state_hash
from .protocol import (
    R15_ARMS,
    R15_CHECKPOINT_SCHEMA,
    R15_CLASS_NAMES,
    R15_SUMMARY_SCHEMA,
    R15Context,
    artifact_path,
    write_sealed_json_if_identical,
)


def _device(context: R15Context) -> torch.device:
    requested = str(context.config.training["device"]).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r15 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"V27d-r15 device is unsupported: {requested}")


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _checkpoint_payload(
    context: R15Context,
    model: R15L1Model,
    optimizer: torch.optim.Optimizer | None,
    *,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    state = model.state_dict()
    payload = {
        "schema_version": R15_CHECKPOINT_SCHEMA,
        "experiment_id": "V27d-r15",
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_sha256": context.config.sha256,
        "config_identity_sha256": context.config.identity_sha256,
        "manifest_sha256": context.manifest_sha256,
        "model_contract": model.model_contract(),
        "model_contract_sha256": canonical_sha256(model.model_contract()),
        "model_state": state,
        "model_state_sha256": model_state_hash(model),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics),
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
    }
    if optimizer is None:
        payload.pop("optimizer_state")
    return payload


def _global_selection_enabled(context: R15Context) -> bool:
    return context.config.protocol.get("checkpoint_selection") == "global_validation_per_class_f1"


def _assess_r15_checkpoint(metrics: Mapping[str, Any], *, epoch: int) -> dict[str, Any]:
    rows = {str(row["class"]): row for row in metrics.get("per_class") or ()}
    if set(rows) != set(R15_CLASS_NAMES) or int(epoch) <= 0:
        raise ValueError("V27d-r15 checkpoint assessment classes or epoch differ")
    values = {name: float(rows[name]["f1"]) for name in R15_CLASS_NAMES}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("V27d-r15 checkpoint assessment is non-finite")
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    return {
        "schema_version": "RADAR-LN3-V27d-r15-checkpoint-selector-v1",
        "epoch": int(epoch),
        "feasible": True,
        "per_class_f1": values,
        "minimum_per_class_f1": min(values.values()),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "selection_policy": "global_validation_per_class_f1",
        "ranking_key": [
            min(values.values()),
            values["KNOWN_TARGET_EVIDENCE"],
            values["KNOWN_CLUTTER_EVIDENCE"],
            macro_f1,
            accuracy,
            -float(epoch),
        ],
    }


def _validate_checkpoint_payload(context: R15Context, payload: Mapping[str, Any], *, arm: str, seed: int) -> None:
    if payload.get("schema_version") != R15_CHECKPOINT_SCHEMA or payload.get("experiment_id") != "V27d-r15":
        raise ValueError("V27d-r15 checkpoint identity differs")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r15 checkpoint arm/seed differs")
    if payload.get("config_sha256") != context.config.sha256 or payload.get("manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27d-r15 checkpoint provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r15 checkpoint model state is missing")
    declared = payload.get("model_state_sha256")
    if not isinstance(declared, str):
        raise ValueError("V27d-r15 checkpoint state hash is missing")
    # Hash against a temporary model is performed by the caller after loading.
    if payload.get("unknown_training_exposure") != 0 or payload.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r15 checkpoint scope differs")


def load_r15_checkpoint(
    context: R15Context,
    *,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[R15L1Model, dict[str, Any]]:
    path = artifact_path(
        context.project_root,
        context.config.artifacts["checkpoint_template"],
        arm=arm,
        seed=seed,
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r15 checkpoint is not an object")
    _validate_checkpoint_payload(context, payload, arm=arm, seed=seed)
    model = build_r15_model(context, arm=arm, seed=seed, device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    if model_state_hash(model) != payload["model_state_sha256"]:
        raise ValueError("V27d-r15 checkpoint state hash differs")
    return model, dict(payload)


def train_r15_arm(
    context: R15Context,
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    if arm not in R15_ARMS:
        raise ValueError(f"V27d-r15 arm is invalid: {arm}")
    if int(seed) not in tuple(int(value) for value in context.config.training["seeds"]):
        raise ValueError(f"V27d-r15 seed is not registered: {seed}")
    _seed(seed)
    device = _device(context)
    train_set = R15L1Dataset(context, view="train")
    validation_set = R15L1Dataset(context, view="validation")
    generator = torch.Generator(device="cpu").manual_seed(int(context.config.training["dataloader_seed"]) + int(seed))
    loader = build_r15_dataloader(
        train_set,
        batch_size=int(context.config.training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    model = build_r15_model(context, arm=arm, seed=seed, device=device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("V27d-r15 model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(context.config.training["learning_rate"]),
        weight_decay=float(context.config.training["weight_decay"]),
    )
    class_weights = torch.tensor(context.config.training["class_weights"], dtype=torch.float32, device=device)
    log_path = artifact_path(
        context.project_root,
        context.config.artifacts["training_log_template"],
        arm=arm,
        seed=seed,
    )
    logger = configure_training_logger(log_path, logger_name=f"radar_ln3.v27d.r15.{arm}.seed_{seed}")
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    global_step = 0
    epochs = int(context.config.training["epochs"])
    global_selection = _global_selection_enabled(context)
    best_assessment: dict[str, Any] | None = None
    if global_selection:
        from .evaluation import _infer_role, _metrics
    try:
        log_event(
            logger,
            "r15_training_start",
            arm=arm,
            seed=seed,
            device=str(device),
            train_parents=len(train_set),
            validation_parents=len(validation_set),
            epochs=epochs,
            batch_size=int(context.config.training["batch_size"]),
            loss=context.config.training["loss_name"],
            v10_semantics=context.config.protocol["v10_time_semantics"],
            unknown=0,
            test=0,
        )
        interval = progress_interval(len(loader))
        for epoch in range(1, epochs + 1):
            model.train()
            model.base.trajectory_heads.eval()
            losses: list[float] = []
            gradient_max = 0.0
            clipped_steps = 0
            epoch_started = time.perf_counter()
            for step, batch in enumerate(loader, start=1):
                inputs = r15_model_input_kwargs(batch, device)
                sidecars = r15_sidecar_kwargs(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(base_inputs=inputs, **sidecars)
                loss = parent_final_fixed_parent_count_ce(
                    output["sequence_logits"],
                    batch["labels"].to(device=device, non_blocking=True),
                    inputs["tr_padding_mask"],
                    class_weights=class_weights,
                )["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r15 loss is non-finite")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(context.config.training["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().item()))
                gradient_max = max(gradient_max, norm)
                clipped_steps += int(norm > float(context.config.training["gradient_clip_norm"]))
                if step == 1 or step == len(loader) or step % interval == 0:
                    log_event(
                        logger,
                        "r15_progress",
                        arm=arm,
                        epoch=f"{epoch:02d}/{epochs:02d}",
                        step=f"{step:04d}/{len(loader):04d}",
                        global_step=global_step,
                        loss=losses[-1],
                        gradient_norm=norm,
                        elapsed=format_duration(time.perf_counter() - epoch_started),
                    )
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm_max": gradient_max,
                "gradient_clipped_steps": clipped_steps,
            }
            if global_selection:
                validation_records = _infer_role(
                    context,
                    arm=arm,
                    seed=seed,
                    view_name="validation",
                    device=device,
                    model=model,
                )
                validation_metrics = _metrics(
                    [row for row in validation_records if row["true_id"] is not None]
                )
                assessment = _assess_r15_checkpoint(validation_metrics, epoch=epoch)
                record["validation_parent_metrics"] = validation_metrics
                record["checkpoint_assessment"] = assessment
                if best_assessment is None or tuple(assessment["ranking_key"]) > tuple(best_assessment["ranking_key"]):
                    best_assessment = dict(assessment)
                log_metric_set(logger, "validation_parent_final", validation_metrics)
                log_event(
                    logger,
                    "r15_selection",
                    arm=arm,
                    epoch=epoch,
                    selection_policy="global_validation_per_class_f1",
                    minimum_per_class_f1=assessment["minimum_per_class_f1"],
                    target_f1=assessment["per_class_f1"]["KNOWN_TARGET_EVIDENCE"],
                    clutter_f1=assessment["per_class_f1"]["KNOWN_CLUTTER_EVIDENCE"],
                    macro_f1=assessment["macro_f1"],
                    accuracy=assessment["accuracy"],
                )
                epoch_payload = _checkpoint_payload(
                    context,
                    model,
                    optimizer,
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                    global_step=global_step,
                    metrics={
                        "train_loss": record["train_loss"],
                        "checkpoint_assessment": assessment,
                        "selection": "global_validation_per_class_f1",
                    },
                )
                epoch_path = artifact_path(
                    context.project_root,
                    context.config.artifacts["epoch_checkpoint_template"],
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                )
                write_torch_if_absent_or_identical(epoch_path, epoch_payload)
            history.append(record)
            log_event(
                logger,
                "r15_epoch_complete",
                arm=arm,
                epoch=f"{epoch:02d}/{epochs:02d}",
                train_loss=history[-1]["train_loss"],
                gradient_norm_max=gradient_max,
                elapsed=format_duration(time.perf_counter() - epoch_started),
            )
        if global_selection:
            if best_assessment is None:
                raise RuntimeError("V27d-r15 global selection lacks validation assessments")
            selected_epoch = int(best_assessment["epoch"])
            selected_epoch_path = artifact_path(
                context.project_root,
                context.config.artifacts["epoch_checkpoint_template"],
                arm=arm,
                seed=seed,
                epoch=selected_epoch,
            )
            selected_epoch_payload = torch.load(selected_epoch_path, map_location="cpu", weights_only=False)
            if not isinstance(selected_epoch_payload, Mapping) or not isinstance(selected_epoch_payload.get("model_state"), Mapping):
                raise ValueError("V27d-r15 global-best checkpoint payload is invalid")
            model.load_state_dict(selected_epoch_payload["model_state"], strict=True)
            selected_path = artifact_path(
                context.project_root,
                context.config.artifacts["checkpoint_template"],
                arm=arm,
                seed=seed,
            )
            metrics = {
                "train_loss": float(selected_epoch_payload["metrics"]["train_loss"]),
                "checkpoint_assessment": dict(best_assessment),
                "selection": "global_validation_per_class_f1",
            }
            payload = _checkpoint_payload(
                context,
                model,
                None,
                arm=arm,
                seed=seed,
                epoch=selected_epoch,
                global_step=int(selected_epoch_payload.get("global_step", 0)),
                metrics=metrics,
            )
            write_torch_if_absent_or_identical(selected_path, payload)
        else:
            selected_epoch = epochs
            selected_path = artifact_path(
                context.project_root,
                context.config.artifacts["checkpoint_template"],
                arm=arm,
                seed=seed,
            )
            metrics = {"final_train_loss": history[-1]["train_loss"], "epochs": epochs}
            payload = _checkpoint_payload(
                context,
                model,
                optimizer,
                arm=arm,
                seed=seed,
                epoch=selected_epoch,
                global_step=global_step,
                metrics=metrics,
            )
            write_torch_if_absent_or_identical(selected_path, payload)
        checkpoint_path = selected_path
        summary = {
            "schema_version": R15_SUMMARY_SCHEMA,
            "experiment_id": "V27d-r15",
            "status": "completed_global_validation_selected_l1_training" if global_selection else "completed_fixed_epoch_l1_training",
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
            "primary_epoch": int(context.config.training["primary_epoch"]),
            "selected_epoch": int(selected_epoch),
            "selection": "global_validation_per_class_f1" if global_selection else "fixed_primary_epoch",
            "train_parent_count": len(train_set),
            "validation_parent_count": len(validation_set),
            "history": history,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_state_sha256": payload["model_state_sha256"],
            "manifest_sha256": context.manifest_sha256,
            "config_sha256": context.config.sha256,
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path = artifact_path(
            context.project_root,
            context.config.artifacts["training_summary_template"],
            arm=arm,
            seed=seed,
        )
        write_sealed_json_if_identical(summary_path, summary)
        log_event(logger, "r15_training_complete", arm=arm, seed=seed, summary_path=str(summary_path), elapsed=format_duration(time.perf_counter() - started))
        return {**summary, "summary_path": str(summary_path)}
    except Exception:
        logger.exception("V27d-r15 training failed | arm=%s | seed=%s", arm, seed)
        raise
    finally:
        close_training_logger(logger)


__all__ = ["load_r15_checkpoint", "train_r15_arm"]