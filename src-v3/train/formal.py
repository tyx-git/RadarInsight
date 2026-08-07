from __future__ import annotations

import argparse
import json
from logging import Logger
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.checkpoint import save_formal_checkpoint
from radar_ln3.config import V3Config, load_config, resolve_project_path
from radar_ln3.contracts import verify_formal_data_contract
from radar_ln3.data import CQ08PointSequenceDataset, build_point_dataloader, collate_point_sequences, require_causal_rd_quality_schema
from radar_ln3.evaluator import (
    LOW_CONFIDENCE_THRESHOLD,
    evaluate_dataset,
    strip_detailed_records,
    write_sample_detail_information,
)
from radar_ln3.factory import build_model
from radar_ln3.losses import five_class_loss
from radar_ln3.losses import causal_prefix_loss
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
    log_low_confidence_records,
    log_metric_set,
    progress_interval,
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: V3Config) -> torch.device:
    return torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")


def _train_formal_impl(
    project_root: Path,
    config: V3Config,
    *,
    max_steps_override: int | None = None,
    epochs_override: int | None = None,
    skip_validation: bool = False,
    logger: Logger,
) -> dict[str, Any]:
    if config.schema_version not in {"RADAR-LN3-V2", "RADAR-LN3-V3", "RADAR-LN3-V4"} or config.train.mode != "formal_supervised":
        raise ValueError("train_formal requires a RADAR-LN3-V2/V3 formal_supervised configuration")
    if not config.train.allow_parameter_updates:
        raise PermissionError("formal training configuration forbids parameter updates")
    contract = verify_formal_data_contract(project_root, config)
    _seed(config.train.seed)
    expected_schema = config.data.index_schema_version or ("RADAR-LN3-V3" if config.schema_version == "RADAR-LN3-V3" else "RADAR-LN3-V1")
    train_dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=config.data.train_split_name or "train",
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        expected_schema_version=expected_schema,
    )
    val_dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=config.data.validation_split_name or "val",
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        expected_schema_version=expected_schema,
    )
    device = _device(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    loader = build_point_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    epochs = max(1, int(epochs_override if epochs_override is not None else config.train.epochs))
    max_steps = max_steps_override if max_steps_override is not None else config.train.max_train_steps
    steps_per_epoch = len(loader)
    planned_steps = epochs * steps_per_epoch
    if max_steps is not None:
        planned_steps = min(planned_steps, max(0, int(max_steps)))
    log_interval = progress_interval(steps_per_epoch)
    logger.info(
        "data | train_parents=%d | val_parents=%d | batch=%d | steps_per_epoch=%d | max_steps=%s",
        len(train_dataset),
        len(val_dataset),
        config.train.batch_size,
        steps_per_epoch,
        "configured" if max_steps is None else str(max_steps),
    )
    logger.info(
        "schedule | device=%s | epochs=%d | planned_steps=%d | lr=%.2e | seed=%d | selection=%s",
        device,
        epochs,
        planned_steps,
        config.train.learning_rate,
        config.train.seed,
        config.train.checkpoint_selection_metric,
    )
    logger.info(
        "contract | tr=%d+missing%d | rd_quality=%d | rd=%dx%d | classes=%d | pretraining=not_required",
        config.model.tr_value_dim,
        config.model.tr_missing_dim,
        config.model.rd_quality_dim,
        config.model.rd_height,
        config.model.rd_width,
        config.model.num_classes,
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    best_validation: dict[str, Any] | None = None
    best_checkpoint = resolve_project_path(project_root, config.train.checkpoint_path)
    last_checkpoint_path = resolve_project_path(project_root, config.train.last_checkpoint_path or config.train.checkpoint_path)
    metadata = {
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "data_contract": contract,
        "index_schema_version": expected_schema,
        "model_contract": {
            "tr_value_dim": config.model.tr_value_dim,
            "tr_missing_dim": config.model.tr_missing_dim,
            "rd_quality_dim": config.model.rd_quality_dim,
            "rd_shape": [config.model.rd_height, config.model.rd_width],
            "num_classes": config.model.num_classes,
            "dt_valid_dim": config.model.dt_valid_dim,
            "rd_quality_schema": config.data.rd_quality_schema,
            "liquid_tau_min_seconds": config.model.liquid_tau_min_seconds,
            "liquid_tau_max_seconds": config.model.liquid_tau_max_seconds,
            "liquid_zero_dt_event_floor_seconds": config.model.liquid_zero_dt_event_floor_seconds,
            "liquid_dt_max_seconds": config.model.liquid_dt_max_seconds,
            "liquid_invalid_dt_update_mode": config.model.liquid_invalid_dt_update_mode,
            "liquid_invalid_dt_observation_gain": config.model.liquid_invalid_dt_observation_gain,
            "liquid_state_normalization": config.model.liquid_state_normalization,
            "rd_frame_aggregation": config.model.rd_frame_aggregation,
            "loss_mode": config.train.loss_mode,
        },
        "class_weights": list(config.train.class_weights),
        "seed": config.train.seed,
        "pretraining": "not_required_and_not_used",
    }
    stop_reason = "epoch_limit"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        started = time.perf_counter()
        for batch in loader:
            tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensor_batch["tr_features"],
                tr_missing_mask=tensor_batch["tr_missing_mask"],
                tr_dt_seconds=tensor_batch["tr_dt_seconds"],
                tr_dt_valid=tensor_batch.get("tr_dt_valid"),
                rd_frames=tensor_batch["rd_frames"],
                rd_frame_mask=tensor_batch["rd_frame_mask"],
                rd_quality=tensor_batch["rd_quality"],
                tr_padding_mask=tensor_batch["tr_padding_mask"],
            )
            if config.train.loss_mode == "causal_prefix":
                loss = causal_prefix_loss(
                    output["sequence_logits"],
                    tensor_batch["labels"],
                    tensor_batch["tr_padding_mask"],
                    class_weights,
                )
            else:
                loss = five_class_loss(output["window_logits"], tensor_batch["labels"], class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.gradient_clip_norm).item())
            optimizer.step()
            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.item())
            if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                raise ValueError("formal training produced non-finite loss or gradient")
            if epoch_steps == 1 or epoch_steps % log_interval == 0 or (max_steps is not None and global_step >= max_steps):
                elapsed = time.perf_counter() - started
                seconds_per_step = elapsed / max(1, epoch_steps)
                remaining_steps = max(0, planned_steps - global_step)
                logger.info(
                    "progress | epoch=%02d/%02d | step=%04d/%04d | global=%05d/%05d | loss=%.6f | grad=%.3f | lr=%.2e | elapsed=%s | eta=%s",
                    epoch,
                    epochs,
                    epoch_steps,
                    steps_per_epoch,
                    global_step,
                    planned_steps,
                    float(loss.item()),
                    grad_norm,
                    optimizer.param_groups[0]["lr"],
                    format_duration(elapsed),
                    format_duration(seconds_per_step * remaining_steps),
                )
            if max_steps is not None and global_step >= max_steps:
                stop_reason = "max_train_steps"
                break
        if skip_validation:
            validation_full = {"status": "not_run_by_argument"}
        else:
            validation_full = evaluate_dataset(model, val_dataset, config=config, device=device, split_name=config.data.validation_split_name or "val")
        metric = validation_full.get("parent_metrics", {}).get(config.train.checkpoint_selection_metric.replace("parent_", "")) if isinstance(validation_full, dict) else None
        if metric is None and isinstance(validation_full, dict):
            metric = validation_full.get("window_metrics", {}).get(config.train.checkpoint_selection_metric)
        metric_value = float(metric) if metric is not None else float("-inf")
        validation_summary = strip_detailed_records(validation_full)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / max(1, epoch_steps),
            "train_steps": epoch_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "validation": validation_summary,
            "selection_metric": config.train.checkpoint_selection_metric,
            "selection_metric_value": None if not np.isfinite(metric_value) else metric_value,
        }
        history.append(epoch_record)
        parent_metrics = validation_full.get("parent_metrics", {}) if isinstance(validation_full, dict) else {}
        window_metrics = validation_full.get("window_metrics", {}) if isinstance(validation_full, dict) else {}
        logger.info(
            "epoch | epoch=%02d/%02d | train_loss=%.6f | steps=%d | elapsed=%s | val_parent_f1=%s | val_window_f1=%s | selected=%s",
            epoch,
            epochs,
            epoch_record["train_loss"],
            epoch_steps,
            format_duration(epoch_record["elapsed_seconds"]),
            format_metric(parent_metrics.get("macro_f1")),
            format_metric(window_metrics.get("macro_f1")),
            format_metric(epoch_record["selection_metric_value"]),
        )
        save_formal_checkpoint(last_checkpoint_path, model, optimizer, config_name=config.config_name, epoch=epoch, global_step=global_step, metrics={"train_loss": epoch_record["train_loss"], "selection_metric_value": epoch_record["selection_metric_value"]}, metadata=metadata, checkpoint_schema=config.train.checkpoint_schema)
        logger.info("checkpoint | kind=last | path=%s", last_checkpoint_path)
        if metric_value > best_metric or best_epoch < 0:
            best_metric = metric_value
            best_epoch = epoch
            best_validation = validation_full
            save_formal_checkpoint(best_checkpoint, model, optimizer, config_name=config.config_name, epoch=epoch, global_step=global_step, metrics={"train_loss": epoch_record["train_loss"], "selection_metric_value": epoch_record["selection_metric_value"]}, metadata=metadata, checkpoint_schema=config.train.checkpoint_schema)
            logger.info("checkpoint | kind=best | path=%s | metric=%s", best_checkpoint, format_metric(epoch_record["selection_metric_value"]))
        if max_steps is not None and global_step >= max_steps:
            logger.info("stop | reason=max_train_steps | global_step=%d", global_step)
            break
        if config.train.early_stopping_patience > 0 and best_epoch > 0 and epoch - best_epoch >= config.train.early_stopping_patience:
            stop_reason = "early_stopping"
            logger.info("stop | reason=early_stopping | epoch=%d | best_epoch=%d", epoch, best_epoch)
            break
    artifact_dir = resolve_project_path(project_root, config.train.formal_artifact_dir or "artifacts/RADAR-LN3-V2-formal")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_value = str(best_checkpoint)
    if best_validation is not None and "records" in best_validation:
        sample_detail_path = write_sample_detail_information(
            artifact_dir,
            best_validation,
            checkpoint_path=best_checkpoint_value,
            epoch=best_epoch,
            selection_metric=config.train.checkpoint_selection_metric,
        )
        logger.info("sample_details | path=%s | confidence_threshold=%.2f | all_records_saved=true", sample_detail_path, LOW_CONFIDENCE_THRESHOLD)
        log_metric_set(logger, "val:parent_final_prefix", best_validation.get("parent_metrics", {}))
        log_metric_set(logger, "val:stable_final", best_validation.get("stable_metrics") or {})
        log_metric_set(logger, "val:window_causal_prefix", best_validation.get("window_metrics", {}))
        log_low_confidence_records(logger, best_validation, threshold=LOW_CONFIDENCE_THRESHOLD)
        logger.info("final_metrics | status=complete | best_checkpoint=%s | sample_details=%s", best_checkpoint_value, sample_detail_path)
    else:
        sample_detail_path = None
        logger.info("sample_details | status=not_written | reason=validation_skipped")
    result: dict[str, Any] = {
        "status": "passed",
        "config_name": config.config_name,
        "device": str(device),
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(val_dataset),
        "epochs_requested": epochs,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_selection_metric_value": None if not np.isfinite(best_metric) else best_metric,
        "best_validation_metrics": None if best_validation is None else strip_detailed_records(best_validation),
        "stop_reason": stop_reason,
        "history": history,
        "best_checkpoint": best_checkpoint_value,
        "last_checkpoint_path": str(last_checkpoint_path),
        "sample_detail_information": None if sample_detail_path is None else str(sample_detail_path),
        "formal_pretraining": "not_required_and_not_used",
        "test_evaluation": "not_executed",
        "production_p95_2s": "not_evaluated",
    }
    (artifact_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "summary | status=%s | global_step=%d | best_epoch=%d | best_metric=%s | stop_reason=%s | path=%s",
        result["status"],
        result["global_step"],
        result["best_epoch"],
        format_metric(result["best_selection_metric_value"]),
        result["stop_reason"],
        artifact_dir / "training-summary.json",
    )
    return result


def train_formal(
    project_root: Path,
    config: V3Config,
    *,
    max_steps_override: int | None = None,
    epochs_override: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Run formal training with compact console and artifact-local logging."""

    require_causal_rd_quality_schema(config.data.rd_quality_schema)
    artifact_dir = resolve_project_path(project_root, config.train.formal_artifact_dir or "artifacts/RADAR-LN3-V2-formal")
    logger = configure_training_logger(artifact_dir / "training.log")
    logger.info(
        "run=start | config=%s | mode=%s | requested_device=%s | batch=%d | artifact=%s",
        config.config_name,
        config.train.mode,
        config.train.device,
        config.train.batch_size,
        artifact_dir,
    )
    try:
        result = _train_formal_impl(
            project_root,
            config,
            max_steps_override=max_steps_override,
            epochs_override=epochs_override,
            skip_validation=skip_validation,
            logger=logger,
        )
        logger.info(
            "run=complete | status=%s | global_step=%d | best_epoch=%d | stop_reason=%s",
            result["status"],
            result["global_step"],
            result["best_epoch"],
            result["stop_reason"],
        )
        return result
    except Exception:
        logger.exception("run=failed")
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal RADAR-LN3 supervised training; does not use pretraining.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Focused contract limit; omit for configured training budget.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true", help="Only for focused finite-step contract checks.")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path.resolve())
    result = train_formal(project_root, config, max_steps_override=args.max_steps, epochs_override=args.epochs, skip_validation=args.skip_validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
