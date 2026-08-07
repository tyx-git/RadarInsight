from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
    log_low_confidence_records,
    log_metric_set,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    evaluate_task,
    save_task_checkpoint,
    seed_everything,
    task_device,
    task_prefix_loss,
    verify_task_data_contract,
    load_task_config,
)


def _strip_records(result: dict[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value.pop("records", None)
    return value


def _finite_metric(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"checkpoint selection metric {name!r} is missing")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"checkpoint selection metric {name!r} is not finite")
    return result


def build_checkpoint_selection(validation: dict[str, Any], metric_name: str) -> dict[str, Any]:
    """Build an auditable, deterministic checkpoint ranking contract.

    ``parent_min_precision_recall`` ranks first by the weakest per-class
    precision/recall value and uses parent Macro-F1 only as a tie-breaker.
    A class with no validation support is rejected because its recall cannot
    establish the requested weak-class floor.
    """

    metric_name = str(metric_name)
    if metric_name == "parent_min_precision_recall":
        metrics = dict(validation.get("parent_metrics") or {})
        per_class = list(metrics.get("per_class") or [])
        if not per_class:
            raise ValueError("parent_min_precision_recall requires per-class parent metrics")
        zero_support = [str(row.get("class")) for row in per_class if int(row.get("support", 0)) <= 0]
        if zero_support:
            raise ValueError(
                "parent_min_precision_recall cannot rank a validation set with zero-support classes: "
                + ",".join(zero_support)
            )
        class_values: list[float] = []
        for row in per_class:
            class_name = str(row.get("class"))
            class_values.append(_finite_metric(row.get("precision"), name=f"{class_name}.precision"))
            class_values.append(_finite_metric(row.get("recall"), name=f"{class_name}.recall"))
        weak_class_floor = min(class_values)
        macro_f1 = _finite_metric(metrics.get("macro_f1"), name="parent_macro_f1")
        return {
            "metric": metric_name,
            "primary_metric": "parent_min_class_precision_recall",
            "primary_value": weak_class_floor,
            "tie_break_metric": "parent_macro_f1",
            "tie_break_value": macro_f1,
            "ranking_key": [weak_class_floor, macro_f1],
        }
    if metric_name.startswith("parent_"):
        scope_name = "parent_metrics"
        value_name = metric_name.removeprefix("parent_")
    elif metric_name.startswith("window_"):
        scope_name = "window_metrics"
        value_name = metric_name.removeprefix("window_")
    else:
        raise ValueError(
            "task checkpoint_selection_metric must start with parent_ or window_; "
            f"received={metric_name!r}"
        )
    metrics = dict(validation.get(scope_name) or {})
    value = _finite_metric(metrics.get(value_name), name=metric_name)
    return {
        "metric": metric_name,
        "primary_metric": metric_name,
        "primary_value": value,
        "tie_break_metric": None,
        "tie_break_value": None,
        "ranking_key": [value],
    }


def unavailable_checkpoint_selection(metric_name: str) -> dict[str, Any]:
    return {
        "metric": str(metric_name),
        "primary_metric": str(metric_name),
        "primary_value": None,
        "tie_break_metric": None,
        "tie_break_value": None,
        "ranking_key": [None],
    }


def checkpoint_selection_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.get("metric") != incumbent.get("metric"):
        raise ValueError("cannot compare checkpoint selections with different metrics")

    def ranking_key(selection: dict[str, Any]) -> tuple[float, ...]:
        values = list(selection.get("ranking_key") or [])
        if not values:
            raise ValueError("checkpoint selection has an empty ranking key")
        return tuple(float("-inf") if value is None else float(value) for value in values)

    return ranking_key(candidate) > ranking_key(incumbent)


def _write_sample_details(
    artifact_dir: Path,
    result: dict[str, Any],
    *,
    checkpoint: Path,
    epoch: int,
    schema_version: str,
) -> Path:
    payload = {
        "schema_version": f"{schema_version}-task-sample-detail-v1",
        "task_name": result.get("task_name"),
        "checkpoint": str(checkpoint),
        "epoch": int(epoch),
        "counts": {key: len(value) for key, value in result.get("records", {}).items()},
        "records": result.get("records", {}),
    }
    path = artifact_dir / "sample_detail_information.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def train_task(
    project_root: Path,
    config_path: Path,
    *,
    max_steps: int | None = None,
    epochs_override: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    config = load_task_config(config_path)
    if not config.train.allow_parameter_updates:
        raise PermissionError("task configuration forbids parameter updates")
    seed_everything(config.train.seed)
    contract = verify_task_data_contract(project_root, config)
    artifact_dir = project_root / config.train.formal_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact_dir / "training.log")
    started_run = time.perf_counter()
    try:
        train_dataset = build_task_dataset(project_root, config, split_name=config.data.train_split_name)
        val_dataset = build_task_dataset(project_root, config, split_name=config.data.validation_split_name)
        train_loader = build_task_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
        device = task_device(config)
        model = LiquidTaskClassifier(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        epochs = max(1, int(epochs_override if epochs_override is not None else config.train.epochs))
        steps_per_epoch = len(train_loader)
        planned_steps = epochs * steps_per_epoch if max_steps is None else min(epochs * steps_per_epoch, max_steps)
        logger.info(
            "run=start | task=%s | config=%s | device=%s | train_parents=%d | val_parents=%d | batch=%d | epochs=%d | planned_steps=%d | pretraining=not_required",
            config.task.name,
            config.config_name,
            device,
            len(train_dataset),
            len(val_dataset),
            config.train.batch_size,
            epochs,
            planned_steps,
        )
        logger.info(
            "contract | index_sha256=%s | contract_sha256=%s | stats_sha256=%s | classes=%s | loss=causal_prefix",
            contract["hashes"]["index_sha256"],
            contract["hashes"]["contract_sha256"],
            contract["hashes"]["stats_sha256"],
            ",".join(config.task.class_names),
        )
        global_step = 0
        best_selection: dict[str, Any] | None = None
        best_epoch = -1
        best_validation: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        best_checkpoint = project_root / config.train.checkpoint_path
        last_checkpoint = project_root / config.train.last_checkpoint_path
        stop_reason = "epoch_limit"
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_started = time.perf_counter()
            for batch in train_loader:
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(
                    tr_features=tensors["tr_features"],
                    tr_missing_mask=tensors["tr_missing_mask"],
                    tr_dt_seconds=tensors["tr_dt_seconds"],
                    tr_dt_valid=tensors["tr_dt_valid"],
                    tr_motion_features=tensors.get("tr_motion_features"),
                    tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
                    rd_frames=tensors["rd_frames"],
                    rd_frame_mask=tensors["rd_frame_mask"],
                    rd_quality=tensors["rd_quality"],
                    tr_padding_mask=tensors["tr_padding_mask"],
                )
                loss = task_prefix_loss(output["sequence_logits"], tensors["labels"], tensors["tr_padding_mask"], class_weights=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                optimizer.step()
                global_step += 1
                epoch_steps += 1
                epoch_loss += float(loss.item())
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("task training produced non-finite loss or gradient")
                if epoch_steps == 1 or epoch_steps % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info("progress | task=%s | epoch=%02d/%02d | step=%04d/%04d | global=%05d/%05d | loss=%.6f | grad=%.3f", config.task.name, epoch, epochs, epoch_steps, steps_per_epoch, global_step, planned_steps, float(loss.item()), grad_norm)
                if max_steps is not None and global_step >= max_steps:
                    stop_reason = "max_train_steps"
                    break
            if skip_validation:
                validation = {"status": "not_run_by_argument"}
                selection = unavailable_checkpoint_selection(config.train.checkpoint_selection_metric)
            else:
                validation = evaluate_task(model, val_dataset, config=config, device=device)
                selection = build_checkpoint_selection(validation, config.train.checkpoint_selection_metric)
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_loss / max(1, epoch_steps),
                "train_steps": epoch_steps,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "selection_metric": config.train.checkpoint_selection_metric,
                "selection_metric_value": selection["primary_value"],
                "checkpoint_selection": selection,
                "validation": _strip_records(validation),
            }
            history.append(epoch_record)
            logger.info(
                "epoch | task=%s | epoch=%02d/%02d | train_loss=%.6f | selection=%s:%s | tie=%s:%s | elapsed=%s",
                config.task.name,
                epoch,
                epochs,
                epoch_record["train_loss"],
                selection["primary_metric"],
                format_metric(selection["primary_value"]),
                selection["tie_break_metric"] or "none",
                format_metric(selection["tie_break_value"]),
                format_duration(epoch_record["elapsed_seconds"]),
            )
            save_task_checkpoint(last_checkpoint, model, optimizer, config=config, epoch=epoch, global_step=global_step, metrics=epoch_record)
            if checkpoint_selection_is_better(selection, best_selection):
                best_selection = selection
                best_epoch = epoch
                best_validation = validation
                save_task_checkpoint(best_checkpoint, model, optimizer, config=config, epoch=epoch, global_step=global_step, metrics=epoch_record)
                logger.info(
                    "checkpoint | task=%s | kind=best | path=%s | selection=%s:%s | tie=%s:%s",
                    config.task.name,
                    best_checkpoint,
                    selection["primary_metric"],
                    format_metric(selection["primary_value"]),
                    selection["tie_break_metric"] or "none",
                    format_metric(selection["tie_break_value"]),
                )
            if max_steps is not None and global_step >= max_steps:
                break
            if config.train.early_stopping_patience and best_epoch > 0 and epoch - best_epoch >= config.train.early_stopping_patience:
                stop_reason = "early_stopping"
                break
        sample_details = None
        if best_validation is not None and "records" in best_validation:
            sample_details = _write_sample_details(
                artifact_dir,
                best_validation,
                checkpoint=best_checkpoint,
                epoch=best_epoch,
                schema_version=config.schema_version,
            )
            log_metric_set(logger, "val:parent_final_prefix", best_validation.get("parent_metrics", {}))
            log_metric_set(logger, "val:window_causal_prefix", best_validation.get("window_metrics", {}))
            log_low_confidence_records(logger, best_validation, threshold=0.60)
        else:
            logger.info("metrics | scope=validation | status=not_executed")
        result = {
            "status": "passed",
            "task_name": config.task.name,
            "config_name": config.config_name,
            "device": str(device),
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(val_dataset),
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_selection_metric_value": None if best_selection is None else best_selection["primary_value"],
            "best_checkpoint_selection": best_selection,
            "best_validation_metrics": None if best_validation is None else _strip_records(best_validation),
            "stop_reason": stop_reason,
            "history": history,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "sample_detail_information": None if sample_details is None else str(sample_details),
            "formal_pretraining": "not_required_and_not_used",
            "test_evaluation": "not_executed",
            "elapsed_seconds": time.perf_counter() - started_run,
        }
        (artifact_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("run=complete | task=%s | status=passed | best_checkpoint=%s | summary=%s", config.task.name, best_checkpoint, artifact_dir / "training-summary.json")
        return result
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one repaired RADAR-LN3 liquid task; formal training is manual-only.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    result = train_task(project_root, config_path.resolve(), max_steps=args.max_steps, epochs_override=args.epochs, skip_validation=args.skip_validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
