"""Training of the V27d-r9 causal Doppler shape residual branch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from radar_ln3.runtime import relative_project_path
from v27.V27d.logging_utils import close_logger, configure_logger, log_event, log_metric_set, progress_interval

from .checkpoint import save_checkpoint
from .features import load_profile_cache, validate_profile_cache_binding
from .model import CausalDopplerShapeResidual, build_model
from .protocol import R9_CLASS_NAMES, R9_SEEDS, artifact_path, load_context, write_sealed_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"]["device"]).strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r9 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r9 device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], split: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(split=split), must_exist=True)


def _inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {
        "baseline_logits": tensors["frozen_l2_logits"][indices].to(device),
        "profile_values": tensors["profile_values"][indices].to(device),
        "profile_valid": tensors["profile_valid"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _parent_final(logits: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    padding = padding.to(device=logits.device)
    lengths = (~padding).long().sum(dim=1)
    if bool(lengths.le(0).any()):
        raise ValueError("V27d-r9 cannot select a final state from an empty parent")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths - 1]


def _loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    padding: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    final_logits = _parent_final(logits, padding)
    target = labels.to(device=logits.device, dtype=torch.long)
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    per_parent = nn.functional.cross_entropy(final_logits, target, reduction="none")
    value = (per_parent * weights[target]).sum() / float(target.numel())
    if not torch.isfinite(value):
        raise FloatingPointError("V27d-r9 loss is non-finite")
    return value


def _class_metrics(payload: Mapping[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    padding = payload["tensors"]["padding_mask"]
    labels = payload["tensors"]["labels"].numpy().astype(np.int64)
    final = _parent_final(logits, padding).detach().cpu().numpy()
    predictions = np.argmax(final, axis=-1).astype(np.int64)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for true_id, pred_id in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[int(true_id), int(pred_id)] += 1
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(R9_CLASS_NAMES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support})
    total = int(matrix.sum())
    return {
        "support": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
    }


def _evaluate(model: CausalDopplerShapeResidual, payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    indices = torch.arange(len(payload["parent_sample_ids"]), dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model(**_inputs(payload, indices, device))
    metrics = _class_metrics(payload, output["sequence_logits"])
    return {"metrics": metrics, "output": output}


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_seed(project_root: str | Path, *, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R9_SEEDS:
        raise ValueError("V27d-r9 seed is not registered")
    train = load_profile_cache(_cache_path(root, config, "train"))
    validation = load_profile_cache(_cache_path(root, config, "val"))
    validate_profile_cache_binding(train, root=root, config=config, split="train")
    validate_profile_cache_binding(validation, root=root, config=config, split="val")
    device = _device(config)
    _seed(seed)
    model = build_model(config["model"], seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    class_weights = torch.tensor(config["training"]["class_weights"], dtype=torch.float32, device=device)
    order_generator = torch.Generator(device="cpu").manual_seed(int(config["training"]["dataloader_seed"]) + int(seed))
    indices = torch.arange(len(train["parent_sample_ids"]), dtype=torch.long)
    batch_size = int(config["training"]["batch_size"])
    steps = (len(indices) + batch_size - 1) // batch_size
    log_path = artifact_path(root, config["artifacts"]["log_template"].format(seed=seed))
    logger = configure_logger(log_path, name=f"v27d.r9.seed.{seed}")
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    primary_state: dict[str, torch.Tensor] | None = None
    primary_epoch = int(config["training"]["primary_epoch"])
    try:
        log_event(logger, "training_start", experiment_id="V27d-r9", seed=seed, device=str(device), train_parents=len(indices), epochs=int(config["training"]["epochs"]), loss=config["training"]["loss_name"], unknown_payload_access_count=0, test_payload_access_count=0)
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            shuffled = indices[torch.randperm(len(indices), generator=order_generator)]
            losses: list[float] = []
            gradient_max = 0.0
            for batch_number, start in enumerate(range(0, len(shuffled), batch_size), start=1):
                batch_indices = shuffled[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(train, batch_indices, device))
                loss = _loss(output["sequence_logits"], train["tensors"]["labels"][batch_indices], train["tensors"]["padding_mask"][batch_indices], class_weights)
                loss.backward()
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                gradient_max = max(gradient_max, norm)
                if batch_number == 1 or batch_number == steps or batch_number % progress_interval(steps) == 0:
                    log_event(logger, "batch", epoch=epoch, batch=f"{batch_number}/{steps}", loss=losses[-1], gradient_norm=norm)
            evaluation = _evaluate(model, validation, device)
            metrics = evaluation["metrics"]
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "gradient_norm_max": gradient_max, "validation_parent_metrics": metrics, "unknown_payload_access_count": 0, "test_payload_access_count": 0}
            history.append(row)
            log_metric_set(logger, "validation_parent_final", metrics)
            per_class = {item["class"]: item for item in metrics["per_class"]}
            if epoch == primary_epoch:
                primary_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            log_event(logger, "epoch_complete", epoch=epoch, train_loss=row["train_loss"], bird_recall=per_class["BirdTarget"]["recall"], macro_f1=metrics["macro_f1"], primary_epoch=primary_epoch, elapsed_seconds=time.perf_counter() - started)
        if primary_state is None:
            raise RuntimeError("V27d-r9 primary epoch state was not captured")
        model.load_state_dict(primary_state, strict=True)
        checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(seed=seed))
        metrics = _evaluate(model, validation, device)["metrics"]
        cache_hashes = {split: load_profile_cache(_cache_path(root, config, split))["cache_sha256"] for split in ("train", "val")}
        save_checkpoint(checkpoint_path, root=root, config=config, manifest=manifest, model=model, seed=seed, epoch=primary_epoch, global_step=primary_epoch * steps, metrics=metrics, cache_hashes=cache_hashes)
        summary = {
            "schema_version": "RADAR-LN3-V27d-r9-training-summary-v1",
            "experiment_id": "V27d-r9",
            "status": "completed",
            "seed": int(seed),
            "selected_epoch": primary_epoch,
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "log_path": relative_project_path(root, log_path),
            "validation_parent_metrics": metrics,
            "history": history,
            "frozen_liquid_v2": True,
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path = artifact_path(root, config["artifacts"]["training_summary_template"].format(seed=seed))
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", selected_epoch=primary_epoch, checkpoint_path=str(checkpoint_path), summary_path=str(summary_path), elapsed_seconds=summary["elapsed_seconds"])
        return {"status": "completed", "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, log_path), "selected_epoch": primary_epoch, "validation_parent_metrics": metrics}
    finally:
        close_logger(logger)


__all__ = ["train_seed"]
