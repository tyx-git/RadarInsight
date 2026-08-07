from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES, PhysicalRDFiveWindowDataset, collate_physical_rd_five_windows
from .data.rd_physical_window_dataset import PhysicalRDWindowBatch
from .device import resolve_cuda_device
from .rd_physical_five_artifacts import (
    build_physical_rd_five_model,
    checkpoint_payload,
    load_physical_rd_five_artifacts,
)
from .rd_physical_five_config import PhysicalRDFiveConfig
from .rd_physical_five_model import PhysicalRDFiveClassifier, PhysicalRDFiveOutput
from .rd_physical_parent_aggregation import mean_parent_probabilities
from .run_paths import prepare_run_output_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(data_root: Path, config: PhysicalRDFiveConfig, split_name: str, shuffle: bool) -> DataLoader:
    dataset = PhysicalRDFiveWindowDataset(
        data_root,
        index_name=config.index_name,
        split_name=split_name,
        normalization_stats_name=config.normalization_stats_name,
        include_missing_rd=config.include_missing_rd,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_physical_rd_five_windows,
        generator=torch.Generator().manual_seed(config.seed + (1 if shuffle else 2)),
    )


def _evidence_weights(batch: PhysicalRDWindowBatch, config: PhysicalRDFiveConfig) -> torch.Tensor:
    """Return per-window loss weights derived only from the window itself.

    Parent/content/sample identifiers are deliberately absent here.  They remain
    available on the batch for split audits and parent-level reporting, but no
    identifier-derived quantity may affect a training loss or sampler.
    """
    evidence = torch.where(
        batch.is_prefix_windows,
        config.prefix_max_loss_weight * (batch.window_frame_counts.float() / float(config.evidence_target_frames)).clamp(max=1.0),
        torch.ones_like(batch.window_frame_counts, dtype=torch.float32),
    )
    present = batch.quality_features[:, RD_FIVE_QUALITY_NAMES.index("rd_present")].bool()
    return evidence * present.to(evidence.dtype)


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor, class_weights: torch.Tensor, gamma: float) -> torch.Tensor:
    losses = nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if gamma > 0.0:
        probability = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1).clamp_min(1.0e-6)
        losses = losses * (1.0 - probability).pow(gamma)
    evidence_weights = weights.to(logits.device)
    weighted = losses * evidence_weights
    # ``cross_entropy(..., weight=..., reduction="mean")`` normalizes by
    # the sum of target-class weights, not by the raw sample count.  Preserve
    # that contract when the additional per-window evidence weight is
    # applied.  Otherwise a rare-class weight also changes the whole batch's
    # gradient scale, making optimization depend on whether that class happens
    # to occur in a small shuffled batch.
    target_class_weights = class_weights.to(logits.device).gather(0, labels)
    effective_weights = evidence_weights * target_class_weights
    valid_weights = torch.where(valid, effective_weights, torch.zeros_like(effective_weights))
    weighted = torch.where(valid, weighted, torch.zeros_like(weighted))
    return weighted.sum() / valid_weights.sum().clamp_min(1.0e-6)


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    if logits.ndim != 2 or logits.shape[1] != len(RD_FIVE_LABELS) or labels.shape != (logits.shape[0],):
        raise ValueError("five-class metrics shape contract failed")
    if labels.numel() == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "sample_count": 0,
            "per_class": {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for label in RD_FIVE_LABELS},
            "confusion_matrix": {"labels": list(RD_FIVE_LABELS), "matrix": [[0 for _ in RD_FIVE_LABELS] for _ in RD_FIVE_LABELS]},
            "directional_errors": {
                "DroneTarget_to_BirdTarget": 0,
                "BirdTarget_to_DroneTarget": 0,
                "BirdTarget_to_BalloonTarget": 0,
                "BirdTarget_to_UnknownTarget": 0,
                "ClutterTarget_to_BirdTarget": 0,
                "ClutterTarget_to_UnknownTarget": 0,
                "UnknownTarget_to_BirdTarget": 0,
                "UnknownTarget_to_ClutterTarget": 0,
            },
        }
    predictions = logits.argmax(dim=-1)
    confusion = [[0 for _ in RD_FIVE_LABELS] for _ in RD_FIVE_LABELS]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(truth)][int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(RD_FIVE_LABELS):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(RD_FIVE_LABELS)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(RD_FIVE_LABELS)) if col != index)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        support = sum(confusion[index])
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "support_stability": "no_support" if support == 0 else ("low" if support < 30 else "adequate"),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "sample_count": int(labels.numel()),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(RD_FIVE_LABELS), "matrix": confusion},
        "directional_errors": {
            "DroneTarget_to_BirdTarget": confusion[0][1],
            "BirdTarget_to_DroneTarget": confusion[1][0],
            "BirdTarget_to_BalloonTarget": confusion[1][2],
            "BirdTarget_to_UnknownTarget": confusion[1][4],
            "ClutterTarget_to_BirdTarget": confusion[3][1],
            "ClutterTarget_to_UnknownTarget": confusion[3][4],
            "UnknownTarget_to_BirdTarget": confusion[4][1],
            "UnknownTarget_to_ClutterTarget": confusion[4][3],
        },
    }


def _selection_record(metrics: dict[str, object], config: PhysicalRDFiveConfig) -> dict[str, object]:
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    references = {
        "DroneTarget": {"precision": config.drone_reference_precision, "recall": config.drone_reference_recall, "f1": config.drone_reference_f1},
        "BirdTarget": {"precision": config.bird_reference_precision, "recall": config.bird_reference_recall, "f1": config.bird_reference_f1},
    }
    checks: dict[str, bool] = {}
    for label, reference in references.items():
        current = per_class[label]
        checks[label] = all(float(current[name]) + config.non_degradation_tolerance >= float(value) for name, value in reference.items())
    parent = metrics["parent_metrics"]["metrics"]
    if int(metrics["sample_count"]) == 0 or int(parent["sample_count"]) == 0:
        return {
            "drone_bird_checks": checks,
            "drone_bird_non_degradation": False,
            "reference_source": config.reference_source,
            "window_macro_f1": 0.0,
            "window_min_class_f1": 0.0,
            "window_min_class_recall": 0.0,
            "window_min_class_precision": 0.0,
            "parent_macro_f1": 0.0,
            "parent_min_class_f1": 0.0,
            "parent_min_class_recall": 0.0,
            "parent_min_class_precision": 0.0,
            "eligible": False,
        }
    parent_per_class = parent["per_class"]
    selection = {
        "drone_bird_checks": checks,
        "drone_bird_non_degradation": bool(all(checks.values())) if config.enforce_drone_bird_non_degradation else True,
        "reference_source": config.reference_source,
        "window_macro_f1": float(metrics["macro_f1"]),
        "window_min_class_f1": min(float(item["f1"]) for item in per_class.values()),
        "window_min_class_recall": min(float(item["recall"]) for item in per_class.values()),
        "window_min_class_precision": min(float(item["precision"]) for item in per_class.values()),
        "parent_macro_f1": float(parent["macro_f1"]),
        "parent_min_class_f1": min(float(item["f1"]) for item in parent_per_class.values()),
        "parent_min_class_recall": min(float(item["recall"]) for item in parent_per_class.values()),
        "parent_min_class_precision": min(float(item["precision"]) for item in parent_per_class.values()),
    }
    selection["eligible"] = bool(selection["drone_bird_non_degradation"])
    return selection


def _selection_key(selection: dict[str, object]) -> tuple[float, ...]:
    return tuple(float(selection[name]) for name in (
        "eligible", "window_macro_f1", "window_min_class_f1", "window_min_class_recall",
        "window_min_class_precision", "parent_macro_f1", "parent_min_class_f1",
        "parent_min_class_recall", "parent_min_class_precision",
    ))


def _parent_metrics(logits: torch.Tensor, labels: torch.Tensor, parent_ids: list[str]) -> dict[str, object]:
    if not parent_ids:
        return {"parent_count": 0, "metrics": _metrics(logits.new_empty((0, len(RD_FIVE_LABELS))), labels.new_empty((0,), dtype=torch.long))}
    parent_probabilities, parent_labels, parent_count = mean_parent_probabilities(logits, labels, parent_ids)
    return {
        "parent_count": parent_count,
        "aggregation": "arithmetic mean of per-window probabilities, then argmax",
        "metrics": _metrics(parent_probabilities, parent_labels),
    }


def _metric_reference(metrics: dict[str, object], config: PhysicalRDFiveConfig) -> dict[str, object]:
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    references = {
        "DroneTarget": {"precision": config.drone_reference_precision, "recall": config.drone_reference_recall, "f1": config.drone_reference_f1},
        "BirdTarget": {"precision": config.bird_reference_precision, "recall": config.bird_reference_recall, "f1": config.bird_reference_f1},
    }
    checks: dict[str, bool] = {}
    for label, reference in references.items():
        current = per_class[label]
        checks[label] = all(float(current[name]) + config.non_degradation_tolerance >= float(value) for name, value in reference.items())
    return {"checks": checks, "met": bool(all(checks.values())) if config.enforce_drone_bird_non_degradation else True, "reference_source": config.reference_source}


def _run_epoch(model: PhysicalRDFiveClassifier, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, config: PhysicalRDFiveConfig, phase: str) -> tuple[dict[str, object], float]:
    training = optimizer is not None
    model.train(training)
    max_batches = config.max_train_batches_per_epoch if training else config.max_val_batches
    max_batches = int(max_batches) if max_batches else len(loader)
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    logits_rows: list[torch.Tensor] = []
    cascade_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    parent_rows: list[str] = []
    losses: list[float] = []
    gradient_norms: list[float] = []
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > max_batches:
            break
        features = batch.features.to(device, non_blocking=True)
        times = batch.time_seconds.to(device, non_blocking=True)
        padding = batch.padding_mask.to(device, non_blocking=True)
        quality = batch.quality_features.to(device, non_blocking=True)
        ranges = batch.range_offset_m.to(device, non_blocking=True)
        velocities = batch.velocity_mps.to(device, non_blocking=True)
        raw_db = batch.raw_amplitude_db.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        weights = _evidence_weights(batch, config).to(device)
        valid = quality[:, RD_FIVE_QUALITY_NAMES.index("rd_present")].bool()
        with torch.set_grad_enabled(training):
            output = model(features, times, padding, quality, ranges, velocities, raw_db)
            flat_loss = _masked_ce(output.flat_logits, labels, valid, weights, class_weights, config.focal_gamma)
            cascade_loss = _masked_ce(output.cascade_logits, labels, valid, weights, class_weights, config.focal_gamma)
            parent_labels = torch.tensor(
                [next(index for index, group in enumerate(config.parent_partition) if int(label) in group) for label in labels.tolist()],
                dtype=torch.long,
                device=device,
            )
            parent_loss = _masked_ce(output.parent_logits, parent_labels, valid, weights, torch.ones(len(config.parent_partition), device=device), config.focal_gamma)
            final_indices = (batch.lengths.to(device) - 1).clamp_min(0)
            row_indices = torch.arange(output.frame_logits.shape[0], device=device)
            prefix_loss = _masked_ce(output.frame_logits[row_indices, final_indices], labels, valid, weights, class_weights, config.focal_gamma)
            micro_loss = _masked_ce(output.micro_logits, labels, valid, weights, class_weights, config.focal_gamma)
            loss = flat_loss + config.cascade_loss_weight * cascade_loss + config.parent_loss_weight * parent_loss + config.prefix_aux_weight * prefix_loss + config.micro_aux_weight * micro_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite five-class RD {phase} loss at batch={batch_index}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"non-finite five-class RD gradient at batch={batch_index}")
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        valid_cpu = valid.detach().cpu()
        logits_rows.append(output.flat_logits.detach().cpu()[valid_cpu])
        cascade_rows.append(output.cascade_logits.detach().cpu()[valid_cpu])
        labels_rows.append(batch.labels.detach().cpu()[valid_cpu])
        parent_rows.extend([parent_id for parent_id, is_valid in zip(batch.parent_sample_ids, valid_cpu.tolist()) if is_valid])
        if batch_index == 1 or batch_index % max(1, config.log_interval_batches) == 0 or batch_index == max_batches:
            print(f"[physical-rd-five][{phase}] batch={batch_index}/{max_batches} loss={losses[-1]:.4f}", flush=True)
    logits = torch.cat(logits_rows) if logits_rows else torch.empty((0, len(RD_FIVE_LABELS)))
    cascade_logits = torch.cat(cascade_rows) if cascade_rows else torch.empty((0, len(RD_FIVE_LABELS)))
    labels = torch.cat(labels_rows) if labels_rows else torch.empty((0,), dtype=torch.long)
    flat_metrics = _metrics(logits, labels)
    cascade_metrics = _metrics(cascade_logits, labels)
    selected_logits = cascade_logits if config.deployment_head == "cascade" else logits
    metrics = _metrics(selected_logits, labels)
    metrics["selected_head"] = config.deployment_head
    metrics["flat_metrics"] = flat_metrics
    metrics["cascade_metrics"] = cascade_metrics
    metrics["parent_metrics"] = _parent_metrics(selected_logits, labels, parent_rows)
    metrics["flat_parent_metrics"] = _parent_metrics(logits, labels, parent_rows)
    metrics["cascade_parent_metrics"] = _parent_metrics(cascade_logits, labels, parent_rows)
    metrics["drone_bird_reference"] = {
        "flat": _metric_reference(flat_metrics, config),
        "cascade": _metric_reference(cascade_metrics, config),
    }
    metrics["gradient_norm"] = {"sample_count": len(gradient_norms), "min": min(gradient_norms) if gradient_norms else None, "mean": float(np.mean(gradient_norms)) if gradient_norms else None, "max": max(gradient_norms) if gradient_norms else None, "all_finite": True}
    return metrics, float(np.mean(losses))


def train_physical_rd_five(config: PhysicalRDFiveConfig) -> dict[str, object]:
    if config.run_test_evaluation:
        raise ValueError("five-class RD training forbids test evaluation")
    set_seed(config.seed)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    data_root = project_root / "data"
    artifacts = load_physical_rd_five_artifacts(project_root, config)
    train_loader = _make_loader(data_root, config, "train", True)
    val_loader = _make_loader(data_root, config, "val", False)
    model = build_physical_rd_five_model(artifacts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    paths = prepare_run_output_dir(config.output_dir)
    history: list[dict[str, object]] = []
    best_metrics: dict[str, object] | None = None
    best_selection: dict[str, object] | None = None
    best_checkpoint: Path | None = None
    best_epoch = -1
    data_summary = {
        "train_window_count": len(train_loader.dataset),
        "validation_window_count": len(val_loader.dataset),
        "train_label_counts": train_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().to_dict(),
        "validation_label_counts": val_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().to_dict(),
        "train_parent_count": train_loader.dataset.frame["parent_sample_id"].astype(str).nunique(),
        "validation_parent_count": val_loader.dataset.frame["parent_sample_id"].astype(str).nunique(),
        "test_split_loaded": False,
        "tr_inputs_loaded": False,
    }
    progress_path = paths.output_dir / "training_progress.json"
    _write_json(progress_path, {"status": "started", "config": asdict(config), "data": data_summary, "test_evaluation_executed": False})
    for epoch in range(1, config.epochs + 1):
        train_metrics, train_loss = _run_epoch(model, train_loader, optimizer, device, config, "train")
        with torch.no_grad():
            val_metrics, val_loss = _run_epoch(model, val_loader, None, device, config, "val")
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics}
        history.append(record)
        selection = _selection_record(val_metrics, config)
        if best_selection is None or _selection_key(selection) > _selection_key(best_selection):
            best_metrics = val_metrics
            best_selection = selection
            best_epoch = epoch
            best_checkpoint = paths.output_dir / "best_rd_physical_five_model.pt"
            torch.save(checkpoint_payload(epoch=epoch, artifacts=artifacts, model=model, validation=val_metrics, selection=selection), best_checkpoint)
        _write_json(paths.output_dir / "training_history.json", history)
        _write_json(progress_path, {"status": "running", "epoch": epoch, "best_epoch": best_epoch, "best_checkpoint": str(best_checkpoint) if best_checkpoint else "", "best_validation": best_metrics, "best_selection": best_selection, "test_evaluation_executed": False})
    summary = {"scope": "physical RD-only five-class training; no TR input, fusion, reject tuning, or test evaluation", "config": asdict(config), "device": device_info, "data": data_summary, "best_checkpoint": str(best_checkpoint) if best_checkpoint else "", "best_epoch": best_epoch, "best_validation": best_metrics, "best_selection": best_selection, "drone_bird_non_degradation_met": bool(best_selection and best_selection.get("drone_bird_non_degradation")), "supports_rejection": False, "test_evaluation_executed": False}
    _write_json(paths.output_dir / "training_summary.json", summary)
    _write_json(progress_path, {"status": "completed", "epoch": config.epochs, "best_epoch": best_epoch, "best_checkpoint": str(best_checkpoint) if best_checkpoint else "", "best_validation": best_metrics, "best_selection": best_selection, "test_evaluation_executed": False})
    return summary


__all__ = ["train_physical_rd_five"]
