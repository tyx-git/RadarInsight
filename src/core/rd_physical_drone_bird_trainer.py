from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.rd_physical_window_dataset import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    PhysicalRDWindowBatch,
    PhysicalRDWindowDataset,
    collate_physical_rd_windows,
)
from .device import resolve_cuda_device
from .rd_physical_drone_bird_config import PhysicalRDDroneBirdConfig
from .rd_physical_artifacts import build_physical_rd_model, checkpoint_payload, load_physical_rd_artifacts
from .rd_physical_drone_bird_model import PhysicalRDDroneBirdClassifier
from .run_paths import prepare_run_output_dir


RD_ONLY_LABELS = ("DroneTarget", "BirdTarget")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(data_root: Path, config: PhysicalRDDroneBirdConfig, split_name: str, shuffle: bool) -> DataLoader:
    dataset = PhysicalRDWindowDataset(
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
        collate_fn=collate_physical_rd_windows,
        generator=torch.Generator().manual_seed(config.seed + (1 if shuffle else 2)),
    )


def _parent_balance_metadata(dataset: PhysicalRDWindowDataset, config: PhysicalRDDroneBirdConfig) -> dict[str, dict[str, float] | dict[str, int]]:
    frame = dataset.frame.copy()
    frame["parent_id"] = frame["parent_sample_id"].astype(str)
    frame["label"] = frame["label_name"].astype(str)
    frame["group_id"] = frame.get("parent_split_group_id", frame["parent_id"]).astype(str)
    if frame.groupby("parent_id")["label"].nunique().max() > 1:
        raise ValueError("physical RD parent has inconsistent labels")
    if frame.groupby("group_id")["label"].nunique().max() > 1:
        raise ValueError("physical RD content group has inconsistent labels")
    parent_labels = frame.drop_duplicates("parent_id").set_index("parent_id")["label"].to_dict()
    parent_groups = frame.drop_duplicates("parent_id").set_index("parent_id")["group_id"].to_dict()
    group_parent_counts = frame[["group_id", "parent_id"]].drop_duplicates().groupby("group_id").size().to_dict()
    class_group_counts = frame[["label", "group_id"]].drop_duplicates().groupby("label").size().to_dict()
    parent_window_counts = frame["parent_id"].value_counts().to_dict()
    frame_fraction = (frame["rd_window_frame_count"].astype(float) / float(config.evidence_target_frames)).clip(upper=1.0)
    evidence = np.where(
        frame["is_prefix_window"].map(_as_bool).to_numpy(),
        config.prefix_max_loss_weight * frame_fraction.to_numpy(),
        1.0,
    )
    present = frame["rd_present"].map(_as_bool).to_numpy()
    evidence = evidence * present
    evidence_sum = frame.assign(_evidence=evidence).groupby("parent_id")["_evidence"].sum().to_dict()
    window_count = len(frame)
    parent_balance = {
        parent_id: (
            window_count / (
            2.0 * max(1, int(class_group_counts[str(label)]))
            * max(1, int(group_parent_counts[str(frame.loc[frame["parent_id"].eq(parent_id), "group_id"].iloc[0])]))
            )
            if float(evidence_sum.get(parent_id, 0.0)) > 0.0
            else 0.0
        )
        for parent_id, label in parent_labels.items()
    }
    return {
        "parent_window_counts": {str(key): int(value) for key, value in parent_window_counts.items()},
        "parent_evidence_sums": {str(key): float(value) for key, value in evidence_sum.items()},
        "parent_balance_weights": {str(key): float(value) for key, value in parent_balance.items()},
        "parent_labels": {str(key): str(value) for key, value in parent_labels.items()},
        "parent_groups": {str(key): str(value) for key, value in parent_groups.items()},
    }


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _rd_present_mask(batch: PhysicalRDWindowBatch) -> torch.Tensor:
    if batch.quality_features.ndim != 2 or batch.quality_features.shape[1] != QUALITY_FEATURE_DIM:
        raise ValueError("physical RD quality features do not match the active schema")
    present = batch.quality_features[:, QUALITY_FEATURE_NAMES.index("rd_present")]
    if not torch.isfinite(present).all():
        raise ValueError("physical RD presence quality feature must be finite")
    if not torch.isin(present, torch.tensor((0.0, 1.0), dtype=present.dtype, device=present.device)).all():
        raise ValueError("physical RD presence quality feature must be binary")
    return present.bool()


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    if valid_mask.shape != labels.shape or logits.shape[0] != labels.shape[0]:
        raise ValueError("physical RD loss mask must match batch labels")
    if focal_gamma < 0.0:
        raise ValueError("physical RD focal gamma must be non-negative")
    losses = nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if focal_gamma > 0.0:
        probabilities = torch.softmax(logits, dim=-1)
        target_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1.0e-6, 1.0)
        losses = losses * (1.0 - target_probability).pow(focal_gamma)
    return torch.where(valid_mask.bool(), losses, torch.zeros_like(losses))


def compute_parent_evidence_weights(
    batch: PhysicalRDWindowBatch,
    parent_window_counts: dict[str, int],
    *,
    prefix_max_loss_weight: float | None = None,
    evidence_target_frames: int,
    parent_evidence_sums: dict[str, float] | None = None,
    parent_balance_weights: dict[str, float] | None = None,
    prefix_loss_weight: float | None = None,
) -> torch.Tensor:
    if prefix_max_loss_weight is None:
        prefix_max_loss_weight = prefix_loss_weight
    if prefix_max_loss_weight is None or not 0.0 < prefix_max_loss_weight <= 1.0 or evidence_target_frames < 1:
        raise ValueError("invalid parent/prefix weighting configuration")
    frame_fraction = (batch.window_frame_counts.float() / float(evidence_target_frames)).clamp(max=1.0)
    prefix = prefix_max_loss_weight * frame_fraction
    evidence = torch.where(batch.is_prefix_windows, prefix, torch.ones_like(prefix))
    evidence = evidence * _rd_present_mask(batch).to(evidence.dtype)
    if parent_evidence_sums is not None and parent_balance_weights is not None:
        parent = torch.tensor(
            [
                float(parent_balance_weights[str(parent_id)]) / max(1.0e-6, float(parent_evidence_sums[str(parent_id)]))
                for parent_id in batch.parent_sample_ids
            ],
            dtype=torch.float32,
        )
    else:
        parent = torch.tensor(
            [1.0 / max(1, int(parent_window_counts[str(parent_id)])) for parent_id in batch.parent_sample_ids],
            dtype=torch.float32,
        )
    weights = parent * evidence
    if parent_evidence_sums is None or parent_balance_weights is None:
        return weights / weights.mean().clamp_min(1.0e-6)
    return weights


def _epoch_weight_normalizer(balance: dict[str, object], loader_batch_count: int) -> float:
    if loader_batch_count < 1:
        raise ValueError("loader_batch_count must be positive")
    parent_balance_weights = balance.get("parent_balance_weights")
    if not isinstance(parent_balance_weights, dict) or not parent_balance_weights:
        raise ValueError("parent balance metadata lacks parent_balance_weights")
    total_weight = sum(float(value) for value in parent_balance_weights.values())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("parent balance metadata has invalid total weight")
    return total_weight / float(loader_batch_count)


def _class_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    if logits.ndim != 2 or logits.shape[1] != len(RD_ONLY_LABELS):
        raise ValueError("physical RD logits must have shape [sample, 2]")
    if labels.shape != (logits.shape[0],):
        raise ValueError("physical RD labels must match logits sample dimension")
    predictions = logits.argmax(dim=-1)
    per_class: dict[str, dict[str, float]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    confusion = [[0 for _ in RD_ONLY_LABELS] for _ in RD_ONLY_LABELS]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(truth)][int(prediction)] += 1
    for index, name in enumerate(RD_ONLY_LABELS):
        true_positive = int(((predictions == index) & (labels == index)).sum())
        false_positive = int(((predictions == index) & (labels != index)).sum())
        false_negative = int(((predictions != index) & (labels == index)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int((labels == index).sum())}
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    macro_precision = float(sum(precision_values) / len(precision_values))
    macro_recall = float(sum(recall_values) / len(recall_values))
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "balanced_accuracy": macro_recall,
        "sample_count": int(labels.numel()),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(RD_ONLY_LABELS), "matrix": confusion},
        "directional_errors": {
            "DroneTarget_to_BirdTarget": int(confusion[0][1]),
            "BirdTarget_to_DroneTarget": int(confusion[1][0]),
        },
    }


def _weight_audit(
    parent_ids: list[str],
    weights: torch.Tensor,
    balance: dict[str, object],
) -> dict[str, object]:
    parent_labels = balance["parent_labels"]
    parent_groups = balance["parent_groups"]
    parent_counts: defaultdict[str, int] = defaultdict(int)
    parent_weights: defaultdict[str, float] = defaultdict(float)
    for parent_id, weight in zip(parent_ids, weights.tolist()):
        parent_counts[str(parent_id)] += 1
        parent_weights[str(parent_id)] += float(weight)
    group_counts: defaultdict[str, int] = defaultdict(int)
    group_weights: defaultdict[str, float] = defaultdict(float)
    class_counts: defaultdict[str, int] = defaultdict(int)
    class_weights: defaultdict[str, float] = defaultdict(float)
    for parent_id, count in parent_counts.items():
        label = str(parent_labels[parent_id])
        group_id = str(parent_groups[parent_id])
        group_counts[group_id] += count
        group_weights[group_id] += parent_weights[parent_id]
        class_counts[label] += count
        class_weights[label] += parent_weights[parent_id]
    sorted_parent_weights = list(parent_weights.values())
    return {
        "class": {
            label: {
                "window_count": int(class_counts[label]),
                "cumulative_weight": float(class_weights[label]),
            }
            for label in sorted(class_counts)
        },
        "content_group": {
            group_id: {
                "window_count": int(group_counts[group_id]),
                "cumulative_weight": float(group_weights[group_id]),
            }
            for group_id in sorted(group_counts)
        },
        "parent": {
            parent_id: {
                "label_name": str(parent_labels[parent_id]),
                "content_group_id": str(parent_groups[parent_id]),
                "window_count": int(parent_counts[parent_id]),
                "cumulative_weight": float(parent_weights[parent_id]),
            }
            for parent_id in sorted(parent_counts)
        },
        "parent_cumulative_weight": {
            "min": float(min(sorted_parent_weights)) if sorted_parent_weights else 0.0,
            "max": float(max(sorted_parent_weights)) if sorted_parent_weights else 0.0,
            "ratio_max_min": float(max(sorted_parent_weights) / max(min(sorted_parent_weights), 1.0e-12)) if sorted_parent_weights else 0.0,
        },
    }


def _metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    parent_ids: list[str],
    aggregation_weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    if aggregation_weights is None:
        aggregation_weights = torch.ones((logits.shape[0],), dtype=logits.dtype)
    if valid_mask is None:
        valid_mask = torch.ones((logits.shape[0],), dtype=torch.bool)
    if valid_mask.shape != (logits.shape[0],) or labels.shape != (logits.shape[0],):
        raise ValueError("physical RD metrics inputs must have matching batch dimensions")
    if len(parent_ids) != logits.shape[0] or aggregation_weights.shape != (logits.shape[0],):
        raise ValueError("physical RD metrics metadata must match logits batch dimension")
    valid_mask = valid_mask.bool()
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    aggregation_weights = aggregation_weights[valid_mask]
    parent_ids = [parent_id for parent_id, is_valid in zip(parent_ids, valid_mask.tolist()) if is_valid]
    excluded_window_count = int((~valid_mask).sum())
    if logits.shape[0] == 0:
        return {
            "window": None,
            "parent_mean_logit": None,
            "parent_unweighted_mean_logit": None,
            "parent_count": 0,
            "window_count": 0,
            "excluded_window_count": excluded_window_count,
        }
    parent_logits: defaultdict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    parent_labels: dict[str, int] = {}
    for parent_id, logit, label, weight in zip(parent_ids, logits, labels.tolist(), aggregation_weights):
        parent_logits[parent_id].append((logit, weight))
        previous = parent_labels.setdefault(parent_id, int(label))
        if previous != int(label):
            raise ValueError(f"parent has inconsistent labels: {parent_id}")
    stacked_parent_logits = torch.stack(
        [sum((logit * weight for logit, weight in values), torch.zeros_like(values[0][0])) / sum((weight for _, weight in values), values[0][0].new_tensor(0.0)).clamp_min(1.0e-6) for values in parent_logits.values()]
    )
    stacked_parent_labels = torch.tensor(list(parent_labels.values()), dtype=torch.long)
    return {
        "window": _class_metrics(logits, labels),
        "parent_mean_logit": _class_metrics(stacked_parent_logits, stacked_parent_labels),
        "parent_unweighted_mean_logit": _class_metrics(
            torch.stack([torch.stack([logit for logit, _ in values]).mean(dim=0) for values in parent_logits.values()]),
            stacked_parent_labels,
        ),
        "parent_count": int(len(parent_logits)),
        "window_count": int(logits.shape[0]),
        "excluded_window_count": excluded_window_count,
    }


def _forward(model: PhysicalRDDroneBirdClassifier, batch: PhysicalRDWindowBatch, device: torch.device):
    return model(
        batch.features.to(device, non_blocking=True),
        batch.time_seconds.to(device, non_blocking=True),
        batch.padding_mask.to(device, non_blocking=True),
        batch.quality_features.to(device, non_blocking=True),
        batch.range_offset_m.to(device, non_blocking=True),
        batch.velocity_mps.to(device, non_blocking=True),
        batch.raw_amplitude_db.to(device, non_blocking=True),
    )


def _run_epoch(model, loader, optimizer, device, config, balance, phase: str) -> tuple[dict[str, object], float]:
    train = optimizer is not None
    model.train(train)
    max_batches = config.max_train_batches_per_epoch if train else config.max_val_batches
    max_batches = int(max_batches) if max_batches else len(loader)
    loss_normalizer = _epoch_weight_normalizer(balance, len(loader))
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    aggregation_weights_rows: list[torch.Tensor] = []
    present_rows: list[torch.Tensor] = []
    losses: list[float] = []
    component_rows: list[dict[str, float]] = []
    gradient_norm_rows: list[float] = []
    class_weights = torch.tensor([1.0, float(config.bird_loss_weight)], dtype=torch.float32, device=device)
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > max_batches:
            break
        with torch.set_grad_enabled(train):
            output = _forward(model, batch, device)
            labels = batch.labels.to(device, non_blocking=True)
            rd_present = _rd_present_mask(batch).to(device, non_blocking=True)
            per_window_loss = _masked_cross_entropy(
                output.logits,
                labels,
                rd_present,
                class_weights=class_weights.to(dtype=output.logits.dtype),
                focal_gamma=config.focal_gamma,
            )
            aggregation_weights = compute_parent_evidence_weights(
                batch,
                balance["parent_window_counts"],
                prefix_max_loss_weight=config.prefix_max_loss_weight,
                evidence_target_frames=config.evidence_target_frames,
                parent_evidence_sums=balance["parent_evidence_sums"],
                parent_balance_weights=balance["parent_balance_weights"],
            ).to(device)
            final_indices = (batch.lengths.to(device) - 1).clamp_min(0)
            row_indices = torch.arange(output.frame_logits.shape[0], device=device)
            prefix_logits = output.frame_logits[row_indices, final_indices]
            prefix_loss = _masked_cross_entropy(
                prefix_logits,
                labels,
                rd_present,
                class_weights=class_weights.to(dtype=prefix_logits.dtype),
                focal_gamma=config.focal_gamma,
            )
            micro_loss = _masked_cross_entropy(
                output.micro_logits,
                labels,
                rd_present,
                class_weights=class_weights.to(dtype=output.micro_logits.dtype),
                focal_gamma=config.focal_gamma,
            )
            weighted_main = (per_window_loss * aggregation_weights).sum()
            weighted_prefix = (prefix_loss * aggregation_weights).sum()
            weighted_micro = (micro_loss * aggregation_weights).sum()
            main_component = weighted_main / loss_normalizer
            prefix_component = weighted_prefix / loss_normalizer
            micro_component = weighted_micro / loss_normalizer
            loss = (
                weighted_main
                + config.prefix_aux_weight * weighted_prefix
                + config.micro_aux_weight * weighted_micro
            ) / loss_normalizer
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite physical RD {phase} loss at batch={batch_index}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"non-finite physical RD gradient norm at batch={batch_index}")
                gradient_norm_rows.append(float(gradient_norm.detach().cpu()))
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_rows.append(
            {
                "main_ce": float(main_component.detach().cpu()),
                "prefix_ce": float(prefix_component.detach().cpu()),
                "micro_ce": float(micro_component.detach().cpu()),
            }
        )
        logits_rows.append(output.logits.detach().cpu())
        label_rows.append(batch.labels.detach().cpu())
        parent_ids.extend(batch.parent_sample_ids)
        aggregation_weights_rows.append(aggregation_weights.detach().cpu())
        present_rows.append(rd_present.detach().cpu())
        if batch_index == 1 or batch_index % max(1, config.log_interval_batches) == 0 or batch_index == max_batches:
            print(f"[physical-rd][{phase}] batch={batch_index}/{max_batches} loss={losses[-1]:.4f}", flush=True)
    logits = torch.cat(logits_rows)
    labels = torch.cat(label_rows)
    aggregation_weights = torch.cat(aggregation_weights_rows)
    metrics = _metrics(logits, labels, parent_ids, aggregation_weights, torch.cat(present_rows))
    metrics["weight_audit"] = _weight_audit(parent_ids, aggregation_weights, balance)
    metrics["loss_components"] = {
        name: float(np.mean([row[name] for row in component_rows]))
        for name in ("main_ce", "prefix_ce", "micro_ce")
    }
    metrics["gradient_norm"] = (
        {
            "sample_count": len(gradient_norm_rows),
            "min": float(min(gradient_norm_rows)),
            "mean": float(np.mean(gradient_norm_rows)),
            "max": float(max(gradient_norm_rows)),
            "all_finite": True,
        }
        if gradient_norm_rows
        else {
            "sample_count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "all_finite": True,
        }
    )
    return metrics, float(np.mean(losses))


def _is_better(candidate: dict[str, object], best: dict[str, object] | None) -> bool:
    if candidate.get("parent_mean_logit") is None or candidate.get("window") is None:
        return False
    if best is None:
        return True
    if best.get("parent_mean_logit") is None or best.get("window") is None:
        return True
    def score(metrics: dict[str, object]) -> tuple[float, float, float, float]:
        parent = metrics["parent_mean_logit"]
        recalls = [float(values["recall"]) for values in parent["per_class"].values()]
        precisions = [float(values.get("precision", 0.0)) for values in parent["per_class"].values()]
        return (float(parent["macro_f1"]), min(recalls), min(precisions), float(metrics["window"]["macro_f1"]))
    return score(candidate) > score(best)


def train_physical_rd_drone_bird(config: PhysicalRDDroneBirdConfig) -> dict[str, object]:
    if config.run_test_evaluation:
        raise ValueError("physical RD training forbids test evaluation")
    set_seed(config.seed)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    data_root = project_root / "data"
    artifacts = load_physical_rd_artifacts(project_root, config)
    train_loader = _make_loader(data_root, config, "train", shuffle=True)
    val_loader = _make_loader(data_root, config, "val", shuffle=False)
    train_balance = _parent_balance_metadata(train_loader.dataset, config)
    model = build_physical_rd_model(artifacts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    paths = prepare_run_output_dir(config.output_dir)
    history_path = paths.output_dir / "training_history.json"
    progress_path = paths.output_dir / "training_progress.json"
    history: list[dict[str, object]] = []
    best_metrics: dict[str, object] | None = None
    best_checkpoint: Path | None = None
    best_epoch = -1
    data_summary = {
        "train_window_count": int(len(train_loader.dataset)),
        "train_parent_count": int(len(train_balance["parent_window_counts"])),
        "train_batch_count": int(len(train_loader)),
        "validation_window_count": int(len(val_loader.dataset)),
        "validation_parent_count": int(val_loader.dataset.frame["parent_sample_id"].astype(str).nunique()),
        "validation_batch_count": int(len(val_loader)),
        "train_label_counts": {
            str(name): int(count)
            for name, count in train_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().items()
        },
        "validation_label_counts": {
            str(name): int(count)
            for name, count in val_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().items()
        },
        "source_names": (
            sorted(
                set(train_loader.dataset.frame["source_name"].astype(str))
                | set(val_loader.dataset.frame["source_name"].astype(str))
            )
            if "source_name" in train_loader.dataset.frame.columns
            and "source_name" in val_loader.dataset.frame.columns
            else []
        ),
        "session_fields": [
            name
            for name in ("session_id", "session_name")
            if name in train_loader.dataset.frame.columns and name in val_loader.dataset.frame.columns
        ],
        "test_split_loaded": False,
    }
    _write_json(
        progress_path,
        {
            "status": "started",
            "epoch": 0,
            "output_dir": paths.output_dir.as_posix(),
            "config": asdict(config),
            "data": data_summary,
            "checkpoint_selection_rule": [
                "validation parent Macro-F1",
                "minimum validation parent per-class Recall",
                "minimum validation parent per-class Precision",
                "validation window Macro-F1",
            ],
            "test_evaluation_executed": False,
        },
    )
    print(
        f"[physical-rd] READY output_dir={paths.output_dir.as_posix()} "
        f"train_windows={data_summary['train_window_count']} "
        f"val_windows={data_summary['validation_window_count']}",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        print(f"[physical-rd][epoch={epoch}] START", flush=True)
        train_metrics, train_loss = _run_epoch(model, train_loader, optimizer, device, config, train_balance, "train")
        with torch.no_grad():
            val_balance = _parent_balance_metadata(val_loader.dataset, config)
            val_metrics, val_loss = _run_epoch(model, val_loader, None, device, config, val_balance, "val")
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if _is_better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_checkpoint = paths.output_dir / "best_rd_physical_drone_bird_model.pt"
            best_epoch = epoch
            torch.save(
                checkpoint_payload(epoch=epoch, artifacts=artifacts, model=model, validation=val_metrics),
                best_checkpoint,
            )
        _write_json(history_path, history)
        _write_json(
            progress_path,
            {
                "status": "running",
                "epoch": epoch,
                "latest": record,
                "best_epoch": best_epoch,
                "best_checkpoint": best_checkpoint.as_posix() if best_checkpoint else "",
                "best_validation": best_metrics,
                "test_evaluation_executed": False,
            },
        )
        parent_metrics = val_metrics.get("parent_mean_logit")
        parent_f1 = float(parent_metrics["macro_f1"]) if isinstance(parent_metrics, dict) else float("nan")
        print(f"[physical-rd][epoch={epoch}] END train_loss={train_loss:.4f} val_parent_f1={parent_f1:.4f}", flush=True)
    summary = {
        "scope": "physical RD-only Drone/Bird diagnostic; no fusion or test evaluation",
        "config": asdict(config),
        "device": device_info,
        "data": data_summary,
        "artifact_manifest": artifacts.manifest,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history_file": "training_history.json",
        "progress_file": "training_progress.json",
        "checkpoint_selection_rule": [
            "validation parent Macro-F1",
            "minimum validation parent per-class Recall",
            "minimum validation parent per-class Precision",
            "validation window Macro-F1",
        ],
        "supports_rejection": False,
        "test_evaluation_executed": False,
    }
    _write_json(paths.output_dir / "training_summary.json", summary)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "epoch": config.epochs,
            "best_epoch": best_epoch,
            "best_checkpoint": best_checkpoint.as_posix() if best_checkpoint else "",
            "best_validation": best_metrics,
            "test_evaluation_executed": False,
        },
    )
    return summary
