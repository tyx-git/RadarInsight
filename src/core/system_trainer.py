from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataloaders import build_dataloaders
from .dataset import TrackDataset
from .labels import DEFAULT_VOCAB
from .router import ROUTE_NAMES
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .system import RadarClassificationSystem
from .system_config import SystemTrainConfig
from .system_metrics import compute_confusion_matrix, compute_per_class_metrics, compute_system_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and cuda_available:
        device = torch.device("cuda")
    elif requested_device == "cuda" and not cuda_available:
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    info: dict[str, object] = {
        "requested_device": requested_device,
        "resolved_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return device, info


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _build_class_weights(loaders) -> torch.Tensor:
    counts = torch.zeros((len(DEFAULT_VOCAB.labels),), dtype=torch.float32)
    for batch in loaders.train_loader:
        labels = batch.labels
        bincount = torch.bincount(labels, minlength=len(DEFAULT_VOCAB.labels)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.25, max=2.0)
    return weights


def _parameters_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _rd_coverage_from_ids(dataset_frame, sample_ids: list[str]) -> dict[str, int]:
    subset = dataset_frame[dataset_frame["sample_id"].astype(str).isin({str(x) for x in sample_ids})].copy()
    if "rd_summary_csv_path" not in subset.columns:
        return {"with_rd": 0, "without_rd": int(len(subset))}
    has_rd = subset["rd_summary_csv_path"].fillna("").astype(str).str.strip()
    has_rd = has_rd.map(lambda x: x.lower() not in {"", "nan", "none"})
    return {
        "with_rd": int(has_rd.sum()),
        "without_rd": int((~has_rd).sum()),
    }


def _route_name_for_label(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


def _project_to_global_logits(batch_labels: torch.Tensor, system_output, num_classes: int, device: torch.device) -> torch.Tensor:
    logits = torch.full((batch_labels.shape[0], num_classes), fill_value=-6.0, device=device)
    logits[:, :] = -6.0
    label_map = {
        "low_altitude_suspect": {"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
        "conventional_air": {"AircraftTarget": 0},
        "ground_target": {"PeopleTarget": 0, "CarTarget": 1, "ShipTarget": 2},
    }
    global_index = DEFAULT_VOCAB.label_to_index

    low = system_output.low_altitude_logits
    logits[:, global_index["DroneTarget"]] = low[:, 0]
    logits[:, global_index["BirdTarget"]] = low[:, 1]
    logits[:, global_index["BalloonTarget"]] = low[:, 2]
    logits[:, global_index["ClutterTarget"]] = low[:, 3]

    conv = system_output.conventional_air_logits
    logits[:, global_index["AircraftTarget"]] = conv[:, 0]

    ground = system_output.ground_logits
    logits[:, global_index["PeopleTarget"]] = ground[:, 0]
    logits[:, global_index["CarTarget"]] = ground[:, 1]
    logits[:, global_index["ShipTarget"]] = ground[:, 2]
    return logits


def _run_epoch(model, loader, class_criterion, route_criterion, optimizer, device: torch.device, train: bool, skip_non_finite_batch: bool) -> dict[str, object]:
    model.train(mode=train)
    class_logits_all: list[torch.Tensor] = []
    class_labels_all: list[torch.Tensor] = []
    route_logits_all: list[torch.Tensor] = []
    route_labels_all: list[torch.Tensor] = []
    loss_values: list[float] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    route_names_all: list[str] = []
    skipped_non_finite_batches = 0

    for batch in loader:
        features = batch.features.to(device)
        physical = batch.physical_features.to(device)
        rd_features = batch.rd_features.to(device)
        labels = batch.labels.to(device)
        route_labels = batch.route_labels.to(device)
        padding_mask = batch.padding_mask.to(device)
        with torch.set_grad_enabled(train):
            output = model(features, physical, rd_features=rd_features, padding_mask=padding_mask)
            class_logits = _project_to_global_logits(labels, output, len(DEFAULT_VOCAB.labels), device)
            class_loss = class_criterion(class_logits, labels)
            route_loss = route_criterion(output.routing.route_logits, route_labels)
            loss = class_loss + 0.3 * route_loss
            if train:
                prev_state = _clone_state_dict_to_cpu(model)
                if not torch.isfinite(loss):
                    if skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite loss detected during system training")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if not _parameters_are_finite(model):
                    if skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        model.load_state_dict(prev_state)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("model parameters became non-finite during system training")
        loss_values.append(float(loss.detach().cpu().item()))
        class_logits_all.append(class_logits.detach().cpu())
        class_labels_all.append(labels.detach().cpu())
        route_logits_all.append(output.routing.route_logits.detach().cpu())
        route_labels_all.append(route_labels.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        route_names_all.extend(batch.route_names)

    if not class_logits_all:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "route_accuracy": 0.0,
            "class_logits": torch.empty((0, len(DEFAULT_VOCAB.labels)), dtype=torch.float32),
            "class_labels": torch.empty((0,), dtype=torch.long),
            "route_logits": torch.empty((0, len(ROUTE_NAMES)), dtype=torch.float32),
            "route_labels": torch.empty((0,), dtype=torch.long),
            "sample_ids": [],
            "source_names": [],
            "label_names": [],
            "route_names": [],
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
        }

    class_logits_cat = torch.cat(class_logits_all, dim=0)
    class_labels_cat = torch.cat(class_labels_all, dim=0)
    route_logits_cat = torch.cat(route_logits_all, dim=0)
    route_labels_cat = torch.cat(route_labels_all, dim=0)
    metrics = compute_system_metrics(
        class_logits=class_logits_cat,
        class_labels=class_labels_cat,
        route_logits=route_logits_cat,
        route_labels=route_labels_cat,
        num_classes=len(DEFAULT_VOCAB.labels),
    )
    return {
        "loss": float(sum(loss_values) / len(loss_values)),
        "accuracy": metrics.classification.accuracy,
        "macro_precision": metrics.classification.macro_precision,
        "macro_recall": metrics.classification.macro_recall,
        "macro_f1": metrics.classification.macro_f1,
        "route_accuracy": metrics.route_accuracy,
        "class_logits": class_logits_cat,
        "class_labels": class_labels_cat,
        "route_logits": route_logits_cat,
        "route_labels": route_labels_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "route_names": route_names_all,
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
    }


def _metric_summary(epoch_metrics: dict[str, object]) -> dict[str, float]:
    return {
        "loss": float(epoch_metrics["loss"]),
        "accuracy": float(epoch_metrics["accuracy"]),
        "macro_precision": float(epoch_metrics["macro_precision"]),
        "macro_recall": float(epoch_metrics["macro_recall"]),
        "macro_f1": float(epoch_metrics["macro_f1"]),
        "route_accuracy": float(epoch_metrics["route_accuracy"]),
        "skipped_non_finite_batches": int(epoch_metrics.get("skipped_non_finite_batches", 0)),
        "has_valid_batches": bool(epoch_metrics.get("has_valid_batches", True)),
    }


def _route_report(route_logits: torch.Tensor, route_labels: torch.Tensor) -> dict[str, object]:
    route_preds = route_logits.argmax(dim=-1)
    matrix = torch.zeros((len(ROUTE_NAMES), len(ROUTE_NAMES)), dtype=torch.long)
    for true_idx, pred_idx in zip(route_labels.tolist(), route_preds.tolist()):
        matrix[int(true_idx), int(pred_idx)] += 1
    per_route = []
    for idx, route_name in enumerate(ROUTE_NAMES):
        support = int((route_labels == idx).sum().item())
        correct = int(((route_labels == idx) & (route_preds == idx)).sum().item())
        recall = correct / support if support > 0 else 0.0
        per_route.append(
            {
                "route_name": route_name,
                "support": support,
                "correct": correct,
                "recall": recall,
            }
        )
    return {
        "route_names": ROUTE_NAMES,
        "confusion_matrix": matrix.tolist(),
        "per_route": per_route,
    }


def _source_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    source_names: list[str],
) -> dict[str, object]:
    preds = class_logits.argmax(dim=-1)
    grouped: dict[str, list[int]] = {}
    for idx, source_name in enumerate(source_names):
        grouped.setdefault(source_name, []).append(idx)
    report: dict[str, object] = {}
    for source_name, indices in grouped.items():
        true = class_labels[indices]
        pred = preds[indices]
        accuracy = float((true == pred).float().mean().item()) if len(indices) > 0 else 0.0
        report[source_name] = {
            "count": len(indices),
            "accuracy": accuracy,
        }
    return report


def _prediction_records(epoch_metrics: dict[str, object]) -> list[dict[str, object]]:
    class_logits = epoch_metrics["class_logits"]
    route_logits = epoch_metrics["route_logits"]
    class_preds = class_logits.argmax(dim=-1).tolist()
    route_preds = route_logits.argmax(dim=-1).tolist()
    records: list[dict[str, object]] = []
    for idx, sample_id in enumerate(epoch_metrics["sample_ids"]):
        true_label = str(epoch_metrics["label_names"][idx])
        pred_label = DEFAULT_VOCAB.decode(int(class_preds[idx]))
        true_route = str(epoch_metrics["route_names"][idx])
        pred_route = ROUTE_NAMES[int(route_preds[idx])]
        records.append(
            {
                "sample_id": sample_id,
                "source_name": str(epoch_metrics["source_names"][idx]),
                "label_name_true": true_label,
                "label_name_pred": pred_label,
                "route_name_true": true_route,
                "route_name_pred": pred_route,
                "is_correct": pred_label == true_label,
                "is_route_correct": pred_route == true_route,
            }
        )
    return records


def _build_evaluation_report(epoch_metrics: dict[str, object]) -> dict[str, object]:
    class_logits = epoch_metrics["class_logits"]
    class_labels = epoch_metrics["class_labels"]
    route_logits = epoch_metrics["route_logits"]
    route_labels = epoch_metrics["route_labels"]
    per_class = compute_per_class_metrics(class_logits, class_labels, DEFAULT_VOCAB.labels)
    confusion = compute_confusion_matrix(class_logits, class_labels, len(DEFAULT_VOCAB.labels))
    return {
        "summary": _metric_summary(epoch_metrics),
        "label_names": DEFAULT_VOCAB.labels,
        "classification_confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in per_class
        ],
        "route": _route_report(route_logits, route_labels),
        "by_source": _source_report(class_logits, class_labels, epoch_metrics["source_names"]),
        "predictions": _prediction_records(epoch_metrics),
    }


def train_system_v1(config: SystemTrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    run_paths = prepare_run_output_dir(config.output_dir)
    output_dir = run_paths.output_dir
    device, device_info = resolve_device(config.device)

    loaders = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    dataset_frame = TrackDataset(config.data_root, trainable_only=True).to_frame()
    model = RadarClassificationSystem(
        sequence_input_dim=10,
        physical_input_dim=15,
        model_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    trainable_parameters = _count_trainable_parameters(model)
    class_weights = _build_class_weights(loaders).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    route_criterion = nn.CrossEntropyLoss()

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {trainable_parameters:,}")

    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, loaders.train_loader, class_criterion, route_criterion, optimizer, device, True, config.skip_non_finite_batch)
        val_metrics = _run_epoch(model, loaders.val_loader, class_criterion, route_criterion, optimizer, device, False, config.skip_non_finite_batch)
        record = {
            "epoch": epoch,
            "train": _metric_summary(train_metrics),
            "val": _metric_summary(val_metrics),
        }
        history.append(record)
        if not train_metrics.get("has_valid_batches", True) or not val_metrics.get("has_valid_batches", True):
            break
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "labels": DEFAULT_VOCAB.labels,
                    "routes": ROUTE_NAMES,
                },
                best_path,
            )

    if not best_path.exists():
        raise RuntimeError("training finished without a valid checkpoint; all epochs became unstable")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, loaders.test_loader, class_criterion, route_criterion, optimizer, device, False, config.skip_non_finite_batch)
    evaluation_report = _build_evaluation_report(test_metrics)
    evaluation_report["device"] = device_info

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output_dir / "training_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_ids": loaders.split.train_ids,
                "val_ids": loaders.split.val_ids,
                "test_ids": loaders.split.test_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "best_checkpoint": best_path.as_posix(),
        "history_file": history_path.as_posix(),
        "manifest_file": manifest_path.as_posix(),
        "evaluation_report": report_path.as_posix(),
        "test_metrics": _metric_summary(test_metrics),
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": DEFAULT_VOCAB.label_to_index,
        "trainable_parameters": trainable_parameters,
        "class_weights": class_weights.detach().cpu().tolist(),
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage_from_ids(dataset_frame, loaders.split.train_ids),
            "val": _rd_coverage_from_ids(dataset_frame, loaders.split.val_ids),
            "test": _rd_coverage_from_ids(dataset_frame, loaders.split.test_ids),
        },
    }
    report_paths = write_real_experiment_report(
        output_dir=output_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=run_paths.timestamp_label,
    )
    summary["real_report_markdown"] = report_paths.markdown_path.as_posix()
    summary["real_report_json"] = report_paths.json_path.as_posix()
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
