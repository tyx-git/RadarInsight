from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .device import resolve_cuda_device
from .forwardlayer.dataset import TrackDataset
from .forwardlayer.labels import DEFAULT_VOCAB
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .router import ROUTE_NAMES, DeepGatingRouter
from .router_train_config import RouterTrainConfig
from .run_paths import prepare_run_output_dir
from .forwardlayer.split import stratified_split
from .forwardlayer.torch_dataset import TorchTrackDataset


TRAINABLE_ROUTE_NAMES = ["low_altitude_suspect", "conventional_air", "ground_target"]
TRAINABLE_ROUTE_INDICES = [ROUTE_NAMES.index(route_name) for route_name in TRAINABLE_ROUTE_NAMES]


def _route_name_for_label(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


class RouterSampleDataset(Dataset):
    def __init__(self, data_root: Path | str, sample_ids: list[str] | None = None) -> None:
        base = TrackDataset(data_root, trainable_only=False, index_name="system_trainable_track_index.csv")
        frame = base.to_frame().copy()
        frame["route_name"] = frame["label_name"].astype(str).map(_route_name_for_label)
        frame = frame[frame["route_name"].isin(TRAINABLE_ROUTE_NAMES)].copy()
        if sample_ids is not None:
            keep = {str(x) for x in sample_ids}
            frame = frame[frame["sample_id"].astype(str).isin(keep)].copy()
        self.torch_dataset = TorchTrackDataset(
            data_root,
            sample_ids=frame["sample_id"].astype(str).tolist(),
            trainable_only=False,
            index_name="system_trainable_track_index.csv",
        )
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str, str]:
        row = self.frame.iloc[index]
        item = self.torch_dataset[index]
        if item.sample_id != str(row["sample_id"]):
            raise RuntimeError(f"router dataset order mismatch: {item.sample_id} != {row['sample_id']}")
        route_name = str(row["route_name"])
        return (
            item.physical_features,
            TRAINABLE_ROUTE_NAMES.index(route_name),
            item.label_index,
            str(row["sample_id"]),
            route_name,
        )


def _collate_router(items):
    physical_features = torch.stack([item[0] for item in items], dim=0).float()
    labels = torch.tensor([item[1] for item in items], dtype=torch.long)
    coarse_labels = torch.tensor([item[2] for item in items], dtype=torch.long)
    sample_ids = [item[3] for item in items]
    route_names = [item[4] for item in items]
    return physical_features, labels, coarse_labels, sample_ids, route_names


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _build_route_weights(dataset: RouterSampleDataset) -> torch.Tensor:
    counts = torch.zeros((len(TRAINABLE_ROUTE_NAMES),), dtype=torch.float32)
    for route_name in dataset.frame["route_name"].astype(str).tolist():
        counts[TRAINABLE_ROUTE_NAMES.index(route_name)] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=3.0)


def _build_balanced_sampler(dataset: RouterSampleDataset) -> WeightedRandomSampler:
    counts: dict[str, int] = {}
    route_names = dataset.frame["route_name"].astype(str).tolist()
    for route_name in route_names:
        counts[route_name] = counts.get(route_name, 0) + 1
    weights = [1.0 / float(max(1, counts[route_name])) for route_name in route_names]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _macro_route_recall(labels: torch.Tensor, preds: torch.Tensor) -> float:
    recalls: list[float] = []
    for idx in range(len(TRAINABLE_ROUTE_NAMES)):
        mask = labels == idx
        if not bool(mask.any()):
            continue
        recalls.append(float((preds[mask] == idx).float().mean().item()))
    return float(np.mean(recalls)) if recalls else 0.0


def _router_metric_summary(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_recall": float(metrics["macro_recall"]),
    }


def train_router_v1(config: RouterTrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)

    base = TrackDataset(config.data_root, trainable_only=False, index_name="system_trainable_track_index.csv")
    frame = base.to_frame().copy()
    frame["label_name"] = frame["label_name"].astype(str).map(_route_name_for_label)
    frame = frame[frame["label_name"].isin(TRAINABLE_ROUTE_NAMES)].copy()
    split = stratified_split(frame.rename(columns={"label_name": "label_name"}), seed=config.seed)

    train_ds = RouterSampleDataset(config.data_root, split.train_ids)
    val_ds = RouterSampleDataset(config.data_root, split.val_ids)
    test_ds = RouterSampleDataset(config.data_root, split.test_ids)
    train_sampler = _build_balanced_sampler(train_ds) if config.balanced_sampling else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=_collate_router,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=_collate_router)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=_collate_router)

    model = DeepGatingRouter(input_dim=len(PHYSICAL_FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    route_weights = _build_route_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=None if config.balanced_sampling else route_weights)
    coarse_criterion = nn.CrossEntropyLoss()

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(model):,}")

    def run_epoch(loader, train: bool):
        model.train(mode=train)
        logits_all = []
        labels_all = []
        loss_values = []
        sample_ids_all = []
        route_names_all = []
        override_all = []
        for physical_features, labels, coarse_labels, sample_ids, route_names in loader:
            physical_features = physical_features.to(device)
            labels = labels.to(device)
            coarse_labels = coarse_labels.to(device)
            with torch.set_grad_enabled(train):
                output = model(physical_features, rule_features=physical_features)
                logits = output.network_logits[:, TRAINABLE_ROUTE_INDICES]
                route_loss = criterion(logits, labels)
                coarse_loss = coarse_criterion(output.coarse_logits, coarse_labels)
                loss = route_loss + config.coarse_loss_weight * coarse_loss
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    optimizer.step()
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
            loss_values.append(float(loss.detach().cpu().item()))
            sample_ids_all.extend(sample_ids)
            route_names_all.extend(route_names)
            override_all.extend([bool(x) for x in output.override_mask.detach().cpu().tolist()])
        logits_cat = torch.cat(logits_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        preds = logits_cat.argmax(dim=-1)
        accuracy = float((preds == labels_cat).float().mean().item())
        macro_recall = _macro_route_recall(labels_cat, preds)
        return {
            "loss": float(np.mean(loss_values)),
            "accuracy": accuracy,
            "macro_recall": macro_recall,
            "logits": logits_cat,
            "labels": labels_cat,
            "sample_ids": sample_ids_all,
            "route_names": route_names_all,
            "override_mask": override_all,
        }

    history = []
    best_metric = -1.0
    best_path = run_paths.output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        history.append(
            {
                "epoch": epoch,
                "train": _router_metric_summary(train_metrics),
                "val": _router_metric_summary(val_metrics),
            }
        )
        if val_metrics["macro_recall"] > best_metric:
            best_metric = float(val_metrics["macro_recall"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "trainable_route_names": TRAINABLE_ROUTE_NAMES,
                    "coarse_label_names": DEFAULT_VOCAB.labels,
                    "balanced_sampling": bool(config.balanced_sampling),
                },
                best_path,
            )

    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = run_epoch(test_loader, False)
    preds = test_metrics["logits"].argmax(dim=-1)
    confusion = torch.zeros((len(TRAINABLE_ROUTE_NAMES), len(TRAINABLE_ROUTE_NAMES)), dtype=torch.long)
    for true_idx, pred_idx in zip(test_metrics["labels"].tolist(), preds.tolist()):
        confusion[int(true_idx), int(pred_idx)] += 1

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps({"train_ids": split.train_ids, "val_ids": split.val_ids, "test_ids": split.test_ids}, ensure_ascii=False, indent=2), encoding="utf-8")

    per_route = []
    for idx, route_name in enumerate(TRAINABLE_ROUTE_NAMES):
        support = int((test_metrics["labels"] == idx).sum().item())
        correct = int(((test_metrics["labels"] == idx) & (preds == idx)).sum().item())
        recall = correct / support if support > 0 else 0.0
        per_route.append({"route_name": route_name, "support": support, "correct": correct, "recall": recall})

    evaluation_report = {
        "summary": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_recall"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_recall"],
            "route_accuracy": test_metrics["accuracy"],
        },
        "label_names": TRAINABLE_ROUTE_NAMES,
        "classification_confusion_matrix": confusion.tolist(),
        "per_class": [
            {"label_name": item["route_name"], "support": item["support"], "precision": item["recall"], "recall": item["recall"], "f1": item["recall"]}
            for item in per_route
        ],
        "route": {
            "label_names": TRAINABLE_ROUTE_NAMES,
            "confusion_matrix": confusion.tolist(),
            "per_route": per_route,
        },
        "by_source": {},
        "device": device_info,
        "predictions": [
            {
                "sample_id": sample_id,
                "source_name": "",
                "label_name_true": route_name,
                "label_name_pred": TRAINABLE_ROUTE_NAMES[int(preds[idx].item())],
                "route_name_true": route_name,
                "route_name_pred": TRAINABLE_ROUTE_NAMES[int(preds[idx].item())],
                "is_correct": TRAINABLE_ROUTE_NAMES[int(preds[idx].item())] == route_name,
                "is_route_correct": TRAINABLE_ROUTE_NAMES[int(preds[idx].item())] == route_name,
                "override_mask": bool(test_metrics["override_mask"][idx]),
            }
            for idx, (sample_id, route_name) in enumerate(zip(test_metrics["sample_ids"], test_metrics["route_names"]))
        ],
    }
    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path.as_posix()),
        "history_file": str(history_path.as_posix()),
        "manifest_file": str(manifest_path.as_posix()),
        "evaluation_report": str(evaluation_report_path.as_posix()),
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_recall"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_recall"],
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(TRAINABLE_ROUTE_NAMES)},
        "trainable_parameters": _count_trainable_parameters(model),
        "route_weights": route_weights.detach().cpu().tolist(),
        "balanced_sampling": bool(config.balanced_sampling),
        "coarse_loss_weight": config.coarse_loss_weight,
        "coarse_label_names": DEFAULT_VOCAB.labels,
        "class_weights_enabled": not bool(config.balanced_sampling),
        "max_grad_norm": config.max_grad_norm,
    }
    summary_path = run_paths.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_paths = write_real_experiment_report(
        output_dir=run_paths.output_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=run_paths.timestamp_label,
    )
    summary["real_report_markdown"] = str(report_paths.markdown_path.as_posix())
    summary["real_report_json"] = str(report_paths.json_path.as_posix())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
