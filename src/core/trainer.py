from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataloaders import build_dataloaders
from .labels import DEFAULT_VOCAB
from .metrics import compute_classification_metrics
from .models import GRUSequenceClassifier
from .run_paths import prepare_run_output_dir
from .train_config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool, num_classes: int) -> dict[str, float]:
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    loss_values: list[float] = []

    for batch in loader:
        features = batch.features.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels.to(device)
        with torch.set_grad_enabled(train):
            output = model(features, lengths)
            loss = criterion(output.logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_values.append(float(loss.detach().cpu().item()))
        logits_all.append(output.logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=num_classes)
    return {
        "loss": float(sum(loss_values) / len(loss_values)),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
    }


def train_baseline(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    run_paths = prepare_run_output_dir(config.output_dir)
    output_dir = run_paths.output_dir
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    model = GRUSequenceClassifier(
        input_dim=10,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=len(DEFAULT_VOCAB.labels),
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, loaders.train_loader, criterion, optimizer, device, True, len(DEFAULT_VOCAB.labels))
        val_metrics = _run_epoch(model, loaders.val_loader, criterion, optimizer, device, False, len(DEFAULT_VOCAB.labels))
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "labels": DEFAULT_VOCAB.labels,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, loaders.test_loader, criterion, optimizer, device, False, len(DEFAULT_VOCAB.labels))

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    split_manifest = {
        "train_ids": loaders.split.train_ids,
        "val_ids": loaders.split.val_ids,
        "test_ids": loaders.split.test_ids,
    }
    manifest_path = output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "best_checkpoint": best_path.as_posix(),
        "history_file": history_path.as_posix(),
        "manifest_file": manifest_path.as_posix(),
        "test_metrics": test_metrics,
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
