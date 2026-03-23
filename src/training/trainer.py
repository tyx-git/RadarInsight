import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.models.fusion_classifier import FusionRadarClassifier
from src.training.config import TrainingConfig
from src.training.dataset import RadarFusionDataset
from src.training.session_manager import build_session_manifest, build_session_subsets, save_session_manifest
from src.utils.logger import logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_classification_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for expected, predicted in zip(labels, predictions):
        confusion[expected, predicted] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for index in range(num_classes):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    divisor = max(num_classes, 1)
    return {
        "accuracy": accuracy,
        "macro_precision": precision_sum / divisor,
        "macro_recall": recall_sum / divisor,
        "macro_f1": f1_sum / divisor,
    }


def _resolve_subset_labels(data_source: Any) -> list[int]:
    if isinstance(data_source, Subset):
        parent = data_source.dataset
        indices = data_source.indices
        return [parent.label_to_index[parent.samples[index].label_name] for index in indices]

    if isinstance(data_source, RadarFusionDataset):
        return [data_source.label_to_index[sample.label_name] for sample in data_source.samples]

    raise TypeError("Unsupported dataset type for label extraction.")


def build_class_balanced_weights(data_source: Any, num_classes: int, beta: float) -> torch.Tensor:
    labels = _resolve_subset_labels(data_source)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num[effective_num <= 0] = 1.0
    weights = (1.0 - beta) / effective_num
    weights[counts == 0] = 0.0
    weights = weights / max(weights.sum(), 1e-8) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if epochs <= 1:
        return None

    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs == 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_dataloaders(
    config: TrainingConfig,
    session_ids: Sequence[str] | None = None,
) -> tuple[RadarFusionDataset, DataLoader, DataLoader | None, DataLoader | None, dict[str, list[str]]]:
    dataset = RadarFusionDataset(
        data_root=config.data_root,
        session_ids=session_ids,
        sequence_length=config.sequence_length,
        rd_frames=config.rd_frames,
        rd_height=config.rd_height,
        rd_width=config.rd_width,
        rd_input_channels=config.rd_input_channels,
        feature_columns=config.track_feature_columns,
        include_rd=config.include_rd,
    )

    if len(dataset) == 1:
        dataset.fit_track_normalization_for_sessions(sorted(dataset.sessions.keys()))
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
        )
        split_map = {"train": sorted(dataset.sessions.keys()), "val": [], "test": []}
        return dataset, train_loader, None, None, split_map

    train_dataset, val_dataset, test_dataset, split_map = build_session_subsets(
        dataset=dataset,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        session_ids=list(session_ids) if session_ids is not None else None,
    )
    dataset.fit_track_normalization_for_sessions(split_map["train"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    return dataset, train_loader, val_loader, test_loader, split_map


def compute_losses(
    outputs: dict[str, Any] | torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    config: TrainingConfig,
    rd_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        logits = outputs
        return logits, criterion(logits, labels)

    logits = outputs["logits"]
    loss = criterion(logits, labels)

    track_logits = outputs.get("track_logits")
    if track_logits is not None and config.track_aux_loss_weight > 0:
        loss = loss + config.track_aux_loss_weight * criterion(track_logits, labels)

    rd_logits = outputs.get("rd_logits")
    if config.include_rd and rd_logits is not None and config.rd_aux_loss_weight > 0:
        if rd_mask is None:
            loss = loss + config.rd_aux_loss_weight * criterion(rd_logits, labels)
        else:
            valid_rd = rd_mask.view(-1) > 0.5
            if valid_rd.any():
                loss = loss + config.rd_aux_loss_weight * criterion(rd_logits[valid_rd], labels[valid_rd])

    return logits, loss


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in data_loader:
        track_features = batch["track_features"].to(device=device, dtype=torch.float32)
        rd_features = None
        rd_mask = None
        if config.include_rd:
            rd_features = batch["rd_features"].to(device=device, dtype=torch.float32)
            rd_mask = batch["has_rd"].to(device=device, dtype=torch.float32).view(-1, 1)
        labels = batch["label"].to(device=device, dtype=torch.long)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(track_features, rd_features, rd_mask)
            logits, loss = compute_losses(outputs, labels, criterion, config, rd_mask)
            if training:
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_predictions, num_classes=max(num_classes, 1))
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_training_artifacts(
    output_dir: Path,
    config: TrainingConfig,
    dataset: RadarFusionDataset,
    history: list[dict[str, Any]],
    num_parameters: int,
    split_map: dict[str, list[str]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "training_metadata.json"
    history_path = output_dir / "training_history.json"

    metadata = {
        "config": config.to_dict(),
        "labels": dataset.index_to_label,
        "num_samples": len(dataset),
        "num_parameters": num_parameters,
        "session_splits": split_map,
        "track_feature_center": dataset.track_feature_center.tolist(),
        "track_feature_scale": dataset.track_feature_scale.tolist(),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    return history_path


def run_training(
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = TrainingConfig.from_dict(config_values)
    set_random_seed(config.random_seed)

    dataset, train_loader, val_loader, test_loader, split_map = build_dataloaders(config, session_ids=session_ids)
    resolved_device = torch.device(device or "cpu")

    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        num_classes=len(dataset.label_to_index),
        include_rd=config.include_rd,
        track_projection_dim=config.track_projection_dim,
        tcn_channels=config.tcn_channels,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_dropout=config.transformer_dropout,
        max_sequence_length=config.sequence_length,
        x3d_in_channels=config.rd_input_channels,
        x3d_stem_channels=config.x3d_stem_channels,
        x3d_stage_channels=config.x3d_stage_channels,
        x3d_stage_depths=config.x3d_stage_depths,
        x3d_spatial_strides=config.x3d_spatial_strides,
        x3d_temporal_strides=config.x3d_temporal_strides,
        x3d_bottleneck_expansion=config.x3d_bottleneck_expansion,
        x3d_se_ratio=config.x3d_se_ratio,
        cross_attention_layers=config.cross_attention_layers,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        fusion_hidden_size=config.fusion_hidden_size,
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = None
    if config.use_class_balanced_loss:
        class_weights = build_class_balanced_weights(
            train_loader.dataset,
            num_classes=len(dataset.label_to_index),
            beta=config.class_balance_beta,
        ).to(resolved_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    scheduler = build_scheduler(optimizer, epochs=config.epochs, warmup_epochs=config.warmup_epochs)
    output_dir = Path(config.output_dir)
    best_checkpoint = output_dir / "best_fusion_model.pt"
    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    num_parameters = count_trainable_parameters(model)

    logger.info(
        f"[*] Training samples: {len(train_loader.dataset)}, "
        f"validation samples: {0 if val_loader is None else len(val_loader.dataset)}, "
        f"classes: {dataset.index_to_label}"
    )
    logger.info(
        f"[*] Session split | train={split_map['train']} | val={split_map['val']} | test={split_map['test']}"
    )
    logger.info(f"[*] Using device: {resolved_device}")
    logger.info(f"[*] Trainable parameters: {num_parameters:,}")
    if class_weights is not None:
        logger.info(f"[*] Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=resolved_device,
            config=config,
            num_classes=len(dataset.label_to_index),
            optimizer=optimizer,
        )

        if val_loader is None:
            val_metrics = train_metrics
        else:
            val_metrics = run_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=resolved_device,
                config=config,
                num_classes=len(dataset.label_to_index),
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        logger.info(
            f"[*] Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if scheduler is not None:
            scheduler.step()

        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "label_to_index": dataset.label_to_index,
                    "best_macro_f1": best_macro_f1,
                    "num_parameters": num_parameters,
                    "track_feature_center": dataset.track_feature_center.tolist(),
                    "track_feature_scale": dataset.track_feature_scale.tolist(),
                },
                best_checkpoint,
            )

    history_path = save_training_artifacts(output_dir, config, dataset, history, num_parameters, split_map)
    manifest = build_session_manifest(data_root=config.data_root, include_rd=config.include_rd, split_map=split_map)
    manifest_path = save_session_manifest(manifest, output_dir)

    if best_checkpoint.exists():
        checkpoint_payload = torch.load(best_checkpoint, map_location=resolved_device)
        model.load_state_dict(checkpoint_payload["model_state_dict"])

    test_metrics = None
    if test_loader is not None:
        test_metrics = run_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=resolved_device,
            config=config,
            num_classes=len(dataset.label_to_index),
        )
        logger.info(
            f"[*] Test metrics | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f} | f1={test_metrics['macro_f1']:.4f}"
        )

    return {
        "best_checkpoint": str(best_checkpoint),
        "history_path": str(history_path),
        "manifest_path": str(manifest_path),
        "num_samples": len(dataset),
        "labels": dataset.index_to_label,
        "num_parameters": num_parameters,
        "session_splits": split_map,
        "test_metrics": test_metrics,
    }
