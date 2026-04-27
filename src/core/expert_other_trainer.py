from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experts import ConventionalAirExpert, GroundTargetExpert
from .expert_other_config import ExpertOtherTrainConfig
from .expert_other_data import build_subset_dataloaders
from .metrics import compute_classification_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .trajectory import TrackRepresentationBuilder


EXPERT_LABELS = {
    "expert2_air": ["AircraftTarget"],
    "expert4_ground": ["PeopleTarget", "CarTarget", "ShipTarget"],
}


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


def _set_track_encoder_trainable(representation_builder: TrackRepresentationBuilder, trainable: bool) -> None:
    for param in representation_builder.sequence_encoder.parameters():
        param.requires_grad = trainable


def _load_pretrained_track_encoder(
    representation_builder: TrackRepresentationBuilder,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {
        key[len("encoder."):] if key.startswith("encoder.") else key: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"no encoder.* weights found in pretrain checkpoint: {checkpoint_path}")
    missing, unexpected = representation_builder.sequence_encoder.load_state_dict(encoder_state, strict=False)
    return {
        "loaded_keys": len(encoder_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _load_shared_representation_builder(
    representation_builder: TrackRepresentationBuilder,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "representation_builder_state_dict" in checkpoint:
        state_dict = checkpoint["representation_builder_state_dict"]
    else:
        raw_state = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key[len("representation_builder."):] if key.startswith("representation_builder.") else key: value
            for key, value in raw_state.items()
            if key.startswith("representation_builder.")
        }
    if not state_dict:
        encoder_state = {
            key[len("encoder."):] if key.startswith("encoder.") else key: value
            for key, value in checkpoint.get("model_state_dict", checkpoint).items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(f"no representation_builder or encoder.* weights found in shared checkpoint: {checkpoint_path}")
        missing, unexpected = representation_builder.sequence_encoder.load_state_dict(encoder_state, strict=False)
        return {
            "loaded_keys": len(encoder_state),
            "loaded_scope": "sequence_encoder",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    missing, unexpected = representation_builder.load_state_dict(state_dict, strict=False)
    return {
        "loaded_keys": len(state_dict),
        "loaded_scope": "representation_builder",
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _fit_source_feature_stats(loader) -> dict[str, dict[str, torch.Tensor]]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        for idx, source_name in enumerate(batch.source_names):
            buckets.setdefault(str(source_name), []).append(batch.features[idx, : batch.lengths[idx]].float())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, tensors in buckets.items():
        cat = torch.cat(tensors, dim=0)
        stats[source_name] = {
            "center": cat.mean(dim=0),
            "scale": cat.std(dim=0).clamp_min(1e-3),
        }
    return stats


def _apply_source_feature_norm(features: torch.Tensor, source_names: list[str], stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    normalized = features.clone()
    for idx, source_name in enumerate(source_names):
        source_key = str(source_name)
        if source_key not in stats:
            continue
        center = stats[source_key]["center"].to(features.device)
        scale = stats[source_key]["scale"].to(features.device)
        normalized[idx] = (normalized[idx] - center) / scale
    return normalized


def _source_feature_stats_from_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, dict[str, torch.Tensor]] | None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stats = checkpoint.get("source_feature_stats")
    if not stats:
        return None
    return stats


def _merge_source_feature_stats(*stats_items: dict[str, dict[str, torch.Tensor]] | None) -> dict[str, dict[str, torch.Tensor]]:
    merged: dict[str, dict[str, torch.Tensor]] = {}
    for stats in stats_items:
        if not stats:
            continue
        for source_name, source_stats in stats.items():
            merged[str(source_name)] = {
                "center": source_stats["center"].detach().cpu()
                if isinstance(source_stats["center"], torch.Tensor)
                else torch.tensor(source_stats["center"], dtype=torch.float32),
                "scale": source_stats["scale"].detach().cpu()
                if isinstance(source_stats["scale"], torch.Tensor)
                else torch.tensor(source_stats["scale"], dtype=torch.float32),
            }
    return merged


def _run_epoch(
    representation_builder,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    global_to_local: dict[int, int],
    labels: list[str],
    expert_name: str,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, object]:
    representation_builder.train(mode=train)
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    sample_ids_all: list[str] = []
    loss_values: list[float] = []

    for batch in loader:
        features = batch.features.to(device)
        if source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = batch.physical_features.to(device)
        if expert_name == "expert2_air":
            labels_local = torch.tensor(
                [1 if name == "AircraftTarget" else 0 for name in batch.label_names],
                dtype=torch.long,
                device=device,
            )
        else:
            labels_local = torch.tensor(
                [global_to_local[int(label)] for label in batch.labels.tolist()],
                dtype=torch.long,
                device=device,
            )
        with torch.set_grad_enabled(train):
            if train and any(param.requires_grad for param in representation_builder.parameters()):
                rep = representation_builder(
                    sequence_features=features,
                    physical_features=physical,
                    padding_mask=batch.padding_mask.to(device),
                )
            else:
                with torch.no_grad():
                    rep = representation_builder(
                        sequence_features=features,
                        physical_features=physical,
                        padding_mask=batch.padding_mask.to(device),
                    )
            if expert_name == "expert2_air":
                output = model(rep.sequence_features, rep.raw_physical_features, padding_mask=batch.padding_mask.to(device))
            else:
                output = model(rep.deep_features, rep.raw_physical_features)
            loss = criterion(output.logits, labels_local)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        logits_all.append(output.logits.detach().cpu())
        labels_all.append(labels_local.detach().cpu())
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        sample_ids_all.extend(batch.sample_ids)
        loss_values.append(float(loss.detach().cpu().item()))

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=len(labels))
    return {
        "loss": float(np.mean(loss_values)),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "logits": logits_cat,
        "labels": labels_cat,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "sample_ids": sample_ids_all,
    }


def train_other_expert(config: ExpertOtherTrainConfig) -> dict[str, object]:
    if config.expert_name not in EXPERT_LABELS:
        raise ValueError(f"unsupported expert_name: {config.expert_name}")
    requested_labels = EXPERT_LABELS[config.expert_name]
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loaders = build_subset_dataloaders(
        data_root=config.data_root,
        labels=requested_labels,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    labels = loaders.labels

    if config.expert_name == "expert2_air":
        model = ConventionalAirExpert(sequence_dim=128, physical_dim=len(PHYSICAL_FEATURE_COLUMNS), dropout=config.dropout).to(device)
    else:
        model = GroundTargetExpert(deep_dim=128, raw_physical_dim=10, dropout=config.dropout).to(device)
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    pretrained_track_info = None
    shared_representation_info = None
    if config.shared_representation_checkpoint:
        shared_representation_info = _load_shared_representation_builder(
            representation_builder,
            config.shared_representation_checkpoint,
            device,
        )
    if config.pretrained_track_checkpoint and not config.shared_representation_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(
            representation_builder,
            config.pretrained_track_checkpoint,
            device,
        )
    if not config.train_representation_builder:
        for param in representation_builder.parameters():
            param.requires_grad = False

    optimizer_params = list(model.parameters())
    if config.train_representation_builder:
        optimizer_params = list(representation_builder.parameters()) + optimizer_params
    optimizer = torch.optim.AdamW(optimizer_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    source_feature_stats = None
    checkpoint_source_feature_stats = None
    if config.source_normalization:
        fitted_source_feature_stats = _fit_source_feature_stats(loaders.train_loader)
        checkpoint_source_feature_stats = _merge_source_feature_stats(
            _source_feature_stats_from_checkpoint(config.shared_representation_checkpoint, device)
            if config.shared_representation_checkpoint
            else None,
            _source_feature_stats_from_checkpoint(config.pretrained_track_checkpoint, device)
            if config.pretrained_track_checkpoint
            else None,
        )
        source_feature_stats = _merge_source_feature_stats(fitted_source_feature_stats, checkpoint_source_feature_stats)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(representation_builder) + _count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    best_val = -1.0
    best_path = run_paths.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        _set_track_encoder_trainable(representation_builder, bool(config.train_representation_builder))
        train_metrics = _run_epoch(representation_builder, model, loaders.train_loader, criterion, optimizer, device, True, loaders.global_to_local, labels, config.expert_name, source_feature_stats)
        val_metrics = _run_epoch(representation_builder, model, loaders.val_loader, criterion, optimizer, device, False, loaders.global_to_local, labels, config.expert_name, source_feature_stats)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["macro_f1"] > best_val:
            best_val = float(val_metrics["macro_f1"])
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "labels": labels,
                "source_feature_stats": source_feature_stats,
                "shared_representation_checkpoint": config.shared_representation_checkpoint,
                "train_representation_builder": bool(config.train_representation_builder),
            }
            if config.train_representation_builder:
                checkpoint["representation_builder_state_dict"] = representation_builder.state_dict()
            torch.save(checkpoint, best_path)

    state = torch.load(best_path, map_location=device, weights_only=True)
    if "representation_builder_state_dict" in state:
        representation_builder.load_state_dict(state["representation_builder_state_dict"])
    model.load_state_dict(state["model_state_dict"])
    test_metrics = _run_epoch(representation_builder, model, loaders.test_loader, criterion, optimizer, device, False, loaders.global_to_local, labels, config.expert_name, source_feature_stats)

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {
        "labels": labels,
        "train_ids": loaders.split.train_ids,
        "val_ids": loaders.split.val_ids,
        "test_ids": loaders.split.test_ids,
    }
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    preds = test_metrics["logits"].argmax(dim=-1)
    for true_idx, pred_idx in zip(test_metrics["labels"].tolist(), preds.tolist()):
        confusion[int(true_idx), int(pred_idx)] += 1

    evaluation_report = {
        "summary": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "route_accuracy": 1.0,
        },
        "label_names": labels,
        "classification_confusion_matrix": confusion.tolist(),
        "per_class": [],
        "route": {
            "label_names": [config.expert_name],
            "confusion_matrix": [[len(test_metrics["sample_ids"])]],
            "per_route": [{"route_name": config.expert_name, "support": len(test_metrics["sample_ids"]), "correct": len(test_metrics["sample_ids"]), "recall": 1.0}],
        },
        "by_source": {},
        "device": device_info,
        "predictions": [],
    }
    for idx, label in enumerate(labels):
        pred_pos = preds == idx
        true_pos = test_metrics["labels"] == idx
        tp = int((pred_pos & true_pos).sum().item())
        fp = int((pred_pos & ~true_pos).sum().item())
        fn = int((~pred_pos & true_pos).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        evaluation_report["per_class"].append({"label_name": label, "support": int(true_pos.sum().item()), "precision": precision, "recall": recall, "f1": f1})

    for source_name in sorted(set(test_metrics["source_names"])):
        indices = [i for i, s in enumerate(test_metrics["source_names"]) if s == source_name]
        if indices:
            acc = float((preds[indices] == test_metrics["labels"][indices]).float().mean().item())
            evaluation_report["by_source"][source_name] = {"count": len(indices), "accuracy": acc}

    for idx, sample_id in enumerate(test_metrics["sample_ids"]):
        evaluation_report["predictions"].append(
            {
                "sample_id": sample_id,
                "source_name": test_metrics["source_names"][idx],
                "label_name_true": test_metrics["label_names"][idx],
                "label_name_pred": labels[int(preds[idx].item())],
                "route_name_true": config.expert_name,
                "route_name_pred": config.expert_name,
                "is_correct": labels[int(preds[idx].item())] == test_metrics["label_names"][idx],
                "is_route_correct": True,
            }
        )

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
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(labels)},
        "trainable_parameters": _count_trainable_parameters(representation_builder) + _count_trainable_parameters(model),
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "shared_representation_checkpoint": config.shared_representation_checkpoint,
        "shared_representation_info": shared_representation_info,
        "train_representation_builder": bool(config.train_representation_builder),
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "source_feature_stats_from_checkpoint": bool(checkpoint_source_feature_stats),
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
