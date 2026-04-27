from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experts import LOW_ALTITUDE_CLASSES, LowAltitudeSuspectExpert
from .expert1_config import Expert1TrainConfig
from .expert1_data import LOW_ALTITUDE_GLOBAL_TO_LOCAL, build_expert1_dataloaders
from .metrics import compute_classification_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .system_metrics import compute_confusion_matrix, compute_per_class_metrics
from .trajectory import TrackRepresentationBuilder


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


def _build_optimizer(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
    config: Expert1TrainConfig,
) -> torch.optim.Optimizer:
    sequence_params = [
        param
        for param in representation_builder.sequence_encoder.parameters()
        if param.requires_grad
    ]
    sequence_param_ids = {id(param) for param in sequence_params}
    other_params = [
        param
        for param in list(representation_builder.parameters()) + list(rd_proj.parameters()) + list(model.parameters())
        if param.requires_grad and id(param) not in sequence_param_ids
    ]
    param_groups: list[dict[str, object]] = []
    if sequence_params:
        param_groups.append(
            {
                "params": sequence_params,
                "lr": config.learning_rate * config.sequence_encoder_lr_multiplier,
                "weight_decay": config.weight_decay,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def _build_class_weights(loader) -> torch.Tensor:
    counts = torch.zeros((len(LOW_ALTITUDE_CLASSES),), dtype=torch.float32)
    for batch in loader:
        local_labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
        )
        bincount = torch.bincount(local_labels, minlength=len(LOW_ALTITUDE_CLASSES)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=2.0)


def _fit_source_feature_stats(loader) -> dict[str, dict[str, torch.Tensor]]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        for idx, source_name in enumerate(batch.source_names):
            buckets.setdefault(str(source_name), []).append(batch.features[idx, : batch.lengths[idx]].float())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, tensors in buckets.items():
        cat = torch.cat(tensors, dim=0)
        center = cat.mean(dim=0)
        scale = cat.std(dim=0).clamp_min(1e-3)
        stats[source_name] = {"center": center, "scale": scale}
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


def _parameters_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _clone_bundle_state_to_cpu(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "representation_builder_state_dict": _clone_state_dict_to_cpu(representation_builder),
        "rd_proj_state_dict": _clone_state_dict_to_cpu(rd_proj),
        "model_state_dict": _clone_state_dict_to_cpu(model),
    }


def _parameters_are_finite_bundle(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
) -> bool:
    return (
        _parameters_are_finite(representation_builder)
        and _parameters_are_finite(rd_proj)
        and _parameters_are_finite(model)
    )


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


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_a, dim=-1)
    log_q = torch.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = torch.sum(p * (log_p - log_q), dim=-1)
    kl_qp = torch.sum(q * (log_q - log_p), dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def _supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    features = torch.nn.functional.normalize(embeddings, dim=-1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(mask, -1e9)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & (~mask)
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=-1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=-1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _rd_coverage(loader) -> dict[str, int]:
    with_rd = 0
    without_rd = 0
    for batch in loader:
        has_rd = batch.rd_is_multimodal.bool()
        with_rd += int(has_rd.sum().item())
        without_rd += int((~has_rd).sum().item())
    return {"with_rd": with_rd, "without_rd": without_rd}


def _subset_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.numel() == 0 or not bool(mask.any()):
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[mask] == labels[mask]).float().mean().item())


def _weighted_ce_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    unknown_index: int,
    unknown_weight: float,
) -> torch.Tensor:
    per_sample = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=criterion.weight,
        reduction="none",
    )
    weights = torch.ones_like(per_sample)
    weights = torch.where(labels == unknown_index, torch.full_like(weights, unknown_weight), weights)
    return (per_sample * weights).mean()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.as_posix())


def _epoch_metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "skipped_non_finite_batches": metrics["skipped_non_finite_batches"],
        "has_valid_batches": metrics["has_valid_batches"],
    }


def _stage2_scale(epoch: int, config: Expert1TrainConfig) -> float:
    if epoch <= config.stage1_traj_only_epochs:
        return 0.0
    if config.stage2_ramp_epochs <= 0:
        return 1.0
    offset = epoch - config.stage1_traj_only_epochs
    return float(min(1.0, max(0.0, offset / float(config.stage2_ramp_epochs))))


def _run_epoch(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    config: Expert1TrainConfig,
    stage2_enabled: bool,
    stage2_scale: float,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, object]:
    representation_builder.train(mode=train)
    rd_proj.train(mode=train)
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    has_rd_all: list[bool] = []
    loss_values: list[float] = []
    skipped_non_finite_batches = 0

    for batch in loader:
        features = batch.features.to(device)
        if config.source_normalization and source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = batch.physical_features.to(device)
        rd_features = batch.rd_features.to(device)
        rd_is_multimodal = batch.rd_is_multimodal.to(device)
        rd_padding_mask = batch.rd_padding_mask.to(device)
        labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
            device=device,
        )

        with torch.set_grad_enabled(train):
            rep = representation_builder(
                sequence_features=features,
                physical_features=physical,
                padding_mask=batch.padding_mask.to(device),
            )
            if rd_features.ndim == 4:
                amp_range = rd_features[..., 0].mean(dim=1)
                amp_dopp = rd_features[..., 2].mean(dim=1)
                rd_summary = torch.stack(
                    [
                        amp_range.mean(dim=-1),
                        amp_range.max(dim=-1).values,
                        amp_range.min(dim=-1).values,
                        amp_dopp.mean(dim=-1),
                        amp_dopp.std(dim=-1),
                        rd_is_multimodal.float(),
                    ],
                    dim=-1,
                )
            else:
                rd_summary = rd_features.mean(dim=1)
            rd_embedding = rd_proj(rd_summary)
            fused_input = torch.cat([rep.fused_features, rd_embedding], dim=-1)
            output = model(
                fused_input,
                rd_features=rd_features,
                low_altitude_targets=labels if train else None,
                rd_is_multimodal=rd_is_multimodal,
                rd_padding_mask=rd_padding_mask,
                enable_reject=(config.enable_reject_in_train if train else config.enable_reject_in_eval) and stage2_enabled,
            )
            unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
            main_loss = _weighted_ce_loss(
                criterion,
                output.logits,
                labels,
                unknown_index=unknown_index,
                unknown_weight=config.unknown_loss_weight,
            )
            aux_loss = _weighted_ce_loss(
                criterion,
                output.traj_logits,
                labels,
                unknown_index=unknown_index,
                unknown_weight=config.unknown_loss_weight,
            )
            has_rd_mask = rd_is_multimodal.bool()
            if stage2_enabled and has_rd_mask.any():
                rd_aux = _weighted_ce_loss(
                    criterion,
                    output.rd_logits[has_rd_mask],
                    labels[has_rd_mask],
                    unknown_index=unknown_index,
                    unknown_weight=config.unknown_loss_weight,
                )
                aux_loss = 0.5 * (aux_loss + rd_aux)
                mutual_loss = _symmetric_kl(
                    output.traj_logits[has_rd_mask],
                    output.rd_logits[has_rd_mask],
                )
                contrastive_mask = has_rd_mask & (labels != unknown_index)
                if contrastive_mask.any():
                    contrastive_loss = _supervised_contrastive_loss(
                        torch.cat(
                            [
                                output.traj_features[contrastive_mask],
                                output.rd_features[contrastive_mask],
                            ],
                            dim=-1,
                        ),
                        labels[contrastive_mask],
                        temperature=config.low_altitude_temperature,
                    )
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            else:
                mutual_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)
            loss = (
                main_loss
                + (config.low_altitude_aux_weight * stage2_scale) * aux_loss
                + (config.low_altitude_mutual_weight * stage2_scale) * mutual_loss
                + (config.low_altitude_contrastive_weight * stage2_scale) * contrastive_loss
            )

            if train:
                prev_state = _clone_bundle_state_to_cpu(representation_builder, rd_proj, model)
                if not torch.isfinite(loss):
                    if config.skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite loss detected during expert1 training")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(representation_builder.parameters()) + list(rd_proj.parameters()) + list(model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                if not _parameters_are_finite_bundle(representation_builder, rd_proj, model):
                    representation_builder.load_state_dict(prev_state["representation_builder_state_dict"])
                    rd_proj.load_state_dict(prev_state["rd_proj_state_dict"])
                    model.load_state_dict(prev_state["model_state_dict"])
                    skipped_non_finite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                if not torch.isfinite(loss):
                    skipped_non_finite_batches += 1
                    continue

        loss_values.append(float(loss.detach().cpu().item()))
        logits_all.append(output.logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        has_rd_all.extend([bool(x) for x in batch.rd_is_multimodal.tolist()])

    if not logits_all:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
            "logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "sample_ids": sample_ids_all,
            "source_names": source_names_all,
            "label_names": label_names_all,
            "has_rd": has_rd_all,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    return {
        "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
        "logits": logits_cat,
        "labels": labels_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "has_rd": has_rd_all,
    }


def train_expert1_v1(config: Expert1TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loaders = build_expert1_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
        unknown_sample_boost=config.unknown_sample_boost,
    )
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    rd_dim = 16
    rd_proj = nn.Sequential(
        nn.Linear(6, rd_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(rd_dim, rd_dim),
        nn.LayerNorm(rd_dim),
    ).to(device)
    model = LowAltitudeSuspectExpert(
        input_dim=128 + 32 + rd_dim,
        dropout=config.dropout,
        cosface_scale=config.cosface_scale,
        cosface_margin=config.cosface_margin,
        reject_threshold=config.reject_threshold,
        rd_activation_threshold=config.rd_activation_threshold,
    ).to(device)
    pretrained_track_info = None
    if config.pretrained_track_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(
            representation_builder,
            config.pretrained_track_checkpoint,
            device,
        )
    class_weights = _build_class_weights(loaders.train_loader).to(device)
    source_feature_stats = _fit_source_feature_stats(loaders.train_loader) if config.source_normalization else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(representation_builder, rd_proj, model, config)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    best_val = -1.0
    best_path = run_paths.output_dir / "best_model.pt"
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        stage2_enabled = epoch > config.stage1_traj_only_epochs
        stage2_scale = _stage2_scale(epoch, config)
        _set_track_encoder_trainable(representation_builder, epoch > config.freeze_track_encoder_epochs)
        train_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.train_loader, criterion, optimizer, device, True, config, stage2_enabled, stage2_scale, source_feature_stats)
        if not train_metrics["has_valid_batches"]:
            break
        val_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.val_loader, criterion, optimizer, device, False, config, stage2_enabled, stage2_scale, source_feature_stats)
        history.append(
            {
                "epoch": epoch,
                "stage2_enabled": stage2_enabled,
                "stage2_scale": stage2_scale,
                "train": _epoch_metric_summary(train_metrics),
                "val": _epoch_metric_summary(val_metrics),
            }
        )
        print(
            f"[*] epoch={epoch:02d} "
            f"stage2={'on' if stage2_enabled else 'off'} "
            f"scale={stage2_scale:.2f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val:
            best_val = float(val_metrics["macro_f1"])
            best_epoch = epoch
            torch.save(
                {
                    "representation_builder_state_dict": representation_builder.state_dict(),
                    "rd_proj_state_dict": rd_proj.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "labels": LOW_ALTITUDE_CLASSES,
                    "source_feature_stats": source_feature_stats,
                },
                best_path,
            )

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=True)
        representation_builder.load_state_dict(state["representation_builder_state_dict"])
        rd_proj.load_state_dict(state["rd_proj_state_dict"])
        model.load_state_dict(state["model_state_dict"])
    test_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.test_loader, criterion, optimizer, device, False, config, True, 1.0, source_feature_stats)

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    manifest = {
        "labels": LOW_ALTITUDE_CLASSES,
        "train_ids": loaders.split.train_ids,
        "val_ids": loaders.split.val_ids,
        "test_ids": loaders.split.test_ids,
    }
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation_report = {
        "summary": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "route_accuracy": 1.0,
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
        },
        "label_names": LOW_ALTITUDE_CLASSES,
        "classification_confusion_matrix": compute_confusion_matrix(
            test_metrics["logits"],
            test_metrics["labels"],
            len(LOW_ALTITUDE_CLASSES),
        ).tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in compute_per_class_metrics(
                test_metrics["logits"],
                test_metrics["labels"],
                LOW_ALTITUDE_CLASSES,
            )
        ],
        "route": {
            "label_names": ["low_altitude_suspect"],
            "confusion_matrix": [[len(test_metrics["sample_ids"])]],
            "per_route": [
                {
                    "route_name": "low_altitude_suspect",
                    "support": len(test_metrics["sample_ids"]),
                    "correct": len(test_metrics["sample_ids"]),
                    "recall": 1.0,
                }
            ],
        },
        "by_source": {
            source_name: {
                "count": sum(1 for x in test_metrics["source_names"] if x == source_name),
                "accuracy": 0.0,
            }
            for source_name in sorted(set(test_metrics["source_names"]))
        },
        "by_rd_presence": {},
        "device": device_info,
        "predictions": [
            {
                "sample_id": sid,
                "source_name": src,
                "label_name_true": true_label,
                "label_name_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["logits"][idx].argmax().item())],
                "route_name_true": "low_altitude_suspect",
                "route_name_pred": "low_altitude_suspect",
                "is_correct": LOW_ALTITUDE_CLASSES[int(test_metrics["logits"][idx].argmax().item())] == true_label,
                "is_route_correct": True,
                "has_rd": rd_flag,
            }
            for idx, (sid, src, true_label, rd_flag) in enumerate(zip(
                test_metrics["sample_ids"],
                test_metrics["source_names"],
                test_metrics["label_names"],
                test_metrics["has_rd"],
            ))
        ],
    }
    for source_name in evaluation_report["by_source"]:
        source_indices = [
            idx for idx, name in enumerate(test_metrics["source_names"])
            if name == source_name
        ]
        if source_indices:
            correct = 0
            for idx in source_indices:
                pred_idx = int(test_metrics["logits"][idx].argmax().item())
                if LOW_ALTITUDE_CLASSES[pred_idx] == test_metrics["label_names"][idx]:
                    correct += 1
            evaluation_report["by_source"][source_name]["accuracy"] = correct / len(source_indices)

    rd_mask = torch.tensor(test_metrics["has_rd"], dtype=torch.bool)
    evaluation_report["by_rd_presence"] = {
        "with_rd": {
            "count": int(rd_mask.sum().item()),
            "accuracy": _subset_accuracy(test_metrics["logits"], test_metrics["labels"], rd_mask),
        },
        "without_rd": {
            "count": int((~rd_mask).sum().item()),
            "accuracy": _subset_accuracy(test_metrics["logits"], test_metrics["labels"], ~rd_mask),
        },
    }
    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": _rel(best_path),
        "history_file": _rel(history_path),
        "manifest_file": _rel(manifest_path),
        "evaluation_report": _rel(evaluation_report_path),
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
            "has_valid_batches": test_metrics["has_valid_batches"],
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(LOW_ALTITUDE_CLASSES)},
        "trainable_parameters": _count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model),
        "class_weights": class_weights.detach().cpu().tolist(),
        "unknown_loss_weight": config.unknown_loss_weight,
        "unknown_sample_boost": config.unknown_sample_boost,
        "rd_activation_threshold": config.rd_activation_threshold,
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "sequence_encoder_lr_multiplier": config.sequence_encoder_lr_multiplier,
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "freeze_track_encoder_epochs": config.freeze_track_encoder_epochs,
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage(loaders.train_loader),
            "val": _rd_coverage(loaders.val_loader),
            "test": _rd_coverage(loaders.test_loader),
        },
        "best_epoch": best_epoch,
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
