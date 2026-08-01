from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.collate import TrackBatch, collate_track_batch
from .data.torch_dataset import TorchTrackDataset
from .device import resolve_cuda_device
from .experts import LowAltitudeSuspectExpert
from .metrics import compute_classification_metrics
from .rd_drone_bird_config import RDDroneBirdDiagnosticConfig
from .rd_parent_stability import (
    ParentStabilityBatch,
    ParentStabilityDataset,
    build_parent_predecessor_lookup,
    collate_parent_stability_batch,
    parent_stability_aggregate_loss,
)
from .run_paths import prepare_run_output_dir
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(
    data_root: str,
    sample_ids: list[str],
    config: RDDroneBirdDiagnosticConfig,
    *,
    shuffle: bool,
    predecessor_by_sample_id: dict[str, str] | None = None,
) -> DataLoader:
    predecessor_by_sample_id = predecessor_by_sample_id or {}
    dataset_sample_ids = list(sample_ids)
    if predecessor_by_sample_id:
        known_ids = set(dataset_sample_ids)
        dataset_sample_ids.extend(
            predecessor_id
            for predecessor_id in predecessor_by_sample_id.values()
            if predecessor_id not in known_ids
        )
    dataset = TorchTrackDataset(
        data_root,
        sample_ids=dataset_sample_ids,
        trainable_only=False,
        index_name=config.index_name,
        use_rd_segment_store=True,
        rd_segment_manifest_name=config.rd_segment_manifest_name,
        window_segment_map_name=config.window_segment_map_name,
        load_only_fusion_eligible_rd=False,
        allow_rd_segment_store_fallback=False,
        rd_cache_size=8,
        rd_segment_store_cache_size=8,
        max_patch_frames=0,
        skip_deep_patch=True,
    )
    paired = bool(predecessor_by_sample_id)
    loader_dataset = (
        ParentStabilityDataset(dataset, predecessor_by_sample_id, sample_ids)
        if paired
        else dataset
    )
    return DataLoader(
        loader_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_parent_stability_batch if paired else collate_track_batch,
        generator=torch.Generator().manual_seed(config.seed + (1 if shuffle else 2)),
    )


def _select_rd_only_ids(index: pd.DataFrame, anchor_map: pd.DataFrame, split_name: str) -> list[str]:
    anchor_ids = set(anchor_map["sample_id"].astype(str))
    subset = index[
        index["split_name"].astype(str).eq(split_name)
        & index["label_name"].astype(str).isin(RD_ONLY_LABELS)
        & index["sample_id"].astype(str).isin(anchor_ids)
    ]
    return subset["sample_id"].astype(str).tolist()


def _balanced_smoke_ids(index: pd.DataFrame, sample_ids: list[str], max_batches: int, batch_size: int) -> list[str]:
    """Return deterministic, label-balanced IDs only for bounded smoke validation."""
    if max_batches <= 0:
        return sample_ids
    target_count = max(1, int(max_batches) * int(batch_size))
    frame = index[index["sample_id"].astype(str).isin(set(sample_ids))].copy()
    by_label = {
        label: frame[frame["label_name"].astype(str).eq(label)]["sample_id"].astype(str).tolist()
        for label in RD_ONLY_LABELS
    }
    selected: list[str] = []
    cursor = {label: 0 for label in RD_ONLY_LABELS}
    while len(selected) < target_count:
        progressed = False
        for label in RD_ONLY_LABELS:
            values = by_label[label]
            position = cursor[label]
            if position < len(values) and len(selected) < target_count:
                selected.append(values[position])
                cursor[label] = position + 1
                progressed = True
        if not progressed:
            break
    return selected


def _restrict_predecessors(
    predecessor_by_sample_id: dict[str, str],
    current_sample_ids: list[str],
) -> dict[str, str]:
    current_ids = {str(sample_id) for sample_id in current_sample_ids}
    return {
        current_id: predecessor_id
        for current_id, predecessor_id in predecessor_by_sample_id.items()
        if current_id in current_ids
    }


def _metric_payload(logits: torch.Tensor, labels: torch.Tensor, parents: list[str]) -> dict[str, object]:
    metrics = compute_classification_metrics(logits, labels, num_classes=2)
    per_class = [asdict(item) for item in compute_per_class_metrics(logits, labels, RD_ONLY_LABELS)]
    parent_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    parent_labels: dict[str, list[int]] = defaultdict(list)
    for parent, logit, label in zip(parents, logits, labels.tolist()):
        parent_logits[str(parent)].append(logit)
        parent_labels[str(parent)].append(int(label))
    parent_mean_logits = torch.stack([torch.stack(values).mean(dim=0) for values in parent_logits.values()])
    parent_targets = torch.tensor([values[0] for values in parent_labels.values()], dtype=torch.long)
    parent_metrics = compute_classification_metrics(parent_mean_logits, parent_targets, num_classes=2)
    return {
        "window": asdict(metrics),
        "per_class": per_class,
        "parent_mean_logit": asdict(parent_metrics),
        "parent_count": int(len(parent_logits)),
        "window_count": int(logits.shape[0]),
    }


def _forward_rd_diagnostic(
    model: LowAltitudeSuspectExpert,
    batch: TrackBatch,
    device: torch.device,
    config: RDDroneBirdDiagnosticConfig,
):
    rd_features = batch.rd_features.to(device, non_blocking=True)
    rd_padding = batch.rd_padding_mask.to(device, non_blocking=True)
    rd_patch_features = batch.rd_patch_features.to(device, non_blocking=True)
    rd_patch_padding = batch.rd_patch_padding_mask.to(device, non_blocking=True)
    rd_time_seconds = batch.rd_time_seconds.to(device, non_blocking=True)
    output = model.rd_drone_bird_diagnostic(
        rd_features,
        rd_patch_features=rd_patch_features if config.use_rd_patch_cnn else None,
        rd_padding_mask=rd_padding,
        rd_patch_padding_mask=rd_patch_padding if config.use_rd_patch_cnn else None,
        rd_time_seconds=rd_time_seconds,
    )
    labels = torch.tensor(
        [1 if name == "BirdTarget" else 0 for name in batch.label_names],
        dtype=torch.long,
        device=device,
    )
    return output, labels


def _run_epoch(
    model: LowAltitudeSuspectExpert,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: RDDroneBirdDiagnosticConfig,
    *,
    epoch: int,
    phase: str,
) -> tuple[dict[str, object], float]:
    train = optimizer is not None
    if (phase == "train") != train:
        raise ValueError(f"phase/optimizer mismatch: phase={phase}, train={train}")
    model.train(train)
    max_batches = (
        int(config.max_train_batches_per_epoch)
        if train and config.max_train_batches_per_epoch > 0
        else int(config.max_val_batches)
        if not train and config.max_val_batches > 0
        else len(loader)
    )
    print(
        f"[rd-only][epoch={epoch}] {phase} START batches={max_batches} "
        f"loader_len={len(loader)}",
        flush=True,
    )
    epoch_start = time.perf_counter()
    first_batch_start: float | None = None
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    parents: list[str] = []
    losses: list[float] = []
    main_losses: list[float] = []
    micro_losses: list[float] = []
    stability_losses: list[float] = []
    paired_window_count = 0
    completed_batches = 0
    interval = max(1, int(config.log_interval_batches))
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > max_batches:
            break
        if first_batch_start is None:
            first_batch_start = time.perf_counter()
            print(f"[rd-only][epoch={epoch}] {phase} first batch loaded", flush=True)
        batch_start = time.perf_counter()
        paired_batch = batch if isinstance(batch, ParentStabilityBatch) else None
        current_batch = paired_batch.current if paired_batch is not None else batch
        with torch.set_grad_enabled(train):
            output, labels = _forward_rd_diagnostic(model, current_batch, device, config)
            main_loss = nn.functional.cross_entropy(output.bird_logits, labels)
            micro_loss = nn.functional.cross_entropy(output.micro_doppler_logits, labels)
            stability_loss = output.bird_logits.new_zeros(())
            pair_count = 0
            if (
                paired_batch is not None
                and paired_batch.predecessor is not None
                and float(config.parent_stability_aux_weight) > 0.0
            ):
                with torch.no_grad():
                    predecessor_output, _ = _forward_rd_diagnostic(
                        model, paired_batch.predecessor, device, config
                    )
                stability_loss, pair_count = parent_stability_aggregate_loss(
                    output.bird_logits,
                    labels,
                    predecessor_output.bird_logits,
                    paired_batch.predecessor_positions,
                )
            loss = (
                main_loss
                + float(config.micro_doppler_aux_weight) * micro_loss
                + float(config.parent_stability_aux_weight) * stability_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite RD-only {phase} loss at epoch={epoch}, batch={batch_index}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        main_losses.append(float(main_loss.detach().cpu()))
        micro_losses.append(float(micro_loss.detach().cpu()))
        stability_losses.append(float(stability_loss.detach().cpu()))
        paired_window_count += pair_count
        logits_all.append(output.bird_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        parents.extend(current_batch.parent_sample_ids)
        completed_batches = batch_index
        if batch_index == 1:
            print(
                f"[rd-only][epoch={epoch}] {phase} first batch complete "
                f"load_to_complete={time.perf_counter() - first_batch_start:.2f}s "
                f"loss={loss_value:.4f} pairs={pair_count}/{len(current_batch.sample_ids)} "
                f"rd_lengths={current_batch.rd_lengths.tolist()}",
                flush=True,
            )
        if batch_index % interval == 0 or batch_index == max_batches:
            elapsed = time.perf_counter() - epoch_start
            average_seconds = elapsed / batch_index
            remaining = max(0, max_batches - batch_index) * average_seconds
            print(
                f"[rd-only][epoch={epoch}] {phase} batch {batch_index}/{max_batches} "
                f"loss={loss_value:.4f} main={main_losses[-1]:.4f} micro={micro_losses[-1]:.4f} "
                f"stability={stability_losses[-1]:.4f} pairs={pair_count} "
                f"batch_time={time.perf_counter() - batch_start:.2f}s "
                f"elapsed={elapsed:.0f}s ETA={remaining:.0f}s",
                flush=True,
            )
    if not logits_all:
        raise RuntimeError("RD-only loader produced no batches")
    metrics = _metric_payload(torch.cat(logits_all), torch.cat(labels_all), parents)
    mean_loss = float(np.mean(losses))
    metrics["loss_components"] = {
        "main_ce": float(np.mean(main_losses)),
        "micro_doppler_ce": float(np.mean(micro_losses)),
        "parent_stability_ce": float(np.mean(stability_losses)),
        "parent_stability_pair_count": int(paired_window_count),
        "parent_stability_pair_coverage": float(paired_window_count / len(parents)) if parents else 0.0,
    }
    print(
        f"[rd-only][epoch={epoch}] {phase} END batches={completed_batches} "
        f"loss={mean_loss:.4f} macro_f1={float(metrics['window']['macro_f1']):.4f}",
        flush=True,
    )
    return metrics, mean_loss


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_rd_drone_bird_diagnostic(config: RDDroneBirdDiagnosticConfig) -> dict[str, object]:
    if config.run_test_evaluation:
        raise ValueError("RD-only diagnostic forbids test evaluation")
    set_seed(config.seed)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    data_root = project_root / "data"
    index = pd.read_csv(data_root / "processed" / config.index_name, low_memory=False)
    anchor_map = pd.read_csv(data_root / "processed" / config.window_segment_map_name, low_memory=False)
    train_ids = _select_rd_only_ids(index, anchor_map, "train")
    val_ids = _select_rd_only_ids(index, anchor_map, "val")
    if not train_ids or not val_ids:
        raise RuntimeError("RD-only diagnostic requires non-empty train and validation anchor windows")
    val_loader_ids = _balanced_smoke_ids(index, val_ids, config.max_val_batches, config.batch_size)
    parent_stability_enabled = float(config.parent_stability_aux_weight) > 0.0
    train_lookup = build_parent_predecessor_lookup(index, train_ids)
    val_lookup = build_parent_predecessor_lookup(index, val_ids)
    train_predecessors = (
        train_lookup.predecessor_by_sample_id if parent_stability_enabled else None
    )
    val_predecessors = (
        _restrict_predecessors(val_lookup.predecessor_by_sample_id, val_loader_ids)
        if parent_stability_enabled
        else None
    )
    train_loader = _make_loader(
        str(data_root), train_ids, config, shuffle=True,
        predecessor_by_sample_id=train_predecessors,
    )
    val_loader = _make_loader(
        str(data_root), val_loader_ids, config, shuffle=False,
        predecessor_by_sample_id=val_predecessors,
    )
    model = LowAltitudeSuspectExpert(
        input_dim=160,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=0,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        use_rd_continuous_time_encoding=config.use_rd_continuous_time_encoding,
        rd_time_encoding_mode=config.rd_time_encoding_mode,
        rd_local_delta_time_clip_seconds=config.rd_local_delta_time_clip_seconds,
        rd_local_delta_time_scale=config.rd_local_delta_time_scale,
        dropout=config.dropout,
    ).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith((
            "range_", "doppler_", "rd_token_proj", "rd_encoder", "rd_norm", "stats_proj",
            "rd_patch", "rd_backbone", "rd_bird_head", "drone_bird_micro_doppler_head",
        ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    paths = prepare_run_output_dir(config.output_dir)
    history_path = paths.output_dir / "training_history.json"
    progress_path = paths.output_dir / "training_progress.json"
    _write_json(
        progress_path,
        {
            "status": "started",
            "epoch": 0,
            "config": asdict(config),
            "data": {
                "train_windows": len(train_ids),
                "val_windows": len(val_ids),
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
            },
            "parent_stability": {
                "enabled": parent_stability_enabled,
                "aux_weight": float(config.parent_stability_aux_weight),
                "train": train_lookup.report,
                "validation_full": val_lookup.report,
                "validation_loaded_pair_count": int(len(val_predecessors or {})),
            },
            "test_evaluation_executed": False,
        },
    )
    print(
        f"[rd-only] READY train_windows={len(train_ids)} val_windows={len(val_ids)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"parent_stability={'on' if parent_stability_enabled else 'off'} "
        f"log_interval={config.log_interval_batches}",
        flush=True,
    )
    history: list[dict[str, object]] = []
    best_macro_f1 = float("-inf")
    best_checkpoint: Path | None = None
    for epoch in range(1, config.epochs + 1):
        print(f"[rd-only][epoch={epoch}] START", flush=True)
        train_metrics, train_loss = _run_epoch(
            model, train_loader, optimizer, device, config, epoch=epoch, phase="train"
        )
        with torch.no_grad():
            val_metrics, val_loss = _run_epoch(
                model, val_loader, None, device, config, epoch=epoch, phase="val"
            )
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics}
        history.append(record)
        _write_json(history_path, history)
        val_f1 = float(val_metrics["window"]["macro_f1"])
        checkpoint_updated = False
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_checkpoint = paths.output_dir / "best_rd_only_drone_bird_model.pt"
            torch.save({"epoch": epoch, "config": asdict(config), "model_state_dict": model.state_dict(), "val": val_metrics}, best_checkpoint)
            checkpoint_updated = True
        _write_json(
            progress_path,
            {
                "status": "running",
                "epoch": epoch,
                "latest": record,
                "best_validation_window_macro_f1": best_macro_f1,
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
                "checkpoint_updated": checkpoint_updated,
                "test_evaluation_executed": False,
            },
        )
        print(
            f"[rd-only][epoch={epoch}] END train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_f1:.4f} best={best_macro_f1:.4f} checkpoint_updated={checkpoint_updated}",
            flush=True,
        )
    summary = {
        "experiment_scope": "RD-only Drone/Bird diagnostic; no Joint fusion, no model selection outside validation, no test evaluation",
        "device": device_info,
        "config": asdict(config),
        "data": {
            "train_windows": len(train_ids),
            "val_windows": len(val_ids),
            "val_windows_loaded": len(val_loader_ids),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
        },
        "parent_stability": {
            "enabled": parent_stability_enabled,
            "aux_weight": float(config.parent_stability_aux_weight),
            "train": train_lookup.report,
            "validation_full": val_lookup.report,
            "validation_loaded_pair_count": int(len(val_predecessors or {})),
        },
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "best_validation_window_macro_f1": best_macro_f1,
        "history_file": "training_history.json",
        "test_evaluation_executed": False,
    }
    _write_json(history_path, history)
    _write_json(paths.output_dir / "training_summary.json", summary)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "epoch": config.epochs,
            "best_validation_window_macro_f1": best_macro_f1,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "test_evaluation_executed": False,
        },
    )
    return summary
