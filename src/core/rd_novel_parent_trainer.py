from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .metrics import compute_classification_metrics
from .rd_novel_parent_config import RDNovelParentConfig
from .rd_novel_parent_data import NovelParentBatch, NovelParentSequenceDataset, collate_novel_parent_batch
from .rd_novel_parent_model import FrozenPhysicalV2NovelParentState
from .run_paths import prepare_run_output_dir
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: NovelParentBatch, device: torch.device) -> NovelParentBatch:
    return NovelParentBatch(
        features=batch.features.to(device, non_blocking=True),
        frame_padding_mask=batch.frame_padding_mask.to(device, non_blocking=True),
        chunk_padding_mask=batch.chunk_padding_mask.to(device, non_blocking=True),
        is_final_chunk=batch.is_final_chunk.to(device, non_blocking=True),
        is_zero_novel_update=batch.is_zero_novel_update.to(device, non_blocking=True),
        delta_time_seconds=batch.delta_time_seconds.to(device, non_blocking=True),
        frame_time_seconds=batch.frame_time_seconds.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        parent_sample_ids=batch.parent_sample_ids,
        anchor_logits=(batch.anchor_logits.to(device, non_blocking=True) if batch.anchor_logits is not None else None),
    )


def _parent_sequence_losses(
    logits: torch.Tensor,
    final_logits: torch.Tensor,
    batch: NovelParentBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    final_losses: list[torch.Tensor] = []
    prefix_losses: list[torch.Tensor] = []
    final_mask = batch.is_final_chunk & ~batch.chunk_padding_mask
    for batch_index in range(logits.shape[0]):
        update_logits = logits[batch_index][final_mask[batch_index]]
        if update_logits.shape[0] == 0:
            raise ValueError("parent sequence has no final update chunk")
        label = batch.labels[batch_index].expand(update_logits.shape[0])
        update_losses = functional.cross_entropy(update_logits, label, reduction="none")
        update_count = update_losses.shape[0]
        weights = torch.linspace(0.25, 1.0, steps=update_count, device=logits.device, dtype=logits.dtype)
        prefix_losses.append((update_losses * weights).sum() / weights.sum())
        final_losses.append(functional.cross_entropy(final_logits[batch_index].unsqueeze(0), batch.labels[batch_index].unsqueeze(0)))
    return torch.stack(final_losses).mean(), torch.stack(prefix_losses).mean()


def _metrics_payload(logits: list[torch.Tensor], labels: list[torch.Tensor]) -> dict[str, object]:
    logits_tensor = torch.cat(logits, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    classification = compute_classification_metrics(logits_tensor, labels_tensor, num_classes=2)
    per_class = [asdict(item) for item in compute_per_class_metrics(logits_tensor, labels_tensor, RD_ONLY_LABELS)]
    return {
        "parent": asdict(classification),
        "per_class": per_class,
        "parent_count": int(logits_tensor.shape[0]),
    }


def _make_loader(dataset: NovelParentSequenceDataset, config: RDNovelParentConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_novel_parent_batch,
        generator=torch.Generator().manual_seed(config.seed + (1 if shuffle else 2)),
    )


def _run_epoch(
    model: FrozenPhysicalV2NovelParentState,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: RDNovelParentConfig,
    *,
    epoch: int,
    phase: str,
    max_batches: int,
) -> tuple[dict[str, object], dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    final_losses: list[float] = []
    prefix_losses: list[float] = []
    residual_l2_losses: list[float] = []
    anchor_harm_losses: list[float] = []
    total_losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    processed_batches = 0
    started = time.perf_counter()
    for batch_index, source_batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        batch = _move_batch(source_batch, device)
        with torch.set_grad_enabled(training):
            output = model(batch)
            final_loss, prefix_loss = _parent_sequence_losses(output.logits, output.final_logits, batch)
            residual_l2_loss = output.logits.new_zeros(())
            anchor_harm_loss = output.logits.new_zeros(())
            if batch.anchor_logits is not None:
                valid_chunks = ~batch.chunk_padding_mask
                residual = output.logits[valid_chunks] - batch.anchor_logits[valid_chunks]
                residual_l2_loss = residual.square().mean()
                expanded_labels = batch.labels.unsqueeze(1).expand_as(batch.chunk_padding_mask)[valid_chunks]
                combined_ce = functional.cross_entropy(output.logits[valid_chunks], expanded_labels, reduction="none")
                anchor_ce = functional.cross_entropy(batch.anchor_logits[valid_chunks], expanded_labels, reduction="none")
                margin = float(getattr(config, "anchor_harm_margin", 0.0))
                anchor_harm_loss = torch.relu(combined_ce - anchor_ce - margin).mean()
            total_loss = (
                final_loss
                + float(config.prefix_loss_weight) * prefix_loss
                + float(getattr(config, "residual_l2_weight", 0.0)) * residual_l2_loss
                + float(getattr(config, "anchor_non_degradation_weight", 0.0)) * anchor_harm_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite {phase} loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
        final_losses.append(float(final_loss.detach().cpu()))
        prefix_losses.append(float(prefix_loss.detach().cpu()))
        residual_l2_losses.append(float(residual_l2_loss.detach().cpu()))
        anchor_harm_losses.append(float(anchor_harm_loss.detach().cpu()))
        total_losses.append(float(total_loss.detach().cpu()))
        all_logits.append(output.final_logits.detach().cpu())
        all_labels.append(batch.labels.detach().cpu())
        processed_batches += 1
        if batch_index == 1 or batch_index % max(1, config.log_interval_batches) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[rd-novel-parent][epoch={epoch}][{phase}] batch={batch_index} "
                f"loss={total_losses[-1]:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if not all_logits:
        raise RuntimeError(f"{phase} epoch processed zero batches")
    return _metrics_payload(all_logits, all_labels), {
        "total": float(np.mean(total_losses)),
        "final_parent_ce": float(np.mean(final_losses)),
        "weighted_prefix_ce": float(np.mean(prefix_losses)),
        "residual_l2": float(np.mean(residual_l2_losses)),
        "anchor_non_degradation": float(np.mean(anchor_harm_losses)),
        "batches": float(processed_batches),
    }


def train_rd_novel_parent_state(config: RDNovelParentConfig) -> dict[str, object]:
    if config.run_test_evaluation:
        raise ValueError("independent test evaluation is forbidden for novel parent-state training")
    if config.max_frames_per_chunk != 32:
        raise ValueError("max_frames_per_chunk must remain frozen at the Stage 009Q train-only value of 32")
    _seed_everything(config.seed)
    data_root = Path.cwd()
    base_checkpoint = Path(config.base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(base_checkpoint)
    train_dataset = NovelParentSequenceDataset(data_root, index_name=config.novel_update_index_name, split_name="train")
    validation_dataset = NovelParentSequenceDataset(data_root, index_name=config.novel_update_index_name, split_name="val")
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    validation_loader = _make_loader(validation_dataset, config, shuffle=False)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = FrozenPhysicalV2NovelParentState(
        base_checkpoint,
        adapter_dim=config.adapter_dim,
        state_dim=config.state_dim,
        dropout=config.dropout,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    paths = prepare_run_output_dir(config.output_dir)
    base_checkpoint_sha256 = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    history_path = paths.output_dir / "training_history.json"
    progress_path = paths.output_dir / "training_progress.json"
    _write_json(
        progress_path,
        {
            "status": "started",
            "config": asdict(config),
            "data": {
                "train_parents": len(train_dataset),
                "validation_parents": len(validation_dataset),
                "train_batches": len(train_loader),
                "validation_batches": len(validation_loader),
            },
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "test_evaluation_executed": False,
            "fusion_enabled": False,
        },
    )
    print(
        f"[rd-novel-parent] READY train_parents={len(train_dataset)} val_parents={len(validation_dataset)} "
        f"train_batches={len(train_loader)} val_batches={len(validation_loader)}",
        flush=True,
    )
    history: list[dict[str, object]] = []
    best_macro_f1 = float("-inf")
    best_checkpoint: Path | None = None
    for epoch in range(1, config.epochs + 1):
        print(f"[rd-novel-parent][epoch={epoch}] START", flush=True)
        train_metrics, train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            epoch=epoch,
            phase="train",
            max_batches=config.max_train_batches_per_epoch,
        )
        with torch.no_grad():
            validation_metrics, validation_loss = _run_epoch(
                model,
                validation_loader,
                None,
                device,
                config,
                epoch=epoch,
                phase="val",
                max_batches=config.max_val_batches,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        _write_json(history_path, history)
        validation_macro_f1 = float(validation_metrics["parent"]["macro_f1"])
        checkpoint_updated = False
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_checkpoint = paths.output_dir / "best_rd_novel_parent_state.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(config),
                    "base_checkpoint": str(base_checkpoint),
                    "base_checkpoint_sha256": base_checkpoint_sha256,
                    "trainable_state_dict": model.trainable_state_dict(),
                    "validation": validation_metrics,
                },
                best_checkpoint,
            )
            checkpoint_updated = True
        _write_json(
            progress_path,
            {
                "status": "running",
                "epoch": epoch,
                "latest": record,
                "best_validation_parent_macro_f1": best_macro_f1,
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
                "checkpoint_updated": checkpoint_updated,
                "test_evaluation_executed": False,
                "fusion_enabled": False,
            },
        )
        print(
            f"[rd-novel-parent][epoch={epoch}] END train_loss={train_loss['total']:.4f} "
            f"val_loss={validation_loss['total']:.4f} val_parent_macro_f1={validation_macro_f1:.4f} "
            f"best={best_macro_f1:.4f} checkpoint_updated={checkpoint_updated}",
            flush=True,
        )
    summary = {
        "experiment_scope": "RD-only physical-v2 novel-evidence causal parent-state; no fusion or test evaluation",
        "config": asdict(config),
        "artifact_dir": str(paths.output_dir),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "best_validation_parent_macro_f1": best_macro_f1,
        "history": history,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    _write_json(paths.output_dir / "training_summary.json", summary)
    return summary


__all__ = ["train_rd_novel_parent_state"]
