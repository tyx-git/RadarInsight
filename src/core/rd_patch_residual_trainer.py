from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .device import resolve_cuda_device
from .rd_drone_bird_trainer import RD_ONLY_LABELS, _balanced_smoke_ids, _make_loader, _metric_payload, _select_rd_only_ids, set_seed
from .rd_patch_residual_config import RDPatchResidualConfig
from .rd_patch_residual_model import FrozenSequencePatchResidual
from .run_paths import prepare_run_output_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_epoch(model: FrozenSequencePatchResidual, loader, optimizer, device: torch.device, config: RDPatchResidualConfig, *, train: bool) -> tuple[dict[str, object], float]:
    model.train(train)
    logits_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    losses: list[float] = []
    deltas: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    phase = "train" if train else "val"
    batch_limit = config.max_train_batches_per_epoch if train else config.max_val_batches
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        if batch_limit > 0 and batch_index > batch_limit:
            break
        labels = torch.tensor([1 if name == "BirdTarget" else 0 for name in batch.label_names], dtype=torch.long, device=device)
        with torch.set_grad_enabled(train):
            output = model(
                batch.rd_features.to(device, non_blocking=True),
                batch.rd_patch_features.to(device, non_blocking=True),
                rd_padding_mask=batch.rd_padding_mask.to(device, non_blocking=True),
                rd_patch_padding_mask=batch.rd_patch_padding_mask.to(device, non_blocking=True),
            )
            main_loss = nn.functional.cross_entropy(output.logits, labels)
            residual_loss = output.residual_delta.square().mean()
            loss = main_loss + float(config.residual_l2_weight) * residual_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        logits_rows.append(output.logits.detach().cpu())
        labels_rows.append(labels.detach().cpu())
        parent_ids.extend(batch.parent_sample_ids)
        deltas.append(output.residual_delta.detach().cpu())
        gates.append(output.residual_gate.detach().cpu())
        losses.append(float(loss.detach().cpu()))
        if batch_index == 1 or (config.log_interval_batches > 0 and batch_index % config.log_interval_batches == 0):
            print(f"[rd-residual] {phase} batch={batch_index} loss={losses[-1]:.4f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    logits = torch.cat(logits_rows)
    labels = torch.cat(labels_rows)
    metrics = _metric_payload(logits, labels, parent_ids)
    delta_tensor = torch.cat(deltas)
    gate_tensor = torch.cat(gates)
    metrics["residual"] = {
        "mean_abs_delta": float(delta_tensor.abs().mean()),
        "p95_abs_delta": float(torch.quantile(delta_tensor.abs().reshape(-1), 0.95)),
        "mean_gate": float(gate_tensor.mean()),
    }
    return metrics, sum(losses) / len(losses)


def train_rd_patch_residual(config: RDPatchResidualConfig) -> dict[str, object]:
    if config.run_test_evaluation or not config.base_checkpoint:
        raise ValueError("residual diagnostic requires base_checkpoint and forbids test evaluation")
    set_seed(config.seed)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    data_root = project_root / "data"
    import pandas as pd
    index = pd.read_csv(data_root / "processed" / config.index_name, low_memory=False)
    anchor_map = pd.read_csv(data_root / "processed" / config.window_segment_map_name, low_memory=False)
    train_ids = _select_rd_only_ids(index, anchor_map, "train")
    val_ids = _select_rd_only_ids(index, anchor_map, "val")
    val_loader_ids = _balanced_smoke_ids(index, val_ids, config.max_val_batches, config.batch_size)
    train_loader = _make_loader(str(data_root), train_ids, config, shuffle=True)
    val_loader = _make_loader(str(data_root), val_loader_ids, config, shuffle=False)
    model = FrozenSequencePatchResidual(config.base_checkpoint, dropout=config.dropout, residual_scale=config.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.patch_branch.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    paths = prepare_run_output_dir(config.output_dir)
    history: list[dict[str, object]] = []
    best_f1 = float("-inf")
    best_checkpoint = paths.output_dir / "best_rd_patch_residual_model.pt"
    for epoch in range(1, config.epochs + 1):
        print(f"[rd-residual] epoch={epoch} START", flush=True)
        train_metrics, train_loss = _run_epoch(model, train_loader, optimizer, device, config, train=True)
        with torch.no_grad():
            val_metrics, val_loss = _run_epoch(model, val_loader, None, device, config, train=False)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics}
        history.append(record)
        _write_json(paths.output_dir / "training_history.json", history)
        if float(val_metrics["window"]["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["window"]["macro_f1"])
            torch.save({"epoch": epoch, "config": asdict(config), "model_state_dict": model.state_dict(), "val": val_metrics, "base_checkpoint": config.base_checkpoint}, best_checkpoint)
        _write_json(paths.output_dir / "training_progress.json", {"status": "running", "epoch": epoch, "best_validation_window_macro_f1": best_f1, "best_checkpoint": str(best_checkpoint), "test_evaluation_executed": False})
    summary = {"experiment_scope": "frozen-sequence patch-residual RD-only diagnostic; no fusion or test", "device": device_info, "config": asdict(config), "data": {"train_windows": len(train_ids), "val_windows": len(val_ids), "val_windows_loaded": len(val_loader_ids), "train_batches": len(train_loader), "val_batches": len(val_loader)}, "best_checkpoint": str(best_checkpoint), "best_validation_window_macro_f1": best_f1, "test_evaluation_executed": False}
    _write_json(paths.output_dir / "training_summary.json", summary)
    _write_json(paths.output_dir / "training_progress.json", {"status": "completed", "epoch": config.epochs, "best_validation_window_macro_f1": best_f1, "best_checkpoint": str(best_checkpoint), "test_evaluation_executed": False})
    return summary
