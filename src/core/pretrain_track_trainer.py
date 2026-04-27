from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import resolve_cuda_device
from .features import TRACK_FEATURE_COLUMNS
from .pretrain_track_config import TrackPretrainConfig
from .pretrain_track_data import build_pretrain_track_loader
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .trajectory import TransformerTrackEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _mean_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (hidden * valid).sum(dim=1) / denom


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    targets = torch.arange(z1.shape[0], device=z1.device)
    loss_a = torch.nn.functional.cross_entropy(logits, targets)
    loss_b = torch.nn.functional.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_b)


class TrackPretrainModel(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TransformerTrackEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.reconstruct_head = nn.Linear(model_dim, input_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, masked: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor, view_a: torch.Tensor, view_a_padding_mask: torch.Tensor, view_b: torch.Tensor, view_b_padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_masked = self.encoder(masked, padding_mask=padding_mask)
        recon = self.reconstruct_head(hidden_masked)
        hidden_a = self.encoder(view_a, padding_mask=view_a_padding_mask)
        hidden_b = self.encoder(view_b, padding_mask=view_b_padding_mask)
        pooled_a = self.proj_head(_mean_pool(hidden_a, view_a_padding_mask))
        pooled_b = self.proj_head(_mean_pool(hidden_b, view_b_padding_mask))
        return {
            "reconstruction": recon,
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
        }


def _encoder_state_dict(model: TrackPretrainModel) -> dict[str, torch.Tensor]:
    return {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }


def pretrain_track_encoder(config: TrackPretrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loader = build_pretrain_track_loader(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mask_ratio=config.mask_ratio,
    )
    model = TrackPretrainModel(
        input_dim=len(TRACK_FEATURE_COLUMNS),
        model_dim=config.model_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(model):,}")

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = run_paths.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_values: list[float] = []
        mtm_values: list[float] = []
        nce_values: list[float] = []
        for batch in loader:
            masked = batch.masked.to(device)
            original = batch.original.to(device)
            mask = batch.mask.to(device)
            padding_mask = batch.padding_mask.to(device)
            view_a = batch.view_a.to(device)
            view_b = batch.view_b.to(device)
            view_a_padding_mask = batch.view_a_padding_mask.to(device)
            view_b_padding_mask = batch.view_b_padding_mask.to(device)

            output = model(masked, mask, padding_mask, view_a, view_a_padding_mask, view_b, view_b_padding_mask)
            recon = output["reconstruction"]
            if mask.any():
                mtm_loss = torch.nn.functional.mse_loss(recon[mask], original[mask])
            else:
                mtm_loss = torch.tensor(0.0, device=device)
            nce_loss = _info_nce(output["pooled_a"], output["pooled_b"], config.contrastive_temperature)
            loss = config.mtm_loss_weight * mtm_loss + config.contrastive_loss_weight * nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_values.append(float(loss.detach().cpu().item()))
            mtm_values.append(float(mtm_loss.detach().cpu().item()))
            nce_values.append(float(nce_loss.detach().cpu().item()))

        record = {
            "epoch": epoch,
            "loss": float(np.mean(loss_values)),
            "mtm_loss": float(np.mean(mtm_values)),
            "contrastive_loss": float(np.mean(nce_values)),
        }
        history.append(record)
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            torch.save(
                {
                    "model_state_dict": _encoder_state_dict(model),
                    "pretrain_model_state_dict": model.state_dict(),
                },
                best_path,
            )

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps({"mode": "pretrain_track", "num_epochs": config.epochs}, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation_report = {
        "summary": {
            "loss": float(history[-1]["loss"]),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "route_accuracy": 0.0,
        },
        "label_names": [],
        "classification_confusion_matrix": [],
        "per_class": [],
        "route": {"label_names": [], "confusion_matrix": [], "per_route": []},
        "by_source": {},
        "device": device_info,
        "predictions": [],
    }
    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path.as_posix()),
        "history_file": str(history_path.as_posix()),
        "manifest_file": str(manifest_path.as_posix()),
        "evaluation_report": str(evaluation_report_path.as_posix()),
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "trainable_parameters": _count_trainable_parameters(model),
        "final_loss": float(history[-1]["loss"]),
        "final_mtm_loss": float(history[-1]["mtm_loss"]),
        "final_contrastive_loss": float(history[-1]["contrastive_loss"]),
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
