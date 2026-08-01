from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import resolve_cuda_device
from .features import (
    track_sequence_binary_feature_indices,
    track_sequence_continuous_feature_indices,
    track_sequence_feature_columns,
    track_sequence_feature_dim,
)
from .pretrain_track_config import TrackPretrainConfig
from .pretrain_track_data import build_pretrain_track_loader
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .trajectory import TEMPORAL_POSITION_REFERENCE, TransformerTrackEncoder
from .trajectory import (
    TEMPORAL_POOLING_MODES,
    TrajectorySequencePool,
)


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


def _mtm_reconstruction_mask(
    mtm_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    feature_missing_mask: torch.Tensor,
) -> torch.Tensor:
    """Return element-level targets valid for MTM reconstruction."""
    if mtm_mask.ndim != 2 or padding_mask.shape != mtm_mask.shape:
        raise ValueError("mtm_mask and padding_mask must have shape [batch, time]")
    if feature_missing_mask.ndim != 3 or feature_missing_mask.shape[:2] != mtm_mask.shape:
        raise ValueError("feature_missing_mask must have shape [batch, time, feature]")
    return (
        mtm_mask.unsqueeze(-1)
        & ~padding_mask.unsqueeze(-1)
        & ~feature_missing_mask.bool()
    )


def _mtm_reconstruction_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mtm_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    feature_missing_mask: torch.Tensor,
) -> torch.Tensor:
    valid_mask = _mtm_reconstruction_mask(
        mtm_mask,
        padding_mask,
        feature_missing_mask,
    )
    if not bool(valid_mask.any()):
        return reconstruction.sum() * 0.0
    return torch.nn.functional.mse_loss(
        reconstruction[valid_mask],
        original[valid_mask],
    )


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    targets = torch.arange(z1.shape[0], device=z1.device)
    loss_a = torch.nn.functional.cross_entropy(logits, targets)
    loss_b = torch.nn.functional.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_b)


class TrackPretrainModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        dropout: float = 0.1,
        temporal_encoding_mode: str = "none",
        temporal_pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = TransformerTrackEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            dropout=dropout,
            temporal_encoding_mode=temporal_encoding_mode,
        )
        self.temporal_encoding_mode = self.encoder.temporal_encoding_mode
        self.temporal_pooling_mode = str(temporal_pooling_mode).strip().lower()
        if self.temporal_pooling_mode not in TEMPORAL_POOLING_MODES:
            raise ValueError(
                "unsupported temporal pooling mode: "
                f"{temporal_pooling_mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
            )
        self.sequence_pool = TrajectorySequencePool(
            model_dim,
            mode=self.temporal_pooling_mode,
            attention_hidden_dim=max(32, model_dim // 2),
        )
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.reconstruct_head = nn.Linear(model_dim, input_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, masked: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor, view_a: torch.Tensor, view_a_padding_mask: torch.Tensor, view_b: torch.Tensor, view_b_padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        effective_mask = mask.bool() & ~padding_mask.bool()
        mask_token = self.mask_token.to(device=masked.device, dtype=masked.dtype).view(1, 1, -1)
        masked_input = torch.where(effective_mask.unsqueeze(-1), mask_token, masked)
        hidden_masked = self.encoder(masked_input, padding_mask=padding_mask)
        recon = self.reconstruct_head(hidden_masked)
        hidden_a = self.encoder(view_a, padding_mask=view_a_padding_mask)
        hidden_b = self.encoder(view_b, padding_mask=view_b_padding_mask)
        pooled_a = self.proj_head(self.sequence_pool(hidden_a, view_a_padding_mask))
        pooled_b = self.proj_head(self.sequence_pool(hidden_b, view_b_padding_mask))
        return {
            "reconstruction": recon,
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
        }


def _parent_ids_sha256(parent_ids: set[str]) -> str:
    payload = "\n".join(sorted(parent_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_allowed_parent_manifest(
    path_value: str,
    *,
    required: bool,
) -> tuple[set[str] | None, dict[str, object]]:
    normalized = str(path_value or "").strip()
    if not normalized:
        if required:
            raise ValueError("require_train_parent_manifest=true requires allowed_parent_manifest_path")
        return None, {"enabled": False, "parent_count": 0, "parent_ids_sha256": ""}
    path = Path(normalized)
    if not path.exists():
        raise FileNotFoundError(f"pretraining parent manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_ids = payload.get("parent_sample_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("pretraining parent manifest parent_sample_ids must be a list")
    parent_ids = {str(item) for item in raw_ids if str(item).strip()}
    if not parent_ids:
        raise ValueError("pretraining parent manifest contains no parent_sample_ids")
    computed_hash = _parent_ids_sha256(parent_ids)
    declared_hash = str(payload.get("parent_ids_sha256", ""))
    if declared_hash and declared_hash != computed_hash:
        raise ValueError("pretraining parent manifest hash does not match parent_sample_ids")
    return parent_ids, {
        "enabled": True,
        "path": path.as_posix(),
        "parent_count": len(parent_ids),
        "parent_ids_sha256": computed_hash,
    }


def _serialized_source_stats(dataset) -> dict[str, dict[str, list[float]]]:
    return {
        str(source_name): {
            "center": center.astype(np.float32, copy=False).tolist(),
            "scale": scale.astype(np.float32, copy=False).tolist(),
        }
        for source_name, (center, scale) in dataset.source_stats.items()
    }


def _encoder_state_dict(model: TrackPretrainModel) -> dict[str, torch.Tensor]:
    return {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }


def pretrain_track_encoder(config: TrackPretrainConfig) -> dict[str, object]:
    config.validate_active_training_contract()
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    effective_config_path = run_paths.output_dir / "effective_config.json"
    effective_config_path.write_text(
        json.dumps(
            {
                "schema_version": "pretrain-track-effective-config-v1",
                "pretrain_track": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    allowed_parent_ids, parent_manifest = _load_allowed_parent_manifest(
        config.allowed_parent_manifest_path,
        required=bool(config.require_train_parent_manifest),
    )
    loader = build_pretrain_track_loader(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mask_ratio=config.mask_ratio,
        trajectory_feature_schema=config.trajectory_feature_schema,
        index_name=config.index_name,
        allowed_parent_ids=allowed_parent_ids,
        sampling_mode=config.sampling_mode,
    )
    actual_parent_ids = set(getattr(loader.dataset, "actual_parent_ids", set()))
    if allowed_parent_ids is not None and actual_parent_ids != allowed_parent_ids:
        raise ValueError("pretraining dataset parent IDs do not exactly match the requested manifest")
    parent_manifest = {
        **parent_manifest,
        "actual_parent_count": len(actual_parent_ids),
        "actual_parent_ids_sha256": _parent_ids_sha256(actual_parent_ids),
    }
    model = TrackPretrainModel(
        input_dim=track_sequence_feature_dim(config.trajectory_feature_schema),
        model_dim=config.model_dim,
        dropout=config.dropout,
        temporal_encoding_mode=config.temporal_encoding_mode,
        temporal_pooling_mode=config.temporal_pooling_mode,
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
        masked_token_counts: list[float] = []
        reconstructable_feature_counts: list[float] = []
        for batch in loader:
            masked = batch.masked.to(device)
            original = batch.original.to(device)
            mask = batch.mask.to(device)
            padding_mask = batch.padding_mask.to(device)
            view_a = batch.view_a.to(device)
            view_b = batch.view_b.to(device)
            view_a_padding_mask = batch.view_a_padding_mask.to(device)
            view_b_padding_mask = batch.view_b_padding_mask.to(device)
            feature_missing_mask = batch.feature_missing_mask.to(device)

            output = model(masked, mask, padding_mask, view_a, view_a_padding_mask, view_b, view_b_padding_mask)
            recon = output["reconstruction"]
            mtm_loss = _mtm_reconstruction_loss(
                recon,
                original,
                mask,
                padding_mask,
                feature_missing_mask,
            )
            nce_loss = _info_nce(output["pooled_a"], output["pooled_b"], config.contrastive_temperature)
            loss = config.mtm_loss_weight * mtm_loss + config.contrastive_loss_weight * nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_values.append(float(loss.detach().cpu().item()))
            mtm_values.append(float(mtm_loss.detach().cpu().item()))
            nce_values.append(float(nce_loss.detach().cpu().item()))

            valid_mtm_mask = _mtm_reconstruction_mask(
                mask,
                padding_mask,
                feature_missing_mask,
            )
            masked_token_count = float((mask & ~padding_mask).sum().detach().cpu().item())
            reconstructable_feature_count = float(valid_mtm_mask.sum().detach().cpu().item())
            masked_token_counts.append(masked_token_count)
            reconstructable_feature_counts.append(reconstructable_feature_count)

        record = {
            "epoch": epoch,
            "loss": float(np.mean(loss_values)),
            "mtm_loss": float(np.mean(mtm_values)),
            "contrastive_loss": float(np.mean(nce_values)),
            "masked_token_count": float(np.mean(masked_token_counts)) if masked_token_counts else 0.0,
            "reconstructable_feature_count": float(np.mean(reconstructable_feature_counts)) if reconstructable_feature_counts else 0.0,
        }
        history.append(record)
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            torch.save(
                {
                    "model_state_dict": _encoder_state_dict(model),
                    "pretrain_model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "trajectory_feature_schema": config.trajectory_feature_schema,
                    "feature_contract": {
                        "feature_columns": track_sequence_feature_columns(config.trajectory_feature_schema),
                        "continuous_indices": track_sequence_continuous_feature_indices(config.trajectory_feature_schema),
                        "binary_missing_indices": track_sequence_binary_feature_indices(config.trajectory_feature_schema),
                        "source_amplitude_transform": "cq08_signed_log1p_v1"
                        if str(config.trajectory_feature_schema).strip().lower() == "v3"
                        else "legacy_none",
                    },
                    "sampling_contract": dict(loader.dataset.sampling_metadata),
                    "temporal_contract": {
                        "encoding_mode": config.temporal_encoding_mode,
                        "pooling_mode": config.temporal_pooling_mode,
                        "position_reference": TEMPORAL_POSITION_REFERENCE,
                    },
                    "parent_manifest": parent_manifest,
                    "source_feature_stats": _serialized_source_stats(loader.dataset),
                    "selection_metric": "training_loss",
                    "epoch": int(epoch),
                    "best_training_loss": float(best_loss),
                },
                best_path,
            )

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "pretrain_track",
                "num_epochs": config.epochs,
                "trajectory_feature_schema": config.trajectory_feature_schema,
                "feature_columns": track_sequence_feature_columns(config.trajectory_feature_schema),
                "index_name": config.index_name,
                "sampling_contract": dict(loader.dataset.sampling_metadata),
                "temporal_contract": {
                    "encoding_mode": config.temporal_encoding_mode,
                    "pooling_mode": config.temporal_pooling_mode,
                    "position_reference": TEMPORAL_POSITION_REFERENCE,
                },
                "parent_manifest": parent_manifest,
                "sample_count": len(loader.dataset),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

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
        "effective_config": str(effective_config_path.as_posix()),
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
