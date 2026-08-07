from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from .data.rd_physical_five_dataset import RD_FIVE_LABELS
from .data.rd_physical_specialist_dataset import specialist_global_indices, specialist_label_names
from .rd_physical_five_artifacts import build_physical_rd_five_model, load_physical_rd_five_artifacts, sha256_file
from .rd_physical_five_config import PhysicalRDFiveConfig
from .rd_physical_specialist_config import PhysicalRDSpecialistConfig
from .rd_physical_specialist_model import SPECIALIST_MODEL_SCHEMA_VERSION, PhysicalRDSpecialistClassifier


SPECIALIST_CHECKPOINT_SCHEMA_VERSION = "physical_rd_specialist_checkpoint_v2"


def _base_config(config: PhysicalRDSpecialistConfig) -> PhysicalRDFiveConfig:
    return PhysicalRDFiveConfig(
        index_name=config.index_name,
        normalization_stats_name=config.normalization_stats_name,
        feature_stats_name=config.feature_stats_name,
        output_dir=config.output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        input_channels=config.input_channels,
        token_dim=config.token_dim,
        quality_dim=config.quality_dim,
        prefix_aux_weight=config.prefix_aux_weight,
        micro_aux_weight=config.micro_aux_weight,
        prefix_max_loss_weight=config.prefix_max_loss_weight,
        evidence_target_frames=config.evidence_target_frames,
        class_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        tr_entry_preserved=config.tr_entry_preserved,
        allow_tr_inputs=config.allow_tr_inputs,
        max_train_batches_per_epoch=config.max_train_batches_per_epoch,
        max_val_batches=config.max_val_batches,
        log_interval_batches=config.log_interval_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        device=config.device,
        run_test_evaluation=config.run_test_evaluation,
    )


def load_physical_rd_specialist_artifacts(project_root: Path, config: PhysicalRDSpecialistConfig) -> dict[str, Any]:
    base = load_physical_rd_five_artifacts(project_root, _base_config(config))
    manifest = {
        "checkpoint_schema_version": SPECIALIST_CHECKPOINT_SCHEMA_VERSION,
        "task_contract": {
            "kind": config.kind,
            "labels": list(specialist_label_names(config.kind)),
            "global_labels": list(RD_FIVE_LABELS),
            "local_indices": list(specialist_global_indices(config.kind)),
            "global_label_groups": (
                [[0, 1, 2], [3, 4]] if config.kind == "parent_gate" else [list(specialist_global_indices(config.kind))]
            ),
            "rd_only_training": True,
            "tr_entry_preserved": True,
            "allow_tr_inputs": False,
            "supports_rejection": False,
            "test_evaluation_executed": False,
        },
        "base_five_artifact_manifest_hash": hashlib.sha256(json.dumps(base.manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_spec": {
            "version": SPECIALIST_MODEL_SCHEMA_VERSION,
            "kind": config.kind,
            "head_mode": config.kind,
            "token_dim": config.token_dim,
            "quality_dim": config.quality_dim,
            "local_class_count": len(specialist_label_names(config.kind)),
        },
        "source": "LOW_ALTITUDE_CLASSES label remap only; no TR/pairing/path/ID forward fields",
    }
    manifest["inference_contract_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"base": base, "manifest": manifest}


def build_physical_rd_specialist_model(artifacts: dict[str, Any], config: PhysicalRDSpecialistConfig, device: torch.device) -> PhysicalRDSpecialistClassifier:
    base = build_physical_rd_five_model(artifacts["base"], head_mode=config.kind)
    return PhysicalRDSpecialistClassifier(kind=config.kind, base=base).to(device)


def specialist_checkpoint_payload(*, epoch: int, config: PhysicalRDSpecialistConfig, artifacts: dict[str, Any], model: PhysicalRDSpecialistClassifier, validation: dict[str, Any], selection: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "checkpoint_schema_version": SPECIALIST_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "config": asdict(config),
        "artifact_manifest": artifacts["manifest"],
        "model_state_dict": model.state_dict(),
        "val": validation,
    }
    if selection is not None:
        payload["selection"] = selection
    return payload


def load_physical_rd_specialist_checkpoint(checkpoint_path: Path, project_root: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != SPECIALIST_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported RD specialist checkpoint schema")
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("RD specialist checkpoint is missing config")
    allowed = {field.name for field in fields(PhysicalRDSpecialistConfig)}
    if set(raw) - allowed:
        raise ValueError(f"unknown RD specialist config fields: {sorted(set(raw) - allowed)}")
    config = PhysicalRDSpecialistConfig(**raw)
    artifacts = load_physical_rd_specialist_artifacts(project_root, config)
    if checkpoint.get("artifact_manifest", {}).get("inference_contract_hash") != artifacts["manifest"]["inference_contract_hash"]:
        raise ValueError("RD specialist inference contract mismatch")
    model = build_physical_rd_specialist_model(artifacts, config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, artifacts, checkpoint


__all__ = [
    "SPECIALIST_CHECKPOINT_SCHEMA_VERSION",
    "build_physical_rd_specialist_model",
    "load_physical_rd_specialist_artifacts",
    "load_physical_rd_specialist_checkpoint",
    "specialist_checkpoint_payload",
]
