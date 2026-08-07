from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .data.rd_physical_window_dataset import PHYSICAL_RD_FEATURE_NAMES
from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES, RD_FIVE_NORMALIZATION_SCHEMA_VERSION
from .rd_physical_drone_bird_model import SPATIAL_POOL_DOPPLER_BINS, SPATIAL_POOL_RANGE_BINS, TEMPORAL_POOLING_VERSION
from .rd_physical_five_config import PhysicalRDFiveConfig
from .rd_physical_five_model import FIVE_RD_MODEL_SCHEMA_VERSION, PhysicalRDFiveClassifier
from .rd_physical_statistics import PHYSICAL_RD_STATISTIC_NAMES


CHECKPOINT_SCHEMA_VERSION = "physical_rd_five_checkpoint_v1"

INFERENCE_CODE_FILES = (
    "src/core/data/rd_physical_window_dataset.py",
    "src/core/data/rd_physical_five_dataset.py",
    "src/core/rd_physical_statistics.py",
    "src/core/rd_physical_drone_bird_model.py",
    "src/core/rd_physical_five_model.py",
    "src/core/rd_physical_five_config.py",
    "src/core/rd_physical_five_artifacts.py",
    "src/core/rd_physical_schema.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_config(values: dict[str, Any]) -> PhysicalRDFiveConfig:
    allowed = {field.name for field in fields(PhysicalRDFiveConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown PhysicalRDFiveConfig fields: {', '.join(unknown)}")
    return PhysicalRDFiveConfig(**values)


def _canonical_tensor_inventory(project_root: Path, index: pd.DataFrame) -> dict[str, Any]:
    paths = sorted({str(value).strip() for value in index["rd_tensor_path"].astype(str) if str(value).strip()})
    digest = hashlib.sha256()
    total_bytes = 0
    for raw_path in paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.exists():
            raise FileNotFoundError(f"physical RD tensor is missing: {resolved}")
        digest.update(resolved.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(resolved.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with resolved.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        total_bytes += resolved.stat().st_size
    return {
        "algorithm": "sha256-canonical-tensor-file-set-v1",
        "file_count": len(paths),
        "total_bytes": int(total_bytes),
        "content_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class PhysicalRDFiveArtifacts:
    config: PhysicalRDFiveConfig
    normalization: dict[str, Any]
    feature_stats: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedPhysicalRDFiveCheckpoint:
    model: PhysicalRDFiveClassifier
    config: PhysicalRDFiveConfig
    artifacts: PhysicalRDFiveArtifacts
    checkpoint: dict[str, Any]


def _inference_contract_hash(manifest: dict[str, Any]) -> str:
    contract = {
        "task_contract": manifest["task_contract"],
        "model_spec": manifest["model_spec"],
        "selection_policy": manifest.get("selection_policy"),
        "artifacts": manifest["artifacts"],
        "inference_code_sha256": manifest["inference_code_sha256"],
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_physical_rd_five_artifacts(project_root: Path, config: PhysicalRDFiveConfig) -> PhysicalRDFiveArtifacts:
    processed = project_root / "data" / "processed"
    index_path = processed / config.index_name
    normalization_path = processed / config.normalization_stats_name
    feature_path = processed / config.feature_stats_name
    if not all(path.exists() for path in (index_path, normalization_path, feature_path)):
        raise FileNotFoundError("five-class physical RD artifact file is missing")
    index = pd.read_csv(index_path, low_memory=False)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    feature_stats = json.loads(feature_path.read_text(encoding="utf-8"))
    if normalization.get("schema_version") != RD_FIVE_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("five-class normalization schema is incompatible")
    if feature_stats.get("schema_version") != "rd_physical_five_feature_stats_v1":
        raise ValueError("five-class feature statistics schema is incompatible")
    if normalization.get("labels") != list(RD_FIVE_LABELS) or feature_stats.get("labels") != list(RD_FIVE_LABELS):
        raise ValueError("five-class artifact label order must match LOW_ALTITUDE_CLASSES")
    if feature_stats.get("quality_names") != list(RD_FIVE_QUALITY_NAMES):
        raise ValueError("five-class feature stats contain an invalid quality field list")
    if "canonical_time" in feature_stats.get("quality_names", []):
        raise ValueError("pairing-derived canonical_time is forbidden in five-class RD inputs")
    index_sha = sha256_file(index_path)
    if normalization.get("index_sha256") != index_sha or feature_stats.get("index_sha256") != index_sha:
        raise ValueError("five-class artifact index provenance mismatch")
    if feature_stats.get("normalization_stats_sha256") != sha256_file(normalization_path):
        raise ValueError("five-class feature stats normalization provenance mismatch")
    expected_normalization_generator = sha256_file(project_root / "src/process/build_rd_physical_five_normalization_stats.py")
    expected_feature_generator = sha256_file(project_root / "src/process/build_rd_physical_five_feature_stats.py")
    if normalization.get("generator_code_sha256") != expected_normalization_generator:
        raise ValueError("five-class normalization generator provenance mismatch")
    if feature_stats.get("generator_code_sha256") != expected_feature_generator:
        raise ValueError("five-class feature stats generator provenance mismatch")
    if normalization.get("split_name") != "train" or feature_stats.get("split_name") != "train":
        raise ValueError("five-class statistics must be train-only")
    if normalization.get("test_evaluation_executed") or feature_stats.get("test_evaluation_executed"):
        raise ValueError("five-class statistics must not use test evaluation")
    center = feature_stats.get("center_median")
    scale = feature_stats.get("scale_iqr_over_1_349")
    if not isinstance(center, list) or not isinstance(scale, list) or len(center) != 14 or len(scale) != 14:
        raise ValueError("five-class feature stats dimensions are invalid")
    if any(float(value) != float(value) for value in center + scale) or any(float(value) <= 0.0 for value in scale):
        raise ValueError("five-class feature stats contain non-finite or non-positive scale")
    frame_names = sorted(index.get("physical_frame_index_name", pd.Series(dtype=str)).astype(str).unique().tolist())
    if len(frame_names) != 1 or not frame_names[0]:
        raise ValueError("five-class index has invalid physical frame index provenance")
    frame_path = processed / frame_names[0]
    if not frame_path.exists():
        raise FileNotFoundError(f"five-class physical frame index missing: {frame_path}")
    inference_hashes = {name: sha256_file(project_root / name) for name in INFERENCE_CODE_FILES}
    manifest: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_contract": {
            "labels": list(RD_FIVE_LABELS),
            "closed_set": True,
            "supports_unknown_or_ood_rejection": False,
            "unknown_label_is_rejection": False,
            "tr_entry_preserved": True,
            "rd_only_training": True,
        },
        "model_spec": {
            "version": FIVE_RD_MODEL_SCHEMA_VERSION,
            "feature_names": list(PHYSICAL_RD_FEATURE_NAMES),
            "quality_names": list(RD_FIVE_QUALITY_NAMES),
            "quality_dim": int(config.quality_dim),
            "input_channels": int(config.input_channels),
            "token_dim": int(config.token_dim),
            "dropout": float(config.dropout),
            "statistic_names": list(PHYSICAL_RD_STATISTIC_NAMES),
            "statistic_dim": len(PHYSICAL_RD_STATISTIC_NAMES),
            "spatial_pool_shape": [SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS],
            "temporal_pooling": TEMPORAL_POOLING_VERSION,
            "parent_partition": [list(group) for group in config.parent_partition],
            "deployment_head": config.deployment_head,
        },
        "selection_policy": {
            "reference_source": config.reference_source,
            "enforce_drone_bird_non_degradation": bool(config.enforce_drone_bird_non_degradation),
            "non_degradation_tolerance": float(config.non_degradation_tolerance),
            "drone_reference": {
                "precision": float(config.drone_reference_precision),
                "recall": float(config.drone_reference_recall),
                "f1": float(config.drone_reference_f1),
            },
            "bird_reference": {
                "precision": float(config.bird_reference_precision),
                "recall": float(config.bird_reference_recall),
                "f1": float(config.bird_reference_f1),
            },
            "priority": [
                "drone_bird_non_degradation",
                "window_macro_f1",
                "window_min_class_f1",
                "window_min_class_recall",
                "window_min_class_precision",
                "parent_macro_f1",
                "parent_min_class_f1",
                "parent_min_class_recall",
                "parent_min_class_precision",
            ],
        },
        "artifacts": {
            "window_index": {"name": config.index_name, "sha256": index_sha, "row_count": int(len(index))},
            "normalization": {"name": config.normalization_stats_name, "sha256": sha256_file(normalization_path)},
            "feature_stats": {"name": config.feature_stats_name, "sha256": sha256_file(feature_path)},
            "physical_frame_index": {"name": frame_names[0], "sha256": sha256_file(frame_path)},
            "canonical_tensors": _canonical_tensor_inventory(project_root, index),
        },
        "inference_code_sha256": inference_hashes,
    }
    manifest["inference_contract_hash"] = _inference_contract_hash(manifest)
    return PhysicalRDFiveArtifacts(config=config, normalization=normalization, feature_stats=feature_stats, manifest=manifest)


def build_physical_rd_five_model(
    artifacts: PhysicalRDFiveArtifacts,
    *,
    head_mode: str = "five",
) -> PhysicalRDFiveClassifier:
    amplitude = artifacts.normalization["amplitude_db"]
    return PhysicalRDFiveClassifier(
        input_channels=artifacts.config.input_channels,
        token_dim=artifacts.config.token_dim,
        quality_dim=artifacts.config.quality_dim,
        dropout=artifacts.config.dropout,
        amplitude_clip_low_db=float(amplitude["clip_low"]),
        amplitude_clip_high_db=float(amplitude["clip_high"]),
        statistics_center=torch.tensor(artifacts.feature_stats["center_median"]),
        statistics_scale=torch.tensor(artifacts.feature_stats["scale_iqr_over_1_349"]),
        parent_partition=artifacts.config.parent_partition,
        head_mode=head_mode,
    )


def checkpoint_payload(*, epoch: int, artifacts: PhysicalRDFiveArtifacts, model: PhysicalRDFiveClassifier, validation: dict[str, Any], selection: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "config": asdict(artifacts.config),
        "artifact_manifest": artifacts.manifest,
        "model_state_dict": model.state_dict(),
        "val": validation,
    }
    if selection is not None:
        payload["selection"] = selection
    return payload


def load_physical_rd_five_checkpoint(checkpoint_path: Path, project_root: Path, device: torch.device) -> LoadedPhysicalRDFiveCheckpoint:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("five-class checkpoint uses an unsupported schema")
    raw_config = checkpoint.get("config")
    manifest = checkpoint.get("artifact_manifest")
    if not isinstance(raw_config, dict) or not isinstance(manifest, dict):
        raise ValueError("five-class checkpoint is missing config or artifact manifest")
    config = _strict_config(dict(raw_config))
    artifacts = load_physical_rd_five_artifacts(project_root, config)
    if manifest.get("inference_contract_hash") != artifacts.manifest.get("inference_contract_hash"):
        raise ValueError("five-class checkpoint inference contract mismatch")
    model = build_physical_rd_five_model(artifacts).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return LoadedPhysicalRDFiveCheckpoint(model=model, config=config, artifacts=artifacts, checkpoint=checkpoint)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LoadedPhysicalRDFiveCheckpoint",
    "PhysicalRDFiveArtifacts",
    "build_physical_rd_five_model",
    "checkpoint_payload",
    "load_physical_rd_five_artifacts",
    "load_physical_rd_five_checkpoint",
    "sha256_file",
]
