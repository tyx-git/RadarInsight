from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .rd_physical_drone_bird_config import PhysicalRDDroneBirdConfig
from .data.rd_physical_window_dataset import PHYSICAL_RD_FEATURE_NAMES, PHYSICAL_RD_INPUT_CHANNELS, QUALITY_FEATURE_DIM, QUALITY_FEATURE_NAMES, QUALITY_METADATA_NAMES, RD_ONLY_QUALITY_NAMES, LEGACY_QUALITY_SCHEMA
from .rd_physical_schema import PHYSICAL_RD_CLEAN_FEATURE_STATS_SCHEMA_VERSION, PHYSICAL_RD_FEATURE_STATS_SCHEMA_VERSION, PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION
from .rd_physical_drone_bird_model import (
    SPATIAL_POOL_DOPPLER_BINS,
    SPATIAL_POOL_RANGE_BINS,
    TEMPORAL_POOLING_VERSION,
    PhysicalRDDroneBirdClassifier,
)
from .rd_physical_statistics import PHYSICAL_RD_STATISTIC_NAMES


CHECKPOINT_SCHEMA_VERSION = "physical_rd_drone_bird_checkpoint_v9"
MODEL_SPEC_VERSION = "physical_rd_drone_bird_model_v9"
INFERENCE_CODE_FILES = (
    "src/core/data/rd_physical_window_dataset.py",
    "src/core/rd_physical_statistics.py",
    "src/core/rd_physical_drone_bird_model.py",
    "src/core/rd_physical_artifacts.py",
    "src/core/rd_physical_schema.py",
)
PROVENANCE_CODE_FILES = (
    "src/core/rd_physical_drone_bird_trainer.py",
    "src/core/evaluate_rd_physical_drone_bird.py",
)
ARTIFACT_GENERATOR_FILES = {
    "normalization_generator_code_sha256": "src/process/build_rd_physical_normalization_stats.py",
    "feature_stats_generator_code_sha256": "src/process/build_rd_physical_feature_stats.py",
}


def _artifact_generator_files(config: PhysicalRDDroneBirdConfig) -> dict[str, str]:
    if config.quality_schema == LEGACY_QUALITY_SCHEMA:
        return ARTIFACT_GENERATOR_FILES
    return {
        "normalization_generator_code_sha256": "src/process/build_rd_physical_normalization_stats.py",
        "feature_stats_generator_code_sha256": "src/process/build_rd_physical_clean_feature_stats.py",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tensor_inventory(project_root: Path, frame_index: pd.DataFrame) -> dict[str, Any]:
    paths = sorted(
        {
            value.strip()
            for value in frame_index["rd_tensor_path"].astype(str).tolist()
            if value.strip() and value.strip().lower() != "nan"
        }
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for raw_path in paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.exists():
            raise FileNotFoundError(f"physical RD canonical tensor is missing: {resolved}")
        relative_name = resolved.relative_to(project_root).as_posix()
        file_size = resolved.stat().st_size
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_size).encode("ascii"))
        digest.update(b"\0")
        with resolved.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        total_bytes += file_size
    return {
        "algorithm": "sha256-canonical-tensor-file-set-v1",
        "file_count": len(paths),
        "total_bytes": int(total_bytes),
        "content_sha256": digest.hexdigest(),
    }


def _strict_config(values: dict[str, Any]) -> PhysicalRDDroneBirdConfig:
    allowed = {field.name for field in fields(PhysicalRDDroneBirdConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown PhysicalRDDroneBirdConfig fields: {', '.join(unknown)}")
    return PhysicalRDDroneBirdConfig(**values)


@dataclass(frozen=True)
class PhysicalRDArtifacts:
    config: PhysicalRDDroneBirdConfig
    normalization: dict[str, Any]
    feature_stats: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedPhysicalRDCheckpoint:
    model: PhysicalRDDroneBirdClassifier
    config: PhysicalRDDroneBirdConfig
    artifacts: PhysicalRDArtifacts
    checkpoint: dict[str, Any]


def _physical_frame_index_name(index_frame: pd.DataFrame) -> str:
    if "physical_frame_index_name" not in index_frame.columns:
        raise ValueError("physical RD window index lacks physical_frame_index_name")
    values = sorted(index_frame["physical_frame_index_name"].astype(str).unique().tolist())
    if len(values) != 1 or not values[0]:
        raise ValueError(f"physical RD window index has invalid physical frame index references: {values}")
    return values[0]


def _model_spec(config: PhysicalRDDroneBirdConfig) -> dict[str, Any]:
    quality_names = list(QUALITY_FEATURE_NAMES) if config.quality_schema == LEGACY_QUALITY_SCHEMA else list(RD_ONLY_QUALITY_NAMES)
    return {
        "version": MODEL_SPEC_VERSION,
        "feature_names": list(PHYSICAL_RD_FEATURE_NAMES),
        "input_channels": int(config.input_channels),
        "quality_names": quality_names,
        "quality_dim": int(config.quality_dim),
        "include_missing_rd": bool(config.include_missing_rd),
        "statistic_names": list(PHYSICAL_RD_STATISTIC_NAMES),
        "statistic_dim": len(PHYSICAL_RD_STATISTIC_NAMES),
        "token_dim": int(config.token_dim),
        "dropout": float(config.dropout),
        "spatial_pool_shape": [SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS],
        "temporal_pooling": TEMPORAL_POOLING_VERSION,
    }


def _inference_contract_hash(payload: dict[str, Any]) -> str:
    contract = {
        "task_contract": payload["task_contract"],
        "model_spec": payload["model_spec"],
        "artifacts": payload["artifacts"],
        "inference_code_sha256": payload["inference_code_sha256"],
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_physical_rd_artifacts(project_root: Path, config: PhysicalRDDroneBirdConfig) -> PhysicalRDArtifacts:
    data_root = project_root / "data"
    processed = data_root / "processed"
    index_path = processed / config.index_name
    normalization_path = processed / config.normalization_stats_name
    feature_stats_path = processed / config.feature_stats_name
    if not index_path.exists() or not normalization_path.exists() or not feature_stats_path.exists():
        raise FileNotFoundError("physical RD artifact file is missing")
    index = pd.read_csv(index_path, low_memory=False)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    feature_stats = json.loads(feature_stats_path.read_text(encoding="utf-8"))
    if normalization.get("schema_version") != PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("physical RD normalization schema is incompatible with the active input contract")
    expected_stats_schema = PHYSICAL_RD_FEATURE_STATS_SCHEMA_VERSION if config.quality_schema == LEGACY_QUALITY_SCHEMA else PHYSICAL_RD_CLEAN_FEATURE_STATS_SCHEMA_VERSION
    if feature_stats.get("schema_version") != expected_stats_schema:
        raise ValueError("physical RD feature statistics schema is incompatible with the active input contract")
    if normalization.get("labels") != ["DroneTarget", "BirdTarget"] or feature_stats.get("labels") != ["DroneTarget", "BirdTarget"]:
        raise ValueError("physical RD artifacts must explicitly declare the Drone/Bird label order")
    if feature_stats.get("feature_names") != list(PHYSICAL_RD_FEATURE_NAMES):
        raise ValueError("physical RD feature statistics do not match the active feature schema")
    expected_quality_names = list(QUALITY_FEATURE_NAMES) if config.quality_schema == LEGACY_QUALITY_SCHEMA else list(RD_ONLY_QUALITY_NAMES)
    if feature_stats.get("quality_names") != expected_quality_names:
        raise ValueError("physical RD feature statistics do not match the active learnable quality schema")
    if feature_stats.get("quality_metadata_names") != expected_quality_names:
        raise ValueError("physical RD feature statistics do not match the active quality metadata schema")
    index_sha = sha256_file(index_path)
    if normalization.get("index_sha256") != index_sha:
        raise ValueError("physical RD normalization does not match active index")
    if feature_stats.get("index_sha256") != index_sha:
        raise ValueError("physical RD feature stats do not match active index")
    if feature_stats.get("normalization_stats_sha256") != sha256_file(normalization_path):
        raise ValueError("physical RD feature stats do not match normalization artifact")
    generator_files = _artifact_generator_files(config)
    if normalization.get("generator_code_sha256") != sha256_file(project_root / generator_files["normalization_generator_code_sha256"]):
        raise ValueError("physical RD normalization generator code does not match artifact")
    if feature_stats.get("generator_code_sha256") != sha256_file(project_root / generator_files["feature_stats_generator_code_sha256"]):
        raise ValueError("physical RD feature statistics generator code does not match artifact")
    frame_index_name = _physical_frame_index_name(index)
    frame_index_path = processed / frame_index_name
    if not frame_index_path.exists():
        raise FileNotFoundError(f"physical RD frame index is missing: {frame_index_path}")
    tensor_inventory = _canonical_tensor_inventory(project_root, index)
    amplitude = normalization.get("amplitude_db")
    if not isinstance(amplitude, dict):
        raise ValueError("physical RD normalization lacks amplitude_db")
    center = feature_stats.get("center_median")
    scale = feature_stats.get("scale_iqr_over_1_349")
    if not isinstance(center, list) or not isinstance(scale, list) or len(center) != 14 or len(scale) != 14:
        raise ValueError("physical RD feature stats have invalid dimensions")
    statistic_names = feature_stats.get("statistic_names")
    if statistic_names != list(PHYSICAL_RD_STATISTIC_NAMES):
        raise ValueError("physical RD feature stats statistic order does not match the active contract")
    inference_code_hashes = {name: sha256_file(project_root / name) for name in INFERENCE_CODE_FILES}
    provenance_code_hashes = {name: sha256_file(project_root / name) for name in PROVENANCE_CODE_FILES}
    manifest = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_contract": {
            "labels": ["DroneTarget", "BirdTarget"],
            "closed_set": True,
            "supports_unknown_or_ood_rejection": False,
        },
        "model_spec": _model_spec(config),
        "feature_names": list(PHYSICAL_RD_FEATURE_NAMES),
        "quality_names": expected_quality_names,
        "quality_metadata_names": expected_quality_names,
        "artifacts": {
            "window_index": {"name": config.index_name, "sha256": index_sha, "row_count": int(len(index))},
            "normalization": {"name": config.normalization_stats_name, "sha256": sha256_file(normalization_path)},
            "feature_stats": {"name": config.feature_stats_name, "sha256": sha256_file(feature_stats_path)},
            "physical_frame_index": {"name": frame_index_name, "sha256": sha256_file(frame_index_path)},
            "canonical_tensors": tensor_inventory,
        },
        "inference_code_sha256": inference_code_hashes,
        "provenance_code_sha256": provenance_code_hashes,
    }
    manifest["inference_contract_hash"] = _inference_contract_hash(manifest)
    return PhysicalRDArtifacts(config=config, normalization=normalization, feature_stats=feature_stats, manifest=manifest)


def build_physical_rd_model(artifacts: PhysicalRDArtifacts) -> PhysicalRDDroneBirdClassifier:
    config = artifacts.config
    amplitude = artifacts.normalization["amplitude_db"]
    return PhysicalRDDroneBirdClassifier(
        input_channels=config.input_channels,
        token_dim=config.token_dim,
        quality_dim=config.quality_dim,
        quality_schema=config.quality_schema,
        dropout=config.dropout,
        amplitude_clip_low_db=float(amplitude["clip_low"]),
        amplitude_clip_high_db=float(amplitude["clip_high"]),
        statistics_center=torch.tensor(artifacts.feature_stats["center_median"]),
        statistics_scale=torch.tensor(artifacts.feature_stats["scale_iqr_over_1_349"]),
    )


def checkpoint_payload(*, epoch: int, artifacts: PhysicalRDArtifacts, model: PhysicalRDDroneBirdClassifier, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "config": asdict(artifacts.config),
        "artifact_manifest": artifacts.manifest,
        "model_state_dict": model.state_dict(),
        "val": validation,
    }


def _assert_manifest_matches(project_root: Path, manifest: dict[str, Any], config: PhysicalRDDroneBirdConfig) -> None:
    expected = load_physical_rd_artifacts(project_root, config).manifest
    if manifest.get("inference_contract_hash") != expected.get("inference_contract_hash"):
        raise ValueError("physical RD checkpoint inference contract does not match active artifacts or code")
    for key in ("task_contract", "model_spec", "feature_names", "quality_names", "quality_metadata_names", "artifacts", "inference_code_sha256"):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"physical RD checkpoint inference contract mismatch: {key}")


def load_physical_rd_checkpoint(checkpoint_path: Path, project_root: Path, device: torch.device) -> LoadedPhysicalRDCheckpoint:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("physical RD checkpoint uses a legacy or unsupported schema")
    raw_config = checkpoint.get("config")
    manifest = checkpoint.get("artifact_manifest")
    if not isinstance(raw_config, dict) or not isinstance(manifest, dict):
        raise ValueError("physical RD checkpoint is missing config or artifact manifest")
    config = _strict_config(dict(raw_config))
    _assert_manifest_matches(project_root, manifest, config)
    artifacts = load_physical_rd_artifacts(project_root, config)
    model = build_physical_rd_model(artifacts).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return LoadedPhysicalRDCheckpoint(model=model, config=config, artifacts=artifacts, checkpoint=checkpoint)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LoadedPhysicalRDCheckpoint",
    "PhysicalRDArtifacts",
    "build_physical_rd_model",
    "checkpoint_payload",
    "load_physical_rd_artifacts",
    "load_physical_rd_checkpoint",
    "sha256_file",
]
