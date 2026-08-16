from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .micro_doppler_residual import (
    MicroDopplerLateResidual,
    V19_CACHE_FIELDS,
    V19_CLASS_COUNT,
    V19_CLASS_NAMES,
    V19_FEATURE_DIM,
    V19_RESIDUAL_BOUND,
    V19_TRAINABLE_PARAMETER_COUNT,
    build_residual_dataloader,
    evaluate_residual,
    normalize_stable8,
    residual_objective,
    sha256_file,
    strip_evaluation_records,
    tensor_state_sha256,
)


V19R_SCHEMA_VERSION = "RADAR-LN3-V19r-micro-doppler-late-residual-v1"
V19R_CACHE_SCHEMA_VERSION = (
    "RADAR-LN3-V19r-cq08-dbb-frozen-v18r-logits-causal-stable8-cache-v1"
)
V19R_CHECKPOINT_SCHEMA_VERSION = (
    "RADAR-LN3-V19r-dbb-micro-doppler-late-residual-checkpoint-v1"
)
V19R_CONFIG_NAME = (
    "RADAR-LN3-V19r-cq08-dbb-specialist-v18r-micro-doppler-late-residual-seed42"
)
V19R_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V19r-cq08-dbb-specialist-v18r-micro-doppler-late-residual-"
    "seed42-config.json"
)
V19R_TRAIN_CACHE_RELATIVE_PATH = Path(
    "data/processed/"
    "RADAR-LN3-V19r-cq08-dbb-train-v18r-micro-doppler-late-residual-cache.npz"
)
V19R_VALIDATION_CACHE_RELATIVE_PATH = Path(
    "data/processed/"
    "RADAR-LN3-V19r-cq08-dbb-val-v18r-micro-doppler-late-residual-cache.npz"
)
V19R_CONTRACT_RELATIVE_PATH = Path(
    "data/processed/"
    "RADAR-LN3-V19r-cq08-dbb-v18r-micro-doppler-late-residual-contract.json"
)
V19R_READINESS_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19r-v18r-micro-doppler-late-residual-train-readiness.json"
)
V19R_REPORT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19r-v18r-micro-doppler-late-residual-validation.json"
)
V19R_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/"
    "RADAR-LN3-V19r-cq08-dbb-specialist-v18r-micro-doppler-late-residual-"
    "seed42-formal"
)
V19R_CACHE_FIELDS = V19_CACHE_FIELDS


class V19rResidualCacheDataset(Dataset[dict[str, Any]]):
    """Version-isolated cache reader for frozen V18r logits and stable8 summaries."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            if tuple(payload.files) != V19R_CACHE_FIELDS:
                raise ValueError(
                    "V19r cache fields differ: "
                    f"expected={V19R_CACHE_FIELDS}, actual={payload.files}"
                )
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        schema = str(self.arrays["schema_version"].item())
        if schema != V19R_CACHE_SCHEMA_VERSION:
            raise ValueError(f"V19r cache schema differs: {schema}")
        self.split_name = str(self.arrays["split_name"].item())
        self.parent_sample_ids = self.arrays["parent_sample_ids"].astype(str)
        self.track_ids = self.arrays["track_ids"].astype(str)
        self.source_names = self.arrays["source_names"].astype(str)
        self.label_names = self.arrays["label_names"].astype(str)
        self.labels = np.asarray(self.arrays["labels"], dtype=np.int64)
        self.lengths = np.asarray(self.arrays["lengths"], dtype=np.int64)
        self.padding_mask = np.asarray(self.arrays["padding_mask"], dtype=np.bool_)
        self.micro_doppler_valid = np.asarray(
            self.arrays["micro_doppler_valid"], dtype=np.bool_
        )
        self.valid_counts = np.asarray(self.arrays["valid_counts"], dtype=np.int64)
        self.baseline_logits = np.asarray(
            self.arrays["baseline_logits"], dtype=np.float32
        )
        self.causal_features = np.asarray(
            self.arrays["causal_features"], dtype=np.float32
        )
        self._validate()

    def _validate(self) -> None:
        count = len(self.parent_sample_ids)
        vectors = (
            self.track_ids,
            self.source_names,
            self.label_names,
            self.labels,
            self.lengths,
        )
        if any(len(value) != count for value in vectors):
            raise ValueError("V19r cache parent metadata lengths differ")
        if len(set(self.parent_sample_ids.tolist())) != count:
            raise ValueError("V19r cache repeats a parent_sample_id")
        if self.padding_mask.ndim != 2:
            raise ValueError("V19r cache padding mask must have shape [parent,sequence]")
        expected_sequence = self.padding_mask.shape
        if (
            self.micro_doppler_valid.shape != expected_sequence
            or self.valid_counts.shape != expected_sequence
            or self.baseline_logits.shape != (*expected_sequence, V19_CLASS_COUNT)
            or self.causal_features.shape != (*expected_sequence, V19_FEATURE_DIM)
        ):
            raise ValueError("V19r cache tensor shapes differ")
        if not np.isfinite(self.baseline_logits).all() or not np.isfinite(
            self.causal_features
        ).all():
            raise ValueError("V19r cache contains non-finite tensors")
        if np.any(self.labels < 0) or np.any(self.labels >= V19_CLASS_COUNT):
            raise ValueError("V19r cache labels are outside [0,2]")
        active_lengths = (~self.padding_mask).sum(axis=1)
        if not np.array_equal(active_lengths, self.lengths):
            raise ValueError("V19r cache lengths differ from padding masks")
        if np.any(self.lengths <= 0):
            raise ValueError("V19r cache contains an empty parent")
        if np.any(self.micro_doppler_valid & self.padding_mask):
            raise ValueError("V19r cache marks padded micro-Doppler evidence as valid")
        if np.any(self.valid_counts[self.padding_mask] != 0):
            raise ValueError("V19r cache padded valid counts must be zero")
        expected_counts = np.cumsum(
            self.micro_doppler_valid & ~self.padding_mask, axis=1, dtype=np.int64
        )
        expected_counts[self.padding_mask] = 0
        if not np.array_equal(expected_counts, self.valid_counts):
            raise ValueError("V19r cache valid counts are not causal cumulative counts")
        if np.any(self.causal_features[self.valid_counts == 0] != 0.0):
            raise ValueError("V19r cache must be zero before the first valid observation")

    def __len__(self) -> int:
        return len(self.parent_sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "parent_sample_id": str(self.parent_sample_ids[index]),
            "track_id": str(self.track_ids[index]),
            "source_name": str(self.source_names[index]),
            "label_name": str(self.label_names[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "length": torch.tensor(int(self.lengths[index]), dtype=torch.long),
            "padding_mask": torch.from_numpy(self.padding_mask[index].copy()),
            "micro_doppler_valid": torch.from_numpy(
                self.micro_doppler_valid[index].copy()
            ),
            "valid_counts": torch.from_numpy(self.valid_counts[index].copy()),
            "baseline_logits": torch.from_numpy(self.baseline_logits[index].copy()),
            "causal_features": torch.from_numpy(self.causal_features[index].copy()),
        }


def _model_contract() -> dict[str, Any]:
    return {
        "feature_dim": V19_FEATURE_DIM,
        "class_count": V19_CLASS_COUNT,
        "bias": False,
        "zero_sum": True,
        "residual_bound": V19_RESIDUAL_BOUND,
        "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
    }


def save_v19r_residual_checkpoint(
    path: Path,
    model: MicroDopplerLateResidual,
    optimizer: torch.optim.Optimizer,
    *,
    config_name: str,
    config_sha256: str,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema": V19R_CHECKPOINT_SCHEMA_VERSION,
        "config_name": str(config_name),
        "config_sha256": str(config_sha256),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_contract": _model_contract(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "training_state": dict(training_state),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_v19r_residual_checkpoint(
    path: Path,
    model: MicroDopplerLateResidual,
    *,
    expected_config_name: str,
    expected_config_sha256: str,
    expected_provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_schema") != V19R_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V19r checkpoint schema differs")
    if payload.get("config_name") != expected_config_name:
        raise ValueError("V19r checkpoint config name differs")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("V19r checkpoint config hash differs")
    if dict(payload.get("provenance") or {}) != dict(expected_provenance):
        raise ValueError("V19r checkpoint provenance differs")
    if dict(payload.get("model_contract") or {}) != _model_contract():
        raise ValueError("V19r checkpoint model contract differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def require_v19r_parent_order(
    actual: Sequence[str], expected_sha256: str, *, epoch: int
) -> str:
    digest = hashlib.sha256()
    for parent_id in actual:
        encoded = str(parent_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    value = digest.hexdigest()
    if value != str(expected_sha256):
        raise ValueError(
            f"V19r epoch {epoch} parent order differs from frozen V18r: "
            f"actual={value}, expected={expected_sha256}"
        )
    return value


__all__ = [
    "MicroDopplerLateResidual",
    "V19R_ARTIFACT_RELATIVE_PATH",
    "V19R_CACHE_FIELDS",
    "V19R_CACHE_SCHEMA_VERSION",
    "V19R_CHECKPOINT_SCHEMA_VERSION",
    "V19R_CONFIG_NAME",
    "V19R_CONFIG_RELATIVE_PATH",
    "V19R_CONTRACT_RELATIVE_PATH",
    "V19R_READINESS_RELATIVE_PATH",
    "V19R_REPORT_RELATIVE_PATH",
    "V19R_SCHEMA_VERSION",
    "V19R_TRAIN_CACHE_RELATIVE_PATH",
    "V19R_VALIDATION_CACHE_RELATIVE_PATH",
    "V19rResidualCacheDataset",
    "V19_CLASS_COUNT",
    "V19_CLASS_NAMES",
    "V19_FEATURE_DIM",
    "V19_RESIDUAL_BOUND",
    "V19_TRAINABLE_PARAMETER_COUNT",
    "build_residual_dataloader",
    "evaluate_residual",
    "load_v19r_residual_checkpoint",
    "normalize_stable8",
    "require_v19r_parent_order",
    "residual_objective",
    "save_v19r_residual_checkpoint",
    "sha256_file",
    "strip_evaluation_records",
    "tensor_state_sha256",
]
