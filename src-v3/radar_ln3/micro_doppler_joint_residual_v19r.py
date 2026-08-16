from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .micro_doppler_residual import (
    MicroDopplerLateResidual,
    V19_CLASS_NAMES,
    V19_FEATURE_DIM,
    V19_RESIDUAL_BOUND,
    V19_TRAINABLE_PARAMETER_COUNT,
    normalize_stable8,
    sha256_file,
    tensor_state_sha256,
)
from .task_models import LiquidTaskClassifier


V19R_JOINT_SCHEMA_VERSION = "RADAR-LN3-V19r-joint-late-residual-v1"
V19R_JOINT_CHECKPOINT_SCHEMA_VERSION = (
    "RADAR-LN3-V19r-dbb-full-joint-late-residual-checkpoint-v1"
)
V19R_JOINT_PAIRED_PROTOCOL = (
    "v19r-dbb-v18r-full-joint-plus-stable8-late-residual-v1"
)
V19R_JOINT_CONFIG_NAME = (
    "RADAR-LN3-V19r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-"
    "stable-density-joint-late-residual-seed42"
)
V19R_JOINT_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/"
    f"{V19R_JOINT_CONFIG_NAME}-config.json"
)
V19R_JOINT_CONTRACT_RELATIVE_PATH = Path(
    "data/processed/"
    "RADAR-LN3-V19r-cq08-dbb-stable8-density-joint-late-residual-contract.json"
)
V19R_JOINT_READINESS_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19r-stable8-density-joint-late-residual-training-readiness.json"
)
V19R_JOINT_REPORT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19r-stable8-density-joint-late-residual-validation.json"
)
V19R_JOINT_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/"
    f"{V19R_JOINT_CONFIG_NAME}-formal"
)


class V19rJointLateResidualModel(nn.Module):
    """Train the complete V18r backbone jointly with a zero-init stable8 residual."""

    def __init__(
        self,
        backbone: LiquidTaskClassifier,
        *,
        normalization_median: torch.Tensor,
        normalization_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if backbone.task_name != "dbb_specialist":
            raise ValueError("V19r joint residual requires a DBB specialist backbone")
        if tuple(backbone.class_names) != V19_CLASS_NAMES:
            raise ValueError("V19r joint residual backbone class order differs")
        if backbone.micro_doppler_dim != V19_FEATURE_DIM:
            raise ValueError("V19r joint residual requires the stable8 backbone input")
        median = normalization_median.detach().to(dtype=torch.float32, device="cpu")
        scale = normalization_scale.detach().to(dtype=torch.float32, device="cpu")
        if median.shape != (V19_FEATURE_DIM,) or scale.shape != (V19_FEATURE_DIM,):
            raise ValueError("V19r joint residual normalization must have shape [8]")
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
            raise ValueError("V19r joint residual normalization must be finite")
        if bool((scale <= 0.0).any()):
            raise ValueError("V19r joint residual normalization scale must be positive")
        self.backbone = backbone
        self.residual = MicroDopplerLateResidual()
        self.register_buffer("normalization_median", median.clone(), persistent=True)
        self.register_buffer("normalization_scale", scale.clone(), persistent=True)
        self.micro_doppler_dim = int(backbone.micro_doppler_dim)
        self.task_name = str(backbone.task_name)
        self.class_names = tuple(backbone.class_names)

    @property
    def backbone_trainable_parameter_count(self) -> int:
        return sum(
            value.numel() for value in self.backbone.parameters() if value.requires_grad
        )

    @property
    def residual_trainable_parameter_count(self) -> int:
        return sum(
            value.numel() for value in self.residual.parameters() if value.requires_grad
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        micro_features = kwargs.get("micro_doppler_features")
        micro_valid = kwargs.get("micro_doppler_valid")
        padding_mask = kwargs.get("tr_padding_mask")
        if (
            not isinstance(micro_features, torch.Tensor)
            or not isinstance(micro_valid, torch.Tensor)
            or not isinstance(padding_mask, torch.Tensor)
        ):
            raise ValueError("V19r joint residual requires stable8 values and masks")
        backbone_output = self.backbone(**kwargs)
        causal_features, valid_counts = normalize_stable8(
            micro_features,
            micro_valid,
            padding_mask,
            median=self.normalization_median,
            scale=self.normalization_scale,
        )
        residual_output = self.residual(
            backbone_output["sequence_logits"],
            causal_features,
            padding_mask,
        )
        output = dict(backbone_output)
        output.update(
            {
                "backbone_sequence_logits": backbone_output["sequence_logits"],
                "residual_logits": residual_output["residual_logits"],
                "causal_stable8_features": causal_features,
                "causal_stable8_valid_counts": valid_counts,
                "sequence_logits": residual_output["sequence_logits"],
            }
        )
        return output


def _checkpoint_model_contract(model: V19rJointLateResidualModel) -> dict[str, Any]:
    return {
        "class_names": list(model.class_names),
        "feature_dim": V19_FEATURE_DIM,
        "bias": False,
        "zero_sum": True,
        "residual_bound": V19_RESIDUAL_BOUND,
        "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
        "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
        "total_trainable_parameter_count": model.trainable_parameter_count,
    }


def save_v19r_joint_checkpoint(
    path: Path,
    model: V19rJointLateResidualModel,
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
        "checkpoint_schema": V19R_JOINT_CHECKPOINT_SCHEMA_VERSION,
        "config_name": str(config_name),
        "config_sha256": str(config_sha256),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_contract": _checkpoint_model_contract(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "training_state": dict(training_state),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_v19r_joint_checkpoint(
    path: Path,
    model: V19rJointLateResidualModel,
    *,
    expected_config_name: str,
    expected_config_sha256: str,
    expected_provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_schema") != V19R_JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V19r joint checkpoint schema differs")
    if payload.get("config_name") != expected_config_name:
        raise ValueError("V19r joint checkpoint config name differs")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("V19r joint checkpoint config hash differs")
    if dict(payload.get("provenance") or {}) != dict(expected_provenance):
        raise ValueError("V19r joint checkpoint provenance differs")
    if dict(payload.get("model_contract") or {}) != _checkpoint_model_contract(model):
        raise ValueError("V19r joint checkpoint model contract differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def require_v19r_joint_parent_order(
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
            f"V19r joint epoch {epoch} parent order differs from V18r: "
            f"actual={value}, expected={expected_sha256}"
        )
    return value


__all__ = [
    "V19R_JOINT_ARTIFACT_RELATIVE_PATH",
    "V19R_JOINT_CHECKPOINT_SCHEMA_VERSION",
    "V19R_JOINT_CONFIG_NAME",
    "V19R_JOINT_CONFIG_RELATIVE_PATH",
    "V19R_JOINT_CONTRACT_RELATIVE_PATH",
    "V19R_JOINT_PAIRED_PROTOCOL",
    "V19R_JOINT_READINESS_RELATIVE_PATH",
    "V19R_JOINT_REPORT_RELATIVE_PATH",
    "V19R_JOINT_SCHEMA_VERSION",
    "V19_TRAINABLE_PARAMETER_COUNT",
    "V19rJointLateResidualModel",
    "load_v19r_joint_checkpoint",
    "require_v19r_joint_parent_order",
    "save_v19r_joint_checkpoint",
    "sha256_file",
    "tensor_state_sha256",
]
