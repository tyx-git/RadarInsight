"""Paired r19 L2 training for the SNR-channel ablation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from v27.V27d.r17.frozen import FrozenR17Models, load_frozen_models
from v27.v27a.checkpoint import tensor_state_hash

from .protocol import R19_ARMS, R19_CHECKPOINT_SCHEMA, R19_EXPERIMENT_ID, R19_L2_CLASSES, R19Config, artifact_path, canonical_sha256, load_manifest, project_path










def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value




def load_checkpoint(
    project_root: str | Path,
    config: R19Config,
    *,
    arm: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    frozen: FrozenR17Models | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an r19 arm only when every provenance binding matches."""

    if arm not in R19_ARMS:
        raise ValueError("r19 checkpoint arm is invalid")
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(
        root, config, allow_code_drift=True
    )
    path = (
        artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        if checkpoint_path is None
        else project_path(root, checkpoint_path, must_exist=True)
    )
    try:
        path.relative_to((root / "src-v3" / "v27" / "V27d" / "r19").resolve())
    except ValueError as exc:
        raise ValueError("r19 checkpoint escaped experiment root") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r19 checkpoint is not an object")
    if (
        payload.get("checkpoint_schema") != R19_CHECKPOINT_SCHEMA
        or payload.get("experiment_id") != R19_EXPERIMENT_ID
        or payload.get("arm") != arm
        or payload.get("task") != "l2"
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", 0)) <= 0
        or int(payload.get("epoch", 0)) > int(config.training["epochs"])
        or payload.get("config_sha256") != config.sha256
        or payload.get("config_identity_sha256") != config.identity_sha256
        or payload.get("manifest_path") != manifest_path.relative_to(root).as_posix()
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("source_r18_manifest_sha256") != config.source["r18_manifest_sha256"]
        or payload.get("source_frozen_l2_checkpoint_sha256") != config.r17.source["r23_checkpoint_sha256"]
        or payload.get("snr_arm") != arm
        or payload.get("radial_speed_retained") is not True
        or payload.get("class_names") != list(R19_L2_CLASSES)
        or payload.get("loss_name") != config.training["loss_name"]
        or payload.get("loss_reduction") != config.training["loss_reduction"]
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or int(payload.get("rd_payload_access_count", -1)) != 0
    ):
        raise ValueError("r19 checkpoint provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("r19 checkpoint model state hash differs")
    source = frozen or load_frozen_models(root, config.r17)
    model = copy.deepcopy(source.l2_model).cpu()
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("r19 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    return model.eval(), dict(payload)




__all__ = ["load_checkpoint"]