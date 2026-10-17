"""Paired r20 training for the structured radial-speed decomposition."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from v27.V27d.r19.training import load_checkpoint as load_r19_checkpoint
from v27.v27a.checkpoint import tensor_state_hash

from .model import R20L2Model
from .protocol import R20_ARMS, R20_CHECKPOINT_SCHEMA, R20_EXPERIMENT_ID, R20_L2_CLASSES, R20Config, artifact_path, canonical_sha256, load_manifest, project_path, sha256_file








def _freeze_unused_heads(model: R20L2Model) -> None:
    heads = getattr(model.base_model, "trajectory_heads", None)
    if heads is None:
        raise ValueError("r20 base model has no trajectory heads to freeze")
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in heads.parameters()):
        raise AssertionError("r20 trajectory heads were not frozen")


def _build_from_r19(root: Path, config: R20Config, *, device: torch.device) -> tuple[R20L2Model, dict[str, Any]]:
    source_path = project_path(root, config.source["r19_checkpoint_path"], must_exist=True)
    model, payload = load_r19_checkpoint(
        root,
        config.r19,
        arm="snr_missing_only",
        seed=int(config.source["r19_checkpoint_seed"]),
        checkpoint_path=source_path,
    )
    if sha256_file(source_path) != config.source["r19_checkpoint_sha256"]:
        raise ValueError("r20 direct r19 checkpoint hash differs")
    if payload.get("model_state_sha256") != config.source["r19_checkpoint_model_state_sha256"]:
        raise ValueError("r20 direct r19 checkpoint state hash differs")
    if (
        payload.get("arm") != config.source["r19_checkpoint_arm"]
        or int(payload.get("seed", -1)) != int(config.source["r19_checkpoint_seed"])
        or int(payload.get("epoch", -1)) != int(config.source["r19_checkpoint_epoch"])
    ):
        raise ValueError("r20 direct r19 checkpoint identity differs")
    wrapped = R20L2Model(config, base_model=copy.deepcopy(model).cpu(), arm="zero_control")
    _freeze_unused_heads(wrapped)
    return wrapped.to(device), dict(payload)




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
    config: R20Config,
    *,
    arm: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> tuple[R20L2Model, dict[str, Any]]:
    if arm not in R20_ARMS:
        raise ValueError("r20 checkpoint arm is invalid")
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r20 checkpoint seed is not registered")
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(root, config, allow_code_drift=True)
    path = (
        artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        if checkpoint_path is None
        else project_path(root, checkpoint_path, must_exist=True)
    )
    try:
        path.relative_to((root / "src-v3" / "v27" / "V27d" / "r20").resolve())
    except ValueError as exc:
        raise ValueError("r20 checkpoint escaped the experiment root") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r20 checkpoint is not an object")
    if (
        payload.get("checkpoint_schema") != R20_CHECKPOINT_SCHEMA
        or payload.get("experiment_id") != R20_EXPERIMENT_ID
        or payload.get("arm") != arm
        or payload.get("task") != "l2"
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", 0)) <= 0
        or int(payload.get("epoch", 0)) > int(config.training["epochs"])
        or payload.get("config_sha256") != config.sha256
        or payload.get("config_identity_sha256") != config.identity_sha256
        or payload.get("manifest_path") != manifest_path.relative_to(root).as_posix()
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("source_r19_checkpoint_sha256") != config.source["r19_checkpoint_sha256"]
        or payload.get("source_r19_checkpoint_model_state_sha256") != config.source["r19_checkpoint_model_state_sha256"]
        or payload.get("raw_radial_scalar_removed") is not True
        or payload.get("radial_sidecar_value_names") != list(config.data["radial_value_names"])
        or payload.get("radial_sidecar_valid_names") != list(config.data["radial_valid_names"])
        or payload.get("class_names") != list(R20_L2_CLASSES)
        or payload.get("loss_name") != config.training["loss_name"]
        or payload.get("loss_reduction") != config.training["loss_reduction"]
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or int(payload.get("rd_payload_access_count", -1)) != 0
    ):
        raise ValueError("r20 checkpoint provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("r20 checkpoint model state hash differs")
    model, _ = _build_from_r19(root, config, device=torch.device("cpu"))
    model.arm = arm
    if payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("r20 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    _freeze_unused_heads(model)
    return model.eval(), dict(payload)




__all__ = ["load_checkpoint"]