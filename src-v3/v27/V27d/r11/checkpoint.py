"""Fail-closed checkpoint helpers for V27d-r11."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, write_torch_if_absent_or_identical

from .model import BirdVerifierR11
from .protocol import R11_CHECKPOINT_SCHEMA_VERSION, artifact_path


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r11 state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    if not parts:
        raise ValueError("V27d-r11 checkpoint state cannot be empty")
    return canonical_sha256(parts)


def save_checkpoint(
    path: str | Path,
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    model: BirdVerifierR11,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    cache_hashes: Mapping[str, str],
) -> Path:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    contract = model.model_contract()
    source = config["source"]
    payload = {
        "checkpoint_schema": R11_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": "V27d-r11",
        "arm": model.arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": str(config["config_path"].relative_to(root)).replace("\\", "/"),
        "config_sha256": config["config_sha256"],
        "config_identity_sha256": config["config_identity_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cache_payload_sha256s": {str(key): str(value) for key, value in sorted(cache_hashes.items())},
        "r6_checkpoint_file_sha256": source["expected_checkpoint_file_sha256s"][str(seed)],
        "r6_checkpoint_head_state_sha256": source["expected_checkpoint_head_state_sha256s"][str(seed)],
        "loss_name": config["training"]["loss_name"],
        "loss_reduction": config["training"]["loss_reduction"],
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "metrics": dict(metrics),
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
        "frozen_liquid_v2_core": True,
        "frozen_base_readout": True,
        "frozen_verifier_head": True,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }
    target = artifact_path(root, path)
    write_torch_if_absent_or_identical(target, payload)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    model: BirdVerifierR11,
    seed: int,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    target = artifact_path(root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r11 checkpoint must be a mapping")
    payload = dict(payload)
    if payload.get("checkpoint_schema") != R11_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V27d-r11 checkpoint schema differs")
    if payload.get("experiment_id") != "V27d-r11" or payload.get("arm") != model.arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r11 checkpoint identity differs")
    if payload.get("config_sha256") != config["config_sha256"] or payload.get("config_identity_sha256") != config["config_identity_sha256"] or payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r11 checkpoint provenance differs")
    expected_cache_hashes = {str(key): str(value) for key, value in sorted(cache_hashes.items())}
    if payload.get("cache_payload_sha256s") != expected_cache_hashes:
        raise ValueError("V27d-r11 checkpoint cache provenance differs")
    source = config["source"]
    if payload.get("r6_checkpoint_file_sha256") != source["expected_checkpoint_file_sha256s"][str(seed)] or payload.get("r6_checkpoint_head_state_sha256") != source["expected_checkpoint_head_state_sha256s"][str(seed)]:
        raise ValueError("V27d-r11 r6 checkpoint provenance differs")
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r11 checkpoint model contract differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27d-r11 checkpoint state hash differs")
    if payload.get("unknown_payload_access_count") != 0 or payload.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r11 checkpoint scope differs")
    model.load_state_dict(state, strict=True)
    return payload


__all__ = ["load_checkpoint", "save_checkpoint", "tensor_state_hash"]
