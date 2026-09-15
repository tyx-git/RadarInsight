"""Fail-closed checkpoint serialization for V27a residual arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import (
    canonical_sha256,
    resolve_project_path,
    write_torch_if_absent_or_identical,
)

from .model import V27AEvidenceResidual
from .protocol import (
    V27A_ARMS,
    V27A_CHECKPOINT_SCHEMA_VERSION,
    V27AContext,
)


_SHA256_HEX = frozenset("0123456789abcdef")


def _cache_payload_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the immutable train/validation cache identities for a run."""

    if set(value) != {"train", "val"}:
        raise ValueError("V27a checkpoint requires train and val cache digests")
    result: dict[str, str] = {}
    for split in ("train", "val"):
        digest = value[split]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _SHA256_HEX for character in digest.lower())
        ):
            raise ValueError("V27a checkpoint cache digest is malformed")
        result[split] = digest.lower()
    return result


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensors deterministically, independent of torch archives."""

    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27a state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    if not parts:
        raise ValueError("V27a checkpoint state cannot be empty")
    return canonical_sha256(parts)


def _checkpoint_metadata(
    *,
    context: V27AContext,
    model: V27AEvidenceResidual,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    kind: str,
    cache_payload_sha256s: Mapping[str, Any],
) -> dict[str, Any]:
    if arm not in V27A_ARMS or model.arm != arm:
        raise ValueError("V27a checkpoint arm/model differs")
    if int(seed) not in context.config.training.seeds:
        raise ValueError("V27a checkpoint seed is not registered")
    if int(epoch) <= 0 or int(global_step) < 0:
        raise ValueError("V27a checkpoint epoch/global_step is invalid")
    if kind not in {"epoch", "selected", "smoke"}:
        raise ValueError("V27a checkpoint kind is invalid")
    contract = model.model_contract()
    return {
        "checkpoint_schema": V27A_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "checkpoint_kind": str(kind),
        "arm": str(arm),
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": context.config.config_path.relative_to(
            context.project_root
        ).as_posix(),
        "config_sha256": context.config.config_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "manifest_path": context.manifest_path.relative_to(
            context.project_root
        ).as_posix(),
        "manifest_sha256": context.manifest_sha256,
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "base_epoch": context.config.base.expected_l2_epoch,
        "cache_payload_sha256s": _cache_payload_hashes(cache_payload_sha256s),
        "loss_name": "parent_final_fixed_parent_count_ce",
        "loss_reduction": "fixed_parent_count",
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "metrics": dict(metrics),
    }


def save_v27a_checkpoint(
    path: str | Path,
    *,
    context: V27AContext,
    model: V27AEvidenceResidual,
    optimizer: torch.optim.Optimizer | None,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    kind: str,
    cache_payload_sha256s: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Save a checkpoint after binding every mutable component to V27a."""

    root = context.project_root
    target = resolve_project_path(root, path)
    metadata = _checkpoint_metadata(
        context=context,
        model=model,
        arm=arm,
        seed=seed,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        kind=kind,
        cache_payload_sha256s=cache_payload_sha256s,
    )
    state = model.state_dict()
    payload: dict[str, Any] = {
        **metadata,
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    # Epoch artifacts are immutable: a rerun may reuse byte-equivalent model
    # state, but it must never replace a different result at the same path.
    write_torch_if_absent_or_identical(target, payload)
    return target


def load_v27a_checkpoint(
    path: str | Path,
    *,
    context: V27AContext,
    model: V27AEvidenceResidual,
    optimizer: torch.optim.Optimizer | None = None,
    arm: str | None = None,
    seed: int | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and verify an exact V27a model/manifest/base checkpoint."""

    target = resolve_project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27a checkpoint must be a mapping")
    payload = dict(payload)
    if payload.get("checkpoint_schema") != V27A_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V27a checkpoint schema differs")
    if payload.get("experiment_id") != "V27a":
        raise ValueError("V27a checkpoint experiment id differs")
    actual_arm = str(payload.get("arm", ""))
    if actual_arm not in V27A_ARMS or actual_arm != model.arm:
        raise ValueError("V27a checkpoint arm differs")
    if arm is not None and str(arm) != actual_arm:
        raise ValueError("V27a requested checkpoint arm differs")
    actual_seed = payload.get("seed")
    if (
        isinstance(actual_seed, bool)
        or not isinstance(actual_seed, int)
        or actual_seed not in context.config.training.seeds
        or (seed is not None and int(seed) != actual_seed)
    ):
        raise ValueError("V27a checkpoint seed differs")
    if int(payload.get("epoch", 0)) <= 0 or int(payload.get("global_step", -1)) < 0:
        raise ValueError("V27a checkpoint epoch/global_step differs")
    from .runtime import cache_payload_digest, load_verified_cache

    cache_payload_sha256s = {
        split: cache_payload_digest(load_verified_cache(context, split))
        for split in ("train", "val")
    }
    expected = _checkpoint_metadata(
        context=context,
        model=model,
        arm=actual_arm,
        seed=actual_seed,
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        metrics=dict(payload.get("metrics") or {}),
        kind=str(payload.get("checkpoint_kind", "")),
        cache_payload_sha256s=cache_payload_sha256s,
    )
    for key, value in expected.items():
        if key == "metrics":
            continue
        if payload.get(key) != value:
            raise ValueError(f"V27a checkpoint metadata differs at {key}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27a checkpoint lacks model_state")
    if payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27a checkpoint model-state hash differs")
    if payload.get("model_contract") != model.model_contract():
        raise ValueError("V27a checkpoint model contract differs")
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        saved = payload.get("optimizer_state")
        if not isinstance(saved, Mapping):
            raise ValueError("V27a checkpoint lacks requested optimizer state")
        optimizer.load_state_dict(saved)
    return payload


__all__ = [
    "load_v27a_checkpoint",
    "save_v27a_checkpoint",
    "tensor_state_hash",
]
