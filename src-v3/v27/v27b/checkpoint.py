"""Fail-closed checkpoint serialization for V27b treatment heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import (
    canonical_sha256,
    resolve_project_path,
    write_torch_if_absent_or_identical,
)

from .cache import cache_payload_sha256, load_feature_cache
from .model import V27BConditionalPairResidual
from .protocol import (
    V27B_CHECKPOINT_SCHEMA_VERSION,
    V27BContext,
    V27B_L2_CLASS_NAMES,
    tensor_state_hash,
)
from .runtime import feature_cache_path


def _cache_hashes(context: V27BContext) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val"):
        payload = load_feature_cache(
            feature_cache_path(context, split), context=context, split=split
        )
        declared = payload.get("cache_payload_sha256")
        actual = cache_payload_sha256(payload)
        if not isinstance(declared, str) or declared != actual:
            raise ValueError(f"V27b {split} feature cache hash cannot be recomputed")
        result[split] = actual.lower()
    return result


def _metadata(
    *,
    context: V27BContext,
    model: V27BConditionalPairResidual,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    kind: str,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if kind not in {"epoch", "smoke"}:
        raise ValueError("V27b checkpoint kind is invalid")
    if isinstance(seed, bool) or int(seed) not in context.config.training.seeds:
        raise ValueError("V27b checkpoint seed is not registered")
    if isinstance(epoch, bool) or int(epoch) <= 0:
        raise ValueError("V27b checkpoint epoch is invalid")
    if isinstance(global_step, bool) or int(global_step) < 0:
        raise ValueError("V27b checkpoint global_step is invalid")
    if set(cache_hashes) != {"train", "val"}:
        raise ValueError("V27b checkpoint cache hash keys differ")
    model_contract = model.model_contract()
    return {
        "checkpoint_schema": V27B_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": "V27b",
        "checkpoint_kind": str(kind),
        "arm": "conditional",
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
        "input_binding_sha256": canonical_sha256(context.input_binding),
        "source_checkpoint_file_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "source_checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "feature_cache_payload_sha256s": {
            "train": str(cache_hashes["train"]).lower(),
            "val": str(cache_hashes["val"]).lower(),
        },
        "loss_name": context.config.training.loss_name,
        "loss_reduction": context.config.training.loss_reduction,
        "class_names": list(V27B_L2_CLASS_NAMES),
        "class_weights": list(context.config.base.expected_l2_class_weights),
        "model_contract": model_contract,
        "model_contract_sha256": canonical_sha256(model_contract),
        "metrics": dict(metrics),
    }


def save_v27b_checkpoint(
    path: str | Path,
    *,
    context: V27BContext,
    model: V27BConditionalPairResidual,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    kind: str,
    cache_hashes: Mapping[str, str],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Write one immutable treatment checkpoint bound to the current manifest."""

    target = resolve_project_path(context.project_root, path)
    metadata = _metadata(
        context=context,
        model=model,
        seed=seed,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        kind=kind,
        cache_hashes=cache_hashes,
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
    write_torch_if_absent_or_identical(target, payload)
    return target


def load_v27b_checkpoint(
    path: str | Path,
    *,
    context: V27BContext,
    model: V27BConditionalPairResidual,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a V27b checkpoint only when every binding and tensor agrees."""

    target = resolve_project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27b checkpoint must be a mapping")
    payload = dict(payload)
    if payload.get("checkpoint_schema") != V27B_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V27b checkpoint schema differs")
    if payload.get("experiment_id") != "V27b" or payload.get("arm") != "conditional":
        raise ValueError("V27b checkpoint experiment/arm differs")
    actual_seed = payload.get("seed")
    if (
        isinstance(actual_seed, bool)
        or not isinstance(actual_seed, int)
        or actual_seed not in context.config.training.seeds
        or (seed is not None and int(seed) != actual_seed)
    ):
        raise ValueError("V27b checkpoint seed differs")
    if int(payload.get("epoch", 0)) <= 0 or int(payload.get("global_step", -1)) < 0:
        raise ValueError("V27b checkpoint epoch/global_step differs")
    cache_hashes = _cache_hashes(context)
    expected = _metadata(
        context=context,
        model=model,
        seed=actual_seed,
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        metrics=dict(payload.get("metrics") or {}),
        kind=str(payload.get("checkpoint_kind", "")),
        cache_hashes=cache_hashes,
    )
    for key, value in expected.items():
        if key == "metrics":
            continue
        if payload.get(key) != value:
            raise ValueError(f"V27b checkpoint metadata differs at {key}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27b checkpoint lacks model_state")
    if payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27b checkpoint model-state hash differs")
    if payload.get("model_contract") != model.model_contract():
        raise ValueError("V27b checkpoint model contract differs")
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        saved = payload.get("optimizer_state")
        if not isinstance(saved, Mapping):
            raise ValueError("V27b checkpoint lacks requested optimizer state")
        optimizer.load_state_dict(saved)
    return payload


__all__ = ["load_v27b_checkpoint", "save_v27b_checkpoint"]
