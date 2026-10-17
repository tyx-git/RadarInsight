"""Training and threshold selection for V27d-r6 Bird verifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256

from .model import BirdVerifier, build_verifier
from .protocol import R6_CHECKPOINT_SCHEMA, artifact_path, load_context






def _inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {name: tensors[name][indices].to(device) for name in ("summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "padding_mask")}


















def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdVerifier, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != R6_CHECKPOINT_SCHEMA or payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r6 checkpoint provenance/schema differs")
    model = build_verifier(config, arm=arm, seed=seed)
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r6 checkpoint contract differs")
    model.load_state_dict(payload["head_state"], strict=True)
    return model, dict(payload), config


__all__ = ["load_checkpoint"]