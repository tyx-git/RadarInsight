"""Training and checkpoint contracts for the Bird-route head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256


from .model import BirdRouteKnownnessHead, build_head
from .protocol import artifact_path, load_context


CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r5-bird-head-checkpoint-v1"






def _head_inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }












def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdRouteKnownnessHead, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, _, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("V27d-r5 checkpoint schema differs")
    if payload.get("manifest_sha256") != manifest["manifest_sha256"] or payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r5 checkpoint provenance differs")
    model = build_head(stable8_enabled=arm == "stable8_treatment", seed=seed)
    if payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("V27d-r5 checkpoint model contract differs")
    state = payload.get("head_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r5 checkpoint state is missing")
    model.load_state_dict(state, strict=True)
    return model, dict(payload), config


__all__ = ["CHECKPOINT_SCHEMA", "load_checkpoint"]