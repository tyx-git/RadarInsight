"""Scan V25a checkpoints with V25f causal-prefix utility selectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from build.known_gate_v25f import validate_v25f_checkpoint_file
from radar_ln3.known_gate import known_gate_config_identity_sha256
from radar_ln3.known_gate import sha256_file
























def _verify_epoch_payload(
    path: Path,
    *,
    epoch: int,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    expected_order_sha256: str,
) -> Mapping[str, Any]:
    validate_v25f_checkpoint_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V25f epoch checkpoint payload must be a mapping: {path}")
    metrics = dict(payload.get("metrics") or {})
    if (
        payload.get("checkpoint_schema") != config.train.checkpoint_schema
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != config.variant
        or int(payload.get("epoch", -1)) != epoch
        or payload.get("config_identity_sha256")
        != known_gate_config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
        or int(metrics.get("epoch", -1)) != epoch
        or metrics.get("parent_order_sha256") != expected_order_sha256
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError(f"V25f epoch checkpoint is stale or invalid: {path}")
    return payload







