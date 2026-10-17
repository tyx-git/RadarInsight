"""Frozen-backbone V28 knownness/quality training and safe selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256

from .protocol import V28_HEAD_SCHEMA


























def _load_head_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != V28_HEAD_SCHEMA:
        raise ValueError("V28 head checkpoint schema differs")
    value = dict(payload)
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("V28 head checkpoint state is missing")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("V28 head checkpoint state contains a non-tensor entry")
    state_hash = canonical_sha256({key: tensor.detach().cpu().numpy().tobytes().hex() for key, tensor in sorted(state.items())})
    if value.get("state_dict_sha256") != state_hash:
        raise ValueError("V28 head checkpoint state hash differs")
    if value.get("model_contract_sha256") != canonical_sha256(value.get("model_contract")):
        raise ValueError("V28 head checkpoint model contract hash differs")
    return value





