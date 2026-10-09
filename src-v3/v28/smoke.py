"""Small V28 contract smoke suite without formal training or test access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper
from radar_ln3.runtime import json_text

from .heads import V28TrajectoryHeads
from .model import V28TaskModel
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, artifact_path, load_config, prepare_manifest, read_sealed_json


def _liquid_equivalence() -> dict[str, Any]:
    torch.manual_seed(28)
    cell = LiquidV2Cell(4, 6, tau_min_seconds=0.05, tau_max_seconds=30.0, zero_dt_event_floor_seconds=0.001, dt_max_seconds=120.0, invalid_dt_update_mode="density_weighted", invalid_dt_observation_gain=0.25)
    wrapper = LiquidV2SequenceWrapper(cell).eval()
    events = torch.randn(1, 5, 4)
    dt = torch.tensor([[0.0, 0.0, 0.25, 0.4, 0.2]])
    valid = torch.tensor([[False, False, True, True, True]])
    reason = torch.tensor([[1, 2, 3, 3, 3]], dtype=torch.long)
    padding = torch.zeros((1, 5), dtype=torch.bool)
    offline = wrapper(events, dt, padding, dt_valid=valid, dt_reason=reason, return_trace=True)
    state = None
    online_states = []
    for index in range(5):
        ordinal = offline.trace["zero_ordinal"][:, index]
        output = wrapper.step(events[:, index], dt[:, index], dt_valid=valid[:, index], dt_reason=reason[:, index], state=state, zero_ordinal=ordinal)
        state = output.next_state
        online_states.append(state)
    online = torch.stack(online_states, dim=1)
    if not torch.allclose(offline.sequence, online, atol=1.0e-6, rtol=0.0):
        raise AssertionError("V28 Liquid V2 offline/online mismatch")
    chunk1 = wrapper(events[:, :3], dt[:, :3], padding[:, :3], dt_valid=valid[:, :3], dt_reason=reason[:, :3], return_trace=True)
    chunk2 = wrapper(events[:, 3:], dt[:, 3:], padding[:, 3:], dt_valid=valid[:, 3:], dt_reason=reason[:, 3:], initial_state=chunk1.final_state, zero_ordinal=offline.trace["zero_ordinal"][:, 3:], return_trace=True)
    if not torch.allclose(torch.cat((chunk1.sequence, chunk2.sequence), dim=1), offline.sequence, atol=1.0e-6, rtol=0.0):
        raise AssertionError("V28 Liquid V2 chunk/offline mismatch")
    return {"status": "passed", "hidden_dim": 6, "sequence_length": 5}


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = prepare_manifest(root, config_path)
    config = load_config(root, config_path, allow_missing_derived=True)
    manifest_path = root / config.artifacts["manifest_path"]
    context = V28Context(root, config, read_sealed_json(manifest_path), manifest_path, manifest["sha256"])
    l1 = V28TaskModel(context, task=V28_L1_TASK)
    l2 = V28TaskModel(context, task=V28_L2_TASK)
    if set(l1.state_dict()).intersection(l2.state_dict()):
        # Parameter names can coincide across independent models; their object
        # identity must still differ.
        if any(left is right for left in l1.parameters() for right in l2.parameters()):
            raise AssertionError("V28 L1/L2 share recurrent parameters")
    head = V28TrajectoryHeads(context, task=V28_L2_TASK)
    if set(head.model_contract()["forbidden_inputs"]) != {"l1_logits", "l2_logits", "softmax", "label_identity"}:
        raise AssertionError("V28 head forbidden-input contract differs")
    result = {"schema_version": "RADAR-LN3-V28-contract-smoke-v2", "status": "passed", "manifest_path": manifest["path"], "manifest_sha256": manifest["sha256"], "liquid_v2": _liquid_equivalence(), "l1_l2_recurrent_state_shared": False, "unknown_classifier_training_exposure": 0, "test_payload_access_count": 0}
    path = artifact_path(root, str(config.artifacts["smoke_report_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(result), encoding="utf-8")
    return result


__all__ = ["run_smoke"]
