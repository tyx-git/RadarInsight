"""Contract checks for V27d-r5-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from v27.V27d.r5.model import build_head

from .audit import audit_error_samples
from .features import load_population_cache
from .protocol import V2_ROLES, artifact_path, load_context, write_sealed_json
from .training import _cache_path, model_contract


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {role: load_population_cache(_cache_path(root, config, role)) for role in V2_ROLES}
    for role, payload in caches.items():
        if payload["v2_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r5-v2 cache provenance differs: {role}")
        tensors = payload["tensors"]
        if bool(tensors["stable8_causal_features"].ne(0).any()):
            raise ValueError(f"V27d-r5-v2 control stable8 input is non-zero: {role}")
        if bool((tensors["event_trajectory"].masked_select(tensors["padding_mask"].unsqueeze(-1))).ne(0).any()):
            raise ValueError(f"V27d-r5-v2 padded trajectory is non-zero: {role}")
        if not torch.equal(tensors["endpoint_final_logits"], tensors["endpoint_sequence_logits"][torch.arange(len(payload["parent_ids"])), tensors["lengths"] - 1]):
            raise ValueError(f"V27d-r5-v2 endpoint final logits differ: {role}")
        if bool((tensors["bird_route_positive"] & tensors["bird_route_negative"]).any()):
            raise ValueError(f"V27d-r5-v2 population overlap: {role}")
    model = build_head(stable8_enabled=False, seed=42)
    contract = model_contract(model)
    if contract.get("experiment_id") != "V27d-r5-bird-route-v2":
        raise ValueError("V27d-r5-v2 model contract identity differs")
    if contract.get("classification_logits_input_to_head") is not False or contract.get("softmax_input_to_head") is not False:
        raise ValueError("V27d-r5-v2 model exposes classifier scores")
    audit = audit_error_samples(root, config_path)
    aggregate = audit["aggregate"]
    if aggregate["negative_support"] != 12 or aggregate["accepted_unknown_count"] != 11 or aggregate["rejected_unknown_count"] != 1:
        raise ValueError("V27d-r5-v2 audit does not reproduce the frozen r5 11/1 split")
    result = {
        "schema_version": "RADAR-LN3-V27d-r5-bird-route-v2-contract-smoke-v1",
        "experiment_id": "V27d-r5-bird-route-v2",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "roles": {
            role: {
                "parent_count": len(payload["parent_ids"]),
                "positive_support": int(payload["tensors"]["bird_route_positive"].sum()),
                "negative_support": int(payload["tensors"]["bird_route_negative"].sum()),
            }
            for role, payload in caches.items()
        },
        "audit_reproduces_r5_control_split": True,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
