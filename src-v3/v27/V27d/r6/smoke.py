"""Contract smoke checks for V27d-r6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import load_population_cache
from .model import build_verifier
from .protocol import R6_ARMS, R6_ROLES, artifact_path, load_context, write_sealed_json
from .training import _inputs


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {role: load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)) for role in R6_ROLES}
    for role, payload in caches.items():
        tensors = payload["tensors"]
        if payload["r6_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r6 cache provenance differs: {role}")
        if bool((tensors["event_trajectory"].masked_select(tensors["padding_mask"].unsqueeze(-1))).ne(0).any()):
            raise ValueError(f"V27d-r6 padded trajectory is non-zero: {role}")
        if bool((tensors["event_facts_sequence"].masked_select(tensors["padding_mask"].unsqueeze(-1))).ne(0).any()):
            raise ValueError(f"V27d-r6 padded event facts are non-zero: {role}")
    gru = build_verifier(config, arm="gru_control", seed=42)
    liquid = build_verifier(config, arm="liquid_treatment", seed=42)
    if gru.recurrent_parameter_count != liquid.recurrent_parameter_count:
        raise ValueError("V27d-r6 recurrent parameter counts differ")
    if gru.model_contract()["classification_logits_input_to_head"] is not False or liquid.model_contract()["softmax_input_to_head"] is not False:
        raise ValueError("V27d-r6 verifier exposes forbidden classifier inputs")
    sample = caches["representation_development"]
    indices = torch.arange(min(4, len(sample["parent_ids"])))
    with torch.inference_mode():
        output = liquid(**{key: value[indices] for key, value in _inputs(sample, indices, torch.device("cpu")).items()})
    if output["liquid_trace"] is None or not torch.isfinite(output["verifier_logit_sequence"]).all():
        raise ValueError("V27d-r6 Liquid smoke output is invalid")
    result = {
        "schema_version": "RADAR-LN3-V27d-r6-contract-smoke-v1",
        "experiment_id": "V27d-r6-bird-liquid-verifier",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "recurrent_parameter_count_gru": gru.recurrent_parameter_count,
        "recurrent_parameter_count_liquid": liquid.recurrent_parameter_count,
        "roles": {role: {"parent_count": len(payload["parent_ids"]), "positive": int(payload["tensors"]["bird_positive"].sum()), "negative": int(payload["tensors"]["route_false_positive_negative"].sum())} for role, payload in caches.items()},
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
