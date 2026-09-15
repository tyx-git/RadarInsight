"""Contract smoke checks for V27d-r7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from radar_ln3.runtime import canonical_sha256
from v27.V27d.r6.training import _inputs as r6_inputs, load_checkpoint as load_r6_checkpoint

from .features import load_population_cache
from .model import build_verifier, initialize_from_r6
from .protocol import R7_ARMS, R7_ROLES, artifact_path, load_context, write_sealed_json, source_path
from .training import _inputs


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {
        role: load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True))
        for role in R7_ROLES
    }
    for role, payload in caches.items():
        tensors = payload["tensors"]
        if payload["r7_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r7 cache provenance differs: {role}")
        if bool(tensors["dispersion_features"].masked_select(tensors["padding_mask"].unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d-r7 padded dispersion is non-zero: {role}")
        if bool(tensors["dispersion_valid_fraction"].masked_select(tensors["padding_mask"]).ne(0).any()):
            raise ValueError(f"V27d-r7 padded dispersion quality is non-zero: {role}")
    r6_model, r6_checkpoint, _ = load_r6_checkpoint(
        root,
        arm="liquid_treatment",
        seed=42,
        config_path=source_path(root, config["source"]["r6_config_path"], must_exist=True),
    )
    zero = build_verifier(config, arm="zero_control", seed=42)
    treatment = build_verifier(config, arm="stable8_treatment", seed=42)
    initialize_from_r6(zero, r6_checkpoint["head_state"])
    initialize_from_r6(treatment, r6_checkpoint["head_state"])
    sample = caches["representation_development"]
    indices = torch.arange(min(4, len(sample["parent_ids"])))
    with torch.inference_mode():
        r6_output = r6_model(**r6_inputs(sample, indices, torch.device("cpu")))
        zero_output = zero(**_inputs(sample, indices, torch.device("cpu"), use_branch=False))
        treatment_output = treatment(**_inputs(sample, indices, torch.device("cpu"), use_branch=True))
    if not torch.equal(r6_output["verifier_logit_sequence"], zero_output["verifier_logit_sequence"]):
        raise ValueError("V27d-r7 zero control does not replay r6 Liquid logits bitwise")
    if not torch.equal(zero_output["verifier_logit_sequence"], treatment_output["verifier_logit_sequence"]):
        raise ValueError("V27d-r7 zero and treatment should initially match before training")
    if not torch.isfinite(treatment_output["verifier_logit_sequence"]).all() or treatment_output["liquid_trace"] is None:
        raise ValueError("V27d-r7 treatment smoke output is invalid")
    result = {
        "schema_version": "RADAR-LN3-V27d-r7-contract-smoke-v1",
        "experiment_id": "V27d-r7-causal-stable8-dispersion",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "r6_initial_checkpoint_sha256": r6_checkpoint.get("head_state_sha256"),
        "zero_replays_r6_bitwise": True,
        "zero_equals_treatment_before_training": True,
        "roles": {role: {"parent_count": len(payload["parent_ids"]), "positive": int(payload["tensors"]["bird_positive"].sum()), "negative": int(payload["tensors"]["route_false_positive_negative"].sum())} for role, payload in caches.items()},
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
