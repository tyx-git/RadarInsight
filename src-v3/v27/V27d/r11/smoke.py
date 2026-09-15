"""Contract smoke checks for V27d-r11."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import load_population_cache, validate_cache_binding
from .model import build_model, initialize_from_r6
from .protocol import (
    R11_ROLES,
    R11_SMOKE_SCHEMA_VERSION,
    artifact_path,
    load_context,
    write_sealed_json_if_identical,
)
from .training import (
    _freeze_except_shape_projection,
    _inputs,
    _load_r6_reference,
    _r6_inputs,
)


def run_smoke(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {
        role: load_population_cache(
            artifact_path(
                root,
                config["artifacts"]["cache_template"].format(role=role),
                must_exist=True,
            )
        )
        for role in R11_ROLES
    }
    for role, payload in caches.items():
        validate_cache_binding(payload, root=root, config=config, role=role)
        if payload.get("r11_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r11 cache manifest differs: {role}")
        tensors = payload["tensors"]
        padding = tensors["padding_mask"]
        if bool(tensors["shape_features"].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d-r11 padded shape feature is non-zero: {role}")
        if bool(tensors["shape_event_valid"].masked_select(padding).any()):
            raise ValueError(f"V27d-r11 padded shape validity is active: {role}")
        if bool(tensors["shape_event_frame_count"].masked_select(padding).ne(0).any()):
            raise ValueError(f"V27d-r11 padded shape frame count is non-zero: {role}")

    reference, reference_payload, _ = _load_r6_reference(root, config, seed=42)
    reference.eval()
    model = build_model(config, arm="shape_treatment", seed=42)
    initialize_from_r6(model, reference_payload["head_state"])
    trainable_names = _freeze_except_shape_projection(model)
    if tuple(trainable_names) != ("shape_projection.weight",):
        raise ValueError("V27d-r11 smoke trainable component differs")
    if not bool(torch.equal(model.shape_projection.weight, torch.zeros_like(model.shape_projection.weight))):
        raise ValueError("V27d-r11 smoke shape projection is not zero initialized")

    sample = caches["representation_development"]
    indices = torch.arange(min(4, len(sample["parent_ids"])), dtype=torch.long)
    with torch.inference_mode():
        treatment_output = model(**_inputs(sample, indices, torch.device("cpu")))
        reference_output = reference(**_r6_inputs(sample, indices, torch.device("cpu")))
    if not torch.equal(
        treatment_output["verifier_logit_sequence"],
        reference_output["verifier_logit_sequence"],
    ):
        raise ValueError("V27d-r11 zero-initialized output differs from r6 reference")
    if bool(treatment_output["liquid_trace"] is None):
        raise ValueError("V27d-r11 Liquid V2 trace is missing")

    result = {
        "schema_version": R11_SMOKE_SCHEMA_VERSION,
        "experiment_id": "V27d-r11",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "roles": {
            role: {
                "parent_count": len(payload["parent_ids"]),
                "candidate": int(payload["tensors"]["candidate_population"].sum()),
                "bird_positive": int(payload["tensors"]["bird_positive"].sum()),
                "route_false_positive_negative": int(payload["tensors"]["route_false_positive_negative"].sum()),
                "shape_valid_events": int(payload["tensors"]["shape_event_valid"].sum()),
            }
            for role, payload in caches.items()
        },
        "r6_zero_initialization_output_exact_match": True,
        "trainable_parameters": list(trainable_names),
        "frozen_liquid_v2_core": True,
        "frozen_base_readout": True,
        "frozen_verifier_head": True,
        "formal_l3_enabled": False,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
