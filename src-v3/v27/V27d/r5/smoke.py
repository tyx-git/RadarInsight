"""Contract smoke checks for V27d-r5 Bird-route population and head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import load_population_cache
from .model import build_head
from .protocol import R5_ARMS, R5_ROLES, artifact_path, load_context, write_sealed_json
from .training import _cache_path


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, _, manifest = load_context(root, config_path)
    caches: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in R5_ARMS:
        for role in R5_ROLES:
            caches[(arm, role)] = load_population_cache(_cache_path(root, config, arm=arm, role=role))
    checks: dict[str, Any] = {}
    for role in R5_ROLES:
        control = caches[("control", role)]
        treatment = caches[("stable8_treatment", role)]
        if control["parent_ids"] != treatment["parent_ids"] or control["label_names"] != treatment["label_names"]:
            raise ValueError(f"V27d-r5 control/treatment parent contract differs: {role}")
        if not torch.equal(control["tensors"]["endpoint_final_logits"], treatment["tensors"]["endpoint_final_logits"]):
            raise ValueError(f"V27d-r5 endpoint logits changed: {role}")
        if bool(control["tensors"]["stable8_causal_features"].ne(0).any()):
            raise ValueError(f"V27d-r5 control stable8 is not zero: {role}")
        if not bool(treatment["tensors"]["stable8_causal_features"].ne(0).any()):
            raise ValueError(f"V27d-r5 treatment stable8 is empty: {role}")
        labels = treatment["label_names"]
        tensors = treatment["tensors"]
        positive = tensors["bird_route_positive"]
        negative = tensors["bird_route_negative"]
        if any(labels[row] != "BirdTarget" for row in torch.where(positive)[0].tolist()):
            raise ValueError(f"V27d-r5 positive label contract differs: {role}")
        if any(labels[row] != "UnknownTarget" for row in torch.where(negative)[0].tolist()):
            raise ValueError(f"V27d-r5 negative label contract differs: {role}")
        if bool((positive & negative).any()) or bool((negative & ~tensors["l1_target_route"]).any()) or bool((negative & ~tensors["l2_predicted_bird"]).any()):
            raise ValueError(f"V27d-r5 negative route contract differs: {role}")
        if bool((positive & ~tensors["quality_target_final"]).any()) or bool((negative & ~tensors["quality_target_final"]).any()):
            raise ValueError(f"V27d-r5 quality population contract differs: {role}")
        checks[role] = {
            "parent_count": len(labels),
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
            "endpoint_logits_bitwise_identical": True,
        }
    for arm in R5_ARMS:
        model = build_head(stable8_enabled=arm == "stable8_treatment", seed=42)
        if model.model_contract()["classification_logits_input_to_head"] is not False:
            raise ValueError("V27d-r5 head exposes classification logits")
    result = {
        "schema_version": "RADAR-LN3-V27d-r5-contract-smoke-v1",
        "experiment_id": "V27d-r5-bird-route",
        "status": "passed",
        "r5_manifest_sha256": manifest["manifest_sha256"],
        "checks": checks,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
