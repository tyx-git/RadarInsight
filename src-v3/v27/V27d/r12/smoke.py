"""Contract smoke checks for V27d-r12."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .prefix_aggregate import aggregate_prefix_quantile
from .protocol import (
    R12_ROLES,
    R12_SMOKE_SCHEMA_VERSION,
    artifact_path,
    load_context,
    write_sealed_json_if_identical,
)
from .evaluation import _load_caches, _load_reference, _model_scores


def run_smoke(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = _load_caches(root, config, manifest)
    reference, reference_checkpoint, _ = _load_reference(root, config, seed=42)
    reference.eval()
    development = caches["representation_development"]
    sequence, final = _model_scores(reference, development)
    aggregated = aggregate_prefix_quantile(
        sequence,
        final,
        development["tensors"]["prefix_route_mask"].bool(),
        development["tensors"]["candidate_population"].bool(),
        development["tensors"]["padding_mask"].bool(),
        quantile=float(config["aggregation"]["quantile"]),
    )
    if not torch.isfinite(aggregated).all():
        raise FloatingPointError("V27d-r12 smoke aggregation is non-finite")
    if not bool((aggregated[development["tensors"]["candidate_population"].bool()] <= 1.0).all()):
        raise ValueError("V27d-r12 smoke scores are out of probability bounds")

    score_sequence = torch.tensor([[0.10, 0.90, 0.80, 0.0]], dtype=torch.float32)
    final_scores = torch.tensor([0.80], dtype=torch.float32)
    prefix_mask = torch.tensor([[True, True, True, False]])
    candidate = torch.tensor([True])
    padding = torch.tensor([[False, False, False, True]])
    expected = torch.quantile(score_sequence[0, :3], 0.25)
    actual = aggregate_prefix_quantile(
        score_sequence,
        final_scores,
        prefix_mask,
        candidate,
        padding,
        quantile=0.25,
    )[0]
    if not torch.equal(actual, expected):
        raise ValueError("V27d-r12 smoke quantile calculation differs")
    if bool(actual >= final_scores[0]):
        raise ValueError("V27d-r12 smoke lower-quartile guard is ineffective")
    no_prefix_rejected = False
    try:
        aggregate_prefix_quantile(
            torch.tensor([[0.4, 0.0]], dtype=torch.float32),
            torch.tensor([0.4], dtype=torch.float32),
            torch.tensor([[False, False]]),
            torch.tensor([True]),
            torch.tensor([[False, False]]),
            quantile=0.25,
        )
    except ValueError:
        no_prefix_rejected = True
    if not no_prefix_rejected:
        raise ValueError("V27d-r12 smoke did not reject a candidate without an eligible prefix")
    if int(reference_checkpoint["selected_epoch"]) <= 0:
        raise ValueError("V27d-r12 reference checkpoint epoch is invalid")

    result = {
        "schema_version": R12_SMOKE_SCHEMA_VERSION,
        "experiment_id": "V27d-r12-causal-prefix-p25",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "aggregation": {
            "method": config["aggregation"]["method"],
            "quantile": float(config["aggregation"]["quantile"]),
            "padding_excluded": True,
            "candidate_without_eligible_prefix_rejected": no_prefix_rejected,
        },
        "roles": {
            role: {
                "parent_count": len(payload["parent_ids"]),
                "candidate": int(payload["tensors"]["candidate_population"].sum()),
                "bird_positive": int(payload["tensors"]["bird_positive"].sum()),
                "negative": int(payload["tensors"]["route_false_positive_negative"].sum()),
            }
            for role, payload in caches.items()
        },
        "synthetic_quantile_check": True,
        "formal_l3_enabled": False,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
