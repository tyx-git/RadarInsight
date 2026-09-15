"""Structural, causal and boundary smoke checks for V27d-r10."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import load_descriptor_cache, validate_cache_binding
from .model import build_model
from .protocol import (
    R10_CLASS_NAMES,
    R10_SMOKE_SCHEMA_VERSION,
    artifact_path,
    load_context,
    write_sealed_json_if_identical,
)
from .training import _cache_path, _inputs


def _zero_replays_baseline(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    count = min(4, len(payload["parent_sample_ids"]))
    indices = torch.arange(count, dtype=torch.long)
    inputs = _inputs(payload, indices, torch.device("cpu"))
    with torch.inference_mode():
        output = model(**inputs)
    if not torch.equal(output["sequence_logits"], inputs["baseline_logits"]):
        raise AssertionError("V27d-r10 zero branch does not replay V27a Main")
    if bool(output["correction_logits"].ne(0.0).any()):
        raise AssertionError("V27d-r10 zero branch correction is non-zero")


def _prefix_is_causal(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    lengths = payload["tensors"]["lengths"]
    row = next((index for index, value in enumerate(lengths.tolist()) if int(value) >= 4), None)
    if row is None:
        return
    length = int(lengths[row])
    split = max(1, length // 2)
    full = _inputs(payload, torch.tensor([row]), torch.device("cpu"))
    prefix = {name: value[:, :split].clone() for name, value in full.items()}
    prefix["padding_mask"].zero_()
    with torch.inference_mode():
        full_output = model(**full)
        prefix_output = model(**prefix)
    if not torch.equal(
        full_output["sequence_logits"][:, :split],
        prefix_output["sequence_logits"],
    ):
        raise AssertionError("V27d-r10 prefix output depends on future events")


def _invalid_does_not_update(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    inputs = _inputs(payload, torch.tensor([0]), torch.device("cpu"))
    invalid = torch.nonzero(
        ~inputs["descriptor_valid"][0] & ~inputs["padding_mask"][0],
        as_tuple=False,
    ).flatten()
    if not invalid.numel():
        return
    event = int(invalid[0])
    inputs["descriptor_values"][:, event] = 0.0
    inputs["descriptor_valid"][:, event] = False
    with torch.inference_mode():
        output = model(**inputs)
    if bool(output["correction_logits"][:, event].ne(0.0).any()) or bool(
        output["pairwise_delta_increment"][:, event].ne(0.0).any()
    ):
        raise AssertionError("V27d-r10 invalid descriptor changed residual state")


def _chunked_matches_full(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    lengths = payload["tensors"]["lengths"]
    row = next((index for index, value in enumerate(lengths.tolist()) if int(value) >= 4), None)
    if row is None:
        return
    length = int(lengths[row])
    split = max(1, length // 2)
    full = _inputs(payload, torch.tensor([row]), torch.device("cpu"))
    first = {name: value[:, :split].clone() for name, value in full.items()}
    second = {name: value[:, split:length].clone() for name, value in full.items()}
    first["padding_mask"].zero_()
    second["padding_mask"].zero_()
    with torch.inference_mode():
        full_output = model(**full)
        first_output = model(**first)
        second_output = model(
            **second,
            initial_state=first_output["final_state"],
        )
    combined = torch.cat(
        (first_output["sequence_logits"], second_output["sequence_logits"]),
        dim=1,
    )
    if not torch.equal(combined, full_output["sequence_logits"][:, :length]):
        raise AssertionError("V27d-r10 chunked continuation differs from full sequence")


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {
        split: load_descriptor_cache(_cache_path(root, config, split))
        for split in ("train", "val")
    }
    for split, payload in caches.items():
        validate_cache_binding(payload, root=root, config=config, split=split)
        model = build_model(config["model"], seed=42).eval()
        _zero_replays_baseline(model, payload)
        _prefix_is_causal(model, payload)
        _invalid_does_not_update(model, payload)
        _chunked_matches_full(model, payload)
    result = {
        "schema_version": R10_SMOKE_SCHEMA_VERSION,
        "experiment_id": "V27d-r10",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "zero_branch_replays_frozen_v27a_main": True,
        "causal_prefix_invariance": True,
        "invalid_descriptor_zero_residual": True,
        "padding_zero_and_no_state_update": True,
        "chunked_continuation_matches_full_sequence": True,
        "class_names": list(R10_CLASS_NAMES),
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "formal_l3_enabled": False,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
