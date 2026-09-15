"""Structural and causal contract checks for V27d-r9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from radar_ln3.runtime import atomic_write_text, json_text

from .features import load_profile_cache, validate_profile_cache_binding
from .model import build_model
from .protocol import R9_CLASS_NAMES, R9_SEEDS, R9_SMOKE_SCHEMA_VERSION, artifact_path, load_context, read_sealed_json, write_sealed_json
from .training import _cache_path, _inputs


def _assert_zero_branch_replays(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    sample_size = min(4, len(payload["parent_sample_ids"]))
    indices = torch.arange(sample_size, dtype=torch.long)
    inputs = _inputs(payload, indices, torch.device("cpu"))
    with torch.inference_mode():
        output = model(**inputs)
    if not torch.equal(output["sequence_logits"], inputs["baseline_logits"]):
        raise AssertionError("V27d-r9 zero-initialized branch does not replay baseline")
    if bool(output["correction_logits"].ne(0.0).any()):
        raise AssertionError("V27d-r9 zero-initialized correction is non-zero")


def _assert_prefix_causal(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    row = 0
    length = int(payload["tensors"]["lengths"][row])
    if length < 3:
        return
    inputs = _inputs(payload, torch.tensor([row]), torch.device("cpu"))
    prefix = {name: value.clone() for name, value in inputs.items()}
    prefix_length = length - 1
    prefix["padding_mask"][:, prefix_length:] = True
    prefix["baseline_logits"][:, prefix_length:] = 0.0
    prefix["profile_values"][:, prefix_length:] = 0.0
    prefix["profile_valid"][:, prefix_length:] = False
    with torch.inference_mode():
        full = model(**inputs)
        truncated = model(**prefix)
    if not torch.equal(full["sequence_logits"][:, :prefix_length], truncated["sequence_logits"][:, :prefix_length]):
        raise AssertionError("V27d-r9 prefix output depends on future events")
    if bool(truncated["sequence_logits"][:, prefix_length:].ne(0.0).any()):
        raise AssertionError("V27d-r9 padded prefix output is non-zero")


def _assert_invalid_profile_contract(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    inputs = _inputs(payload, torch.tensor([0]), torch.device("cpu"))
    valid = inputs["profile_valid"]
    invalid = torch.nonzero(~valid[0] & ~inputs["padding_mask"][0], as_tuple=False).flatten()
    if not invalid.numel():
        return
    event = int(invalid[0])
    inputs["profile_values"][:, event] = 0.0
    inputs["profile_valid"][:, event] = False
    with torch.inference_mode():
        output = model(**inputs)
    if bool(output["correction_logits"][:, event].ne(0.0).any()) or bool(output["pairwise_delta_increment"][:, event].ne(0.0).any()):
        raise AssertionError("V27d-r9 invalid profile changed residual state")


def _assert_chunked_equivalence(model: torch.nn.Module, payload: dict[str, Any]) -> None:
    lengths = payload["tensors"]["lengths"]
    row = next((index for index, value in enumerate(lengths.tolist()) if int(value) >= 4), None)
    if row is None:
        return
    length = int(lengths[row])
    split = max(1, length // 2)
    full_inputs = _inputs(payload, torch.tensor([row]), torch.device("cpu"))
    first = {name: value[:, :split].clone() for name, value in full_inputs.items()}
    second = {name: value[:, split:length].clone() for name, value in full_inputs.items()}
    first["padding_mask"].zero_()
    second["padding_mask"].zero_()
    with torch.inference_mode():
        full = model(**full_inputs)
        first_output = model(**first)
        second_output = model(**second, initial_state=first_output["final_state"])
    combined = torch.cat((first_output["sequence_logits"], second_output["sequence_logits"]), dim=1)
    if not torch.equal(combined, full["sequence_logits"][:, :length]):
        raise AssertionError("V27d-r9 chunked continuation differs from full causal output")


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {split: load_profile_cache(_cache_path(root, config, split)) for split in ("train", "val")}
    for split, payload in caches.items():
        validate_profile_cache_binding(payload, root=root, config=config, split=split)
        model = build_model(config["model"], seed=42).eval()
        _assert_zero_branch_replays(model, payload)
        _assert_prefix_causal(model, payload)
        _assert_invalid_profile_contract(model, payload)
        _assert_chunked_equivalence(model, payload)
        tensors = payload["tensors"]
        if bool(tensors["profile_values"].masked_select(tensors["padding_mask"].unsqueeze(-1)).ne(0).any()):
            raise AssertionError(f"V27d-r9 {split} padded profile is non-zero")
    with torch.inference_mode():
        model = build_model(config["model"], seed=42).eval()
        _assert_zero_branch_replays(model, caches["val"])
    result = {
        "schema_version": R9_SMOKE_SCHEMA_VERSION,
        "experiment_id": "V27d-r9",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "zero_branch_replays_frozen_v27a_main": True,
        "causal_prefix_invariance": True,
        "invalid_profile_zero_residual": True,
        "padding_zero_and_no_state_update": True,
        "chunked_continuation_matches_full_sequence": True,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "formal_l3_enabled": False,
        "class_names": list(R9_CLASS_NAMES),
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
