"""Contract smoke checks for V27d-r8."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from radar_ln3.runtime import atomic_write_text, json_text
from v27.V27d.r6.training import _inputs as r6_inputs, load_checkpoint as load_r6_checkpoint

from .features import load_population_cache
from .model import R8_ARM_FLAGS, build_verifier, initialize_from_r6
from .protocol import R8_ARMS, R8_ROLES, artifact_path, load_context, read_sealed_json, write_sealed_json, source_path
from .training import _inputs


def _expect_failure(check: Callable[[], Any], name: str) -> None:
    try:
        check()
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError, KeyError, IndexError):
        return
    raise AssertionError(f"V27d-r8 smoke expected failure did not occur: {name}")


def run_smoke(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    caches = {
        role: load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True))
        for role in R8_ROLES
    }
    for role, payload in caches.items():
        if payload["r8_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r8 cache provenance differs: {role}")
        tensors = payload["tensors"]
        if bool(tensors["course_features"].masked_select(tensors["padding_mask"].unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d-r8 padded course is non-zero: {role}")
        if bool(tensors["dispersion_features"].masked_select(tensors["padding_mask"].unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d-r8 padded dispersion is non-zero: {role}")
        if bool(tensors["course_valid"].masked_select(tensors["padding_mask"]).any()):
            raise ValueError(f"V27d-r8 padded course valid mask is non-zero: {role}")

    r6_config = source_path(root, config["source"]["r6_config_path"], must_exist=True)
    r6_model, r6_checkpoint, _ = load_r6_checkpoint(root, arm="liquid_treatment", seed=42, config_path=r6_config)
    models = {}
    for arm in R8_ARMS:
        model = build_verifier(config, arm=arm, seed=42)
        initialize_from_r6(model, r6_checkpoint["head_state"])
        models[arm] = model
    sample = caches["representation_development"]
    indices = torch.arange(min(4, len(sample["parent_ids"])))
    with torch.inference_mode():
        r6_output = r6_model(**r6_inputs(sample, indices, torch.device("cpu")))
        outputs = {
            arm: model(**_inputs(sample, indices, torch.device("cpu"), use_course=flags[0], use_stable8=flags[1]))
            for arm, (flags, model) in ((arm, (R8_ARM_FLAGS[arm], models[arm])) for arm in R8_ARMS)
        }
    for arm, output in outputs.items():
        if not torch.equal(r6_output["verifier_logit_sequence"], output["verifier_logit_sequence"]):
            raise ValueError(f"V27d-r8 {arm} does not replay r6 Liquid logits bitwise")
    for arm in R8_ARMS:
        if not torch.isfinite(outputs[arm]["verifier_logit_sequence"]).all() or outputs[arm]["liquid_trace"] is None:
            raise ValueError(f"V27d-r8 {arm} output is invalid")

    with tempfile.TemporaryDirectory(dir=str(root / "src-v3" / "v27" / "V27d" / "r8" / "reports")) as folder:
        path = Path(folder) / "sealed.json"
        write_sealed_json(path, {"schema_version": "smoke", "value": 1})
        if read_sealed_json(path, expected_schema="smoke")["value"] != 1:
            raise AssertionError("V27d-r8 sealed JSON round trip differs")
        atomic_write_text(path, json_text({"schema_version": "smoke", "value": 2, "payload_sha256": "0" * 64}))
        _expect_failure(lambda: read_sealed_json(path), "sealed hash tamper")
    _expect_failure(lambda: artifact_path(root, "src-v3/v27/V27d/r8/cache/../../../../outside.json"), "artifact path escape")
    result = {
        "schema_version": "RADAR-LN3-V27d-r8-contract-smoke-v1",
        "experiment_id": "V27d-r8-joint-course-stable8",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "arms": list(R8_ARMS),
        "arm_flags": {arm: {"course": flags[0], "stable8": flags[1]} for arm, flags in R8_ARM_FLAGS.items()},
        "zero_replays_r6_bitwise": True,
        "course_and_dispersion_padding_zero": True,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["smoke_path"])
    write_sealed_json(path, result)
    return {**result, "report_path": str(path.relative_to(root)).replace("\\", "/")}


__all__ = ["run_smoke"]
