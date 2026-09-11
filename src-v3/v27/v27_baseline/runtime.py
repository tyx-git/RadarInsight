"""Shared execution helpers for the sealed V27 baseline.

This module is intentionally small and policy-free beyond the V27 protocol:
it is the single place where an entry point resolves the verified context,
chooses the configured device, seeds the three RNG families, and writes a
mutable JSON artifact.  It never opens test or Unknown payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.runtime import atomic_write_text, canonical_sha256, json_text

from .model import V27LiquidV2TaskModel
from .protocol import (
    V27Config,
    load_and_verify_v27_manifest,
    load_v27_config,
    verify_v27_data_contract,
)


@dataclass(frozen=True)
class V27Context:
    """All provenance objects an entry point may consume."""

    project_root: Path
    config: V27Config
    data_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def seed_v27(seed: int, *, deterministic_cuda: bool = False) -> None:
    """Seed Python, NumPy and Torch without changing the configured policy."""

    value = int(seed)
    if value < 0:
        raise ValueError("V27 seed must be non-negative")
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic_cuda:
        # The flag is explicit in the manifest.  Failures are preferable to a
        # silently nondeterministic baseline when a future operator enables it.
        torch.use_deterministic_algorithms(True, warn_only=False)


def configured_device(config: V27Config) -> torch.device:
    """Resolve the exact configured device and fail closed if unavailable."""

    requested = str(config.training.device).strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27 configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported V27 device: {requested}")


def load_v27_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27Context:
    """Load and re-verify the current V27 config, data registry and manifest."""

    root = Path(project_root).resolve()
    config = load_v27_config(root, config_path)
    binding = verify_v27_data_contract(root, config)
    manifest, manifest_path, manifest_sha256 = load_and_verify_v27_manifest(
        root, config, binding
    )
    return V27Context(
        project_root=root,
        config=config,
        data_binding=dict(binding),
        manifest=dict(manifest),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def build_v27_model(
    context: V27Context,
    *,
    task: str,
    device: torch.device | None = None,
) -> V27LiquidV2TaskModel:
    """Construct one task-local V27 model and move it to the verified device."""

    target = configured_device(context.config) if device is None else device
    model = V27LiquidV2TaskModel(context.config, task=task)
    model.to(target)
    return model


def write_v27_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace a mutable V27 summary/report artifact."""

    target = Path(path)
    atomic_write_text(target, json_text(payload, indent=2))
    return target


def write_v27_json_if_identical(path: str | Path, payload: Any) -> Path:
    """Write an immutable-style JSON artifact, accepting only identical data."""

    target = Path(path)
    text = json_text(payload, indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V27 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def seal_v27_json_payload(payload: Any) -> dict[str, Any]:
    """Add a recomputable integrity field to a V27 JSON artifact.

    The digest is over the payload *before* the integrity field is added.  A
    reader can therefore verify the artifact without trusting a separately
    declared file hash or relying on JSON formatting and key order.
    """

    if not isinstance(payload, dict):
        raise TypeError("V27 sealed JSON payload must be an object")
    if "payload_sha256" in payload:
        raise ValueError("V27 payload already contains payload_sha256")
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def verify_v27_json_payload(
    payload: Any,
    *,
    expected_schema: str | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Verify a sealed V27 JSON object and its optional identity fields."""

    if not isinstance(payload, dict):
        raise ValueError("V27 JSON artifact must contain an object")
    declared = payload.get("payload_sha256")
    if not isinstance(declared, str) or declared != canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    ):
        raise ValueError("V27 JSON artifact payload_sha256 cannot be recomputed")
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        raise ValueError("V27 JSON artifact schema differs")
    if expected_status is not None and payload.get("status") != expected_status:
        raise ValueError("V27 JSON artifact status differs")
    return payload


def write_v27_sealed_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a JSON artifact with a recomputable payload digest."""

    return write_v27_json(path, seal_v27_json_payload(payload))


def write_v27_sealed_json_if_absent_or_identical(
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """Create an immutable sealed artifact or reuse an equivalent result.

    Fitted heads, calibration registries, and evaluation reports are terminal
    evidence for one manifest.  They must not be silently replaced by a later
    run at the same path.  Mutable progress files keep using
    :func:`write_v27_sealed_json` deliberately.
    """

    return write_v27_json_if_identical(path, seal_v27_json_payload(payload))


def read_v27_sealed_json(path: str | Path) -> dict[str, Any]:
    """Read and integrity-check a sealed V27 JSON artifact."""

    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return verify_v27_json_payload(payload)


def environment_snapshot() -> dict[str, Any]:
    """Return only deterministic runtime facts useful in a training summary."""

    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


__all__ = [
    "V27Context",
    "build_v27_model",
    "configured_device",
    "environment_snapshot",
    "load_v27_context",
    "seed_v27",
    "seal_v27_json_payload",
    "read_v27_sealed_json",
    "write_v27_json",
    "write_v27_json_if_identical",
    "write_v27_sealed_json",
    "write_v27_sealed_json_if_absent_or_identical",
    "verify_v27_json_payload",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a sealed V27 JSON artifact")
    parser.add_argument("--verify-json", type=Path, required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    artifact = args.verify_json.resolve()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    verify_v27_json_payload(
        payload,
        expected_schema=args.schema,
        expected_status=args.status,
    )
    print(json.dumps({"status": "verified", "path": str(artifact)}, ensure_ascii=False))
