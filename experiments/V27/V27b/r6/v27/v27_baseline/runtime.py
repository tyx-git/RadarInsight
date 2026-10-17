"""Shared execution helpers for the sealed V27 baseline.

This module is intentionally small and policy-free beyond the V27 protocol:
it is the single place where an entry point resolves the verified context,
chooses the configured device, seeds the three RNG families, and writes a
mutable JSON artifact.  It never opens test or Unknown payloads.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.runtime import canonical_sha256

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






def read_v27_sealed_json(path: str | Path) -> dict[str, Any]:
    """Read and integrity-check a sealed V27 JSON artifact."""

    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return verify_v27_json_payload(payload)




__all__ = ["V27Context", "load_v27_context", "seed_v27", "read_v27_sealed_json", "verify_v27_json_payload"]