"""Versioned paths and integrity helpers for the V26e long-horizon L1 OOF.

V26e changes only the L1 training horizon of V26c from 20 to 32 epochs.  The
fold assignment, feature caches, fold-train normalization, and V26c L2
epoch-15 outputs are reused read-only.  Unknown parents remain outside every
training and calibration population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


from radar_ln3 import known_gate_v26c_oof as _v26c


FOLD_COUNT = 5
L1_TRAIN_EPOCHS = 32
L1_EPOCH = 32

V26C_CONTRACT = _v26c.CONTRACT
V26C_READINESS = _v26c.READINESS
V26C_SOURCE_FEATURE_MANIFEST = _v26c.SOURCE_FEATURE_MANIFEST
V26C_CONTRACT_SCHEMA = "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1"
V26C_READINESS_SCHEMA = "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness-v1"
BOUND_FILE_MAX_BYTES = 64 * 1024 * 1024

CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness.json"
)
ARTIFACT_ROOT = Path(
    "artifacts/RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration"
)

CONTRACT_SCHEMA = (
    "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-contract-v1"
)
READINESS_SCHEMA = (
    "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness-v1"
)
L1_CHECKPOINT_SCHEMA = "RADAR-LN3-V26e-l1-epoch32-known-only-oof-checkpoint-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def bound_path(root: Path, value: str | Path, description: str) -> Path:
    """Resolve a contract path while refusing project-root escape."""

    path = resolve_path(root, value)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V26e {description} escapes the project root: {value}") from exc
    return path


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = BOUND_FILE_MAX_BYTES,
) -> Path:
    """Resolve, bound, size-check and optionally hash-check one artifact."""

    path = bound_path(root, value, description)
    if not path.is_file():
        raise FileNotFoundError(f"V26e {description} is missing: {path}")
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(
            f"V26e {description} exceeds the bounded size: "
            f"{path.stat().st_size} > {int(max_bytes)} bytes"
        )
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise ValueError(f"V26e {description} hash differs: {path}")
    return path






def validate_v26c_source_payload(
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    *,
    contract: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a V26c source pair and every hash transitively named by it."""

    contract_path = bound_path(root, contract_path, "V26c source contract")
    readiness_path = bound_path(root, readiness_path, "V26c source readiness")
    if contract_path != bound_path(root, V26C_CONTRACT, "V26c source contract"):
        raise ValueError("V26e V26c source contract path is not canonical")
    if readiness_path != bound_path(root, V26C_READINESS, "V26c source readiness"):
        raise ValueError("V26e V26c source readiness path is not canonical")
    require_bound_file(root, contract_path, "V26c source contract")
    require_bound_file(root, readiness_path, "V26c source readiness")
    contract_payload = dict(contract or json.loads(contract_path.read_text(encoding="utf-8")))
    readiness_payload = dict(readiness or json.loads(readiness_path.read_text(encoding="utf-8")))
    checks = dict(readiness_payload.get("checks") or {})
    if (
        contract_payload.get("schema_version") != V26C_CONTRACT_SCHEMA
        or readiness_payload.get("schema_version") != V26C_READINESS_SCHEMA
        or readiness_payload.get("status") != "passed"
        or readiness_payload.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract_payload.get("known_train_parent_count", -1)) != 986
        or int(contract_payload.get("unknown_training_exposure", -1)) != 0
        or int(contract_payload.get("unknown_normalization_exposure", -1)) != 0
        or int(contract_payload.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract_payload.get("validation_access_count", -1)) != 0
        or int(contract_payload.get("test_access_count", -1)) != 0
    ):
        raise ValueError("V26e bound V26c source contract/readiness is invalid")

    source_bindings = dict(contract_payload.get("source_bindings") or {})
    if not source_bindings:
        raise ValueError("V26e V26c source contract has no transitive source bindings")
    resolved_bindings: dict[str, Path] = {}
    for name, raw_binding in source_bindings.items():
        binding = dict(raw_binding or {})
        path = require_bound_file(
            root,
            str(binding.get("path", "")),
            f"V26c source binding {name}",
            expected_sha256=str(binding.get("sha256", "")),
        )
        resolved_bindings[str(name)] = path

    rebuild = dict(contract_payload.get("feature_value_rebuild") or {})
    manifest_path = require_bound_file(
        root,
        str(rebuild.get("manifest_path", "")),
        "V26c source feature rebuild manifest",
        expected_sha256=str(rebuild.get("manifest_sha256", "")),
    )
    if manifest_path != bound_path(root, V26C_SOURCE_FEATURE_MANIFEST, "V26c source feature rebuild manifest"):
        raise ValueError("V26e V26c source feature rebuild manifest path is not canonical")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "RADAR-LN3-V26c-label-free-source-feature-rebuild-manifest-v1"
        or int(manifest.get("known_train_parent_count", -1)) != 986
        or int(manifest.get("dbb_train_parent_count", -1)) != 953
        or int(manifest.get("unknown_training_exposure", -1)) != 0
        or int(manifest.get("validation_access_count", -1)) != 0
        or int(manifest.get("test_access_count", -1)) != 0
        or manifest.get("feature_value_label_free_after_scope_selection") is not True
        or manifest.get("previous_v21e_or_v18_cache_values_reused") is not False
    ):
        raise ValueError("V26e V26c source feature rebuild manifest is invalid")
    index_binding = dict(source_bindings.get("index") or {})
    if (
        manifest.get("source_index") != index_binding.get("path")
        or manifest.get("source_index_sha256") != index_binding.get("sha256")
    ):
        raise ValueError("V26e V26c source feature manifest is not bound to its source index")
    for name in (
        "course",
        "rd_physical4",
        "micro_doppler8",
        "mixed_l1_known_inference_only_rd_physical4",
        "mixed_l1_known_inference_only_micro_doppler8",
    ):
        entry = dict(manifest.get(name) or {})
        require_bound_file(
            root,
            str(entry.get("path", "")),
            f"V26c source rebuild cache {name}",
            expected_sha256=str(entry.get("sha256", "")),
        )
    return {
        "contract_path": contract_path,
        "readiness_path": readiness_path,
        "contract": contract_payload,
        "readiness": readiness_payload,
        "source_bindings": resolved_bindings,
        "feature_manifest_path": manifest_path,
        "feature_manifest": manifest,
    }


def validate_frozen_v26c_sources(root: Path, contract: Mapping[str, Any]) -> dict[str, Path]:
    """Re-check the immutable V26c inputs after V26e preparation.

    Preparation records the V26c source hashes, but a later train/report
    invocation must verify the files again.  Otherwise a replaced source
    contract could silently change the meaning of an already prepared V26e
    experiment.
    """

    source = dict(contract.get("source_v26c") or {})
    expected = (
        ("contract_path", "contract_sha256", V26C_CONTRACT, "V26c source contract"),
        ("readiness_path", "readiness_sha256", V26C_READINESS, "V26c source readiness"),
    )
    result: dict[str, Path] = {}
    for path_key, hash_key, canonical, description in expected:
        path_value = source.get(path_key)
        actual_hash = source.get(hash_key)
        path = bound_path(root, str(path_value or ""), description)
        if path != bound_path(root, canonical, description):
            raise ValueError(f"V26e {description} path differs from the canonical source")
        require_bound_file(
            root,
            path,
            description,
            expected_sha256=str(actual_hash),
        )
        result[path_key] = path
    validate_v26c_source_payload(
        root,
        result["contract_path"],
        result["readiness_path"],
    )
    return result




def fold_directory(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26e fold: {fold_id}")
    return ARTIFACT_ROOT / f"fold-{int(fold_id):02d}"


def l1_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-seed43-config.json"
    )


def l1_data_contract_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-data-contract.json"
    )


def l1_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "epoch-32-model.pt"




def l1_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "training-summary.json"


def l1_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1-held-out-predictions.json"




# These are deliberately aliases to V26c's physical files.  V26e changes no
# feature values, normalization values, or L2 outputs.
l1_normalization_path = _v26c.l1_normalization_path
l1_train_feature_cache_path = _v26c.l1_train_feature_cache_path
l1_held_out_feature_cache_path = _v26c.l1_held_out_feature_cache_path
