"""Protocol and immutable provenance for the independent Bird-route experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file

from v28.protocol import load_context as load_v28_context
from v28.protocol import read_sealed_json, write_sealed_json
from v28.registry import load_registry as load_v28_registry


EXPERIMENT_ID = "V28-bird-route-r1-profile-shape"
CONFIG_PATH = Path("src-v3/v28_experiments/bird_route_r1_profile_shape/config/V28-bird-route-r1-profile-shape.json")
MANIFEST_SCHEMA = "RADAR-LN3-V28-bird-route-r1-profile-shape-manifest-v1"
FEATURE_SCHEMA = "RADAR-LN3-V28-bird-route-r1-profile-shape-features-v1"
CHECKPOINT_SCHEMA = "RADAR-LN3-V28-bird-route-r1-profile-shape-head-v1"


@dataclass(frozen=True)
class R1Context:
    project_root: Path
    config: dict[str, Any]
    config_path: Path
    config_sha256: str
    v28_context: Any
    v28_registry: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _resolve(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V28 Bird-route r1 path escaped project root: {value}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def load_config(root: Path) -> tuple[dict[str, Any], Path, str]:
    path = _resolve(root, CONFIG_PATH, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("V28 Bird-route r1 config identity differs")
    value = dict(payload)
    audit_hash = value.get("base", {}).get("v28_audit_registry_sha256")
    if not isinstance(audit_hash, str) or len(audit_hash) != 64:
        raise ValueError("V28 Bird-route r1 audit binding is malformed")
    if int(value["training"]["seed"]) != 42:
        raise ValueError("V28 Bird-route r1 uses exactly seed 42")
    return value, path, sha256_file(path)


def _code_hashes(root: Path) -> dict[str, str]:
    package = root / "src-v3/v28_experiments/bird_route_r1_profile_shape"
    result: dict[str, str] = {}
    for path in sorted(package.glob("*.py")):
        result[relative_project_path(root, path)] = sha256_file(path)
    return result


def _artifact_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "v28_manifest": config["base"]["v28_manifest_path"],
        "v28_registry": config["base"]["v28_registry_path"],
        "v28_audit_report": config["base"]["v28_audit_report_path"],
        "v28_pool_registry": config["base"]["pool_registry_path"],
        "index": config["data"]["index_path"],
        "provenance": config["data"]["provenance_path"],
        "canonical_frame_index": config["data"]["canonical_frame_index_path"],
    }
    result = {}
    for name, value in paths.items():
        result[name] = sha256_file(_resolve(root, value, must_exist=True))
    return result


def build_manifest_body(root: Path, config: Mapping[str, Any], config_sha256: str) -> dict[str, Any]:
    evidence_dir = root / "src-v3/v28_experiments/bird_route_r1_profile_shape/cache"
    evidence = {
        relative_project_path(root, path): sha256_file(path)
        for path in sorted(evidence_dir.glob("V28-bird-route-r1-*-profile-evidence.npz"))
        if path.is_file()
        and path.name not in {"V28-bird-route-r1-test-profile-evidence.npz"}
    }
    features_dir = root / "src-v3/v28_experiments/bird_route_r1_profile_shape/cache"
    feature_files = {
        relative_project_path(root, path): sha256_file(path)
        for path in sorted(features_dir.glob("V28-bird-route-r1-*-profile-features.pt"))
        if path.is_file()
        and path.name not in {"V28-bird-route-r1-post_selection_audit-profile-features.pt"}
    }
    artifact_files: dict[str, str] = {}
    for key in (
        "audit_report_path",
        "diagnostic_report_path",
        "training_checkpoint_path",
        "training_summary_path",
        "calibration_report_path",
    ):
        path = _resolve(root, config["artifacts"][key])
        if path.is_file():
            artifact_files[relative_project_path(root, path)] = sha256_file(path)
    body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config": {"path": relative_project_path(root, _resolve(root, CONFIG_PATH)), "sha256": config_sha256},
        "base": dict(config["base"]),
        "source_hashes": _artifact_hashes(root, config),
        "code_hashes": _code_hashes(root),
        "evidence_cache_hashes": evidence,
        "feature_cache_hashes": feature_files,
        "experiment_artifact_hashes": artifact_files,
        "protocol": dict(config["protocol"]),
        "model": dict(config["model"]),
        "selection": dict(config["selection"]),
    }
    return body


def write_manifest(root: Path) -> dict[str, Any]:
    config, config_path, config_sha256 = load_config(root)
    body = build_manifest_body(root, config, config_sha256)
    payload = {**body, "manifest_sha256": canonical_sha256(body)}
    path = _resolve(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, payload)
    return payload


def load_context(root: str | Path) -> R1Context:
    project_root = Path(root).resolve()
    config, config_path, config_sha256 = load_config(project_root)
    v28_context = load_v28_context(project_root)
    v28_registry = load_v28_registry(v28_context)
    audit_path = _resolve(project_root, config["base"]["v28_audit_report_path"], must_exist=True)
    if sha256_file(audit_path) != config["base"]["v28_audit_report_sha256"]:
        raise ValueError("V28 Bird-route r1 audit report hash differs")
    audit = read_sealed_json(audit_path)
    if audit.get("registry_sha256") != config["base"]["v28_audit_registry_sha256"]:
        raise ValueError("V28 Bird-route r1 audit registry binding differs")
    if v28_registry.get("registry_sha256") != config["base"]["v28_registry_sha256"]:
        raise ValueError("V28 Bird-route r1 current V28 registry differs")
    manifest_path = _resolve(project_root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "payload_sha256" and key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(body):
        raise ValueError("V28 Bird-route r1 manifest digest differs")
    expected = build_manifest_body(project_root, config, config_sha256)
    if body != expected:
        raise ValueError("V28 Bird-route r1 manifest differs from current artifacts")
    return R1Context(project_root, config, config_path, config_sha256, v28_context, v28_registry, manifest, manifest_path, sha256_file(manifest_path))


def load_unsealed_context(root: str | Path) -> R1Context:
    """Load immutable V28 bindings before the r1 manifest exists."""

    project_root = Path(root).resolve()
    config, config_path, config_sha256 = load_config(project_root)
    v28_context = load_v28_context(project_root)
    v28_registry = load_v28_registry(v28_context)
    manifest_path = _resolve(project_root, config["artifacts"]["manifest_path"])
    return R1Context(project_root, config, config_path, config_sha256, v28_context, v28_registry, {}, manifest_path, "")


def resolve_artifact(context: R1Context, key: str, *, must_exist: bool = False) -> Path:
    return _resolve(context.project_root, context.config["artifacts"][key], must_exist=must_exist)


__all__ = ["CHECKPOINT_SCHEMA", "EXPERIMENT_ID", "FEATURE_SCHEMA", "MANIFEST_SCHEMA", "R1Context", "build_manifest_body", "load_context", "load_config", "load_unsealed_context", "read_sealed_json", "resolve_artifact", "write_manifest"]
