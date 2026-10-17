"""Provenance and artifact contracts for the isolated Bird-route experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256, relative_project_path, resolve_project_path, sha256_file

from v27.V27d.protocol import V27DContext, load_context as load_v27d_context


R5_EXPERIMENT_ID = "V27d-r5-bird-route"
R5_SCHEMA_VERSION = "RADAR-LN3-V27d-r5-bird-route-v1"
R5_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r5-manifest-v1"
R5_ROOT = Path("src-v3/v27/V27d/r5")
R5_PACKAGE_ROOT = Path("src-v3/v27/V27d/r5")
R5_DEFAULT_CONFIG = R5_ROOT / "config" / "V27d-r5-bird-route.json"
R5_ROLES = ("representation_fit", "representation_development", "calibration")
R5_ARMS = ("control", "stable8_treatment")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r5 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r5 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r5 {name}.{key} must be positive")
    return int(value)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R5_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r5 configuration must be an object")
    value = dict(payload)
    if value.get("schema_version") != R5_SCHEMA_VERSION or value.get("experiment_id") != R5_EXPERIMENT_ID:
        raise ValueError("V27d-r5 configuration identity differs")
    if value.get("seed") != 42:
        raise ValueError("V27d-r5 is fixed to diagnostic seed 42")
    source = _mapping(value.get("source"), "source")
    _text(source, "v27d_config_path", "source")
    model = _mapping(value.get("model"), "model")
    _positive_int(model, "hidden_dim", "model")
    _positive_int(model, "projection_dim", "model")
    training = _mapping(value.get("training"), "training")
    for key in ("batch_size", "epochs"):
        _positive_int(training, key, "training")
    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key in ("manifest_path", "cache_template", "checkpoint_template", "summary_template", "log_template", "report_path", "smoke_path"):
        _text(artifacts, key, "artifacts")
    value["config_path"] = path
    value["config_sha256"] = sha256_file(path)
    value["config_identity_sha256"] = canonical_sha256({key: item for key, item in value.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}})
    return value


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    artifact_root = resolve_project_path(root, R5_ROOT)
    try:
        target.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("V27d-r5 artifact escaped its experiment directory") from exc
    return target


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R5_PACKAGE_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }






def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r5 sealed JSON must be an object")
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d-r5 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r5 sealed JSON schema differs")
    return dict(value)


def load_r1_context(project_root: Path, config: Mapping[str, Any]) -> V27DContext:
    source = _mapping(config["source"], "source")
    return load_v27d_context(project_root, source["v27d_config_path"])


def build_manifest(project_root: Path, config: Mapping[str, Any], r1: V27DContext) -> dict[str, Any]:
    source = _mapping(config["source"], "source")
    return {
        "schema_version": R5_MANIFEST_SCHEMA,
        "experiment_id": R5_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(project_root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "v27d_manifest_path": relative_project_path(project_root, r1.manifest_path),
            "v27d_manifest_sha256": r1.manifest_sha256,
            "v27d_config_path": source["v27d_config_path"],
            "v27d_endpoint_manifest_sha256": r1.manifest.get("endpoint_manifest_sha256", ""),
        },
        "code_hashes": package_code_hashes(project_root),
        "arms": list(R5_ARMS),
        "roles": list(R5_ROLES),
        "seed": 42,
        "protocol": {
            "unknown_is_classifier_class": False,
            "classification_logits_input_to_head": False,
            "calibration_scope": "exploratory_threshold_transfer_only",
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
        },
    }




def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], V27DContext, dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    r1 = load_r1_context(root, config)
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R5_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, r1)
    actual = {key: item for key, item in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r5 manifest differs from configuration, source or implementation")
    return config, r1, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = ["R5_ARMS", "R5_EXPERIMENT_ID", "R5_MANIFEST_SCHEMA", "R5_ROLES", "R5_ROOT", "artifact_path", "build_manifest", "load_config", "load_context", "load_r1_context", "read_sealed_json"]