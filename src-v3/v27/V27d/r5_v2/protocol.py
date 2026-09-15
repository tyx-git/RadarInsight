"""Provenance and artifact contracts for V27d-r5 Bird-route v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)

from v27.V27d.protocol import load_context as load_v27d_context


V2_EXPERIMENT_ID = "V27d-r5-bird-route-v2"
V2_SCHEMA_VERSION = "RADAR-LN3-V27d-r5-bird-route-v2-v1"
V2_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r5-bird-route-v2-manifest-v1"
V2_CACHE_SCHEMA = "RADAR-LN3-V27d-r5-bird-route-v2-population-cache-v1"
V2_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r5-bird-route-v2-checkpoint-v1"
V2_ROOT = Path("src-v3/v27/V27d/r5_v2")
V2_PACKAGE_ROOT = Path("src-v3/v27/V27d/r5_v2")
V2_DEFAULT_CONFIG = V2_ROOT / "config" / "V27d-r5-bird-route-v2.json"
V2_ROLES = ("representation_fit", "representation_development", "calibration")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r5-v2 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r5-v2 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r5-v2 {name}.{key} must be positive")
    return int(value)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / V2_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r5-v2 configuration must be an object")
    value = dict(value)
    if value.get("schema_version") != V2_SCHEMA_VERSION or value.get("experiment_id") != V2_EXPERIMENT_ID:
        raise ValueError("V27d-r5-v2 configuration identity differs")
    if value.get("seed") != 42:
        raise ValueError("V27d-r5-v2 is registered for seed 42")
    source = _mapping(value.get("source"), "source")
    for key in ("r5_config_path", "r5_manifest_path", "r5_report_path", "r5_population_cache_template", "v27d_config_path", "v27d_endpoint_cache_template"):
        _text(source, key, "source")
    model = _mapping(value.get("model"), "model")
    for key in ("hidden_dim", "projection_dim"):
        _positive_int(model, key, "model")
    training = _mapping(value.get("training"), "training")
    for key in ("batch_size", "epochs", "minimum_quality_events"):
        _positive_int(training, key, "training")
    for key in ("prefix_consistency_weight", "unknown_terminal_uplift_weight", "unknown_terminal_uplift_margin"):
        if isinstance(training.get(key), bool) or not isinstance(training.get(key), (int, float)) or float(training[key]) < 0.0:
            raise ValueError(f"V27d-r5-v2 training.{key} must be non-negative")
    selection = _mapping(value.get("selection"), "selection")
    for key in ("bird_recall_floor",):
        if isinstance(selection.get(key), bool) or not isinstance(selection.get(key), (int, float)) or not 0.0 < float(selection[key]) <= 1.0:
            raise ValueError(f"V27d-r5-v2 selection.{key} must be within (0, 1]")
    for key in ("max_correct_bird_new_rejections", "calibration_max_additional_correct_bird_rejections"):
        if isinstance(selection.get(key), bool) or not isinstance(selection.get(key), int) or int(selection[key]) < 0:
            raise ValueError(f"V27d-r5-v2 selection.{key} must be non-negative")
    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key in ("manifest_path", "cache_template", "checkpoint_path", "summary_path", "log_path", "audit_path", "smoke_path", "evaluation_path", "experiment_doc_path"):
        _text(artifacts, key, "artifacts")
    value["config_path"] = path
    value["config_sha256"] = sha256_file(path)
    value["config_identity_sha256"] = canonical_sha256({key: item for key, item in value.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}})
    return value


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    artifact_root = resolve_project_path(root, V2_ROOT)
    try:
        target.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("V27d-r5-v2 artifact escaped the experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / V2_PACKAGE_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r5-v2 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json_text(seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r5-v2 sealed JSON must be an object")
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d-r5-v2 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r5-v2 sealed JSON schema differs")
    return dict(value)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    source = config["source"]
    r5_manifest_path = source_path(root, source["r5_manifest_path"], must_exist=True)
    r5_report_path = source_path(root, source["r5_report_path"], must_exist=True)
    r5_config_path = source_path(root, source["r5_config_path"], must_exist=True)
    v27d_context = load_v27d_context(root, source["v27d_config_path"])
    manifest_path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(manifest_path, expected_schema=V2_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, v27d_context)
    actual = {key: item for key, item in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r5-v2 manifest differs from configuration or source hashes")
    return config, {**manifest, "manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path), "r5_manifest_path": r5_manifest_path, "r5_report_path": r5_report_path, "r5_config_path": r5_config_path}


def build_manifest(project_root: Path, config: Mapping[str, Any], v27d_context: Any) -> dict[str, Any]:
    root = project_root.resolve()
    source = config["source"]
    paths = {
        key: source_path(root, source[key], must_exist=True)
        for key in ("r5_config_path", "r5_manifest_path", "r5_report_path")
    }
    return {
        "schema_version": V2_MANIFEST_SCHEMA,
        "experiment_id": V2_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "r5_config_path": source["r5_config_path"],
            "r5_config_sha256": sha256_file(paths["r5_config_path"]),
            "r5_manifest_path": source["r5_manifest_path"],
            "r5_manifest_sha256": sha256_file(paths["r5_manifest_path"]),
            "r5_report_path": source["r5_report_path"],
            "r5_report_sha256": sha256_file(paths["r5_report_path"]),
            "v27d_manifest_sha256": v27d_context.manifest_sha256,
        },
        "code_hashes": package_code_hashes(root),
        "roles": [*V2_ROLES],
        "seed": 42,
        "protocol": {
            "direct_baseline": "V27d-r5-control",
            "unique_variable": "causal_prefix_consistency_objective",
            "unknown_is_classifier_class": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "formal_l3_enabled": False,
            "calibration_scope": "exploratory_threshold_transfer_only",
            "test_payload_access_count": 0,
        },
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    v27d_context = load_v27d_context(root, config["source"]["v27d_config_path"])
    manifest = build_manifest(root, config, v27d_context)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, manifest)
    return config, path, sha256_file(path)


__all__ = [
    "V2_CACHE_SCHEMA",
    "V2_CHECKPOINT_SCHEMA",
    "V2_DEFAULT_CONFIG",
    "V2_EXPERIMENT_ID",
    "V2_MANIFEST_SCHEMA",
    "V2_ROLES",
    "artifact_path",
    "build_manifest",
    "load_config",
    "load_context",
    "package_code_hashes",
    "prepare_manifest",
    "read_sealed_json",
    "source_path",
    "write_sealed_json",
]
