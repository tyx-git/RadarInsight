"""Provenance and artifact contracts for V27d-r6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import atomic_write_text, canonical_sha256, json_text, relative_project_path, resolve_project_path, sha256_file

from v27.V27d.protocol import load_context as load_v27d_context


R6_EXPERIMENT_ID = "V27d-r6-bird-liquid-verifier"
R6_SCHEMA_VERSION = "RADAR-LN3-V27d-r6-bird-liquid-verifier-v1"
R6_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r6-manifest-v1"
R6_CACHE_SCHEMA = "RADAR-LN3-V27d-r6-population-cache-v1"
R6_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r6-checkpoint-v1"
R6_ROOT = Path("src-v3/v27/V27d/r6")
R6_PACKAGE_ROOT = Path("src-v3/v27/V27d/r6")
R6_DEFAULT_CONFIG = R6_ROOT / "config" / "V27d-r6-bird-liquid-verifier.json"
R6_ROLES = ("representation_fit", "representation_development", "calibration")
R6_ARMS = ("gru_control", "liquid_treatment")
R6_SEEDS = (42, 43, 44)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r6 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r6 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r6 {name}.{key} must be positive")
    return int(value)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R6_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r6 configuration must be an object")
    value = dict(payload)
    if value.get("schema_version") != R6_SCHEMA_VERSION or value.get("experiment_id") != R6_EXPERIMENT_ID:
        raise ValueError("V27d-r6 configuration identity differs")
    seeds = tuple(int(seed) for seed in value.get("seeds", ()))
    if seeds != R6_SEEDS:
        raise ValueError("V27d-r6 seed registry must be exactly 42,43,44")
    source = _mapping(value.get("source"), "source")
    for key in ("r5_v2_config_path", "r5_v2_manifest_path", "r5_v2_cache_template", "r5_control_report_path", "r5_control_config_path", "r5_control_checkpoint_path", "v27d_config_path", "v27d_l1_endpoint_cache_template"):
        _text(source, key, "source")
    model = _mapping(value.get("model"), "model")
    for key in ("event_trajectory_dim", "task_fact_dim", "input_dim", "hidden_dim", "summary_dim", "projection_dim"):
        _positive_int(model, key, "model")
    if int(model["input_dim"]) != int(model["event_trajectory_dim"]) + int(model["task_fact_dim"]):
        raise ValueError("V27d-r6 input dimension does not equal event plus task facts")
    training = _mapping(value.get("training"), "training")
    for key in ("batch_size", "epochs", "minimum_quality_events"):
        _positive_int(training, key, "training")
    selection = _mapping(value.get("selection"), "selection")
    for key in ("max_additional_bird_rejections_per_pool",):
        if isinstance(selection.get(key), bool) or not isinstance(selection.get(key), int) or int(selection[key]) < 0:
            raise ValueError(f"V27d-r6 selection.{key} must be non-negative")
    for key in ("min_precision_gain", "min_development_unknown_rejection", "min_calibration_unknown_rejection"):
        if isinstance(selection.get(key), bool) or not isinstance(selection.get(key), (int, float)) or not 0.0 <= float(selection[key]) <= 1.0:
            raise ValueError(f"V27d-r6 selection.{key} must be in [0,1]")
    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key in ("manifest_path", "cache_template", "checkpoint_template", "summary_template", "log_template", "smoke_path", "evaluation_template", "summary_report_path", "experiment_doc_path"):
        _text(artifacts, key, "artifacts")
    value["config_path"] = path
    value["config_sha256"] = sha256_file(path)
    value["config_identity_sha256"] = canonical_sha256({key: item for key, item in value.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}})
    return value


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    experiment_root = resolve_project_path(root, R6_ROOT)
    try:
        target.relative_to(experiment_root)
    except ValueError as exc:
        raise ValueError("V27d-r6 artifact escaped its experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R6_PACKAGE_ROOT
    return {relative_project_path(project_root, path): sha256_file(path) for path in sorted(package_root.rglob("*.py")) if "__pycache__" not in path.parts}


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r6 payload is already sealed")
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
        raise ValueError("V27d-r6 sealed JSON must be an object")
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d-r6 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r6 sealed JSON schema differs")
    return dict(value)


def build_manifest(project_root: Path, config: Mapping[str, Any], v27d_context: Any) -> dict[str, Any]:
    root = project_root.resolve()
    source = config["source"]
    source_keys = ("r5_v2_config_path", "r5_v2_manifest_path", "r5_control_report_path", "r5_control_config_path", "r5_control_checkpoint_path")
    paths = {key: source_path(root, source[key], must_exist=True) for key in source_keys}
    return {
        "schema_version": R6_MANIFEST_SCHEMA,
        "experiment_id": R6_EXPERIMENT_ID,
        "config": {"path": relative_project_path(root, config["config_path"]), "sha256": config["config_sha256"], "identity_sha256": config["config_identity_sha256"]},
        "source": {
            "r5_v2_config_path": source["r5_v2_config_path"],
            "r5_v2_config_sha256": sha256_file(paths["r5_v2_config_path"]),
            "r5_v2_manifest_path": source["r5_v2_manifest_path"],
            "r5_v2_manifest_sha256": sha256_file(paths["r5_v2_manifest_path"]),
            "r5_control_report_path": source["r5_control_report_path"],
            "r5_control_report_sha256": sha256_file(paths["r5_control_report_path"]),
            "r5_control_config_path": source["r5_control_config_path"],
            "r5_control_config_sha256": sha256_file(paths["r5_control_config_path"]),
            "r5_control_checkpoint_path": source["r5_control_checkpoint_path"],
            "r5_control_checkpoint_sha256": sha256_file(paths["r5_control_checkpoint_path"]),
            "v27d_manifest_sha256": v27d_context.manifest_sha256,
        },
        "code_hashes": package_code_hashes(root),
        "roles": list(R6_ROLES),
        "arms": list(R6_ARMS),
        "seeds": list(R6_SEEDS),
        "protocol": dict(config["protocol"]),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    v27d_context = load_v27d_context(root, config["source"]["v27d_config_path"])
    manifest = build_manifest(root, config, v27d_context)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, manifest)
    return config, path, sha256_file(path)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    v27d_context = load_v27d_context(root, config["source"]["v27d_config_path"])
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R6_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, v27d_context)
    actual = {key: item for key, item in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r6 manifest differs from configuration, source or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = ["R6_ARMS", "R6_CACHE_SCHEMA", "R6_CHECKPOINT_SCHEMA", "R6_DEFAULT_CONFIG", "R6_EXPERIMENT_ID", "R6_MANIFEST_SCHEMA", "R6_ROLES", "R6_SEEDS", "artifact_path", "load_config", "load_context", "prepare_manifest", "read_sealed_json", "source_path", "write_sealed_json"]
