"""Configuration and provenance contracts for V27d-r8."""

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


R8_EXPERIMENT_ID = "V27d-r8-joint-course-stable8"
R8_SCHEMA_VERSION = "RADAR-LN3-V27d-r8-joint-course-stable8-v1"
R8_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r8-manifest-v1"
R8_CACHE_SCHEMA = "RADAR-LN3-V27d-r8-population-cache-v1"
R8_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r8-checkpoint-v1"
R8_ROOT = Path("src-v3/v27/V27d/r8")
R8_PACKAGE_ROOT = Path("src-v3/v27/V27d/r8")
R8_DEFAULT_CONFIG = R8_ROOT / "config" / "V27d-r8-joint-course-stable8.json"
R8_ROLES = ("representation_fit", "representation_development", "calibration")
R8_ARMS = ("zero_control", "stable8_treatment", "course_treatment", "joint_treatment")
R8_SEEDS = (42, 43, 44)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r8 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r8 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r8 {name}.{key} must be positive")
    return int(value)


def _bounded_float(mapping: Mapping[str, Any], key: str, name: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"V27d-r8 {name}.{key} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"V27d-r8 {name}.{key} must be in [0,1]")
    return result


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R8_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V27d-r8 configuration must be an object")
    config = dict(raw)
    if config.get("schema_version") != R8_SCHEMA_VERSION or config.get("experiment_id") != R8_EXPERIMENT_ID:
        raise ValueError("V27d-r8 configuration identity differs")
    seeds = tuple(int(value) for value in config.get("seeds", ()))
    if seeds != R8_SEEDS:
        raise ValueError("V27d-r8 seeds must be exactly 42,43,44")
    for section_name in ("source", "model", "training", "selection", "calibration", "artifacts", "protocol"):
        _mapping(config.get(section_name), section_name)
    source = _mapping(config["source"], "source")
    for key in ("r6_config_path", "r6_manifest_path", "r6_cache_template", "r6_liquid_checkpoint_template", "r6_report_template", "course_cache_template", "index_path", "forensics_report_path"):
        _text(source, key, "source")
    model = _mapping(config["model"], "model")
    for key in ("event_trajectory_dim", "task_fact_dim", "input_dim", "hidden_dim", "summary_dim", "projection_dim", "dispersion_dim", "course_dim"):
        _positive_int(model, key, "model")
    if int(model["input_dim"]) != int(model["event_trajectory_dim"]) + int(model["task_fact_dim"]):
        raise ValueError("V27d-r8 input dimension differs")
    training = _mapping(config["training"], "training")
    for key in ("batch_size", "epochs", "minimum_quality_events"):
        _positive_int(training, key, "training")
    selection = _mapping(config["selection"], "selection")
    for key in ("max_additional_bird_rejections_per_pool", "minimum_calibration_seed_wins"):
        value = selection.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V27d-r8 selection.{key} differs")
    for key in ("min_development_unknown_rejection", "min_calibration_unknown_rejection"):
        _bounded_float(selection, key, "selection")
    artifacts = _mapping(config["artifacts"], "artifacts")
    for key in ("manifest_path", "cache_template", "checkpoint_template", "summary_template", "log_template", "smoke_path", "evaluation_template", "summary_path", "experiment_doc_path"):
        _text(artifacts, key, "artifacts")
    config["config_path"] = path
    config["config_sha256"] = sha256_file(path)
    config["config_identity_sha256"] = canonical_sha256({key: value for key, value in config.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}})
    return config


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    experiment_root = resolve_project_path(root, R8_ROOT)
    try:
        target.relative_to(experiment_root)
    except ValueError as exc:
        raise ValueError("V27d-r8 artifact escaped experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R8_PACKAGE_ROOT
    return {relative_project_path(project_root, path): sha256_file(path) for path in sorted(package_root.rglob("*.py")) if "__pycache__" not in path.parts}


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r8 payload is already sealed")
    return {**dict(payload), "payload_sha256": canonical_sha256(payload)}


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json_text(seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r8 sealed JSON must be an object")
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d-r8 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r8 sealed JSON schema differs")
    return dict(value)


def build_manifest(root: Path, config: Mapping[str, Any], r6_manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    source_keys = ("r6_config_path", "r6_manifest_path", "course_cache_template", "index_path", "forensics_report_path")
    paths = {key: source_path(root, source[key].format(role="representation_fit") if "template" in key else source[key], must_exist=True) for key in source_keys}
    return {
        "schema_version": R8_MANIFEST_SCHEMA,
        "experiment_id": R8_EXPERIMENT_ID,
        "config": {"path": relative_project_path(root, config["config_path"]), "sha256": config["config_sha256"], "identity_sha256": config["config_identity_sha256"]},
        "source": {
            "r6_config_path": source["r6_config_path"],
            "r6_config_sha256": sha256_file(paths["r6_config_path"]),
            "r6_manifest_path": source["r6_manifest_path"],
            "r6_manifest_sha256": sha256_file(paths["r6_manifest_path"]),
            "r6_manifest_payload_sha256": r6_manifest["payload_sha256"],
            "course_cache_template": source["course_cache_template"],
            "course_fit_sha256": sha256_file(source_path(root, source["course_cache_template"].format(role="representation_fit"), must_exist=True)),
            "course_development_sha256": sha256_file(source_path(root, source["course_cache_template"].format(role="representation_development"), must_exist=True)),
            "course_calibration_sha256": sha256_file(source_path(root, source["course_cache_template"].format(role="calibration"), must_exist=True)),
            "index_path": source["index_path"],
            "index_sha256": sha256_file(paths["index_path"]),
            "forensics_report_path": source["forensics_report_path"],
            "forensics_report_sha256": sha256_file(paths["forensics_report_path"]),
        },
        "code_hashes": package_code_hashes(root),
        "arms": list(R8_ARMS),
        "roles": list(R8_ROLES),
        "seeds": list(R8_SEEDS),
        "protocol": dict(config["protocol"]),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    r6_manifest = read_sealed_json(source_path(root, config["source"]["r6_manifest_path"], must_exist=True))
    manifest = build_manifest(root, config, r6_manifest)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, manifest)
    return config, path, sha256_file(path)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    r6_manifest = read_sealed_json(source_path(root, config["source"]["r6_manifest_path"], must_exist=True))
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R8_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, r6_manifest)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r8 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = ["R8_ARMS", "R8_CACHE_SCHEMA", "R8_CHECKPOINT_SCHEMA", "R8_DEFAULT_CONFIG", "R8_EXPERIMENT_ID", "R8_MANIFEST_SCHEMA", "R8_ROLES", "R8_SCHEMA_VERSION", "R8_SEEDS", "artifact_path", "load_config", "load_context", "prepare_manifest", "read_sealed_json", "source_path", "write_sealed_json"]
