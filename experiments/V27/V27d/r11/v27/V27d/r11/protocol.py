"""Configuration and provenance contracts for V27d-r11."""

from __future__ import annotations

import json
import math
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


R11_EXPERIMENT_ID = "V27d-r11"
R11_SCHEMA_VERSION = "RADAR-LN3-V27d-r11-full-stable8-causal-shape-v1"
R11_CONFIG_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-config"
R11_MANIFEST_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-manifest"
R11_CACHE_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-cache"
R11_CHECKPOINT_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-checkpoint"
R11_REPORT_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-evaluation"
R11_SMOKE_SCHEMA_VERSION = f"{R11_SCHEMA_VERSION}-smoke"
R11_DEFAULT_CONFIG = "src-v3/v27/V27d/r11/config/V27d-r11-full-stable8-causal-shape.json"
R11_ROOT = "src-v3/v27/V27d/r11"
R11_SEEDS = (42, 44)
R11_ROLES = ("representation_fit", "representation_development", "calibration")
R11_STABLE8_DIM = 8
R11_SHAPE_FEATURE_DIM = 32
_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r11 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r11 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r11 {name}.{key} must be positive")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27d-r11 {name}.{key} must be non-negative")
    return int(value)


def _finite_float(mapping: Mapping[str, Any], key: str, name: str, *, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"V27d-r11 {name}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d-r11 {name}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27d-r11 {name}.{key} is invalid")
    return result


def _sha(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _text(mapping, key, name).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27d-r11 {name}.{key} must be a SHA-256 digest")
    return value


def _sha_map(mapping: Mapping[str, Any], key: str, name: str, keys: tuple[str, ...]) -> dict[str, str]:
    raw = _mapping(mapping.get(key), f"{name}.{key}")
    if set(raw) != set(keys):
        raise ValueError(f"V27d-r11 {name}.{key} keys differ")
    return {item: _sha(raw, item, f"{name}.{key}") for item in keys}


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R11_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if config.get("schema_version") != R11_CONFIG_SCHEMA_VERSION or config.get("experiment_id") != R11_EXPERIMENT_ID:
        raise ValueError("V27d-r11 configuration identity differs")
    if tuple(config.get("seeds", ())) != R11_SEEDS:
        raise ValueError("V27d-r11 seeds must be exactly 42 and 44")

    source = _mapping(config.get("source"), "source")
    for key in ("r6_config_path", "r6_manifest_path", "r6_cache_template", "r6_checkpoint_template"):
        _text(source, key, "source")
    _sha(source, "expected_r6_config_sha256", "source")
    _sha(source, "expected_r6_manifest_sha256", "source")
    _sha_map(source, "expected_cache_file_sha256s", "source", R11_ROLES)
    _sha_map(source, "expected_cache_payload_sha256s", "source", R11_ROLES)
    _sha_map(source, "expected_checkpoint_file_sha256s", "source", tuple(str(seed) for seed in R11_SEEDS))
    _sha_map(source, "expected_checkpoint_head_state_sha256s", "source", tuple(str(seed) for seed in R11_SEEDS))

    model = _mapping(config.get("model"), "model")
    if _positive_int(model, "stable8_dim", "model") != R11_STABLE8_DIM or _positive_int(model, "shape_feature_dim", "model") != R11_SHAPE_FEATURE_DIM:
        raise ValueError("V27d-r11 stable8 shape dimensions differ")
    for key in ("event_trajectory_dim", "task_fact_dim", "hidden_dim", "summary_dim", "projection_dim"):
        _positive_int(model, key, "model")
    if int(model.get("input_dim", -1)) != int(model["event_trajectory_dim"]) + int(model["task_fact_dim"]):
        raise ValueError("V27d-r11 recurrent input dimension differs")
    _finite_float(model, "liquid_tau_min_seconds", "model", positive=True)
    _finite_float(model, "liquid_tau_max_seconds", "model", positive=True)
    _finite_float(model, "liquid_zero_dt_event_floor_seconds", "model", positive=True)
    _finite_float(model, "liquid_dt_max_seconds", "model", positive=True)
    _finite_float(model, "liquid_invalid_dt_observation_gain", "model")

    training = _mapping(config.get("training"), "training")
    for key in ("dataloader_seed", "batch_size", "epochs", "bootstrap_replicates", "minimum_quality_events"):
        _positive_int(training, key, "training")
    _finite_float(training, "learning_rate", "training", positive=True)
    _finite_float(training, "weight_decay", "training")
    _finite_float(training, "gradient_clip_norm", "training", positive=True)
    if _text(training, "device", "training").lower() not in {"cpu", "cuda"}:
        raise ValueError("V27d-r11 training.device must be cpu or cuda")
    if training.get("loss_name") != "parent_balanced_bce_on_eligible_prefixes" or training.get("loss_reduction") != "equal_parent_then_equal_positive_negative_group":
        raise ValueError("V27d-r11 knownness loss contract differs")

    selection = _mapping(config.get("selection"), "selection")
    _nonnegative_int(selection, "max_additional_bird_rejections_per_pool", "selection")
    calibration = _mapping(config.get("calibration"), "calibration")
    _positive_int(calibration, "formal_unknown_support_required", "calibration")
    if calibration.get("formal_l3_enabled") is not False:
        raise ValueError("V27d-r11 formal L3 must remain disabled")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    for key in ("manifest_path", "cache_template", "checkpoint_template", "summary_template", "log_template", "smoke_path", "evaluation_template", "summary_path", "experiment_doc_path"):
        value = _text(artifacts, key, "artifacts")
        try:
            value.format(role="representation_fit", seed=42)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r11 artifacts.{key} contains an invalid placeholder") from exc

    protocol = _mapping(config.get("protocol"), "protocol")
    expected_protocol = {
        "direct_baseline": "V27d-r6-liquid_treatment",
        "unique_variable": "full_stable8_causal_vector_shape_projection",
        "trainable_component": "zero_initialized_shape_projection_only",
        "frozen_liquid_v2_core": True,
        "frozen_base_readout": True,
        "frozen_verifier_head": True,
        "classification_logits_input_to_branch": False,
        "softmax_input_to_branch": False,
        "label_identity_input_to_branch": False,
        "formal_l3_enabled": False,
        "gru_used": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r11 protocol.{key} differs from the registered contract")

    config["config_path"] = path
    config["config_sha256"] = sha256_file(path)
    config["config_identity_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}}
    )
    return config


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    scope = resolve_project_path(root, R11_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r11 artifact escaped its experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R11_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    config_path = source_path(root, source["r6_config_path"], must_exist=True)
    manifest_path = source_path(root, source["r6_manifest_path"], must_exist=True)
    cache_paths = {
        role: source_path(root, source["r6_cache_template"].format(role=role), must_exist=True)
        for role in R11_ROLES
    }
    checkpoint_paths = {
        str(seed): source_path(root, source["r6_checkpoint_template"].format(seed=seed), must_exist=True)
        for seed in R11_SEEDS
    }
    return {
        "schema_version": R11_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R11_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "r6_config_path": relative_project_path(root, config_path),
            "r6_config_sha256": sha256_file(config_path),
            "r6_manifest_path": relative_project_path(root, manifest_path),
            "r6_manifest_sha256": sha256_file(manifest_path),
            "cache_paths": {role: relative_project_path(root, path) for role, path in cache_paths.items()},
            "cache_file_sha256s": {role: sha256_file(path) for role, path in cache_paths.items()},
            "cache_payload_sha256s": dict(source["expected_cache_payload_sha256s"]),
            "checkpoint_paths": {seed: relative_project_path(root, path) for seed, path in checkpoint_paths.items()},
            "checkpoint_file_sha256s": {seed: sha256_file(path) for seed, path in checkpoint_paths.items()},
            "checkpoint_head_state_sha256s": dict(source["expected_checkpoint_head_state_sha256s"]),
        },
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "selection": dict(config["selection"]),
        "calibration": dict(config["calibration"]),
        "protocol": dict(config["protocol"]),
        "seeds": list(R11_SEEDS),
        "roles": list(R11_ROLES),
        "code_hashes": package_code_hashes(root),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r11 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27d-r11 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r11 sealed JSON must be an object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r11 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r11 sealed JSON schema differs")
    return dict(value)


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    manifest = build_manifest(root, config)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json_if_identical(path, manifest)
    return config, path, sha256_file(path)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R11_MANIFEST_SCHEMA_VERSION)
    expected = build_manifest(root, config)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r11 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = ["R11_CACHE_SCHEMA_VERSION", "R11_CHECKPOINT_SCHEMA_VERSION", "R11_CONFIG_SCHEMA_VERSION", "R11_DEFAULT_CONFIG", "R11_EXPERIMENT_ID", "R11_MANIFEST_SCHEMA_VERSION", "R11_REPORT_SCHEMA_VERSION", "R11_ROLES", "R11_SCHEMA_VERSION", "R11_SEEDS", "R11_SHAPE_FEATURE_DIM", "R11_SMOKE_SCHEMA_VERSION", "R11_STABLE8_DIM", "artifact_path", "build_manifest", "load_config", "load_context", "package_code_hashes", "prepare_manifest", "read_sealed_json", "source_path", "write_sealed_json_if_identical"]