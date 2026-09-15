"""Configuration and provenance contracts for V27d-r10."""

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


R10_EXPERIMENT_ID = "V27d-r10"
R10_SCHEMA_VERSION = "RADAR-LN3-V27d-r10-compact-causal-doppler-shape-v1"
R10_CONFIG_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-config"
R10_MANIFEST_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-manifest"
R10_CACHE_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-cache"
R10_CHECKPOINT_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-checkpoint"
R10_REPORT_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-evaluation"
R10_SMOKE_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-smoke"
R10_SUMMARY_SCHEMA_VERSION = f"{R10_SCHEMA_VERSION}-summary"
R10_DEFAULT_CONFIG = "src-v3/v27/V27d/r10/config/V27d-r10-compact-causal-doppler-shape.json"
R10_ROOT = "src-v3/v27/V27d/r10"
R10_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R10_SEEDS = (42, 44)
R10_DESCRIPTOR_NAMES = (
    "profile_centroid_over_90mps",
    "profile_spread_over_90mps",
    "profile_entropy_over_log_181",
    "profile_signed_positive_negative_balance",
    "profile_top4_mass",
    "profile_scaled_rms",
    "intra_event_profile_cosine_mean",
    "intra_event_peak_drift_over_180_bins",
)
R10_DESCRIPTOR_DIM = len(R10_DESCRIPTOR_NAMES)
_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r10 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r10 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r10 {name}.{key} must be positive")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27d-r10 {name}.{key} must be non-negative")
    return int(value)


def _finite_float(mapping: Mapping[str, Any], key: str, name: str, *, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"V27d-r10 {name}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d-r10 {name}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27d-r10 {name}.{key} is invalid")
    return result


def _sha(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _text(mapping, key, name).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27d-r10 {name}.{key} must be a SHA-256 digest")
    return value


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R10_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if config.get("schema_version") != R10_CONFIG_SCHEMA_VERSION or config.get("experiment_id") != R10_EXPERIMENT_ID:
        raise ValueError("V27d-r10 configuration identity differs")
    if tuple(config.get("seeds", ())) != R10_SEEDS:
        raise ValueError("V27d-r10 seeds must be exactly 42 and 44")

    source = _mapping(config.get("source"), "source")
    for key in (
        "v27b_train_cache_path",
        "v27b_validation_cache_path",
        "raw_train_evidence_path",
        "raw_validation_evidence_path",
    ):
        _text(source, key, "source")
    for key in (
        "expected_v27b_train_cache_file_sha256",
        "expected_v27b_validation_cache_file_sha256",
        "expected_v27b_train_cache_payload_sha256",
        "expected_v27b_validation_cache_payload_sha256",
        "expected_raw_train_evidence_sha256",
        "expected_raw_validation_evidence_sha256",
    ):
        _sha(source, key, "source")

    model = _mapping(config.get("model"), "model")
    if _positive_int(model, "descriptor_dim", "model") != R10_DESCRIPTOR_DIM:
        raise ValueError("V27d-r10 descriptor_dim differs from the registered descriptor contract")
    for key in ("projection_dim", "hidden_dim"):
        _positive_int(model, key, "model")
    _finite_float(model, "pairwise_delta_bound", "model", positive=True)
    if model.get("descriptor_names") != list(R10_DESCRIPTOR_NAMES):
        raise ValueError("V27d-r10 descriptor names differ")

    training = _mapping(config.get("training"), "training")
    for key in ("dataloader_seed", "batch_size", "epochs", "primary_epoch", "bootstrap_replicates"):
        _positive_int(training, key, "training")
    if int(training["primary_epoch"]) > int(training["epochs"]):
        raise ValueError("V27d-r10 primary_epoch exceeds epochs")
    _finite_float(training, "learning_rate", "training", positive=True)
    _finite_float(training, "weight_decay", "training")
    _finite_float(training, "gradient_clip_norm", "training", positive=True)
    if _text(training, "device", "training").lower() not in {"cpu", "cuda"}:
        raise ValueError("V27d-r10 training.device must be cpu or cuda")
    if training.get("loss_name") != "parent_final_fixed_parent_count_ce" or training.get("loss_reduction") != "fixed_parent_count":
        raise ValueError("V27d-r10 loss contract differs")
    weights = training.get("class_weights")
    if not isinstance(weights, list) or len(weights) != 3 or any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0 for value in weights):
        raise ValueError("V27d-r10 class weights differ")

    promotion = _mapping(config.get("promotion"), "promotion")
    bird_recall = _finite_float(promotion, "minimum_bird_recall", "promotion")
    if not 0.0 <= bird_recall <= 1.0:
        raise ValueError("V27d-r10 minimum_bird_recall must be in [0,1]")
    _nonnegative_int(promotion, "maximum_new_errors", "promotion")
    _positive_int(promotion, "minimum_drone_bird_corrections", "promotion")
    _positive_int(promotion, "minimum_total_correct_gain", "promotion")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    for key in (
        "manifest_path",
        "cache_template",
        "checkpoint_template",
        "training_summary_template",
        "log_template",
        "smoke_path",
        "evaluation_template",
        "summary_path",
        "experiment_doc_path",
    ):
        value = _text(artifacts, key, "artifacts")
        try:
            value.format(seed=42, split="train")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r10 artifacts.{key} contains an invalid placeholder") from exc

    protocol = _mapping(config.get("protocol"), "protocol")
    expected_protocol = {
        "direct_baseline": "V27a-r3-main-epoch-020",
        "liquid_v2_backbone": "frozen",
        "unique_variable": "compact_causal_doppler_shape_descriptors",
        "envelope_profile_used": False,
        "classification_logits_input_to_branch": False,
        "softmax_input_to_branch": False,
        "label_identity_input_to_branch": False,
        "unknown_training_exposure": 0,
        "formal_l3_enabled": False,
        "gru_used": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r10 protocol.{key} differs from the registered contract")

    config["config_path"] = path
    config["config_sha256"] = sha256_file(path)
    config["config_identity_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}}
    )
    return config


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    scope = resolve_project_path(root, R10_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r10 artifact escaped its experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R10_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    paths = {
        "v27b_train_cache": source_path(root, source["v27b_train_cache_path"], must_exist=True),
        "v27b_validation_cache": source_path(root, source["v27b_validation_cache_path"], must_exist=True),
        "raw_train_evidence": source_path(root, source["raw_train_evidence_path"], must_exist=True),
        "raw_validation_evidence": source_path(root, source["raw_validation_evidence_path"], must_exist=True),
    }
    return {
        "schema_version": R10_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R10_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "paths": {name: relative_project_path(root, path) for name, path in paths.items()},
            "sha256": {name: sha256_file(path) for name, path in paths.items()},
            "v27b_train_cache_payload_sha256": source["expected_v27b_train_cache_payload_sha256"],
            "v27b_validation_cache_payload_sha256": source["expected_v27b_validation_cache_payload_sha256"],
        },
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "promotion": dict(config["promotion"]),
        "protocol": dict(config["protocol"]),
        "seeds": list(R10_SEEDS),
        "code_hashes": package_code_hashes(root),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r10 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    atomic_write_text(target, json_text(_seal(payload), indent=2))
    return target


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27d-r10 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r10 sealed JSON must be an object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r10 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r10 sealed JSON schema differs")
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
    manifest = read_sealed_json(path, expected_schema=R10_MANIFEST_SCHEMA_VERSION)
    expected = build_manifest(root, config)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r10 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = [
    "R10_CACHE_SCHEMA_VERSION",
    "R10_CHECKPOINT_SCHEMA_VERSION",
    "R10_CLASS_NAMES",
    "R10_CONFIG_SCHEMA_VERSION",
    "R10_DEFAULT_CONFIG",
    "R10_DESCRIPTOR_DIM",
    "R10_DESCRIPTOR_NAMES",
    "R10_EXPERIMENT_ID",
    "R10_MANIFEST_SCHEMA_VERSION",
    "R10_REPORT_SCHEMA_VERSION",
    "R10_SCHEMA_VERSION",
    "R10_SEEDS",
    "R10_SMOKE_SCHEMA_VERSION",
    "R10_SUMMARY_SCHEMA_VERSION",
    "artifact_path",
    "build_manifest",
    "load_config",
    "load_context",
    "package_code_hashes",
    "prepare_manifest",
    "read_sealed_json",
    "source_path",
    "write_sealed_json",
    "write_sealed_json_if_identical",
]
