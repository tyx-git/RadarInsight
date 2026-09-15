"""Configuration and provenance contracts for V27d-r9.

The experiment consumes only the frozen V27a Main event logits carried by the
V27b-r5 cache and the immutable V18 Doppler-profile evidence.  All trainable
artifacts are owned by this directory.
"""

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


R9_EXPERIMENT_ID = "V27d-r9"
R9_SCHEMA_VERSION = "RADAR-LN3-V27d-r9-causal-doppler-profile-shape-v1"
R9_CONFIG_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-config"
R9_MANIFEST_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-manifest"
R9_CACHE_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-cache"
R9_CHECKPOINT_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-checkpoint"
R9_REPORT_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-evaluation"
R9_SUMMARY_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-summary"
R9_SMOKE_SCHEMA_VERSION = f"{R9_SCHEMA_VERSION}-smoke"
R9_DEFAULT_CONFIG = "src-v3/v27/V27d/r9/config/V27d-r9-causal-doppler-profile-shape.json"
R9_ROOT = "src-v3/v27/V27d/r9"
R9_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R9_SEEDS = (42, 44)
R9_PROFILE_BINS = 181

_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r9 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r9 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r9 {name}.{key} must be positive")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27d-r9 {name}.{key} must be non-negative")
    return int(value)


def _finite_float(mapping: Mapping[str, Any], key: str, name: str, *, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"V27d-r9 {name}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d-r9 {name}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27d-r9 {name}.{key} is invalid")
    return result


def _sha(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _text(mapping, key, name).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27d-r9 {name}.{key} must be a SHA-256 digest")
    return value


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R9_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if config.get("schema_version") != R9_CONFIG_SCHEMA_VERSION:
        raise ValueError("V27d-r9 configuration schema differs")
    if config.get("experiment_id") != R9_EXPERIMENT_ID:
        raise ValueError("V27d-r9 configuration experiment identity differs")
    if tuple(config.get("seeds", ())) != R9_SEEDS:
        raise ValueError("V27d-r9 seeds must be exactly 42 and 44")

    source = _mapping(config.get("source"), "source")
    source_keys = (
        "v27b_train_cache_path",
        "v27b_validation_cache_path",
        "raw_train_evidence_path",
        "raw_validation_evidence_path",
    )
    for key in source_keys:
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
    if _positive_int(model, "profile_bins", "model") != R9_PROFILE_BINS:
        raise ValueError("V27d-r9 profile_bins must be 181")
    _positive_int(model, "profile_projection_dim", "model")
    _positive_int(model, "hidden_dim", "model")
    _finite_float(model, "pairwise_delta_bound", "model", positive=True)

    training = _mapping(config.get("training"), "training")
    for key in ("batch_size", "epochs", "primary_epoch", "bootstrap_replicates"):
        _positive_int(training, key, "training")
    if int(training["primary_epoch"]) > int(training["epochs"]):
        raise ValueError("V27d-r9 primary_epoch exceeds epochs")
    _positive_int(training, "dataloader_seed", "training")
    _finite_float(training, "learning_rate", "training", positive=True)
    _finite_float(training, "weight_decay", "training")
    _finite_float(training, "gradient_clip_norm", "training", positive=True)
    device = _text(training, "device", "training").lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("V27d-r9 training.device must be cpu or cuda")

    promotion = _mapping(config.get("promotion"), "promotion")
    bird_recall = _finite_float(promotion, "minimum_bird_recall", "promotion")
    if not 0.0 <= bird_recall <= 1.0:
        raise ValueError("V27d-r9 minimum_bird_recall must be in [0,1]")
    _nonnegative_int(promotion, "maximum_new_errors", "promotion")
    _positive_int(promotion, "minimum_drone_bird_corrections", "promotion")
    _positive_int(promotion, "minimum_total_correct_gain", "promotion")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    artifact_keys = (
        "manifest_path",
        "cache_template",
        "checkpoint_template",
        "training_summary_template",
        "log_template",
        "smoke_path",
        "evaluation_template",
        "summary_path",
        "experiment_doc_path",
    )
    for key in artifact_keys:
        value = _text(artifacts, key, "artifacts")
        try:
            value.format(seed=42, epoch=1, split="train")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r9 artifacts.{key} contains an invalid placeholder") from exc

    protocol = _mapping(config.get("protocol"), "protocol")
    required_protocol = {
        "direct_baseline": "V27a-r3-main-epoch-020",
        "liquid_v2_backbone": "frozen",
        "unknown_training_exposure": 0,
        "formal_l3_enabled": False,
        "classification_logits_input_to_branch": False,
        "softmax_input_to_branch": False,
        "label_identity_input_to_branch": False,
        "gru_used": False,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r9 protocol.{key} differs from the registered contract")

    config["config_path"] = path
    config["config_sha256"] = sha256_file(path)
    config["config_identity_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key not in {"config_path", "config_sha256", "config_identity_sha256"}}
    )
    return config


def artifact_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    scope = resolve_project_path(root, R9_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r9 artifact escaped its experiment directory") from exc
    return target


def source_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R9_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _source_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    values = {
        "v27b_train_cache_file_sha256": sha256_file(source_path(root, source["v27b_train_cache_path"], must_exist=True)),
        "v27b_validation_cache_file_sha256": sha256_file(source_path(root, source["v27b_validation_cache_path"], must_exist=True)),
        "raw_train_evidence_sha256": sha256_file(source_path(root, source["raw_train_evidence_path"], must_exist=True)),
        "raw_validation_evidence_sha256": sha256_file(source_path(root, source["raw_validation_evidence_path"], must_exist=True)),
    }
    for name, expected_key in (
        ("v27b_train_cache_file_sha256", "expected_v27b_train_cache_file_sha256"),
        ("v27b_validation_cache_file_sha256", "expected_v27b_validation_cache_file_sha256"),
        ("raw_train_evidence_sha256", "expected_raw_train_evidence_sha256"),
        ("raw_validation_evidence_sha256", "expected_raw_validation_evidence_sha256"),
    ):
        if values[name] != source[expected_key]:
            raise ValueError(f"V27d-r9 source hash differs at {name}")
    return values


def build_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    file_hashes = _source_hashes(root, config)
    return {
        "schema_version": R9_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R9_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "v27b_train_cache_path": source["v27b_train_cache_path"],
            "v27b_validation_cache_path": source["v27b_validation_cache_path"],
            "raw_train_evidence_path": source["raw_train_evidence_path"],
            "raw_validation_evidence_path": source["raw_validation_evidence_path"],
            **file_hashes,
            "v27b_train_cache_payload_sha256": source["expected_v27b_train_cache_payload_sha256"],
            "v27b_validation_cache_payload_sha256": source["expected_v27b_validation_cache_payload_sha256"],
        },
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "promotion": dict(config["promotion"]),
        "protocol": dict(config["protocol"]),
        "code_hashes": package_code_hashes(root),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r9 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    atomic_write_text(target, json_text(_seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r9 sealed JSON must be an object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r9 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r9 sealed JSON schema differs")
    return dict(value)


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    manifest = build_manifest(root, config)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, manifest)
    return config, path, sha256_file(path)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R9_MANIFEST_SCHEMA_VERSION)
    expected = build_manifest(root, config)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r9 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = [
    "R9_CACHE_SCHEMA_VERSION",
    "R9_CHECKPOINT_SCHEMA_VERSION",
    "R9_CLASS_NAMES",
    "R9_CONFIG_SCHEMA_VERSION",
    "R9_DEFAULT_CONFIG",
    "R9_EXPERIMENT_ID",
    "R9_MANIFEST_SCHEMA_VERSION",
    "R9_PROFILE_BINS",
    "R9_REPORT_SCHEMA_VERSION",
    "R9_SCHEMA_VERSION",
    "R9_SEEDS",
    "R9_SMOKE_SCHEMA_VERSION",
    "R9_SUMMARY_SCHEMA_VERSION",
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
