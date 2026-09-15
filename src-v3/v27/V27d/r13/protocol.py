"""Configuration and provenance contracts for V27d-r13."""

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


R13_EXPERIMENT_ID = "V27d-r13"
R13_SCHEMA_VERSION = "RADAR-LN3-V27d-r13-raw-envelope-profile-shape-v1"
R13_CONFIG_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-config"
R13_MANIFEST_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-manifest"
R13_CACHE_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-cache"
R13_CHECKPOINT_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-checkpoint"
R13_REPORT_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-evaluation"
R13_SUMMARY_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-summary"
R13_SMOKE_SCHEMA_VERSION = f"{R13_SCHEMA_VERSION}-smoke"
R13_DEFAULT_CONFIG = "src-v3/v27/V27d/r13/config/V27d-r13-raw-envelope-profile-shape.json"
R13_ROOT = "src-v3/v27/V27d/r13"
R13_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R13_SEEDS = (42, 44)
R13_PROFILE_BINS = 64
R13_PROFILE_AXIS_HZ = tuple(
    300.0 + (2000.0 - 300.0) * index / (R13_PROFILE_BINS - 1)
    for index in range(R13_PROFILE_BINS)
)

_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r13 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r13 {name}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d-r13 {name}.{key} must be positive")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27d-r13 {name}.{key} must be non-negative")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    name: str,
    *,
    positive: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"V27d-r13 {name}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d-r13 {name}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27d-r13 {name}.{key} is invalid")
    return result


def _sha(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _text(mapping, key, name).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27d-r13 {name}.{key} must be a SHA-256 digest")
    return value


def load_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R13_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if (
        config.get("schema_version") != R13_CONFIG_SCHEMA_VERSION
        or config.get("experiment_id") != R13_EXPERIMENT_ID
    ):
        raise ValueError("V27d-r13 configuration identity differs")
    if tuple(config.get("seeds", ())) != R13_SEEDS:
        raise ValueError("V27d-r13 seeds must be exactly 42 and 44")

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
    if _positive_int(model, "profile_bins", "model") != R13_PROFILE_BINS:
        raise ValueError("V27d-r13 profile_bins must be 64")
    for key in ("profile_projection_dim", "hidden_dim"):
        _positive_int(model, key, "model")
    _finite_float(model, "pairwise_delta_bound", "model", positive=True)

    training = _mapping(config.get("training"), "training")
    for key in ("batch_size", "epochs", "primary_epoch", "bootstrap_replicates"):
        _positive_int(training, key, "training")
    if int(training["primary_epoch"]) > int(training["epochs"]):
        raise ValueError("V27d-r13 primary_epoch exceeds epochs")
    _positive_int(training, "dataloader_seed", "training")
    _finite_float(training, "learning_rate", "training", positive=True)
    _finite_float(training, "weight_decay", "training")
    _finite_float(training, "gradient_clip_norm", "training", positive=True)
    if _text(training, "device", "training").lower() not in {"cpu", "cuda"}:
        raise ValueError("V27d-r13 training.device must be cpu or cuda")

    promotion = _mapping(config.get("promotion"), "promotion")
    recall = _finite_float(promotion, "minimum_bird_recall", "promotion")
    if not 0.0 <= recall <= 1.0:
        raise ValueError("V27d-r13 minimum_bird_recall must be in [0,1]")
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
            value.format(seed=42, epoch=1, split="train")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r13 artifacts.{key} contains an invalid placeholder") from exc

    protocol = _mapping(config.get("protocol"), "protocol")
    required_protocol = {
        "direct_baseline": "V27a-r3-main-epoch-020",
        "liquid_v2_backbone": "frozen",
        "unique_variable": "raw_envelope_modulation_profile_shape",
        "profile_source": "V18 envelope_modulation_profiles",
        "envelope_profile_used": True,
        "residual_doppler_profile_used": False,
        "classification_logits_input_to_branch": False,
        "softmax_input_to_branch": False,
        "label_identity_input_to_branch": False,
        "unknown_training_exposure": 0,
        "formal_l3_enabled": False,
        "gru_used": False,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r13 protocol.{key} differs from the registered contract")

    config["config_path"] = path
    config["config_sha256"] = sha256_file(path)
    config["config_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in config.items()
            if key not in {"config_path", "config_sha256", "config_identity_sha256"}
        }
    )
    return config


def artifact_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value, must_exist=must_exist)
    scope = resolve_project_path(root, R13_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r13 artifact escaped its experiment directory") from exc
    return target


def source_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R13_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _source_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    source = config["source"]
    values = {
        "v27b_train_cache_file_sha256": sha256_file(
            source_path(root, source["v27b_train_cache_path"], must_exist=True)
        ),
        "v27b_validation_cache_file_sha256": sha256_file(
            source_path(root, source["v27b_validation_cache_path"], must_exist=True)
        ),
        "raw_train_evidence_sha256": sha256_file(
            source_path(root, source["raw_train_evidence_path"], must_exist=True)
        ),
        "raw_validation_evidence_sha256": sha256_file(
            source_path(root, source["raw_validation_evidence_path"], must_exist=True)
        ),
    }
    expected = {
        "v27b_train_cache_file_sha256": "expected_v27b_train_cache_file_sha256",
        "v27b_validation_cache_file_sha256": "expected_v27b_validation_cache_file_sha256",
        "raw_train_evidence_sha256": "expected_raw_train_evidence_sha256",
        "raw_validation_evidence_sha256": "expected_raw_validation_evidence_sha256",
    }
    for name, key in expected.items():
        if values[name] != source[key]:
            raise ValueError(f"V27d-r13 source hash differs at {name}")
    return values


def build_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    return {
        "schema_version": R13_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R13_EXPERIMENT_ID,
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
            **_source_hashes(root, config),
            "v27b_train_cache_payload_sha256": source["expected_v27b_train_cache_payload_sha256"],
            "v27b_validation_cache_payload_sha256": source["expected_v27b_validation_cache_payload_sha256"],
        },
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "promotion": dict(config["promotion"]),
        "protocol": dict(config["protocol"]),
        "code_hashes": package_code_hashes(root),
        "seeds": list(R13_SEEDS),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r13 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    atomic_write_text(target, json_text(_seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r13 sealed JSON must be an object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r13 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r13 sealed JSON schema differs")
    return dict(value)


def prepare_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    manifest = build_manifest(root, config)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json(path, manifest)
    return config, path, sha256_file(path)


def load_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R13_MANIFEST_SCHEMA_VERSION)
    expected = build_manifest(root, config)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r13 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = [
    "R13_CACHE_SCHEMA_VERSION",
    "R13_CHECKPOINT_SCHEMA_VERSION",
    "R13_CLASS_NAMES",
    "R13_CONFIG_SCHEMA_VERSION",
    "R13_DEFAULT_CONFIG",
    "R13_EXPERIMENT_ID",
    "R13_MANIFEST_SCHEMA_VERSION",
    "R13_PROFILE_AXIS_HZ",
    "R13_PROFILE_BINS",
    "R13_REPORT_SCHEMA_VERSION",
    "R13_ROOT",
    "R13_SCHEMA_VERSION",
    "R13_SEEDS",
    "R13_SMOKE_SCHEMA_VERSION",
    "R13_SUMMARY_SCHEMA_VERSION",
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
