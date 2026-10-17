"""Provenance and artifact contracts for V27d-r12."""

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
    verify_file_hash,
)


R12_EXPERIMENT_ID = "V27d-r12-causal-prefix-p25"
R12_SCHEMA_VERSION = "RADAR-LN3-V27d-r12-causal-prefix-p25-v1"
R12_CONFIG_SCHEMA_VERSION = f"{R12_SCHEMA_VERSION}-config"
R12_MANIFEST_SCHEMA_VERSION = f"{R12_SCHEMA_VERSION}-manifest"
R12_REPORT_SCHEMA_VERSION = f"{R12_SCHEMA_VERSION}-evaluation"
R12_SMOKE_SCHEMA_VERSION = f"{R12_SCHEMA_VERSION}-smoke"
R12_DEFAULT_CONFIG = "src-v3/v27/V27d/r12/config/V27d-r12-causal-prefix-p25.json"
R12_ROOT = "src-v3/v27/V27d/r12"
R12_ROLES = ("representation_fit", "representation_development", "calibration")
R12_SEEDS = (42, 44)
_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r12 {name} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r12 {name}.{key} must be non-empty text")
    return value.strip()


def _sha(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _text(mapping, key, name).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27d-r12 {name}.{key} must be a SHA-256 digest")
    return value


def _sha_map(
    mapping: Mapping[str, Any],
    key: str,
    name: str,
    keys: tuple[str, ...],
) -> dict[str, str]:
    raw = _mapping(mapping.get(key), f"{name}.{key}")
    if set(raw) != set(keys):
        raise ValueError(f"V27d-r12 {name}.{key} keys differ")
    return {item: _sha(raw, item, f"{name}.{key}") for item in keys}


def load_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / R12_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if (
        config.get("schema_version") != R12_CONFIG_SCHEMA_VERSION
        or config.get("experiment_id") != R12_EXPERIMENT_ID
    ):
        raise ValueError("V27d-r12 configuration identity differs")
    if tuple(config.get("seeds", ())) != R12_SEEDS:
        raise ValueError("V27d-r12 seeds must be exactly 42 and 44")

    source = _mapping(config.get("source"), "source")
    for key in (
        "r6_config_path",
        "r6_manifest_path",
        "r6_cache_template",
        "r6_checkpoint_template",
    ):
        _text(source, key, "source")
    _sha(source, "expected_r6_config_sha256", "source")
    _sha(source, "expected_r6_manifest_sha256", "source")
    _sha_map(source, "expected_cache_file_sha256s", "source", R12_ROLES)
    _sha_map(source, "expected_cache_payload_sha256s", "source", R12_ROLES)
    _sha_map(
        source,
        "expected_checkpoint_file_sha256s",
        "source",
        tuple(str(seed) for seed in R12_SEEDS),
    )
    _sha_map(
        source,
        "expected_checkpoint_head_state_sha256s",
        "source",
        tuple(str(seed) for seed in R12_SEEDS),
    )

    aggregation = _mapping(config.get("aggregation"), "aggregation")
    quantile = aggregation.get("quantile")
    if isinstance(quantile, bool):
        raise ValueError("V27d-r12 aggregation.quantile must be numeric")
    try:
        quantile_value = float(quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("V27d-r12 aggregation.quantile must be numeric") from exc
    if not math.isfinite(quantile_value) or not 0.0 < quantile_value < 1.0:
        raise ValueError("V27d-r12 aggregation.quantile must be in (0,1)")
    if aggregation.get("method") != "torch_linear_quantile_over_eligible_causal_prefixes":
        raise ValueError("V27d-r12 aggregation method differs")

    selection = _mapping(config.get("selection"), "selection")
    maximum_loss = selection.get("max_additional_bird_rejections_per_pool")
    if isinstance(maximum_loss, bool) or not isinstance(maximum_loss, int) or maximum_loss < 0:
        raise ValueError("V27d-r12 selection Bird rejection budget differs")
    minimum_recall = selection.get("minimum_bird_recall")
    if isinstance(minimum_recall, bool):
        raise ValueError("V27d-r12 selection.minimum_bird_recall must be numeric")
    try:
        minimum_recall_value = float(minimum_recall)
    except (TypeError, ValueError) as exc:
        raise ValueError("V27d-r12 selection.minimum_bird_recall must be numeric") from exc
    if not 0.0 < minimum_recall_value <= 1.0:
        raise ValueError("V27d-r12 selection.minimum_bird_recall is invalid")

    calibration = _mapping(config.get("calibration"), "calibration")
    support = calibration.get("formal_unknown_support_required")
    if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
        raise ValueError("V27d-r12 formal Unknown support is invalid")
    if calibration.get("formal_l3_enabled") is not False:
        raise ValueError("V27d-r12 formal L3 must remain disabled")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    for key in (
        "manifest_path",
        "smoke_path",
        "evaluation_template",
        "log_template",
        "summary_path",
        "experiment_doc_path",
    ):
        value = _text(artifacts, key, "artifacts")
        try:
            value.format(seed=42, role="calibration")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r12 artifacts.{key} contains an invalid placeholder") from exc

    protocol = _mapping(config.get("protocol"), "protocol")
    expected_protocol = {
        "direct_baseline": "V27d-r6-liquid_treatment",
        "unique_variable": "eligible_causal_prefix_score_lower_quartile_aggregation",
        "liquid_v2_frozen": True,
        "knownness_head_frozen": True,
        "classification_endpoints_frozen": True,
        "classification_logits_input_to_aggregator": False,
        "softmax_input_to_aggregator": False,
        "label_identity_input_to_aggregator": False,
        "formal_l3_enabled": False,
        "unknown_is_classifier_class": False,
        "training_performed": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r12 protocol.{key} differs from the registered contract")

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
    scope = resolve_project_path(root, R12_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r12 artifact escaped its experiment directory") from exc
    return target


def source_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def package_code_hashes(project_root: Path) -> dict[str, str]:
    package_root = project_root / R12_ROOT
    return {
        relative_project_path(project_root, path): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    r6_config = source_path(root, source["r6_config_path"], must_exist=True)
    r6_manifest = source_path(root, source["r6_manifest_path"], must_exist=True)
    cache_paths = {
        role: source_path(root, source["r6_cache_template"].format(role=role), must_exist=True)
        for role in R12_ROLES
    }
    checkpoint_paths = {
        str(seed): source_path(root, source["r6_checkpoint_template"].format(seed=seed), must_exist=True)
        for seed in R12_SEEDS
    }
    return {
        "schema_version": R12_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R12_EXPERIMENT_ID,
        "config": {
            "path": relative_project_path(root, config["config_path"]),
            "sha256": config["config_sha256"],
            "identity_sha256": config["config_identity_sha256"],
        },
        "source": {
            "r6_config_path": relative_project_path(root, r6_config),
            "r6_config_sha256": sha256_file(r6_config),
            "r6_manifest_path": relative_project_path(root, r6_manifest),
            "r6_manifest_sha256": sha256_file(r6_manifest),
            "cache_paths": {role: relative_project_path(root, path) for role, path in cache_paths.items()},
            "cache_file_sha256s": {role: sha256_file(path) for role, path in cache_paths.items()},
            "cache_payload_sha256s": dict(source["expected_cache_payload_sha256s"]),
            "checkpoint_paths": {seed: relative_project_path(root, path) for seed, path in checkpoint_paths.items()},
            "checkpoint_file_sha256s": {seed: sha256_file(path) for seed, path in checkpoint_paths.items()},
            "checkpoint_head_state_sha256s": dict(source["expected_checkpoint_head_state_sha256s"]),
        },
        "aggregation": dict(config["aggregation"]),
        "selection": dict(config["selection"]),
        "calibration": dict(config["calibration"]),
        "protocol": dict(config["protocol"]),
        "roles": list(R12_ROLES),
        "seeds": list(R12_SEEDS),
        "code_hashes": package_code_hashes(root),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r12 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27d-r12 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r12 sealed JSON must be an object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r12 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r12 sealed JSON schema differs")
    return dict(value)


def prepare_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    source = config["source"]
    r6_config = source_path(root, source["r6_config_path"], must_exist=True)
    r6_manifest = source_path(root, source["r6_manifest_path"], must_exist=True)
    verify_file_hash(r6_config, source["expected_r6_config_sha256"], field="V27d-r12 r6 config")
    verify_file_hash(r6_manifest, source["expected_r6_manifest_sha256"], field="V27d-r12 r6 manifest")
    manifest = build_manifest(root, config)
    path = artifact_path(root, config["artifacts"]["manifest_path"])
    write_sealed_json_if_identical(path, manifest)
    return config, path, sha256_file(path)


def load_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config["artifacts"]["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R12_MANIFEST_SCHEMA_VERSION)
    expected = build_manifest(root, config)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r12 manifest differs from configuration, sources or code")
    return config, {**manifest, "manifest_path": path, "manifest_sha256": sha256_file(path)}


__all__ = ["R12_CONFIG_SCHEMA_VERSION", "R12_DEFAULT_CONFIG", "R12_EXPERIMENT_ID", "R12_MANIFEST_SCHEMA_VERSION", "R12_REPORT_SCHEMA_VERSION", "R12_ROLES", "R12_ROOT", "R12_SCHEMA_VERSION", "R12_SEEDS", "R12_SMOKE_SCHEMA_VERSION", "artifact_path", "build_manifest", "load_config", "load_context", "package_code_hashes", "prepare_manifest", "read_sealed_json", "source_path", "write_sealed_json_if_identical"]