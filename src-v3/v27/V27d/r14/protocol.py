"""Provenance and artifact contracts for the V27d-r14 paired L2 experiment.

The experiment changes one Liquid V2 time-policy field.  It deliberately
reuses the current V27 known-class data contract and never opens Unknown or
test payloads during training or known-class evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
from v27.v27_baseline.protocol import load_v27_config, verify_v27_data_contract


R14_EXPERIMENT_ID = "V27d-r14"
R14_CONFIG_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-config"
R14_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-manifest"
R14_SMOKE_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-smoke"
R14_REPORT_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-paired-report"
R14_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-summary"
R14_ROOT = Path("src-v3") / "v27" / "V27d" / "r14"
R14_DEFAULT_CONFIG = R14_ROOT / "config" / "V27d-r14-density-weighted.json"
R14_ARMS = ("control", "treatment")
R14_MODES = {"control": "bounded_observation", "treatment": "density_weighted"}
R14_SEEDS = (42, 44)
R14_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


@dataclass(frozen=True)
class R14Config:
    payload: dict[str, Any]
    path: Path
    sha256: str
    identity_sha256: str

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.payload["protocol"]


@dataclass(frozen=True)
class R14Context:
    project_root: Path
    config: R14Config
    baseline_config: Any
    data_binding: dict[str, Any]
    source_hashes: dict[str, str]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r14 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r14 {name} must be non-empty text")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"V27d-r14 {name} must be a SHA-256 digest")
    return digest


def load_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R14Config:
    root = Path(project_root).resolve()
    candidate = root / R14_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = _mapping(raw, "configuration")
    if value.get("schema_version") != R14_CONFIG_SCHEMA:
        raise ValueError("V27d-r14 configuration schema differs")
    if value.get("experiment_id") != R14_EXPERIMENT_ID:
        raise ValueError("V27d-r14 configuration identity differs")
    if tuple(value.get("seeds", ())) != R14_SEEDS:
        raise ValueError("V27d-r14 seeds must be exactly 42 and 44")

    source = _mapping(value.get("source"), "source")
    paths = _mapping(source.get("paths"), "source.paths")
    expected = _mapping(source.get("expected_hashes"), "source.expected_hashes")
    if set(paths) != set(expected):
        raise ValueError("V27d-r14 source path/hash keys differ")
    for key, item in expected.items():
        expected[str(key)] = _sha(item, f"source.expected_hashes.{key}")
    source["paths"] = paths
    source["expected_hashes"] = expected
    source["expected_data_registry_sha256"] = _sha(
        source.get("expected_data_registry_sha256"),
        "source.expected_data_registry_sha256",
    )
    value["source"] = source

    training = _mapping(value.get("training"), "training")
    if int(training.get("epochs", 0)) <= 0 or int(training.get("primary_epoch", -1)) != int(training.get("epochs", -2)):
        raise ValueError("V27d-r14 primary_epoch must equal a positive epochs value")
    if int(training.get("batch_size", 0)) <= 0 or float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("V27d-r14 training dimensions/rate are invalid")
    if training.get("loss_name") != "parent_final_fixed_parent_count_ce" or training.get("loss_reduction") != "fixed_parent_count":
        raise ValueError("V27d-r14 loss contract differs")
    value["training"] = training

    modes = _mapping(value.get("modes"), "modes")
    if modes != R14_MODES:
        raise ValueError("V27d-r14 control/treatment time-policy modes differ")
    value["modes"] = modes

    protocol = _mapping(value.get("protocol"), "protocol")
    required = {
        "unique_variable": "liquid_v2_invalid_dt_update_mode",
        "control_mode": "bounded_observation",
        "treatment_mode": "density_weighted",
        "l2_classes": list(R14_CLASS_NAMES),
        "unknown_is_l2_classifier_class": False,
        "unknown_training_exposure": 0,
        "knownness_training": "not_in_scope",
        "quality_training": "not_in_scope",
        "l3": "not_in_scope",
        "test_payload_access_count": 0,
        "oof": "not_executed",
        "cross_source": "not_executed",
        "padding": "never_updates_state",
        "positive_dt_policy": "unchanged_from_current_v27",
        "same_second_policy": "density_weighted_zero_ordinal",
    }
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise ValueError(f"V27d-r14 protocol.{key} differs")
    value["protocol"] = protocol

    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key, template in artifacts.items():
        _text(template, f"artifacts.{key}")
        try:
            str(template).format(arm="control", seed=42, epoch=1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r14 artifacts.{key} has an invalid template") from exc
    value["artifacts"] = artifacts

    identity = canonical_sha256(value)
    return R14Config(value, path, sha256_file(path), identity)


def artifact_path(
    project_root: str | Path,
    value: str | Path,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    must_exist: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    try:
        rendered = str(value).format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid V27d-r14 artifact template: {value}") from exc
    target = resolve_project_path(root, rendered, must_exist=must_exist)
    scope = resolve_project_path(root, R14_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r14 artifact escaped its experiment directory") from exc
    return target


def project_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r14 JSON artifact is not an object: {path}")
    return dict(value)


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r14 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    atomic_write_text(target, json_text(seal(payload), indent=2))
    return target


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27d-r14 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r14 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r14 sealed JSON schema differs")
    return value


def verify_sources(root: Path, config: R14Config) -> tuple[Any, dict[str, Any], dict[str, str]]:
    observed: dict[str, str] = {}
    for key, value in config.source["paths"].items():
        path = project_path(root, value, must_exist=True)
        observed[str(key)] = sha256_file(path)
        if observed[str(key)] != config.source["expected_hashes"][str(key)]:
            raise ValueError(f"V27d-r14 source hash differs for {key}")

    baseline_config = load_v27_config(root, config.source["paths"]["v27_config"])
    data_binding = verify_v27_data_contract(root, baseline_config)
    if data_binding["data_registry_sha256"] != config.source["expected_data_registry_sha256"]:
        raise ValueError("V27d-r14 data registry hash differs")

    for key in ("comparator_report_seed42", "comparator_report_seed43"):
        report = read_json(project_path(root, config.source["paths"][key], must_exist=True))
        if report.get("experiment_id") != "V27c-r23" or "frozen_v27a_main_reference" not in report.get("arms", {}):
            raise ValueError(f"V27d-r14 comparator report is not the frozen V27a reference: {key}")
        metrics = report["arms"]["frozen_v27a_main_reference"].get("parent_metrics")
        if not isinstance(metrics, Mapping) or int(metrics.get("support", 0)) != 223:
            raise ValueError(f"V27d-r14 comparator support differs: {key}")
        body = {name: item for name, item in report.items() if name != "payload_sha256"}
        if report.get("payload_sha256") != canonical_sha256(body):
            raise ValueError(f"V27d-r14 comparator payload hash differs: {key}")
    return baseline_config, dict(data_binding), observed


def package_code_hashes(root: Path) -> dict[str, str]:
    package = root / R14_ROOT
    return {
        relative_project_path(root, path): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(
    root: Path,
    config: R14Config,
    baseline_config: Any,
    data_binding: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    from v27.v27_baseline.protocol import V27_SOURCE_PATHS

    v27_code = {
        path.as_posix(): sha256_file(root / path)
        for path in V27_SOURCE_PATHS
    }
    return {
        "schema_version": R14_MANIFEST_SCHEMA,
        "experiment_id": R14_EXPERIMENT_ID,
        "variant": config.payload["variant"],
        "objective": config.payload["objective"],
        "config": {
            "path": relative_project_path(root, config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "source_hashes": dict(source_hashes),
        "data_registry_sha256": data_binding["data_registry_sha256"],
        "parent_split_sha256": data_binding["parent_split_sha256"],
        "feature_config_sha256": data_binding["feature_config_sha256"],
        "v27_config_sha256": baseline_config.config_sha256,
        "v27_manifest_reverification": "stale_against_current_source_preserved_as_read_only_reference",
        "v27_code_hashes": v27_code,
        "r14_code_hashes": package_code_hashes(root),
        "modes": dict(config.payload["modes"]),
        "training": dict(config.training),
        "protocol": dict(config.protocol),
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
            "oof": "not_executed",
            "cross_source": "not_executed",
        },
    }


def prepare_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[R14Context, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    baseline_config, data_binding, source_hashes = verify_sources(root, config)
    manifest = build_manifest(root, config, baseline_config, data_binding, source_hashes)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_sealed_json(path, manifest)
    sealed = read_sealed_json(path, expected_schema=R14_MANIFEST_SCHEMA)
    return (
        R14Context(root, config, baseline_config, data_binding, source_hashes, sealed, path, sha256_file(path)),
        path,
        sha256_file(path),
    )


def load_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R14Context:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    baseline_config, data_binding, source_hashes = verify_sources(root, config)
    path = artifact_path(root, config.artifacts["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R14_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, baseline_config, data_binding, source_hashes)
    actual = {key: item for key, item in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r14 manifest differs from configuration, sources or code")
    return R14Context(root, config, baseline_config, data_binding, source_hashes, manifest, path, sha256_file(path))


def mode_config(context: R14Context, *, mode: str, seed: int) -> Any:
    if mode not in set(R14_MODES.values()):
        raise ValueError(f"unsupported V27d-r14 Liquid mode: {mode}")
    if int(seed) not in R14_SEEDS:
        raise ValueError("V27d-r14 seed is not registered")
    training = replace(
        context.baseline_config.training,
        seeds=(int(seed),),
        dataloader_seed=int(seed),
    )
    model = replace(context.baseline_config.model, invalid_dt_update_mode=mode)
    return replace(context.baseline_config, model=model, training=training)


__all__ = [
    "R14_ARMS",
    "R14_CLASS_NAMES",
    "R14_CONFIG_SCHEMA",
    "R14Context",
    "R14_DEFAULT_CONFIG",
    "R14_EXPERIMENT_ID",
    "R14_MANIFEST_SCHEMA",
    "R14_MODES",
    "R14_REPORT_SCHEMA",
    "R14_SEEDS",
    "R14_SMOKE_SCHEMA",
    "R14_SUMMARY_SCHEMA",
    "R14Config",
    "artifact_path",
    "build_manifest",
    "load_config",
    "load_context",
    "mode_config",
    "package_code_hashes",
    "prepare_manifest",
    "project_path",
    "read_json",
    "read_sealed_json",
    "verify_sources",
    "write_sealed_json",
    "write_sealed_json_if_identical",
]
