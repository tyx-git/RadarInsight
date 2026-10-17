"""Provenance and artifact contracts for the V27d-r14 paired L2 experiment.

The experiment changes one Liquid V2 time-policy field.  It deliberately
reuses the current V27 known-class data contract and never opens Unknown or
test payloads during training or known-class evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file


R14_EXPERIMENT_ID = "V27d-r14"
R14_CONFIG_SCHEMA = "RADAR-LN3-V27d-r14-density-weighted-v1-config"
R14_ROOT = Path("src-v3") / "v27" / "V27d" / "r14"
R14_DEFAULT_CONFIG = R14_ROOT / "config" / "V27d-r14-density-weighted.json"
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
























__all__ = ["R14_CLASS_NAMES", "R14_CONFIG_SCHEMA", "R14Context", "R14_DEFAULT_CONFIG", "R14_EXPERIMENT_ID", "R14_MODES", "R14_SEEDS", "R14Config", "artifact_path", "load_config", "project_path"]