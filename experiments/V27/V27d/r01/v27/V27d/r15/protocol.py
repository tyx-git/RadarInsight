"""V27d-r15 的配置、来源绑定和实验产物契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file


R15_EXPERIMENT_ID = "V27d-r15"
R15_CONFIG_SCHEMA = "RADAR-LN3-V27d-r15-v10-observation-increment-v3-config"
R15_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r15-checkpoint-v2"
R15_ARMS = ("control", "v10_observation_increment")
R15_ROLES = ("representation_fit", "representation_development", "calibration")
R15_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
R15_ROOT = Path("src-v3") / "v27" / "V27d" / "r15"


@dataclass(frozen=True)
class R15Config:
    """保留经过校验的原始配置，避免重复定义嵌套配置对象。"""

    payload: dict[str, Any]
    path: Path
    sha256: str
    identity_sha256: str

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def data(self) -> Mapping[str, Any]:
        return self.payload["data"]

    @property
    def model(self) -> Mapping[str, Any]:
        return self.payload["model"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.payload["evaluation"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.payload["protocol"]


@dataclass(frozen=True)
class R15Context:
    project_root: Path
    config: R15Config
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r15 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r15 {name} must be non-empty text")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"V27d-r15 {name} must be a SHA-256 digest")
    return digest


def load_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R15Config:
    root = Path(project_root).resolve()
    candidate = (
        root / R15_ROOT / "config" / "V27d-r15-v10-observation-increment-l1-v3.json"
        if config_path is None
        else Path(config_path)
    )
    path = resolve_project_path(root, candidate, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r15 configuration root must be an object")
    value = dict(payload)
    if value.get("schema_version") != R15_CONFIG_SCHEMA:
        raise ValueError("V27d-r15 configuration schema differs")
    if value.get("experiment_id") != R15_EXPERIMENT_ID:
        raise ValueError("V27d-r15 configuration identity differs")

    source = _mapping(value.get("source"), "source")
    expected_hashes = _mapping(source.get("expected_hashes"), "source.expected_hashes")
    for key, digest in expected_hashes.items():
        expected_hashes[str(key)] = _sha(digest, f"source.expected_hashes.{key}")
    source["expected_hashes"] = expected_hashes
    role_cache_paths = _mapping(source.get("role_course_cache_paths"), "source.role_course_cache_paths")
    role_cache_hashes = _mapping(
        source.get("expected_role_course_cache_hashes"),
        "source.expected_role_course_cache_hashes",
    )
    for role in R15_ROLES:
        role_cache_paths[role] = _text(
            role_cache_paths.get(role),
            f"source.role_course_cache_paths.{role}",
        )
        role_cache_hashes[role] = _sha(
            role_cache_hashes.get(role),
            f"source.expected_role_course_cache_hashes.{role}",
        )
    source["role_course_cache_paths"] = role_cache_paths
    source["expected_role_course_cache_hashes"] = role_cache_hashes
    value["source"] = source

    data = _mapping(value.get("data"), "data")
    if data.get("observation_increment_schema") != "causal-tr-observation-increment-v1":
        raise ValueError("V27d-r15 must use the V10 observation-increment schema")
    if int(data.get("observation_increment_dim", -1)) != 9:
        raise ValueError("V27d-r15 observation-increment dimension differs")
    if data.get("observation_increment_mask_mode") != "gating_only":
        raise ValueError("V27d-r15 observation-increment mask mode differs")
    value["data"] = data

    model = _mapping(value.get("model"), "model")
    if model.get("injection_boundary") != "after_v27_fusion_before_liquid_v2_residual":
        raise ValueError("V27d-r15 injection boundary differs")
    if int(model.get("motion_output_dim", -1)) <= 0:
        raise ValueError("V27d-r15 residual output dimension is invalid")
    if any(bool(model.get(name)) for name in ("motion_projection_bias", "motion_output_bias")):
        raise ValueError("V27d-r15 V10 branch must be bias-free")
    value["model"] = model

    training = _mapping(value.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or seeds != [42]:
        raise ValueError("V27d-r15 training seeds are invalid")
    if int(training.get("primary_epoch", -1)) != int(training.get("epochs", -2)):
        raise ValueError("V27d-r15 primary epoch must be fixed at the configured epoch")
    value["training"] = training

    evaluation = _mapping(value.get("evaluation"), "evaluation")
    if evaluation.get("unknown_classifier_class") is not False:
        raise ValueError("V27d-r15 Unknown must not become an L1 classifier class")
    if int(evaluation.get("unknown_training_exposure", -1)) != 0:
        raise ValueError("V27d-r15 training cannot expose Unknown")
    value["evaluation"] = evaluation

    protocol = _mapping(value.get("protocol"), "protocol")
    required_protocol = {
        "v10_time_semantics": "values_are_changes_per_radar_report_not_physical_rates_per_second",
        "same_second_policy": "same-second_events_are_valid_observations_and_increment_state_advances_without_dt",
        "negative_dt_policy": "observation_increment_does_not_read_dt_or_dt_valid",
        "invalid_value_policy": "invalid_components_are_zero_and_validity_is_gating_only",
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d-r15 protocol.{key} differs from the V10 contract")
    value["protocol"] = protocol

    return R15Config(
        payload=value,
        path=path,
        sha256=sha256_file(path),
        identity_sha256=canonical_sha256(value),
    )






















__all__ = ["R15_ARMS", "R15_CHECKPOINT_SCHEMA", "R15_CLASS_NAMES", "R15_CONFIG_SCHEMA", "R15Context", "R15_EXPERIMENT_ID", "R15_ROLES", "R15Config", "load_config"]