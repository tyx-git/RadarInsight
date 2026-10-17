"""V27d-r15 的配置、来源绑定和实验产物契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


R15_EXPERIMENT_ID = "V27d-r15"
R15_CONFIG_SCHEMA = "RADAR-LN3-V27d-r15-v10-observation-increment-v3-config"
R15_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r15-manifest-v3"
R15_CACHE_SCHEMA = "RADAR-LN3-V27d-r15-observation-increment-cache-v3"
R15_CACHE_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r15-observation-increment-manifest-v3"
R15_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r15-checkpoint-v2"
R15_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r15-training-summary-v1"
R15_REPORT_SCHEMA = "RADAR-LN3-V27d-r15-evaluation-report-v1"
R15_SUMMARY_REPORT_SCHEMA = "RADAR-LN3-V27d-r15-two-seed-summary-v1"
R15_SMOKE_SCHEMA = "RADAR-LN3-V27d-r15-contract-smoke-v1"
R15_ARMS = ("control", "v10_observation_increment")
R15_ROLES = ("representation_fit", "representation_development", "calibration")
R15_CACHE_VIEWS = ("train", "validation", *R15_ROLES)
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


def artifact_path(
    project_root: str | Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    root = Path(project_root).resolve()
    try:
        rendered = str(template).format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid V27d-r15 artifact template: {template}") from exc
    target = resolve_project_path(root, rendered)
    scope = resolve_project_path(root, R15_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r15 artifact escaped its experiment directory") from exc
    return target


def project_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r15 JSON artifact is not an object: {path}")
    return dict(value)


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r15 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    sealed = seal(payload)
    text = json_text(sealed, indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V27d-r15 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d-r15 JSON payload hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError("V27d-r15 JSON schema differs")
    return value


def verify_sources(root: Path, config: R15Config) -> dict[str, str]:
    observed: dict[str, str] = {}
    paths = {
        "v27_config": config.source["v27_config_path"],
        "v27_manifest": config.source["v27_manifest_path"],
        "v27_l1_checkpoint": config.source["v27_l1_checkpoint_path"],
        "v27d_endpoint_manifest": config.source["v27d_endpoint_manifest_path"],
        "index": config.source["index_path"],
        "observation_increment_source": config.source["observation_increment_source"],
    }
    for key, value in paths.items():
        path = project_path(root, value, must_exist=True)
        observed[key] = sha256_file(path)
        expected = config.source["expected_hashes"].get(key)
        if expected is not None and observed[key] != expected:
            raise ValueError(f"V27d-r15 source hash differs for {key}")
    for role in R15_ROLES:
        path = project_path(
            root,
            config.source["role_course_cache_paths"][role],
            must_exist=True,
        )
        digest = sha256_file(path)
        observed[f"role_course_cache:{role}"] = digest
        if digest != config.source["expected_role_course_cache_hashes"][role]:
            raise ValueError(f"V27d-r15 role course cache hash differs: {role}")
    endpoint_manifest = read_json(project_path(root, paths["v27d_endpoint_manifest"], must_exist=True))
    tasks = endpoint_manifest.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("V27d-r15 endpoint manifest lacks tasks")
    for role, expected_file_hash in config.source["expected_endpoint_file_hashes"].items():
        entry = tasks.get(f"known_gate:{role}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"V27d-r15 endpoint manifest lacks known_gate/{role}")
        path = project_path(root, str(entry["path"]), must_exist=True)
        if sha256_file(path) != expected_file_hash or entry.get("sha256") != expected_file_hash:
            raise ValueError(f"V27d-r15 endpoint file hash differs: {role}")
    return observed


def build_manifest(root: Path, config: R15Config, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    code_paths = sorted(
        path
        for path in (root / R15_ROOT).glob("*.py")
        if "__pycache__" not in path.parts
    )
    code_hashes = {
        relative_project_path(root, path): sha256_file(path)
        for path in code_paths
    }
    return {
        "schema_version": R15_MANIFEST_SCHEMA,
        "experiment_id": R15_EXPERIMENT_ID,
        "variant": config.payload["variant"],
        "objective": config.payload["objective"],
        "config": {
            "path": relative_project_path(root, config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "source_hashes": dict(source_hashes),
        "code_hashes": code_hashes,
        "observation_increment": {
            "schema": "causal-tr-observation-increment-v1",
            "value_names": [
                "reported_speed_delta_scaled",
                "radial_speed_delta_scaled",
                "course_delta_scaled",
                "geometric_step_distance_scaled",
                "range_delta_scaled",
                "radial_range_direction_agreement",
                "radial_sign_flip",
                "consecutive_step_direction_cosine",
                "causal_path_straightness",
            ],
            "validity_names": [
                "reported_speed_delta_scaled_valid",
                "radial_speed_delta_scaled_valid",
                "course_delta_scaled_valid",
                "geometric_step_distance_scaled_valid",
                "range_delta_scaled_valid",
                "radial_range_direction_agreement_valid",
                "radial_sign_flip_valid",
                "consecutive_step_direction_cosine_valid",
                "causal_path_straightness_valid",
            ],
            "clip_absolute": 5.0,
            "mask_mode": "gating_only",
            "time_semantics": "per_report_change_not_per_second_rate",
            "same_second_events": "state_advances_and_zero_dt_is_not_read",
        },
        "roles": list(R15_ROLES),
        "cache_views": list(R15_CACHE_VIEWS),
        "arms": list(R15_ARMS),
        "unknown_is_classifier_class": False,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R15Context, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    source_hashes = verify_sources(root, config)
    payload = build_manifest(root, config, source_hashes)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_sealed_json_if_identical(path, payload)
    return R15Context(root, config, seal(payload), path, sha256_file(path)), path, sha256_file(path)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> R15Context:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    source_hashes = verify_sources(root, config)
    path = artifact_path(root, config.artifacts["manifest_path"])
    manifest = read_sealed_json(path, expected_schema=R15_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, source_hashes)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r15 manifest differs from configuration, sources or code")
    return R15Context(root, config, manifest, path, sha256_file(path))


__all__ = ["R15_ARMS", "R15_CACHE_MANIFEST_SCHEMA", "R15_CACHE_SCHEMA", "R15_CHECKPOINT_SCHEMA", "R15_CLASS_NAMES", "R15_CONFIG_SCHEMA", "R15Context", "R15_EXPERIMENT_ID", "R15_MANIFEST_SCHEMA", "R15_REPORT_SCHEMA", "R15_ROLES", "R15_SMOKE_SCHEMA", "R15_SUMMARY_REPORT_SCHEMA", "R15_SUMMARY_SCHEMA", "R15Config", "artifact_path", "build_manifest", "load_config", "load_context", "prepare_manifest", "project_path", "read_json", "read_sealed_json", "seal", "verify_sources", "write_sealed_json_if_identical"]