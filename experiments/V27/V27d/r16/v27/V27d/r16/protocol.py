"""Configuration and provenance contracts for V27d-r16."""

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
from v27.v27_baseline.protocol import load_v27_config, verify_v27_data_contract


R16_EXPERIMENT_ID = "V27d-r16"
R16_CONFIG_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-config"
R16_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-manifest"
R16_SMOKE_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-smoke"
R16_REPORT_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-evaluation"
R16_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-summary"
R16_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-checkpoint"
R16_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r16-rd-frame-liquid-v1-training-summary"
R16_ROOT = Path("src-v3") / "v27" / "V27d" / "r16"
R16_ARMS = ("control", "treatment")
R16_SEEDS = (42, 44)
R16_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


@dataclass(frozen=True)
class R16Config:
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
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.payload["protocol"]


@dataclass(frozen=True)
class R16Context:
    project_root: Path
    config: R16Config
    baseline_config: Any
    r14_context: Any
    data_binding: dict[str, Any]
    source_hashes: dict[str, str]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r16 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d-r16 {name} must be non-empty text")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"V27d-r16 {name} must be a SHA-256 digest")
    return digest


def load_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R16Config:
    root = Path(project_root).resolve()
    candidate = (
        root / R16_ROOT / "config" / "V27d-r16-rd-frame-liquid.json"
        if config_path is None
        else Path(config_path)
    )
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = _mapping(raw, "configuration")
    if value.get("schema_version") != R16_CONFIG_SCHEMA:
        raise ValueError("V27d-r16 configuration schema differs")
    if value.get("experiment_id") != R16_EXPERIMENT_ID:
        raise ValueError("V27d-r16 configuration identity differs")
    if tuple(value.get("seeds", ())) != R16_SEEDS:
        raise ValueError("V27d-r16 seeds must be exactly 42 and 44")

    source = _mapping(value.get("source"), "source")
    paths = _mapping(source.get("paths"), "source.paths")
    expected = _mapping(source.get("expected_hashes"), "source.expected_hashes")
    if set(paths) != set(expected):
        raise ValueError("V27d-r16 source path/hash keys differ")
    for key in expected:
        expected[str(key)] = _sha(expected[key], f"source.expected_hashes.{key}")
    source["paths"] = paths
    source["expected_hashes"] = expected
    source["expected_data_registry_sha256"] = _sha(
        source.get("expected_data_registry_sha256"),
        "source.expected_data_registry_sha256",
    )
    value["source"] = source

    data = _mapping(value.get("data"), "data")
    if data.get("frame_time_source") != "canonical_rd_tensor.time_seconds":
        raise ValueError("V27d-r16 frame-time source differs")
    if data.get("frame_time_semantics") != "cpi_relative_elapsed_not_tr_wall_clock":
        raise ValueError("V27d-r16 frame-time semantics differ")
    if int(data.get("max_rd_frames_per_point", 0)) != 12:
        raise ValueError("V27d-r16 RD frame cap differs")
    if int(data.get("rd_height", 0)) != 21 or int(data.get("rd_width", 0)) != 384:
        raise ValueError("V27d-r16 RD shape differs")
    value["data"] = data

    model = _mapping(value.get("model"), "model")
    if model.get("control_aggregation") != "masked_mean":
        raise ValueError("V27d-r16 control aggregation differs")
    if model.get("treatment_aggregation") != "liquid_v2_frame_time_residual":
        raise ValueError("V27d-r16 treatment aggregation differs")
    if model.get("recurrent_cell") != "radar_ln3.liquid_v2.LiquidV2Cell":
        raise ValueError("V27d-r16 recurrent cell differs")
    if float(model.get("residual_scale", 0.0)) <= 0.0:
        raise ValueError("V27d-r16 residual scale must be positive")
    value["model"] = model

    training = _mapping(value.get("training"), "training")
    if int(training.get("epochs", 0)) <= 0 or int(training.get("primary_epoch", -1)) != int(training.get("epochs", -2)):
        raise ValueError("V27d-r16 primary_epoch must equal a positive epochs value")
    if int(training.get("batch_size", 0)) <= 0 or float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("V27d-r16 training dimensions/rate are invalid")
    if training.get("loss_name") != "parent_final_fixed_parent_count_ce" or training.get("loss_reduction") != "fixed_parent_count":
        raise ValueError("V27d-r16 loss contract differs")
    if training.get("freeze_existing_parameters") is not True:
        raise ValueError("V27d-r16 must freeze the existing control model")
    value["training"] = training

    protocol = _mapping(value.get("protocol"), "protocol")
    required = {
        "arms": list(R16_ARMS),
        "l2_classes": list(R16_CLASS_NAMES),
        "unknown_is_l2_classifier_class": False,
        "unknown_training_exposure": 0,
        "knownness_training": "not_in_scope",
        "quality_training": "not_in_scope",
        "l3": "not_in_scope",
        "test_payload_access_count": 0,
        "oof": "not_executed",
        "cross_source": "not_executed",
        "tr_event_liquid_policy": "unchanged_from_r14_control",
        "padding": "never_updates_outer_or_rd_state",
        "nonmonotonic_frame_policy": "masked_mean_and_zero_residual;diagnostic_count_recorded",
        "single_frame_policy": "masked_mean_and_zero_residual",
        "baseline_logits_input_to_recurrence": False,
    }
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise ValueError(f"V27d-r16 protocol.{key} differs")
    value["protocol"] = protocol

    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key, template in artifacts.items():
        _text(template, f"artifacts.{key}")
        try:
            str(template).format(arm="control", seed=42, epoch=1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27d-r16 artifacts.{key} has an invalid template") from exc
    value["artifacts"] = artifacts

    return R16Config(value, path, sha256_file(path), canonical_sha256(value))


def project_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    return resolve_project_path(Path(project_root).resolve(), value, must_exist=must_exist)


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
        raise ValueError(f"invalid V27d-r16 artifact template: {value}") from exc
    target = resolve_project_path(root, rendered, must_exist=must_exist)
    scope = resolve_project_path(root, R16_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27d-r16 artifact escaped its experiment directory") from exc
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d-r16 JSON artifact is not an object: {path}")
    return dict(value)


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d-r16 payload is already sealed")
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
            raise FileExistsError(f"refusing to replace a different V27d-r16 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"V27d-r16 sealed JSON hash differs: {path}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError(f"V27d-r16 sealed JSON schema differs: {path}")
    return value


def _tensor_state_hash(state: Mapping[str, Any]) -> str:
    parts: list[Any] = []
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            raise TypeError(f"V27d-r16 checkpoint state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((name, str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def _verify_r14_checkpoint(root: Path, path: Path, expected_hash: str, seed: int, manifest_hash: str) -> None:
    import torch

    if sha256_file(path) != expected_hash:
        raise ValueError(f"V27d-r16 r14 control checkpoint file hash differs: seed={seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r16 r14 control checkpoint is not a mapping")
    if (
        payload.get("experiment_id") != "V27d-r14"
        or payload.get("arm") != "control"
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", -1)) != 20
        or payload.get("r14_manifest_sha256") != manifest_hash
    ):
        raise ValueError(f"V27d-r16 r14 checkpoint identity differs: seed={seed}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != _tensor_state_hash(state):
        raise ValueError(f"V27d-r16 r14 checkpoint state hash differs: seed={seed}")


def verify_sources(root: Path, config: R16Config) -> tuple[Any, Any, dict[str, Any], dict[str, str]]:
    observed: dict[str, str] = {}
    for key, value in config.source["paths"].items():
        path = project_path(root, value, must_exist=True)
        observed[str(key)] = sha256_file(path)
        if observed[str(key)] != config.source["expected_hashes"][str(key)]:
            raise ValueError(f"V27d-r16 source hash differs for {key}")

    baseline_config = load_v27_config(root, config.source["paths"]["v27_config"])
    data_binding = verify_v27_data_contract(root, baseline_config)
    if data_binding["data_registry_sha256"] != config.source["expected_data_registry_sha256"]:
        raise ValueError("V27d-r16 data registry hash differs")
    if baseline_config.model.rd_frame_aggregation != "masked_mean":
        raise ValueError("V27d-r16 requires the frozen V27 masked-mean source model")

    r14_manifest = read_sealed_json(
        project_path(root, config.source["paths"]["r14_manifest"], must_exist=True)
    )
    if r14_manifest.get("experiment_id") != "V27d-r14" or r14_manifest.get("schema_version") != "RADAR-LN3-V27d-r14-density-weighted-v1-manifest":
        raise ValueError("V27d-r16 r14 manifest identity differs")
    from v27.V27d.r14.protocol import R14Context, load_config as load_r14_config

    r14_config = load_r14_config(
        root,
        config.source["paths"]["r14_config"],
    )
    r14_manifest_path = project_path(
        root,
        config.source["paths"]["r14_manifest"],
        must_exist=True,
    )
    r14_context = R14Context(
        project_root=root,
        config=r14_config,
        baseline_config=baseline_config,
        data_binding=dict(data_binding),
        source_hashes=observed,
        manifest=r14_manifest,
        manifest_path=r14_manifest_path,
        manifest_sha256=sha256_file(r14_manifest_path),
    )
    r14_manifest_hash = config.source["expected_hashes"]["r14_manifest"]
    for seed in R16_SEEDS:
        _verify_r14_checkpoint(
            root,
            project_path(root, config.source["paths"][f"r14_control_seed{seed}_checkpoint"], must_exist=True),
            config.source["expected_hashes"][f"r14_control_seed{seed}_checkpoint"],
            seed,
            r14_manifest_hash,
        )
    return baseline_config, r14_context, dict(data_binding), observed


def package_code_hashes(root: Path) -> dict[str, str]:
    package = root / R16_ROOT
    return {
        relative_project_path(root, path): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(
    root: Path,
    config: R16Config,
    baseline_config: Any,
    data_binding: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": R16_MANIFEST_SCHEMA,
        "experiment_id": R16_EXPERIMENT_ID,
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
        "r16_code_hashes": package_code_hashes(root),
        "model": dict(config.model),
        "data": dict(config.data),
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
) -> tuple[R16Context, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    baseline_config, r14_context, data_binding, source_hashes = verify_sources(root, config)
    payload = build_manifest(root, config, baseline_config, data_binding, source_hashes)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_sealed_json(path, payload)
    sealed = read_sealed_json(path, expected_schema=R16_MANIFEST_SCHEMA)
    return R16Context(root, config, baseline_config, r14_context, data_binding, source_hashes, sealed, path, sha256_file(path)), path, sha256_file(path)


def load_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R16Context:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    baseline_config, r14_context, data_binding, source_hashes = verify_sources(root, config)
    path = artifact_path(root, config.artifacts["manifest_path"], must_exist=True)
    manifest = read_sealed_json(path, expected_schema=R16_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, baseline_config, data_binding, source_hashes)
    actual = {key: item for key, item in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d-r16 manifest differs from configuration, sources or code")
    return R16Context(root, config, baseline_config, r14_context, data_binding, source_hashes, manifest, path, sha256_file(path))


__all__ = ["R16_ARMS", "R16_CHECKPOINT_SCHEMA", "R16_CLASS_NAMES", "R16_CONFIG_SCHEMA", "R16Context", "R16Config", "R16_EXPERIMENT_ID", "R16_MANIFEST_SCHEMA", "R16_REPORT_SCHEMA", "R16_ROOT", "R16_SEEDS", "R16_SMOKE_SCHEMA", "R16_SUMMARY_SCHEMA", "R16_TRAINING_SUMMARY_SCHEMA", "artifact_path", "build_manifest", "load_config", "load_context", "package_code_hashes", "prepare_manifest", "project_path", "read_json", "read_sealed_json", "verify_sources", "write_sealed_json", "write_sealed_json_if_identical"]