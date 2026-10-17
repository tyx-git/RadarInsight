"""Immutable protocol for V27d-r22 radial-rate scale ablation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r17.protocol import (
    canonical_sha256,
    project_path,
    relative_path,
    sha256_file,
    write_torch_if_identical,
)
from v27.V27d.r20.protocol import (
    R20_ARMS,
    load_config as load_r20_config,
    load_manifest as load_r20_manifest,
)

R22_EXPERIMENT_ID = "V27d-r22"
R22_SCHEMA_VERSION = "RADAR-LN3-V27d-r22-radial-rate-scale-ablation-v1"
R22_CONFIG_SCHEMA = "RADAR-LN3-V27d-r22-config-v1"
R22_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r22-manifest-v1"
R22_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r22-l2-checkpoint-v1"
R22_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r22-training-summary-v1"
R22_REPORT_SCHEMA = "RADAR-LN3-V27d-r22-trained-comparison-v1"
R22_SMOKE_SCHEMA = "RADAR-LN3-V27d-r22-contract-smoke-v1"
R22_ROOT = Path("src-v3") / "v27" / "V27d" / "r22"
R22_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R22_ARMS = R20_ARMS
R22_RATE_SCALE = 10.0


class R22Config:
    """Validated r22 configuration and its immutable r20 parent."""

    def __init__(self, payload: dict[str, Any], path: Path, r20: Any) -> None:
        self.payload = payload
        self.path = path
        self.r20 = r20
        self.r19 = r20.r19
        self.r18 = r20.r18
        self.r17 = r20.r17
        self.sha256 = sha256_file(path)
        self.identity_sha256 = canonical_sha256(payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"r22 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r22 {name} must be non-empty text")
    return value.strip()


def _same_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"r22 {name} differs from r20; only radial rate scale may change")


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R22Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path or R22_ROOT / "config" / "V27d-r22-radial-rate-scale.json",
        must_exist=True,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("r22 configuration is not an object")
    payload = dict(raw)
    if payload.get("schema_version") != R22_CONFIG_SCHEMA:
        raise ValueError("r22 configuration schema differs")
    if payload.get("experiment_id") != R22_EXPERIMENT_ID:
        raise ValueError("r22 experiment identity differs")

    source = _mapping(payload.get("source"), "source")
    parent = _mapping(source.get("parent"), "source.parent")
    parent_config_text = _text(parent.get("r20_config_path"), "source.parent.r20_config_path")
    parent_config_path = project_path(root, parent_config_text, must_exist=True)
    if sha256_file(parent_config_path) != parent.get("r20_config_sha256"):
        raise ValueError("r22 r20 config hash differs")
    r20 = load_r20_config(root, parent_config_path)
    parent_manifest_text = _text(parent.get("r20_manifest_path"), "source.parent.r20_manifest_path")
    parent_manifest_path = project_path(root, parent_manifest_text, must_exist=True)
    if sha256_file(parent_manifest_path) != parent.get("r20_manifest_sha256"):
        raise ValueError("r22 r20 manifest hash differs")
    r20_manifest, _, _ = load_r20_manifest(root, r20, allow_code_drift=False)
    if r20_manifest.get("experiment_id") != "V27d-r20":
        raise ValueError("r22 direct parent is not V27d-r20")
    if parent_config_text != r20.path.relative_to(root).as_posix():
        raise ValueError("r22 parent config path is not canonical")

    r20_source = dict(r20.source)
    for key, expected in r20_source.items():
        if source.get(key) != expected:
            raise ValueError(f"r22 source.{key} differs from r20")
    payload["source"] = source

    data = _mapping(payload.get("data"), "data")
    r20_data = dict(r20.data)
    if set(data) != set(r20_data):
        raise ValueError("r22 data schema differs from r20")
    for key, expected in r20_data.items():
        if key == "radial_rate_scale_mps_per_s":
            continue
        if data.get(key) != expected:
            raise ValueError(f"r22 data.{key} differs from r20")
    if float(data.get("radial_rate_scale_mps_per_s", -1.0)) != R22_RATE_SCALE:
        raise ValueError("r22 radial rate scale must be exactly 10 m/s/s")
    if float(r20_data.get("radial_rate_scale_mps_per_s", -1.0)) != 100.0:
        raise ValueError("r20 parent radial rate scale is not the registered 100 m/s/s baseline")
    payload["data"] = data

    model = _mapping(payload.get("model"), "model")
    _same_mapping(model, r20.model, "model")
    payload["model"] = model
    training = _mapping(payload.get("training"), "training")
    _same_mapping(training, r20.training, "training")
    payload["training"] = training

    artifacts = _mapping(payload.get("artifacts"), "artifacts")
    for key in (
        "manifest_path",
        "smoke_path",
        "comparison_report_path",
        "checkpoint_template",
        "epoch_checkpoint_template",
        "training_summary_template",
    ):
        _text(artifacts.get(key), f"artifacts.{key}")
    payload["artifacts"] = artifacts
    return R22Config(payload, path, r20)


def artifact_path(
    project_root: str | Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    root = Path(project_root).resolve()
    if "{arm}" in template and arm not in R22_ARMS:
        raise ValueError("r22 artifact arm is invalid")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("r22 artifact seed is invalid")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("r22 artifact epoch is invalid")
    try:
        rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("r22 artifact template is invalid") from exc
    target = project_path(root, rendered)
    try:
        target.relative_to((root / R22_ROOT).resolve())
    except ValueError as exc:
        raise ValueError("r22 artifact escaped the experiment root") from exc
    return target


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("r22 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"r22 refuses to overwrite a different artifact: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _code_hashes(root: Path) -> dict[str, str]:
    scope = (root / R22_ROOT).resolve()
    paths = sorted(path for path in scope.rglob("*.py") if "__pycache__" not in path.parts)
    return {relative_path(root, path): sha256_file(path) for path in paths}


def build_manifest(project_root: str | Path, config: R22Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parent_manifest_path = project_path(root, config.source["parent"]["r20_manifest_path"], must_exist=True)
    parent_manifest, _, _ = load_r20_manifest(root, config.r20, allow_code_drift=False)
    return {
        "schema_version": R22_MANIFEST_SCHEMA,
        "experiment_id": R22_EXPERIMENT_ID,
        "variant": config["variant"],
        "config": {
            "path": relative_path(root, config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "direct_parent": {
            "experiment_id": "V27d-r20",
            "config_path": relative_path(root, project_path(root, config.source["parent"]["r20_config_path"], must_exist=True)),
            "config_sha256": sha256_file(project_path(root, config.source["parent"]["r20_config_path"], must_exist=True)),
            "manifest_path": relative_path(root, parent_manifest_path),
            "manifest_sha256": sha256_file(parent_manifest_path),
        },
        "source": {
            "dataset": config.source["dataset"],
            "r19_config_path": config.source["r19_config_path"],
            "r19_config_sha256": config.source["r19_config_sha256"],
            "r19_manifest_path": config.source["r19_manifest_path"],
            "r19_manifest_sha256": config.source["r19_manifest_sha256"],
            "r19_checkpoint_path": config.source["r19_checkpoint_path"],
            "r19_checkpoint_sha256": config.source["r19_checkpoint_sha256"],
            "r19_checkpoint_model_state_sha256": config.source["r19_checkpoint_model_state_sha256"],
            "r19_checkpoint_arm": config.source["r19_checkpoint_arm"],
            "r19_checkpoint_seed": int(config.source["r19_checkpoint_seed"]),
            "r19_checkpoint_epoch": int(config.source["r19_checkpoint_epoch"]),
        },
        "split": parent_manifest["split"],
        "parent_assignments": parent_manifest["parent_assignments"],
        "input_contract": {
            "copied_from_r20": True,
            "raw_radial_speed_scalar": "removed from ordinary TR encoder in both arms",
            "radial_sidecar_values": list(config.data["radial_value_names"]),
            "radial_sidecar_validity": list(config.data["radial_valid_names"]),
            "radial_abs_scale_mps": float(config.data["radial_abs_scale_mps"]),
            "radial_rate_scale_mps_per_s": float(config.data["radial_rate_scale_mps_per_s"]),
            "r20_parent_rate_scale_mps_per_s": 100.0,
            "positive_dt_boundary": bool(config.data["positive_dt_only_for_rate_and_consistency"]),
            "sidecar_clip_absolute": float(config.data["sidecar_clip_absolute"]),
        },
        "model": {
            "task": "l2",
            "class_names": list(R22_L2_CLASSES),
            "unknown_classifier_class": False,
            "structure": "identical_to_r20",
            "trainable_scope": "full_r20_scope; trajectory_heads_frozen",
        },
        "training_boundary": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "rd_payload_access_count": 0,
            "l3": "not_executed",
        },
        "code_hashes": _code_hashes(root),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R22Config, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_json_if_identical(path, seal(build_manifest(root, config)))
    return config, path, sha256_file(path)


def load_manifest(
    project_root: str | Path,
    config: R22Config,
    *,
    allow_code_drift: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    path = artifact_path(root, config.artifacts["manifest_path"])
    if not path.exists():
        raise FileNotFoundError(f"r22 manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r22 manifest is not an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("r22 manifest payload hash cannot be recomputed")
    expected = build_manifest(root, config)
    if body != expected:
        body_without_code = {key: value for key, value in body.items() if key != "code_hashes"}
        expected_without_code = {key: value for key, value in expected.items() if key != "code_hashes"}
        if not allow_code_drift or body_without_code != expected_without_code:
            raise ValueError("r22 manifest differs from current config, parent, split or code")
    return dict(payload), path, sha256_file(path)


__all__ = ["R22_ARMS", "R22_CHECKPOINT_SCHEMA", "R22Config", "R22_EXPERIMENT_ID", "R22_L2_CLASSES", "R22_MANIFEST_SCHEMA", "R22_RATE_SCALE", "R22_REPORT_SCHEMA", "R22_ROOT", "R22_SCHEMA_VERSION", "R22_SMOKE_SCHEMA", "R22_TRAINING_SUMMARY_SCHEMA", "artifact_path", "build_manifest", "canonical_sha256", "load_config", "load_manifest", "prepare_manifest", "project_path", "seal", "sha256_file", "write_json_if_identical", "write_torch_if_identical"]