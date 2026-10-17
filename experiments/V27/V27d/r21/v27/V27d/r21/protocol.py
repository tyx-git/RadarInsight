"""Immutable protocol for V27d-r21 adapter-only radial-sidecar training."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r20.protocol import (
    R20_RADIAL_VALID_NAMES,
    R20_RADIAL_VALUE_DIM,
    R20_RADIAL_VALUE_NAMES,
)
from v27.V27d.r20.protocol import load_config as load_r20_config
from v27.V27d.r20.protocol import load_manifest as load_r20_manifest
from v27.V27d.r17.protocol import (
    canonical_sha256,
    project_path,
    relative_path,
    sha256_file,
    write_torch_if_identical,
)

R21_EXPERIMENT_ID = "V27d-r21"
R21_SCHEMA_VERSION = "RADAR-LN3-V27d-r21-radial-adapter-only-v1"
R21_CONFIG_SCHEMA = "RADAR-LN3-V27d-r21-config-v1"
R21_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r21-manifest-v1"
R21_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r21-l2-checkpoint-v1"
R21_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r21-training-summary-v1"
R21_REPORT_SCHEMA = "RADAR-LN3-V27d-r21-trained-comparison-v1"
R21_SMOKE_SCHEMA = "RADAR-LN3-V27d-r21-contract-smoke-v1"
R21_ROOT = Path("src-v3") / "v27" / "V27d" / "r21"
R21_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R21_ARMS = ("zero_control", "radial_decomposed")


class R21Config:
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
        raise ValueError(f"r21 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r21 {name} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"r21 {name} must be a positive integer")
    return int(value)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R21Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path or R21_ROOT / "config" / "V27d-r21-radial-adapter-only.json",
        must_exist=True,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("r21 configuration is not an object")
    payload = dict(raw)
    if payload.get("schema_version") != R21_CONFIG_SCHEMA:
        raise ValueError("r21 configuration schema differs")
    if payload.get("experiment_id") != R21_EXPERIMENT_ID:
        raise ValueError("r21 experiment identity differs")

    source = _mapping(payload.get("source"), "source")
    r20_path_text = _text(source.get("r20_config_path"), "source.r20_config_path")
    r20 = load_r20_config(root, r20_path_text)
    r20_path = project_path(root, r20_path_text, must_exist=True)
    if sha256_file(r20_path) != source.get("r20_config_sha256"):
        raise ValueError("r21 r20 config hash differs")
    r20_manifest_text = _text(source.get("r20_manifest_path"), "source.r20_manifest_path")
    r20_manifest_path = project_path(root, r20_manifest_text, must_exist=True)
    if sha256_file(r20_manifest_path) != source.get("r20_manifest_sha256"):
        raise ValueError("r21 r20 manifest hash differs")
    r20_manifest, _, _ = load_r20_manifest(root, r20, allow_code_drift=True)
    if r20_manifest.get("experiment_id") != "V27d-r20":
        raise ValueError("r21 direct parent is not V27d-r20")

    source_checkpoint_text = _text(
        source.get("r20_source_checkpoint_path"), "source.r20_source_checkpoint_path"
    )
    source_checkpoint = project_path(root, source_checkpoint_text, must_exist=True)
    if sha256_file(source_checkpoint) != source.get("r20_source_checkpoint_sha256"):
        raise ValueError("r21 source checkpoint hash differs")
    if source.get("r20_source_checkpoint_arm") != "zero_control":
        raise ValueError("r21 source checkpoint arm differs")
    if isinstance(source.get("r20_source_checkpoint_seed"), bool) or int(source.get("r20_source_checkpoint_seed", -1)) != 42:
        raise ValueError("r21 source checkpoint seed differs")
    if isinstance(source.get("r20_source_checkpoint_epoch"), bool) or int(source.get("r20_source_checkpoint_epoch", -1)) != 19:
        raise ValueError("r21 source checkpoint epoch differs")
    payload["source"] = source

    data = _mapping(payload.get("data"), "data")
    if data.get("dataset") != r20.data.get("dataset"):
        raise ValueError("r21 dataset differs from r20")
    if data.get("split_protocol") != r20.data.get("split_protocol"):
        raise ValueError("r21 split protocol differs from r20")
    if data.get("radial_value_names") != list(R20_RADIAL_VALUE_NAMES):
        raise ValueError("r21 radial value names differ from r20")
    if data.get("radial_valid_names") != list(R20_RADIAL_VALID_NAMES):
        raise ValueError("r21 radial validity names differ from r20")
    if _positive_int(data.get("radial_sidecar_dim"), "data.radial_sidecar_dim") != R20_RADIAL_VALUE_DIM:
        raise ValueError("r21 radial sidecar dimension differs from r20")
    if data.get("positive_dt_only_for_rate_and_consistency") is not True:
        raise ValueError("r21 dt boundary differs from r20")
    payload["data"] = data

    model = _mapping(payload.get("model"), "model")
    for key in ("radial_projection_dim", "radial_adapter_dim"):
        if int(model.get(key, -1)) != int(r20.model[key]):
            raise ValueError(f"r21 model.{key} differs from r20")
    for key in ("bias_free", "output_zero_initialization", "validity_mask_is_model_input"):
        if model.get(key) != r20.model[key]:
            raise ValueError(f"r21 model.{key} differs from r20")
    if model.get("injection") != r20.model["injection"]:
        raise ValueError("r21 model injection boundary differs from r20")
    if model.get("trainable_scope") != "radial_adapter_only":
        raise ValueError("r21 trainable scope is not adapter-only")
    payload["model"] = model

    training = _mapping(payload.get("training"), "training")
    for key in (
        "seeds",
        "batch_size",
        "epochs",
        "gradient_clip_norm",
        "l2_learning_rate",
        "weight_decay",
        "loss_name",
        "loss_reduction",
        "optimizer",
        "device",
        "checkpoint_selection",
    ):
        if training.get(key) != r20.training.get(key):
            raise ValueError(f"r21 training.{key} differs from r20")
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
    return R21Config(payload, path, r20)


def artifact_path(
    project_root: str | Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    root = Path(project_root).resolve()
    if "{arm}" in template and arm not in R21_ARMS:
        raise ValueError("r21 artifact arm is invalid")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("r21 artifact seed is invalid")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("r21 artifact epoch is invalid")
    try:
        rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("r21 artifact template is invalid") from exc
    target = project_path(root, rendered)
    try:
        target.relative_to((root / R21_ROOT).resolve())
    except ValueError as exc:
        raise ValueError("r21 artifact escaped the experiment root") from exc
    return target


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("r21 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"r21 refuses to overwrite a different artifact: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
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
    scope = (root / R21_ROOT).resolve()
    paths = sorted(path for path in scope.rglob("*.py") if "__pycache__" not in path.parts)
    return {relative_path(root, path): sha256_file(path) for path in paths}


def build_manifest(project_root: str | Path, config: R21Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    r20_manifest_path = project_path(root, config.source["r20_manifest_path"], must_exist=True)
    r20_manifest, _, _ = load_r20_manifest(root, config.r20, allow_code_drift=True)
    return {
        "schema_version": R21_MANIFEST_SCHEMA,
        "experiment_id": R21_EXPERIMENT_ID,
        "variant": config["variant"],
        "config": {
            "path": relative_path(root, config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "source": {
            "dataset": config.source["dataset"],
            "r20_config_path": config.source["r20_config_path"],
            "r20_config_sha256": config.r20.sha256,
            "r20_manifest_path": relative_path(root, r20_manifest_path),
            "r20_manifest_sha256": sha256_file(r20_manifest_path),
            "r20_source_checkpoint_path": config.source["r20_source_checkpoint_path"],
            "r20_source_checkpoint_sha256": config.source["r20_source_checkpoint_sha256"],
            "r20_source_checkpoint_model_state_sha256": config.source["r20_source_checkpoint_model_state_sha256"],
            "r20_source_checkpoint_arm": config.source["r20_source_checkpoint_arm"],
            "r20_source_checkpoint_seed": int(config.source["r20_source_checkpoint_seed"]),
            "r20_source_checkpoint_epoch": int(config.source["r20_source_checkpoint_epoch"]),
        },
        "split": r20_manifest["split"],
        "parent_assignments": r20_manifest["parent_assignments"],
        "input_contract": {
            "copied_from_r20": True,
            "raw_radial_scalar_removed": True,
            "radial_sidecar_values": list(R20_RADIAL_VALUE_NAMES),
            "radial_sidecar_validity": list(R20_RADIAL_VALID_NAMES),
            "positive_dt_boundary": True,
        },
        "model": {
            "task": "l2",
            "class_names": list(R21_L2_CLASSES),
            "unknown_classifier_class": False,
            "structure": "identical_to_r20",
            "trainable_scope": "radial_adapter_only",
            "frozen_scope": "V27 Liquid V2 L2 base, fusion, TR encoder, classifier, trajectory heads",
        },
        "training_boundary": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "rd_payload_access_count": 0,
            "l3": "not_executed",
        },
        "code_hashes": _code_hashes(root),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R21Config, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_json_if_identical(path, seal(build_manifest(root, config)))
    return config, path, sha256_file(path)


def load_manifest(
    project_root: str | Path,
    config: R21Config,
    *,
    allow_code_drift: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    path = artifact_path(root, config.artifacts["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r21 manifest is not an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("r21 manifest payload hash cannot be recomputed")
    expected = build_manifest(root, config)
    if body != expected:
        body_without_code = {key: value for key, value in body.items() if key != "code_hashes"}
        expected_without_code = {key: value for key, value in expected.items() if key != "code_hashes"}
        if not allow_code_drift or body_without_code != expected_without_code:
            raise ValueError("r21 manifest differs from current config, source, split or code")
    return dict(payload), path, sha256_file(path)


__all__ = ["R21_ARMS", "R21_CHECKPOINT_SCHEMA", "R21Config", "R21_EXPERIMENT_ID", "R21_L2_CLASSES", "R21_MANIFEST_SCHEMA", "R21_REPORT_SCHEMA", "R21_ROOT", "R21_SCHEMA_VERSION", "R21_SMOKE_SCHEMA", "R21_TRAINING_SUMMARY_SCHEMA", "artifact_path", "build_manifest", "canonical_sha256", "load_config", "load_manifest", "prepare_manifest", "project_path", "seal", "sha256_file", "write_json_if_identical", "write_torch_if_identical"]