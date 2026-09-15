"""Immutable protocol for the V27d-r19 SNR-channel ablation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r18.protocol import load_config as load_r18_config
from v27.V27d.r18.protocol import build_source_stratified_split
from v27.V27d.r17.protocol import (
    R17Config,
    canonical_sha256,
    project_path,
    relative_path,
    sha256_file,
    write_torch_if_identical as _write_r17_torch_if_identical,
)

R19_EXPERIMENT_ID = "V27d-r19"
R19_SCHEMA_VERSION = "RADAR-LN3-V27d-r19-snr-channel-ablation-v1"
R19_CONFIG_SCHEMA = "RADAR-LN3-V27d-r19-config-v1"
R19_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r19-manifest-v1"
R19_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r19-l2-checkpoint-v1"
R19_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r19-training-summary-v1"
R19_REPORT_SCHEMA = "RADAR-LN3-V27d-r19-trained-comparison-v1"
R19_SMOKE_SCHEMA = "RADAR-LN3-V27d-r19-contract-smoke-v1"
R19_ROOT = Path("src-v3") / "v27" / "V27d" / "r19"
R19_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R19_ARMS = ("snr_missing_only", "snr_channel_removed")
R19_SNR_VALUE_INDEX = 10
R19_SNR_MISSING_INDEX = 7
R19_RADIAL_VALUE_INDEX = 1
R19_RADIAL_MISSING_INDEX = 1


class R19Config:
    def __init__(self, payload: dict[str, Any], path: Path, r18: Any) -> None:
        self.payload = payload
        self.path = path
        self.r18 = r18
        self.r17: R17Config = r18.r17
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
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r19 {name} must be non-empty text")
    return value.strip()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"r19 {name} must be an object")
    return dict(value)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R19Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path or R19_ROOT / "config" / "V27d-r19-snr-channel-ablation.json",
        must_exist=True,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("r19 configuration is not an object")
    payload = dict(raw)
    if payload.get("schema_version") != R19_CONFIG_SCHEMA or payload.get("experiment_id") != R19_EXPERIMENT_ID:
        raise ValueError("r19 configuration identity/schema differs")
    source = _mapping(payload.get("source"), "source")
    r18_path = _text(source.get("r18_config_path"), "source.r18_config_path")
    r18 = load_r18_config(root, r18_path)
    r18_config_path = project_path(root, r18_path, must_exist=True)
    if sha256_file(r18_config_path) != source.get("r18_config_sha256"):
        raise ValueError("r19 r18 config hash differs")
    r18_manifest_path = project_path(root, source["r18_manifest_path"], must_exist=True)
    if sha256_file(r18_manifest_path) != source.get("r18_manifest_sha256"):
        raise ValueError("r19 r18 manifest hash differs")
    r18_manifest = json.loads(r18_manifest_path.read_text(encoding="utf-8"))
    r18_manifest_body = dict(r18_manifest)
    declared = r18_manifest_body.pop("payload_sha256", None)
    if declared != canonical_sha256(r18_manifest_body):
        raise ValueError("r19 r18 manifest payload hash differs")
    if r18_manifest_body.get("schema_version") != "RADAR-LN3-V27d-r18-manifest-v3":
        raise ValueError("r19 direct r18 manifest schema differs")
    current_split = build_source_stratified_split(root, r18)
    if current_split["assignments"] != r18_manifest_body.get("parent_assignments"):
        raise ValueError("r19 r18 parent split differs from its sealed manifest")
    payload["source"] = source
    data = _mapping(payload.get("data"), "data")
    if data.get("dataset") != "OriData/excel/TR-only-processed-r17":
        raise ValueError("r19 dataset differs")
    if data.get("split_protocol") != "r18 sealed source_suffix+label parent split":
        raise ValueError("r19 split protocol differs")
    if int(data.get("snr_value_index", -1)) != R19_SNR_VALUE_INDEX or int(data.get("snr_missing_index", -1)) != R19_SNR_MISSING_INDEX:
        raise ValueError("r19 SNR indices differ from TR schema")
    if int(data.get("radial_speed_value_index", -1)) != R19_RADIAL_VALUE_INDEX:
        raise ValueError("r19 radial speed index differs")
    payload["data"] = data
    training = _mapping(payload.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("r19 seeds are invalid")
    for key in ("epochs", "batch_size", "gradient_clip_norm"):
        if float(training.get(key, 0)) <= 0.0:
            raise ValueError(f"r19 training.{key} is invalid")
    if training.get("loss_name") != r18.training["loss_name"] or training.get("loss_reduction") != r18.training["loss_reduction"]:
        raise ValueError("r19 loss differs from r18")
    payload["training"] = training
    artifacts = _mapping(payload.get("artifacts"), "artifacts")
    for key in ("manifest_path", "smoke_path", "comparison_report_path", "checkpoint_template", "epoch_checkpoint_template", "training_summary_template"):
        _text(artifacts.get(key), f"artifacts.{key}")
    payload["artifacts"] = artifacts
    return R19Config(payload, path, r18)


def artifact_path(project_root: str | Path, template: str, *, arm: str | None = None, seed: int | None = None, epoch: int | None = None) -> Path:
    root = Path(project_root).resolve()
    if "{arm}" in template and arm not in set(R19_ARMS):
        raise ValueError("r19 artifact arm is invalid")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("r19 artifact seed is invalid")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("r19 artifact epoch is invalid")
    target = project_path(root, template.format(arm=arm, seed=seed, epoch=epoch))
    try:
        target.relative_to((root / R19_ROOT).resolve())
    except ValueError as exc:
        raise ValueError("r19 artifact escaped experiment root") from exc
    return target


def _code_hashes(root: Path) -> dict[str, str]:
    files = sorted(path for path in (root / R19_ROOT).glob("*.py") if "__pycache__" not in path.parts)
    return {relative_path(root, path): sha256_file(path) for path in files}


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("r19 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"r19 refuses to overwrite a different artifact: {target}")
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


def write_torch_if_identical(path: str | Path, payload: Any) -> None:
    _write_r17_torch_if_identical(path, payload)


def build_manifest(root: str | Path, config: R19Config) -> dict[str, Any]:
    project_root = Path(root).resolve()
    split = build_source_stratified_split(project_root, config.r18)
    r18_manifest_path = project_path(project_root, config.source["r18_manifest_path"], must_exist=True)
    return {
        "schema_version": R19_MANIFEST_SCHEMA,
        "experiment_id": R19_EXPERIMENT_ID,
        "variant": config["variant"],
        "config": {"path": relative_path(project_root, config.path), "sha256": config.sha256, "identity_sha256": config.identity_sha256},
        "source": {"dataset": config.source["dataset"], "r18_config_path": config.source["r18_config_path"], "r18_config_sha256": config.r18.sha256, "r18_manifest_path": relative_path(project_root, r18_manifest_path), "r18_manifest_sha256": sha256_file(r18_manifest_path), "frozen_l2_checkpoint_sha256": config.r17.source["r23_checkpoint_sha256"]},
        "split": {key: value for key, value in split.items() if key != "assignments"},
        "parent_assignments": split["assignments"],
        "input_contract": {"snr_control": "snr_db missing value with snr_db_missing=1", "snr_ablation": "snr_db value=0 and snr_db_missing=0", "untrusted_excel_noise": "excluded", "radial_speed": "retained in both arms", "radial_derived_observation_increment": "retained in both arms"},
        "model": {"frozen_l2": "V27c-r23 robust_control seed42 epoch1", "class_names": list(R19_L2_CLASSES), "main_variable": "SNR channel value+missingness removal"},
        "training_boundary": {"unknown_training_exposure": 0, "test_payload_access_count": 0, "rd_payload_access_count": 0, "l3": "not_executed"},
        "code_hashes": _code_hashes(project_root),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R19Config, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_json_if_identical(path, seal(build_manifest(root, config)))
    return config, path, sha256_file(path)


def load_manifest(
    project_root: str | Path,
    config: R19Config,
    *,
    allow_code_drift: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    path = artifact_path(root, config.artifacts["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r19 manifest is not an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("r19 manifest payload hash cannot be recomputed")
    expected = build_manifest(root, config)
    if body != expected:
        body_without_code = {key: value for key, value in body.items() if key != "code_hashes"}
        expected_without_code = {key: value for key, value in expected.items() if key != "code_hashes"}
        if not allow_code_drift or body_without_code != expected_without_code:
            raise ValueError("r19 manifest differs from current config/source/split/code")
    return dict(payload), path, sha256_file(path)


__all__ = ["R19_CHECKPOINT_SCHEMA", "R19Config", "R19_EXPERIMENT_ID", "R19_L2_CLASSES", "R19_MANIFEST_SCHEMA", "R19_REPORT_SCHEMA", "R19_ROOT", "R19_SCHEMA_VERSION", "R19_SMOKE_SCHEMA", "R19_TRAINING_SUMMARY_SCHEMA", "R19_ARMS", "R19_SNR_VALUE_INDEX", "R19_SNR_MISSING_INDEX", "R19_RADIAL_VALUE_INDEX", "artifact_path", "build_manifest", "canonical_sha256", "load_config", "load_manifest", "prepare_manifest", "project_path", "relative_path", "seal", "sha256_file", "write_json_if_identical", "write_torch_if_identical"]
