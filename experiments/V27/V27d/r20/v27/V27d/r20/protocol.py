"""Immutable protocol for V27d-r20 radial-speed decomposition."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r18.protocol import build_source_stratified_split
from v27.V27d.r19.protocol import (
    load_config as load_r19_config,
    load_manifest as load_r19_manifest,
)
from v27.V27d.r17.protocol import (
    canonical_sha256,
    project_path,
    relative_path,
    sha256_file,
    write_torch_if_identical,
)

R20_EXPERIMENT_ID = "V27d-r20"
R20_SCHEMA_VERSION = "RADAR-LN3-V27d-r20-radial-speed-decomposition-v1"
R20_CONFIG_SCHEMA = "RADAR-LN3-V27d-r20-config-v1"
R20_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r20-manifest-v1"
R20_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r20-l2-checkpoint-v1"
R20_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r20-training-summary-v1"
R20_REPORT_SCHEMA = "RADAR-LN3-V27d-r20-trained-comparison-v1"
R20_SMOKE_SCHEMA = "RADAR-LN3-V27d-r20-contract-smoke-v1"
R20_ROOT = Path("src-v3") / "v27" / "V27d" / "r20"
R20_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R20_ARMS = ("zero_control", "radial_decomposed")
R20_RADIAL_VALUE_NAMES = (
    "abs_radial_speed_scaled",
    "radial_speed_sign",
    "radial_speed_rate_scaled",
    "radial_range_direction_consistency",
)
R20_RADIAL_VALID_NAMES = tuple(f"{name}_valid" for name in R20_RADIAL_VALUE_NAMES)
R20_RADIAL_VALUE_DIM = len(R20_RADIAL_VALUE_NAMES)


class R20Config:
    """Validated r20 configuration with its immutable r19 parent config."""

    def __init__(self, payload: dict[str, Any], path: Path, r19: Any) -> None:
        self.payload = payload
        self.path = path
        self.r19 = r19
        self.r18 = r19.r18
        self.r17 = r19.r17
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
        raise ValueError(f"r20 {name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r20 {name} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"r20 {name} must be a positive integer")
    return int(value)


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"r20 {name} must be numeric") from exc
    if result <= 0.0 or result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"r20 {name} must be finite and positive")
    return result


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R20Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path or R20_ROOT / "config" / "V27d-r20-radial-speed-decomposition.json",
        must_exist=True,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("r20 configuration is not an object")
    payload = dict(raw)
    if payload.get("schema_version") != R20_CONFIG_SCHEMA:
        raise ValueError("r20 configuration schema differs")
    if payload.get("experiment_id") != R20_EXPERIMENT_ID:
        raise ValueError("r20 experiment identity differs")

    source = _mapping(payload.get("source"), "source")
    r19_path_text = _text(source.get("r19_config_path"), "source.r19_config_path")
    r19 = load_r19_config(root, r19_path_text)
    r19_path = project_path(root, r19_path_text, must_exist=True)
    if sha256_file(r19_path) != source.get("r19_config_sha256"):
        raise ValueError("r20 r19 config hash differs")
    r19_manifest_text = _text(source.get("r19_manifest_path"), "source.r19_manifest_path")
    r19_manifest_path = project_path(root, r19_manifest_text, must_exist=True)
    if sha256_file(r19_manifest_path) != source.get("r19_manifest_sha256"):
        raise ValueError("r20 r19 manifest hash differs")
    r19_manifest, _, _ = load_r19_manifest(root, r19, allow_code_drift=True)
    if r19_manifest.get("experiment_id") != "V27d-r19":
        raise ValueError("r20 direct parent is not V27d-r19")
    r19_checkpoint_text = _text(
        source.get("r19_checkpoint_path"), "source.r19_checkpoint_path"
    )
    r19_checkpoint_path = project_path(root, r19_checkpoint_text, must_exist=True)
    if sha256_file(r19_checkpoint_path) != source.get("r19_checkpoint_sha256"):
        raise ValueError("r20 direct r19 checkpoint hash differs")
    if source.get("r19_checkpoint_arm") != "snr_missing_only":
        raise ValueError("r20 direct r19 checkpoint arm differs")
    if isinstance(source.get("r19_checkpoint_seed"), bool) or int(source.get("r19_checkpoint_seed", -1)) != 42:
        raise ValueError("r20 direct r19 checkpoint seed differs")
    if isinstance(source.get("r19_checkpoint_epoch"), bool) or int(source.get("r19_checkpoint_epoch", -1)) != 20:
        raise ValueError("r20 direct r19 checkpoint epoch differs")
    if _text(source.get("dataset"), "source.dataset") != "OriData/excel/TR-only-processed-r17":
        raise ValueError("r20 dataset source differs")
    payload["source"] = source

    data = _mapping(payload.get("data"), "data")
    if _text(data.get("split_protocol"), "data.split_protocol") != "r18 sealed source_suffix+label parent split":
        raise ValueError("r20 split protocol differs")
    if data.get("dataset") != "OriData/excel/TR-only-processed-r17":
        raise ValueError("r20 data dataset differs")
    if data.get("radial_scalar_removed") is not True:
        raise ValueError("r20 must remove the raw radial scalar from the TR encoder")
    if data.get("positive_dt_only_for_rate_and_consistency") is not True:
        raise ValueError("r20 positive-dt boundary is not enabled")
    if data.get("radial_value_names") != list(R20_RADIAL_VALUE_NAMES):
        raise ValueError("r20 radial sidecar value names differ")
    if data.get("radial_valid_names") != list(R20_RADIAL_VALID_NAMES):
        raise ValueError("r20 radial sidecar validity names differ")
    if _positive_int(data.get("radial_sidecar_dim"), "data.radial_sidecar_dim") != R20_RADIAL_VALUE_DIM:
        raise ValueError("r20 radial sidecar dimension differs")
    if _positive_int(data.get("radial_sidecar_input_dim"), "data.radial_sidecar_input_dim") != 2 * R20_RADIAL_VALUE_DIM:
        raise ValueError("r20 radial sidecar input dimension differs")
    _finite_positive(data.get("radial_abs_scale_mps"), "data.radial_abs_scale_mps")
    _finite_positive(data.get("radial_rate_scale_mps_per_s"), "data.radial_rate_scale_mps_per_s")
    clip = _finite_positive(data.get("sidecar_clip_absolute"), "data.sidecar_clip_absolute")
    if clip > 5.0:
        raise ValueError("r20 sidecar clip is outside the registered bound")
    if data.get("untrusted_excel_noise_policy") != r19.data.get("untrusted_noise_policy"):
        raise ValueError("r20 SNR/noise policy differs from r19")
    payload["data"] = data

    model = _mapping(payload.get("model"), "model")
    if _positive_int(model.get("radial_projection_dim"), "model.radial_projection_dim") <= 0:
        raise ValueError("r20 radial projection dimension is invalid")
    if _positive_int(model.get("radial_adapter_dim"), "model.radial_adapter_dim") <= 0:
        raise ValueError("r20 radial adapter dimension is invalid")
    if model.get("output_zero_initialization") is not True or model.get("bias_free") is not True:
        raise ValueError("r20 adapter must be zero-initialized and bias-free")
    if model.get("injection") != "post_fusion_pre_liquid":
        raise ValueError("r20 injection boundary differs")
    payload["model"] = model

    training = _mapping(payload.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds):
        raise ValueError("r20 training seeds are invalid")
    for key in ("epochs", "batch_size", "gradient_clip_norm"):
        _finite_positive(training.get(key), f"training.{key}")
    for key in ("loss_name", "loss_reduction", "optimizer", "checkpoint_selection"):
        if training.get(key) != r19.training.get(key):
            raise ValueError(f"r20 training.{key} differs from r19")
    for key in ("batch_size", "epochs", "gradient_clip_norm", "l2_learning_rate", "weight_decay", "device"):
        if training.get(key) != r19.training.get(key):
            raise ValueError(f"r20 training.{key} differs from r19")
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
    return R20Config(payload, path, r19)


def artifact_path(
    project_root: str | Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    root = Path(project_root).resolve()
    if "{arm}" in template and arm not in R20_ARMS:
        raise ValueError("r20 artifact arm is invalid")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("r20 artifact seed is invalid")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("r20 artifact epoch is invalid")
    try:
        rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("r20 artifact template is invalid") from exc
    target = project_path(root, rendered)
    try:
        target.relative_to((root / R20_ROOT).resolve())
    except ValueError as exc:
        raise ValueError("r20 artifact escaped the experiment root") from exc
    return target


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("r20 payload is already sealed")
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"r20 refuses to overwrite a different artifact: {target}")
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
    scope = (root / R20_ROOT).resolve()
    paths = sorted(path for path in scope.rglob("*.py") if "__pycache__" not in path.parts)
    return {relative_path(root, path): sha256_file(path) for path in paths}


def build_manifest(project_root: str | Path, config: R20Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    split = build_source_stratified_split(root, config.r19.r18)
    r19_manifest_path = project_path(root, config.source["r19_manifest_path"], must_exist=True)
    r19_manifest, _, _ = load_r19_manifest(root, config.r19, allow_code_drift=True)
    if split["assignments"] != r19_manifest.get("parent_assignments"):
        raise ValueError("r20 parent split differs from sealed r19 parent split")
    frozen_l2 = project_path(root, config.r17.source["r23_checkpoint_path"], must_exist=True)
    return {
        "schema_version": R20_MANIFEST_SCHEMA,
        "experiment_id": R20_EXPERIMENT_ID,
        "variant": config["variant"],
        "config": {
            "path": relative_path(root, config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "source": {
            "dataset": config.source["dataset"],
            "r19_config_path": config.source["r19_config_path"],
            "r19_config_sha256": config.r19.sha256,
            "r19_manifest_path": relative_path(root, r19_manifest_path),
            "r19_manifest_sha256": sha256_file(r19_manifest_path),
            "r19_checkpoint_path": config.source["r19_checkpoint_path"],
            "r19_checkpoint_sha256": config.source["r19_checkpoint_sha256"],
            "r19_checkpoint_model_state_sha256": config.source["r19_checkpoint_model_state_sha256"],
            "r19_checkpoint_arm": config.source["r19_checkpoint_arm"],
            "r19_checkpoint_seed": int(config.source["r19_checkpoint_seed"]),
            "r19_checkpoint_epoch": int(config.source["r19_checkpoint_epoch"]),
            "frozen_l2_checkpoint_path": relative_path(root, frozen_l2),
            "frozen_l2_checkpoint_sha256": sha256_file(frozen_l2),
            "frozen_l2_identity": "V27c-r23 robust_control seed42 epoch1",
        },
        "split": {key: value for key, value in split.items() if key != "assignments"},
        "parent_assignments": split["assignments"],
        "input_contract": {
            "raw_radial_speed_scalar": "removed from ordinary TR encoder in both arms",
            "zero_control": "all four sidecar values and validity flags are zero/false",
            "radial_decomposed": {
                "values": list(R20_RADIAL_VALUE_NAMES),
                "validity": list(R20_RADIAL_VALID_NAMES),
                "rate_and_consistency": "positive dt only; dt=0/invalid/first event is invalid",
                "injection": "post_fusion_pre_liquid",
            },
            "snr": "r19 snr_missing_only; untrusted Excel noise excluded",
        },
        "model": {
            "task": "l2",
            "class_names": list(R20_L2_CLASSES),
            "unknown_classifier_class": False,
            "direct_initialization": "frozen V27c-r23 robust_control seed42 epoch1 base L2",
            "main_variable": "radial speed scalar replaced by four causal sidecar values plus validity flags",
        },
        "training_boundary": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "rd_payload_access_count": 0,
            "l3": "not_executed",
        },
        "code_hashes": _code_hashes(root),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R20Config, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_json_if_identical(path, seal(build_manifest(root, config)))
    return config, path, sha256_file(path)


def load_manifest(
    project_root: str | Path,
    config: R20Config,
    *,
    allow_code_drift: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    path = artifact_path(root, config.artifacts["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r20 manifest is not an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("r20 manifest payload hash cannot be recomputed")
    expected = build_manifest(root, config)
    if body != expected:
        body_without_code = {key: value for key, value in body.items() if key != "code_hashes"}
        expected_without_code = {key: value for key, value in expected.items() if key != "code_hashes"}
        if not allow_code_drift or body_without_code != expected_without_code:
            raise ValueError("r20 manifest differs from current config, source, split or code")
    return dict(payload), path, sha256_file(path)


__all__ = ["R20_ARMS", "R20_CHECKPOINT_SCHEMA", "R20Config", "R20_EXPERIMENT_ID", "R20_L2_CLASSES", "R20_MANIFEST_SCHEMA", "R20_RADIAL_VALID_NAMES", "R20_RADIAL_VALUE_DIM", "R20_RADIAL_VALUE_NAMES", "R20_REPORT_SCHEMA", "R20_ROOT", "R20_SCHEMA_VERSION", "R20_SMOKE_SCHEMA", "R20_TRAINING_SUMMARY_SCHEMA", "artifact_path", "build_manifest", "canonical_sha256", "load_config", "load_manifest", "prepare_manifest", "project_path", "seal", "sha256_file", "write_json_if_identical", "write_torch_if_identical"]