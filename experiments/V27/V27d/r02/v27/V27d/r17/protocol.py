"""V27d/r17 的独立实验协议。

本实验只读取 ``OriData/excel``。parent 身份是
``source_suffix + StartTimer + Batch``，旧的仅按 Batch 合并结果只作为
历史对照，不作为 r17 输入。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


R17_EXPERIMENT_ID = "V27d-r17"
R17_SCHEMA_VERSION = "RADAR-LN3-V27d-r17-excel-tr-only-v2"
R17_CONFIG_SCHEMA = "RADAR-LN3-V27d-r17-excel-tr-only-config-v2"
R17_ROOT = Path("src-v3") / "v27" / "V27d" / "r17"
R17_DATA_ROOT = Path("data") / "processed" / "v27d_r17"

R17_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
R17_RAW_LABEL_TO_CLASS = {
    "无人机": "DroneTarget",
    "微型机": "DroneTarget",
    "鸟": "BirdTarget",
    "疑似气球": "BalloonTarget",
    "气球": "BalloonTarget",
    "杂波": "ClutterTarget",
}
R17_ALLOWED_RAW_LABELS = frozenset(R17_RAW_LABEL_TO_CLASS)
R17_DT_REASON = {
    "padding": 0,
    "first_event_no_predecessor": 1,
    "zero_delta_at_second_precision": 2,
    "positive_delta_at_second_precision": 3,
}




def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    raw = str(value).replace("\\", "/").strip()
    if not raw:
        raise ValueError("r17 project path must not be empty")
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"r17 path escapes project root: {resolved}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def relative_path(project_root: str | Path, path: str | Path) -> str:
    root = Path(project_root).resolve()
    return project_path(root, path).relative_to(root).as_posix()


def _inside(root: Path, path: Path, description: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"r17 {description} escaped its root: {path}") from exc
    return path


def data_path(project_root: str | Path, value: str | Path) -> Path:
    root = Path(project_root).resolve()
    return _inside((root / R17_DATA_ROOT).resolve(), project_path(root, value), "data artifact")




def _mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"r17 {context}.{key} must be an object")
    return dict(value)


def _text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r17 {context}.{key} must be non-empty text")
    return value.strip()


def _sha(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(payload, key, context).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"r17 {context}.{key} must be a SHA-256 digest")
    return value


class R17Config:
    def __init__(self, *, payload: dict[str, Any], path: Path) -> None:
        self.payload = payload
        self.path = path
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
    def frozen_layers(self) -> Mapping[str, Any]:
        return self.payload["frozen_layers"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R17Config:
    root = Path(project_root).resolve()
    candidate = (
        root / R17_ROOT / "config" / "V27d-r17-excel-tr-only.json"
        if config_path is None
        else project_path(root, config_path)
    )
    path = project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("r17 configuration root must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != R17_CONFIG_SCHEMA or payload.get("experiment_id") != R17_EXPERIMENT_ID:
        raise ValueError("r17 configuration identity/schema differs")
    source = _mapping(payload, "source", "root")
    if _text(source, "semantic_resolution", "source") != "column_schema_over_filename_role":
        raise ValueError("r17 semantic resolution differs")
    if _text(source, "snr_input_policy", "source") != "exclude_untrusted_excel_noise_and_derived_snr":
        raise ValueError("r17 SNR policy differs")
    for key in ("raw_excel_root", "r15_config_path", "r15_checkpoint_path", "r23_config_path", "r23_checkpoint_path", "l3_example_registry_path"):
        _text(source, key, "source")
    for key in ("r15_checkpoint_sha256", "r15_model_contract_sha256", "r15_model_state_sha256", "r23_checkpoint_sha256", "r23_model_contract_sha256", "r23_model_state_sha256"):
        _sha(source, key, "source")
    for key in ("r15_config_sha256", "r23_config_sha256", "l3_example_registry_sha256"):
        _sha(source, key, "source")
    for path_key, hash_key in (
        ("r15_config_path", "r15_config_sha256"),
        ("r23_config_path", "r23_config_sha256"),
        ("l3_example_registry_path", "l3_example_registry_sha256"),
    ):
        source_path = project_path(root, source[path_key], must_exist=True)
        if sha256_file(source_path) != source[hash_key]:
            raise ValueError(f"r17 source file hash differs: {path_key}")
    payload["source"] = source
    data = _mapping(payload, "data", "root")
    if data.get("class_names") != list(R17_CLASS_NAMES):
        raise ValueError("r17 class order differs")
    if data.get("allowed_raw_labels") != sorted(R17_ALLOWED_RAW_LABELS):
        raise ValueError("r17 allowed raw labels differ")
    for key, expected in (("parent_key", "source_suffix+StartTimer+Batch"), ("event_order", "tck_workbook_row_order_within_parent"), ("radial_speed_transform", "radial_speed_mps = -Speed_R"), ("time_unit", "milliseconds_to_seconds_relative_to_parent_first_event"), ("snr_model_input", "missing_zero_with_validity_mask")):
        if data.get(key) != expected:
            raise ValueError(f"r17 data.{key} differs")
    if int(data.get("max_sequence_length", 0)) <= 0 or not 0.0 < float(data.get("validation_fraction", 0.0)) < 1.0:
        raise ValueError("r17 data bounds differ")
    if int(data.get("rd_placeholder_frames", 0)) != 1:
        raise ValueError("r17 TR-only RD placeholder width must be one")
    clipping = data.get("l2_robust_clipping")
    if not isinstance(clipping, Mapping):
        raise ValueError("r17 L2 robust clipping is missing")
    for field in ("range_m", "speed_mps"):
        bounds = clipping.get(field)
        if not isinstance(bounds, Mapping):
            raise ValueError(f"r17 L2 robust clipping lacks {field}")
        lower = float(bounds.get("lower", -1.0))
        upper = float(bounds.get("upper", -1.0))
        if lower < 0.0 or not lower <= upper:
            raise ValueError(f"r17 L2 robust clipping bounds are invalid: {field}")
    payload["data"] = data
    frozen = _mapping(payload, "frozen_layers", "root")
    for task, expected in (("l1", ("V27d-r15", "v10_observation_increment", 11)), ("l2", ("V27c-r23", "robust_control", 1))):
        section = _mapping(frozen, task, "frozen_layers")
        if _text(section, "experiment_id", f"frozen_layers.{task}") != expected[0] or _text(section, "arm", f"frozen_layers.{task}") != expected[1] or int(section.get("seed", -1)) != 42 or int(section.get("epoch", -1)) != expected[2]:
            raise ValueError(f"r17 frozen {task} identity differs")
        for key in ("checkpoint_sha256", "model_contract_sha256", "model_state_sha256"):
            _sha(section, key, f"frozen_layers.{task}")
    payload["frozen_layers"] = frozen
    training = _mapping(payload, "training", "root")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("r17 training seeds are invalid")
    for key in ("batch_size", "epochs", "gradient_clip_norm"):
        if float(training.get(key, 0)) <= 0.0:
            raise ValueError(f"r17 training.{key} is invalid")
    payload["training"] = training
    artifacts = _mapping(payload, "artifacts", "root")
    for key in ("manifest_path", "data_manifest_path", "report_template", "frozen_report_path", "smoke_path", "checkpoint_template", "epoch_checkpoint_template", "training_summary_template", "training_log_template", "experiment_doc_path"):
        _text(artifacts, key, "artifacts")
    payload["artifacts"] = artifacts
    return R17Config(payload=payload, path=path)














def _torch_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_torch_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_torch_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right












__all__ = ["R17_ALLOWED_RAW_LABELS", "R17_CLASS_NAMES", "R17_CONFIG_SCHEMA", "R17_DATA_ROOT", "R17_DT_REASON", "R17_EXPERIMENT_ID", "R17_RAW_LABEL_TO_CLASS", "R17_ROOT", "R17_SCHEMA_VERSION", "R17Config", "canonical_sha256", "data_path", "load_config", "project_path", "relative_path", "sha256_file"]