from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    index_path: str
    contract_path: str
    split_name: str
    max_sequence_length: int
    max_rd_frames_per_point: int
    exclude_anomalous_parents: bool
    sample_limit: int | None
    rd_amplitude_center: float
    rd_amplitude_scale: float
    normalization_stats_path: str | None = None
    train_split_name: str | None = None
    validation_split_name: str | None = None
    test_split_name: str | None = None
    expected_index_sha256: str | None = None
    expected_contract_sha256: str | None = None
    expected_stats_sha256: str | None = None
    index_schema_version: str | None = None
    dt_valid_field: str | None = None
    contract_schema_version: str | None = None
    rd_quality_schema: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    stable_confirm_count: int = 2
    stable_ema_decay: float = 0.65


@dataclass(frozen=True)
class ModelConfig:
    tr_value_dim: int
    tr_missing_dim: int
    rd_quality_dim: int
    hidden_dim: int
    rd_embedding_dim: int
    liquid_hidden_dim: int
    num_classes: int
    rd_height: int
    rd_width: int
    liquid_tau_min_seconds: float
    liquid_tau_max_seconds: float
    liquid_zero_dt_event_floor_seconds: float
    liquid_dt_max_seconds: float
    dt_valid_dim: int = 0
    liquid_invalid_dt_update_mode: str = "floor"
    liquid_invalid_dt_observation_gain: float = 0.25
    liquid_state_normalization: str = "recurrent"
    rd_frame_aggregation: str = "masked_mean"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    smoke_steps: int
    learning_rate: float
    seed: int
    device: str
    checkpoint_path: str
    allow_parameter_updates: bool
    mode: str = "smoke"
    epochs: int = 1
    early_stopping_patience: int = 0
    gradient_clip_norm: float = 100.0
    checkpoint_selection_metric: str = "val_macro_f1"
    evaluate_test: bool = False
    max_train_steps: int | None = None
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    last_checkpoint_path: str | None = None
    formal_artifact_dir: str | None = None
    loss_mode: str = "final_window"
    checkpoint_schema: str = "RADAR-LN3-V2-formal-checkpoint-v1"


@dataclass(frozen=True)
class V3Config:
    schema_version: str
    config_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"configuration is missing field: {key}")
    return mapping[key]


def load_config(path: str | Path) -> V3Config:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {"RADAR-LN3-V0", "RADAR-LN3-V1", "RADAR-LN3-V2", "RADAR-LN3-V3", "RADAR-LN3-V4"}:
        raise ValueError("RADAR-LN3 config must use schema_version RADAR-LN3-V0 through RADAR-LN3-V4")
    data_payload = dict(_require(payload, "data"))
    model_payload = dict(_require(payload, "model"))
    train_payload = dict(_require(payload, "train"))
    data = DataConfig(
        index_path=str(_require(data_payload, "index_path")),
        contract_path=str(_require(data_payload, "contract_path")),
        split_name=str(_require(data_payload, "split_name")),
        max_sequence_length=max(1, int(_require(data_payload, "max_sequence_length"))),
        max_rd_frames_per_point=max(1, int(_require(data_payload, "max_rd_frames_per_point"))),
        exclude_anomalous_parents=bool(_require(data_payload, "exclude_anomalous_parents")),
        sample_limit=(None if data_payload.get("sample_limit") is None else max(1, int(data_payload["sample_limit"]))),
        rd_amplitude_center=float(_require(data_payload, "rd_amplitude_center")),
        rd_amplitude_scale=float(_require(data_payload, "rd_amplitude_scale")),
        normalization_stats_path=(None if data_payload.get("normalization_stats_path") is None else str(data_payload["normalization_stats_path"])),
        train_split_name=(None if data_payload.get("train_split_name") is None else str(data_payload["train_split_name"])),
        validation_split_name=(None if data_payload.get("validation_split_name") is None else str(data_payload["validation_split_name"])),
        test_split_name=(None if data_payload.get("test_split_name") is None else str(data_payload["test_split_name"])),
        expected_index_sha256=(None if data_payload.get("expected_index_sha256") is None else str(data_payload["expected_index_sha256"])),
        expected_contract_sha256=(None if data_payload.get("expected_contract_sha256") is None else str(data_payload["expected_contract_sha256"])),
        expected_stats_sha256=(None if data_payload.get("expected_stats_sha256") is None else str(data_payload["expected_stats_sha256"])),
        index_schema_version=(None if data_payload.get("index_schema_version") is None else str(data_payload["index_schema_version"])),
        dt_valid_field=(None if data_payload.get("dt_valid_field") is None else str(data_payload["dt_valid_field"])),
        contract_schema_version=(None if data_payload.get("contract_schema_version") is None else str(data_payload["contract_schema_version"])),
        rd_quality_schema=(None if data_payload.get("rd_quality_schema") is None else str(data_payload["rd_quality_schema"])),
    )
    if data.rd_amplitude_scale <= 0.0:
        raise ValueError("data.rd_amplitude_scale must be positive")
    model = ModelConfig(
        tr_value_dim=int(_require(model_payload, "tr_value_dim")),
        tr_missing_dim=int(_require(model_payload, "tr_missing_dim")),
        rd_quality_dim=int(_require(model_payload, "rd_quality_dim")),
        hidden_dim=int(_require(model_payload, "hidden_dim")),
        rd_embedding_dim=int(_require(model_payload, "rd_embedding_dim")),
        liquid_hidden_dim=int(_require(model_payload, "liquid_hidden_dim")),
        num_classes=int(_require(model_payload, "num_classes")),
        rd_height=int(_require(model_payload, "rd_height")),
        rd_width=int(_require(model_payload, "rd_width")),
        liquid_tau_min_seconds=float(_require(model_payload, "liquid_tau_min_seconds")),
        liquid_tau_max_seconds=float(_require(model_payload, "liquid_tau_max_seconds")),
        liquid_zero_dt_event_floor_seconds=float(_require(model_payload, "liquid_zero_dt_event_floor_seconds")),
        liquid_dt_max_seconds=float(_require(model_payload, "liquid_dt_max_seconds")),
        dt_valid_dim=int(model_payload.get("dt_valid_dim", 0)),
        liquid_invalid_dt_update_mode=str(model_payload.get("liquid_invalid_dt_update_mode", "floor")),
        liquid_invalid_dt_observation_gain=float(model_payload.get("liquid_invalid_dt_observation_gain", 0.25)),
        liquid_state_normalization=str(model_payload.get("liquid_state_normalization", "recurrent")),
        rd_frame_aggregation=str(model_payload.get("rd_frame_aggregation", "masked_mean")),
    )
    if model.num_classes != 5 or model.rd_height != 21 or model.rd_width != 384:
        raise ValueError("RADAR-LN3 model contract requires five classes and RD shape 21x384")
    if model.tr_value_dim != 14 or model.tr_missing_dim != 11 or model.rd_quality_dim != 7:
        raise ValueError("RADAR-LN3 model contract requires TR value dim 14, missing dim 11 and RD quality dim 7")
    if model.dt_valid_dim not in {0, 1}:
        raise ValueError("model.dt_valid_dim must be 0 or 1")
    if model.liquid_invalid_dt_update_mode not in {"floor", "bounded_observation"}:
        raise ValueError("model.liquid_invalid_dt_update_mode must be floor or bounded_observation")
    if not 0.0 < model.liquid_invalid_dt_observation_gain <= 1.0:
        raise ValueError("model.liquid_invalid_dt_observation_gain must be in (0, 1]")
    if not 0.0 < model.liquid_tau_min_seconds <= model.liquid_tau_max_seconds:
        raise ValueError("invalid liquid time scale bounds")
    if model.liquid_state_normalization not in {"recurrent", "readout_only"}:
        raise ValueError("model.liquid_state_normalization must be recurrent or readout_only")
    if model.rd_frame_aggregation not in {"masked_mean", "causal_gru"}:
        raise ValueError("model.rd_frame_aggregation must be masked_mean or causal_gru")
    if model.liquid_state_normalization != "recurrent" and model.rd_frame_aggregation != "masked_mean":
        raise ValueError("liquid normalization and RD temporal aggregation ablations must be run separately")
    train = TrainConfig(
        batch_size=max(1, int(_require(train_payload, "batch_size"))),
        smoke_steps=max(1, int(train_payload.get("smoke_steps", 1))),
        learning_rate=float(_require(train_payload, "learning_rate")),
        seed=int(_require(train_payload, "seed")),
        device=str(_require(train_payload, "device")),
        checkpoint_path=str(_require(train_payload, "checkpoint_path")),
        allow_parameter_updates=bool(train_payload.get("allow_parameter_updates", True)),
        mode=str(train_payload.get("mode", "smoke")),
        epochs=max(1, int(train_payload.get("epochs", 1))),
        early_stopping_patience=max(0, int(train_payload.get("early_stopping_patience", 0))),
        gradient_clip_norm=float(train_payload.get("gradient_clip_norm", 100.0)),
        checkpoint_selection_metric=str(train_payload.get("checkpoint_selection_metric", "val_macro_f1")),
        evaluate_test=bool(train_payload.get("evaluate_test", False)),
        max_train_steps=(None if train_payload.get("max_train_steps") is None else max(1, int(train_payload["max_train_steps"]))),
        class_weights=tuple(float(value) for value in train_payload.get("class_weights", [1.0] * 5)),
        last_checkpoint_path=(None if train_payload.get("last_checkpoint_path") is None else str(train_payload["last_checkpoint_path"])),
        formal_artifact_dir=(None if train_payload.get("formal_artifact_dir") is None else str(train_payload["formal_artifact_dir"])),
        loss_mode=str(train_payload.get("loss_mode", "final_window")),
        checkpoint_schema=str(train_payload.get("checkpoint_schema", "RADAR-LN3-V2-formal-checkpoint-v1")),
    )
    if train.learning_rate <= 0.0:
        raise ValueError("train.learning_rate must be positive")
    if train.gradient_clip_norm <= 0.0:
        raise ValueError("train.gradient_clip_norm must be positive")
    if len(train.class_weights) != 5 or any(value <= 0.0 for value in train.class_weights):
        raise ValueError("train.class_weights must contain five positive values")
    if train.loss_mode not in {"final_window", "causal_prefix"}:
        raise ValueError("train.loss_mode must be final_window or causal_prefix")
    evaluation_payload = dict(payload.get("evaluation", {}))
    evaluation = EvaluationConfig(
        stable_confirm_count=max(1, int(evaluation_payload.get("stable_confirm_count", 2))),
        stable_ema_decay=float(evaluation_payload.get("stable_ema_decay", 0.65)),
    )
    if not 0.0 <= evaluation.stable_ema_decay < 1.0:
        raise ValueError("evaluation.stable_ema_decay must be in [0, 1)")
    if str(payload["schema_version"]) in {"RADAR-LN3-V2", "RADAR-LN3-V3", "RADAR-LN3-V4"}:
        if train.mode != "formal_supervised":
            raise ValueError("RADAR-LN3-V2/V3/V4 requires train.mode=formal_supervised")
        if data.train_split_name != "train" or data.validation_split_name != "val":
            raise ValueError("RADAR-LN3-V2/V3/V4 requires explicit train/val split names")
        expected_index_schema = "RADAR-LN3-V1" if str(payload["schema_version"]) == "RADAR-LN3-V2" else "RADAR-LN3-V3"
        if data.index_schema_version != expected_index_schema:
            raise ValueError(f"{payload['schema_version']} formal config must explicitly consume {expected_index_schema} data rows")
        if data.sample_limit is not None:
            raise ValueError("formal RADAR-LN3-V2/V3/V4 config cannot set data.sample_limit")
        if data.split_name != "train" or data.test_split_name != "test":
            raise ValueError("formal RADAR-LN3-V2/V3/V4 config requires split_name=train and test_split_name=test")
        if data.normalization_stats_path is None:
            raise ValueError("formal RADAR-LN3-V2/V3/V4 requires train-only normalization stats")
        if train.formal_artifact_dir is None or train.last_checkpoint_path is None:
            raise ValueError("RADAR-LN3-V2/V3/V4 requires formal artifact and last checkpoint paths")
        if str(payload["schema_version"]) in {"RADAR-LN3-V3", "RADAR-LN3-V4"}:
            if data.dt_valid_field != "tr_dt_valid":
                raise ValueError("RADAR-LN3-V3/V4 requires data.dt_valid_field=tr_dt_valid")
            if model.dt_valid_dim != 1:
                raise ValueError("RADAR-LN3-V3/V4 requires model.dt_valid_dim=1")
            if train.loss_mode != "causal_prefix":
                raise ValueError("RADAR-LN3-V3/V4 requires train.loss_mode=causal_prefix")
            expected_checkpoint_schema = f"{payload['schema_version']}-formal-checkpoint-v1"
            if train.checkpoint_schema != expected_checkpoint_schema:
                raise ValueError(f"{payload['schema_version']} requires its dedicated checkpoint schema")
        if str(payload["schema_version"]) == "RADAR-LN3-V4":
            if data.contract_schema_version != "RADAR-LN3-V3":
                raise ValueError("RADAR-LN3-V4 must explicitly reuse RADAR-LN3-V3 data contract")
            if model.liquid_invalid_dt_update_mode != "bounded_observation":
                raise ValueError("RADAR-LN3-V4 requires bounded_observation invalid dt updates")
    return V3Config(
        schema_version=str(payload["schema_version"]),
        config_name=str(payload.get("config_name", config_path.name)),
        data=data,
        model=model,
        train=train,
        evaluation=evaluation,
    )


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else project_root / candidate


__all__ = ["DataConfig", "EvaluationConfig", "ModelConfig", "TrainConfig", "V3Config", "load_config", "resolve_project_path"]
