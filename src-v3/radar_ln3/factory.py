from __future__ import annotations

from .config import V3Config
from .data import require_causal_rd_quality_schema
from .model import RadarLiquidClassifier


def build_model(config: V3Config) -> RadarLiquidClassifier:
    require_causal_rd_quality_schema(config.data.rd_quality_schema)
    return RadarLiquidClassifier(
        tr_value_dim=config.model.tr_value_dim,
        tr_missing_dim=config.model.tr_missing_dim,
        rd_quality_dim=config.model.rd_quality_dim,
        hidden_dim=config.model.hidden_dim,
        rd_embedding_dim=config.model.rd_embedding_dim,
        liquid_hidden_dim=config.model.liquid_hidden_dim,
        num_classes=config.model.num_classes,
        tau_min_seconds=config.model.liquid_tau_min_seconds,
        tau_max_seconds=config.model.liquid_tau_max_seconds,
        zero_dt_event_floor_seconds=config.model.liquid_zero_dt_event_floor_seconds,
        dt_max_seconds=config.model.liquid_dt_max_seconds,
        dt_valid_dim=config.model.dt_valid_dim,
        liquid_invalid_dt_update_mode=config.model.liquid_invalid_dt_update_mode,
        liquid_invalid_dt_observation_gain=config.model.liquid_invalid_dt_observation_gain,
        liquid_state_normalization=config.model.liquid_state_normalization,
        rd_frame_aggregation=config.model.rd_frame_aggregation,
        rd_quality_schema=config.data.rd_quality_schema,
    )


__all__ = ["build_model"]
