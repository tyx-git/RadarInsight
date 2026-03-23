from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class TrainingConfig:
    data_root: str = "data/sessions"
    output_dir: str = "artifacts/checkpoints"
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    num_workers: int = 0
    use_class_balanced_loss: bool = True
    class_balance_beta: float = 0.999
    warmup_epochs: int = 2
    max_grad_norm: float = 1.0
    sequence_length: int = 24
    rd_frames: int = 24
    rd_height: int = 64
    rd_width: int = 64
    rd_input_channels: int = 8
    include_rd: bool = True
    track_feature_columns: tuple[str, ...] = (
        "range_m",
        "azimuth_deg",
        "radial_speed_mps",
        "snr_db",
        "delta_range_m",
        "delta_azimuth_deg",
        "speed_delta_mps",
        "acceleration_mps2",
        "turn_rate_degps",
        "range_jitter_m",
        "snr_delta_db",
    )
    track_projection_dim: int = 128
    tcn_channels: tuple[int, ...] = (128, 128, 256)
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    cross_attention_layers: int = 2
    cross_attention_heads: int = 4
    cross_attention_dropout: float = 0.1
    fusion_hidden_size: int = 256
    x3d_stem_channels: int = 24
    x3d_stage_channels: tuple[int, ...] = (32, 64, 128, 192)
    x3d_stage_depths: tuple[int, ...] = (2, 3, 5, 3)
    x3d_spatial_strides: tuple[int, ...] = (1, 2, 2, 2)
    x3d_temporal_strides: tuple[int, ...] = (1, 1, 2, 2)
    x3d_bottleneck_expansion: float = 2.25
    x3d_se_ratio: float = 0.0625
    label_smoothing: float = 0.05
    track_aux_loss_weight: float = 0.1
    rd_aux_loss_weight: float = 0.3

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "TrainingConfig":
        payload = dict(values or {})
        tuple_fields = {
            "track_feature_columns",
            "tcn_channels",
            "x3d_stage_channels",
            "x3d_stage_depths",
            "x3d_spatial_strides",
            "x3d_temporal_strides",
        }
        for field_name in tuple_fields:
            if field_name in payload:
                payload[field_name] = tuple(payload[field_name])

        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_payload = {key: value for key, value in payload.items() if key in valid_keys}
        return cls(**filtered_payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["track_feature_columns"] = list(self.track_feature_columns)
        payload["tcn_channels"] = list(self.tcn_channels)
        payload["x3d_stage_channels"] = list(self.x3d_stage_channels)
        payload["x3d_stage_depths"] = list(self.x3d_stage_depths)
        payload["x3d_spatial_strides"] = list(self.x3d_spatial_strides)
        payload["x3d_temporal_strides"] = list(self.x3d_temporal_strides)
        return payload

    @property
    def feature_dim(self) -> int:
        return len(self.track_feature_columns)
