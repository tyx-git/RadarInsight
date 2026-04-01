from dataclasses import asdict, dataclass, field
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
    sequence_length: int = 16
    track_short_length: int = 8
    track_mid_length: int = 16
    track_long_length: int = 16
    track_window_points: int = 24
    track_window_stride_points: int = 8
    track_min_window_points: int = 8
    rd_frames: int = 24
    rd_height: int = 64
    rd_width: int = 64
    rd_input_channels: int = 12
    include_track: bool = True
    include_rd: bool = True
    label_granularity: str = "coarse"
    track_feature_columns: tuple[str, ...] = (
        "range_m",
        "azimuth_deg",
        "radial_speed_mps",
        "abs_radial_speed_mps",
        "speed_mps",
        "height_m",
        "elevation_deg",
        "course_deg",
        "amplitude_db",
        "noise_db",
        "snr_db",
        "delta_range_m",
        "delta_azimuth_deg",
        "speed_delta_mps",
        "abs_speed_delta_mps",
        "acceleration_mps2",
        "turn_rate_degps",
        "jerk_mps3",
        "turn_acceleration_degps2",
        "range_jitter_m",
        "snr_delta_db",
        "height_delta_m",
        "elevation_delta_deg",
        "course_delta_deg",
        "amplitude_delta_db",
        "noise_delta_db",
    )
    track_global_feature_columns: tuple[str, ...] = (
        "clip_range_mean_m",
        "clip_range_span_m",
        "clip_speed_mean_mps",
        "clip_speed_std_mps",
        "clip_speed_span_mps",
        "clip_abs_speed_mean_mps",
        "clip_abs_speed_std_mps",
        "clip_abs_speed_max_mps",
        "clip_speed_direction_consistency",
        "clip_range_monotonicity",
        "clip_abs_acceleration_mean_mps2",
        "clip_abs_turn_rate_mean_degps",
        "clip_range_jitter_std_m",
        "clip_speed_zero_crossing_rate",
        "clip_azimuth_std_deg",
        "clip_snr_mean_db",
        "clip_height_mean_m",
        "clip_height_span_m",
        "clip_elevation_mean_deg",
        "clip_elevation_std_deg",
        "clip_course_std_deg",
        "clip_amplitude_mean_db",
        "clip_noise_mean_db",
        "clip_speed_energy_low",
        "clip_speed_energy_mid",
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
    track_aux_loss_weight: float = 0.25
    rd_aux_loss_weight: float = 0.15
    hierarchical_aux_loss_weight: float = 0.2
    detail_aux_loss_weight: float = 0.1
    detail_boundary_loss_weight: float = 0.2
    detail_boundary_margin: float = 0.2
    detail_logit_residual_weight: float = 0.2
    detail_cosine_scale: float = 16.0
    notes_aux_loss_weight: float = 0.0
    bird_drone_aux_loss_weight: float = 0.0
    coarse_label_map: dict[str, str] = field(
        default_factory=lambda: {
            "human": "HumanTarget",
            "bird": "BirdTarget",
            "drone": "DroneTarget",
            "micro_aircraft": "AircraftTarget",
            "small_aircraft": "AircraftTarget",
            "large_aircraft": "AircraftTarget",
            "vehicle": "SurfaceTarget",
            "vessel": "SurfaceTarget",
            "suspected_balloon": "UnknownTarget",
            "unknown": "UnknownTarget",
            "clutter": "NoiseTarget",
        }
    )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "TrainingConfig":
        payload = dict(values or {})
        tuple_fields = {
            "track_feature_columns",
            "track_global_feature_columns",
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
        payload["track_global_feature_columns"] = list(self.track_global_feature_columns)
        payload["tcn_channels"] = list(self.tcn_channels)
        payload["x3d_stage_channels"] = list(self.x3d_stage_channels)
        payload["x3d_stage_depths"] = list(self.x3d_stage_depths)
        payload["x3d_spatial_strides"] = list(self.x3d_spatial_strides)
        payload["x3d_temporal_strides"] = list(self.x3d_temporal_strides)
        return payload

    @property
    def feature_dim(self) -> int:
        return len(self.track_feature_columns)

    @property
    def global_feature_dim(self) -> int:
        return len(self.track_global_feature_columns)
