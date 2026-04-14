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
    deterministic: bool = True
    num_workers: int = 0
    use_class_balanced_loss: bool = True
    class_balance_beta: float = 0.995
    use_focal_loss: bool = True
    focal_gamma: float = 1.25
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
    track_backbone_type: str = "temporal"
    use_track_window_mix: bool = True
    track_window_mix_prob: float = 0.35
    temporal_2d_branch_channels: int = 96
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
    bird_surface_aux_loss_weight: float = 0.25
    bird_surface_logit_residual_weight: float = 0.2
    rd_dominant_fusion_bias: float = 0.35
    rd_unimodal_logit_residual_weight: float = 0.18
    min_rd_fusion_weight: float = 0.15
    ambiguous_rd_fusion_boost: float = 0.15
    surface_rd_fusion_boost: float = 0.1
    use_weighted_fusion_feature: bool = False
    rd_fusion_curriculum_enabled: bool = True
    rd_fusion_curriculum_start_epoch: int = 3
    rd_fusion_curriculum_full_epoch: int = 6
    min_rd_fusion_weight_start: float = 0.0
    ambiguous_rd_fusion_boost_start: float = 0.0
    surface_rd_fusion_boost_start: float = 0.0
    bird_rd_aux_loss_weight: float = 0.15
    bird_surface_rd_aux_loss_weight: float = 0.2
    bird_surface_prototype_loss_weight: float = 0.2
    bird_surface_prototype_margin: float = 0.35
    bird_surface_sampling_boost: float = 1.25
    hard_ship_session_sampling_boost: float = 1.0
    hard_ship_session_ids: tuple[str, ...] = ()
    hard_bird_session_sampling_boost: float = 1.0
    hard_bird_session_ids: tuple[str, ...] = ()
    suspicious_ship_downweight_enabled: bool = False
    suspicious_ship_downweight_factor: float = 1.0
    suspicious_ship_session_ids: tuple[str, ...] = ()
    suspicious_ship_speed_mean_threshold: float = 0.0
    suspicious_ship_direction_consistency_threshold: float = 0.0
    suspicious_ship_surface_bias_threshold: float = 0.0
    suspicious_ship_snr_mean_threshold: float = 0.0
    confusion_pair_sampling_enabled: bool = False
    confusion_pair_sampling_boost: float = 1.0
    confusion_pair_bird_ship_session_ids: tuple[str, ...] = ()
    confusion_pair_ship_drone_session_ids: tuple[str, ...] = ()
    ship_drone_margin_loss_weight: float = 0.0
    ship_drone_margin: float = 0.0
    bird_ship_margin_loss_weight: float = 0.0
    bird_ship_margin: float = 0.0
    conditional_ship_drone_margin_enabled: bool = False
    conditional_ship_drone_margin_weight: float = 0.0
    conditional_ship_drone_margin: float = 0.0
    conditional_ship_drone_session_ids: tuple[str, ...] = ()
    conditional_ship_abs_speed_mean_threshold: float = 0.0
    conditional_ship_range_monotonicity_min: float = 0.0
    conditional_ship_snr_mean_min: float = 0.0
    curated_ship_drone_manifest_path: str = ""
    curated_ship_drone_margin_enabled: bool = False
    curated_ship_drone_margin_weight: float = 0.0
    curated_ship_drone_margin: float = 0.0
    curated_bird_ship_manifest_path: str = ""
    curated_bird_ship_margin_enabled: bool = False
    curated_bird_ship_margin_weight: float = 0.0
    curated_bird_ship_margin: float = 0.0
    confusion_pair_handler_enabled: bool = False
    confusion_pair_handler_hidden_dim: int = 64
    confusion_pair_handler_dropout: float = 0.1
    confusion_pair_handler_weight: float = 0.0
    confusion_pair_handler_gate_threshold: float = 0.55
    confusion_ship_drone_handler_weight: float = 0.0
    confusion_bird_ship_handler_weight: float = 0.0
    bird_surface_triplet_loss_weight: float = 0.15
    bird_surface_triplet_margin: float = 0.4
    session_grouped_batches: bool = False
    session_logit_loss_weight: float = 0.0
    track_router_aux_loss_weight: float = 0.15
    rd_bird_drone_aux_loss_weight: float = 0.15
    rd_ship_drone_aux_loss_weight: float = 0.06
    rd_ship_bird_aux_loss_weight: float = 0.06
    ship_specialist_logit_residual_weight: float = 0.0
    rd_ship_air_aux_loss_weight: float = 0.04
    best_metric_name: str = "val_composite"
    best_metric_weights: dict[str, float] = field(default_factory=lambda: {"val_macro_f1": 0.7, "val_accuracy": 0.3})
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.002
    coarse_label_map: dict[str, str] = field(
        default_factory=lambda: {
            "human": "HumanTarget",
            "bird": "BirdTarget",
            "drone": "DroneTarget",
            "micro_aircraft": "DroneTarget",
            "small_aircraft": "DroneTarget",
            "large_aircraft": "DroneTarget",
            "vehicle": "ShipTarget",
            "vessel": "ShipTarget",
            "suspected_balloon": "OtherTarget",
            "unknown": "UnknownTarget",
            "clutter": "OtherTarget",
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
            "hard_ship_session_ids",
            "hard_bird_session_ids",
            "suspicious_ship_session_ids",
            "confusion_pair_bird_ship_session_ids",
            "confusion_pair_ship_drone_session_ids",
            "conditional_ship_drone_session_ids",
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
