from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPretrainConfig:
    data_root: str = "data"
    index_name: str = "system_trainable_track_index.csv"
    allowed_parent_manifest_path: str = ""
    require_train_parent_manifest: bool = False
    require_manifest_provenance: bool = False
    parent_split_manifest_path: str = ""
    require_parent_validation: bool = False
    # ``full_track`` preserves the historical parent-level objective.  The
    # causal-window mode consumes a persisted train-only causal index (for
    # example V9), so pretraining sees the same bounded evidence as Expert1
    # and streaming inference.
    sampling_mode: str = "full_track"
    # Historical checkpoints used every persisted point with equal weight.
    # New causal-window experiments can opt into parent_equal_v1 so overlapping
    # windows from long parents do not dominate source normalization statistics.
    source_statistics_weighting: str = "point_weighted_v1"
    output_dir: str = "artifacts/pretrain_track"
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    model_dim: int = 128
    dropout: float = 0.1
    mask_ratio: float = 0.15
    contrastive_temperature: float = 0.1
    # ``diagonal_v1`` preserves the historical one-positive objective.
    # ``parent_multi_positive_v1`` treats windows from the same parent as
    # positives in both directions, preventing overlapping causal windows
    # from becoming false negatives.
    contrastive_pairing_mode: str = "diagonal_v1"
    # Historical one-pass window exposure versus parent-equal sampling.
    parent_sampling_mode: str = "window_shuffle_v1"
    mtm_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.2
    trajectory_feature_schema: str = "v1"
    temporal_encoding_mode: str = "none"
    temporal_pooling_mode: str = "mean"
    # Historical V1/V2 pretraining remains available only for explicit replay.
    allow_legacy_feature_contract: bool = False
    seed: int = 42
    device: str = "cuda"

    def validate_active_training_contract(self) -> None:
        """Prevent the active pretraining command from silently using V1/V2."""
        if bool(self.allow_legacy_feature_contract):
            return
        if str(self.trajectory_feature_schema).strip().lower() != "v3":
            raise ValueError(
                "active track pretraining requires trajectory_feature_schema='v3'; "
                "set allow_legacy_feature_contract=true only for explicit historical V1/V2 replay"
            )
        if str(self.contrastive_pairing_mode).strip().lower() not in {
            "diagonal_v1",
            "parent_multi_positive_v1",
        }:
            raise ValueError(
                "unsupported contrastive_pairing_mode: "
                f"{self.contrastive_pairing_mode!r}; expected 'diagonal_v1' "
                "or 'parent_multi_positive_v1'"
            )
        if str(self.parent_sampling_mode).strip().lower() not in {
            "window_shuffle_v1",
            "parent_equal_v1",
        }:
            raise ValueError(
                "unsupported parent_sampling_mode: "
                f"{self.parent_sampling_mode!r}; expected 'window_shuffle_v1' "
                "or 'parent_equal_v1'"
            )
