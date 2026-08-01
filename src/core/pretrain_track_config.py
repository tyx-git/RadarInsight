from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPretrainConfig:
    data_root: str = "data"
    index_name: str = "system_trainable_track_index.csv"
    allowed_parent_manifest_path: str = ""
    require_train_parent_manifest: bool = False
    # ``full_track`` preserves the historical parent-level objective.  The
    # causal-window mode consumes a persisted train-only causal index (for
    # example V9), so pretraining sees the same bounded evidence as Expert1
    # and streaming inference.
    sampling_mode: str = "full_track"
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
