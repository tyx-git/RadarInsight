from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RDNovelAnchorResidualConfig:
    novel_update_index_name: str = "expert1_cq08_rd_physical_novel_update_index_v2.csv"
    anchor_logit_cache_name: str = "RETIRED_incompatible_physical_v2_window_logit_cache.csv"
    base_checkpoint: str = (
        "artifacts/20260727-145406/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs/"
        "best_rd_only_drone_bird_model.pt"
    )
    output_dir: str = "artifacts/rd_physical_v2_anchor_residual"
    epochs: int = 18
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.0
    adapter_dim: int = 96
    state_dim: int = 96
    max_residual_logit_delta: float = 0.75
    prefix_loss_weight: float = 0.13291639806178684
    residual_l2_weight: float = 0.05
    anchor_non_degradation_weight: float = 1.0
    anchor_harm_margin: float = 0.0
    allow_rejected_residual_experiment: bool = False
    max_frames_per_chunk: int = 32
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False
