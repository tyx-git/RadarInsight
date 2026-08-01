from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RDNovelParentConfig:
    novel_update_index_name: str = "expert1_cq08_rd_physical_novel_update_index_v2.csv"
    base_checkpoint: str = (
        "artifacts/20260727-145406/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs/"
        "best_rd_only_drone_bird_model.pt"
    )
    output_dir: str = "artifacts/rd_physical_v2_novel_parent_state"
    epochs: int = 18
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.0
    adapter_dim: int = 96
    state_dim: int = 96
    prefix_loss_weight: float = 0.43358851852102304
    max_frames_per_chunk: int = 32
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False
