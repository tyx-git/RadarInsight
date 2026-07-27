from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RDDroneBirdDiagnosticConfig:
    index_name: str = "expert1_realtime_causal_window_index_v7.csv"
    rd_segment_manifest_name: str = "expert1_cq08_rd_anchor_manifest_v1.csv"
    window_segment_map_name: str = "expert1_realtime_causal_window_cq08_anchor_map_v1.csv"
    output_dir: str = "artifacts/rd_drone_bird_diagnostic"
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    micro_doppler_aux_weight: float = 0.20
    use_rd_patch_cnn: bool = False
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False
