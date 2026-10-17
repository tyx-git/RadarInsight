"""Run V25a's full V24-equivalent gradient-telemetry audit training."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
import torch



V25A_EPOCH_COUNT = 20
V25A_BATCH_SIZE = 2
V25A_LEARNING_RATE = 1.0e-4
V25A_GRADIENT_CLIP_NORM = 100.0
V25A_CLASS_WEIGHT_NORMALIZATION = "batch_weight_sum"
V25A_FROZEN_MODEL_FIELDS = {
    "tr_value_dim": 14,
    "tr_missing_dim": 11,
    "rd_quality_dim": 7,
    "hidden_dim": 48,
    "rd_embedding_dim": 24,
    "liquid_hidden_dim": 48,
    "rd_height": 21,
    "rd_width": 384,
    "liquid_tau_min_seconds": 0.05,
    "liquid_tau_max_seconds": 30.0,
    "liquid_zero_dt_event_floor_seconds": 0.001,
    "liquid_dt_max_seconds": 120.0,
    "liquid_invalid_dt_update_mode": "bounded_observation",
    "liquid_invalid_dt_observation_gain": 0.25,
    "liquid_state_normalization": "recurrent",
    "rd_frame_aggregation": "masked_mean",
    "dt_reason_mode": "v3_reason_aware_dynamics_only",
    "evidence_dim": 1,
    "rd_temporal_residual_dim": 0,
}


def _require_v24_equivalent_config(config: Any) -> None:
    """Fail closed if V25a drifts from the V24 training semantics."""

    if (
        int(config.train.epochs) != V25A_EPOCH_COUNT
        or int(config.train.batch_size) != V25A_BATCH_SIZE
        or abs(float(config.train.learning_rate) - V25A_LEARNING_RATE) > 1.0e-15
        or abs(float(config.train.gradient_clip_norm) - V25A_GRADIENT_CLIP_NORM)
        > 1.0e-12
        or str(config.train.class_weight_normalization)
        != V25A_CLASS_WEIGHT_NORMALIZATION
        or tuple(float(value) for value in config.train.class_weights)
        != (0.5173137460650578, 14.93939393939394)
        or float(config.train.prefix_auxiliary_weight) != 0.1
        or int(config.train.prefix_auxiliary_min_events) != 3
        or {
            name: getattr(config.model, name)
            for name in V25A_FROZEN_MODEL_FIELDS
        }
        != V25A_FROZEN_MODEL_FIELDS
        or config.train.loss_mode != "parent_final_task_ce_with_prefix_auxiliary"
        or config.train.checkpoint_selection_metric != "parent_min_precision_recall"
        or abs(float(config.train.task_min_precision_recall_target) - 0.9) > 1.0e-12
        or int(config.train.seed) != 43
        or int(config.train.dataloader_seed) != 43
        or int(config.train.early_stopping_patience) != 0
        or config.train.cuda_deterministic_algorithms
        or not config.train.record_parent_order_hash
        or not config.train.save_every_epoch
        or not config.train.full_parameter_training
    ):
        raise ValueError("V25a config drifted from the frozen V24 training semantics")






def _restore_training_state(
    payload: Mapping[str, Any], generator: torch.Generator
) -> dict[str, Any]:
    state = dict(payload.get("training_state") or {})
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "run_state",
    }
    if set(state) != required:
        raise ValueError("V25a checkpoint lacks complete epoch-boundary recovery state")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(state["torch_cuda_rng_states"] or ())
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("V25a checkpoint CUDA RNG state count differs from this host")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, device="cpu") for value in cuda_states]
        )
    generator.set_state(torch.as_tensor(state["dataloader_generator_state"], device="cpu"))
    run_state = state["run_state"]
    if not isinstance(run_state, Mapping):
        raise ValueError("V25a checkpoint training_state lacks run_state")
    return dict(run_state)











