"""第二版液态算法的结构契约（写入每个 V27 checkpoint）。

来源：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa，src-v3/radar_ln3/liquid_v2.py 逐字拆分。
"""

from __future__ import annotations

from typing import Any

from .cell import (
    EQUATION_VERSION,
    LiquidV2Cell,
)

from .time_policy import (
    time_policy_contract,
)

def liquid_v2_contract(cell: LiquidV2Cell) -> dict[str, Any]:
    """Return the structural contract bound into every V27 checkpoint."""

    return {
        "equation_version": EQUATION_VERSION,
        "cell_class": "radar_ln3.liquid_v2.LiquidV2Cell",
        "sequence_wrapper_class": "radar_ln3.liquid_v2.LiquidV2SequenceWrapper",
        "input_dim": cell.input_dim,
        "hidden_dim": cell.hidden_dim,
        "memory_state": "raw_r_t",
        "compute_state": "n_t=LayerNorm(r_t) temporary only",
        "post_update_normalization": False,
        "candidate_function": {
            "inputs": ["x_t", "n_{t-1}"],
            "candidate": "tanh(candidate_projection([x_t,n_{t-1}]) + input_projection(x_t) + state_projection(n_{t-1}))",
            "gate": "sigmoid(gate_projection([x_t,n_{t-1}]))",
            "v1_capacity_preserved": True,
        },
        "classifier_readout_must_be_independent": True,
        "innovation_names": {
            "candidate": "candidate_innovation=c_t-r_{t-1}",
            "mixed_memory": "mixed_memory_innovation=m_t-r_{t-1}",
            "actual_update": "actual_update=r_t-r_{t-1}",
        },
        "trace_semantics": {
            "raw_state_before_update": "r_{t-1}, before the current active event update; padded rows retain the previous state",
            "raw_state": "r_t, after the current active event update; padded rows retain the previous state",
        },
        "time_policy": time_policy_contract(
            tau_min_seconds=cell.tau_min_seconds,
            tau_max_seconds=cell.tau_max_seconds,
            zero_dt_event_floor_seconds=cell.zero_dt_event_floor_seconds,
            dt_max_seconds=cell.dt_max_seconds,
            invalid_dt_update_mode=cell.invalid_dt_update_mode,
            invalid_dt_observation_gain=cell.invalid_dt_observation_gain,
        ),
        "padding_calls_cell": False,
    }


__all__ = ["liquid_v2_contract"]
