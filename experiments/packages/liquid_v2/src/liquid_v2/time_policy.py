"""第二版液态算法的时间策略（raw-memory / pre-norm，V27）。

来源：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa，src-v3/radar_ln3/liquid_v2.py 逐字拆分。
"""

from __future__ import annotations

from typing import Any

DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3

TIME_POLICY_VERSION = "RADAR-LN3-V27-reason-aware-time-policy-v2"

def time_policy_contract(
    *,
    tau_min_seconds: float,
    tau_max_seconds: float,
    zero_dt_event_floor_seconds: float,
    dt_max_seconds: float,
    invalid_dt_update_mode: str,
    invalid_dt_observation_gain: float,
) -> dict[str, Any]:
    """Serialize every time-dynamics choice used by a V27 cell."""

    mode = str(invalid_dt_update_mode)
    mode_rules = {
        "floor": {
            "zero_delta": "use zero_dt_event_floor_seconds through exponential alpha",
            "invalid_dt": "use zero_dt_event_floor_seconds through exponential alpha",
        },
        "bounded_observation": {
            "zero_delta": "use invalid_dt_observation_gain observation update",
            "invalid_dt": "use invalid_dt_observation_gain observation update",
        },
        "skip": {
            "zero_delta": "identity update; candidate/gate function is bypassed",
            "invalid_dt": "use invalid_dt_observation_gain observation update",
        },
        "strict_skip": {
            "zero_delta": "identity update; candidate/gate function is bypassed",
            "invalid_dt": "use invalid_dt_observation_gain observation update",
        },
        "density_weighted": {
            "zero_delta": "use clip(invalid_dt_observation_gain * zero_ordinal, 0, 3 * invalid_dt_observation_gain)",
            "invalid_dt": "use invalid_dt_observation_gain observation update",
        },
    }
    if mode not in mode_rules:
        raise ValueError(f"unsupported Liquid V2 time-policy mode: {mode}")
    return {
        "version": TIME_POLICY_VERSION,
        "tau_min_seconds": float(tau_min_seconds),
        "tau_max_seconds": float(tau_max_seconds),
        "zero_dt_event_floor_seconds": float(zero_dt_event_floor_seconds),
        "dt_max_seconds": float(dt_max_seconds),
        "invalid_dt_update_mode": mode,
        "active_mode": mode,
        "invalid_dt_observation_gain": float(invalid_dt_observation_gain),
        "dt_reason_ids": {
            "padding": DT_REASON_PADDING,
            "first_event_no_predecessor": DT_REASON_FIRST_EVENT,
            "zero_delta_at_second_precision": DT_REASON_ZERO_DELTA,
            "positive_delta_at_second_precision": DT_REASON_POSITIVE_DELTA,
        },
        "first_event_rule": "alpha=1; current observation initializes evidence",
        "zero_delta_rule": mode_rules[mode]["zero_delta"],
        "positive_delta_rule": "alpha=1-exp(-positive_dt/tau)",
        "padding_rule": "padding never calls the cell and leaves raw memory unchanged",
        "active_zero_delta_rule": mode_rules[mode]["zero_delta"],
        "active_invalid_dt_rule": mode_rules[mode]["invalid_dt"],
        "supported_modes": sorted(mode_rules),
        "mode_rules": mode_rules,
        "strict_skip_rule": "supported mode only: zero-delta rows are identity updates and do not enter F",
    }


__all__ = ["TIME_POLICY_VERSION", "time_policy_contract"]
