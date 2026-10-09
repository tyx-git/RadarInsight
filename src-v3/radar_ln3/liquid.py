"""Canonical entry points for Radar-LN3 liquid backends.

The current product deliberately uses the project-owned liquid-style encoder
implemented in :mod:`radar_ln3.model`.  Keeping construction here makes the
backend choice explicit without changing the module/class name layout used by
historical checkpoints.  The official ``ncps.torch.CfC`` implementation stays
in ``official_cfc.py`` as an experimental backend and is never selected by
this module implicitly.
"""

from __future__ import annotations

from typing import Any

from .model import CfCLiquidCausalEncoder


CURRENT_LIQUID_BACKEND = "radar_ln3.liquid.build_cfc_liquid_encoder"
CURRENT_LIQUID_IMPLEMENTATION = "radar_ln3.model.CfCLiquidCausalEncoder"
EXPERIMENTAL_OFFICIAL_LIQUID_BACKEND = "ncps.torch.CfC"


def build_cfc_liquid_encoder(
    input_dim: int,
    hidden_dim: int,
    *,
    tau_min_seconds: float,
    tau_max_seconds: float,
    zero_dt_event_floor_seconds: float,
    dt_max_seconds: float,
    invalid_dt_update_mode: str,
    invalid_dt_observation_gain: float,
    state_normalization_mode: str,
) -> CfCLiquidCausalEncoder:
    """Build the current self-developed causal liquid encoder.

    All arguments are forwarded unchanged.  This function intentionally does
    not add defaults: product and experiment callers must bind every dynamics
    choice from their versioned configuration.
    """

    return CfCLiquidCausalEncoder(
        input_dim,
        hidden_dim,
        tau_min_seconds=tau_min_seconds,
        tau_max_seconds=tau_max_seconds,
        zero_dt_event_floor_seconds=zero_dt_event_floor_seconds,
        dt_max_seconds=dt_max_seconds,
        invalid_dt_update_mode=invalid_dt_update_mode,
        invalid_dt_observation_gain=invalid_dt_observation_gain,
        state_normalization_mode=state_normalization_mode,
    )


def current_liquid_backend_binding() -> dict[str, Any]:
    """Return the explicit backend binding for product provenance records."""

    return {
        "backend": "self_developed_cfc_style",
        "class_path": CURRENT_LIQUID_BACKEND,
        "implementation_path": CURRENT_LIQUID_IMPLEMENTATION,
        "official_backend": EXPERIMENTAL_OFFICIAL_LIQUID_BACKEND,
        "official_backend_default": False,
    }


__all__ = [
    "CURRENT_LIQUID_BACKEND",
    "CURRENT_LIQUID_IMPLEMENTATION",
    "EXPERIMENTAL_OFFICIAL_LIQUID_BACKEND",
    "build_cfc_liquid_encoder",
    "current_liquid_backend_binding",
]
