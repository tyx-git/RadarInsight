"""r22 L2 model with the r20 structured radial sidecar contract."""

from __future__ import annotations

from typing import Any

from v27.V27d.r20.model import R20L2Model, R20RadialAdapter

from .protocol import R22_RATE_SCALE


class R22L2Model(R20L2Model):
    """The r20 architecture with an explicit r22 model contract."""

    def model_contract(self) -> dict[str, Any]:
        contract = dict(super().model_contract())
        contract["schema_version"] = "RADAR-LN3-V27d-r22-l2-model-contract-v1"
        contract["experiment_id"] = "V27d-r22"
        contract["radial_rate_scale_mps_per_s"] = R22_RATE_SCALE
        contract["r20_parent_rate_scale_mps_per_s"] = 100.0
        contract["main_variable"] = "radial_rate_scale_mps_per_s: 100.0 -> 10.0"
        return contract


R22RadialAdapter = R20RadialAdapter

__all__ = ["R22L2Model", "R22RadialAdapter"]
