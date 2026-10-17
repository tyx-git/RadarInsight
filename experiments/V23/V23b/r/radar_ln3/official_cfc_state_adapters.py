"""External causal-state adapters for controlled V23 official-CfC experiments.

The adapters in this module deliberately operate *after* an unmodified
``ncps.torch.CfC`` call.  They never alter or reimplement ``CfCCell``.  Each
adapted state is fed back as the next official call's ``hx`` so the experiment
can isolate one historical project mechanism at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .data import DT_REASON_POSITIVE_DELTA, DT_REASON_ZERO_DELTA


ADAPTER_NONE = "none"
ADAPTER_POSITIVE_DT_FIXED_RETENTION = "positive_dt_fixed_retention"
ADAPTER_POSITIVE_DT_PER_DIMENSION_RETENTION = "positive_dt_per_dimension_retention"
ADAPTER_RECURRENT_LAYER_NORM = "recurrent_layer_norm"
ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY = "recurrent_layer_norm_scale_only"
ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION = "zero_delta_bounded_observation"
ADAPTER_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY = (
    "zero_delta_density_weighted_multiplicity"
)
ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY = (
    "positive_dt_fixed_retention_zero_delta_density_weighted_multiplicity"
)

ADAPTER_KINDS = (
    ADAPTER_NONE,
    ADAPTER_POSITIVE_DT_FIXED_RETENTION,
    ADAPTER_POSITIVE_DT_PER_DIMENSION_RETENTION,
    ADAPTER_RECURRENT_LAYER_NORM,
    ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY,
    ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION,
    ADAPTER_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY,
    ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY,
)


@dataclass(frozen=True)
class ExternalStateAdapterSpec:
    """Immutable, externally audited state-adapter definition."""

    kind: str
    hidden_dim: int
    fixed_tau_seconds: float | None = None
    zero_delta_gain: float | None = None
    layer_norm_eps: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ADAPTER_KINDS:
            raise ValueError(f"unsupported external state adapter: {self.kind}")
        if self.hidden_dim <= 0:
            raise ValueError("external state adapter hidden_dim must be positive")

    @property
    def trainable_parameter_count(self) -> int:
        if self.kind == ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY:
            return self.hidden_dim
        return 0

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return state-dict suffixes introduced by this adapter only."""

        if self.kind == ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY:
            return ("recurrent_norm.weight",)
        return ()

    @property
    def requires_causal_zero_ordinal(self) -> bool:
        """Whether the online adapter needs an externally retained ordinal."""

        return self.kind in (
            ADAPTER_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY,
            ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY,
        )

    def as_dict(self) -> dict[str, Any]:
        if self.kind == ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY:
            return {
                "kind": self.kind,
                "state_equation_owner": "official ncps.torch.CfC plus external post-state adapter",
                "applies_to": "every valid causal event",
                "formula": "h=LayerNorm_scale_only(h_official)",
                "layer_norm_eps": self.layer_norm_eps,
                "affine": True,
                "bias": False,
                "position": "post-official-state; emitted and fed back as the next official CfC hx",
                "trainable_parameter_count": self.trainable_parameter_count,
            }
        return {
            "kind": self.kind,
            "state_equation_owner": "official ncps.torch.CfC",
            "applies_to": "disabled",
            "formula": "h=h_official",
            "trainable_parameter_count": 0,
        }


class ExternalOfficialCfCStateAdapter(nn.Module):
    """Apply exactly one pre-registered state adaptation after official CfC."""

    def __init__(self, spec: ExternalStateAdapterSpec) -> None:
        super().__init__()
        self.spec = spec
        self.recurrent_norm: nn.LayerNorm | None
        self.log_tau_ratio: nn.Parameter | None
        if spec.kind in (
            ADAPTER_RECURRENT_LAYER_NORM,
            ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY,
        ):
            self.recurrent_norm = nn.LayerNorm(
                spec.hidden_dim,
                eps=float(spec.layer_norm_eps),
                elementwise_affine=True,
                bias=spec.kind == ADAPTER_RECURRENT_LAYER_NORM,
            )

    def specification(self) -> dict[str, Any]:
        return self.spec.as_dict()

    def _validate(
        self,
        official_state: torch.Tensor,
        previous_state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor | None,
    ) -> None:
        if official_state.ndim != 2 or official_state.shape[-1] != self.spec.hidden_dim:
            raise ValueError("external adapter official state has an invalid shape")
        if previous_state.shape != official_state.shape:
            raise ValueError("external adapter previous state shape differs from official state")
        if dt_seconds.ndim != 1 or dt_seconds.shape[0] != official_state.shape[0]:
            raise ValueError("external adapter dt must have shape [batch]")
        if dt_reason.ndim != 1 or dt_reason.shape != dt_seconds.shape:
            raise ValueError("external adapter dt_reason must have shape [batch]")
        if zero_ordinal is not None:
            if (
                zero_ordinal.ndim != 1
                or zero_ordinal.shape != dt_seconds.shape
                or zero_ordinal.dtype not in (torch.int64, torch.long)
                or bool((zero_ordinal < 1).any())
            ):
                raise ValueError(
                    "external adapter zero_ordinal must be positive long with shape [batch]"
                )
        if not all(torch.isfinite(value).all() for value in (official_state, previous_state, dt_seconds)):
            raise ValueError("external adapter inputs must be finite")

    def forward(
        self,
        official_state: torch.Tensor,
        previous_state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(
            official_state,
            previous_state,
            dt_seconds,
            dt_reason,
            zero_ordinal,
        )
        if self.spec.kind == ADAPTER_NONE:
            return official_state
        if self.spec.kind in (
            ADAPTER_RECURRENT_LAYER_NORM,
            ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY,
        ):
            if self.recurrent_norm is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("recurrent LayerNorm adapter is not initialized")
            return self.recurrent_norm(official_state)
        if self.spec.kind == ADAPTER_POSITIVE_DT_FIXED_RETENTION:
            tau = torch.as_tensor(
                self.spec.fixed_tau_seconds,
                dtype=official_state.dtype,
                device=official_state.device,
            )
            alpha = 1.0 - torch.exp(-dt_seconds.unsqueeze(-1) / tau)
            retained = (1.0 - alpha) * previous_state + alpha * official_state
            applies = (dt_reason == DT_REASON_POSITIVE_DELTA).unsqueeze(-1)
            return torch.where(applies, retained, official_state)
        if self.spec.kind == ADAPTER_POSITIVE_DT_PER_DIMENSION_RETENTION:
            if self.log_tau_ratio is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("per-dimension tau adapter is not initialized")
            base_tau = torch.as_tensor(
                self.spec.fixed_tau_seconds,
                dtype=official_state.dtype,
                device=official_state.device,
            )
            tau = base_tau * torch.exp(self.log_tau_ratio.to(dtype=official_state.dtype))
            alpha = 1.0 - torch.exp(-dt_seconds.unsqueeze(-1) / tau.unsqueeze(0))
            retained = (1.0 - alpha) * previous_state + alpha * official_state
            applies = (dt_reason == DT_REASON_POSITIVE_DELTA).unsqueeze(-1)
            return torch.where(applies, retained, official_state)
        if self.spec.kind == ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION:
            gain = torch.as_tensor(
                self.spec.zero_delta_gain,
                dtype=official_state.dtype,
                device=official_state.device,
            )
            bounded = (1.0 - gain) * previous_state + gain * official_state
            applies = (dt_reason == DT_REASON_ZERO_DELTA).unsqueeze(-1)
            return torch.where(applies, bounded, official_state)
        if self.spec.kind == ADAPTER_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY:
            ordinal = (
                torch.ones_like(dt_seconds, dtype=torch.long)
                if zero_ordinal is None
                else zero_ordinal
            )
            gain = torch.clamp(
                torch.as_tensor(
                    self.spec.zero_delta_gain,
                    dtype=official_state.dtype,
                    device=official_state.device,
                )
                * ordinal.to(dtype=official_state.dtype),
                max=3.0 * float(self.spec.zero_delta_gain),
            ).unsqueeze(-1)
            weighted = (1.0 - gain) * previous_state + gain * official_state
            applies = (dt_reason == DT_REASON_ZERO_DELTA).unsqueeze(-1)
            return torch.where(applies, weighted, official_state)
        if (
            self.spec.kind
            == ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY
        ):
            tau = torch.as_tensor(
                self.spec.fixed_tau_seconds,
                dtype=official_state.dtype,
                device=official_state.device,
            )
            positive_alpha = 1.0 - torch.exp(-dt_seconds.unsqueeze(-1) / tau)
            retained = (
                (1.0 - positive_alpha) * previous_state
                + positive_alpha * official_state
            )
            ordinal = (
                torch.ones_like(dt_seconds, dtype=torch.long)
                if zero_ordinal is None
                else zero_ordinal
            )
            zero_gain = torch.clamp(
                torch.as_tensor(
                    self.spec.zero_delta_gain,
                    dtype=official_state.dtype,
                    device=official_state.device,
                )
                * ordinal.to(dtype=official_state.dtype),
                max=3.0 * float(self.spec.zero_delta_gain),
            ).unsqueeze(-1)
            weighted = (1.0 - zero_gain) * previous_state + zero_gain * official_state
            positive = (dt_reason == DT_REASON_POSITIVE_DELTA).unsqueeze(-1)
            zero = (dt_reason == DT_REASON_ZERO_DELTA).unsqueeze(-1)
            return torch.where(positive, retained, torch.where(zero, weighted, official_state))
        raise RuntimeError(f"unhandled external state adapter: {self.spec.kind}")


__all__ = ["ADAPTER_KINDS", "ADAPTER_NONE", "ADAPTER_POSITIVE_DT_FIXED_RETENTION", "ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY", "ADAPTER_POSITIVE_DT_PER_DIMENSION_RETENTION", "ADAPTER_RECURRENT_LAYER_NORM", "ADAPTER_RECURRENT_LAYER_NORM_SCALE_ONLY", "ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION", "ADAPTER_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY", "ExternalOfficialCfCStateAdapter", "ExternalStateAdapterSpec"]