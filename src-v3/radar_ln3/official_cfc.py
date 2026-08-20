"""Official ``ncps`` Closed-form Continuous-time (CfC) adapter for V23.

The project historically used a locally implemented, CfC-inspired recurrent
unit.  This module deliberately does *not* reimplement the CfC equations:
the recurrent operator is ``ncps.torch.CfC`` from the Neural Circuit Policies
project.  The surrounding code only applies Radar-LN3's causal padding and
``dt`` tensor contracts.
"""

from __future__ import annotations

import json
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)


OFFICIAL_CFC_DISTRIBUTION = "ncps"
OFFICIAL_CFC_DISTRIBUTION_VERSION = "1.0.1"
OFFICIAL_CFC_GIT_COMMIT = "695ec22d7d1831f2ced67346ef7d2f08c525bd47"
OFFICIAL_CFC_CLASS_PATH = "ncps.torch.CfC"
OFFICIAL_CFC_MODE = "default"
OFFICIAL_CFC_ACTIVATION = "lecun_tanh"


@dataclass
class OfficialCfCForwardState:
    """CfC sequence output plus Radar-LN3 causal-time audit masks."""

    sequence: torch.Tensor
    final_state: torch.Tensor
    dt_clamped_mask: torch.Tensor
    dt_zero_mask: torch.Tensor
    dt_invalid_mask: torch.Tensor
    dt_first_event_mask: torch.Tensor | None = None
    dt_zero_delta_mask: torch.Tensor | None = None


def official_cfc_runtime_binding() -> dict[str, Any]:
    """Return and validate the exact official package installed for V23.

    ``ncps.__version__`` is not a reliable distribution-version source in the
    current upstream package, so this deliberately uses package metadata and
    the PEP 610 direct-url record instead.
    """

    try:
        from ncps.torch import CfC
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "V23 requires the official ncps package. Install the pinned "
            "src-v3/requirements-v23-official-cfc.txt dependency into .venv."
        ) from error
    site_packages = Path(sysconfig.get_paths()["purelib"])
    dist_infos = sorted(site_packages.glob("ncps-*.dist-info"))
    if len(dist_infos) != 1:
        raise RuntimeError(
            "V23 requires exactly one installed ncps distribution metadata directory; "
            f"found {len(dist_infos)} under {site_packages}"
        )
    dist_info = dist_infos[0]
    metadata_path = dist_info / "METADATA"
    direct_url_path = dist_info / "direct_url.json"
    if not metadata_path.is_file() or not direct_url_path.is_file():
        raise RuntimeError("V23 official ncps installation lacks direct_url.json")
    version = next(
        (
            line.partition(":")[2].strip()
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Version:")
        ),
        "",
    )
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    vcs_info = dict(direct_url.get("vcs_info") or {})
    commit = str(vcs_info.get("commit_id") or "")
    if version != OFFICIAL_CFC_DISTRIBUTION_VERSION or commit != OFFICIAL_CFC_GIT_COMMIT:
        raise RuntimeError(
            "V23 requires official ncps=="
            f"{OFFICIAL_CFC_DISTRIBUTION_VERSION} at commit "
            f"{OFFICIAL_CFC_GIT_COMMIT}; found version={version!r}, "
            f"commit={commit!r}"
        )
    if CfC.__module__ != "ncps.torch.cfc":
        raise RuntimeError("V23 CfC import is not the official ncps.torch.cfc class")
    return {
        "distribution": OFFICIAL_CFC_DISTRIBUTION,
        "distribution_version": version,
        "direct_url": str(direct_url.get("url") or ""),
        "git_commit": commit,
        "class_path": OFFICIAL_CFC_CLASS_PATH,
        "class_module": CfC.__module__,
    }


class OfficialCfCCausalEncoder(nn.Module):
    """Causal adapter around the unmodified official ``ncps.torch.CfC``.

    V23 intentionally gives official CfC the native event time spans: first
    observations and same-second observations have a zero time span, while
    positive deltas retain their measured, bounded wall-clock duration.  No
    project-specific floor, observation gain, state normalization, or hand
    written state equation is applied inside this adapter.

    The upstream ``ncps==1.0.1`` sequence wrapper squeezes the per-event
    time-span tensor before passing it to ``CfCCell``.  Calling it one parent
    at a time preserves its official implementation while avoiding that
    upstream batch-time-span broadcast limitation.  This is acceptable for
    Radar-LN3's fixed batch size of two and, importantly, it lets each parent
    stop at its true causal length: no padding event reaches the official CfC
    recurrence at all.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dt_max_seconds: float,
        mode: str = OFFICIAL_CFC_MODE,
        activation: str = OFFICIAL_CFC_ACTIVATION,
        backbone_units: int | None = None,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        mixed_memory: bool = False,
    ) -> None:
        super().__init__()
        runtime = official_cfc_runtime_binding()
        if mode != OFFICIAL_CFC_MODE:
            raise ValueError("V23 fixes the official CfC mode to default")
        if activation != OFFICIAL_CFC_ACTIVATION:
            raise ValueError("V23 fixes the official CfC activation to lecun_tanh")
        if backbone_layers != 1 or backbone_dropout != 0.0 or mixed_memory:
            raise ValueError("V23 fixes one dropout-free, non-mixed official CfC layer")
        if input_dim <= 0 or hidden_dim <= 0 or dt_max_seconds <= 0.0:
            raise ValueError("V23 official CfC dimensions/dt maximum are invalid")
        units = int(hidden_dim)
        backbone = units if backbone_units is None else int(backbone_units)
        if backbone != units:
            raise ValueError("V23 fixes official CfC backbone_units equal to hidden_dim")
        try:
            from ncps.torch import CfC
        except ImportError as error:  # pragma: no cover - checked above
            raise RuntimeError("V23 official ncps.CfC became unavailable") from error
        self.input_dim = int(input_dim)
        self.hidden_dim = units
        self.dt_max_seconds = float(dt_max_seconds)
        self.runtime_binding = runtime
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone
        self.backbone_layers = int(backbone_layers)
        self.backbone_dropout = float(backbone_dropout)
        self.mixed_memory = bool(mixed_memory)
        self.cfc = CfC(
            self.input_dim,
            self.hidden_dim,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode=self.mode,
            activation=self.activation,
            backbone_units=self.backbone_units,
            backbone_layers=self.backbone_layers,
            backbone_dropout=self.backbone_dropout,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def specification(self) -> dict[str, Any]:
        return {
            **self.runtime_binding,
            "mode": self.mode,
            "activation": self.activation,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
            "mixed_memory": self.mixed_memory,
            "return_sequences": True,
            "batch_first": True,
            "dt_semantics": (
                "official_cfc_raw_nonnegative_wall_clock_seconds; "
                "first_event_and_same_second_use_zero_timespan; "
                "positive_delta_preserves_measured_dt; "
                "only_positive_dt_is_capped"
            ),
            "parameter_count": self.parameter_count,
        }

    @staticmethod
    def _validate_reason_tensors(
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor | None,
        *,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if dt_reason is None:
            return None, None
        if dt_reason.shape != dt_seconds.shape or dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("V23 official CfC dt_reason must be long with shape [batch, sequence]")
        active = ~padding_mask
        padded_reasons = dt_reason.masked_select(padding_mask)
        active_reasons = dt_reason.masked_select(active)
        if bool((padded_reasons != DT_REASON_PADDING).any()):
            raise ValueError("V23 padded events must use the padding dt reason")
        if bool((active_reasons == DT_REASON_PADDING).any()):
            raise ValueError("V23 active events must not use the padding dt reason")
        first = (dt_reason == DT_REASON_FIRST_EVENT) & active
        if initial_state is None:
            first_indices = active.to(dtype=torch.int64).argmax(dim=1)
            batch_indices = torch.arange(dt_seconds.shape[0], device=dt_seconds.device)
            if bool((first.sum(dim=1) != 1).any()) or bool(
                (dt_reason[batch_indices, first_indices] != DT_REASON_FIRST_EVENT).any()
            ):
                raise ValueError("fresh V23 sequences require first-event at the first active observation")
        elif bool(first.any()):
            raise ValueError("continued V23 sequences cannot contain first-event dt reasons")
        positive = dt_reason == DT_REASON_POSITIVE_DELTA
        zero_like = (dt_reason == DT_REASON_FIRST_EVENT) | (dt_reason == DT_REASON_ZERO_DELTA)
        if bool(dt_seconds.masked_select(zero_like & active).abs().gt(1.0e-9).any()):
            raise ValueError("V23 first/same-second observations require raw dt=0")
        if bool(dt_seconds.masked_select(positive & active).le(0.0).any()):
            raise ValueError("V23 positive-delta observations require raw dt>0")
        if not torch.equal(positive & active, dt_valid & active):
            raise ValueError("V23 dt_reason and dt_valid are inconsistent")
        return first, (dt_reason == DT_REASON_ZERO_DELTA) & active

    def _timespans(
        self,
        dt_seconds: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.isfinite(dt_seconds).all():
            raise ValueError("V23 official CfC dt must be finite")
        negative = dt_seconds < 0.0
        if bool(negative.masked_select(active).any()):
            raise ValueError("V23 official CfC does not accept negative wall-clock dt")
        clamped = (dt_seconds > self.dt_max_seconds) & active
        safe = dt_seconds.clamp(min=0.0, max=self.dt_max_seconds)
        safe = torch.where(active, safe, torch.zeros_like(safe))
        zero = (dt_seconds.abs() <= 1.0e-9) & active
        return safe, clamped, zero

    def forward(
        self,
        events: torch.Tensor,
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        dt_valid: torch.Tensor | None = None,
        dt_reason: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> OfficialCfCForwardState:
        if events.ndim != 3:
            raise ValueError("V23 official CfC events must have shape [batch, sequence, features]")
        if events.shape[-1] != self.input_dim:
            raise ValueError("V23 official CfC event feature dimension differs from its config")
        if dt_seconds.shape != events.shape[:2] or padding_mask.shape != events.shape[:2]:
            raise ValueError("V23 official CfC dt/padding shapes must match events")
        if dt_valid is None:
            dt_valid = torch.ones_like(dt_seconds, dtype=torch.bool)
        if dt_valid.shape != dt_seconds.shape or dt_valid.dtype != torch.bool:
            raise ValueError("V23 official CfC dt_valid must be bool with shape [batch, sequence]")
        if not torch.isfinite(events).all():
            raise ValueError("V23 official CfC events must be finite")
        active = ~padding_mask
        if bool((active.sum(dim=1) <= 0).any()):
            raise ValueError("V23 official CfC cannot consume an all-padding sequence")
        lengths = active.to(dtype=torch.long).sum(dim=1)
        expected_active = torch.arange(events.shape[1], device=events.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(active, expected_active):
            raise ValueError("V23 official CfC requires right-sided causal padding")
        if initial_state is not None and initial_state.shape != (events.shape[0], self.hidden_dim):
            raise ValueError("V23 official CfC initial state shape is invalid")
        first_mask, zero_delta_mask = self._validate_reason_tensors(
            dt_seconds,
            padding_mask,
            dt_valid,
            dt_reason,
            initial_state=initial_state,
        )
        timespans, clamped_mask, zero_mask = self._timespans(dt_seconds, active)
        parent_sequences: list[torch.Tensor] = []
        parent_final_states: list[torch.Tensor] = []
        for batch_index, length in enumerate(lengths.tolist()):
            hx = None if initial_state is None else initial_state[batch_index : batch_index + 1]
            parent_sequence, parent_final = self.cfc(
                events[batch_index : batch_index + 1, :length],
                hx=hx,
                timespans=timespans[batch_index : batch_index + 1, :length],
            )
            if parent_sequence.shape != (1, length, self.hidden_dim) or parent_final.shape != (
                1,
                self.hidden_dim,
            ):
                raise RuntimeError("official ncps.CfC returned an unexpected per-parent shape")
            parent_sequences.append(parent_sequence.squeeze(0))
            parent_final_states.append(parent_final)
        sequence = events.new_zeros((events.shape[0], events.shape[1], self.hidden_dim))
        for batch_index, parent_sequence in enumerate(parent_sequences):
            sequence[batch_index, : parent_sequence.shape[0]] = parent_sequence
        final_state = torch.cat(parent_final_states, dim=0)
        return OfficialCfCForwardState(
            sequence=sequence,
            final_state=final_state,
            dt_clamped_mask=clamped_mask,
            dt_zero_mask=zero_mask,
            dt_invalid_mask=(~dt_valid) & active,
            dt_first_event_mask=(
                torch.zeros_like(active) if first_mask is None else first_mask
            ),
            dt_zero_delta_mask=(
                torch.zeros_like(active) if zero_delta_mask is None else zero_delta_mask
            ),
        )

    def step(
        self,
        event: torch.Tensor,
        dt_seconds: torch.Tensor,
        state: torch.Tensor | None = None,
        dt_valid: torch.Tensor | None = None,
        dt_reason: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event.ndim != 2 or event.shape[-1] != self.input_dim:
            raise ValueError("V23 official CfC online event must have shape [batch, features]")
        if dt_seconds.ndim != 1 or dt_seconds.shape[0] != event.shape[0]:
            raise ValueError("V23 official CfC online dt must have shape [batch]")
        if dt_valid is None:
            dt_valid = torch.ones_like(dt_seconds, dtype=torch.bool)
        if dt_valid.shape != dt_seconds.shape or dt_valid.dtype != torch.bool:
            raise ValueError("V23 official CfC online dt_valid must be bool with shape [batch]")
        if dt_reason is not None:
            if dt_reason.shape != dt_seconds.shape or dt_reason.dtype not in (torch.int64, torch.long):
                raise ValueError("V23 official CfC online dt_reason shape is invalid")
            first = dt_reason == DT_REASON_FIRST_EVENT
            if state is None and not bool(first.all()):
                raise ValueError("fresh V23 official CfC online state requires first-event reasons")
            if state is not None and bool(first.any()):
                raise ValueError("existing V23 official CfC online state cannot receive first-event")
            allowed = first | (dt_reason == DT_REASON_ZERO_DELTA) | (dt_reason == DT_REASON_POSITIVE_DELTA)
            if not bool(allowed.all()):
                raise ValueError("V23 official CfC online dt_reason contains padding/unsupported code")
            positive = dt_reason == DT_REASON_POSITIVE_DELTA
            zero_like = first | (dt_reason == DT_REASON_ZERO_DELTA)
            if bool(dt_seconds.masked_select(zero_like).abs().gt(1.0e-9).any()) or bool(
                dt_seconds.masked_select(positive).le(0.0).any()
            ):
                raise ValueError("V23 official CfC online dt_reason/raw-dt mismatch")
            if not torch.equal(positive, dt_valid):
                raise ValueError("V23 official CfC online dt_reason/dt_valid mismatch")
        if state is not None and state.shape != (event.shape[0], self.hidden_dim):
            raise ValueError("V23 official CfC online state shape is invalid")
        active = torch.ones_like(dt_seconds, dtype=torch.bool)
        timespans, clamped_mask, zero_mask = self._timespans(dt_seconds, active)
        next_states: list[torch.Tensor] = []
        for batch_index in range(event.shape[0]):
            hx = None if state is None else state[batch_index : batch_index + 1]
            output, next_state = self.cfc(
                event[batch_index : batch_index + 1].unsqueeze(1),
                hx=hx,
                timespans=timespans[batch_index : batch_index + 1].unsqueeze(1),
            )
            if output.shape != (1, 1, self.hidden_dim) or next_state.shape != (1, self.hidden_dim):
                raise RuntimeError("official ncps.CfC returned an unexpected online state shape")
            next_states.append(next_state)
        return torch.cat(next_states, dim=0), clamped_mask, zero_mask

    @staticmethod
    def readout(state: torch.Tensor) -> torch.Tensor:
        """The official CfC output is the hidden state because proj_size is None."""

        return state


__all__ = [
    "OFFICIAL_CFC_ACTIVATION",
    "OFFICIAL_CFC_CLASS_PATH",
    "OFFICIAL_CFC_DISTRIBUTION",
    "OFFICIAL_CFC_DISTRIBUTION_VERSION",
    "OFFICIAL_CFC_GIT_COMMIT",
    "OFFICIAL_CFC_MODE",
    "OfficialCfCCausalEncoder",
    "OfficialCfCForwardState",
    "official_cfc_runtime_binding",
]
