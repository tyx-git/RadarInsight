"""Transport-level source contracts.

The source contract contains physical observations only.  It is intentionally
smaller than the full model feature batch and therefore remains usable by
clients that do not run RadarInsight inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RRTSPRDFrame:
    """One finite RD frame associated with an event."""

    time_seconds: float
    amplitude: np.ndarray
    phase: np.ndarray | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        amplitude = np.asarray(self.amplitude, dtype=np.float32)
        if amplitude.ndim != 2 or min(amplitude.shape) <= 0:
            raise ValueError(f"RD amplitude must be a non-empty 2D array: {amplitude.shape}")
        if not np.isfinite(amplitude).all():
            raise ValueError("RD amplitude contains non-finite values")
        if self.phase is not None:
            phase = np.asarray(self.phase, dtype=np.float32)
            if phase.shape != amplitude.shape:
                raise ValueError("RD amplitude and phase shapes differ")
            if not np.isfinite(phase).all():
                raise ValueError("RD phase contains non-finite values")
        if not np.isfinite(float(self.time_seconds)):
            raise ValueError("RD frame time must be finite")


@dataclass(frozen=True)
class RRTSPEvent:
    """One ordered track update and its optional RD frames."""

    event_index: int
    timestamp_seconds: float
    track_id: int
    state: int
    class_id: int
    tr_values: Mapping[str, object]
    rd_frames: tuple[RRTSPRDFrame, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    parent_sample_id: str = ""
    model_event: Any | None = None
    model_sequence: Any | None = None

    def __post_init__(self) -> None:
        if int(self.event_index) < 0:
            raise ValueError("event_index must be non-negative")
        if not np.isfinite(float(self.timestamp_seconds)):
            raise ValueError("event timestamp must be finite")
        if not 0 <= int(self.track_id) <= 0xFFFFFFFF:
            raise ValueError("track_id must fit uint32")
        if not 0 <= int(self.state) <= 0xFF or not 0 <= int(self.class_id) <= 0xFF:
            raise ValueError("state/class_id must fit uint8")


class StreamSource(Protocol):
    """Source interface consumed by the R-RTSP session."""

    @property
    def events(self) -> Sequence[RRTSPEvent]:
        ...

    @property
    def has_rd(self) -> bool:
        ...

    @property
    def first_rd_shape(self) -> tuple[int, int] | None:
        ...

    @property
    def source_name(self) -> str:
        ...

    @property
    def metadata(self) -> Mapping[str, object]:
        ...


__all__ = ["RRTSPEvent", "RRTSPRDFrame", "StreamSource"]
