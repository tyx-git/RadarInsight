# -*- coding: utf-8 -*-
"""Connection-scoped history of received track measurements, independent of UI limits."""
from __future__ import annotations

import math
from dataclasses import dataclass

from aerosight.core.radar import RadarFrame


RTP_CLOCK_HZ = 90_000
TrackKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class TrackAnalysisSample:
    elapsed_seconds: float
    radial_speed_mps: float | None
    altitude_m: float | None


@dataclass
class _StreamClock:
    last_frame_id: int | None = None
    last_rtp_timestamp: int | None = None
    last_source_seconds: float | None = None
    elapsed_seconds: float = 0.0

    def advance(self, frame: RadarFrame) -> float | None:
        # Frame IDs increase within one receiver connection. Re-rendering or
        # receiving a duplicate/stale frame must not append measurements twice.
        if self.last_frame_id is not None and frame.frame_id <= self.last_frame_id:
            return None
        rtp = frame.rtp_timestamp
        if rtp is not None:
            rtp = int(rtp) & 0xFFFFFFFF
            source_seconds = rtp / RTP_CLOCK_HZ
        else:
            source_seconds = float(frame.timestamp)
        if not math.isfinite(source_seconds):
            return None

        delta = 0.0
        if self.last_source_seconds is not None:
            if rtp is not None and self.last_rtp_timestamp is not None:
                ticks = (rtp - self.last_rtp_timestamp) & 0xFFFFFFFF
                # A small forward difference also covers the 32-bit RTP wrap.
                # Backwards timestamps are stale data, not a 13-hour jump.
                if ticks >= 0x80000000:
                    return None
                delta = ticks / RTP_CLOCK_HZ
            else:
                delta = source_seconds - self.last_source_seconds
                if delta < 0.0:
                    return None

        self.elapsed_seconds += delta
        self.last_frame_id = int(frame.frame_id)
        self.last_rtp_timestamp = rtp
        self.last_source_seconds = source_seconds
        return self.elapsed_seconds


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


class TrackAnalysisHistory:
    """Record every delivered batch; no ASD reads, interpolation, or timers."""

    def __init__(self):
        self._clocks: dict[str, _StreamClock] = {}
        self._origins: dict[TrackKey, float] = {}
        self._samples: dict[TrackKey, list[TrackAnalysisSample]] = {}

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(self._clocks)

    def record_frame(
        self, frame: RadarFrame,
    ) -> dict[TrackKey, tuple[TrackAnalysisSample, ...]]:
        stream_id = str(frame.stream_id)
        clock = self._clocks.setdefault(stream_id, _StreamClock())
        elapsed = clock.advance(frame)
        if elapsed is None:
            return {}

        added: dict[TrackKey, tuple[TrackAnalysisSample, ...]] = {}
        for observation in frame.observations:
            key = (stream_id, int(observation.track_id))
            if key in added:
                continue
            origin = self._origins.setdefault(key, elapsed)
            sample = TrackAnalysisSample(
                elapsed_seconds=elapsed - origin,
                radial_speed_mps=_finite_or_none(observation.radial_speed_mps),
                altitude_m=_finite_or_none(observation.altitude_m),
            )
            self._samples.setdefault(key, []).append(sample)
            added[key] = (sample,)
        return added

    def samples_for(self, stream_id: str, track_id: int) -> tuple[TrackAnalysisSample, ...]:
        return tuple(self._samples.get((str(stream_id), int(track_id)), ()))

    def reset_stream(self, stream_id: str):
        """Start a new connection segment without affecting other radars."""
        self._clocks.pop(stream_id, None)
        for key in list(self._samples):
            if key[0] == stream_id:
                self._samples.pop(key)
                self._origins.pop(key)

    def clear(self):
        self._clocks.clear()
        self._origins.clear()
        self._samples.clear()
