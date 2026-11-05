"""AeroSight ``.asd`` replay source for the R-RTSP server."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import zlib

import numpy as np

from radar_ln3.data.asd import load_asd
from radar_ln3.data.contracts import EventSequence

from .base import RRTSPEvent, RRTSPRDFrame


_TRACK_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "speed_mps",
    "course_deg",
    "snr_db",
    "amplitude_db",
)
MODEL_CONTEXT_EVENTS = 29
QUEUE_MAX_EVENTS = 60
# Backward-compatible name used by existing callers. It describes the
# default causal model context, not the queue-mode source capacity.
MAX_PRODUCT_EVENTS = MODEL_CONTEXT_EVENTS


def _stable_track_id(value: object) -> int:
    text = str(value).strip()
    try:
        parsed = int(text, 10)
    except (TypeError, ValueError):
        parsed = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
    return parsed & 0xFFFFFFFF


class ASDStreamSource:
    """Load one ASD parent and expose its first causal events to R-RTSP.

    The shared ASD adapter performs timestamp conversion, finite-value checks,
    raw-axis canonicalization and the per-event RD frame cap. Causal replay
    normally requests 29 events; queue replay may request up to 60 source
    events and applies the model's 29-event sliding window later.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_unit: str | None = None,
        amplitude_mode: str | None = None,
        max_events: int = MAX_PRODUCT_EVENTS,
    ) -> None:
        self.path = Path(path).resolve()
        if isinstance(max_events, bool) or not 1 <= int(max_events) <= QUEUE_MAX_EVENTS:
            raise ValueError(f"max_events must be between 1 and {QUEUE_MAX_EVENTS}")
        sequence = load_asd(
            self.path,
            timestamp_unit=timestamp_unit,
            amplitude_mode=amplitude_mode,
            max_events=max_events,
        )
        source_events = list(sequence.events)
        if not source_events:
            raise ValueError(f"ASD package contains no events: {self.path}")
        self.sequence = sequence
        safe_events = [
            replace(event, label_id=4, label_name="UnknownTarget")
            for event in source_events
        ]
        self.model_sequence = EventSequence(
            parent_sample_id=sequence.parent_sample_id,
            track_id=sequence.track_id,
            label_id=4,
            label_name="UnknownTarget",
            split_name="inference",
            events=safe_events,
            source_name="asd",
            metadata={
                "source_name": "asd",
                "timestamp_unit": sequence.metadata.get("timestamp_unit"),
                "amplitude_mode": sequence.metadata.get("amplitude_mode"),
                "model_input_event_count": len(safe_events),
            },
        )
        self.original_event_count = int(
            sequence.metadata.get("original_event_count", len(source_events))
        )
        self.truncated = bool(sequence.metadata.get("sequence_truncated", False))
        self._events = tuple(self._convert_event(event) for event in source_events)
        self._first_rd_shape = next(
            (
                tuple(int(value) for value in frame.amplitude.shape)
                for event in self._events
                for frame in event.rd_frames
            ),
            None,
        )
        self._metadata = {
            "source_path": str(self.path),
            "source_name": "asd",
            "parent_sample_id": sequence.parent_sample_id,
            "track_id": sequence.track_id,
            "label_id": sequence.label_id,
            "label_name": sequence.label_name,
            "original_event_count": self.original_event_count,
            "event_count": len(self._events),
            "sequence_truncated": self.truncated,
            "timestamp_unit": sequence.metadata.get("timestamp_unit"),
            "amplitude_mode": sequence.metadata.get("amplitude_mode"),
            "canonical_rd_shape": list(self._first_rd_shape)
            if self._first_rd_shape is not None
            else None,
        }

    @property
    def events(self) -> tuple[RRTSPEvent, ...]:
        return self._events

    @property
    def has_rd(self) -> bool:
        return any(event.rd_frames for event in self._events)

    @property
    def first_rd_shape(self) -> tuple[int, int] | None:
        return self._first_rd_shape

    @property
    def source_name(self) -> str:
        return "asd"

    @property
    def metadata(self) -> Mapping[str, object]:
        return dict(self._metadata)

    def _convert_event(self, event) -> RRTSPEvent:
        tr_values = {
            name: event.tr_values[name]
            for name in _TRACK_FIELDS
            if isinstance(event.tr_values, Mapping) and name in event.tr_values
        }
        timestamp = event.tr_time_seconds
        if timestamp is None and isinstance(event.tr_values, Mapping):
            timestamp = event.tr_values.get("time_seconds")
        if timestamp is None:
            raise ValueError(f"ASD event has no timestamp: {event.causal_event_index}")
        frames: list[RRTSPRDFrame] = []
        rd_values = event.rd_values if event.rd_available else None
        if isinstance(rd_values, Mapping):
            for frame in rd_values.get("frames", ()):
                if not isinstance(frame, Mapping):
                    raise ValueError("ASD RD frame must be an object")
                frames.append(
                    RRTSPRDFrame(
                        time_seconds=float(frame.get("time_seconds", 0.0)),
                        amplitude=np.asarray(frame["amplitude"], dtype=np.float32),
                        phase=(
                            None
                            if frame.get("phase") is None
                            else np.asarray(frame["phase"], dtype=np.float32)
                        ),
                        metadata={
                            key: value
                            for key, value in frame.items()
                            if key not in {"amplitude", "phase"}
                        },
                    )
                )
        return RRTSPEvent(
            event_index=int(event.causal_event_index),
            timestamp_seconds=float(timestamp),
            track_id=_stable_track_id(event.track_id),
            # ASD labels are evaluation ground truth, never radar observations.
            state=0,
            class_id=5,
            tr_values=tr_values,
            rd_frames=tuple(frames),
            metadata={
                "sample_id": event.sample_id,
                "parent_sample_id": event.parent_sample_id,
                "parent_event_count": len(self.sequence.events),
                "source_track_id": str(event.track_id),
                "label_name": str(event.label_name),
            },
            parent_sample_id=str(event.parent_sample_id),
            model_event=self.model_sequence.events[int(event.causal_event_index)],
            model_sequence=self.model_sequence,
        )


class ASDDirectoryStreamSource:
    """Replay ASD packages as interleaved radar scans.

    Each scan contributes at most one causal point from every package.  This
    preserves the time order within each target while presenting the directory
    as a multi-target radar stream instead of playing one target to completion
    before starting the next one.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_unit: str | None = None,
        amplitude_mode: str | None = None,
        max_events_per_package: int = MODEL_CONTEXT_EVENTS,
        send_interval_seconds: float = 1.0,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"ASD directory does not exist: {self.path}")
        if isinstance(max_events_per_package, bool) or not 1 <= int(max_events_per_package) <= QUEUE_MAX_EVENTS:
            raise ValueError(
                f"max_events_per_package must be between 1 and {QUEUE_MAX_EVENTS}"
            )
        if float(send_interval_seconds) <= 0.0:
            raise ValueError("send_interval_seconds must be positive")
        paths = sorted(self.path.glob("*.asd"), key=lambda item: item.name.casefold())
        if not paths:
            raise FileNotFoundError(f"No .asd packages found in: {self.path}")

        sources = [
            ASDStreamSource(
                item,
                timestamp_unit=timestamp_unit,
                amplitude_mode=amplitude_mode,
                max_events=max_events_per_package,
            )
            for item in paths
        ]
        self.sources = tuple(sources)
        events: list[RRTSPEvent] = []
        used_track_ids: set[int] = set()
        track_id_by_parent: dict[str, int] = {}
        elapsed = 0.0
        scan_count = max((len(source.events) for source in sources), default=0)
        max_scan_targets = 0
        for scan_index in range(scan_count):
            candidates = [
                (package_index, source, source.events[scan_index])
                for package_index, source in enumerate(sources)
                if scan_index < len(source.events)
            ]
            target_count = len(candidates)
            max_scan_targets = max(max_scan_targets, target_count)
            for package_index, source, event in candidates:
                local_index = scan_index
                parent_id = str(event.parent_sample_id)
                track_id = track_id_by_parent.get(parent_id)
                if track_id is None:
                    track_id = int(event.track_id)
                if parent_id not in track_id_by_parent and track_id in used_track_ids:
                    identity = parent_id.encode("utf-8")
                    track_id = zlib.crc32(identity) & 0xFFFFFFFF
                    while track_id in used_track_ids:
                        track_id = (track_id + 1) & 0xFFFFFFFF
                if parent_id not in track_id_by_parent:
                    track_id_by_parent[parent_id] = track_id
                    used_track_ids.add(track_id)
                events.append(
                    replace(
                        event,
                        event_index=len(events),
                        timestamp_seconds=elapsed,
                        track_id=track_id,
                        metadata={
                            **dict(event.metadata),
                            "package_index": package_index,
                            "package_path": str(source.path),
                            "package_event_index": local_index,
                            "scan_index": scan_index,
                            "scan_target_count": target_count,
                        },
                    )
                )
            if candidates:
                elapsed += float(send_interval_seconds)
        self._events = tuple(events)
        self._first_rd_shape = next(
            (
                tuple(int(value) for value in frame.amplitude.shape)
                for event in self._events
                for frame in event.rd_frames
            ),
            None,
        )
        self._metadata = {
            "source_name": "asd-directory",
            "source_path": str(self.path),
            "package_count": len(sources),
            "package_names": [item.path.name for item in sources],
            "package_event_counts": [len(item.events) for item in sources],
            "package_original_event_counts": [
                item.original_event_count for item in sources
            ],
            "truncated_package_count": sum(item.truncated for item in sources),
            "event_count": len(self._events),
            "scan_count": scan_count,
            "scan_target_count_max": max_scan_targets,
            "send_interval_seconds": float(send_interval_seconds),
            "replay_mode": "round_robin_scan",
            "canonical_rd_shape": list(self._first_rd_shape)
            if self._first_rd_shape is not None
            else None,
        }

    @property
    def events(self) -> tuple[RRTSPEvent, ...]:
        return self._events

    @property
    def has_rd(self) -> bool:
        return any(event.rd_frames for event in self._events)

    @property
    def first_rd_shape(self) -> tuple[int, int] | None:
        return self._first_rd_shape

    @property
    def source_name(self) -> str:
        return "asd-directory"

    @property
    def metadata(self) -> Mapping[str, object]:
        return dict(self._metadata)


__all__ = [
    "ASDDirectoryStreamSource",
    "ASDStreamSource",
    "MODEL_CONTEXT_EVENTS",
    "QUEUE_MAX_EVENTS",
    "MAX_PRODUCT_EVENTS",
]
