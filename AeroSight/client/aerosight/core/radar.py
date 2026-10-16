# -*- coding: utf-8 -*-
"""Radar data objects shared between AeroSight processes."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from aerosight.codec.decoder import spherical_to_xyz


@dataclass
class RadarObservation:
    """One radar track at one point in time."""

    stream_id: str
    track_id: int
    state: str = ""
    # Radar-provided class; this is not the ONNX classification result.
    radar_class: str = ""
    range_m: float = 0.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    ground_range_m: float = 0.0
    altitude_m: float = 0.0
    radial_speed_mps: float = 0.0
    speed_mps: float = 0.0
    course_deg: float = 0.0
    snr_db: float = 0.0
    amplitude_db: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    @classmethod
    def from_track(cls, stream_id: str, track: dict) -> "RadarObservation":
        fields = track.get("fields", {})
        try:
            r = float(fields["range_m"])
            az = float(fields["azimuth_deg"])
            el = float(fields["elevation_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "track position requires finite range_m, azimuth_deg, "
                "and elevation_deg"
            ) from exc
        if not all(math.isfinite(value) for value in (r, az, el)):
            raise ValueError(
                "track position requires finite range_m, azimuth_deg, "
                "and elevation_deg"
            )
        if r < 0.0:
            raise ValueError("track range_m must be non-negative")
        if not -180.0 <= az <= 180.0:
            raise ValueError("track azimuth_deg must be in [-180, 180]")
        if not -90.0 <= el <= 90.0:
            raise ValueError("track elevation_deg must be in [-90, 90]")
        xyz = spherical_to_xyz(r, az, el).reshape(3)
        radial_speed = float(fields.get("radial_speed_mps", 0.0))
        speed = float(fields.get("speed_mps", radial_speed))
        return cls(
            stream_id=stream_id,
            track_id=int(track.get("id", 0)),
            state=str(track.get("state", "")),
            radar_class=str(track.get("class_name", "")),
            range_m=r,
            azimuth_deg=az,
            elevation_deg=el,
            ground_range_m=float(r * math.cos(math.radians(el))),
            altitude_m=float(r * math.sin(math.radians(el))),
            radial_speed_mps=radial_speed,
            speed_mps=speed,
            course_deg=float(fields.get("course_deg", 0.0)),
            snr_db=float(fields.get("snr_db", 0.0)),
            amplitude_db=float(fields.get("amplitude_db", 0.0)),
            x_m=float(xyz[0]),
            y_m=float(xyz[1]),
            z_m=float(xyz[2]),
        )

    def feature_vector(self) -> list[float]:
        """Legacy 8-D vector."""
        return [
            self.range_m,
            self.azimuth_deg,
            self.elevation_deg,
            self.radial_speed_mps,
            self.speed_mps,
            self.course_deg,
            self.snr_db,
            self.amplitude_db,
        ]


@dataclass
class RadarFrame:
    stream_id: str
    stream_name: str
    frame_id: int
    timestamp: float
    # RTP timestamp in the sender's 90 kHz clock. ``timestamp`` remains the
    # seconds representation used by existing frontend consumers.
    rtp_timestamp: int | None = None
    send_interval_seconds: float = 1.0
    observations: list[RadarObservation] = field(default_factory=list)
    # Current complete RD map for this frame. Lists are used because they are
    # trivial to pickle across multiprocessing/spawn.
    rd_amplitude: list[list[float]] = field(default_factory=list)
    rd_phase: list[list[float]] = field(default_factory=list)
    remote_labels: dict[int, tuple[str, float]] = field(default_factory=dict)

    @property
    def scan_interval_seconds(self) -> float:
        """Compatibility alias; frontend animation has its own period."""
        return float(self.send_interval_seconds)


@dataclass
class StreamStatus:
    id: str
    name: str
    state: str
    detail: str = ""
    url: str = ""


@dataclass
class RadarInference:
    stream_id: str
    frame_id: int
    labels: dict[int, tuple[str, float]] = field(default_factory=dict)
    status: str = "ok"
