"""Isolated RadarInsight RADAR-LN3 implementation."""

SCHEMA_VERSION = "RADAR-LN3-V0"
LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)

__all__ = ["LABELS", "SCHEMA_VERSION"]
