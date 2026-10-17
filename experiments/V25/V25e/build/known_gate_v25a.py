"""Prepare V25a's gradient-telemetry-only replication of the V24 baseline.

V25a deliberately changes no model, data, loss, optimiser or checkpoint
criterion.  It creates a new artifact namespace solely because its trainer
records pre-clip gradient and batch-composition observations on every update.
"""

from __future__ import annotations

from pathlib import Path




FEATURE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V25a-cq08-known-gate-course-preclip-gradient-"
    "telemetry-feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-"
    "gradient-telemetry-seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25a-known-gate-course-preclip-gradient-"
    "telemetry-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-"
    "telemetry-seed43-formal"
)






















