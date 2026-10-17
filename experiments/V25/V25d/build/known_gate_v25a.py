"""Prepare V25a's gradient-telemetry-only replication of the V24 baseline.

V25a deliberately changes no model, data, loss, optimiser or checkpoint
criterion.  It creates a new artifact namespace solely because its trainer
records pre-clip gradient and batch-composition observations on every update.
"""

from __future__ import annotations

from pathlib import Path


from build.known_gate_course_reason_aware import CORE_CONTRACT, CORE_NORMALIZATION, EXPECTED_CACHE_EVENT_COUNTS, SOURCE_INDEX, TRAIN_PARENT_COUNT, V21E_TRAIN_CACHE, V21E_VALIDATION_CACHE, VALIDATION_CLUTTER_PARENT_SUPPORT, VALIDATION_PARENT_COUNT


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






















