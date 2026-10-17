"""Prepare the sealed V25c module-gradient conflict localization diagnosis.

V25c is intentionally a read-only analysis of the completed V25a run.  It
does not create a new model, mutate a V25a/V24 checkpoint, construct a held-out
Dataset, or fit a threshold.  Preparation only freezes provenance and code
bindings that the later exact replay must satisfy.
"""

from __future__ import annotations

from pathlib import Path



REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-"
    "conflict-localization.json"
)























