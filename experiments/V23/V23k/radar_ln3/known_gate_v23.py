"""Frozen contract for the V23 pure official-CfC KnownGate baseline.

V23 is deliberately a single full-training baseline, not an OOF experiment.
It replaces the project-local recurrent encoder in the frozen V21 seed43
baseline with dense ``ncps.torch.CfC`` and leaves five-fold OOF for a later,
separately approved confirmation phase.
"""

from __future__ import annotations

import hashlib
from pathlib import Path



FIXED_EPOCHS = 20
TRAINING_SEED = 43
SOURCE_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
)
SOURCE_ARTIFACT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal"
)
SOURCE_SUMMARY_RELATIVE = SOURCE_ARTIFACT_RELATIVE / "training-summary.json"
RUNTIME_COMPATIBILITY_REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V23-official-cfc-runtime-compatibility.json"
)

RUNTIME_COMPATIBILITY_REPORT_SCHEMA = "RADAR-LN3-V23-official-cfc-runtime-compatibility-v1"
RUNTIME_COMPATIBILITY_REQUIRED_CHECKS = (
    "exact_ncps_distribution_version",
    "exact_official_runtime_binding",
    "official_cfc_forward_has_no_padding_mask",
    "official_cfc_batch_two_heterogeneous_timespans_are_not_supported",
    "official_cfc_zero_dt_is_finite_but_not_state_identity",
    "official_ltc_zero_dt_is_not_safe_for_v23_direct_swap",
    "official_cfc_baseline_is_dense_not_wired",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


















__all__ = ["FIXED_EPOCHS", "RUNTIME_COMPATIBILITY_REPORT_RELATIVE", "RUNTIME_COMPATIBILITY_REQUIRED_CHECKS", "RUNTIME_COMPATIBILITY_REPORT_SCHEMA", "SOURCE_CONFIG_RELATIVE", "SOURCE_ARTIFACT_RELATIVE", "SOURCE_SUMMARY_RELATIVE", "TRAINING_SEED", "sha256_file"]