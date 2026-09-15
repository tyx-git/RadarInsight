"""r19 SNR channel and radial-speed retention smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import R19Dataset, build_r19_loader
from .protocol import R19_SMOKE_SCHEMA, R19Config, R19_RADIAL_VALUE_INDEX, R19_SNR_MISSING_INDEX, R19_SNR_VALUE_INDEX, artifact_path, load_manifest, seal, write_json_if_identical


def run_smoke(project_root: str | Path, config: R19Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(
        root, config, allow_code_drift=True
    )
    control = R19Dataset(root, config, split_name="validation", arm="snr_missing_only")
    ablation = R19Dataset(root, config, split_name="validation", arm="snr_channel_removed")
    checks = {
        "same_parent_population": control.parent_ids == ablation.parent_ids,
        "nonempty": len(control) > 0 and len(ablation) > 0,
        "snr_value_missing_in_control": all(bool(item.tr_features[..., R19_SNR_VALUE_INDEX].eq(0).all()) for item in control),
        "snr_missing_flag_one_in_control": all(bool(item.tr_missing_mask[..., R19_SNR_MISSING_INDEX].eq(1).all()) for item in control),
        "snr_channel_removed_in_ablation": all(bool(item.tr_missing_mask[..., R19_SNR_MISSING_INDEX].eq(0).all()) for item in ablation),
        "radial_speed_retained": all(torch.equal(a.tr_features[..., R19_RADIAL_VALUE_INDEX], b.tr_features[..., R19_RADIAL_VALUE_INDEX]) for a, b in zip(control, ablation, strict=True)),
        "radial_speed_missingness_retained": all(torch.equal(a.tr_missing_mask[..., 1], b.tr_missing_mask[..., 1]) for a, b in zip(control, ablation, strict=True)),
        "snr_ablation_changes_input": any(not torch.equal(a.tr_missing_mask, b.tr_missing_mask) for a, b in zip(control, ablation, strict=True)),
        "unknown_not_in_protocol": manifest["training_boundary"]["unknown_training_exposure"] == 0,
        "test_not_accessed": manifest["training_boundary"]["test_payload_access_count"] == 0,
    }
    payload = {"schema_version": R19_SMOKE_SCHEMA, "experiment_id": "V27d-r19", "status": "passed" if all(checks.values()) else "failed", "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "checks": checks, "training": "not_executed", "unknown_payload_access_count": 0, "test_payload_access_count": 0}
    path = artifact_path(root, config.artifacts["smoke_path"])
    write_json_if_identical(path, seal(payload))
    if not all(checks.values()):
        raise AssertionError(f"r19 smoke failed: {checks}")
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
