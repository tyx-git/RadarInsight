"""r22 scale-flow, population and sidecar contract smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from radar_ln3.data import DT_REASON_ZERO_DELTA
from v27.V27d.r20.data import R20Dataset

from .data import R22Dataset, build_r22_loader
from .protocol import (
    R22_RATE_SCALE,
    R22_SMOKE_SCHEMA,
    R22Config,
    artifact_path,
    load_manifest,
    seal,
    write_json_if_identical,
)


def run_smoke(project_root: str | Path, config: R22Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    control = R22Dataset(root, config, split_name="validation", arm="zero_control")
    treatment = R22Dataset(root, config, split_name="validation", arm="radial_decomposed")
    parent_treatment = R20Dataset(root, config.r20, split_name="validation", arm="radial_decomposed")
    checks: dict[str, bool] = {
        "direct_parent_is_r20": manifest["direct_parent"]["experiment_id"] == "V27d-r20",
        "registered_rate_scale_is_10": float(config.data["radial_rate_scale_mps_per_s"]) == R22_RATE_SCALE,
        "r20_parent_rate_scale_is_100": float(config.r20.data["radial_rate_scale_mps_per_s"]) == 100.0,
        "same_parent_population": control.parent_ids == treatment.parent_ids == parent_treatment.parent_ids,
        "nonempty": len(control) > 0 and len(treatment) > 0,
        "raw_radial_removed_in_control": all(
            bool(item.sequence.tr_features[..., 1].eq(0).all()) and bool(item.sequence.tr_missing_mask[..., 1].eq(1).all())
            for item in control
        ),
        "raw_radial_removed_in_treatment": all(
            bool(item.sequence.tr_features[..., 1].eq(0).all()) and bool(item.sequence.tr_missing_mask[..., 1].eq(1).all())
            for item in treatment
        ),
        "control_sidecar_exact_zero": all(
            bool(item.radial_values.eq(0).all()) and not bool(item.radial_valid.any()) for item in control
        ),
        "treatment_sidecar_has_signal": any(
            bool(item.radial_valid.any()) and bool(item.radial_values.abs().gt(0).any()) for item in treatment
        ),
        "invalid_sidecar_values_zero": all(
            bool(item.radial_values.masked_select(~item.radial_valid).eq(0).all()) for item in treatment
        ),
        "dt_zero_rate_and_consistency_invalid": all(
            not bool(
                item.radial_valid[..., 2:4].masked_select(
                    item.sequence.tr_dt_reason.eq(DT_REASON_ZERO_DELTA).unsqueeze(-1)
                ).any()
            )
            for item in treatment
        ),
        "first_event_rate_and_consistency_invalid": all(
            not bool(item.radial_valid[0, 2:4].any())
            for item in treatment
            if item.sequence.event_count > 0
        ),
        "sidecar_values_within_clip": all(
            bool(item.radial_values.abs().le(float(config.data["sidecar_clip_absolute"]) + 1.0e-6).all())
            for item in treatment
        ),
        "unknown_not_in_protocol": manifest["training_boundary"]["unknown_training_exposure"] == 0,
        "test_not_accessed": manifest["training_boundary"]["test_payload_access_count"] == 0,
        "rd_not_accessed": manifest["training_boundary"]["rd_payload_access_count"] == 0,
    }
    same_non_rate_values = True
    same_validity = True
    rate_changed = False
    for current, parent in zip(treatment, parent_treatment, strict=True):
        same_non_rate_values = same_non_rate_values and bool(
            torch.equal(current.radial_values[..., (0, 1, 3)], parent.radial_values[..., (0, 1, 3)])
        )
        same_validity = same_validity and bool(torch.equal(current.radial_valid, parent.radial_valid))
        rate_mask = current.radial_valid[..., 2] & parent.radial_valid[..., 2]
        if bool(rate_mask.any()) and bool(
            (current.radial_values[..., 2].masked_select(rate_mask) != parent.radial_values[..., 2].masked_select(rate_mask)).any()
        ):
            rate_changed = True
    checks["non_rate_sidecar_values_unchanged"] = same_non_rate_values
    checks["sidecar_validity_unchanged"] = same_validity
    checks["rate_scale_changes_rate_values"] = rate_changed

    control_batch = next(iter(build_r22_loader(control, batch_size=2, shuffle=False)))
    treatment_batch = next(iter(build_r22_loader(treatment, batch_size=2, shuffle=False)))
    checks["padded_control_sidecar_zero"] = bool(
        control_batch["radial_sidecar_values"].masked_select(control_batch["tr_padding_mask"].unsqueeze(-1)).eq(0).all()
    )
    checks["padded_treatment_sidecar_zero"] = bool(
        treatment_batch["radial_sidecar_values"].masked_select(treatment_batch["tr_padding_mask"].unsqueeze(-1)).eq(0).all()
    )
    checks["padded_treatment_valid_false"] = not bool(
        treatment_batch["radial_sidecar_valid"].masked_select(treatment_batch["tr_padding_mask"].unsqueeze(-1)).any()
    )
    payload = {
        "schema_version": R22_SMOKE_SCHEMA,
        "experiment_id": "V27d-r22",
        "status": "passed" if all(checks.values()) else "failed",
        "direct_parent_experiment_id": "V27d-r20",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "checks": checks,
        "main_variable": "radial_rate_scale_mps_per_s: 100 -> 10",
        "training": "not_executed",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "deployment_eligible": False,
    }
    path = artifact_path(root, config.artifacts["smoke_path"])
    write_json_if_identical(path, seal(payload))
    if not all(checks.values()):
        raise AssertionError(f"r22 smoke failed: {checks}")
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
