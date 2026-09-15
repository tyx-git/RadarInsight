"""r21 contract smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from v27.v27a.checkpoint import tensor_state_hash

from .data import R21Dataset, build_r21_loader
from .model import R21L2Model, freeze_base_train_adapter
from .protocol import R21_SMOKE_SCHEMA, R21Config, artifact_path, load_manifest, seal, write_json_if_identical
from .training import _build_from_r20


def run_smoke(project_root: str | Path, config: R21Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config, allow_code_drift=True)
    control = R21Dataset(root, config, split_name="validation", arm="zero_control")
    treatment = R21Dataset(root, config, split_name="validation", arm="radial_decomposed")
    model, _ = _build_from_r20(root, config, device=torch.device("cpu"), arm="radial_decomposed")
    reference_base = model.base_state()
    freeze_base_train_adapter(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    checks: dict[str, bool] = {
        "same_parent_population": control.parent_ids == treatment.parent_ids,
        "nonempty": len(control) > 0 and len(treatment) > 0,
        "same_r20_sidecar_shapes": all(
            a.radial_values.shape == b.radial_values.shape
            for a, b in zip(control, treatment, strict=True)
        ),
        "control_sidecar_exact_zero": all(
            bool(item.radial_values.eq(0).all()) and not bool(item.radial_valid.any())
            for item in control
        ),
        "treatment_sidecar_has_signal": any(
            bool(item.radial_valid.any()) and bool(item.radial_values.abs().gt(0).any())
            for item in treatment
        ),
        "only_adapter_trainable": bool(trainable_names) and all(
            name.startswith("base_model.radial_adapter.") for name in trainable_names
        ),
        "base_model_frozen": bool(frozen_names) and all(
            not name.startswith("base_model.radial_adapter.") for name in frozen_names
        ),
        "unknown_not_in_protocol": manifest["training_boundary"]["unknown_training_exposure"] == 0,
        "test_not_accessed": manifest["training_boundary"]["test_payload_access_count"] == 0,
        "rd_not_accessed": manifest["training_boundary"]["rd_payload_access_count"] == 0,
    }
    control_batch = next(iter(build_r21_loader(control, batch_size=2, shuffle=False)))
    treatment_batch = next(iter(build_r21_loader(treatment, batch_size=2, shuffle=False)))
    checks["padded_values_zero"] = bool(
        treatment_batch["radial_sidecar_values"]
        .masked_select(treatment_batch["tr_padding_mask"].unsqueeze(-1))
        .eq(0)
        .all()
    )
    checks["padded_valid_false"] = not bool(
        treatment_batch["radial_sidecar_valid"]
        .masked_select(treatment_batch["tr_padding_mask"].unsqueeze(-1))
        .any()
    )
    payload = {
        "schema_version": R21_SMOKE_SCHEMA,
        "experiment_id": "V27d-r21",
        "status": "passed" if all(checks.values()) else "failed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "checks": checks,
        "trainable_parameter_names": trainable_names,
        "frozen_base_state_sha256": tensor_state_hash(reference_base),
        "training": "not_executed",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "deployment_eligible": False,
    }
    path = artifact_path(root, config.artifacts["smoke_path"])
    write_json_if_identical(path, seal(payload))
    if not all(checks.values()):
        raise AssertionError(f"r21 smoke failed: {checks}")
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
