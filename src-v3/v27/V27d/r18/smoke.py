"""r18 structural and data-contract smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import R18Dataset, build_r18_loader
from .protocol import R18_SMOKE_SCHEMA, R18Config, artifact_path, load_manifest, seal, write_json_if_identical


def run_smoke(project_root: str | Path, config: R18Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    checks: dict[str, bool] = {}
    full = R18Dataset(root, config, split_name="validation", radial_speed_ablation=False)
    ablated = R18Dataset(root, config, split_name="validation", radial_speed_ablation=True)
    checks["same_parent_population"] = full.parent_ids == ablated.parent_ids
    checks["nonempty_split"] = len(full) > 0 and len(ablated) > 0
    checks["radial_speed_ablation_changes_input"] = any(not torch.equal(a.tr_features, b.tr_features) for a, b in zip(full, ablated, strict=True))
    checks["ablation_marks_radial_speed_missing"] = any(bool(item.tr_missing_mask[..., 1].gt(0.5).any()) for item in ablated)
    batch = next(iter(build_r18_loader(ablated, batch_size=2, shuffle=False)))
    checks["padded_values_zero"] = bool(batch["tr_features"].masked_select(batch["tr_padding_mask"].unsqueeze(-1)).eq(0).all())
    checks["unknown_not_in_protocol"] = manifest["training_boundary"]["unknown_training_exposure"] == 0
    checks["test_not_accessed"] = manifest["training_boundary"]["test_payload_access_count"] == 0
    payload = {"schema_version": R18_SMOKE_SCHEMA, "experiment_id": "V27d-r18", "status": "passed" if all(checks.values()) else "failed", "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "checks": checks, "train": "not_executed", "unknown_payload_access_count": 0, "test_payload_access_count": 0, "deployment_eligible": False}
    path = artifact_path(root, config.artifacts["smoke_path"])
    write_json_if_identical(path, seal(payload))
    if not all(checks.values()):
        raise AssertionError(f"r18 smoke failed: {checks}")
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
