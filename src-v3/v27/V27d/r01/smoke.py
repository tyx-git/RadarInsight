"""r01 最小契约 smoke 测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .denoising import DenoisingConfig, score_x5a_record
from .protocol import R01_EXPERIMENT_ID, R01_SCHEMA_VERSION, load_config, r01_root, validate_r20_reference
from .x5a import X5ALayout, iter_x5a_records


def run_smoke(project_root: str | Path, *, sample_file: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(root)
    directory = root / "OriData" / "梁子湖X5A"
    path = Path(sample_file) if sample_file is not None else sorted(directory.glob("*.dat"))[0]
    if not path.is_absolute():
        path = root / path
    records = iter_x5a_records(path, layout=X5ALayout())
    first = next(records)
    points = first.point_values
    result = score_x5a_record(points, config=DenoisingConfig())
    checks = {
        "config_pointwise_input": config["data"]["pointwise_input"] is True,
        "x5a_layout_record_shape": points.ndim == 2 and points.shape[1] == 12,
        "x5a_layout_uint16_values": points.dtype == np.dtype("uint16"),
        "score_shape_matches_points": result.keep_weight.shape == (points.shape[0],),
        "score_finite": bool(np.isfinite(result.remove_score).all() and np.isfinite(result.keep_weight).all()),
        "keep_weight_bounded": bool((result.keep_weight >= 0.0).all() and (result.keep_weight <= 1.0).all()),
        "r20_reference_is_read_only": validate_r20_reference(root, config)["read_only_reference"] is True,
        "r01_artifacts_inside_r01": r01_root(root).is_dir(),
        "no_x5a_label_claim": config["source"]["x5a_label_policy"] == "no_point_labels; filename hints are not point truth",
    }
    payload = {
        "schema_version": "RADAR-LN3-V27d-r01-contract-smoke-v1",
        "experiment_id": R01_EXPERIMENT_ID,
        "r01_schema": R01_SCHEMA_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "sample_file": str(path),
        "sample_record_index": first.record_index,
        "sample_point_count": first.point_count,
        "checks": checks,
        "training": "not_required_for_replication",
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return payload


__all__ = ["run_smoke"]
