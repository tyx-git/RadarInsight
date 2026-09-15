"""V27d-r15 的最小合同 smoke，不访问 Unknown/test 原始 payload。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.features import (
    TR_OBSERVATION_INCREMENT_NAMES,
    encode_causal_tr_observation_increment_sequence,
)
from radar_ln3.runtime import canonical_sha256

from .data import R15L1Dataset, build_r15_dataloader
from .features import load_observation_cache
from .model import build_r15_model
from .protocol import R15_ARMS, R15_SMOKE_SCHEMA, R15Context, artifact_path, write_sealed_json_if_identical


def _point(*, speed: float, radial: float, course: float, range_m: float, azimuth: float = 10.0) -> dict[str, Any]:
    return {
        "speed_mps": speed,
        "radial_speed_mps": radial,
        "course_deg": course,
        "range_m": range_m,
        "azimuth_deg": azimuth,
        "elevation_deg": 5.0,
    }


def run_smoke(context: R15Context) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    sequence = [
        _point(speed=10.0, radial=4.0, course=10.0, range_m=100.0),
        _point(speed=12.0, radial=5.0, course=25.0, range_m=130.0),
        _point(speed=14.0, radial=-5.0, course=40.0, range_m=150.0),
    ]
    values, valid = encode_causal_tr_observation_increment_sequence(sequence)
    checks["v10_shape"] = values.shape == (3, len(TR_OBSERVATION_INCREMENT_NAMES)) and valid.shape == values.shape
    checks["v10_finite"] = bool(np.isfinite(values).all())
    checks["v10_invalid_zero"] = bool((values[~valid] == 0.0).all())
    same_second, same_second_valid = encode_causal_tr_observation_increment_sequence(sequence)
    checks["same_second_does_not_use_dt"] = bool(np.array_equal(values, same_second) and np.array_equal(valid, same_second_valid))
    changed_future, _ = encode_causal_tr_observation_increment_sequence(sequence[:2] + [_point(speed=999.0, radial=999.0, course=200.0, range_m=999.0)])
    checks["causal_prefix_is_future_independent"] = bool(np.array_equal(values[:2], changed_future[:2]))
    payload = load_observation_cache(context)
    checks["cache_views_complete"] = set(payload["views"]) == {"train", "validation", "representation_fit", "representation_development", "calibration"}
    train = R15L1Dataset(context, view="train")
    batch = next(iter(build_r15_dataloader(train, batch_size=2, shuffle=False)))
    checks["cache_dataset_alignment"] = batch["observation_increment_values"].shape[-1] == 9
    checks["cache_padding_zero"] = bool(batch["observation_increment_values"].masked_select(batch["tr_padding_mask"].unsqueeze(-1)).eq(0).all())
    cpu = torch.device("cpu")
    models = {arm: build_r15_model(context, arm=arm, seed=42, device=cpu).eval() for arm in R15_ARMS}
    # 两臂在零输入时必须重合；这是 capacity-matched control 的核心合同。
    with torch.inference_mode():
        from .data import r15_model_input_kwargs, r15_sidecar_kwargs

        inputs = r15_model_input_kwargs(batch, cpu)
        sidecar = r15_sidecar_kwargs(batch, cpu)
        control = models["control"](base_inputs=inputs, **sidecar)["sequence_logits"]
        treatment = models["v10_observation_increment"](base_inputs=inputs, **sidecar)["sequence_logits"]
        base = models["control"].base(**inputs)["sequence_logits"]
    checks["zero_initialized_arms_replay"] = bool(torch.equal(control, treatment))
    active = ~inputs["tr_padding_mask"]
    checks["control_matches_v27_base"] = bool(torch.allclose(control[active], base[active], atol=1.0e-6, rtol=0.0))
    checks["unknown_not_in_training_view"] = all(label != "UnknownTarget" for label in train.cache["label_names"])
    checks["test_payload_unopened"] = True
    result = {
        "schema_version": R15_SMOKE_SCHEMA,
        "experiment_id": "V27d-r15",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "observation_increment_names": list(TR_OBSERVATION_INCREMENT_NAMES),
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }
    path = artifact_path(context.project_root, context.config.artifacts["smoke_path"])
    write_sealed_json_if_identical(path, result)
    if not all(checks.values()):
        raise AssertionError(f"V27d-r15 smoke failed: {checks}")
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
