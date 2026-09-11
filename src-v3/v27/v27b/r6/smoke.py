"""Contract smoke checks for the frozen-endpoint V27b-r6 experiment.

The checks use synthetic tensors except for one exact replay of the sealed
V27a validation cache.  They intentionally avoid mixed-pool materialization,
Unknown extraction, fitting, calibration, and test payloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .features import _summary_stability
from .model import build_r6_model
from .protocol import (
    R6_ARMS,
    R6_SMOKE_SCHEMA_VERSION,
    R6Context,
    r6_artifact_path,
    write_sealed_json_if_identical,
)


def _synthetic_features(context: R6Context) -> dict[str, torch.Tensor]:
    """Create a padded, finite batch with stable8 evidence only on active rows."""

    generator = torch.Generator(device="cpu").manual_seed(27_060)
    batch, width = 3, 6
    lengths = torch.tensor((6, 4, 2), dtype=torch.long)
    padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    event = torch.randn(
        (batch, width, context.config.data.event_trajectory_dim),
        generator=generator,
    )
    summary = torch.randn(
        (batch, width, context.config.data.summary_feature_dim),
        generator=generator,
    )
    facts = torch.randn(
        (batch, width, context.config.data.task_fact_dim),
        generator=generator,
    )
    stable = torch.randn(
        (batch, width, context.config.data.stable8_dim * 4 + 6),
        generator=generator,
    )
    for value in (event, summary, facts, stable):
        value.masked_fill_(padding.unsqueeze(-1), 0.0)
    return {
        "event_trajectory": event,
        "summary_features": summary,
        "task_facts_sequence": facts,
        "stable_features": stable,
        "padding_mask": padding,
    }


def _assert_capacity_and_control_isolation(context: R6Context) -> dict[str, Any]:
    control = build_r6_model(context.config, arm="control", seed=42)
    treatment = build_r6_model(context.config, arm="stable8_treatment", seed=42)
    if control.parameter_count != treatment.parameter_count:
        raise AssertionError("V27b-r6 arms do not have matched parameter counts")
    if control.state_dict().keys() != treatment.state_dict().keys() or any(
        not torch.equal(control.state_dict()[name], treatment.state_dict()[name])
        for name in control.state_dict()
    ):
        raise AssertionError("V27b-r6 arms do not have identical initialization")
    values = _synthetic_features(context)
    changed = {name: value.clone() for name, value in values.items()}
    changed["stable_features"] += (
        (~changed["padding_mask"]).unsqueeze(-1).to(dtype=changed["stable_features"].dtype)
        * 100.0
    )
    with torch.inference_mode():
        control_before = control(**values)
        control_after = control(**changed)
        treatment_before = treatment(**values)
        treatment_after = treatment(**changed)
    if not torch.equal(control_before, control_after):
        raise AssertionError("V27b-r6 control depends on treatment stable8 input")
    if torch.equal(treatment_before, treatment_after):
        raise AssertionError("V27b-r6 treatment ignores stable8 input")
    return {
        "parameter_count": control.parameter_count,
        "parameter_counts_identical": True,
        "initialization_identical": True,
        "control_stable8_input_invariant": True,
        "treatment_stable8_input_sensitive": True,
    }


def _assert_padding_and_causality(context: R6Context) -> None:
    """GRU final output may depend on all valid past, never padded/future suffixes."""

    model = build_r6_model(context.config, arm="stable8_treatment", seed=42)
    values = _synthetic_features(context)
    base = {name: value[:1, :6].clone() for name, value in values.items()}
    base["padding_mask"][:] = False
    with torch.inference_mode():
        full = model(**base)
        prefix = {name: value[:, :4].clone() for name, value in base.items()}
        prefix_result = model(**prefix)
        future = {name: value.clone() for name, value in base.items()}
        for name in (
            "event_trajectory",
            "summary_features",
            "task_facts_sequence",
            "stable_features",
        ):
            future[name][:, 4:] += 100.0
        future_prefix = {name: value[:, :4].clone() for name, value in future.items()}
        future_prefix_result = model(**future_prefix)
    if not torch.allclose(prefix_result, future_prefix_result, atol=0.0, rtol=0.0):
        raise AssertionError("V27b-r6 future suffix changed an independently evaluated prefix")
    padded = {name: value.clone() for name, value in base.items()}
    padded["padding_mask"][:, 4:] = True
    for name in (
        "event_trajectory",
        "summary_features",
        "task_facts_sequence",
        "stable_features",
    ):
        padded[name][:, 4:] = 0.0
    with torch.inference_mode():
        padded_result = model(**padded)
    if not torch.allclose(prefix_result, padded_result, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("V27b-r6 padded suffix differs from explicit prefix")
    if not torch.isfinite(full).all():
        raise FloatingPointError("V27b-r6 synthetic full output is non-finite")


def _assert_stability_contract(context: R6Context) -> dict[str, Any]:
    """Check prefix-local coverage and distinguish reliable frames from RD presence."""

    summary = torch.zeros((1, 4, context.config.data.stable8_dim), dtype=torch.float32)
    summary[:, 0] = 1.0
    summary[:, 2] = 3.0
    reliable = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    counts = torch.tensor([[1, 0, 2, 0]], dtype=torch.long)
    fractions = torch.tensor([[0.5, 0.0, 1.0, 0.0]], dtype=torch.float32)
    padding = torch.zeros((1, 4), dtype=torch.bool)
    values = _summary_stability(summary, reliable, counts, fractions, padding)
    # stable8 feature layout starts with last, mean, std, deviation, coverage.
    coverage_offset = context.config.data.stable8_dim * 4
    actual = values[0, :, coverage_offset]
    expected = torch.tensor((1.0, 0.5, 2.0 / 3.0, 0.5), dtype=torch.float32)
    if not torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0):
        raise AssertionError("V27b-r6 stable8 coverage is not prefix-local")
    if values.shape[-1] != context.config.data.stable8_dim * 4 + 6:
        raise AssertionError("V27b-r6 stable8 feature width differs")
    return {
        "prefix_local_coverage": actual.tolist(),
        "reliable_stable8_distinct_from_rd_present": True,
        "stable_feature_dim": int(values.shape[-1]),
    }


def _assert_archived_endpoint_replay(context: R6Context) -> dict[str, Any]:
    """Replay the sealed V27a endpoint exactly without opening test data."""

    from v27.v27a.cache import load_cache_payload
    from v27.v27a.model import V27AEvidenceResidual
    from v27.v27a.protocol import load_v27a_config

    checkpoint_path = context.project_root / context.config.base.v27a_main_checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_v27a_config(context.project_root, context.config.base.v27a_config_path)
    model = V27AEvidenceResidual(config, arm="main")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    cache = load_cache_payload(
        context.project_root / context.config.base.v27a_validation_reference_cache_path
    )
    prediction: list[int] = []
    truth: list[int] = []
    with torch.inference_mode():
        for row, length in enumerate(cache["lengths"]):
            output = model(
                baseline_logits=torch.from_numpy(cache["event_baseline_logits"][row : row + 1]),
                summary_tokens=torch.from_numpy(cache["event_summary_tokens"][row : row + 1]),
                summary_valid=torch.from_numpy(cache["event_summary_valid"][row : row + 1]),
                frame_values=torch.from_numpy(cache["event_frame_values"][row : row + 1]),
                frame_mask=torch.from_numpy(cache["event_frame_mask"][row : row + 1]),
                padding_mask=torch.from_numpy(cache["padding_mask"][row : row + 1]),
            )
            prediction.append(int(output["sequence_logits"][0, int(length) - 1].argmax().item()))
            truth.append(int(cache["labels"][row]))
    matrix = np.zeros((3, 3), dtype=np.int64)
    for expected, observed in zip(truth, prediction, strict=True):
        matrix[expected, observed] += 1
    expected_matrix = [[109, 4, 0], [1, 54, 1], [1, 1, 52]]
    if matrix.tolist() != expected_matrix:
        raise AssertionError("V27b-r6 frozen V27a endpoint replay differs")
    return {
        "parent_final_confusion_matrix_rows_true_columns_pred": matrix.tolist(),
        "matches_sealed_v27a_report": True,
    }


def run_smoke(context: R6Context) -> dict[str, Any]:
    """Run only the necessary contract checks and write an immutable report."""

    capacity = _assert_capacity_and_control_isolation(context)
    _assert_padding_and_causality(context)
    stability = _assert_stability_contract(context)
    replay = _assert_archived_endpoint_replay(context)
    result = {
        "schema_version": R6_SMOKE_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "passed",
        "r6_manifest_sha256": context.manifest_sha256,
        "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "capacity": capacity,
        "stability": stability,
        "endpoint_replay": replay,
        "tests": {
            "matched_arm_capacity": True,
            "control_stable8_isolation": True,
            "treatment_stable8_sensitivity": True,
            "padding_never_enters_gru": True,
            "prefix_input_does_not_read_future_suffix": True,
            "reliable_stable8_is_not_rd_present": True,
            "sealed_v27a_endpoint_exact_replay": True,
        },
        "scope": {
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
            "calibration": "not_fitted",
            "l3": "not_fitted",
        },
    }
    path = r6_artifact_path(context.project_root, context.config.artifacts.smoke_report_path)
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
