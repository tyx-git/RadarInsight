"""Necessary structural and frozen-endpoint checks for V27c."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .features import load_feature_cache, replay_frozen_main_logits
from .model import build_v27c_model
from .protocol import (
    V27C_FROZEN_MAIN_CONFUSION,
    V27C_SMOKE_SCHEMA_VERSION,
    V27CContext,
    v27c_artifact_path,
    write_sealed_json_if_identical,
)


def _synthetic(context: V27CContext) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(27_300)
    lengths = torch.tensor((5, 3), dtype=torch.long)
    width = 5
    padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    logits = torch.randn((2, width, 3), generator=generator)
    motion = torch.randn((2, width, context.config.data.motion_dim), generator=generator).clamp(-1.0, 1.0)
    logits.masked_fill_(padding.unsqueeze(-1), 0.0)
    motion.masked_fill_(padding.unsqueeze(-1), 0.0)
    return {"frozen_endpoint_logits": logits, "motion_values": motion, "padding_mask": padding}


def _assert_model_contract(context: V27CContext) -> dict[str, Any]:
    model = build_v27c_model(context.config, seed=42)
    values = _synthetic(context)
    with torch.inference_mode():
        output = model(**values)
    if not torch.equal(output["baseline_logits"], values["frozen_endpoint_logits"]):
        raise AssertionError("V27c model altered frozen input logits")
    if not torch.allclose(output["residual_logits"].sum(dim=-1), torch.zeros_like(output["residual_logits"][..., 0]), atol=1.0e-6, rtol=0.0):
        raise AssertionError("V27c residual is not zero-sum")
    if bool((output["residual_logits"].abs() > context.config.model.residual_bound + 1.0e-6).any()):
        raise AssertionError("V27c residual exceeds its bound")
    zeroed = dict(values)
    zeroed["motion_values"] = torch.zeros_like(values["motion_values"])
    with torch.inference_mode():
        zero_result = model(**zeroed)
    if not torch.equal(zero_result["sequence_logits"], zeroed["frozen_endpoint_logits"]):
        raise AssertionError("V27c zero motion does not exactly replay frozen logits")
    with torch.no_grad():
        model.residual_projection.weight[0, 0] = 0.25
        model.residual_projection.weight[1, 0] = -0.25
    with torch.inference_mode():
        zero_after_nonzero_weights = model(**zeroed)["sequence_logits"]
    if not torch.equal(zero_after_nonzero_weights, zeroed["frozen_endpoint_logits"]):
        raise AssertionError("V27c zero motion acquired a correction after output-head training")
    empty = dict(values)
    empty["frozen_endpoint_logits"] = torch.zeros_like(values["frozen_endpoint_logits"][:, :1])
    empty["motion_values"] = torch.zeros_like(values["motion_values"][:, :1])
    empty["padding_mask"] = torch.ones_like(values["padding_mask"][:, :1])
    try:
        model(**empty)
    except ValueError as exc:
        if str(exc) != "V27c received an empty parent":
            raise
    else:
        raise AssertionError("V27c accepted a new empty parent without an initial state")
    changed = dict(values)
    changed["motion_values"] = values["motion_values"].clone()
    changed["motion_values"][0, 0, 0] += 0.5
    with torch.inference_mode():
        baseline_response = model(**values)["sequence_logits"]
        changed_response = model(**changed)["sequence_logits"]
    if torch.equal(baseline_response, changed_response):
        raise AssertionError("V27c treatment ignores a non-zero motion token")
    altered = dict(values)
    altered["motion_values"] = values["motion_values"].clone()
    altered["motion_values"][0, 4, 0] += 0.5
    with torch.inference_mode():
        before = model(**values)
        after = model(**altered)
    if not torch.equal(before["sequence_logits"][:, :4], after["sequence_logits"][:, :4]):
        raise AssertionError("V27c future suffix changed a previously emitted prefix")
    with torch.inference_mode():
        full = before
        first = model(**{key: value[:, :4] for key, value in values.items()})
        suffix = {
            key: value[:, 4:]
            for key, value in values.items()
        }
        second = model(**suffix, initial_state=first["final_state"])
    if (
        not torch.equal(full["sequence_logits"][:, 4:], second["sequence_logits"])
        or not torch.equal(full["final_state"], second["final_state"])
    ):
        raise AssertionError("V27c chunked causal continuation differs from whole-sequence inference")
    return {
        "parameter_count": model.parameter_count,
        "frozen_logits_passthrough": True,
        "zero_sum_residual": True,
        "bounded_residual": True,
        "zero_motion_exact_control": True,
        "zero_motion_exact_control_after_nonzero_output_weights": True,
        "new_empty_parent_rejected": True,
        "nonzero_motion_token_changes_treatment": True,
        "prefix_does_not_read_future_suffix": True,
        "chunked_continuation_matches_whole_sequence": True,
    }


def _assert_component_support_permutation() -> dict[str, Any]:
    from .evaluation import _component_support_preserving_event_value_permutation

    values = np.zeros((4, 3, 2), dtype=np.float32)
    values[:, 0, 0] = np.arange(1.0, 5.0)
    values[:, 1, :] = np.arange(5.0, 13.0, dtype=np.float32).reshape(4, 2)
    lengths = np.full((4,), 2, dtype=np.int64)
    permuted, moved_events, moved_parents = (
        _component_support_preserving_event_value_permutation(
            values, lengths, seed=27_811
        )
    )
    if (
        moved_events != 8
        or moved_parents != 4
        or not np.array_equal(values == 0.0, permuted == 0.0)
        or np.array_equal(values, permuted)
        or bool((permuted[:, 2] != 0.0).any())
    ):
        raise AssertionError("V27c component-support value permutation differs")
    return {
        "component_support_preserving_value_permutation": True,
        "synthetic_moved_event_count": moved_events,
    }


def _assert_materialized_alignment(context: V27CContext) -> dict[str, Any]:
    train = load_feature_cache(context, split="train")
    val = load_feature_cache(context, split="val")
    if len(train["parent_ids"]) != 953 or len(val["parent_ids"]) != 223:
        raise AssertionError("V27c frozen parent support differs")
    motion = val["tensors"]["motion_values"]
    padding = val["tensors"]["padding_mask"]
    if bool(motion.masked_select(padding.unsqueeze(-1).expand_as(motion)).ne(0).any()):
        raise AssertionError("V27c materialized padded motion is non-zero")
    final = val["tensors"]["lengths"] - 1
    rows = torch.arange(len(final))
    for split, cache in (("train", train), ("val", val)):
        replayed = replay_frozen_main_logits(context, split=split)
        if not torch.equal(cache["tensors"]["frozen_endpoint_logits"], replayed):
            raise AssertionError(
                f"V27c {split} sidecar frozen endpoint differs from exact V27a Main replay"
            )
    return {
        "train_parent_count": len(train["parent_ids"]),
        "validation_parent_count": len(val["parent_ids"]),
        "validation_active_motion_nonzero_fraction": float(
            motion.abs().amax(dim=-1).masked_select(~padding).gt(0.0).float().mean().item()
        ),
        "validation_final_motion_nonzero_fraction": float(
            motion[rows, final].abs().amax(dim=-1).gt(0.0).float().mean().item()
        ),
        "motion_mask_not_serialized": "tr_motion_valid_mask" not in val["tensors"],
        "padding_zero": True,
        "frozen_endpoint_logits_exact_replay": True,
    }


def _assert_endpoint_replay(context: V27CContext) -> dict[str, Any]:
    from v27.v27a.model import V27AEvidenceResidual
    from v27.v27a.protocol import load_v27a_config
    from v27.v27a.cache import load_cache_payload

    config = load_v27a_config(context.project_root, context.config.base.v27a_runtime_config_path)
    checkpoint = torch.load(context.project_root / context.config.base.v27a_main_checkpoint_path, map_location="cpu", weights_only=False)
    model = V27AEvidenceResidual(config, arm="main")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    cache = load_cache_payload(context.project_root / context.config.base.v27a_validation_reference_cache_path)
    matrix = np.zeros((3, 3), dtype=np.int64)
    with torch.inference_mode():
        for row, length in enumerate(cache["lengths"]):
            output = model(
                baseline_logits=torch.from_numpy(cache["event_baseline_logits"][row:row + 1]),
                summary_tokens=torch.from_numpy(cache["event_summary_tokens"][row:row + 1]),
                summary_valid=torch.from_numpy(cache["event_summary_valid"][row:row + 1]),
                frame_values=torch.from_numpy(cache["event_frame_values"][row:row + 1]),
                frame_mask=torch.from_numpy(cache["event_frame_mask"][row:row + 1]),
                padding_mask=torch.from_numpy(cache["padding_mask"][row:row + 1]),
            )
            truth = int(cache["labels"][row])
            prediction = int(output["sequence_logits"][0, int(length) - 1].argmax().item())
            matrix[truth, prediction] += 1
    if matrix.tolist() != [list(row) for row in V27C_FROZEN_MAIN_CONFUSION]:
        raise AssertionError("V27c frozen V27a endpoint replay differs")
    return {"parent_final_confusion_matrix_rows_true_columns_pred": matrix.tolist(), "matches_archived_v27a_main": True}


def run_smoke(context: V27CContext) -> dict[str, Any]:
    result = {
        "schema_version": V27C_SMOKE_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "passed",
        "manifest_sha256": context.manifest_sha256,
        "model": _assert_model_contract(context),
        "counterfactual": _assert_component_support_permutation(),
        "sidecar": _assert_materialized_alignment(context),
        "endpoint_replay": _assert_endpoint_replay(context),
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
    }
    path = v27c_artifact_path(context.project_root, context.config.artifacts.smoke_report_path)
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
