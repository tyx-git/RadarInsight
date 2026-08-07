from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from radar_ln3.data import CQ08PointSequenceDataset
from radar_ln3.features import (
    CausalTRObservationIncrementState,
    TR_OBSERVATION_INCREMENT_NAMES,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    encode_causal_tr_observation_increment_sequence,
    parse_feature_json,
    resolve_causal_tr_feature_contract,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    verify_task_data_contract,
)


V9_CONFIG = "config/RADAR-LN3/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json"
V10_CONFIG = "config/RADAR-LN3/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json"
V9_CHECKPOINT = "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/best_model.pt"


def _point(
    *,
    range_m: float | None = 1000.0,
    radial: float | None = 10.0,
    azimuth: float | None = 0.0,
    elevation: float | None = 0.0,
    course: float | None = 0.0,
    speed: float | None = 10.0,
) -> dict[str, Any]:
    return {
        "range_m": range_m,
        "radial_speed_mps": radial,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "course_deg": course,
        "speed_mps": speed,
        "amplitude_db": 1.0,
        "snr_db": 10.0,
        "height_m": None,
        "latitude": None,
        "longitude": None,
    }


def _assert_validity(values: np.ndarray, valid: np.ndarray) -> None:
    expected = len(TR_OBSERVATION_INCREMENT_NAMES)
    if values.shape != valid.shape or values.shape[-1] != expected:
        raise AssertionError("V10 observation-increment value/mask shape mismatch")
    if not np.isfinite(values).all():
        raise AssertionError("V10 observation-increment values are non-finite")
    if np.any(values[~valid] != 0.0):
        raise AssertionError("invalid V10 observation-increment values are not exactly zero")
    contract = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
    if float(np.abs(values).max(initial=0.0)) > contract.clip_absolute + 1.0e-6:
        raise AssertionError("V10 observation increment escaped its frozen clip")


def _synthetic_contracts() -> dict[str, Any]:
    signature = inspect.signature(CausalTRObservationIncrementState.update)
    if tuple(signature.parameters) != ("self", "values"):
        raise AssertionError(f"V10 state unexpectedly accepts time information: {signature}")

    single, single_valid = encode_causal_tr_observation_increment_sequence([_point()])
    _assert_validity(single, single_valid)
    if single_valid.any():
        raise AssertionError("V10 first event unexpectedly produced a valid increment")

    sequence = [
        _point(course=120.0),
        _point(range_m=1010.0, radial=11.0, course=359.0, speed=12.0),
        _point(range_m=1020.0, radial=-12.0, course=0.0, speed=13.0),
    ]
    values, valid = encode_causal_tr_observation_increment_sequence(sequence)
    _assert_validity(values, valid)
    course_index = TR_OBSERVATION_INCREMENT_NAMES.index("course_delta_scaled")
    if abs(float(values[2, course_index]) - 1.0 / 45.0) > 1.0e-6:
        raise AssertionError("V10 359-to-0 course transition did not use the shortest circular delta")
    if valid[1, TR_OBSERVATION_INCREMENT_NAMES.index("reported_speed_delta_scaled")]:
        raise AssertionError("unconfirmed first-event speed seeded a V10 second-event delta")
    if valid[1, course_index]:
        raise AssertionError("unconfirmed first-event course seeded a V10 second-event delta")
    if not valid[2, TR_OBSERVATION_INCREMENT_NAMES.index("radial_sign_flip")]:
        raise AssertionError("V10 radial sign flip was not valid on a complete pair")

    missing, missing_valid = encode_causal_tr_observation_increment_sequence(
        [_point(), _point(range_m=None, radial=None, azimuth=None, elevation=None, course=None, speed=None)]
    )
    _assert_validity(missing, missing_valid)
    if missing_valid[1].any():
        raise AssertionError("missing V10 current fields generated a valid increment")

    extreme, extreme_valid = encode_causal_tr_observation_increment_sequence(
        [_point(), _point(range_m=100000.0, radial=-10000.0, course=180.0, speed=10000.0)]
    )
    _assert_validity(extreme, extreme_valid)

    base = [_point(), _point(range_m=1010.0, course=5.0), _point(range_m=1020.0, course=10.0)]
    changed_future = [dict(value) for value in base]
    changed_future[2] = _point(range_m=4000.0, radial=-80.0, course=170.0, speed=90.0)
    first, first_valid = encode_causal_tr_observation_increment_sequence(base)
    second, second_valid = encode_causal_tr_observation_increment_sequence(changed_future)
    if not np.array_equal(first[:2], second[:2]) or not np.array_equal(first_valid[:2], second_valid[:2]):
        raise AssertionError("future-point mutation changed an earlier V10 causal prefix")

    state = CausalTRObservationIncrementState()
    state.update(_point())
    state.update(_point(range_m=1010.0))
    state.reset()
    reset_values, reset_valid = state.update(_point(range_m=5000.0))
    if reset_valid.any() or np.any(reset_values != 0.0):
        raise AssertionError("V10 increment state leaked across a reset boundary")

    return {
        "state_signature_has_no_time_input": True,
        "single_point": "passed",
        "circular_359_to_0": "passed",
        "first_tracker_kinematics_not_amplified": True,
        "missing_fields": "passed",
        "extreme_increment_clipped": "passed",
        "future_prefix_invariance": "passed",
        "state_reset": "passed",
    }


def _changed_keys(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


def _real_dt_invariance(
    project_root: Path,
    config: Any,
    dataset: Any,
    smoke_dir: Path,
) -> dict[str, Any]:
    parent_id = dataset.parent_ids[0]
    group = dataset.base._groups[parent_id].copy()
    original = group.copy()
    perturbed = group.copy()
    row_count = len(perturbed)
    perturbed["tr_dt_seconds"] = np.asarray(
        [(-123.0 if index % 3 == 0 else 0.0 if index % 3 == 1 else 999.0) for index in range(row_count)],
        dtype=np.float64,
    )
    perturbed["tr_dt_valid"] = np.asarray([index % 2 == 0 for index in range(row_count)], dtype=bool)
    if "tr_dt_reason" in perturbed.columns:
        perturbed["tr_dt_reason"] = [f"v10_forbidden_time_probe_{index}" for index in range(row_count)]
    if "tr_time_seconds" in perturbed.columns:
        perturbed["tr_time_seconds"] = np.arange(row_count, dtype=np.float64)[::-1] * 777.0

    with tempfile.TemporaryDirectory(prefix="v10-dt-invariance-", dir=smoke_dir) as temporary:
        temporary_root = Path(temporary)
        original_path = temporary_root / "original.csv"
        perturbed_path = temporary_root / "perturbed.csv"
        original.to_csv(original_path, index=False)
        perturbed.to_csv(perturbed_path, index=False)

        kwargs = {
            "split_name": config.data.train_split_name,
            "max_sequence_length": config.data.max_sequence_length,
            "max_rd_frames_per_point": config.data.max_rd_frames_per_point,
            "rd_amplitude_center": config.data.rd_amplitude_center,
            "rd_amplitude_scale": config.data.rd_amplitude_scale,
            "exclude_anomalous_parents": False,
            "expected_schema_version": config.data.index_schema_version,
            "tr_motion_schema": config.data.tr_motion_schema,
        }
        left = CQ08PointSequenceDataset(project_root, index_path=original_path, **kwargs)[0]
        right = CQ08PointSequenceDataset(project_root, index_path=perturbed_path, **kwargs)[0]
    if not torch.equal(left.tr_motion_features, right.tr_motion_features):
        raise AssertionError("V10 real Dataset increments changed after dt/timestamp perturbation")
    if not torch.equal(left.tr_motion_valid_mask, right.tr_motion_valid_mask):
        raise AssertionError("V10 real Dataset validity changed after dt/timestamp perturbation")
    if torch.equal(left.tr_dt_seconds, right.tr_dt_seconds) or torch.equal(left.tr_dt_valid, right.tr_dt_valid):
        raise AssertionError("V10 dt invariance test did not actually perturb its forbidden time inputs")
    return {
        "parent_sample_id": parent_id,
        "event_count": int(left.event_count),
        "motion_values_exactly_equal": True,
        "motion_valid_mask_exactly_equal": True,
        "dt_values_changed": True,
        "dt_valid_values_changed": True,
        "timestamp_and_reason_changed_when_present": True,
    }


def _model_contracts(project_root: Path) -> dict[str, Any]:
    np.random.seed(20260807)
    torch.manual_seed(20260807)
    v9_path = project_root / V9_CONFIG
    v10_path = project_root / V10_CONFIG
    v9_payload = json.loads(v9_path.read_text(encoding="utf-8"))
    v10_payload = json.loads(v10_path.read_text(encoding="utf-8"))
    if v9_payload["task"] != v10_payload["task"]:
        raise AssertionError("V10 changed the DBB/CU task or label mapping")
    if _changed_keys(v9_payload["model"], v10_payload["model"]):
        raise AssertionError("V10 changed model dimensions or liquid/RD architecture")
    expected_data_changes = {
        "contract_path",
        "contract_schema_version",
        "expected_contract_sha256",
        "tr_motion_schema",
    }
    expected_train_changes = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "checkpoint_schema",
    }
    data_changes = _changed_keys(v9_payload["data"], v10_payload["data"])
    train_changes = _changed_keys(v9_payload["train"], v10_payload["train"])
    if data_changes != expected_data_changes:
        raise AssertionError(f"V10 changed unexpected data config fields: {sorted(data_changes)}")
    if train_changes != expected_train_changes:
        raise AssertionError(f"V10 changed unexpected train config fields: {sorted(train_changes)}")

    v9 = load_task_config(v9_path)
    v10 = load_task_config(v10_path)
    contract = verify_task_data_contract(project_root, v10)
    dataset = build_task_dataset(
        project_root,
        v10,
        split_name=v10.data.train_split_name,
        parent_limit=4,
    )
    loader = build_task_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    values = batch["tr_motion_features"]
    valid = batch["tr_motion_valid_mask"]
    padding = batch["tr_padding_mask"]
    expected_shape = (*batch["tr_features"].shape[:2], len(TR_OBSERVATION_INCREMENT_NAMES))
    if values.shape != expected_shape or valid.shape != expected_shape or valid.dtype != torch.bool:
        raise AssertionError("real V10 Dataset observation-increment shape/mask is invalid")
    if bool((values.masked_select(~valid) != 0).any()):
        raise AssertionError("real V10 Dataset contains a nonzero invalid increment")
    if bool(valid.masked_select(padding.unsqueeze(-1)).any()):
        raise AssertionError("real V10 Dataset padding contains a valid increment")

    smoke_dir = project_root / "src-v3" / "smoke"
    dt_invariance = _real_dt_invariance(project_root, v10, dataset, smoke_dir)

    first_parent_id = dataset.parent_ids[0]
    first_item = dataset[0]
    first_group = dataset.base._groups[first_parent_id].iloc[: v10.data.max_sequence_length]
    online_state = CausalTRObservationIncrementState()
    online = [
        online_state.update(parse_feature_json(str(row["tr_feature_values_json"])))
        for _, row in first_group.iterrows()
    ]
    online_values = torch.from_numpy(np.stack([item[0] for item in online])).float()
    online_valid = torch.from_numpy(np.stack([item[1] for item in online])).bool()
    if not torch.equal(online_values, first_item.tr_motion_features):
        raise AssertionError("V10 online raw-point state does not match the offline Dataset")
    if not torch.equal(online_valid, first_item.tr_motion_valid_mask):
        raise AssertionError("V10 online raw-point validity does not match the offline Dataset")

    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    model = LiquidTaskClassifier(v10).eval()
    with torch.inference_mode():
        output = model(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            tr_motion_features=tensors["tr_motion_features"],
            tr_motion_valid_mask=tensors["tr_motion_valid_mask"],
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    if not torch.isfinite(output["sequence_logits"]).all():
        raise AssertionError("V10 real Dataset forward produced a non-finite logit")

    liquid_state: torch.Tensor | None = None
    online_logits: list[torch.Tensor] = []
    first_length = int(batch["lengths"][0].item())
    with torch.inference_mode():
        for event_index in range(first_length):
            point = model.forward_online_point(
                tr_features=tensors["tr_features"][0:1, event_index],
                tr_missing_mask=tensors["tr_missing_mask"][0:1, event_index],
                tr_dt_seconds=tensors["tr_dt_seconds"][0:1, event_index],
                tr_dt_valid=tensors["tr_dt_valid"][0:1, event_index],
                tr_motion_features=tensors["tr_motion_features"][0:1, event_index],
                tr_motion_valid_mask=tensors["tr_motion_valid_mask"][0:1, event_index],
                rd_frames=tensors["rd_frames"][0:1, event_index],
                rd_frame_mask=tensors["rd_frame_mask"][0:1, event_index],
                rd_quality=tensors["rd_quality"][0:1, event_index],
                liquid_state=liquid_state,
            )
            liquid_state = point["liquid_state"]
            online_logits.append(point["window_logits"])
    online_logit_tensor = torch.stack(online_logits, dim=1)
    offline_logit_tensor = output["sequence_logits"][0:1, :first_length]
    max_difference = float((online_logit_tensor - offline_logit_tensor).abs().max().item())
    if not torch.allclose(online_logit_tensor, offline_logit_tensor, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"V10 online/offline prefix logits differ: {max_difference}")

    v9_checkpoint = project_root / V9_CHECKPOINT
    v9_model = LiquidTaskClassifier(v9).eval()
    load_task_checkpoint(v9_checkpoint, v9_model, expected_schema=v9.train.checkpoint_schema, map_location="cpu")
    with tempfile.TemporaryDirectory(prefix="v10-checkpoint-contract-", dir=smoke_dir) as temporary:
        temporary_root = Path(temporary)
        spoofed = torch.load(v9_checkpoint, map_location="cpu", weights_only=False)
        spoofed["checkpoint_schema"] = v10.train.checkpoint_schema
        spoofed["data_contract"] = {
            "index_sha256": v10.data.expected_index_sha256,
            "contract_sha256": v10.data.expected_contract_sha256,
            "stats_sha256": v10.data.expected_stats_sha256,
        }
        spoofed_path = temporary_root / "v9-spoofed-as-v10.pt"
        torch.save(spoofed, spoofed_path)
        try:
            load_task_checkpoint(
                spoofed_path,
                LiquidTaskClassifier(v10),
                expected_schema=v10.train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError as error:
            spoofed_rejected = "model/input contract" in str(error)
        else:
            spoofed_rejected = False
        if not spoofed_rejected:
            raise AssertionError("a schema-spoofed V9 checkpoint bypassed the V10 feature contract")

        checkpoint_path = temporary_root / "v10.pt"
        save_task_checkpoint(checkpoint_path, model, None, config=v10, epoch=0, global_step=0, metrics={})
        restored = LiquidTaskClassifier(v10).eval()
        load_task_checkpoint(
            checkpoint_path,
            restored,
            expected_schema=v10.train.checkpoint_schema,
            map_location="cpu",
        )
        with torch.inference_mode():
            restored_output = restored(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_motion_features=tensors["tr_motion_features"],
                tr_motion_valid_mask=tensors["tr_motion_valid_mask"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
        if not torch.allclose(output["window_logits"], restored_output["window_logits"], atol=1.0e-6, rtol=1.0e-5):
            raise AssertionError("V10 checkpoint round-trip changed logits")

    return {
        "contract_hashes": contract["hashes"],
        "paired_config_changes": {
            "data": sorted(data_changes),
            "model": [],
            "train_output_identity": sorted(train_changes),
            "unchanged_task_and_label_mapping": True,
            "unchanged_model_dimensions_and_architecture": True,
        },
        "real_batch_shapes": {
            key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)
        },
        "real_valid_count": int(valid.sum().item()),
        "real_invalid_count": int((~valid & ~padding.unsqueeze(-1)).sum().item()),
        "real_padding_count": int(padding.sum().item()),
        "real_dt_invariance": dt_invariance,
        "raw_point_online_state_matches_dataset": True,
        "forward": "passed",
        "online_offline_prefix_max_difference": max_difference,
        "v9_real_checkpoint_regression_load": "passed",
        "schema_spoofed_v9_to_v10_rejected": spoofed_rejected,
        "v10_checkpoint_round_trip": "passed",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(project_root: Path) -> dict[str, Any]:
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V10",
        "synthetic_contracts": _synthetic_contracts(),
        "model_contracts": _model_contracts(project_root),
        "formal_training": "未执行",
        "validation_performance": "未评估",
        "test": "未执行",
    }
    path = project_root / "src-v3" / "smoke" / "RADAR-LN3-V10-observation-increment-contract-result.json"
    result["result_path"] = str(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V10 no-dt causal TR observation increments and checkpoint isolation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
