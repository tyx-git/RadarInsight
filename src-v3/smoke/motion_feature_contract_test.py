from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.features import (
    CausalTRMotionState,
    TR_MOTION_NAMES,
    encode_causal_tr_motion_sequence,
    parse_feature_json,
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


def _assert_invalid_is_zero(values: np.ndarray, valid: np.ndarray) -> None:
    if values.shape != valid.shape or values.shape[-1] != len(TR_MOTION_NAMES):
        raise AssertionError("motion value/mask shape mismatch")
    if not np.isfinite(values).all():
        raise AssertionError("motion values are non-finite")
    if np.any(values[~valid] != 0.0):
        raise AssertionError("invalid motion values are not exactly zero")


def _synthetic_contracts() -> dict[str, Any]:
    single, single_valid = encode_causal_tr_motion_sequence([_point()], [0.0], [False])
    _assert_invalid_is_zero(single, single_valid)
    if single_valid.any():
        raise AssertionError("single first event unexpectedly produced a valid derivative")

    two, two_valid = encode_causal_tr_motion_sequence(
        [_point(course=359.0), _point(range_m=1010.0, radial=11.0, course=0.0, speed=12.0)],
        [0.0, 1.0],
        [False, True],
    )
    _assert_invalid_is_zero(two, two_valid)
    if two_valid[1, TR_MOTION_NAMES.index("reported_speed_acceleration_scaled")]:
        raise AssertionError("the unconfirmed first-point speed seeded a second-event derivative")
    if two_valid[1, TR_MOTION_NAMES.index("course_turn_rate_scaled")]:
        raise AssertionError("the unconfirmed first-point course seeded a second-event derivative")

    circular, circular_valid = encode_causal_tr_motion_sequence(
        [
            _point(course=120.0),
            _point(range_m=1010.0, radial=11.0, course=359.0, speed=12.0),
            _point(range_m=1020.0, radial=12.0, course=0.0, speed=13.0),
        ],
        [0.0, 1.0, 1.0],
        [False, True, True],
    )
    _assert_invalid_is_zero(circular, circular_valid)
    turn_index = TR_MOTION_NAMES.index("course_turn_rate_scaled")
    if abs(float(circular[2, turn_index]) - 1.0 / 45.0) > 1.0e-6:
        raise AssertionError("359-to-0 course transition did not use the shortest circular delta")

    invalid_cases: dict[str, tuple[float, bool]] = {
        "repeated_time": (0.0, False),
        "negative_time": (-1.0, False),
        "invalid_positive_value": (1.0, False),
    }
    for name, (dt, is_valid) in invalid_cases.items():
        values, valid = encode_causal_tr_motion_sequence(
            [_point(), _point(range_m=1010.0, radial=-10.0, course=90.0, speed=20.0)],
            [0.0, dt],
            [False, is_valid],
        )
        _assert_invalid_is_zero(values, valid)
        if valid[1].any():
            raise AssertionError(f"{name} generated a valid motion value")

    missing, missing_valid = encode_causal_tr_motion_sequence(
        [_point(), _point(range_m=None, radial=None, course=None, speed=None)],
        [0.0, 1.0],
        [False, True],
    )
    _assert_invalid_is_zero(missing, missing_valid)
    if missing_valid[1].any():
        raise AssertionError("missing current fields generated a valid motion value")

    extreme, extreme_valid = encode_causal_tr_motion_sequence(
        [_point(), _point(range_m=1100.0, radial=-100.0, course=180.0, speed=200.0)],
        [0.0, 1.0e-6],
        [False, True],
    )
    _assert_invalid_is_zero(extreme, extreme_valid)
    if float(np.abs(extreme).max()) > 5.0:
        raise AssertionError("extreme short interval escaped the frozen feature clip")

    long_gap, long_gap_valid = encode_causal_tr_motion_sequence(
        [_point(), _point(range_m=1100.0, radial=11.0, course=10.0, speed=12.0)],
        [0.0, 120.0],
        [False, True],
    )
    _assert_invalid_is_zero(long_gap, long_gap_valid)

    base = [_point(), _point(range_m=1010.0, course=5.0), _point(range_m=1020.0, course=10.0)]
    changed_future = [dict(value) for value in base]
    changed_future[2] = _point(range_m=4000.0, radial=-80.0, course=170.0, speed=90.0)
    first, first_valid = encode_causal_tr_motion_sequence(base, [0.0, 1.0, 1.0], [False, True, True])
    second, second_valid = encode_causal_tr_motion_sequence(
        changed_future,
        [0.0, 1.0, 1.0],
        [False, True, True],
    )
    if not np.array_equal(first[:2], second[:2]) or not np.array_equal(first_valid[:2], second_valid[:2]):
        raise AssertionError("future-point mutation changed an earlier causal prefix")

    state = CausalTRMotionState()
    state.update(_point(), dt_seconds=0.0, dt_valid=False)
    state.update(_point(range_m=1010.0), dt_seconds=1.0, dt_valid=True)
    state.reset()
    reset_values, reset_valid = state.update(_point(range_m=5000.0), dt_seconds=1.0, dt_valid=True)
    if reset_valid.any() or np.any(reset_values != 0.0):
        raise AssertionError("motion state leaked across a reset boundary")

    return {
        "single_point": "passed",
        "double_point": "passed",
        "circular_359_to_0": "passed",
        "invalid_dt_cases": list(invalid_cases),
        "missing_fields": "passed",
        "extreme_short_interval_clipped": "passed",
        "long_interval": "passed",
        "future_prefix_invariance": "passed",
        "state_reset": "passed",
    }


def _model_contracts(project_root: Path) -> dict[str, Any]:
    np.random.seed(20260807)
    torch.manual_seed(20260807)
    config_dir = project_root / "config" / "RADAR-LN3"
    v8_path = config_dir / "RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-config.json"
    v9_path = config_dir / "RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json"
    v8_payload = json.loads(v8_path.read_text(encoding="utf-8"))
    v9_payload = json.loads(v9_path.read_text(encoding="utf-8"))
    if v8_payload["task"] != v9_payload["task"]:
        raise AssertionError("V9 changed the DBB/CU task or label mapping")

    def changed_keys(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
        return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}

    data_changes = changed_keys(v8_payload["data"], v9_payload["data"])
    expected_data_changes = {
        "contract_path",
        "contract_schema_version",
        "expected_contract_sha256",
        "tr_motion_schema",
        "tr_motion_mask_mode",
    }
    model_changes = changed_keys(v8_payload["model"], v9_payload["model"])
    train_changes = changed_keys(v8_payload["train"], v9_payload["train"])
    expected_train_changes = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "checkpoint_schema",
    }
    if data_changes != expected_data_changes:
        raise AssertionError(f"V9 changed unexpected data config fields: {sorted(data_changes)}")
    if model_changes != {"tr_motion_dim"}:
        raise AssertionError(f"V9 changed unexpected model config fields: {sorted(model_changes)}")
    if train_changes != expected_train_changes:
        raise AssertionError(f"V9 changed unexpected train config fields: {sorted(train_changes)}")

    smoke_dir = project_root / "src-v3" / "smoke"
    with tempfile.TemporaryDirectory(prefix="v8-motion-config-reject-", dir=smoke_dir) as temporary:
        illegal_v8 = json.loads(json.dumps(v8_payload))
        illegal_v8["data"]["tr_motion_schema"] = "causal-tr-motion-v1"
        illegal_v8["data"]["tr_motion_mask_mode"] = "gating_only"
        illegal_v8["model"]["tr_motion_dim"] = len(TR_MOTION_NAMES)
        illegal_path = Path(temporary) / "illegal-v8-motion.json"
        illegal_path.write_text(json.dumps(illegal_v8, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            load_task_config(illegal_path)
        except ValueError as error:
            pre_v9_motion_rejected = "before RADAR-LN3-V9" in str(error)
        else:
            pre_v9_motion_rejected = False
    if not pre_v9_motion_rejected:
        raise AssertionError("a V8 configuration was allowed to enable V9 motion inputs")

    v8 = load_task_config(v8_path)
    v9 = load_task_config(v9_path)
    contract = verify_task_data_contract(project_root, v9)
    dataset = build_task_dataset(project_root, v9, split_name=v9.data.train_split_name, parent_limit=4)
    loader = build_task_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    motion = batch["tr_motion_features"]
    motion_valid = batch["tr_motion_valid_mask"]
    padding = batch["tr_padding_mask"]
    if motion.shape != (*batch["tr_features"].shape[:2], len(TR_MOTION_NAMES)):
        raise AssertionError("real V9 Dataset motion shape is invalid")
    if motion_valid.dtype != torch.bool or motion_valid.shape != motion.shape:
        raise AssertionError("real V9 Dataset motion valid mask is invalid")
    if bool((motion.masked_select(~motion_valid) != 0).any()):
        raise AssertionError("real V9 Dataset has a nonzero invalid motion value")
    if bool(motion_valid.masked_select(padding.unsqueeze(-1)).any()):
        raise AssertionError("real V9 Dataset padding has a valid motion value")

    first_parent_id = dataset.parent_ids[0]
    first_item = dataset[0]
    first_group = dataset.base._groups[first_parent_id].iloc[: v9.data.max_sequence_length]
    online_motion_state = CausalTRMotionState()
    online_motion: list[torch.Tensor] = []
    online_motion_valid: list[torch.Tensor] = []
    for _, row in first_group.iterrows():
        value, valid = online_motion_state.update(
            parse_feature_json(str(row["tr_feature_values_json"])),
            dt_seconds=float(row["tr_dt_seconds"]),
            dt_valid=str(row["tr_dt_valid"]).strip().lower() in {"1", "true", "yes", "y"},
        )
        online_motion.append(torch.from_numpy(value))
        online_motion_valid.append(torch.from_numpy(valid))
    online_motion_tensor = torch.stack(online_motion).float()
    online_motion_valid_tensor = torch.stack(online_motion_valid).bool()
    if not torch.equal(online_motion_tensor, first_item.tr_motion_features):
        raise AssertionError("online raw-point motion state does not match the offline Dataset")
    if not torch.equal(online_motion_valid_tensor, first_item.tr_motion_valid_mask):
        raise AssertionError("online raw-point motion validity does not match the offline Dataset")

    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    v9_model = LiquidTaskClassifier(v9).eval()
    with torch.inference_mode():
        output = v9_model(
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
        raise AssertionError("V9 real Dataset forward produced a non-finite logit")
    online_liquid_state: torch.Tensor | None = None
    online_logits: list[torch.Tensor] = []
    first_length = int(batch["lengths"][0].item())
    with torch.inference_mode():
        for event_index in range(first_length):
            online_output = v9_model.forward_online_point(
                tr_features=tensors["tr_features"][0:1, event_index],
                tr_missing_mask=tensors["tr_missing_mask"][0:1, event_index],
                tr_dt_seconds=tensors["tr_dt_seconds"][0:1, event_index],
                tr_dt_valid=tensors["tr_dt_valid"][0:1, event_index],
                tr_motion_features=tensors["tr_motion_features"][0:1, event_index],
                tr_motion_valid_mask=tensors["tr_motion_valid_mask"][0:1, event_index],
                rd_frames=tensors["rd_frames"][0:1, event_index],
                rd_frame_mask=tensors["rd_frame_mask"][0:1, event_index],
                rd_quality=tensors["rd_quality"][0:1, event_index],
                liquid_state=online_liquid_state,
            )
            online_liquid_state = online_output["liquid_state"]
            online_logits.append(online_output["window_logits"])
    online_logit_tensor = torch.stack(online_logits, dim=1)
    offline_logit_tensor = output["sequence_logits"][0:1, :first_length]
    online_offline_max_difference = float((online_logit_tensor - offline_logit_tensor).abs().max().item())
    if not torch.allclose(online_logit_tensor, offline_logit_tensor, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            f"V9 online/offline prefix logits differ: max_difference={online_offline_max_difference}"
        )
    try:
        v9_model(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    except ValueError as error:
        missing_motion_rejected = "requires causal TR motion" in str(error)
    else:
        missing_motion_rejected = False
    if not missing_motion_rejected:
        raise AssertionError("V9 model accepted a batch without motion tensors")
    out_of_range_motion = tensors["tr_motion_features"].clone()
    out_of_range_valid = tensors["tr_motion_valid_mask"].clone()
    out_of_range_motion[0, 0, 0] = 6.0
    out_of_range_valid[0, 0, 0] = True
    try:
        v9_model(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            tr_motion_features=out_of_range_motion,
            tr_motion_valid_mask=out_of_range_valid,
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    except ValueError as error:
        out_of_range_motion_rejected = "outside the frozen clip" in str(error)
    else:
        out_of_range_motion_rejected = False
    if not out_of_range_motion_rejected:
        raise AssertionError("V9 model accepted a motion value outside the frozen clip")

    v8_dataset = build_task_dataset(project_root, v8, split_name=v8.data.train_split_name, parent_limit=2)
    v8_batch = next(iter(build_task_dataloader(v8_dataset, batch_size=2, shuffle=False)))
    if v8_batch["tr_motion_features"].shape[-1] != 0:
        raise AssertionError("V8 Dataset unexpectedly computes V9 motion features")
    v8_model = LiquidTaskClassifier(v8).eval()
    real_v8_checkpoint = (
        project_root
        / "artifacts"
        / "RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-formal"
        / "best_model.pt"
    )
    if not real_v8_checkpoint.exists():
        raise FileNotFoundError(f"V8 paired baseline checkpoint is missing: {real_v8_checkpoint}")
    load_task_checkpoint(
        real_v8_checkpoint,
        v8_model,
        expected_schema=v8.train.checkpoint_schema,
        map_location="cpu",
    )
    v8_tensors = {key: value for key, value in v8_batch.items() if isinstance(value, torch.Tensor)}
    with torch.inference_mode():
        v8_output = v8_model(
            tr_features=v8_tensors["tr_features"],
            tr_missing_mask=v8_tensors["tr_missing_mask"],
            tr_dt_seconds=v8_tensors["tr_dt_seconds"],
            tr_dt_valid=v8_tensors["tr_dt_valid"],
            rd_frames=v8_tensors["rd_frames"],
            rd_frame_mask=v8_tensors["rd_frame_mask"],
            rd_quality=v8_tensors["rd_quality"],
            tr_padding_mask=v8_tensors["tr_padding_mask"],
        )
    if not torch.isfinite(v8_output["window_logits"]).all():
        raise AssertionError("V8 regression forward produced a non-finite logit")

    with tempfile.TemporaryDirectory(prefix="v9-motion-contract-", dir=smoke_dir) as temporary:
        temporary_root = Path(temporary)
        v8_checkpoint = temporary_root / "v8.pt"
        save_task_checkpoint(v8_checkpoint, v8_model, None, config=v8, epoch=0, global_step=0, metrics={})
        try:
            load_task_checkpoint(
                v8_checkpoint,
                LiquidTaskClassifier(v9),
                expected_schema=v9.train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError:
            v8_to_v9_rejected = True
        else:
            v8_to_v9_rejected = False
        if not v8_to_v9_rejected:
            raise AssertionError("V8 checkpoint was accepted by the V9 model contract")
        spoofed_payload = torch.load(v8_checkpoint, map_location="cpu", weights_only=False)
        spoofed_payload["checkpoint_schema"] = v9.train.checkpoint_schema
        spoofed_payload["data_contract"] = {
            "index_sha256": v9.data.expected_index_sha256,
            "contract_sha256": v9.data.expected_contract_sha256,
            "stats_sha256": v9.data.expected_stats_sha256,
        }
        spoofed_checkpoint = temporary_root / "v8-spoofed-as-v9.pt"
        torch.save(spoofed_payload, spoofed_checkpoint)
        try:
            load_task_checkpoint(
                spoofed_checkpoint,
                LiquidTaskClassifier(v9),
                expected_schema=v9.train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError as error:
            spoofed_v8_contract_rejected = "model/input contract" in str(error)
        else:
            spoofed_v8_contract_rejected = False
        if not spoofed_v8_contract_rejected:
            raise AssertionError("a schema-spoofed V8 checkpoint bypassed the V9 input contract")

        v9_checkpoint = temporary_root / "v9.pt"
        save_task_checkpoint(v9_checkpoint, v9_model, None, config=v9, epoch=0, global_step=0, metrics={})
        restored = LiquidTaskClassifier(v9).eval()
        load_task_checkpoint(v9_checkpoint, restored, expected_schema=v9.train.checkpoint_schema, map_location="cpu")
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
        if not torch.allclose(output["window_logits"], restored_output["window_logits"], atol=1e-6, rtol=1e-5):
            raise AssertionError("V9 checkpoint round-trip changed logits")

    return {
        "v9_contract_hashes": contract["hashes"],
        "paired_config_changes": {
            "data": sorted(data_changes),
            "model": sorted(model_changes),
            "train_output_identity": sorted(train_changes),
            "unchanged_task_and_label_mapping": True,
            "pre_v9_motion_config_rejected": pre_v9_motion_rejected,
        },
        "real_batch_shapes": {
            key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)
        },
        "real_motion_valid_count": int(motion_valid.sum().item()),
        "real_motion_invalid_count": int((~motion_valid & ~padding.unsqueeze(-1)).sum().item()),
        "real_padding_count": int(padding.sum().item()),
        "raw_point_online_motion_matches_dataset": True,
        "v9_forward": "passed",
        "v9_online_offline_prefix_max_difference": online_offline_max_difference,
        "v9_missing_motion_rejected": missing_motion_rejected,
        "v9_out_of_range_motion_rejected": out_of_range_motion_rejected,
        "v8_dataset_motion_dim": int(v8_batch["tr_motion_features"].shape[-1]),
        "v8_real_checkpoint": str(real_v8_checkpoint),
        "v8_real_checkpoint_regression_forward": "passed",
        "v8_to_v9_checkpoint_rejected": v8_to_v9_rejected,
        "schema_spoofed_v8_model_contract_rejected": spoofed_v8_contract_rejected,
        "v9_checkpoint_round_trip": "passed",
    }


def run(project_root: Path) -> dict[str, Any]:
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V9",
        "synthetic_contracts": _synthetic_contracts(),
        "model_contracts": _model_contracts(project_root),
        "formal_training": "未执行",
        "validation_performance": "未评估",
        "test": "未执行",
    }
    path = project_root / "src-v3" / "smoke" / "RADAR-LN3-V9-motion-feature-contract-result.json"
    result["result_path"] = str(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V9 causal TR motion features and checkpoint isolation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
