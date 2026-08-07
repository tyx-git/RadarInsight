from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from radar_ln3.cascade import EXPECTED_TASKS, LiquidSoftCascade, OnlineLiquidSoftCascade, load_cascade
from radar_ln3.data import CQ08PointSequenceDataset, RD_QUALITY_SCHEMA, encode_causal_rd_quality
from radar_ln3.metrics import summarize_records
from radar_ln3.stable import TargetStableStateMachine
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    LiquidTaskConfig,
    REPAIRED_TASK_SCHEMA_VERSION,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    summarize_task_records,
)


def _repaired(config: LiquidTaskConfig, *, normalization: str = "recurrent", rd_aggregation: str = "masked_mean") -> LiquidTaskConfig:
    return replace(
        config,
        schema_version=REPAIRED_TASK_SCHEMA_VERSION,
        data=replace(
            config.data,
            rd_quality_schema=RD_QUALITY_SCHEMA,
            contract_schema_version=REPAIRED_TASK_SCHEMA_VERSION,
        ),
        train=replace(
            config.train,
            checkpoint_schema=config.train.checkpoint_schema.replace("RADAR-LN3-V5", REPAIRED_TASK_SCHEMA_VERSION, 1),
        ),
        model=replace(
            config.model,
            liquid_state_normalization=normalization,
            rd_frame_aggregation=rd_aggregation,
        ),
    )


def _synthetic_batch(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    batch_size, length, frames = 1, 3, 2
    qualities = [
        encode_causal_rd_quality(
            present=True,
            original_count=2,
            duration_seconds=0.03,
            local_time_non_monotonic=False,
            truncated=False,
            max_rd_frames_per_point=12,
        ),
        encode_causal_rd_quality(
            present=True,
            original_count=1,
            duration_seconds=0.0,
            local_time_non_monotonic=False,
            truncated=False,
            max_rd_frames_per_point=12,
        ),
        encode_causal_rd_quality(
            present=False,
            original_count=0,
            duration_seconds=0.0,
            local_time_non_monotonic=False,
            truncated=False,
            max_rd_frames_per_point=12,
        ),
    ]
    return {
        "tr_features": torch.randn(batch_size, length, 14, generator=generator).to(device),
        "tr_missing_mask": torch.zeros(batch_size, length, 11, device=device),
        "tr_dt_seconds": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=device),
        "tr_dt_valid": torch.tensor([[False, True, False]], dtype=torch.bool, device=device),
        "rd_frames": torch.randn(batch_size, length, frames, 2, 21, 384, generator=generator).to(device),
        "rd_frame_mask": torch.tensor([[[True, True], [True, False], [False, False]]], dtype=torch.bool, device=device),
        "rd_quality": torch.from_numpy(np.stack(qualities, axis=0)).to(device).unsqueeze(0),
        "tr_padding_mask": torch.zeros(batch_size, length, dtype=torch.bool, device=device),
    }


def _models(configs: dict[str, LiquidTaskConfig], device: torch.device) -> dict[str, LiquidTaskClassifier]:
    return {name: LiquidTaskClassifier(configs[name]).to(device).eval() for name in EXPECTED_TASKS}


def _cascade(models: dict[str, LiquidTaskClassifier]) -> LiquidSoftCascade:
    return LiquidSoftCascade(
        models["dbb_cu_gate"], models["dbb_specialist"], models["cu_specialist"]
    ).eval()


def _online_matches_batch(cascade: LiquidSoftCascade, batch: dict[str, torch.Tensor]) -> float:
    online = OnlineLiquidSoftCascade(cascade, confirm_count=2)
    online_probabilities: list[torch.Tensor] = []
    for event_index in range(batch["tr_features"].shape[1]):
        result = online.step(
            tr_features=batch["tr_features"][:, event_index],
            tr_missing_mask=batch["tr_missing_mask"][:, event_index],
            tr_dt_seconds=batch["tr_dt_seconds"][:, event_index],
            tr_dt_valid=batch["tr_dt_valid"][:, event_index],
            rd_frames=batch["rd_frames"][:, event_index],
            rd_frame_mask=batch["rd_frame_mask"][:, event_index],
            rd_quality=batch["rd_quality"][:, event_index],
        )
        if event_index == 0 and result.stable_confirmed:
            raise AssertionError("stable output was confirmed before confirm_count")
        online_probabilities.append(result.window_probabilities)
    with torch.inference_mode():
        expected = cascade(**batch)["sequence_probabilities"]
    actual = torch.stack(online_probabilities, dim=1)
    max_difference = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"cascade batch/online mismatch: max_difference={max_difference}")
    first = online_probabilities[0].clone()
    online.reset()
    reset = online.step(
        tr_features=batch["tr_features"][:, 0],
        tr_missing_mask=batch["tr_missing_mask"][:, 0],
        tr_dt_seconds=batch["tr_dt_seconds"][:, 0],
        tr_dt_valid=batch["tr_dt_valid"][:, 0],
        rd_frames=batch["rd_frames"][:, 0],
        rd_frame_mask=batch["rd_frame_mask"][:, 0],
        rd_quality=batch["rd_quality"][:, 0],
    )
    if not torch.allclose(first, reset.window_probabilities, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("cascade reset did not restore the initial target state")
    return max_difference


def run(project_root: Path) -> dict[str, object]:
    config_dir = project_root / "config" / "RADAR-LN3"
    paths = {
        "dbb_cu_gate": config_dir / "RADAR-LN3-V5-cq08-liquid-dbb-cu-gate-config.json",
        "dbb_specialist": config_dir / "RADAR-LN3-V5-cq08-liquid-dbb-specialist-config.json",
        "cu_specialist": config_dir / "RADAR-LN3-V5-cq08-liquid-cu-specialist-config.json",
    }
    legacy = {name: load_task_config(path) for name, path in paths.items()}
    try:
        LiquidTaskClassifier(legacy["dbb_cu_gate"])
    except ValueError as error:
        legacy_rejected = "audit-only" in str(error)
    else:
        legacy_rejected = False
    if not legacy_rejected:
        raise AssertionError("legacy task config was not rejected by the repaired input contract")

    repaired = {name: _repaired(config) for name, config in legacy.items()}
    gate_data = repaired["dbb_cu_gate"].data
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=project_root / gate_data.index_path,
        split_name=gate_data.train_split_name,
        max_sequence_length=gate_data.max_sequence_length,
        max_rd_frames_per_point=gate_data.max_rd_frames_per_point,
        rd_amplitude_center=gate_data.rd_amplitude_center,
        rd_amplitude_scale=gate_data.rd_amplitude_scale,
        exclude_anomalous_parents=gate_data.exclude_anomalous_parents,
        expected_schema_version=gate_data.index_schema_version,
    )
    partial_parent = next(
        parent_id
        for parent_id in dataset.parent_ids
        if dataset._groups[parent_id]["rd_quality_partial_parent"].astype(str).str.lower().isin({"1", "true", "yes", "y"}).any()
    )
    partial_index = dataset.parent_ids.index(partial_parent)
    quality_before = dataset[partial_index].rd_quality.clone()
    group = dataset._groups[partial_parent]
    group["rd_quality_partial_parent"] = ~group["rd_quality_partial_parent"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    group["rd_quality_parent_order_anomaly"] = ~group["rd_quality_parent_order_anomaly"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    quality_after = dataset[partial_index].rd_quality
    parent_global_quality_max_difference = float((quality_before - quality_after).abs().max().item())
    if parent_global_quality_max_difference != 0.0:
        raise AssertionError("parent-global RD quality fields still affect the model input")
    quality_a = encode_causal_rd_quality(
        present=True,
        original_count=3,
        duration_seconds=0.05,
        local_time_non_monotonic=False,
        truncated=False,
        max_rd_frames_per_point=12,
    )
    quality_b = encode_causal_rd_quality(
        present=True,
        original_count=3,
        duration_seconds=0.05,
        local_time_non_monotonic=False,
        truncated=False,
        max_rd_frames_per_point=12,
    )
    if not torch.equal(torch.from_numpy(quality_a), torch.from_numpy(quality_b)):
        raise AssertionError("causal RD quality changed without a current-event input change")

    five = summarize_records([{"true_id": value, "pred_id": 0} for value in range(5)], scope="probe")
    task = summarize_task_records(
        [{"true_id": 0, "pred_id": 0}, {"true_id": 1, "pred_id": 0}],
        class_names=("A", "B"),
        scope="probe",
    )
    if abs(float(five["macro_precision"]) - 0.04) > 1.0e-12 or abs(float(task["macro_precision"]) - 0.25) > 1.0e-12:
        raise AssertionError("fixed-class macro precision still drops zero-prediction classes")

    stable = TargetStableStateMachine(confirm_count=2)
    first = stable.update(torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0]))
    second = stable.update(torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0]))
    if first.stable_confirmed or first.stable_label_id is not None or first.stable_label_name is not None:
        raise AssertionError("unconfirmed stable state was encoded as a class")
    if not second.stable_confirmed or second.stable_label_name != "DroneTarget":
        raise AssertionError("stable state did not confirm after confirm_count")
    probability_state = TargetStableStateMachine(confirm_count=1)
    probability_decision = probability_state.update_probabilities(torch.tensor([0.80, 0.05, 0.05, 0.05, 0.05]))
    if abs(probability_decision.window_confidence - 0.80) > 1.0e-6:
        raise AssertionError("probability stable update applied an unintended second softmax")
    try:
        probability_state.update(torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0]))
    except RuntimeError:
        stable_mixed_mode_rejected = True
    else:
        stable_mixed_mode_rejected = False
    if not stable_mixed_mode_rejected:
        raise AssertionError("stable state machine accepted mixed probability/logit updates")

    device = torch.device("cpu")
    batch = _synthetic_batch(device)
    models = _models(repaired, device)
    cascade = _cascade(models)
    baseline_difference = _online_matches_batch(cascade, batch)
    with torch.inference_mode():
        probabilities = cascade(**batch)["sequence_probabilities"]
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]), atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("soft cascade probabilities are not finite and normalized")

    smoke_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="cascade-contract-", dir=smoke_dir) as temporary:
        root = Path(temporary)
        checkpoints: dict[str, Path] = {}
        for name, model in models.items():
            path = root / f"{name}.pt"
            save_task_checkpoint(path, model, None, config=repaired[name], epoch=0, global_step=0, metrics={})
            checkpoints[name] = path
        restored, payloads = load_cascade(repaired, checkpoints, device=device)
        restored_difference = _online_matches_batch(restored, batch)
        if set(payloads) != set(EXPECTED_TASKS):
            raise AssertionError("cascade checkpoint loader did not restore all tasks")
        broken_payload = torch.load(checkpoints["dbb_cu_gate"], map_location="cpu", weights_only=False)
        broken_payload["model_contract"].pop("rd_quality_schema")
        broken = root / "legacy.pt"
        torch.save(broken_payload, broken)
        try:
            load_task_checkpoint(
                broken,
                LiquidTaskClassifier(repaired["dbb_cu_gate"]),
                expected_schema=repaired["dbb_cu_gate"].train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError:
            checkpoint_rejected = True
        else:
            checkpoint_rejected = False
        if not checkpoint_rejected:
            raise AssertionError("checkpoint missing repaired input semantics was not rejected")
        hash_payload = torch.load(checkpoints["dbb_cu_gate"], map_location="cpu", weights_only=False)
        hash_payload["data_contract"]["index_sha256"] = "0" * 64
        hash_broken = root / "wrong-data-hash.pt"
        torch.save(hash_payload, hash_broken)
        try:
            load_task_checkpoint(
                hash_broken,
                LiquidTaskClassifier(repaired["dbb_cu_gate"]),
                expected_schema=repaired["dbb_cu_gate"].train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError:
            checkpoint_data_hash_rejected = True
        else:
            checkpoint_data_hash_rejected = False
        if not checkpoint_data_hash_rejected:
            raise AssertionError("checkpoint with mismatched data hashes was not rejected")

    normalization_configs = {
        name: _repaired(config, normalization="readout_only") for name, config in legacy.items()
    }
    normalization_difference = _online_matches_batch(_cascade(_models(normalization_configs, device)), batch)
    rd_temporal_configs = {
        name: _repaired(config, rd_aggregation="causal_gru") for name, config in legacy.items()
    }
    rd_temporal_difference = _online_matches_batch(_cascade(_models(rd_temporal_configs, device)), batch)
    future_configs = {
        name: replace(
            config,
            schema_version="RADAR-LN3-V7",
            model=replace(config.model, dt_valid_feature_mode="dynamics_only"),
            train=replace(
                config.train,
                checkpoint_schema=config.train.checkpoint_schema.replace("RADAR-LN3-V6", "RADAR-LN3-V7", 1),
            ),
        )
        for name, config in repaired.items()
    }
    future_version_difference = _online_matches_batch(_cascade(_models(future_configs, device)), batch)
    if any(config.data.contract_schema_version != REPAIRED_TASK_SCHEMA_VERSION for config in future_configs.values()):
        raise AssertionError("future model-only configuration did not reuse the repaired V6 data contract")
    mixed_versions = dict(repaired)
    mixed_versions["dbb_cu_gate"] = future_configs["dbb_cu_gate"]
    try:
        _cascade(_models(mixed_versions, device))
    except ValueError:
        mixed_cascade_versions_rejected = True
    else:
        mixed_cascade_versions_rejected = False
    if not mixed_cascade_versions_rejected:
        raise AssertionError("soft cascade accepted mixed run/configuration versions")
    invalid = _repaired(legacy["dbb_cu_gate"], normalization="readout_only", rd_aggregation="causal_gru")
    try:
        LiquidTaskClassifier(invalid)
    except ValueError:
        simultaneous_ablation_rejected = True
    else:
        simultaneous_ablation_rejected = False
    if not simultaneous_ablation_rejected:
        raise AssertionError("simultaneous liquid/RD temporal ablations were not rejected")

    result = {
        "status": "passed",
        "repaired_config_schema": REPAIRED_TASK_SCHEMA_VERSION,
        "repaired_contract_schema": repaired["dbb_cu_gate"].data.contract_schema_version,
        "future_model_only_config_schema": future_configs["dbb_cu_gate"].schema_version,
        "future_model_only_contract_schema": future_configs["dbb_cu_gate"].data.contract_schema_version,
        "legacy_config_rejected": legacy_rejected,
        "rd_quality_schema": RD_QUALITY_SCHEMA,
        "real_partial_parent": partial_parent,
        "parent_global_quality_max_difference": parent_global_quality_max_difference,
        "five_class_macro_precision_probe": five["macro_precision"],
        "two_class_macro_precision_probe": task["macro_precision"],
        "stable_unconfirmed_is_null": first.stable_label_id is None,
        "stable_probability_no_double_softmax": probability_decision.window_confidence,
        "stable_mixed_mode_rejected": stable_mixed_mode_rejected,
        "soft_cascade_probability_sum": probabilities.sum(dim=-1).tolist(),
        "baseline_batch_online_max_difference": baseline_difference,
        "checkpoint_round_trip_batch_online_max_difference": restored_difference,
        "legacy_checkpoint_rejected": checkpoint_rejected,
        "checkpoint_data_hash_rejected": checkpoint_data_hash_rejected,
        "readout_only_batch_online_max_difference": normalization_difference,
        "causal_rd_gru_batch_online_max_difference": rd_temporal_difference,
        "future_version_reuses_v6_contract_batch_online_max_difference": future_version_difference,
        "mixed_cascade_versions_rejected": mixed_cascade_versions_rejected,
        "simultaneous_ablation_rejected": simultaneous_ablation_rejected,
        "formal_training": "未执行",
        "validation": "未执行",
        "test": "未执行",
    }
    result_path = Path(__file__).resolve().parent / "causal_cascade_contract_result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check repaired causal quality, metrics, stable and liquid cascade contracts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
