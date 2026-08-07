from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from radar_ln3.cascade import EXPECTED_TASKS, LiquidSoftCascade, OnlineLiquidSoftCascade, validate_cascade_configs
from radar_ln3.rejection import (
    MAX_CLASS_PROBABILITY_SCORE,
    REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    RejectionCalibration,
    RejectionPolicy,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    sha256_file,
    task_prefix_loss,
    verify_task_data_contract,
)
from train.tasks import build_checkpoint_selection, checkpoint_selection_is_better


CONFIG_STEMS = {
    "dbb_cu_gate": "dbb-cu-gate",
    "dbb_specialist": "dbb-specialist",
    "cu_specialist": "cu-specialist",
}


def _load_version(project_root: Path, version: int):
    config_dir = project_root / "config" / "RADAR-LN3"
    return {
        task: load_task_config(
            config_dir / f"RADAR-LN3-V{version}-cq08-liquid-{stem}-config.json"
        )
        for task, stem in CONFIG_STEMS.items()
    }


def _config_path(project_root: Path, version: int, task: str) -> Path:
    return (
        project_root
        / "config"
        / "RADAR-LN3"
        / f"RADAR-LN3-V{version}-cq08-liquid-{CONFIG_STEMS[task]}-config.json"
    )


def _normalized_payload(project_root: Path, version: int, task: str, *, drop_selection: bool) -> dict:
    payload = json.loads(_config_path(project_root, version, task).read_text(encoding="utf-8"))
    value = copy.deepcopy(payload)
    value["schema_version"] = "<run-version>"
    value["config_name"] = "<config-name>"
    value["model"].pop("dt_valid_feature_mode", None)
    train = value["train"]
    train["checkpoint_path"] = "<best-checkpoint>"
    train["last_checkpoint_path"] = "<last-checkpoint>"
    train["formal_artifact_dir"] = "<artifact-dir>"
    train["checkpoint_schema"] = "<checkpoint-schema>"
    if drop_selection:
        train.pop("checkpoint_selection_metric", None)
    return value


def _synthetic_batch(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    return {
        "tr_features": torch.randn(1, 3, 14, generator=generator).to(device),
        "tr_missing_mask": torch.zeros(1, 3, 11, device=device),
        "tr_dt_seconds": torch.tensor([[0.0, 1.0, 0.0]], device=device),
        "tr_dt_valid": torch.tensor([[False, True, False]], dtype=torch.bool, device=device),
        "rd_frames": torch.randn(1, 3, 1, 2, 21, 384, generator=generator).to(device),
        "rd_frame_mask": torch.ones(1, 3, 1, dtype=torch.bool, device=device),
        "rd_quality": torch.zeros(1, 3, 7, device=device),
        "tr_padding_mask": torch.zeros(1, 3, dtype=torch.bool, device=device),
    }


def _cascade(configs, device: torch.device) -> LiquidSoftCascade:
    models = {name: LiquidTaskClassifier(configs[name]).to(device).eval() for name in EXPECTED_TASKS}
    return LiquidSoftCascade(
        models["dbb_cu_gate"], models["dbb_specialist"], models["cu_specialist"]
    ).to(device).eval()


def _batch_online_difference(cascade: LiquidSoftCascade, batch: dict[str, torch.Tensor]) -> float:
    with torch.inference_mode():
        expected = cascade(**batch)["sequence_probabilities"]
    online = OnlineLiquidSoftCascade(cascade)
    rows = []
    for index in range(batch["tr_features"].shape[1]):
        result = online.step(
            tr_features=batch["tr_features"][:, index],
            tr_missing_mask=batch["tr_missing_mask"][:, index],
            tr_dt_seconds=batch["tr_dt_seconds"][:, index],
            tr_dt_valid=batch["tr_dt_valid"][:, index],
            rd_frames=batch["rd_frames"][:, index],
            rd_frame_mask=batch["rd_frame_mask"][:, index],
            rd_quality=batch["rd_quality"][:, index],
        )
        if result.rejected is not None or result.decision_state != "not_calibrated":
            raise AssertionError("default online rejection policy emitted a calibrated decision")
        rows.append(result.window_probabilities)
    actual = torch.stack(rows, dim=1)
    difference = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"V7 cascade batch/online mismatch: {difference}")
    return difference


def _metric_result(per_class: list[dict[str, float | int | str]], macro_f1: float):
    return {"parent_metrics": {"per_class": per_class, "macro_f1": macro_f1}}


def run(project_root: Path) -> dict[str, object]:
    torch.manual_seed(20260806)
    versions = {version: _load_version(project_root, version) for version in (6, 7, 8)}
    config_hashes = {
        f"V{version}_{task}": sha256_file(_config_path(project_root, version, task))
        for version in (7, 8)
        for task in EXPECTED_TASKS
    }
    for version in (7, 8):
        validate_cascade_configs(versions[version])
        for config in versions[version].values():
            verify_task_data_contract(project_root, config)
            if config.model.dt_valid_feature_mode != "dynamics_only":
                raise AssertionError(f"V{version} did not disable dt_valid as a fusion feature")
    if any(config.train.checkpoint_selection_metric != "parent_macro_f1" for config in versions[7].values()):
        raise AssertionError("V7 changed checkpoint selection in the dt_valid-only ablation")
    if any(config.train.checkpoint_selection_metric != "parent_min_precision_recall" for config in versions[8].values()):
        raise AssertionError("V8 did not enable weak-class checkpoint selection")
    for task in EXPECTED_TASKS:
        if _normalized_payload(project_root, 6, task, drop_selection=False) != _normalized_payload(
            project_root, 7, task, drop_selection=False
        ):
            raise AssertionError(f"V7 {task} changes more than dt_valid exposure and versioned output identity")
        if _normalized_payload(project_root, 7, task, drop_selection=True) != _normalized_payload(
            project_root, 8, task, drop_selection=True
        ):
            raise AssertionError(f"V8 {task} changes more than checkpoint selection and versioned output identity")

    v6_gate = versions[6]["dbb_cu_gate"]
    v7_gate = versions[7]["dbb_cu_gate"]
    v6_model = LiquidTaskClassifier(v6_gate)
    load_task_checkpoint(
        project_root / v6_gate.train.checkpoint_path,
        v6_model,
        expected_schema=v6_gate.train.checkpoint_schema,
        map_location="cpu",
    )
    v7_model = LiquidTaskClassifier(v7_gate)
    if v6_model.fusion_input_dim != 80 or v7_model.fusion_input_dim != 79:
        raise AssertionError("dt_valid fusion removal did not change exactly one input dimension")
    try:
        load_task_checkpoint(
            project_root / v6_gate.train.checkpoint_path,
            v7_model,
            expected_schema=v6_gate.train.checkpoint_schema,
            map_location="cpu",
        )
    except ValueError:
        v6_v7_checkpoint_rejected = True
    else:
        raise AssertionError("V7 accepted a V6 checkpoint with the old fusion contract")

    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().clone())

    hook = v7_model.fusion.register_forward_pre_hook(capture)
    batch = _synthetic_batch(torch.device("cpu"))
    with torch.inference_mode():
        v7_model(**batch)
        changed = dict(batch)
        changed["tr_dt_valid"] = ~batch["tr_dt_valid"]
        v7_model(**changed)
    hook.remove()
    if len(captured) != 2 or not torch.equal(captured[0], captured[1]):
        raise AssertionError("V7 dt_valid still changes the direct fusion/classification input")
    with torch.no_grad():
        event = v7_model.fusion(captured[0])[:, 1]
    state = torch.zeros(1, v7_gate.model.liquid_hidden_dim)
    valid_state, _, _ = v7_model.liquid.step(
        event, torch.ones(1), state, torch.ones(1, dtype=torch.bool)
    )
    invalid_state, _, _ = v7_model.liquid.step(
        event, torch.ones(1), state, torch.zeros(1, dtype=torch.bool)
    )
    dynamics_difference = float((valid_state - invalid_state).abs().max().item())
    if dynamics_difference <= 0.0:
        raise AssertionError("V7 dt_valid no longer controls liquid dynamics")

    cascade = _cascade(versions[7], torch.device("cpu"))
    batch_online_difference = _batch_online_difference(cascade, batch)
    mixed = dict(versions[7])
    mixed["cu_specialist"] = versions[8]["cu_specialist"]
    try:
        validate_cascade_configs(mixed)
    except ValueError:
        mixed_v7_v8_rejected = True
    else:
        raise AssertionError("cascade accepted mixed V7/V8 task configurations")

    with tempfile.TemporaryDirectory(prefix="v7-v8-repair-", dir=project_root / "src-v3" / "smoke") as temporary:
        checkpoint = Path(temporary) / "v7.pt"
        save_task_checkpoint(
            checkpoint, v7_model, None, config=v7_gate, epoch=0, global_step=0, metrics={}
        )
        payload = load_task_checkpoint(
            checkpoint,
            LiquidTaskClassifier(v7_gate),
            expected_schema=v7_gate.train.checkpoint_schema,
            map_location="cpu",
        )
        if payload["model_contract"].get("dt_valid_feature_mode") != "dynamics_only":
            raise AssertionError("V7 checkpoint omitted dt_valid_feature_mode")
        try:
            load_task_checkpoint(
                checkpoint,
                LiquidTaskClassifier(versions[8]["dbb_cu_gate"]),
                expected_schema=versions[8]["dbb_cu_gate"].train.checkpoint_schema,
                map_location="cpu",
            )
        except ValueError:
            v7_v8_checkpoint_rejected = True
        else:
            raise AssertionError("V8 accepted a V7 checkpoint with different selection/version semantics")

    stronger_floor = build_checkpoint_selection(
        _metric_result(
            [
                {"class": "A", "precision": 0.80, "recall": 0.82, "support": 10},
                {"class": "B", "precision": 0.84, "recall": 0.81, "support": 10},
            ],
            0.90,
        ),
        "parent_min_precision_recall",
    )
    higher_macro_weaker_floor = build_checkpoint_selection(
        _metric_result(
            [
                {"class": "A", "precision": 0.99, "recall": 0.99, "support": 10},
                {"class": "B", "precision": 0.70, "recall": 0.99, "support": 10},
            ],
            0.95,
        ),
        "parent_min_precision_recall",
    )
    if not checkpoint_selection_is_better(stronger_floor, higher_macro_weaker_floor):
        raise AssertionError("weak-class floor did not outrank a higher Macro-F1 candidate")
    tie_better = dict(stronger_floor)
    tie_better["tie_break_value"] = 0.91
    tie_better["ranking_key"] = [stronger_floor["primary_value"], 0.91]
    if not checkpoint_selection_is_better(tie_better, stronger_floor):
        raise AssertionError("Macro-F1 did not break an equal weak-class-floor tie")
    try:
        build_checkpoint_selection(
            _metric_result(
                [
                    {"class": "A", "precision": 1.0, "recall": 1.0, "support": 10},
                    {"class": "B", "precision": 0.0, "recall": 0.0, "support": 0},
                ],
                0.50,
            ),
            "parent_min_precision_recall",
        )
    except ValueError:
        zero_support_rejected = True
    else:
        raise AssertionError("weak-class selector accepted a zero-support validation class")

    probabilities = torch.tensor([0.05, 0.10, 0.10, 0.20, 0.55])
    default_decision = RejectionPolicy().decide(probabilities)
    if default_decision.rejected is not None or default_decision.class_label_name != "UnknownTarget":
        raise AssertionError("uncalibrated rejection changed the UnknownTarget class label")
    try:
        RejectionPolicy(enabled=True)
    except RuntimeError:
        uncalibrated_enable_rejected = True
    else:
        raise AssertionError("rejection enabled without calibration")
    try:
        RejectionCalibration(
            schema_version=REJECTION_CALIBRATION_SCHEMA,
            source="validation",
            score_name=MAX_CLASS_PROBABILITY_SCORE,
            acceptance_threshold=0.60,
        )
    except ValueError:
        invalid_calibration_source_rejected = True
    else:
        raise AssertionError("rejection accepted validation-derived calibration")
    calibration = RejectionCalibration(
        schema_version=REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        score_name=MAX_CLASS_PROBABILITY_SCORE,
        acceptance_threshold=0.60,
    )
    calibrated_decision = RejectionPolicy(calibration, enabled=True).decide(probabilities)
    if not calibrated_decision.rejected or calibrated_decision.class_label_name != "UnknownTarget":
        raise AssertionError("reject action overwrote the five-class UnknownTarget label")

    logits = v7_model(**batch)["sequence_logits"]
    loss = task_prefix_loss(
        logits,
        torch.tensor([0]),
        batch["tr_padding_mask"],
        class_weights=torch.tensor(v7_gate.train.class_weights),
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("V7 focused loss is non-finite")

    result = {
        "status": "passed",
        "config_sha256": config_hashes,
        "v7_single_factor_contract": "dt_valid_feature_mode only, excluding versioned output identity",
        "v8_single_factor_contract": "checkpoint_selection_metric only, excluding versioned output identity",
        "v6_legacy_mode": v6_gate.model.dt_valid_feature_mode,
        "v6_fusion_input_dim": v6_model.fusion_input_dim,
        "v6_checkpoint_compatible": True,
        "v7_mode": v7_gate.model.dt_valid_feature_mode,
        "v7_fusion_input_dim": v7_model.fusion_input_dim,
        "direct_fusion_max_difference_when_only_dt_valid_changes": float(
            (captured[0] - captured[1]).abs().max().item()
        ),
        "liquid_dynamics_max_difference_for_validity_change": dynamics_difference,
        "batch_online_max_difference": batch_online_difference,
        "v6_v7_checkpoint_rejected": v6_v7_checkpoint_rejected,
        "v7_v8_checkpoint_rejected": v7_v8_checkpoint_rejected,
        "mixed_v7_v8_cascade_rejected": mixed_v7_v8_rejected,
        "weak_class_selection_primary": stronger_floor["primary_value"],
        "weak_class_selection_tie_break": stronger_floor["tie_break_value"],
        "higher_macro_weaker_floor_rejected": True,
        "zero_support_selection_rejected": zero_support_rejected,
        "uncalibrated_rejection_state": default_decision.decision_state,
        "uncalibrated_enable_rejected": uncalibrated_enable_rejected,
        "invalid_calibration_source_rejected": invalid_calibration_source_rejected,
        "unknown_class_label_preserved_when_rejected": calibrated_decision.class_label_name,
        "focused_loss": float(loss.item()),
        "formal_training": "未执行",
        "validation": "未执行",
        "test": "未执行",
        "oof_calibration": "未执行；测试阈值仅为接口单元测试常量，不是正式阈值",
    }
    path = project_root / "src-v3" / "smoke" / "RADAR-LN3-V7-V8-repair-contract-result.json"
    result["result_path"] = str(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the isolated RADAR-LN3 V7/V8 repair contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
