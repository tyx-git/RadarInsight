from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from radar_ln3.angular_knownness_oof import (
    NESTED_OOF_FORWARD_CLAIM_SCHEMA,
    claim_outer_held_out_forward_once,
    complete_outer_held_out_forward_claim,
    load_nested_oof_contract,
    nested_config_identities,
    nested_fold_record,
    nested_prediction_path,
    validate_completed_outer_held_out_forward_claim,
    verify_nested_dataset,
)
from radar_ln3.cascade import (
    EXPECTED_TASKS,
    OnlineLiquidSoftCascade,
    build_cascade,
    build_cascade_dataset,
    validate_cascade_configs,
)
from radar_ln3.data import collate_point_sequences
from radar_ln3.losses import strict_open_set_final_state_nll
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_COSINE_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    V15_CONTROL_REJECTION_CALIBRATION_SCHEMA,
    V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA,
    StrictOpenSetRejectionCalibration,
    StrictOpenSetRejectionPolicy,
    validate_strict_open_set_calibration_model_binding,
)
from radar_ln3.strict_open_set_oof import predict_oof_parent_final, sha256_file
from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    UNKNOWN_BASE_LABEL,
    load_task_config,
)
from train.cascade import cascade_forward_batch, cascade_training_objective, load_hierarchical_configs


TASK_SLUGS = {
    "dbb_cu_gate": "dbb-cu-gate",
    "dbb_specialist": "dbb-specialist",
    "cu_specialist": "cu-specialist",
}


def _expect_failure(call: Callable[[], Any], contains: str) -> str:
    try:
        call()
    except (ValueError, RuntimeError, PermissionError, FileExistsError) as error:
        message = str(error)
        if contains.lower() not in message.lower():
            raise AssertionError(f"failure did not mention {contains!r}: {message}") from error
        return message
    raise AssertionError(f"expected failure containing {contains!r}")


def _config_paths(project_root: Path, arm: str) -> dict[str, Path]:
    return {
        name: project_root
        / "config"
        / "RADAR-LN3"
        / (
            "RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-"
            f"{arm}-{slug}-config.json"
        )
        for name, slug in TASK_SLUGS.items()
    }


def _calibration(*, arm: str) -> StrictOpenSetRejectionCalibration:
    treatment = arm == "treatment"
    return StrictOpenSetRejectionCalibration(
        schema_version=(
            V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA
            if treatment
            else V15_CONTROL_REJECTION_CALIBRATION_SCHEMA
        ),
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        dbb_known_score_name=(
            DBB_KNOWN_COSINE_SCORE if treatment else DBB_KNOWN_PROBABILITY_SCORE
        ),
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=0.60,
        gate_accept_dbb_margin=0.10,
        dbb_accept_known_threshold=-0.25 if treatment else 0.60,
        dbb_accept_known_margin=0.10,
        cu_accept_clutter_threshold=0.60,
        cu_accept_clutter_margin=0.10,
        calibration_quality_status="deployment_eligible",
        deployment_eligible=True,
        model_schema_version=ANGULAR_KNOWNNESS_SCHEMA_VERSION,
        paired_arm=arm,
        min_valid_points=1,
        min_track_quality=0.0,
        max_gap_fraction=1.0,
    )


def _margin_contract() -> dict[str, Any]:
    torch.manual_seed(20260811)
    gate = torch.randn(2, 4, 2)
    dbb = torch.randn(2, 4, 3, requires_grad=True)
    cu = torch.randn(2, 4, 2)
    labels = torch.tensor([1, 3], dtype=torch.long)
    padding = torch.tensor(
        [[False, False, False, False], [False, False, False, True]], dtype=torch.bool
    )
    base = strict_open_set_final_state_nll(
        gate,
        dbb,
        cu,
        labels,
        padding,
        prefix_auxiliary_weight=0.10,
        prefix_auxiliary_min_events=3,
        dbb_angular_margin=0.0,
        dbb_cosine_scale=16.0,
    )
    treatment = strict_open_set_final_state_nll(
        gate,
        dbb,
        cu,
        labels,
        padding,
        prefix_auxiliary_weight=0.10,
        prefix_auxiliary_min_events=3,
        dbb_angular_margin=0.20,
        dbb_cosine_scale=16.0,
    )
    if float(treatment["loss"].item()) == float(base["loss"].item()):
        raise AssertionError("V15 angular margin did not change the DBB training objective")
    if abs(float(treatment["dbb_angular_margin"].item()) - 0.20) > 1.0e-6:
        raise AssertionError("V15 angular margin was not retained in the objective record")
    return {
        "base_loss": float(base["loss"].item()),
        "margin_loss": float(treatment["loss"].item()),
        "margin_changes_training_loss": True,
    }


def _initialized_pair_contract(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control = load_hierarchical_configs(_config_paths(project_root, "control"))
    treatment = load_hierarchical_configs(_config_paths(project_root, "treatment"))
    validate_cascade_configs(control)
    validate_cascade_configs(treatment)
    control_model = build_cascade(control, device=torch.device("cpu")).eval()
    treatment_model = build_cascade(treatment, device=torch.device("cpu")).eval()
    exact_modules = ("gate", "cu_specialist")
    for name in exact_modules:
        control_state = getattr(control_model, name).state_dict()
        treatment_state = getattr(treatment_model, name).state_dict()
        if set(control_state) != set(treatment_state) or any(
            not torch.equal(control_state[key], treatment_state[key]) for key in control_state
        ):
            raise AssertionError(f"V15 paired initialization changed non-intervention module {name}")
    control_state = control_model.dbb_specialist.state_dict()
    treatment_state = treatment_model.dbb_specialist.state_dict()
    shared_names = sorted(set(control_state) & set(treatment_state))
    unequal = [
        name
        for name in shared_names
        if control_state[name].shape == treatment_state[name].shape
        and not torch.equal(control_state[name], treatment_state[name])
    ]
    if unequal:
        raise AssertionError(
            "V15 DBB shared encoder/fusion/liquid initialization differs across paired arms: "
            f"{unequal[:3]}"
        )
    return control, treatment, {
        "gate_exact": True,
        "cu_exact": True,
        "dbb_common_shape_parameters_exact": True,
        "dbb_control_readout": control["dbb_specialist"].model.dbb_readout_mode,
        "dbb_treatment_readout": treatment["dbb_specialist"].model.dbb_readout_mode,
    }


def _real_forward_contract(
    project_root: Path,
    configs: dict[str, Any],
) -> dict[str, Any]:
    dataset = build_cascade_dataset(
        project_root,
        configs["dbb_cu_gate"],
        split_name=configs["dbb_cu_gate"].data.train_split_name,
        label_scope="known_only",
    )
    if UNKNOWN_BASE_LABEL in set(dataset.frame["label_name"].astype(str)):
        raise AssertionError("V15 known-only Dataset exposes UnknownTarget")
    dbb_index = next(
        index
        for index, parent_id in enumerate(dataset.parent_ids)
        if str(dataset._groups[str(parent_id)].iloc[0]["label_name"])
        in {"DroneTarget", "BirdTarget", "BalloonTarget"}
    )
    item = dataset[dbb_index]
    batch = collate_point_sequences([item])
    tensors = {
        key: value.to(torch.device("cpu")) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = build_cascade(configs, device=torch.device("cpu")).eval()
    with torch.inference_mode():
        offline = cascade_forward_batch(model, tensors)
    dbb_logits = offline["dbb_sequence_logits"]
    knownness = offline["dbb_knownness_sequence_scores"]
    if knownness.shape != dbb_logits.shape or not torch.isfinite(knownness).all():
        raise AssertionError("V15 DBB knownness output has an invalid shape or values")
    if not torch.equal(dbb_logits.argmax(dim=-1), knownness.argmax(dim=-1)):
        raise AssertionError("V15 inference logits and knownness score top class diverged")
    before = {key: value.detach().clone() for key, value in offline.items() if isinstance(value, torch.Tensor)}
    _ = cascade_training_objective(offline, tensors, configs)
    after = cascade_forward_batch(model, tensors)
    inference_keys = (
        "dbb_sequence_logits",
        "dbb_knownness_sequence_scores",
        "sequence_open_set_evidence",
    )
    maximum_margin_pollution = max(
        float((before[key] - after[key]).abs().max().item()) for key in inference_keys
    )
    if maximum_margin_pollution != 0.0:
        raise AssertionError("V15 training margin mutated label-free inference outputs")
    online = OnlineLiquidSoftCascade(model, confirm_count=2)
    online_evidence: list[torch.Tensor] = []
    for event_index in range(int(batch["lengths"][0].item())):
        step = online.step(
            tr_features=tensors["tr_features"][:, event_index],
            tr_missing_mask=tensors["tr_missing_mask"][:, event_index],
            tr_dt_seconds=tensors["tr_dt_seconds"][:, event_index],
            tr_dt_valid=tensors["tr_dt_valid"][:, event_index],
            tr_dt_reason=tensors["tr_dt_reason"][:, event_index],
            rd_frames=tensors["rd_frames"][:, event_index],
            rd_frame_mask=tensors["rd_frame_mask"][:, event_index],
            rd_quality=tensors["rd_quality"][:, event_index],
        )
        if step.open_set_audit_evidence is None:
            raise AssertionError("V15 online output omitted strict-open-set audit evidence")
        online_evidence.append(step.open_set_audit_evidence)
    # Reproduce the DBB stateful path to compare the exact score consumed by
    # the online cascade with the offline causal-prefix score.
    specialist_state = None
    specialist_scores: list[torch.Tensor] = []
    for event_index in range(int(batch["lengths"][0].item())):
        point = model.dbb_specialist.forward_online_point(
            tr_features=tensors["tr_features"][:, event_index],
            tr_missing_mask=tensors["tr_missing_mask"][:, event_index],
            tr_dt_seconds=tensors["tr_dt_seconds"][:, event_index],
            tr_dt_valid=tensors["tr_dt_valid"][:, event_index],
            tr_dt_reason=tensors["tr_dt_reason"][:, event_index],
            rd_frames=tensors["rd_frames"][:, event_index],
            rd_frame_mask=tensors["rd_frame_mask"][:, event_index],
            rd_quality=tensors["rd_quality"][:, event_index],
            liquid_state=specialist_state,
        )
        specialist_state = point["liquid_state"]
        specialist_scores.append(point["dbb_knownness_window_scores"])
    online_score_tensor = torch.stack(specialist_scores, dim=1)
    offline_evidence = offline["sequence_open_set_evidence"]
    online_evidence_tensor = torch.stack(online_evidence, dim=1)
    score_difference = float(
        (offline["dbb_knownness_sequence_scores"] - online_score_tensor).abs().max().item()
    )
    evidence_difference = float((offline_evidence - online_evidence_tensor).abs().max().item())
    if score_difference > 1.0e-5 or evidence_difference > 1.0e-5:
        raise AssertionError(
            f"V15 offline/online knownness or evidence mismatch: score={score_difference}, evidence={evidence_difference}"
        )
    return {
        "known_only_parent_count": len(dataset),
        "unknown_training_exposure": 0,
        "dbb_knownness_score_name": offline["dbb_knownness_score_name"],
        "inference_margin_pollution_max_difference": maximum_margin_pollution,
        "offline_online_knownness_max_difference": score_difference,
        "offline_online_evidence_max_difference": evidence_difference,
    }


def _calibration_binding_contract(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    control_calibration = _calibration(arm="control")
    treatment_calibration = _calibration(arm="treatment")
    validate_strict_open_set_calibration_model_binding(
        control_calibration,
        model_schema_version=ANGULAR_KNOWNNESS_SCHEMA_VERSION,
        paired_arm="control",
        dbb_known_score_name=control["dbb_specialist"].model.dbb_knownness_score_name,
    )
    validate_strict_open_set_calibration_model_binding(
        treatment_calibration,
        model_schema_version=ANGULAR_KNOWNNESS_SCHEMA_VERSION,
        paired_arm="treatment",
        dbb_known_score_name=treatment["dbb_specialist"].model.dbb_knownness_score_name,
    )
    arm_error = _expect_failure(
        lambda: validate_strict_open_set_calibration_model_binding(
            control_calibration,
            model_schema_version=ANGULAR_KNOWNNESS_SCHEMA_VERSION,
            paired_arm="treatment",
            dbb_known_score_name=treatment["dbb_specialist"].model.dbb_knownness_score_name,
        ),
        "does not match",
    )
    score_error = _expect_failure(
        lambda: StrictOpenSetRejectionPolicy(treatment_calibration, enabled=True).decide(
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.1, 0.8, 0.1]),
            torch.tensor([0.8, 0.2]),
            dbb_knownness_scores=None,
            valid_point_count=3,
            track_quality=1.0,
            gap_fraction=0.0,
        ),
        "cosine",
    )
    return {
        "control_binding_passed": True,
        "treatment_binding_passed": True,
        "cross_arm_calibration_rejected": True,
        "missing_cosine_score_rejected": True,
        "cross_arm_error": arm_error,
        "missing_cosine_error": score_error,
    }


def _nested_contract(project_root: Path, configs: dict[str, Any]) -> dict[str, Any]:
    contract = load_nested_oof_contract(project_root)
    fold = nested_fold_record(contract, 0)
    expected_outer = list(fold["outer_held_out_parent_ids"])
    dataset = build_cascade_dataset(
        project_root,
        configs["dbb_cu_gate"],
        split_name="train",
        label_scope="known_only",
        parent_ids=expected_outer,
    )
    scope = verify_nested_dataset(
        dataset,
        expected_parent_ids=expected_outer,
        scope="smoke_outer_held_out",
    )
    contaminated = copy.deepcopy(dataset)
    contaminated.frame = contaminated.frame.copy()
    contaminated.frame.loc[contaminated.frame.index[0], "label_name"] = UNKNOWN_BASE_LABEL
    unknown_error = _expect_failure(
        lambda: verify_nested_dataset(
            contaminated,
            expected_parent_ids=expected_outer,
            scope="smoke_contaminated_unknown",
        ),
        "known train-parent",
    )
    wrong_ids = expected_outer[:-1]
    held_out_error = _expect_failure(
        lambda: verify_nested_dataset(
            dataset,
            expected_parent_ids=wrong_ids,
            scope="smoke_wrong_held_out",
        ),
        "violates",
    )
    outer_identities = nested_config_identities(configs)
    score_name = configs["dbb_specialist"].model.dbb_knownness_score_name
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v15-forward-claim-") as temporary:
        root = Path(temporary)
        fake_project = root
        claim = claim_outer_held_out_forward_once(
            fake_project,
            contract=contract,
            arm="control",
            fold=fold,
            outer_config_identities=outer_identities,
            dbb_knownness_score_name=score_name,
        )
        claimed = json.loads(claim.read_text(encoding="utf-8"))
        if claimed.get("schema_version") != NESTED_OOF_FORWARD_CLAIM_SCHEMA:
            raise AssertionError("V15 outer-held-out forward claim schema is invalid")
        duplicate_error = _expect_failure(
            lambda: claim_outer_held_out_forward_once(
                fake_project,
                contract=contract,
                arm="control",
                fold=fold,
                outer_config_identities=outer_identities,
                dbb_knownness_score_name=score_name,
            ),
            "already claimed",
        )
        prediction_path = nested_prediction_path(fake_project, "control", 0)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("{}\n", encoding="utf-8")
        completed = complete_outer_held_out_forward_claim(claim, prediction_path=prediction_path)
        completed_claim = validate_completed_outer_held_out_forward_claim(
            fake_project,
            contract=contract,
            arm="control",
            fold=fold,
            outer_config_identities=outer_identities,
            dbb_knownness_score_name=score_name,
            prediction_path=prediction_path,
        )
        if completed != completed_claim:
            raise AssertionError("V15 completed outer-forward claim does not round trip")
    return {
        "known_parent_count": scope["parent_count"],
        "unknown_training_exposure": 0,
        "unknown_dataset_exposure_rejected": True,
        "held_out_partition_mismatch_rejected": True,
        "duplicate_outer_held_out_forward_rejected": True,
        "unknown_error": unknown_error,
        "held_out_error": held_out_error,
        "duplicate_error": duplicate_error,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    margin = _margin_contract()
    control, treatment, initialized = _initialized_pair_contract(root)
    forward = _real_forward_contract(root, treatment)
    calibration = _calibration_binding_contract(control, treatment)
    nested = _nested_contract(root, control)
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V15-angular-knownness-focused-contract-v1",
        "margin": margin,
        "paired_initialization": initialized,
        "treatment_forward": forward,
        "calibration_binding": calibration,
        "nested_oof": nested,
        "formal_training": "未执行",
        "nested_oof_training": "未执行",
        "threshold_fitting": "未执行",
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    result_path = root / "src-v3" / "smoke" / "RADAR-LN3-V15-angular-knownness-focused-contract-result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V15 angular-knownness focused contract tests.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
