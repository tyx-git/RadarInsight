"""Focused isolation/forward smoke checks for V26c's L1/L2 OOF protocol.

This check does not train a fold or score a held-out parent.  It validates the
prepared partitions, cache populations, runtime bindings, and first train-batch
forwards before the ten full twenty-epoch CUDA jobs are launched.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    build_known_gate_dataloader,
    build_known_gate_model_for_config,
    build_known_gate_oof_dataset,
)
from radar_ln3.known_gate_v26c_oof import (
    CONTRACT,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    L1_EPOCH,
    L2_EPOCH,
    READINESS,
    READINESS_SCHEMA,
    V21E_ASSIGNMENT,
    fold_record,
    l1_config_path,
    l1_held_out_feature_cache_path,
    l1_train_feature_cache_path,
    l2_config_path,
    l2_held_out_micro_cache_path,
    l2_held_out_mixed_micro_cache_path,
    l2_held_out_mixed_physical_cache_path,
    l2_held_out_physical_cache_path,
    l2_train_micro_cache_path,
    l2_train_physical_cache_path,
    relative_path,
    sequence_sha256,
    sha256_file,
)
from radar_ln3.l1_l2_raw_routing import (
    FrozenL2MixedAuditDataset,
    build_frozen_l2_mixed_audit_dataloader,
    l2_model_inputs as mixed_l2_model_inputs,
    verify_l1_l2_event_alignment,
)
from report.known_gate_v26c_oof import (
    _actions as apply_v26c_actions,
    _l1_candidate_metrics,
)
from radar_ln3.task_models import (
    build_paired_model_for_config,
    build_task_dataloader,
    build_task_dataset,
)
from train.known_gate_v21 import _model_inputs as l1_model_inputs
from train.known_gate_v26c_oof import (
    L1_LABELS,
    L2_LABELS,
    _dataset_scope,
    _runtime_l1_config,
    _runtime_l2_config,
    _source_protocols,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26c smoke expected JSON object: {path}")
    return value


def _cache_ids(path: Path) -> list[str]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise AssertionError(f"V26c cache fields differ: {path}")
        sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
        values = archive["values"]
        valid = archive["valid"]
    if (
        len(sample_ids) != len(set(sample_ids))
        or values.shape[0] != len(sample_ids)
        or valid.shape[0] != len(sample_ids)
    ):
        raise AssertionError(f"V26c cache population is malformed: {path}")
    return sample_ids


def _known_parent_assignment(root: Path) -> dict[str, dict[str, Any]]:
    assignment = _read(root / V21E_ASSIGNMENT)
    rows = {str(row["parent_sample_id"]): dict(row) for row in assignment["assignments"]}
    if (
        assignment.get("grouping") != "capture_content_provenance_component"
        or int(assignment.get("known_parent_count", -1)) != 986
        or int(assignment.get("unknown_training_exposure", -1)) != 0
        or len(rows) != 986
    ):
        raise AssertionError("V26c assignment scope is invalid")
    return rows


def _first_l1_forward(root: Path, fold: Mapping[str, Any]) -> dict[str, Any]:
    config, _, _ = _runtime_l1_config(root, int(fold["fold_id"]), dict(fold["l1"]))
    dataset = build_known_gate_oof_dataset(
        root,
        config,
        parent_ids=fold["train_parent_ids"],
        feature_cache_path=root / l1_train_feature_cache_path(int(fold["fold_id"])),
    )
    batch = next(
        iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False, pin_memory=False))
    )
    source, _, _, _ = _source_protocols(root)
    model, reference, _ = build_known_gate_model_for_config(root, source)
    if reference is not None:
        del reference
    model.config = config
    model.eval()
    with torch.inference_mode():
        output = model(**l1_model_inputs(batch))
    logits = output["sequence_logits"]
    if logits.shape[:2] != batch["tr_features"].shape[:2] or logits.shape[-1] != 2 or not torch.isfinite(logits).all():
        raise AssertionError("V26c L1 first train batch forward is invalid")
    return {
        "parent_ids": [str(value) for value in batch["parent_sample_ids"]],
        "logits_shape": list(logits.shape),
        "finite": True,
        "class_names": list(KNOWN_GATE_CLASS_NAMES),
    }


def _first_l2_forward(root: Path, fold: Mapping[str, Any]) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    config, _, _ = _runtime_l2_config(root, fold_id, dict(fold["l2"]))
    dataset = build_task_dataset(
        root,
        config,
        split_name="train",
        parent_ids=fold["l2"]["train_parent_ids"],
        rd_physical_cache_path_override=root / l2_train_physical_cache_path(fold_id),
        micro_doppler_cache_path_override=root / l2_train_micro_cache_path(fold_id),
    )
    batch = next(iter(build_task_dataloader(dataset, batch_size=2, shuffle=False)))
    _, source, _, _ = _source_protocols(root)
    model, reference, _ = build_paired_model_for_config(root, source)
    if reference is not None:
        del reference
    model.config = config
    model.eval()
    with torch.inference_mode():
        output = model(**mixed_l2_model_inputs(batch))
    logits = output["sequence_logits"]
    if logits.shape[:2] != batch["tr_features"].shape[:2] or logits.shape[-1] != 3 or not torch.isfinite(logits).all():
        raise AssertionError("V26c L2 first train batch forward is invalid")
    return {
        "parent_ids": [str(value) for value in batch["parent_sample_ids"]],
        "logits_shape": list(logits.shape),
        "finite": True,
        "class_names": list(model.class_names),
    }


def _mixed_alignment(root: Path, fold: Mapping[str, Any]) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    l1_config, _, _ = _runtime_l1_config(root, fold_id, dict(fold["l1"]))
    l2_config, _, _ = _runtime_l2_config(root, fold_id, dict(fold["l2"]))
    l1_dataset = build_known_gate_oof_dataset(
        root,
        l1_config,
        parent_ids=fold["held_out_parent_ids"],
        feature_cache_path=root / l1_held_out_feature_cache_path(fold_id),
    )
    mixed = FrozenL2MixedAuditDataset(
        root,
        l2_config,
        split_name="train",
        rd_physical_cache_path=root / l2_held_out_mixed_physical_cache_path(fold_id),
        micro_doppler_cache_path=root / l2_held_out_mixed_micro_cache_path(fold_id),
        parent_ids=tuple(fold["held_out_parent_ids"]),
    )
    alignment = verify_l1_l2_event_alignment(l1_dataset.base, mixed.base)
    if sorted(mixed.parent_ids) != fold["held_out_parent_ids"]:
        raise AssertionError("V26c mixed L2 held-out parent scope differs")
    labels = mixed.label_counts()
    if labels["UnknownTarget"] != 0 or labels["ClutterTarget"] != int(fold["l1"]["held_out_label_parent_counts"]["ClutterTarget"]):
        raise AssertionError("V26c mixed L2 inference scope contains Unknown or lacks held-out Clutter")
    batch = next(iter(build_frozen_l2_mixed_audit_dataloader(mixed, batch_size=1)))
    if str(batch["label_names"][0]) not in L1_LABELS:
        raise AssertionError("V26c mixed L2 batch contains a forbidden label")
    return {"alignment": alignment, "label_counts": labels, "first_parent": str(batch["parent_sample_ids"][0])}


def _clutter_l2_audit_is_not_action_evidence() -> dict[str, Any]:
    """Regression test for the inference-only Clutter L2 route.

    L2 is materialized for a held-out Clutter parent only to reproduce the
    call path.  Even an artificially confident DBB argmax must not override
    a valid L1 Clutter acceptance.
    """

    row = {
        "label_name": "ClutterTarget",
        "l1_known_target_probability": 0.01,
        "l1_known_clutter_probability": 0.99,
        "l2_top_subclass": "DroneTarget",
        "l2_subclass_probability": 0.99,
        "l2_inference_only_for_clutter": True,
        "l2_training_exposure_for_this_parent": 0,
        "product_action_eligible": True,
        "window_length": 5,
        "event_index": 4,
        "fold_id": 0,
        "parent_sample_id": "synthetic-clutter",
    }
    result = apply_v26c_actions(
        [row],
        {
            "l1_known_target_acceptance": 0.80,
            "l1_known_clutter_acceptance": 0.80,
            "l2_subclass_acceptance": 0.80,
        },
    )[0]
    if (
        result["semantic_action"] != "ClutterTarget"
        or result["dbb_subclass_accepted"] is not False
        or result["l2_subclass_evidence_used_for_action"] is not False
    ):
        raise AssertionError("V26c Clutter audit-only L2 evidence leaked into action resolution")
    return {
        "semantic_action": result["semantic_action"],
        "dbb_subclass_evidence_used_for_action": result[
            "l2_subclass_evidence_used_for_action"
        ],
    }


def _l1_background_preempts_unrouted_l2_evidence() -> dict[str, Any]:
    """Exercise the L1-background/L2-high boundary and grid equivalence.

    L2 is deliberately evaluated for audit alignment before L3 acts.  A high
    DBB score must not convert an L1 KnownClutter acceptance into Pending just
    because the same parent is non-Clutter in ground truth.  The candidate-grid
    counters used for threshold selection must therefore agree with the actual
    semantic actions at this boundary.
    """

    thresholds = {
        "l1_known_target_acceptance": 0.80,
        "l1_known_clutter_acceptance": 0.80,
        "l2_subclass_acceptance": 0.80,
    }
    rows = [
        {
            "label_name": "DroneTarget",
            "l1_known_target_probability": 0.10,
            "l1_known_clutter_probability": 0.90,
            "l2_top_subclass": "DroneTarget",
            "l2_subclass_probability": 0.99,
            "l2_inference_only_for_clutter": False,
            "l2_training_exposure_for_this_parent": 1,
            "product_action_eligible": True,
            "window_length": 5,
            "event_index": 4,
            "fold_id": 0,
            "parent_sample_id": "synthetic-known-routed-clutter",
        },
        {
            "label_name": "ClutterTarget",
            "l1_known_target_probability": 0.10,
            "l1_known_clutter_probability": 0.90,
            "l2_top_subclass": "DroneTarget",
            "l2_subclass_probability": 0.99,
            "l2_inference_only_for_clutter": True,
            "l2_training_exposure_for_this_parent": 0,
            "product_action_eligible": True,
            "window_length": 5,
            "event_index": 4,
            "fold_id": 0,
            "parent_sample_id": "synthetic-clutter-routed-clutter",
        },
        {
            "label_name": "BirdTarget",
            "l1_known_target_probability": 0.90,
            "l1_known_clutter_probability": 0.10,
            "l2_top_subclass": "BirdTarget",
            "l2_subclass_probability": 0.99,
            "l2_inference_only_for_clutter": False,
            "l2_training_exposure_for_this_parent": 1,
            "product_action_eligible": True,
            "window_length": 5,
            "event_index": 4,
            "fold_id": 0,
            "parent_sample_id": "synthetic-known-routed-target",
        },
        {
            "label_name": "BalloonTarget",
            "l1_known_target_probability": 0.60,
            "l1_known_clutter_probability": 0.40,
            "l2_top_subclass": "BalloonTarget",
            "l2_subclass_probability": 0.99,
            "l2_inference_only_for_clutter": False,
            "l2_training_exposure_for_this_parent": 1,
            "product_action_eligible": True,
            "window_length": 5,
            "event_index": 4,
            "fold_id": 0,
            "parent_sample_id": "synthetic-known-rejected-by-l1",
        },
    ]
    actions = apply_v26c_actions(rows, thresholds)
    boundary = actions[0]
    if (
        boundary["semantic_action"] != "ClutterTarget"
        or boundary["semantic_reason"] != "known_clutter_evidence_accepted"
        or boundary["dbb_subclass_accepted"] is not False
        or boundary["l2_subclass_evidence_used_for_action"] is not False
    ):
        raise AssertionError("V26c L1 KnownClutter acceptance did not preempt unrouted L2 evidence")

    import numpy as np

    labels = np.asarray([str(row["label_name"]) for row in rows], dtype=object)
    target_scores = np.asarray(
        [float(row["l1_known_target_probability"]) for row in rows], dtype=np.float64
    )
    clutter_scores = np.asarray(
        [float(row["l1_known_clutter_probability"]) for row in rows], dtype=np.float64
    )
    l2_accepted = np.asarray(
        [
            bool(not row["l2_inference_only_for_clutter"])
            and float(row["l2_subclass_probability"]) >= thresholds["l2_subclass_acceptance"]
            for row in rows
        ],
        dtype=np.bool_,
    )
    l2_correct = np.asarray(
        [
            bool(accepted) and str(row["l2_top_subclass"]) == str(row["label_name"])
            for row, accepted in zip(rows, l2_accepted, strict=True)
        ],
        dtype=np.bool_,
    )
    candidate = _l1_candidate_metrics(
        labels,
        target_scores,
        clutter_scores,
        target_thresholds=np.asarray([thresholds["l1_known_target_acceptance"]]),
        clutter_thresholds=np.asarray([thresholds["l1_known_clutter_acceptance"]]),
        l2_subclass_accepted=l2_accepted,
        l2_correct_subclass_accepted=l2_correct,
    )
    actual_clutter = [row for row in actions if row["semantic_action"] == "ClutterTarget"]
    actual_known_to_clutter = [
        row
        for row in actions
        if row["label_name"] != "ClutterTarget" and row["semantic_action"] == "ClutterTarget"
    ]
    actual_dbb = [row for row in actions if row["semantic_action"] in {"DroneTarget", "BirdTarget", "BalloonTarget"}]
    actual_correct_dbb = [
        row for row in actual_dbb if row["semantic_action"] == row["label_name"]
    ]
    expected = {
        "clutter_tp": sum(row["label_name"] == "ClutterTarget" for row in actual_clutter),
        "clutter_predicted": len(actual_clutter),
        "known_to_clutter": len(actual_known_to_clutter),
        "clutter_to_target_route": sum(
            row["label_name"] == "ClutterTarget" and row["known_target_accepted"]
            for row in actions
        ),
        "target_final_action": len(actual_dbb),
        "target_correct_action": len(actual_correct_dbb),
    }
    observed = {name: int(candidate[name][0, 0]) for name in expected}
    if observed != expected:
        raise AssertionError(
            "V26c L1 candidate-grid counters diverge from semantic actions: "
            f"observed={observed}; expected={expected}"
        )
    return {
        "boundary_action": boundary["semantic_action"],
        "boundary_l2_evidence_used": boundary["l2_subclass_evidence_used_for_action"],
        "candidate_action_counter_equivalence": True,
    }


def _l2_task_directory_and_dataset_lifecycle_are_isolated() -> dict[str, bool]:
    """Static guard for the two execution-boundary regressions fixed in V26c."""

    src_root = Path(__file__).resolve().parents[1]
    pipeline = (src_root / "scripts" / "v26c_known_gate_l1_l2_oof_pipeline.ps1").read_text(
        encoding="utf-8"
    )
    trainer = (src_root / "train" / "known_gate_v26c_oof.py").read_text(
        encoding="utf-8"
    )
    invoke_start = pipeline.index("function Invoke-FoldTraining")
    invoke_end = pipeline.index("\nfunction Invoke-TrainingStage", invoke_start)
    invoke = pipeline[invoke_start:invoke_end]
    l2_start = trainer.index("def _train_l2(")
    l2_end = trainer.index("\ndef train(", l2_start)
    l2 = trainer[l2_start:l2_end]
    l2_after_dataset_release = l2.split("del held_out_dataset", 1)[1]
    lifecycle = {
        "task_specific_directory_guard": "$taskDirectory = Join-Path $foldDirectory $Task" in invoke,
        "no_fold_root_sibling_guard": "Get-ChildItem -LiteralPath $foldDirectory -Force" not in invoke,
        "held_out_count_saved_before_release": l2.index("held_out_parent_count = len(held_out_dataset)") < l2.index("del held_out_dataset"),
        "released_dataset_not_used_after_release": re.search(
            r"(?<![A-Za-z0-9_])held_out_dataset(?![A-Za-z0-9_])",
            l2_after_dataset_release,
        )
        is None,
        "saved_held_out_count_logged": "held_out_parent_count)" in l2,
    }
    if not all(lifecycle.values()):
        raise AssertionError("V26c task-directory or held-out Dataset lifecycle guard regressed")
    return lifecycle


def _single_held_out_forward_protocol() -> dict[str, Any]:
    """Guard the V26c trainer against reintroducing duplicate held-out calls.

    The executable L2 path must join a sealed L1 prediction to one mixed L2
    held-out pass.  This static guard deliberately rejects both the historical
    independent ``evaluate_task`` pass and any direct L1 model forward in the
    joint-score helper before expensive fold jobs are launched.
    """

    trainer_path = Path(__file__).resolve().parents[1] / "train" / "known_gate_v26c_oof.py"
    text = trainer_path.read_text(encoding="utf-8")
    start = text.index("def _l2_evaluation_and_joint_scores(")
    end = text.index("\ndef _load_l1_for_joint_scores(", start)
    helper = text[start:end]
    train_l2_start = text.index("def _train_l2(")
    train_l2_end = text.index("\ndef train(", train_l2_start)
    train_l2 = text[train_l2_start:train_l2_end]
    forbidden = {
        "independent_l2_evaluate_task": "evaluate_task(" not in train_l2,
        "no_l1_model_forward_in_l2_helper": "l1_model(" not in helper,
        "single_l2_forward_site": helper.count("l2_model(**mixed_l2_model_inputs(tensors))") == 1,
        "sealed_l1_reuse_declared": "l1_scores_reused_from_sealed_prediction" in helper,
        "l2_forward_count_declared": '"l2_held_out_forward_count": 1' in helper,
    }
    if not all(forbidden.values()):
        raise AssertionError("V26c single held-out forward protocol is not structurally enforced")
    return forbidden


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract, readiness = _read(contract_path), _read(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise AssertionError("V26c prepared contract/readiness is invalid")
    assignment = _known_parent_assignment(root)
    folds: list[dict[str, Any]] = []
    held_seen: list[str] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        train_ids, held_ids = fold["train_parent_ids"], fold["held_out_parent_ids"]
        train_components = {assignment[parent]["provenance_component_id"] for parent in train_ids}
        held_components = {assignment[parent]["provenance_component_id"] for parent in held_ids}
        if train_components & held_components:
            raise AssertionError(f"V26c fold {fold_id} crosses a provenance component")
        if (
            set(train_ids) & set(held_ids)
            or len(train_ids) + len(held_ids) != 986
            or any(assignment[parent]["label_name"] == "UnknownTarget" for parent in (*train_ids, *held_ids))
        ):
            raise AssertionError(f"V26c fold {fold_id} parent isolation is invalid")
        # Cache membership must be disjoint at every training boundary.
        l1_train_cache = _cache_ids(root / l1_train_feature_cache_path(fold_id))
        l1_held_cache = _cache_ids(root / l1_held_out_feature_cache_path(fold_id))
        l2_train_phys = _cache_ids(root / l2_train_physical_cache_path(fold_id))
        l2_held_phys = _cache_ids(root / l2_held_out_physical_cache_path(fold_id))
        l2_train_micro = _cache_ids(root / l2_train_micro_cache_path(fold_id))
        l2_held_micro = _cache_ids(root / l2_held_out_micro_cache_path(fold_id))
        l2_mixed_phys = _cache_ids(root / l2_held_out_mixed_physical_cache_path(fold_id))
        l2_mixed_micro = _cache_ids(root / l2_held_out_mixed_micro_cache_path(fold_id))
        if (
            set(l1_train_cache) & set(l1_held_cache)
            or set(l2_train_phys) & set(l2_held_phys)
            or set(l2_train_micro) & set(l2_held_micro)
            or l2_train_phys != l2_train_micro
            or l2_held_phys != l2_held_micro
            or l2_mixed_phys != l2_mixed_micro
            or not set(l2_held_phys).issubset(set(l2_mixed_phys))
            or set(l2_train_phys) & set(l2_mixed_phys)
        ):
            raise AssertionError(f"V26c fold {fold_id} cache isolation differs")
        expected_l2_train = sorted(fold["l2"]["train_parent_ids"])
        expected_l2_held = sorted(fold["l2"]["held_out_parent_ids"])
        if (
            not expected_l2_train
            or not expected_l2_held
            or set(expected_l2_train) & set(expected_l2_held)
            or any(assignment[parent]["label_name"] not in L2_LABELS for parent in (*expected_l2_train, *expected_l2_held))
        ):
            raise AssertionError(f"V26c fold {fold_id} L2 Known-only scope differs")
        folds.append(
            {
                "fold_id": fold_id,
                "l1_train_parent_count": len(train_ids),
                "l1_held_out_parent_count": len(held_ids),
                "l2_train_parent_count": len(expected_l2_train),
                "l2_held_out_parent_count": len(expected_l2_held),
                "component_overlap_count": 0,
                "l1_cache_event_overlap_count": 0,
                "l2_cache_event_overlap_count": 0,
                "mixed_l2_inference_only_cache_contains_held_out_dbb_events": True,
            }
        )
        held_seen.extend(held_ids)
    if sorted(held_seen) != sorted(assignment) or len(held_seen) != len(set(held_seen)):
        raise AssertionError("V26c L1 OOF held-out coverage does not contain each Known parent once")
    # Fold zero is sufficient to prove the explicit-cache dataset wiring and
    # mixed inference call signatures without launching any optimizer step.
    fold_zero = fold_record(contract, 0)
    forward = {
        "l1": _first_l1_forward(root, fold_zero),
        "l2": _first_l2_forward(root, fold_zero),
        "mixed_l1_l2": _mixed_alignment(root, fold_zero),
    }
    action_regression = _clutter_l2_audit_is_not_action_evidence()
    l1_route_regression = _l1_background_preempts_unrouted_l2_evidence()
    execution_boundary = _l2_task_directory_and_dataset_lifecycle_are_isolated()
    single_forward = _single_held_out_forward_protocol()
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V26c-l1-l2-known-only-oof-contract-smoke-v1",
        "contract_path": relative_path(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "fold_count": FOLD_COUNT,
        "known_parent_count": len(assignment),
        "provenance_component_count": len({row["provenance_component_id"] for row in assignment.values()}),
        "fixed_snapshots": {"l1_epoch": L1_EPOCH, "l2_epoch": L2_EPOCH, "full_training_epochs": 20},
        "fold_checks": folds,
        "first_train_batch_forwards": forward,
        "semantic_action_regressions": {
            "clutter_l2_audit_is_not_action_evidence": action_regression,
            "l1_background_preempts_unrouted_l2_evidence": l1_route_regression,
        },
        "execution_boundary_regressions": execution_boundary,
        "held_out_forward_protocol": single_forward,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_cache_selection_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "L3_calibration": "not_executed",
        "mixed_unknown_audit": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V26c strict L1/L2 OOF isolation.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
