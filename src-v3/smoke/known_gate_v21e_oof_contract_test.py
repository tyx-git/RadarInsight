from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from build.known_gate_v21e_oof import (
    _config_payload,
    _fold_data_contract,
    _parent_records,
    build_assignments,
)
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    BASE_CONTROL_CONFIG_RELATIVE,
    BASE_TREATMENT_CONFIG_RELATIVE,
    CORE_CONTRACT_RELATIVE,
    FOLD_COUNT,
    INDEX_RELATIVE,
    canonical_sha256,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    records = _parent_records(root / INDEX_RELATIVE)
    first = build_assignments(records)
    second = build_assignments(records)
    if canonical_sha256(first) != canonical_sha256(second):
        raise AssertionError("V21e OOF assignment is not deterministic")
    assignments = list(first["assignments"])
    component_folds: dict[str, set[int]] = {}
    for row in assignments:
        component_folds.setdefault(str(row["provenance_component_id"]), set()).add(
            int(row["fold_id"])
        )
    if (
        int(first["fold_count"]) != FOLD_COUNT
        or int(first["known_parent_count"]) != 986
        or int(first["provenance_component_count"]) != 975
        or any(len(values) != 1 for values in component_folds.values())
    ):
        raise AssertionError("V21e OOF assignment isolation failed")
    fold_zero = [row for row in assignments if int(row["fold_id"]) == 0]
    fold_train = [row for row in assignments if int(row["fold_id"]) != 0]
    fold_train_ids = sorted(str(row["parent_sample_id"]) for row in fold_train)
    fold_held_out_ids = sorted(str(row["parent_sample_id"]) for row in fold_zero)
    stats_path = root / "data/processed/RADAR-LN3-V21-cq08-known-only-train-normalization-stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    core_contract = json.loads((root / CORE_CONTRACT_RELATIVE).read_text(encoding="utf-8"))
    train_counts = {
        label: sum(str(row["label_name"]) == label for row in fold_train)
        for label in ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    }
    held_out_counts = {
        label: sum(str(row["label_name"]) == label for row in fold_zero)
        for label in ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    }
    fold_contract_probe = _fold_data_contract(
        core_contract,
        root=root,
        fold_id=0,
        train_ids=fold_train_ids,
        held_out_ids=fold_held_out_ids,
        train_counts=train_counts,
        held_out_counts=held_out_counts,
        stats_path=stats_path,
        stats=stats,
    )
    if (
        fold_contract_probe["oof_fold_contract"]["held_out_checkpoint_selection"]
        is not False
        or fold_contract_probe["oof_fold_contract"]["grouping"]
        != "capture_content_provenance_component"
    ):
        raise AssertionError("V21e OOF fold data contract probe failed")
    control_payload = json.loads(
        (root / BASE_CONTROL_CONFIG_RELATIVE).read_text(encoding="utf-8")
    )
    derived_control = _config_payload(
        control_payload,
        root=root,
        arm="control",
        fold_id=0,
        data_contract_path=root / CORE_CONTRACT_RELATIVE,
        stats_path=stats_path,
        stats=stats,
        class_weights=(0.52, 15.0),
    )
    if (
        derived_control["train"]["epochs"] != 20
        or derived_control["train"]["save_every_epoch"] is not False
        or derived_control["train"]["paired_baseline_config_path"] is not None
    ):
        raise AssertionError("V21e OOF derived Control config probe failed")
    treatment_payload = json.loads(
        (root / BASE_TREATMENT_CONFIG_RELATIVE).read_text(encoding="utf-8")
    )
    derived_treatment = _config_payload(
        treatment_payload,
        root=root,
        arm="treatment",
        fold_id=0,
        data_contract_path=root / CORE_CONTRACT_RELATIVE,
        stats_path=stats_path,
        stats=stats,
        class_weights=(0.52, 15.0),
    )
    for payload in (derived_control, derived_treatment):
        payload["data"]["contract_path"] = (
            "data/processed/"
            "RADAR-LN3-V21e-cq08-known-gate-oof-fold-00-data-contract.json"
        )
        payload["data"]["normalization_stats_path"] = (
            "data/processed/"
            "RADAR-LN3-V21e-cq08-known-gate-oof-fold-00-"
            "train-normalization-stats.json"
        )
    with tempfile.TemporaryDirectory(prefix="v21e-oof-config-smoke-") as temporary:
        config_dir = Path(temporary) / "config" / "RADAR-LN3"
        config_dir.mkdir(parents=True)
        loaded_derived = {}
        for arm, payload in (
            ("control", derived_control),
            ("treatment", derived_treatment),
        ):
            path = config_dir / (
                f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-00-{arm}-"
                "seed43-config.json"
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            loaded_derived[arm] = load_known_gate_config(path)
        malformed = copy.deepcopy(derived_control)
        malformed["train"]["formal_artifact_dir"] += "-wrong"
        malformed_path = config_dir / (
            "RADAR-LN3-V21e-cq08-known-gate-oof-fold-01-control-"
            "seed43-config.json"
        )
        malformed["config_name"] = malformed_path.name.removesuffix("-config.json")
        malformed_path.write_text(
            json.dumps(malformed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            load_known_gate_config(malformed_path)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed V21e OOF config bypassed its narrow gate")
    if (
        loaded_derived["control"].train.save_every_epoch is not False
        or loaded_derived["treatment"].train.save_every_epoch is not False
        or loaded_derived["control"].variant != "baseline"
        or loaded_derived["treatment"].variant != "v21e_course_prefix"
    ):
        raise AssertionError("V21e OOF derived configs did not load as frozen support arms")
    treatment_config = load_known_gate_config(root / BASE_TREATMENT_CONFIG_RELATIVE)
    target_ids = [
        str(row["parent_sample_id"])
        for row in records
        if str(row["label_name"]) != "ClutterTarget"
    ][:2]
    clutter_ids = [
        str(row["parent_sample_id"])
        for row in records
        if str(row["label_name"]) == "ClutterTarget"
    ][:2]
    selected_ids = target_ids + clutter_ids
    dataset = build_known_gate_dataset(
        root,
        treatment_config,
        split_name="train",
        parent_ids=selected_ids,
        allow_feature_cache_superset=True,
    )
    if sorted(dataset.parent_ids) != sorted(selected_ids):
        raise AssertionError("V21e OOF Dataset parent subset differs")
    metadata = dict(dataset.feature_cache_metadata or {})
    if (
        metadata.get("subset_from_superset") is not True
        or int(metadata.get("event_count", -1))
        != int(dataset.base.frame["sample_id"].nunique())
        or int(metadata.get("source_cache_event_count", 0))
        <= int(metadata.get("event_count", 0))
    ):
        raise AssertionError("V21e OOF feature-cache subset contract failed")
    model, baseline, evidence = build_known_gate_model_for_config(
        root, treatment_config
    )
    if baseline is None or evidence.get("new_fusion_columns_exactly_zero") is not True:
        raise AssertionError("V21e OOF paired initialization failed")
    loader = build_known_gate_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    model.train()
    baseline.eval()
    treatment_output = model(**_model_inputs(batch))
    baseline_inputs = _model_inputs(batch)
    baseline_inputs["known_gate_features"] = None
    baseline_inputs["known_gate_feature_valid"] = None
    with torch.inference_mode():
        baseline_output = baseline(**baseline_inputs)
    if not torch.equal(
        treatment_output["sequence_logits"].detach(),
        baseline_output["sequence_logits"],
    ):
        raise AssertionError("V21e OOF initial paired logits differ")
    objective = parent_final_task_ce_with_prefix_auxiliary(
        treatment_output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=torch.tensor(treatment_config.train.class_weights),
        prefix_auxiliary_weight=treatment_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=treatment_config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    new_column_gradient = model.fusion[0].weight.grad[
        :, -treatment_config.model.evidence_dim :
    ]
    if (
        not torch.isfinite(objective["loss"])
        or new_column_gradient is None
        or not torch.isfinite(new_column_gradient).all()
        or float(new_column_gradient.abs().sum().item()) <= 0.0
    ):
        raise AssertionError("V21e OOF subset path has no finite feature gradient")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21e-known-gate-oof-contract-smoke-v1",
        "known_parent_count": int(first["known_parent_count"]),
        "provenance_component_count": int(first["provenance_component_count"]),
        "folds": first["folds"],
        "assignment_is_deterministic": True,
        "no_provenance_component_crosses_folds": True,
        "feature_cache_superset_subset_is_exact": True,
        "fold_data_contract_and_config_derivation_passed": True,
        "derived_control_and_treatment_configs_load": True,
        "malformed_oof_config_rejected": True,
        "paired_initial_logits_exactly_equal": True,
        "new_feature_column_receives_gradient": True,
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V21e OOF contracts.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
