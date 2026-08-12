from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .cascade import EXPECTED_TASKS, validate_cascade_configs
from .data import CQ08PointSequenceDataset
from .strict_open_set_oof import predict_oof_parent_final, sequence_sha256, sha256_file
from .task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    KNOWN_BASE_LABELS,
    UNKNOWN_BASE_LABEL,
    LiquidTaskConfig,
    task_config_identity_sha256,
)


NESTED_OOF_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V15-cq08-known-only-nested-oof-contract.json"
)
NESTED_OOF_CONTRACT_SCHEMA = (
    "RADAR-LN3-V15-known-only-nested-parent-grouped-oof-contract-v1"
)
NESTED_OOF_PREDICTION_SCHEMA = (
    "RADAR-LN3-V15-known-only-nested-parent-grouped-oof-prediction-v1"
)
NESTED_OOF_FORWARD_CLAIM_SCHEMA = (
    "RADAR-LN3-V15-known-only-nested-parent-grouped-oof-forward-claim-v1"
)
NESTED_OOF_ARTIFACT_PREFIX = (
    "RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness"
)
NESTED_OOF_MAX_EPOCHS = 20


def _resolve(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _relative(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def _inverse_frequency_weights(counts: tuple[int, ...]) -> tuple[float, ...]:
    total = sum(int(value) for value in counts)
    if total <= 0 or any(int(value) <= 0 for value in counts):
        raise ValueError("V15 nested OOF class-weight counts must all be positive")
    return tuple(total / (len(counts) * int(value)) for value in counts)


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        encoded = name.encode("utf-8")
        tensor = value.detach().cpu().contiguous()
        digest.update(len(encoded).to_bytes(4, "big", signed=False))
        digest.update(encoded)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(bytes(tensor.numpy().tobytes()))
    return digest.hexdigest()


def load_nested_oof_contract(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract_path = root / NESTED_OOF_CONTRACT_RELATIVE
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"V15 nested OOF contract is missing; run build.angular_knownness_oof: {contract_path}"
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    folds = list(payload.get("folds") or [])
    fold_count = int(payload.get("fold_count", -1))
    if (
        payload.get("schema_version") != NESTED_OOF_CONTRACT_SCHEMA
        or payload.get("model_schema_version") != ANGULAR_KNOWNNESS_SCHEMA_VERSION
        or payload.get("fit_split") != "train"
        or payload.get("fit_population") != "known_only"
        or payload.get("grouping") != "parent"
        or payload.get("prediction_protocol") != "nested_out_of_fold"
        or payload.get("epoch_protocol")
        != "inner_parent_selection_then_outer_train_retrain"
        or int(payload.get("known_parent_count", -1)) != 1011
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("unknown_oof_prediction_count", -1)) != 0
        or int(payload.get("validation_access_count", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or payload.get("outer_held_out_used_for_normalization") is not False
        or payload.get("outer_held_out_used_for_epoch_selection") is not False
        or int(payload.get("outer_held_out_prediction_pass_count", -1)) != 1
        or fold_count != 5
        or len(folds) != fold_count
    ):
        raise ValueError("V15 nested OOF contract violates the frozen execution protocol")
    source_files = {
        "assignment": (
            payload.get("source_assignment_manifest"),
            payload.get("source_assignment_manifest_sha256"),
        ),
        "v14_fold_contract": (
            payload.get("source_v14_outer_fold_contract"),
            payload.get("source_v14_outer_fold_contract_sha256"),
        ),
        "experiment_contract": (
            payload.get("source_experiment_contract"),
            payload.get("source_experiment_contract_sha256"),
        ),
    }
    for name, (relative, expected_hash) in source_files.items():
        path = _resolve(root, str(relative or ""))
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"V15 nested OOF {name} path/hash changed")
    assignment_path = _resolve(root, str(source_files["assignment"][0]))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignments = list(assignment.get("assignments") or [])
    all_ids = sorted(str(row["parent_sample_id"]) for row in assignments)
    if (
        len(assignments) != 1011
        or len(set(all_ids)) != 1011
        or sequence_sha256(all_ids) != payload["known_parent_ids_sha256"]
        or any(str(row.get("label_name")) == UNKNOWN_BASE_LABEL for row in assignments)
    ):
        raise ValueError("V15 nested OOF assignment exposes Unknown or repeats a parent")
    payload["_contract_path"] = str(contract_path)
    payload["_contract_sha256"] = sha256_file(contract_path)
    payload["_assignments"] = assignments
    return payload


def nested_fold_record(contract: Mapping[str, Any], outer_fold_id: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(contract.get("folds") or [])
        if int(row.get("outer_fold_id", -1)) == int(outer_fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V15 nested OOF outer fold is unavailable or repeated: {outer_fold_id}")
    fold = matches[0]
    outer_ids = sorted(str(value) for value in fold.get("outer_held_out_parent_ids") or [])
    inner_selection_ids = sorted(
        str(value) for value in fold.get("inner_selection_parent_ids") or []
    )
    inner_train_ids = sorted(
        str(value) for value in dict(fold.get("inner_train") or {}).get("train_parent_ids") or []
    )
    outer_train_ids = sorted(
        str(value) for value in dict(fold.get("outer_retrain") or {}).get("train_parent_ids") or []
    )
    populations = [set(outer_ids), set(inner_selection_ids), set(inner_train_ids)]
    all_ids = sorted(
        str(row["parent_sample_id"])
        for row in list(contract.get("_assignments") or [])
    )
    if (
        any(populations[i] & populations[j] for i in range(3) for j in range(i + 1, 3))
        or sorted(outer_ids + inner_selection_ids + inner_train_ids) != all_ids
        or sorted(inner_train_ids + inner_selection_ids) != outer_train_ids
        or len(outer_ids) != int(fold.get("outer_held_out_parent_count", -1))
        or len(inner_selection_ids) != int(fold.get("inner_selection_parent_count", -1))
        or sequence_sha256(outer_ids) != fold.get("outer_held_out_parent_ids_sha256")
        or sequence_sha256(inner_selection_ids)
        != fold.get("inner_selection_parent_ids_sha256")
        or fold.get("outer_held_out_used_for_normalization") is not False
        or fold.get("outer_held_out_used_for_epoch_selection") is not False
        or int(fold.get("outer_held_out_prediction_pass_count", -1)) != 1
        or int(fold.get("unknown_training_exposure", -1)) != 0
        or int(fold.get("validation_access_count", -1)) != 0
        or int(fold.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError(f"V15 nested OOF outer fold {outer_fold_id} violates isolation")
    fold["outer_held_out_parent_ids"] = outer_ids
    fold["inner_selection_parent_ids"] = inner_selection_ids
    fold["inner_train"]["train_parent_ids"] = inner_train_ids
    fold["outer_retrain"]["train_parent_ids"] = outer_train_ids
    return fold


def nested_artifact_root(project_root: Path, arm: str, outer_fold_id: int) -> Path:
    if arm not in {"control", "treatment"}:
        raise ValueError("V15 nested OOF arm must be control or treatment")
    return (
        project_root.resolve()
        / "artifacts"
        / f"{NESTED_OOF_ARTIFACT_PREFIX}-{arm}-nested-oof"
        / f"outer-{int(outer_fold_id):02d}"
    )


def nested_prediction_path(project_root: Path, arm: str, outer_fold_id: int) -> Path:
    return nested_artifact_root(project_root, arm, outer_fold_id) / "outer-held-out-predictions.json"


def nested_forward_claim_path(project_root: Path, arm: str, outer_fold_id: int) -> Path:
    return nested_artifact_root(project_root, arm, outer_fold_id) / "outer-held-out-forward-claim.json"


def _forward_claim_payload(
    *,
    contract: Mapping[str, Any],
    arm: str,
    fold: Mapping[str, Any],
    outer_config_identities: Mapping[str, str],
    dbb_knownness_score_name: str,
) -> dict[str, Any]:
    outer_ids = sorted(str(value) for value in fold["outer_held_out_parent_ids"])
    return {
        "schema_version": NESTED_OOF_FORWARD_CLAIM_SCHEMA,
        "status": "claimed",
        "arm": str(arm),
        "outer_fold_id": int(fold["outer_fold_id"]),
        "nested_oof_contract": str(contract["_contract_path"]),
        "nested_oof_contract_sha256": str(contract["_contract_sha256"]),
        "outer_held_out_parent_count": len(outer_ids),
        "outer_held_out_parent_ids_sha256": sequence_sha256(outer_ids),
        "outer_config_identities": dict(outer_config_identities),
        "dbb_knownness_score_name": str(dbb_knownness_score_name),
        "outer_held_out_forward_count_before_final": 0,
        "outer_held_out_prediction_pass_count": 1,
    }


def claim_outer_held_out_forward_once(
    project_root: Path,
    *,
    contract: Mapping[str, Any],
    arm: str,
    fold: Mapping[str, Any],
    outer_config_identities: Mapping[str, str],
    dbb_knownness_score_name: str,
) -> Path:
    """Persist a fail-closed claim before an outer held-out forward pass.

    A crash after this claim is intentionally not retried automatically: the
    protocol promises exactly one outer-held-out forward and cannot prove that
    a terminated CUDA call observed zero samples.  The caller must retain the
    claim alongside the completed prediction for later provenance checks.
    """

    path = nested_forward_claim_path(project_root, arm, int(fold["outer_fold_id"]))
    expected = _forward_claim_payload(
        contract=contract,
        arm=arm,
        fold=fold,
        outer_config_identities=outer_config_identities,
        dbb_knownness_score_name=dbb_knownness_score_name,
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"V15 nested OOF forward claim is not valid JSON: {path}") from exc
        if not isinstance(existing, dict):
            raise ValueError(f"V15 nested OOF forward claim is not an object: {path}")
        raise RuntimeError(
            "V15 nested OOF outer held-out forward was already claimed; "
            "refusing a duplicate pass. If no completed prediction exists, preserve the "
            f"claim for manual audit: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def complete_outer_held_out_forward_claim(
    claim_path: Path,
    *,
    prediction_path: Path,
) -> dict[str, Any]:
    if not claim_path.is_file():
        raise FileNotFoundError(f"V15 nested OOF outer held-out forward claim is missing: {claim_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"V15 nested OOF outer held-out prediction is missing before claim completion: {prediction_path}"
        )
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != NESTED_OOF_FORWARD_CLAIM_SCHEMA
        or payload.get("status") != "claimed"
    ):
        raise ValueError("V15 nested OOF forward claim is not in its one-time claimed state")
    payload["status"] = "completed"
    payload["prediction_path"] = str(prediction_path)
    payload["prediction_sha256"] = sha256_file(prediction_path)
    claim_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_completed_outer_held_out_forward_claim(
    project_root: Path,
    *,
    contract: Mapping[str, Any],
    arm: str,
    fold: Mapping[str, Any],
    outer_config_identities: Mapping[str, str],
    dbb_knownness_score_name: str,
    prediction_path: Path,
) -> dict[str, Any]:
    claim_path = nested_forward_claim_path(project_root, arm, int(fold["outer_fold_id"]))
    if not claim_path.is_file():
        raise FileNotFoundError(
            "V15 nested OOF completed prediction has no one-time outer-held-out forward claim: "
            f"{claim_path}"
        )
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    expected = _forward_claim_payload(
        contract=contract,
        arm=arm,
        fold=fold,
        outer_config_identities=outer_config_identities,
        dbb_knownness_score_name=dbb_knownness_score_name,
    )
    completed = {
        **expected,
        "status": "completed",
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
    }
    if payload != completed:
        raise ValueError("V15 nested OOF completed forward claim does not match the prediction contract")
    return payload


def derive_nested_stage_configs(
    project_root: Path,
    base_configs: Mapping[str, LiquidTaskConfig],
    fold: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, LiquidTaskConfig]:
    root = project_root.resolve()
    validate_cascade_configs(base_configs)
    if any(
        config.schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION
        for config in base_configs.values()
    ):
        raise ValueError("V15 nested OOF requires three V15 base configurations")
    arms = {config.train.paired_arm for config in base_configs.values()}
    if len(arms) != 1 or next(iter(arms)) not in {"control", "treatment"}:
        raise ValueError("V15 nested OOF configurations do not belong to one paired arm")
    arm = str(next(iter(arms)))
    if stage not in {"inner_selection", "outer_retrain"}:
        raise ValueError(f"unsupported V15 nested OOF stage: {stage}")
    outer_fold_id = int(fold["outer_fold_id"])
    stage_record = dict(fold["inner_train"] if stage == "inner_selection" else fold["outer_retrain"])
    train_ids = sorted(str(value) for value in stage_record.get("train_parent_ids") or [])
    if not train_ids or sequence_sha256(train_ids) != stage_record.get("train_parent_ids_sha256"):
        raise ValueError(f"V15 nested OOF {stage} train parent hash mismatch")
    stats_path = _resolve(root, str(stage_record.get("normalization_stats_path", "")))
    contract_path = _resolve(root, str(stage_record.get("data_contract_path", "")))
    if (
        not stats_path.is_file()
        or sha256_file(stats_path) != stage_record.get("normalization_stats_sha256")
        or not contract_path.is_file()
        or sha256_file(contract_path) != stage_record.get("data_contract_sha256")
    ):
        raise ValueError(f"V15 nested OOF {stage} data/normalization path or hash changed")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stage_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    oof_contract = dict(stage_contract.get("oof_fold_training_contract") or {})
    train_counts = {
        str(key): int(value)
        for key, value in dict(stage_record.get("train_label_parent_counts") or {}).items()
    }
    if (
        stats.get("schema_version")
        not in {"RADAR-LN3-V14", ANGULAR_KNOWNNESS_SCHEMA_VERSION}
        or int(stats.get("parent_count", -1)) != len(train_ids)
        or stats.get("allowed_parent_ids_sha256") != sequence_sha256(train_ids)
        or stats.get("forbidden_labels") != [UNKNOWN_BASE_LABEL]
        or set(train_counts) != set(KNOWN_BASE_LABELS)
        or stage_contract.get("schema_version") != ANGULAR_KNOWNNESS_SCHEMA_VERSION
        or oof_contract.get("nested_oof_stage") != stage
        or int(oof_contract.get("fold_id", -1)) != outer_fold_id
        or oof_contract.get("train_parent_ids_sha256") != sequence_sha256(train_ids)
        or int(oof_contract.get("unknown_training_exposure", -1)) != 0
        or int(oof_contract.get("validation_access_count", -1)) != 0
        or int(oof_contract.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError(f"V15 nested OOF {stage} contract permits leakage or changed support")
    center = float(stats["amplitude"]["center"])
    scale = float(stats["amplitude"]["scale"])
    if not torch.isfinite(torch.tensor([center, scale])).all() or scale <= 0.0:
        raise ValueError(f"V15 nested OOF {stage} normalization values are invalid")
    dbb_count = sum(train_counts[label] for label in KNOWN_BASE_LABELS[:3])
    clutter_count = train_counts["ClutterTarget"]
    weights = {
        "dbb_cu_gate": _inverse_frequency_weights((dbb_count, clutter_count)),
        "dbb_specialist": _inverse_frequency_weights(
            tuple(train_counts[label] for label in KNOWN_BASE_LABELS[:3])
        ),
        "cu_specialist": _inverse_frequency_weights((clutter_count, dbb_count)),
    }
    stage_slug = stage.replace("_", "-")
    stage_root = nested_artifact_root(root, arm, outer_fold_id) / stage_slug
    derived: dict[str, LiquidTaskConfig] = {}
    for name in EXPECTED_TASKS:
        base = base_configs[name]
        task_root = stage_root / name
        data = replace(
            base.data,
            contract_path=_relative(root, contract_path),
            normalization_stats_path=_relative(root, stats_path),
            expected_contract_sha256=sha256_file(contract_path),
            expected_stats_sha256=sha256_file(stats_path),
            rd_amplitude_center=center,
            rd_amplitude_scale=scale,
        )
        schema_stem = (
            f"RADAR-LN3-V15-angular-knownness-{arm}-nested-oof-"
            f"outer-{outer_fold_id:02d}-{stage_slug}"
        )
        train = replace(
            base.train,
            epochs=NESTED_OOF_MAX_EPOCHS,
            early_stopping_patience=0,
            class_weights=weights[name],
            checkpoint_path=_relative(root, task_root / "final_model.pt"),
            last_checkpoint_path=_relative(root, task_root / "last_model.pt"),
            formal_artifact_dir=_relative(root, task_root),
            epoch_checkpoint_dir=_relative(root, task_root / "epochs"),
            joint_artifact_dir=_relative(root, stage_root),
            joint_last_checkpoint_path=_relative(root, stage_root / "joint_last_model.pt"),
            joint_best_checkpoint_path=_relative(root, stage_root / "joint_best_model.pt"),
            joint_checkpoint_schema=f"{schema_stem}-joint-checkpoint-v1",
            checkpoint_schema=f"{schema_stem}-{name.replace('_', '-')}-checkpoint-v1",
        )
        derived[name] = replace(
            base,
            config_name=(
                f"{base.config_name}-nested-oof-outer-{outer_fold_id:02d}-{stage_slug}"
            ),
            data=data,
            train=train,
        )
    validate_cascade_configs(derived)
    if any(config.train.epochs != NESTED_OOF_MAX_EPOCHS for config in derived.values()):
        raise ValueError("V15 nested OOF derived configurations changed the epoch ceiling")
    return derived


def verify_nested_dataset(
    dataset: CQ08PointSequenceDataset,
    *,
    expected_parent_ids: list[str],
    scope: str,
) -> dict[str, Any]:
    expected = sorted(str(value) for value in expected_parent_ids)
    actual = sorted(str(value) for value in dataset.parent_ids)
    labels = set(dataset.frame["label_name"].astype(str))
    if (
        actual != expected
        or len(actual) != len(set(actual))
        or labels - set(KNOWN_BASE_LABELS)
        or UNKNOWN_BASE_LABEL in labels
        or set(dataset.frame["split_name"].astype(str)) != {"train"}
    ):
        raise ValueError(f"V15 nested OOF {scope} Dataset violates known train-parent scope")
    return {
        "scope": scope,
        "parent_count": len(actual),
        "parent_ids_sha256": sequence_sha256(actual),
        "labels": sorted(labels),
        "unknown_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }


def predict_outer_held_out_once(
    model: Any,
    dataset: CQ08PointSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    outer_fold_id: int,
) -> list[dict[str, Any]]:
    records = predict_oof_parent_final(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        fold_id=outer_fold_id,
    )
    for record in records:
        record["nested_oof_outer_fold_id"] = int(outer_fold_id)
        record["prediction_protocol"] = "nested_out_of_fold"
        record["outer_held_out_prediction_pass_count"] = 1
    return records


def nested_config_identities(
    configs: Mapping[str, LiquidTaskConfig],
) -> dict[str, str]:
    if set(configs) != set(EXPECTED_TASKS):
        raise ValueError("V15 nested OOF identity requires exactly three task configs")
    return {name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS}


__all__ = [
    "NESTED_OOF_ARTIFACT_PREFIX",
    "NESTED_OOF_CONTRACT_RELATIVE",
    "NESTED_OOF_CONTRACT_SCHEMA",
    "NESTED_OOF_FORWARD_CLAIM_SCHEMA",
    "NESTED_OOF_MAX_EPOCHS",
    "NESTED_OOF_PREDICTION_SCHEMA",
    "claim_outer_held_out_forward_once",
    "complete_outer_held_out_forward_claim",
    "derive_nested_stage_configs",
    "load_nested_oof_contract",
    "model_state_sha256",
    "nested_artifact_root",
    "nested_config_identities",
    "nested_fold_record",
    "nested_forward_claim_path",
    "nested_prediction_path",
    "predict_outer_held_out_once",
    "validate_completed_outer_held_out_forward_claim",
    "verify_nested_dataset",
]
