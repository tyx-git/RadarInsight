from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import LABELS
from .cascade import EXPECTED_TASKS, LiquidSoftCascade, validate_cascade_configs
from .data import CQ08PointSequenceDataset, build_point_dataloader
from .task_models import (
    DBB_KNOWNNESS_COSINE,
    DBB_KNOWNNESS_PROBABILITY,
    KNOWN_BASE_LABELS,
    STRICT_OPEN_SET_SCHEMA_VERSION,
    UNKNOWN_BASE_LABEL,
    LiquidTaskConfig,
    is_strict_open_set_schema,
    task_config_identity_sha256,
)


OOF_ASSIGNMENT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"
)
OOF_FOLD_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"
)
OOF_ARTIFACT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-parent-grouped-oof"
)
OOF_PREDICTION_SCHEMA = "RADAR-LN3-V14-known-only-parent-grouped-oof-prediction-v1"
OOF_FIXED_EPOCHS = 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _relative(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def _inverse_frequency_weights(counts: tuple[int, ...]) -> tuple[float, ...]:
    total = sum(int(value) for value in counts)
    if total <= 0 or any(int(value) <= 0 for value in counts):
        raise ValueError("V14 OOF class-weight counts must all be positive")
    return tuple(total / (len(counts) * int(value)) for value in counts)


def load_oof_fold_contract(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    fold_path = root / OOF_FOLD_CONTRACT_RELATIVE
    if not fold_path.is_file():
        raise FileNotFoundError(fold_path)
    fold_contract = json.loads(fold_path.read_text(encoding="utf-8"))
    assignment_path = _resolve(root, str(fold_contract.get("assignment_manifest", "")))
    if not assignment_path.is_file():
        raise FileNotFoundError(assignment_path)
    if (
        assignment_path != (root / OOF_ASSIGNMENT_RELATIVE).resolve()
        or sha256_file(assignment_path) != fold_contract.get("assignment_manifest_sha256")
    ):
        raise ValueError("V14 OOF assignment manifest path/hash does not match the fold contract")
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    folds = list(fold_contract.get("folds") or [])
    fold_count = int(fold_contract.get("fold_count", -1))
    if (
        fold_contract.get("schema_version")
        != "RADAR-LN3-V14-known-only-parent-grouped-oof-fold-contract-v2"
        or fold_contract.get("fit_split") != "train"
        or fold_contract.get("fit_population") != "known_only"
        or fold_contract.get("grouping") != "parent"
        or fold_contract.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
        or int(fold_contract.get("known_parent_count", -1)) != 1011
        or int(fold_contract.get("unknown_training_exposure", -1)) != 0
        or int(fold_contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(fold_contract.get("validation_access_count", -1)) != 0
        or int(fold_contract.get("test_payload_access_count", -1)) != 0
        or len(folds) != fold_count
        or fold_count < 2
    ):
        raise ValueError("V14 OOF fold contract violates the frozen known-only protocol")
    policy = dict(fold_contract.get("execution_policy") or {})
    if (
        int(policy.get("epochs_per_fold", -1)) != OOF_FIXED_EPOCHS
        or policy.get("early_stopping") is not False
        or policy.get("held_out_checkpoint_selection") is not False
        or policy.get("held_out_prediction_time") != "once_after_fixed_final_epoch"
    ):
        raise ValueError("V14 OOF execution policy is not fixed-final-epoch")
    if (
        assignment.get("fit_split") != "train"
        or assignment.get("fit_population") != "known_only"
        or assignment.get("grouping") != "parent"
        or assignment.get("prediction_protocol") != "out_of_fold"
        or int(assignment.get("known_parent_count", -1)) != 1011
        or int(assignment.get("unknown_oof_parent_count", -1)) != 0
        or int(assignment.get("validation_parent_count", -1)) != 0
        or int(assignment.get("test_parent_count", -1)) != 0
    ):
        raise ValueError("V14 OOF assignment manifest is not known-only train-parent OOF")
    assignments = list(assignment.get("assignments") or [])
    assigned_ids = sorted(str(row["parent_sample_id"]) for row in assignments)
    if (
        len(assignments) != 1011
        or len(set(assigned_ids)) != 1011
        or any(str(row.get("label_name")) == UNKNOWN_BASE_LABEL for row in assignments)
        or sequence_sha256(assigned_ids) != assignment.get("known_parent_ids_sha256")
    ):
        raise ValueError("V14 OOF assignment does not cover each known parent exactly once")
    return fold_contract, assignment


def fold_record(fold_contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    matches = [
        dict(value)
        for value in list(fold_contract.get("folds") or [])
        if int(value.get("fold_id", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V14 OOF fold id is unavailable or repeated: {fold_id}")
    record = matches[0]
    train_ids = sorted(str(value) for value in record.get("train_parent_ids") or [])
    held_out_ids = sorted(str(value) for value in record.get("held_out_parent_ids") or [])
    if (
        not train_ids
        or not held_out_ids
        or set(train_ids) & set(held_out_ids)
        or len(train_ids) != int(record.get("train_parent_count", -1))
        or len(held_out_ids) != int(record.get("held_out_parent_count", -1))
        or len(train_ids) + len(held_out_ids) != 1011
        or sequence_sha256(train_ids) != record.get("train_parent_ids_sha256")
        or sequence_sha256(held_out_ids) != record.get("held_out_parent_ids_sha256")
        or int(record.get("unknown_train_parent_count", -1)) != 0
        or int(record.get("unknown_held_out_parent_count", -1)) != 0
        or int(record.get("validation_parent_count", -1)) != 0
        or int(record.get("test_parent_count", -1)) != 0
        or record.get("held_out_used_for_normalization") is not False
    ):
        raise ValueError(f"V14 OOF fold {fold_id} violates its parent partition")
    record["train_parent_ids"] = train_ids
    record["held_out_parent_ids"] = held_out_ids
    return record


def derive_oof_fold_configs(
    project_root: Path,
    base_configs: Mapping[str, LiquidTaskConfig],
    fold: Mapping[str, Any],
) -> dict[str, LiquidTaskConfig]:
    root = project_root.resolve()
    validate_cascade_configs(base_configs)
    if any(config.schema_version != STRICT_OPEN_SET_SCHEMA_VERSION for config in base_configs.values()):
        raise ValueError("V14 OOF requires three strict-open-set base configurations")
    fold_id = int(fold["fold_id"])
    stats_path = _resolve(root, str(fold.get("normalization_stats_path", "")))
    if not stats_path.is_file() or sha256_file(stats_path) != fold.get("normalization_stats_sha256"):
        raise ValueError(f"V14 OOF fold {fold_id} normalization path/hash mismatch")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_counts = {str(key): int(value) for key, value in dict(fold["train_label_parent_counts"]).items()}
    expected_labels = set(KNOWN_BASE_LABELS)
    if (
        set(train_counts) != expected_labels
        or stats.get("schema_version") != STRICT_OPEN_SET_SCHEMA_VERSION
        or int(stats.get("parent_count", -1)) != int(fold["train_parent_count"])
        or stats.get("selected_label_parent_counts") != {
            key: train_counts[key] for key in sorted(train_counts)
        }
        or stats.get("allowed_labels") != sorted(KNOWN_BASE_LABELS)
        or stats.get("forbidden_labels") != [UNKNOWN_BASE_LABEL]
        or int(stats.get("allowed_parent_ids_count", -1)) != int(fold["train_parent_count"])
        or stats.get("allowed_parent_ids_sha256") != fold.get("train_parent_ids_sha256")
        or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
        or "no validation or test statistics" not in str(stats.get("scope", ""))
    ):
        raise ValueError(f"V14 OOF fold {fold_id} normalization contains held-out or Unknown exposure")
    amplitude = dict(stats.get("amplitude") or {})
    center = float(amplitude["center"])
    scale = float(amplitude["scale"])
    if not torch.isfinite(torch.tensor([center, scale])).all() or scale <= 0.0:
        raise ValueError(f"V14 OOF fold {fold_id} normalization values are invalid")
    contract_path = _resolve(root, str(fold.get("data_contract_path", "")))
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != fold.get("data_contract_sha256")
    ):
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    oof_contract = dict(contract.get("oof_fold_training_contract") or {})
    normalization_contract = dict(contract.get("normalization_contract") or {})
    if (
        contract.get("schema_version") != STRICT_OPEN_SET_SCHEMA_VERSION
        or int(oof_contract.get("fold_id", -1)) != fold_id
        or oof_contract.get("fit_split") != "train"
        or oof_contract.get("fit_population") != "known_only"
        or oof_contract.get("grouping") != "parent"
        or int(oof_contract.get("train_parent_count", -1)) != len(fold["train_parent_ids"])
        or int(oof_contract.get("held_out_parent_count", -1)) != len(fold["held_out_parent_ids"])
        or oof_contract.get("train_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or oof_contract.get("held_out_parent_ids_sha256") != fold["held_out_parent_ids_sha256"]
        or int(oof_contract.get("unknown_training_exposure", -1)) != 0
        or int(oof_contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(oof_contract.get("validation_access_count", -1)) != 0
        or int(oof_contract.get("test_payload_access_count", -1)) != 0
        or int(oof_contract.get("epochs", -1)) != OOF_FIXED_EPOCHS
        or oof_contract.get("early_stopping") is not False
        or oof_contract.get("held_out_checkpoint_selection") is not False
        or normalization_contract.get("sha256") != sha256_file(stats_path)
        or normalization_contract.get("allowed_parent_ids_sha256") != fold["train_parent_ids_sha256"]
    ):
        raise ValueError(f"V14 OOF fold {fold_id} data contract is not fold-train-only")
    dbb_count = sum(train_counts[label] for label in LABELS[:3])
    clutter_count = train_counts["ClutterTarget"]
    weights = {
        "dbb_cu_gate": _inverse_frequency_weights((dbb_count, clutter_count)),
        "dbb_specialist": _inverse_frequency_weights(
            tuple(train_counts[label] for label in LABELS[:3])
        ),
        "cu_specialist": _inverse_frequency_weights((clutter_count, dbb_count)),
    }
    fold_root = root / OOF_ARTIFACT_RELATIVE / f"fold-{fold_id:02d}"
    derived: dict[str, LiquidTaskConfig] = {}
    for name in EXPECTED_TASKS:
        base = base_configs[name]
        task_root = fold_root / name
        data = replace(
            base.data,
            contract_path=_relative(root, contract_path),
            normalization_stats_path=_relative(root, stats_path),
            expected_contract_sha256=sha256_file(contract_path),
            expected_stats_sha256=sha256_file(stats_path),
            rd_amplitude_center=center,
            rd_amplitude_scale=scale,
        )
        train = replace(
            base.train,
            epochs=OOF_FIXED_EPOCHS,
            early_stopping_patience=0,
            class_weights=weights[name],
            checkpoint_path=_relative(root, task_root / "final_model.pt"),
            last_checkpoint_path=_relative(root, task_root / "last_model.pt"),
            formal_artifact_dir=_relative(root, task_root),
            epoch_checkpoint_dir=_relative(root, task_root / "epochs"),
            joint_artifact_dir=_relative(root, fold_root),
            joint_last_checkpoint_path=_relative(root, fold_root / "joint_last_model.pt"),
            joint_best_checkpoint_path=_relative(root, fold_root / "joint_final_model.pt"),
            joint_checkpoint_schema=(
                f"RADAR-LN3-V14-strict-open-set-oof-fold-{fold_id:02d}-joint-checkpoint-v1"
            ),
            checkpoint_schema=(
                f"RADAR-LN3-V14-{name.replace('_', '-')}-oof-fold-{fold_id:02d}-checkpoint-v1"
            ),
        )
        derived[name] = replace(
            base,
            config_name=f"{base.config_name}-oof-fold-{fold_id:02d}",
            data=data,
            train=train,
        )
    validate_cascade_configs(derived)
    if any(config.train.epochs != OOF_FIXED_EPOCHS for config in derived.values()):
        raise ValueError("V14 OOF derived configurations changed the fixed epoch budget")
    return derived


def verify_oof_dataset(
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
        raise ValueError(f"V14 OOF {scope} Dataset violates known train-parent scope")
    return {
        "scope": scope,
        "parent_count": len(actual),
        "parent_ids_sha256": sequence_sha256(actual),
        "labels": sorted(labels),
        "unknown_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }


def predict_oof_parent_final(
    model: LiquidSoftCascade,
    dataset: CQ08PointSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    fold_id: int,
) -> list[dict[str, Any]]:
    if not is_strict_open_set_schema(model.gate.config.schema_version):
        raise ValueError("OOF prediction requires a strict-open-set cascade")
    if UNKNOWN_BASE_LABEL in set(dataset.frame["label_name"].astype(str)):
        raise ValueError("UnknownTarget entered V14 OOF held-out prediction")
    loader = build_point_dataloader(dataset, batch_size=batch_size, shuffle=False)
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_dt_reason=tensors["tr_dt_reason"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
            gate = torch.softmax(output["gate_sequence_logits"], dim=-1)
            dbb = torch.softmax(output["dbb_sequence_logits"], dim=-1)
            cu = torch.softmax(output["cu_sequence_logits"], dim=-1)
            dbb_knownness = output["dbb_knownness_sequence_scores"]
            dbb_score_name = str(output["dbb_knownness_score_name"])
            if dbb_knownness.shape != dbb.shape or dbb_score_name not in {
                DBB_KNOWNNESS_PROBABILITY,
                DBB_KNOWNNESS_COSINE,
            }:
                raise ValueError("OOF DBB knownness output violates its score contract")
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][index].item())
                true_id = int(batch["labels"][index].item())
                if not 0 <= true_id < 4 or length < 1:
                    raise ValueError("V14 OOF held-out prediction received an invalid known parent")
                final_index = length - 1
                gate_final = gate[index, final_index]
                dbb_final = dbb[index, final_index]
                cu_final = cu[index, final_index]
                dbb_id = int(dbb_final.argmax().item())
                dbb_known_final = dbb_knownness[index, final_index]
                dbb_known_top = torch.topk(dbb_known_final, k=2)
                if int(dbb_known_top.indices[0].item()) != dbb_id:
                    raise ValueError("OOF DBB logits and knownness scores disagree on the top class")
                rd_observed = int(
                    (tensors["rd_quality"][index, :length, 0] >= 0.5).sum().item()
                )
                zero_delta = int(
                    (tensors["tr_dt_reason"][index, :length] == 2).sum().item()
                )
                record = {
                    "parent_sample_id": str(parent_id),
                    "fold_id": int(fold_id),
                    "split_name": "train_oof_held_out",
                    "label_name": str(batch["label_names"][index]),
                    "true_id": true_id,
                    "event_count": length,
                    "gate_dbb_probability": float(gate_final[0].item()),
                    "gate_margin": float((gate_final[0] - gate_final[1]).item()),
                    "dbb_pred_id": dbb_id,
                    "dbb_pred_name": LABELS[dbb_id],
                    "dbb_knownness_score_name": dbb_score_name,
                    "dbb_known_confidence": float(dbb_known_top.values[0].item()),
                    "dbb_known_margin": float(
                        (dbb_known_top.values[0] - dbb_known_top.values[1]).item()
                    ),
                    "cu_clutter_probability": float(cu_final[0].item()),
                    "cu_clutter_margin": float((cu_final[0] - cu_final[1]).item()),
                    "rd_observed_fraction": rd_observed / length,
                    "zero_delta_fraction": zero_delta / max(1, length - 1),
                    "unknown_is_model_class": False,
                    "unknown_training_exposure": 0,
                }
                if dbb_score_name == DBB_KNOWNNESS_PROBABILITY:
                    record["dbb_top_probability"] = record["dbb_known_confidence"]
                    record["dbb_top_margin"] = record["dbb_known_margin"]
                numeric = [
                    record["gate_dbb_probability"],
                    record["gate_margin"],
                    record["dbb_known_confidence"],
                    record["dbb_known_margin"],
                    record["cu_clutter_probability"],
                    record["cu_clutter_margin"],
                    record["rd_observed_fraction"],
                    record["zero_delta_fraction"],
                ]
                if not torch.isfinite(torch.tensor(numeric)).all():
                    raise ValueError("V14 OOF prediction produced a non-finite score")
                records.append(record)
    records.sort(key=lambda value: str(value["parent_sample_id"]))
    if (
        len(records) != len(dataset)
        or len({str(value["parent_sample_id"]) for value in records}) != len(records)
        or any(value["label_name"] == UNKNOWN_BASE_LABEL for value in records)
    ):
        raise ValueError("V14 OOF held-out prediction does not cover each parent exactly once")
    return records


def fold_prediction_path(project_root: Path, fold_id: int) -> Path:
    return (
        project_root.resolve()
        / OOF_ARTIFACT_RELATIVE
        / f"fold-{int(fold_id):02d}"
        / "held-out-predictions.json"
    )


def fold_config_identities(configs: Mapping[str, LiquidTaskConfig]) -> dict[str, str]:
    if set(configs) != set(EXPECTED_TASKS):
        raise ValueError("V14 OOF configuration identity requires exactly three tasks")
    return {name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS}


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(
    truth_positive: np.ndarray,
    predicted_positive: np.ndarray,
    *,
    positive_name: str,
    negative_name: str,
) -> dict[str, Any]:
    truth = np.asarray(truth_positive, dtype=bool)
    predicted = np.asarray(predicted_positive, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("V14 OOF binary metrics require matching non-empty vectors")
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    pr_values: list[float] = []
    for name, true_mask, pred_mask in (
        (positive_name, truth, predicted),
        (negative_name, ~truth, ~predicted),
    ):
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        rows.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        f1_values.append(f1)
        pr_values.extend((precision, recall))
    return {
        "accuracy": float((truth == predicted).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "minimum_precision_recall": min(pr_values),
        "per_class": rows,
    }


def _dbb_acceptance_metrics(
    records: list[dict[str, Any]],
    *,
    probability_threshold: float,
    margin_threshold: float,
    confidence_field: str = "dbb_top_probability",
    margin_field: str = "dbb_top_margin",
) -> dict[str, Any]:
    dbb_records = [record for record in records if int(record["true_id"]) < 3]
    if not dbb_records:
        raise ValueError("V14 OOF DBB calibration has no DBB parent")
    accepted = [
        float(record[confidence_field]) >= float(probability_threshold)
        and float(record[margin_field]) >= float(margin_threshold)
        for record in dbb_records
    ]
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    pr_values: list[float] = []
    for class_id, name in enumerate(LABELS[:3]):
        true_mask = np.asarray(
            [int(record["true_id"]) == class_id for record in dbb_records],
            dtype=bool,
        )
        pred_mask = np.asarray(
            [
                is_accepted and int(record["dbb_pred_id"]) == class_id
                for record, is_accepted in zip(dbb_records, accepted)
            ],
            dtype=bool,
        )
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        rows.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        f1_values.append(f1)
        pr_values.extend((precision, recall))
    return {
        "parent_count": len(dbb_records),
        "accepted_parent_count": int(sum(accepted)),
        "coverage": float(np.mean(accepted)),
        "macro_f1": float(np.mean(f1_values)),
        "minimum_precision_recall": min(pr_values),
        "per_class": rows,
    }


def _quantile_candidates(
    values: np.ndarray,
    *,
    maximum: int = 65,
    lower: float = 0.0,
    upper: float = 1.0 - 1.0e-7,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("V14 OOF threshold candidates require finite scores")
    if len(np.unique(array)) <= maximum:
        candidates = np.unique(array)
    else:
        candidates = np.quantile(array, np.linspace(0.0, 1.0, maximum))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("OOF threshold candidate bounds are invalid")
    return sorted({float(np.clip(value, lower, upper)) for value in candidates})


def _select_binary_boundary(
    scores: np.ndarray,
    truth_positive: np.ndarray,
    *,
    target: float,
    positive_name: str,
    negative_name: str,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth_positive, dtype=bool)
    if values.shape != truth.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("V14 OOF binary boundary received invalid score/label vectors")
    candidates = sorted(
        {
            0.5,
            *(
                float(np.clip(value, 0.5, 1.0 - 1.0e-7))
                for value in np.unique(values[values >= 0.5])
            ),
        }
    )
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in candidates:
        metrics = _binary_metrics(
            truth,
            values >= threshold,
            positive_name=positive_name,
            negative_name=negative_name,
        )
        pr_values = [
            float(row[field])
            for row in metrics["per_class"]
            for field in ("precision", "recall")
        ]
        passed = sum(value >= target for value in pr_values)
        key = (
            float(passed),
            min(value / target for value in pr_values),
            min(pr_values),
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "probability_threshold": float(threshold),
                "margin_threshold": float(max(0.0, 2.0 * threshold - 1.0)),
                "target_min_precision_recall": float(target),
                "passed_precision_recall_count": int(passed),
                "total_precision_recall_count": len(pr_values),
                "all_precision_recall_targets_passed": passed == len(pr_values),
                "metrics": metrics,
                "candidate_count": len(candidates),
                "selection": (
                    "lexicographic known-only OOF: passed P/R count, normalized floor, "
                    "raw floor, Macro-F1, Accuracy, then less restrictive threshold"
                ),
                "binary_margin_note": (
                    "for two-class softmax, margin=2*p-1; the recorded margin boundary "
                    "is derived from the selected probability boundary"
                ),
            }
    if best is None:
        raise RuntimeError("V14 OOF binary boundary search produced no candidate")
    return best


def _select_dbb_boundary(
    records: list[dict[str, Any]],
    *,
    target: float,
    score_name: str = DBB_KNOWNNESS_PROBABILITY,
) -> dict[str, Any]:
    dbb_records = [record for record in records if int(record["true_id"]) < 3]
    if score_name == DBB_KNOWNNESS_PROBABILITY:
        confidence_field = "dbb_top_probability"
        margin_field = "dbb_top_margin"
        confidence_floor = 1.0e-7
        confidence_bounds = (0.0, 1.0 - 1.0e-7)
        margin_bounds = (0.0, 1.0 - 1.0e-7)
    elif score_name == DBB_KNOWNNESS_COSINE:
        confidence_field = "dbb_known_confidence"
        margin_field = "dbb_known_margin"
        confidence_floor = -1.0 + 1.0e-7
        confidence_bounds = (-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        margin_bounds = (0.0, 2.0 - 1.0e-7)
    else:
        raise ValueError(f"unsupported DBB OOF knownness score: {score_name!r}")
    observed_scores = {str(record.get("dbb_knownness_score_name", score_name)) for record in records}
    if observed_scores != {score_name}:
        raise ValueError("OOF records mix DBB knownness score semantics")
    confidences = np.asarray(
        [float(record[confidence_field]) for record in dbb_records],
        dtype=np.float64,
    )
    margins = np.asarray(
        [float(record[margin_field]) for record in dbb_records],
        dtype=np.float64,
    )
    probability_candidates = sorted({
        confidence_floor,
        *_quantile_candidates(
            confidences,
            lower=confidence_bounds[0],
            upper=confidence_bounds[1],
        ),
    })
    margin_candidates = sorted({
        0.0,
        *_quantile_candidates(
            margins,
            lower=margin_bounds[0],
            upper=margin_bounds[1],
        ),
    })
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    candidate_count = 0
    for probability_threshold in probability_candidates:
        for margin_threshold in margin_candidates:
            candidate_count += 1
            metrics = _dbb_acceptance_metrics(
                records,
                probability_threshold=probability_threshold,
                margin_threshold=margin_threshold,
                confidence_field=confidence_field,
                margin_field=margin_field,
            )
            pr_values = [
                float(row[field])
                for row in metrics["per_class"]
                for field in ("precision", "recall")
            ]
            passed = sum(value >= target for value in pr_values)
            key = (
                float(passed),
                min(value / target for value in pr_values),
                min(pr_values),
                float(metrics["macro_f1"]),
                float(metrics["coverage"]),
                -float(probability_threshold),
                -float(margin_threshold),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "probability_threshold": float(probability_threshold),
                    "margin_threshold": float(margin_threshold),
                    "target_min_precision_recall": float(target),
                    "passed_precision_recall_count": int(passed),
                    "total_precision_recall_count": len(pr_values),
                    "all_precision_recall_targets_passed": passed == len(pr_values),
                    "metrics": metrics,
                    "selection": (
                        "lexicographic known-only OOF finite grid: passed P/R count, "
                        "normalized floor, raw floor, Macro-F1, coverage, then less restrictive boundaries"
                    ),
                    "dbb_knownness_score_name": score_name,
                    "confidence_field": confidence_field,
                    "margin_field": margin_field,
                }
    if best is None:
        raise RuntimeError("V14 OOF DBB boundary search produced no candidate")
    best["candidate_count"] = candidate_count
    best["probability_candidate_count"] = len(probability_candidates)
    best["margin_candidate_count"] = len(margin_candidates)
    return best


def summarize_known_oof_actions(
    records: list[dict[str, Any]],
    *,
    gate_probability_threshold: float,
    gate_margin_threshold: float,
    dbb_probability_threshold: float,
    dbb_margin_threshold: float,
    cu_probability_threshold: float,
    cu_margin_threshold: float,
    dbb_score_name: str = DBB_KNOWNNESS_PROBABILITY,
) -> dict[str, Any]:
    if not records or any(int(record["true_id"]) >= 4 for record in records):
        raise ValueError("V14 OOF action summary accepts known parents only")
    predicted: list[int] = []
    confidence_field = (
        "dbb_known_confidence"
        if dbb_score_name == DBB_KNOWNNESS_COSINE
        else "dbb_top_probability"
    )
    margin_field = (
        "dbb_known_margin"
        if dbb_score_name == DBB_KNOWNNESS_COSINE
        else "dbb_top_margin"
    )
    for record in records:
        gate_accepts = (
            float(record["gate_dbb_probability"]) >= gate_probability_threshold
            and float(record["gate_margin"]) >= gate_margin_threshold
        )
        if gate_accepts:
            dbb_accepts = (
                float(record[confidence_field]) >= dbb_probability_threshold
                and float(record[margin_field]) >= dbb_margin_threshold
            )
            predicted.append(int(record["dbb_pred_id"]) if dbb_accepts else 4)
        else:
            clutter_accepts = (
                float(record["cu_clutter_probability"]) >= cu_probability_threshold
                and float(record["cu_clutter_margin"]) >= cu_margin_threshold
            )
            predicted.append(3 if clutter_accepts else 4)
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    pr_values: list[float] = []
    for class_id, name in enumerate(LABELS[:4]):
        true_mask = np.asarray(
            [int(record["true_id"]) == class_id for record in records],
            dtype=bool,
        )
        pred_mask = np.asarray([value == class_id for value in predicted], dtype=bool)
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        rows.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        f1_values.append(f1)
        pr_values.extend((precision, recall))
    accepted = np.asarray([value < 4 for value in predicted], dtype=bool)
    true_ids = np.asarray([int(record["true_id"]) for record in records], dtype=np.int64)
    pred_ids = np.asarray(predicted, dtype=np.int64)
    return {
        "known_parent_count": len(records),
        "accepted_known_parent_count": int(accepted.sum()),
        "rejected_known_parent_count": int((~accepted).sum()),
        "known_coverage": float(accepted.mean()),
        "known_false_rejection_rate": float((~accepted).mean()),
        "accepted_accuracy": _safe_divide(
            int(((pred_ids == true_ids) & accepted).sum()),
            int(accepted.sum()),
        ),
        "overall_known_accuracy_counting_reject_as_error": float((pred_ids == true_ids).mean()),
        "macro_f1_counting_reject_as_error": float(np.mean(f1_values)),
        "minimum_precision_recall": min(pr_values),
        "per_class": rows,
        "unknown_detection_metrics": "not_available_from_known-only OOF",
    }


def fit_known_only_oof_boundaries(
    records: list[dict[str, Any]],
    *,
    gate_target: float,
    dbb_target: float,
    cu_target: float,
    dbb_score_name: str = DBB_KNOWNNESS_PROBABILITY,
) -> dict[str, Any]:
    parent_ids = [str(record["parent_sample_id"]) for record in records]
    true_ids = np.asarray([int(record["true_id"]) for record in records], dtype=np.int64)
    if (
        len(records) != 1011
        or len(set(parent_ids)) != 1011
        or any(not 0 <= value < 4 for value in true_ids.tolist())
        or any(record.get("label_name") == UNKNOWN_BASE_LABEL for record in records)
    ):
        raise ValueError("V14 threshold fitting requires every known train OOF parent exactly once")
    gate = _select_binary_boundary(
        np.asarray([float(record["gate_dbb_probability"]) for record in records]),
        true_ids < 3,
        target=float(gate_target),
        positive_name="DBB_ACCEPT_EVIDENCE",
        negative_name="ROUTE_TO_CU_EVIDENCE",
    )
    dbb = _select_dbb_boundary(
        records,
        target=float(dbb_target),
        score_name=dbb_score_name,
    )
    cu = _select_binary_boundary(
        np.asarray([float(record["cu_clutter_probability"]) for record in records]),
        true_ids == 3,
        target=float(cu_target),
        positive_name="CLUTTER_ACCEPT_EVIDENCE",
        negative_name="NON_CLUTTER_KNOWN_EVIDENCE",
    )
    combined = summarize_known_oof_actions(
        records,
        gate_probability_threshold=gate["probability_threshold"],
        gate_margin_threshold=gate["margin_threshold"],
        dbb_probability_threshold=dbb["probability_threshold"],
        dbb_margin_threshold=dbb["margin_threshold"],
        cu_probability_threshold=cu["probability_threshold"],
        cu_margin_threshold=cu["margin_threshold"],
        dbb_score_name=dbb_score_name,
    )
    return {
        "fit_population": "known_only",
        "dbb_knownness_score_name": dbb_score_name,
        "known_parent_count": len(records),
        "unknown_threshold_fit_count": 0,
        "validation_used_for_threshold_fitting": False,
        "test_used": False,
        "gate": gate,
        "dbb_knownness": dbb,
        "clutter_acceptance": cu,
        "combined_known_action": combined,
        "interpretation": (
            "thresholds optimize only known-parent routing, known-class acceptance and known false "
            "rejection; they do not estimate or optimize Unknown detection"
        ),
    }


__all__ = [
    "OOF_ARTIFACT_RELATIVE",
    "OOF_ASSIGNMENT_RELATIVE",
    "OOF_FIXED_EPOCHS",
    "OOF_FOLD_CONTRACT_RELATIVE",
    "OOF_PREDICTION_SCHEMA",
    "canonical_sha256",
    "derive_oof_fold_configs",
    "fold_config_identities",
    "fold_prediction_path",
    "fold_record",
    "load_oof_fold_contract",
    "fit_known_only_oof_boundaries",
    "predict_oof_parent_final",
    "sequence_sha256",
    "sha256_file",
    "summarize_known_oof_actions",
    "verify_oof_dataset",
]
