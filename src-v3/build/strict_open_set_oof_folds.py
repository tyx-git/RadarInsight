from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from build.radar_ln3_normalization import build as build_normalization


VERSION = "RADAR-LN3-V14"
KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
INDEX_RELATIVE = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
BASE_CONTRACT_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-strict-open-set-data-contract.json")
BASE_STATS_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-train-normalization-stats.json")
ASSIGNMENT_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json")
FOLD_CONTRACT_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V14 OOF fold contract: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _write_prepared_fold_contract(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return "created"
    existing_text = path.read_text(encoding="utf-8")
    if existing_text == text:
        return "reused_identical"
    existing = json.loads(existing_text)
    existing_policy = dict(existing.get("execution_policy") or {})
    if (
        existing.get("schema_version")
        not in {
            "RADAR-LN3-V14-known-only-parent-grouped-oof-fold-contract-v1",
            "RADAR-LN3-V14-known-only-parent-grouped-oof-fold-contract-v2",
        }
        or existing.get("assignment_manifest_sha256") != payload["assignment_manifest_sha256"]
        or existing.get("source_index_sha256") != payload["source_index_sha256"]
        or existing.get("base_data_contract_sha256") != payload["base_data_contract_sha256"]
        or existing.get("base_normalization_stats_sha256")
        != payload["base_normalization_stats_sha256"]
        or existing_policy.get("formal_training") != "未执行"
        or existing_policy.get("oof_prediction") != "未执行"
        or existing_policy.get("threshold_fitting") != "未执行"
        or existing_policy.get("test") != "未执行"
    ):
        raise FileExistsError(f"refusing to overwrite an executed or unrelated V14 OOF contract: {path}")
    path.write_text(text, encoding="utf-8")
    return "upgraded_unexecuted_prepared_contract"


def _fold_data_contract(
    base_contract: dict[str, Any],
    *,
    fold_id: int,
    stats_relative: Path,
    stats_hash: str,
    stats: dict[str, Any],
    train_ids: list[str],
    held_out_ids: list[str],
    train_counts: dict[str, int],
    held_out_counts: dict[str, int],
) -> dict[str, Any]:
    payload = copy.deepcopy(base_contract)
    payload["contract_name"] = f"cq08_strict_open_set_oof_fold_{fold_id:02d}_training_v1"
    payload["contract_status"] = "oof_calibration_support_training_prepared_not_executed"
    payload["source_normalization_stats"] = _relative(stats_relative)
    payload["source_normalization_stats_sha256"] = stats_hash
    data_scope = dict(payload.get("data_scope") or {})
    data_scope.update({
        "oof_fold_id": fold_id,
        "optimizer_known_parent_count": len(train_ids),
        "held_out_known_parent_count": len(held_out_ids),
        "optimizer_label_parent_counts": train_counts,
        "held_out_label_parent_counts": held_out_counts,
        "unknown_optimizer_parent_count": 0,
        "unknown_held_out_parent_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
    })
    payload["data_scope"] = data_scope
    strict = dict(payload.get("strict_open_set_contract") or {})
    strict.update({
        "normalization_known_parent_count": len(train_ids),
        "normalization_excluded_unknown_parent_count": int(
            dict(stats.get("excluded_label_parent_counts") or {}).get(UNKNOWN_LABEL, 0)
        ),
        "normalization_selected_excluded_tensor_overlap_count": int(
            stats["selected_excluded_rd_tensor_path_overlap_count"]
        ),
        "validation_known_checkpoint_selection_parent_count": 0,
        "validation_unknown_post_freeze_development_audit_parent_count": 0,
        "oof_fold_train_parent_count": len(train_ids),
        "oof_fold_held_out_parent_count": len(held_out_ids),
        "oof_fold_held_out_checkpoint_selection_allowed": False,
        "oof_fold_validation_access_count": 0,
        "oof_fold_test_payload_access_count": 0,
    })
    payload["strict_open_set_contract"] = strict
    normalization = dict(payload.get("normalization_contract") or {})
    normalization.update({
        "path": _relative(stats_relative),
        "sha256": stats_hash,
        "known_parent_count": len(train_ids),
        "selected_label_parent_counts": stats["selected_label_parent_counts"],
        "forbidden_labels": stats["forbidden_labels"],
        "excluded_label_parent_counts": stats["excluded_label_parent_counts"],
        "selected_parent_ids_sha256": stats["selected_parent_ids_sha256"],
        "selected_rd_tensor_paths_sha256": stats["selected_rd_tensor_paths_sha256"],
        "excluded_parent_ids_sha256": stats["excluded_parent_ids_sha256"],
        "excluded_rd_tensor_paths_sha256": stats["excluded_rd_tensor_paths_sha256"],
        "rd_amplitude_center": float(stats["amplitude"]["center"]),
        "rd_amplitude_scale": float(stats["amplitude"]["scale"]),
        "allowed_parent_ids_count": int(stats["allowed_parent_ids_count"]),
        "allowed_parent_ids_sha256": stats["allowed_parent_ids_sha256"],
    })
    payload["normalization_contract"] = normalization
    feature = dict(payload.get("feature_contract") or {})
    feature["rd_normalization_stats"] = _relative(stats_relative)
    feature["rd_normalization_stats_sha256"] = stats_hash
    payload["feature_contract"] = feature
    training = dict(payload.get("training_contract") or {})
    training.update({
        "entry": "src-v3/train/strict_open_set_oof.py",
        "optimization_population": f"known train parents excluding OOF fold {fold_id:02d}",
        "optimizer_parent_count": len(train_ids),
        "unknown_optimizer_parent_count": 0,
        "epochs": 20,
        "early_stopping": False,
        "checkpoint_selection": "fixed final epoch; held-out fold is never used for checkpoint selection",
    })
    payload["training_contract"] = training
    payload["oof_fold_training_contract"] = {
        "fold_id": fold_id,
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "train_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "train_parent_ids_sha256": _sequence_sha256(train_ids),
        "held_out_parent_ids_sha256": _sequence_sha256(held_out_ids),
        "parent_overlap_count": 0,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "epochs": 20,
        "early_stopping": False,
        "held_out_checkpoint_selection": False,
        "held_out_prediction_time": "once_after_fixed_final_epoch",
        "training": "未执行",
        "prediction": "未执行",
    }
    reuse = dict(payload.get("reuse") or {})
    reuse.update({
        "base_v14_contract_reused": False,
        "fold_specific_normalization": True,
        "version_reason": (
            "V14 OOF support contract changes only the train-parent subset and fold-train "
            "normalization; it is not a deployment data version"
        ),
    })
    payload["reuse"] = reuse
    validation = dict(payload.get("validation_status") or {})
    validation.update({
        "formal_training": "不适用（OOF校准支持折模型）",
        "oof_fold_training": "未执行",
        "oof_prediction": "未执行",
        "oof_calibration": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    })
    payload["validation_status"] = validation
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    index_path = root / INDEX_RELATIVE
    base_contract_path = root / BASE_CONTRACT_RELATIVE
    base_stats_path = root / BASE_STATS_RELATIVE
    assignment_path = root / ASSIGNMENT_RELATIVE
    output_path = root / FOLD_CONTRACT_RELATIVE
    for path in (index_path, base_contract_path, base_stats_path, assignment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    if base_contract.get("schema_version") != VERSION:
        raise ValueError("V14 base strict-open-set contract schema changed")
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
        raise ValueError("V14 OOF assignment manifest violates known-only train-parent grouping")
    assignments = list(assignment.get("assignments") or [])
    fold_count = int(assignment.get("fold_count", 0))
    if len(assignments) != 1011 or fold_count < 2:
        raise ValueError("V14 OOF assignment count/fold count is invalid")
    if any(str(row.get("label_name")) == UNKNOWN_LABEL for row in assignments):
        raise ValueError("UnknownTarget entered the V14 OOF assignment manifest")
    all_ids = sorted(str(row["parent_sample_id"]) for row in assignments)
    if len(set(all_ids)) != 1011:
        raise ValueError("V14 OOF assignment repeats a known parent")
    fold_records: list[dict[str, Any]] = []
    for fold_id in range(fold_count):
        train_ids = sorted(
            str(row["parent_sample_id"])
            for row in assignments
            if int(row["fold_id"]) != fold_id
        )
        held_out_ids = sorted(
            str(row["parent_sample_id"])
            for row in assignments
            if int(row["fold_id"]) == fold_id
        )
        if set(train_ids) & set(held_out_ids) or sorted(train_ids + held_out_ids) != all_ids:
            raise ValueError(f"V14 OOF fold {fold_id} does not partition known train parents")
        stats_relative = Path(
            f"data/processed/RADAR-LN3-V14-cq08-known-only-oof-fold-{fold_id:02d}-train-normalization-stats.json"
        )
        stats_path = root / stats_relative
        if not stats_path.is_file():
            build_normalization(
                root,
                index_path,
                stats_path,
                "train",
                allowed_labels=KNOWN_LABELS,
                forbidden_labels=(UNKNOWN_LABEL,),
                allowed_parent_ids=tuple(train_ids),
                schema_version=VERSION,
                stats_name=f"cq08_known_only_oof_fold_{fold_id:02d}_train_normalization_v1",
                scope=(
                    f"V3 train split OOF fold {fold_id:02d}; CQ08 known fold-train parents only; "
                    "held-out fold and Unknown excluded; no validation or test statistics"
                ),
            )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            int(stats.get("parent_count", -1)) != len(train_ids)
            or int(stats.get("allowed_parent_ids_count", -1)) != len(train_ids)
            or stats.get("allowed_parent_ids_sha256") != _sequence_sha256(train_ids)
            or stats.get("forbidden_labels") != [UNKNOWN_LABEL]
            or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
            or set(stats.get("selected_label_parent_counts") or {}) != set(KNOWN_LABELS)
        ):
            raise ValueError(f"V14 OOF fold {fold_id} normalization is not fold-train known-only")
        held_out_counts = {
            label: sum(
                int(row["fold_id"]) == fold_id and str(row["label_name"]) == label
                for row in assignments
            )
            for label in KNOWN_LABELS
        }
        train_counts = {
            label: sum(
                int(row["fold_id"]) != fold_id and str(row["label_name"]) == label
                for row in assignments
            )
            for label in KNOWN_LABELS
        }
        fold_contract_relative = Path(
            f"data/processed/RADAR-LN3-V14-cq08-known-only-oof-fold-{fold_id:02d}-data-contract.json"
        )
        fold_contract_path = root / fold_contract_relative
        fold_contract_payload = _fold_data_contract(
            base_contract,
            fold_id=fold_id,
            stats_relative=stats_relative,
            stats_hash=_sha256(stats_path),
            stats=stats,
            train_ids=train_ids,
            held_out_ids=held_out_ids,
            train_counts=train_counts,
            held_out_counts=held_out_counts,
        )
        _write_new_or_identical(fold_contract_path, fold_contract_payload)
        fold_contract_hash = _sha256(fold_contract_path)
        fold_records.append({
            "fold_id": fold_id,
            "train_parent_count": len(train_ids),
            "held_out_parent_count": len(held_out_ids),
            "train_label_parent_counts": train_counts,
            "held_out_label_parent_counts": held_out_counts,
            "train_parent_ids": train_ids,
            "held_out_parent_ids": held_out_ids,
            "train_parent_ids_sha256": _sequence_sha256(train_ids),
            "held_out_parent_ids_sha256": _sequence_sha256(held_out_ids),
            "parent_overlap_count": 0,
            "unknown_train_parent_count": 0,
            "unknown_held_out_parent_count": 0,
            "validation_parent_count": 0,
            "test_parent_count": 0,
            "normalization_stats_path": _relative(stats_relative),
            "normalization_stats_sha256": _sha256(stats_path),
            "normalization_parent_count": int(stats["parent_count"]),
            "normalization_rd_amplitude_center": float(stats["amplitude"]["center"]),
            "normalization_rd_amplitude_scale": float(stats["amplitude"]["scale"]),
            "held_out_used_for_normalization": False,
            "data_contract_path": _relative(fold_contract_relative),
            "data_contract_sha256": fold_contract_hash,
        })
    payload = {
        "schema_version": f"{VERSION}-known-only-parent-grouped-oof-fold-contract-v2",
        "assignment_manifest": _relative(ASSIGNMENT_RELATIVE),
        "assignment_manifest_sha256": _sha256(assignment_path),
        "source_index": _relative(INDEX_RELATIVE),
        "source_index_sha256": _sha256(index_path),
        "base_data_contract": _relative(BASE_CONTRACT_RELATIVE),
        "base_data_contract_sha256": _sha256(base_contract_path),
        "base_normalization_stats": _relative(BASE_STATS_RELATIVE),
        "base_normalization_stats_sha256": _sha256(base_stats_path),
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "fold_count": fold_count,
        "known_parent_count": 1011,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "folds": fold_records,
        "execution_policy": {
            "epochs_per_fold": 20,
            "early_stopping": False,
            "held_out_checkpoint_selection": False,
            "held_out_prediction_time": "once_after_fixed_final_epoch",
            "formal_training": "未执行",
            "oof_prediction": "未执行",
            "threshold_fitting": "未执行",
            "test": "未执行",
        },
    }
    write_status = _write_prepared_fold_contract(output_path, payload)
    return {
        "status": "passed",
        "schema_version": payload["schema_version"],
        "contract_path": str(output_path),
        "contract_sha256": _sha256(output_path),
        "write_status": write_status,
        "fold_count": fold_count,
        "known_parent_count": 1011,
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "fold_training": "未执行",
        "threshold_fitting": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fold-train-only normalization and provenance for V14 parent-grouped OOF."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
