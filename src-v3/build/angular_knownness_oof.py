from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from build.radar_ln3_normalization import build as build_normalization
from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    KNOWN_BASE_LABELS,
    UNKNOWN_BASE_LABEL,
)


VERSION = ANGULAR_KNOWNNESS_SCHEMA_VERSION
INDEX_RELATIVE = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
BASE_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V15-cq08-angular-knownness-experiment-contract.json"
)
ASSIGNMENT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"
)
V14_FOLD_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"
)
NESTED_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V15-cq08-known-only-nested-oof-contract.json"
)
NESTED_CONTRACT_SCHEMA = "RADAR-LN3-V15-known-only-nested-parent-grouped-oof-contract-v1"
FOLD_COUNT = 5


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


def _relative(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V15 OOF artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _nested_contract_equal(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Accept the one pre-freeze V15 contract with runtime-only status fields.

    Early preparation wrote ``write_status`` into each nested-stage record.
    That field describes whether the builder happened to create or reuse a
    file; it is not a data, split, model, or OOF execution semantic.  New
    payloads intentionally omit it, while this compatibility comparison keeps
    the already frozen, hash-recorded artifact read-only.
    """

    normalized = copy.deepcopy(dict(existing))
    for fold in list(normalized.get("folds") or []):
        if not isinstance(fold, dict):
            return False
        for stage in ("inner_train", "outer_retrain"):
            value = fold.get(stage)
            if isinstance(value, dict):
                value.pop("write_status", None)
    return normalized == dict(expected)


def _label_counts(
    assignments: list[dict[str, Any]],
    parent_ids: list[str],
) -> dict[str, int]:
    selected = set(parent_ids)
    counts = {
        label: sum(
            str(row["parent_sample_id"]) in selected and str(row["label_name"]) == label
            for row in assignments
        )
        for label in KNOWN_BASE_LABELS
    }
    if any(value <= 0 for value in counts.values()) or sum(counts.values()) != len(parent_ids):
        raise ValueError("V15 nested OOF population lacks one or more known labels")
    return counts


def _normalization_payload_fields(
    stats: Mapping[str, Any],
    *,
    relative_path: Path,
    absolute_path: Path,
) -> dict[str, Any]:
    return {
        "path": _relative(relative_path),
        "sha256": sha256_file(absolute_path),
        "schema_version": str(stats["schema_version"]),
        "source_split": "train",
        "known_parent_count": int(stats["parent_count"]),
        "selected_label_parent_counts": dict(stats["selected_label_parent_counts"]),
        "forbidden_labels": list(stats["forbidden_labels"]),
        "excluded_label_parent_counts": dict(stats["excluded_label_parent_counts"]),
        "selected_parent_ids_sha256": str(stats["selected_parent_ids_sha256"]),
        "selected_rd_tensor_paths_sha256": str(stats["selected_rd_tensor_paths_sha256"]),
        "excluded_parent_ids_sha256": str(stats["excluded_parent_ids_sha256"]),
        "excluded_rd_tensor_paths_sha256": str(stats["excluded_rd_tensor_paths_sha256"]),
        "rd_amplitude_center": float(stats["amplitude"]["center"]),
        "rd_amplitude_scale": float(stats["amplitude"]["scale"]),
    }


def _stage_contract(
    base_contract: Mapping[str, Any],
    *,
    outer_fold_id: int,
    stage: str,
    train_ids: list[str],
    outer_held_out_ids: list[str],
    inner_selection_ids: list[str],
    train_counts: dict[str, int],
    stats_relative: Path,
    stats_path: Path,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in {"inner_selection", "outer_retrain"}:
        raise ValueError(f"unsupported V15 nested OOF stage: {stage}")
    payload = copy.deepcopy(dict(base_contract))
    stats_hash = sha256_file(stats_path)
    train_hash = sequence_sha256(train_ids)
    outer_hash = sequence_sha256(outer_held_out_ids)
    inner_hash = sequence_sha256(inner_selection_ids)
    payload["contract_name"] = (
        f"cq08_v15_nested_oof_outer_{outer_fold_id:02d}_{stage}_v1"
    )
    payload["contract_status"] = "nested_oof_prepared_formal_execution_not_started"
    payload["source_normalization_stats"] = _relative(stats_relative)
    payload["source_normalization_stats_sha256"] = stats_hash
    data_scope = dict(payload["data_scope"])
    data_scope.update(
        {
            "nested_oof_outer_fold_id": outer_fold_id,
            "nested_oof_stage": stage,
            "optimizer_known_parent_count": len(train_ids),
            "optimizer_label_parent_counts": train_counts,
            "outer_held_out_known_parent_count": len(outer_held_out_ids),
            "inner_selection_known_parent_count": len(inner_selection_ids),
            "unknown_optimizer_parent_count": 0,
            "unknown_selection_parent_count": 0,
            "unknown_outer_held_out_parent_count": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
        }
    )
    payload["data_scope"] = data_scope
    strict = dict(payload["strict_open_set_contract"])
    strict.update(
        {
            "normalization_known_parent_count": len(train_ids),
            "normalization_excluded_unknown_parent_count": int(
                dict(stats.get("excluded_label_parent_counts") or {}).get(
                    UNKNOWN_BASE_LABEL, 0
                )
            ),
            "normalization_selected_excluded_tensor_overlap_count": int(
                stats["selected_excluded_rd_tensor_path_overlap_count"]
            ),
            "validation_known_checkpoint_selection_parent_count": 0,
            "validation_unknown_post_freeze_development_audit_parent_count": 0,
            "nested_oof_stage": stage,
            "nested_oof_optimizer_parent_count": len(train_ids),
            "nested_oof_inner_selection_parent_count": len(inner_selection_ids),
            "nested_oof_outer_held_out_parent_count": len(outer_held_out_ids),
            "nested_oof_outer_held_out_normalization_allowed": False,
            "nested_oof_outer_held_out_checkpoint_selection_allowed": False,
            "nested_oof_outer_held_out_forward_before_retrain_complete": False,
        }
    )
    payload["strict_open_set_contract"] = strict
    normalization = _normalization_payload_fields(
        stats,
        relative_path=stats_relative,
        absolute_path=stats_path,
    )
    payload["normalization_contract"] = normalization
    feature = dict(payload["feature_contract"])
    feature["rd_normalization_stats"] = _relative(stats_relative)
    feature["rd_normalization_stats_sha256"] = stats_hash
    payload["feature_contract"] = feature
    training = dict(payload["training_contract"])
    training.update(
        {
            "entry": "src-v3/train/angular_knownness_oof.py",
            "optimization_population": f"known train parents in V15 {stage}",
            "optimizer_parent_count": len(train_ids),
            "unknown_optimizer_parent_count": 0,
            "epochs": 20,
            "early_stopping": False,
            "nested_oof_stage": stage,
            "outer_held_out_checkpoint_selection": False,
        }
    )
    payload["training_contract"] = training
    oof = dict(payload["oof_calibration_contract"])
    oof.update(
        {
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold",
            "epoch_protocol": "inner_parent_selection_then_outer_train_retrain",
            "outer_held_out_used_for_normalization": False,
            "outer_held_out_used_for_checkpoint_selection": False,
            "outer_held_out_prediction_pass_count": 1,
            "unknown_oof_count": 0,
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
        }
    )
    payload["oof_calibration_contract"] = oof
    payload["oof_fold_training_contract"] = {
        "fold_id": outer_fold_id,
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "nested_oof_stage": stage,
        "train_parent_count": len(train_ids),
        "train_parent_ids_sha256": train_hash,
        "held_out_parent_count": len(outer_held_out_ids),
        "held_out_parent_ids_sha256": outer_hash,
        "inner_selection_parent_count": len(inner_selection_ids),
        "inner_selection_parent_ids_sha256": inner_hash,
        "parent_overlap_count": 0,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "epochs": 20,
        "early_stopping": False,
        "held_out_checkpoint_selection": False,
        "outer_held_out_forward_count": 0,
        "training": "未执行",
        "prediction": "未执行",
    }
    reuse = dict(payload["reuse"])
    reuse.update(
        {
            "v14_oof_assignment_reused_read_only": True,
            "v14_outer_train_normalization_reused_read_only": stage == "outer_retrain",
            "inner_train_normalization_is_stage_specific": stage == "inner_selection",
        }
    )
    payload["reuse"] = reuse
    validation = dict(payload["validation_status"])
    validation.update(
        {
            "nested_oof_stage": "未执行",
            "oof_calibration": "未执行",
            "development_unknown_audit": "未执行",
            "test": "未执行",
            "p95": "未执行",
        }
    )
    payload["validation_status"] = validation
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "index": root / INDEX_RELATIVE,
        "base_contract": root / BASE_CONTRACT_RELATIVE,
        "assignment": root / ASSIGNMENT_RELATIVE,
        "v14_fold_contract": root / V14_FOLD_CONTRACT_RELATIVE,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    assignment = json.loads(paths["assignment"].read_text(encoding="utf-8"))
    v14_fold_contract = json.loads(
        paths["v14_fold_contract"].read_text(encoding="utf-8")
    )
    base_contract = json.loads(paths["base_contract"].read_text(encoding="utf-8"))
    assignments = list(assignment.get("assignments") or [])
    if (
        base_contract.get("schema_version") != VERSION
        or int(assignment.get("fold_count", -1)) != FOLD_COUNT
        or int(assignment.get("known_parent_count", -1)) != 1011
        or int(assignment.get("unknown_oof_parent_count", -1)) != 0
        or len(assignments) != 1011
        or any(str(row.get("label_name")) == UNKNOWN_BASE_LABEL for row in assignments)
    ):
        raise ValueError("V15 nested OOF source contracts violate the known-only protocol")
    v14_folds = {
        int(row["fold_id"]): dict(row)
        for row in list(v14_fold_contract.get("folds") or [])
    }
    if set(v14_folds) != set(range(FOLD_COUNT)):
        raise ValueError("V14 outer folds do not provide exactly five partitions")
    all_ids = sorted(str(row["parent_sample_id"]) for row in assignments)
    if len(set(all_ids)) != 1011 or sequence_sha256(all_ids) != assignment["known_parent_ids_sha256"]:
        raise ValueError("V14 assignment no longer covers every known train parent exactly once")

    fold_records: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for outer_fold_id in range(FOLD_COUNT):
        inner_selection_fold_id = (outer_fold_id + 1) % FOLD_COUNT
        outer_held_out_ids = sorted(
            str(row["parent_sample_id"])
            for row in assignments
            if int(row["fold_id"]) == outer_fold_id
        )
        inner_selection_ids = sorted(
            str(row["parent_sample_id"])
            for row in assignments
            if int(row["fold_id"]) == inner_selection_fold_id
        )
        inner_train_ids = sorted(
            str(row["parent_sample_id"])
            for row in assignments
            if int(row["fold_id"]) not in {outer_fold_id, inner_selection_fold_id}
        )
        outer_train_ids = sorted(set(all_ids) - set(outer_held_out_ids))
        populations = [
            set(inner_train_ids),
            set(inner_selection_ids),
            set(outer_held_out_ids),
        ]
        if (
            any(populations[i] & populations[j] for i in range(3) for j in range(i + 1, 3))
            or sorted(inner_train_ids + inner_selection_ids + outer_held_out_ids) != all_ids
            or sorted(inner_train_ids + inner_selection_ids) != outer_train_ids
        ):
            raise ValueError(f"V15 nested OOF outer fold {outer_fold_id} is not disjoint")

        inner_stats_relative = Path(
            "data/processed/"
            f"RADAR-LN3-V15-cq08-nested-oof-outer-{outer_fold_id:02d}-inner-train-normalization-stats.json"
        )
        inner_stats_path = root / inner_stats_relative
        if not inner_stats_path.is_file():
            build_normalization(
                root,
                paths["index"],
                inner_stats_path,
                "train",
                allowed_labels=tuple(KNOWN_BASE_LABELS),
                forbidden_labels=(UNKNOWN_BASE_LABEL,),
                allowed_parent_ids=tuple(inner_train_ids),
                schema_version=VERSION,
                stats_name=f"cq08_v15_nested_oof_outer_{outer_fold_id:02d}_inner_train_v1",
                scope=(
                    f"V3 train split; V15 outer fold {outer_fold_id:02d} inner-train known parents only; "
                    "inner-selection, outer-held-out, Unknown, validation and test excluded"
                ),
            )
        inner_stats = json.loads(inner_stats_path.read_text(encoding="utf-8"))
        v14_outer = v14_folds[outer_fold_id]
        outer_stats_relative = Path(str(v14_outer["normalization_stats_path"]))
        outer_stats_path = root / outer_stats_relative
        if (
            not outer_stats_path.is_file()
            or sha256_file(outer_stats_path) != v14_outer["normalization_stats_sha256"]
        ):
            raise ValueError(f"V14 outer-train normalization changed for fold {outer_fold_id}")
        outer_stats = json.loads(outer_stats_path.read_text(encoding="utf-8"))

        stage_outputs: dict[str, Any] = {}
        for stage, train_ids, stats_relative, stats in (
            ("inner_selection", inner_train_ids, inner_stats_relative, inner_stats),
            ("outer_retrain", outer_train_ids, outer_stats_relative, outer_stats),
        ):
            train_counts = _label_counts(assignments, train_ids)
            contract_relative = Path(
                "data/processed/"
                f"RADAR-LN3-V15-cq08-nested-oof-outer-{outer_fold_id:02d}-{stage.replace('_', '-')}-data-contract.json"
            )
            contract_path = root / contract_relative
            contract_payload = _stage_contract(
                base_contract,
                outer_fold_id=outer_fold_id,
                stage=stage,
                train_ids=train_ids,
                outer_held_out_ids=outer_held_out_ids,
                inner_selection_ids=inner_selection_ids,
                train_counts=train_counts,
                stats_relative=stats_relative,
                stats_path=root / stats_relative,
                stats=stats,
            )
            status = _write_new_or_identical(contract_path, contract_payload)
            stage_outputs[stage] = {
                "train_parent_count": len(train_ids),
                "train_parent_ids": train_ids,
                "train_parent_ids_sha256": sequence_sha256(train_ids),
                "train_label_parent_counts": train_counts,
                "normalization_stats_path": _relative(stats_relative),
                "normalization_stats_sha256": sha256_file(root / stats_relative),
                "data_contract_path": _relative(contract_relative),
                "data_contract_sha256": sha256_file(contract_path),
            }
            generated.append({"path": str(contract_path), "write_status": status})
        fold_records.append(
            {
                "outer_fold_id": outer_fold_id,
                "inner_selection_fold_id": inner_selection_fold_id,
                "outer_held_out_parent_count": len(outer_held_out_ids),
                "outer_held_out_parent_ids": outer_held_out_ids,
                "outer_held_out_parent_ids_sha256": sequence_sha256(outer_held_out_ids),
                "inner_selection_parent_count": len(inner_selection_ids),
                "inner_selection_parent_ids": inner_selection_ids,
                "inner_selection_parent_ids_sha256": sequence_sha256(inner_selection_ids),
                "inner_train": stage_outputs["inner_selection"],
                "outer_retrain": stage_outputs["outer_retrain"],
                "unknown_training_exposure": 0,
                "validation_access_count": 0,
                "test_payload_access_count": 0,
                "outer_held_out_used_for_normalization": False,
                "outer_held_out_used_for_epoch_selection": False,
                "outer_held_out_prediction_pass_count": 1,
            }
        )

    nested_payload = {
        "schema_version": NESTED_CONTRACT_SCHEMA,
        "model_schema_version": VERSION,
        "source_assignment_manifest": _relative(ASSIGNMENT_RELATIVE),
        "source_assignment_manifest_sha256": sha256_file(paths["assignment"]),
        "source_v14_outer_fold_contract": _relative(V14_FOLD_CONTRACT_RELATIVE),
        "source_v14_outer_fold_contract_sha256": sha256_file(paths["v14_fold_contract"]),
        "source_experiment_contract": _relative(BASE_CONTRACT_RELATIVE),
        "source_experiment_contract_sha256": sha256_file(paths["base_contract"]),
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "prediction_protocol": "nested_out_of_fold",
        "epoch_protocol": "inner_parent_selection_then_outer_train_retrain",
        "inner_selection_rule": "next frozen V14 fold modulo five",
        "fold_count": FOLD_COUNT,
        "known_parent_count": 1011,
        "known_parent_ids_sha256": sequence_sha256(all_ids),
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "outer_held_out_used_for_normalization": False,
        "outer_held_out_used_for_epoch_selection": False,
        "outer_held_out_prediction_pass_count": 1,
        "folds": fold_records,
        "execution_status": {
            "control_nested_oof": "未执行",
            "treatment_nested_oof": "未执行",
            "threshold_fitting": "未执行",
            "development_unknown_audit": "未执行",
            "test": "未执行",
            "p95": "未执行",
        },
    }
    nested_path = root / NESTED_CONTRACT_RELATIVE
    if nested_path.exists():
        existing_nested = json.loads(nested_path.read_text(encoding="utf-8"))
        if existing_nested == nested_payload:
            nested_status = "reused_identical"
        elif _nested_contract_equal(existing_nested, nested_payload):
            nested_status = "reused_legacy_runtime_status_only"
        else:
            raise FileExistsError(
                f"refusing to overwrite a different V15 OOF artifact: {nested_path}"
            )
    else:
        nested_status = _write_new_or_identical(nested_path, nested_payload)
    return {
        "status": "passed",
        "schema_version": NESTED_CONTRACT_SCHEMA,
        "nested_contract_path": str(nested_path),
        "nested_contract_sha256": sha256_file(nested_path),
        "write_status": nested_status,
        "fold_count": FOLD_COUNT,
        "known_parent_count": 1011,
        "unknown_training_exposure": 0,
        "generated_stage_contract_count": len(generated),
        "formal_training": "未执行",
        "nested_oof": "未执行",
        "threshold_fitting": "未执行",
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V15 nested parent-grouped OOF contracts without training."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
