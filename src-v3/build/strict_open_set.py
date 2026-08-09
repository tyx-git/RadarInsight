from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build.radar_ln3_normalization import build as build_normalization
from radar_ln3.data import RD_QUALITY_NAMES, RD_QUALITY_SCHEMA


VERSION = "RADAR-LN3-V14"
KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
INDEX_RELATIVE = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
SOURCE_CONTRACT_RELATIVE = Path("data/processed/RADAR-LN3-V13-cq08-hierarchical-cascade-data-contract.json")
STATS_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-train-normalization-stats.json")
CONTRACT_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-strict-open-set-data-contract.json")
CONFIG_ROOT_RELATIVE = Path("config/RADAR-LN3")
FORMAL_ARTIFACT_RELATIVE = Path("artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path).replace("\\", "/")


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V14 artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _parent_scope(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("V3 index contains a parent with multiple labels")
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("V3 index contains a parent crossing splits")
    return (
        frame[["parent_sample_id", "split_name", "label_name", "source_name", "rd_tensor_path"]]
        .drop_duplicates("parent_sample_id")
        .assign(
            parent_sample_id=lambda value: value["parent_sample_id"].astype(str),
            split_name=lambda value: value["split_name"].astype(str),
            label_name=lambda value: value["label_name"].astype(str),
            source_name=lambda value: value["source_name"].astype(str),
            rd_tensor_path=lambda value: value["rd_tensor_path"].astype(str),
        )
    )


def _counts(parent_frame: pd.DataFrame, split: str) -> dict[str, int]:
    subset = parent_frame[parent_frame["split_name"].eq(split)]
    counts = subset.groupby("label_name").size()
    return {label: int(counts.get(label, 0)) for label in (*KNOWN_LABELS, UNKNOWN_LABEL)}


def _inverse_frequency_weights(counts: tuple[int, ...]) -> list[float]:
    total = sum(counts)
    classes = len(counts)
    if total <= 0 or any(value <= 0 for value in counts):
        raise ValueError("class-weight counts must all be positive")
    return [total / (classes * value) for value in counts]


def _data_contract(
    *,
    index_hash: str,
    source_contract_hash: str,
    stats_hash: str,
    stats: dict[str, Any],
    parent_frame: pd.DataFrame,
    point_frame: pd.DataFrame,
) -> dict[str, Any]:
    split_parent_counts = {
        split: int(parent_frame["split_name"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    split_point_counts = {
        split: int(point_frame["split_name"].astype(str).eq(split).sum())
        for split in ("train", "val", "test")
    }
    label_parent_counts = {
        label: int(parent_frame["label_name"].eq(label).sum())
        for label in (*KNOWN_LABELS, UNKNOWN_LABEL)
    }
    train_counts = _counts(parent_frame, "train")
    validation_counts = _counts(parent_frame, "val")
    test_counts = _counts(parent_frame, "test")
    known_train_count = sum(train_counts[label] for label in KNOWN_LABELS)
    known_validation_count = sum(validation_counts[label] for label in KNOWN_LABELS)
    if train_counts != {
        "DroneTarget": 488,
        "BirdTarget": 243,
        "BalloonTarget": 245,
        "ClutterTarget": 35,
        "UnknownTarget": 70,
    }:
        raise ValueError(f"V14 frozen train parent counts changed: {train_counts}")
    if validation_counts != {
        "DroneTarget": 105,
        "BirdTarget": 52,
        "BalloonTarget": 52,
        "ClutterTarget": 8,
        "UnknownTarget": 15,
    }:
        raise ValueError(f"V14 frozen validation parent counts changed: {validation_counts}")
    return {
        "schema_version": VERSION,
        "contract_name": "cq08_tr_rd_known_only_strict_open_set_liquid_cascade_v1",
        "contract_status": "implementation_prepared_formal_training_not_executed",
        "source_index": _relative(INDEX_RELATIVE),
        "source_index_sha256": index_hash,
        "source_contract": _relative(SOURCE_CONTRACT_RELATIVE),
        "source_contract_sha256": source_contract_hash,
        "source_normalization_stats": _relative(STATS_RELATIVE),
        "source_normalization_stats_sha256": stats_hash,
        "source_purity": {
            "allowed_sources": ["CQ08"],
            "excluded_sources": ["raw", "Excel", "RD-only"],
        },
        "data_scope": {
            "index_schema_version": "RADAR-LN3-V3",
            "parent_count": int(len(parent_frame)),
            "point_count": int(len(point_frame)),
            "split_parent_counts": split_parent_counts,
            "split_point_counts": split_point_counts,
            "label_parent_counts": label_parent_counts,
            "train_label_parent_counts": train_counts,
            "validation_label_parent_counts": validation_counts,
            "test_label_parent_counts_metadata_only": test_counts,
            "known_train_parent_count": known_train_count,
            "known_validation_parent_count": known_validation_count,
            "unknown_train_source_parent_count": train_counts[UNKNOWN_LABEL],
            "unknown_validation_audit_parent_count": validation_counts[UNKNOWN_LABEL],
            "index_rows_changed": False,
            "split_changed": False,
            "rd_tensor_payloads_changed": False,
            "test_payload_access_count": 0,
        },
        "strict_open_set_contract": {
            "known_training_labels": list(KNOWN_LABELS),
            "unknown_label": UNKNOWN_LABEL,
            "unknown_is_model_class": False,
            "unknown_is_training_target": False,
            "unknown_is_normalization_population": False,
            "unknown_is_sampler_population": False,
            "unknown_is_optimizer_population": False,
            "unknown_is_checkpoint_ranking_population": False,
            "unknown_is_oof_population": False,
            "unknown_is_threshold_fitting_population": False,
            "unknown_training_exposure": 0,
            "normalization_known_parent_count": int(stats["parent_count"]),
            "normalization_excluded_unknown_parent_count": int(stats["excluded_parent_count"]),
            "normalization_selected_excluded_tensor_overlap_count": int(
                stats["selected_excluded_rd_tensor_path_overlap_count"]
            ),
            "validation_known_checkpoint_selection_parent_count": known_validation_count,
            "validation_unknown_post_freeze_development_audit_parent_count": validation_counts[UNKNOWN_LABEL],
            "validation_unknown_threshold_fitting_allowed": False,
            "validation_unknown_checkpoint_selection_allowed": False,
            "test_is_sealed": True,
            "test_evaluation": "未执行",
        },
        "normalization_contract": {
            "path": _relative(STATS_RELATIVE),
            "sha256": stats_hash,
            "schema_version": VERSION,
            "source_split": "train",
            "known_parent_count": int(stats["parent_count"]),
            "selected_label_parent_counts": stats["selected_label_parent_counts"],
            "forbidden_labels": stats["forbidden_labels"],
            "excluded_label_parent_counts": stats["excluded_label_parent_counts"],
            "selected_parent_ids_sha256": stats["selected_parent_ids_sha256"],
            "selected_rd_tensor_paths_sha256": stats["selected_rd_tensor_paths_sha256"],
            "excluded_parent_ids_sha256": stats["excluded_parent_ids_sha256"],
            "excluded_rd_tensor_paths_sha256": stats["excluded_rd_tensor_paths_sha256"],
            "rd_amplitude_center": float(stats["amplitude"]["center"]),
            "rd_amplitude_scale": float(stats["amplitude"]["scale"]),
        },
        "time_contract": {
            "raw_precision_seconds": 1.0,
            "dt_field": "tr_dt_seconds",
            "dt_valid_field": "tr_dt_valid",
            "dt_reason_field": "tr_dt_reason",
            "reason_encoding": {
                "padding": 0,
                "first_event_no_predecessor": 1,
                "zero_delta_at_second_precision": 2,
                "positive_delta_at_second_precision": 3,
            },
            "dynamics_only": True,
            "first_event_rule": "initialize from the current observation without claiming elapsed physical time",
            "zero_delta_rule": "apply bounded current-observation update without time decay",
            "positive_delta_rule": "apply continuous-time liquid update with the positive second-precision delta",
            "padding_rule": "never update state",
            "no_high_precision_dt_recovery_claim": True,
        },
        "feature_contract": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "rd_quality_schema": RD_QUALITY_SCHEMA,
            "rd_quality_names": list(RD_QUALITY_NAMES),
            "rd_frame_shape": [2, 21, 384],
            "rd_normalization_stats": _relative(STATS_RELATIVE),
            "rd_normalization_stats_sha256": stats_hash,
            "excluded_model_inputs": [
                "label_name",
                "path",
                "filename",
                "parent_sample_id",
                "track_id",
                "split_name",
                "UnknownTarget indicator",
            ],
            "dt_reason_input_scope": "liquid dynamics control only; never concatenated into classification Fusion",
        },
        "model_io": {
            "tr_features": "[B,L,14]",
            "tr_missing_mask": "[B,L,11]",
            "tr_dt_seconds": "[B,L]",
            "tr_dt_valid": "[B,L] bool",
            "tr_dt_reason": "[B,L] int64 dynamics-only",
            "rd_frames": "[B,L,K,2,21,384]",
            "rd_frame_mask": "[B,L,K] bool",
            "rd_quality": "[B,L,7] causal-current-event-v1",
            "tr_padding_mask": "[B,L] bool",
            "gate_logits": "[B,L,2] DBB_ACCEPT_EVIDENCE/ROUTE_TO_CU_EVIDENCE",
            "dbb_logits": "[B,L,3] Drone/Bird/Balloon known conditional evidence",
            "cu_logits": "[B,L,2] CLUTTER_ACCEPT_EVIDENCE/NON_CLUTTER_KNOWN_EVIDENCE",
            "open_set_audit_evidence": "[B,L,5] four known evidence values plus UNRESOLVED_OPEN_SET_EVIDENCE; not Unknown posterior",
            "hierarchical_action": "DBB_PENDING|DBB_ACCEPTED|CU_PENDING|CLUTTER_ACCEPTED|UNKNOWN_REJECTED",
        },
        "training_contract": {
            "entry": "src-v3/train/cascade.py",
            "single_task_entry_forbidden": True,
            "optimization_population": "known train parents only",
            "optimizer_parent_count": known_train_count,
            "unknown_optimizer_parent_count": 0,
            "gate_target": "DBB known versus Clutter route evidence; Unknown absent",
            "dbb_target": "Drone/Bird/Balloon only",
            "cu_target": "Clutter accept versus DBB known non-Clutter evidence; Unknown absent",
            "main_loss": "parent-final known-only strict-open-set NLL",
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "checkpoint_selection": "known validation Gate/DBB/Clutter-acceptance/Drone-Bird guardrails only",
            "unknown_checkpoint_selection_exposure": 0,
            "epochs": 20,
            "early_stopping": False,
            "pretraining": "not_required_and_not_used",
        },
        "oof_calibration_contract": {
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold",
            "unknown_oof_count": 0,
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
            "thresholds": [
                "gate_accept_dbb_probability_and_margin",
                "dbb_accept_known_probability_and_margin",
                "cu_accept_clutter_probability_and_margin",
            ],
            "status": "未执行",
        },
        "action_contract": {
            "gate_action": "accept DBB route when calibrated; otherwise route to CU",
            "dbb_action": "accept Drone/Bird/Balloon only when DBB knownness is calibrated and passes",
            "dbb_knownness_failure": "UNKNOWN_REJECTED",
            "cu_action": "accept Clutter only when calibrated and passes",
            "cu_clutter_failure": "UNKNOWN_REJECTED",
            "unknown_is_rejection_action_only": True,
            "unknown_is_probability_output": False,
            "uncalibrated_mode": "named evidence audit only; no Coverage/FAR/FRR/reject/stable claim",
        },
        "reuse": {
            "index_reused": True,
            "split_reused": True,
            "rd_tensors_reused_read_only": True,
            "normalization_reused": False,
            "v13_checkpoint_compatibility": "forbidden",
            "v13_calibration_compatibility": "forbidden",
            "version_reason": "V14 changes train population, normalization, task labels, loss, checkpoint ranking, evidence semantics and three-boundary rejection while reusing V3 rows/split/TR/RD tensors",
        },
        "validation_status": {
            "focused": "未执行",
            "bounded_cpu_smoke": "未执行",
            "bounded_cuda_smoke": "未执行",
            "formal_training": "未执行",
            "oof_calibration": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "test": "未执行",
            "p95": "未执行",
        },
    }


def _base_model(num_classes: int) -> dict[str, Any]:
    return {
        "tr_value_dim": 14,
        "tr_missing_dim": 11,
        "rd_quality_dim": 7,
        "hidden_dim": 48,
        "rd_embedding_dim": 24,
        "liquid_hidden_dim": 48,
        "rd_height": 21,
        "rd_width": 384,
        "liquid_tau_min_seconds": 0.05,
        "liquid_tau_max_seconds": 30.0,
        "liquid_zero_dt_event_floor_seconds": 0.001,
        "liquid_dt_max_seconds": 120.0,
        "dt_valid_dim": 1,
        "dt_valid_feature_mode": "dynamics_only",
        "dt_reason_mode": "v3_reason_aware_dynamics_only",
        "liquid_invalid_dt_update_mode": "bounded_observation",
        "liquid_invalid_dt_observation_gain": 0.25,
        "num_classes": num_classes,
        "liquid_state_normalization": "recurrent",
        "rd_frame_aggregation": "masked_mean",
    }


def _config(
    *,
    task_name: str,
    class_names: list[str],
    label_to_id: dict[str, int],
    class_weights: list[float],
    num_classes: int,
    index_hash: str,
    contract_hash: str,
    stats_hash: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    task_slug = {
        "dbb_cu_gate": "dbb-cu-gate",
        "dbb_specialist": "dbb-specialist",
        "cu_specialist": "cu-specialist",
    }[task_name]
    config_name = f"{VERSION}-cq08-liquid-strict-open-set-{task_slug}"
    artifact = FORMAL_ARTIFACT_RELATIVE / task_name
    return {
        "schema_version": VERSION,
        "config_name": config_name,
        "task": {
            "name": task_name,
            "class_names": class_names,
            "label_to_id": label_to_id,
        },
        "data": {
            "index_path": _relative(INDEX_RELATIVE),
            "contract_path": _relative(CONTRACT_RELATIVE),
            "normalization_stats_path": _relative(STATS_RELATIVE),
            "train_split_name": "train",
            "validation_split_name": "val",
            "test_split_name": "test",
            "index_schema_version": "RADAR-LN3-V3",
            "contract_schema_version": VERSION,
            "rd_quality_schema": RD_QUALITY_SCHEMA,
            "expected_index_sha256": index_hash,
            "expected_contract_sha256": contract_hash,
            "expected_stats_sha256": stats_hash,
            "max_sequence_length": 29,
            "max_rd_frames_per_point": 12,
            "rd_amplitude_center": float(stats["amplitude"]["center"]),
            "rd_amplitude_scale": float(stats["amplitude"]["scale"]),
            "exclude_anomalous_parents": False,
        },
        "model": _base_model(num_classes),
        "train": {
            "batch_size": 2,
            "learning_rate": 0.0001,
            "seed": 42,
            "device": "cuda",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 100.0,
            "checkpoint_selection_metric": "known_only_parent_route_guarded",
            "checkpoint_path": _relative(artifact / "best_model.pt"),
            "last_checkpoint_path": _relative(artifact / "last_model.pt"),
            "formal_artifact_dir": _relative(artifact),
            "epoch_checkpoint_dir": _relative(artifact / "epochs"),
            "joint_artifact_dir": _relative(FORMAL_ARTIFACT_RELATIVE),
            "joint_best_checkpoint_path": _relative(FORMAL_ARTIFACT_RELATIVE / "joint_best_model.pt"),
            "joint_last_checkpoint_path": _relative(FORMAL_ARTIFACT_RELATIVE / "joint_last_model.pt"),
            "joint_checkpoint_schema": f"{VERSION}-strict-open-set-joint-checkpoint-v1",
            "gate_min_precision_recall_target": 0.95,
            "dbb_min_precision_recall_target": 0.95,
            "cu_min_precision_recall_target": 0.90,
            "drone_bird_min_precision_recall_target": 0.90,
            "class_weights": class_weights,
            "loss_mode": "parent_final_known_only_open_set_nll_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "checkpoint_schema": f"{VERSION}-{task_slug}-checkpoint-v1",
            "allow_parameter_updates": True,
            "dataloader_seed": 42,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    index_path = root / INDEX_RELATIVE
    source_contract_path = root / SOURCE_CONTRACT_RELATIVE
    stats_path = root / STATS_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    if not index_path.is_file() or not source_contract_path.is_file():
        raise FileNotFoundError("V14 source index or V13 contract is missing")
    point_frame = pd.read_csv(index_path, low_memory=False)
    required = {"parent_sample_id", "split_name", "label_name", "source_name", "rd_tensor_path"}
    missing = sorted(required - set(point_frame.columns))
    if missing:
        raise ValueError(f"V3 index lacks V14 build columns: {missing}")
    if point_frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V14 source index is not CQ08-only")
    parent_frame = _parent_scope(point_frame)
    stats_result = build_normalization(
        root,
        index_path,
        stats_path,
        "train",
        allowed_labels=KNOWN_LABELS,
        forbidden_labels=(UNKNOWN_LABEL,),
        schema_version=VERSION,
        stats_name="cq08_known_only_train_normalization_v1",
        scope="V3 train split; CQ08 Drone/Bird/Balloon/Clutter parents only; Unknown excluded; no validation or test statistics",
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if (
        int(stats.get("parent_count", -1)) != 1011
        or int(stats.get("excluded_parent_count", -1)) != 70
        or stats.get("forbidden_labels") != [UNKNOWN_LABEL]
        or stats.get("excluded_label_parent_counts") != {UNKNOWN_LABEL: 70}
        or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
    ):
        raise ValueError("V14 known-only normalization did not prove zero Unknown exposure")
    index_hash = sha256_file(index_path)
    stats_hash = sha256_file(stats_path)
    source_contract_hash = sha256_file(source_contract_path)
    contract = _data_contract(
        index_hash=index_hash,
        source_contract_hash=source_contract_hash,
        stats_hash=stats_hash,
        stats=stats,
        parent_frame=parent_frame,
        point_frame=point_frame,
    )
    contract_status = _write_new_or_identical(contract_path, contract)
    contract_hash = sha256_file(contract_path)
    gate_weights = _inverse_frequency_weights((976, 35))
    dbb_weights = _inverse_frequency_weights((488, 243, 245))
    cu_weights = _inverse_frequency_weights((35, 976))
    configurations = {
        "dbb_cu_gate": _config(
            task_name="dbb_cu_gate",
            class_names=["DBB_ACCEPT_EVIDENCE", "ROUTE_TO_CU_EVIDENCE"],
            label_to_id={
                "DroneTarget": 0,
                "BirdTarget": 0,
                "BalloonTarget": 0,
                "ClutterTarget": 1,
            },
            class_weights=gate_weights,
            num_classes=2,
            index_hash=index_hash,
            contract_hash=contract_hash,
            stats_hash=stats_hash,
            stats=stats,
        ),
        "dbb_specialist": _config(
            task_name="dbb_specialist",
            class_names=["DroneTarget", "BirdTarget", "BalloonTarget"],
            label_to_id={"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
            class_weights=dbb_weights,
            num_classes=3,
            index_hash=index_hash,
            contract_hash=contract_hash,
            stats_hash=stats_hash,
            stats=stats,
        ),
        "cu_specialist": _config(
            task_name="cu_specialist",
            class_names=["CLUTTER_ACCEPT_EVIDENCE", "NON_CLUTTER_KNOWN_EVIDENCE"],
            label_to_id={
                "ClutterTarget": 0,
                "DroneTarget": 1,
                "BirdTarget": 1,
                "BalloonTarget": 1,
            },
            class_weights=cu_weights,
            num_classes=2,
            index_hash=index_hash,
            contract_hash=contract_hash,
            stats_hash=stats_hash,
            stats=stats,
        ),
    }
    config_paths: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    config_statuses: dict[str, str] = {}
    for task_name, config in configurations.items():
        path = root / CONFIG_ROOT_RELATIVE / f"{config['config_name']}-config.json"
        config_statuses[task_name] = _write_new_or_identical(path, config)
        config_paths[task_name] = str(path)
        config_hashes[task_name] = sha256_file(path)
    return {
        "status": "passed",
        "schema_version": VERSION,
        "source_index": str(index_path),
        "source_index_sha256": index_hash,
        "normalization": {
            "path": str(stats_path),
            "sha256": stats_hash,
            "write_status": stats_result.get("write_status"),
            "known_parent_count": stats["parent_count"],
            "excluded_unknown_parent_count": stats["excluded_parent_count"],
            "selected_excluded_rd_tensor_path_overlap_count": stats[
                "selected_excluded_rd_tensor_path_overlap_count"
            ],
            "rd_amplitude_center": stats["amplitude"]["center"],
            "rd_amplitude_scale": stats["amplitude"]["scale"],
        },
        "data_contract": {
            "path": str(contract_path),
            "sha256": contract_hash,
            "write_status": contract_status,
        },
        "configs": {
            task_name: {
                "path": config_paths[task_name],
                "sha256": config_hashes[task_name],
                "write_status": config_statuses[task_name],
            }
            for task_name in configurations
        },
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "formal_training": "未执行",
        "oof_calibration": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the V14 known-only strict-open-set statistics, contract and three configs."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
