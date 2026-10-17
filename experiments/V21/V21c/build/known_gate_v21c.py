from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_V21C_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    FEATURE_INDICES_BY_VARIANT,
    ISO_FEATURE_NAMES,
    KNOWN_GATE_FEATURE_DIM,
    KNOWN_GATE_FEATURE_SCHEMA,
    V21C_VARIANT,
    feature_contract,
)


SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
CORE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json"
)
CORE_NORMALIZATION = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-only-train-normalization-stats.json"
)
CORE_FEATURE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-feature-cache-manifest.json"
)
BASELINE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
)
BASELINE_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal"
)
BASELINE_SELECTION = Path(
    "data/processed/RADAR-LN3-V21-known-gate-baseline-development-selection.json"
)
EXPERIMENT_DOCUMENT = Path(
    "docs/radar_ln3_v21c_target_row_iso_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21c-cq08-known-gate-target-row-iso-clutter-drone-contract.json"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-clutter-drone-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-clutter-drone-"
    "seed43-formal"
)
OLD_SEED42_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-seed42-config.json"
)
KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
SEED = 43


def _clone(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "support": int(data.size),
        "mean": float(np.mean(data)) if data.size else None,
        "median": float(np.median(data)) if data.size else None,
        "q25": float(np.quantile(data, 0.25)) if data.size else None,
        "q75": float(np.quantile(data, 0.75)) if data.size else None,
        "minimum": float(np.min(data)) if data.size else None,
        "maximum": float(np.max(data)) if data.size else None,
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not pos.size or not neg.size:
        raise ValueError("V21c AUC requires two non-empty parent populations")
    greater = (pos[:, None] > neg[None, :]).sum(dtype=np.int64)
    equal = (pos[:, None] == neg[None, :]).sum(dtype=np.int64)
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def _train_only_diagnostic(
    index_path: Path,
    train_cache_path: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(
        index_path,
        usecols=[
            "sample_id",
            "parent_sample_id",
            "split_name",
            "label_name",
            "causal_event_index",
        ],
        low_memory=False,
    )
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    with np.load(train_cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError("V21c core cache fields changed")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if (
        sample_ids.dtype.kind not in {"U", "S"}
        or values.shape != (len(selected), KNOWN_GATE_FEATURE_DIM)
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or [str(value) for value in sample_ids.tolist()]
        != selected["sample_id"].astype(str).tolist()
    ):
        raise ValueError("V21c train cache no longer aligns the frozen train index")
    indices = FEATURE_INDICES_BY_VARIANT[V21C_VARIANT]
    if len(indices) != 1:
        raise ValueError("V21c must select exactly one iso feature")
    iso = values[:, indices[0]].astype(np.float64, copy=False)
    iso_valid = valid[:, indices[0]]
    if np.any(iso[~iso_valid] != 0.0):
        raise ValueError("V21c invalid iso evidence is not exactly zero")

    parent_rows: list[dict[str, Any]] = []
    for parent_id, group in selected.groupby("parent_sample_id", sort=True):
        row_indices = group.index.to_numpy(dtype=np.int64)
        labels = group["label_name"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"V21c train parent label is not unique: {parent_id}")
        active = iso_valid[row_indices]
        if not active.any():
            raise ValueError(f"V21c train parent has no valid iso event: {parent_id}")
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": labels[0],
                "iso": float(iso[row_indices[active]].mean()),
                "valid_event_count": int(active.sum()),
            }
        )
    parents = pd.DataFrame(parent_rows)
    support = {
        label: int((parents["label_name"] == label).sum()) for label in KNOWN_LABELS
    }
    expected_support = {
        "DroneTarget": 476,
        "BirdTarget": 239,
        "BalloonTarget": 238,
        "ClutterTarget": 33,
    }
    if support != expected_support:
        raise ValueError(
            f"V21c train parent support changed: expected={expected_support}, actual={support}"
        )

    distributions: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        event_mask = selected["label_name"].astype(str).eq(label).to_numpy()
        parent_group = parents[parents["label_name"].eq(label)]
        distributions[label] = {
            "parent_support": int(len(parent_group)),
            "event": _summary(iso[event_mask & iso_valid]),
            "parent_mean_over_valid_events": _summary(
                parent_group["iso"].to_numpy(dtype=np.float64)
            ),
        }
    clutter = parents[parents["label_name"].eq("ClutterTarget")]["iso"].to_numpy(
        dtype=np.float64
    )
    separations: dict[str, Any] = {}
    for other_label in ("DroneTarget", "BirdTarget", "BalloonTarget"):
        other = parents[parents["label_name"].eq(other_label)]["iso"].to_numpy(
            dtype=np.float64
        )
        clutter_higher = _auc(clutter, other)
        separations[f"ClutterTarget_vs_{other_label}"] = {
            "clutter_higher_auc": clutter_higher,
            "other_higher_auc": float(1.0 - clutter_higher),
            "direction_free_separation_auc": float(
                max(clutter_higher, 1.0 - clutter_higher)
            ),
            "clutter_minus_other_mean": float(np.mean(clutter) - np.mean(other)),
        }
    clutter_drone = separations["ClutterTarget_vs_DroneTarget"]
    return {
        "role": (
            "train-only physical attribution; it freezes the expected direction but "
            "does not become a readiness or performance gate"
        ),
        "event_count": int(len(selected)),
        "valid_event_count": int(iso_valid.sum()),
        "valid_event_fraction": float(iso_valid.mean()),
        "parent_label_distributions": distributions,
        "parent_separation": separations,
        "interpretation_frozen_before_treatment_validation": {
            "drone_parent_mean_is_higher_than_clutter": (
                float(clutter_drone["clutter_minus_other_mean"]) < 0.0
            ),
            "clutter_drone_direction_free_auc": float(
                clutter_drone["direction_free_separation_auc"]
            ),
            "clutter_bird_univariate_separation_is_weak": (
                float(
                    separations["ClutterTarget_vs_BirdTarget"][
                        "direction_free_separation_auc"
                    ]
                )
                < 0.60
            ),
            "formula_changed_after_this_diagnostic": False,
        },
    }


def _config_payload(
    baseline: Mapping[str, Any],
    *,
    manifest_hash: str,
    train_cache_path: str,
    train_cache_hash: str,
    validation_cache_path: str,
    validation_cache_hash: str,
    experiment_contract_hash: str,
) -> dict[str, Any]:
    payload = _clone(baseline)
    payload["config_name"] = (
        "RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-"
        "clutter-drone-seed43"
    )
    payload["variant"] = V21C_VARIANT
    payload["task"]["feature_names"] = list(ISO_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": CORE_FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": train_cache_path,
            "validation_feature_cache_path": validation_cache_path,
            "expected_train_feature_cache_sha256": train_cache_hash,
            "expected_validation_feature_cache_sha256": validation_cache_hash,
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": experiment_contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = len(ISO_FEATURE_NAMES)
    artifact = ARTIFACT.as_posix()
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V21-v21c-clutter-drone-seed43-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": BASELINE_CONFIG.as_posix(),
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "feature_manifest": root / CORE_FEATURE_MANIFEST,
        "baseline_config": root / BASELINE_CONFIG,
        "baseline_summary": root / BASELINE_ARTIFACT / "training-summary.json",
        "baseline_checkpoint": root / BASELINE_ARTIFACT / "best_model.pt",
        "selection": root / BASELINE_SELECTION,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
        "old_seed42_config": root / OLD_SEED42_CONFIG,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21c source files are missing: {missing}")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    selection = json.loads(source_paths["selection"].read_text(encoding="utf-8"))
    baseline_summary = json.loads(
        source_paths["baseline_summary"].read_text(encoding="utf-8")
    )
    if (
        selection.get("status") != "selected_development_champion"
        or int(selection.get("selected_seed", -1)) != SEED
        or selection.get("config_sha256") != source_hashes["baseline_config"]
        or selection.get("summary_sha256") != source_hashes["baseline_summary"]
        or selection.get("checkpoint_sha256") != source_hashes["baseline_checkpoint"]
        or baseline_summary.get("status") != "completed"
        or baseline_summary.get("variant") != "baseline"
        or int(baseline_summary.get("seed", -1)) != SEED
        or len(list(baseline_summary.get("history") or [])) != 20
        or baseline_summary.get("test_evaluation") != "not_executed"
        or baseline_summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V21c selected seed43 Baseline binding is stale or incomplete")
    baseline_config_payload = json.loads(
        source_paths["baseline_config"].read_text(encoding="utf-8")
    )
    baseline_config = load_known_gate_config(source_paths["baseline_config"])
    rebuilt, reference, evidence = build_known_gate_model_for_config(
        root, baseline_config
    )
    rebuilt_initial_sha = _state_dict_sha256(rebuilt.state_dict())
    if (
        reference is not None
        or evidence.get("paired_baseline") is not False
        or rebuilt_initial_sha
        != str(baseline_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError(
            "V21c current code no longer reconstructs the selected Baseline initial state"
        )
    del rebuilt

    manifest = json.loads(
        source_paths["feature_manifest"].read_text(encoding="utf-8")
    )
    core_contract = json.loads(
        source_paths["core_contract"].read_text(encoding="utf-8")
    )
    manifest_caches = dict(manifest.get("cache_files") or {})
    train_binding = dict(manifest_caches.get("train") or {})
    validation_binding = dict(manifest_caches.get("val") or {})
    train_cache = root / str(train_binding.get("path", ""))
    validation_cache = root / str(validation_binding.get("path", ""))
    iso_contract = dict(
        dict(dict(core_contract.get("feature_contract") or {}).get("features") or {}).get(
            "target_row_iso"
        )
        or {}
    )
    expected_iso_contract = dict(feature_contract()["features"]["target_row_iso"])
    if (
        manifest.get("schema_version")
        != "RADAR-LN3-V21-known-gate-feature-cache-manifest-v1"
        or manifest.get("feature_schema") != KNOWN_GATE_FEATURE_SCHEMA
        or manifest.get("data_contract_sha256") != source_hashes["core_contract"]
        or manifest.get("test_excluded") is not True
        or int(manifest.get("test_payload_access_count", -1)) != 0
        or manifest.get("valid_mask_is_model_input") is not False
        or iso_contract != expected_iso_contract
        or not train_cache.is_file()
        or not validation_cache.is_file()
        or sha256_file(train_cache) != train_binding.get("sha256")
        or sha256_file(validation_cache) != validation_binding.get("sha256")
    ):
        raise ValueError("V21c frozen core feature contract/cache binding changed")

    diagnostic = _train_only_diagnostic(source_paths["index"], train_cache)
    contract = {
        "schema_version": (
            "RADAR-LN3-V21c-known-gate-target-row-iso-"
            "clutter-drone-experiment-contract-v1"
        ),
        "experiment_name": "V21c target-row energy isolation for Clutter versus Drone",
        "internal_variant": V21C_VARIANT,
        "feature_names": list(ISO_FEATURE_NAMES),
        "formula": iso_contract["formula"],
        "source": iso_contract["source"],
        "causal_scope": "current event only; no previous/future event or parent-final aggregation",
        "target_row_semantics": (
            "the zero-offset row of each canonical 21x384 RD frame; the row is not "
            "selected by a label, global energy argmax or validation result"
        ),
        "power_semantics": (
            "convert current canonical dB amplitudes to relative linear power after "
            "subtracting each frame maximum; take target-row power divided by full-frame "
            "power per frame and then the current-event mean"
        ),
        "value_bounds": [0.0, 1.0],
        "missing_policy": (
            "an event without canonical RD has value zero and invalid; TR radial/range "
            "missingness does not invalidate iso"
        ),
        "invalid_policy": "zero value plus internal gating mask; validity is not concatenated",
        "experiment_document": str(source_paths["experiment_document"]),
        "experiment_document_sha256": source_hashes["experiment_document"],
        "core_data_contract": str(source_paths["core_contract"]),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "core_feature_manifest": str(source_paths["feature_manifest"]),
        "core_feature_manifest_sha256": source_hashes["feature_manifest"],
        "train_cache": str(train_cache),
        "train_cache_sha256": str(train_binding["sha256"]),
        "validation_cache": str(validation_cache),
        "validation_cache_sha256": str(validation_binding["sha256"]),
        "cache_rebuilt_for_v21c_seed43": False,
        "raw_echo_payload_access_count": 0,
        "validation_feature_payload_access_count_for_readiness": 0,
        "test_payload_access_count": 0,
        "train_only_feature_diagnostic": diagnostic,
        "historical_exploration_disclosure": {
            "not_formal_evidence": True,
            "historical_script": "tmp/v3_probe_archive/tmp_feature_mining.py",
            "sampled_track_count": 93,
            "sampled_split_counts": {"train": 60, "val": 13, "test": 20},
            "test_track_count_observed_during_feature_conception": 20,
            "reported_drone_clutter_values_are_not_formal_v21c_values": True,
            "non_equivalence_reasons": [
                "the historical probe sampled all V16 splits rather than train only",
                "it used native 31-by-PRT absolute-amplitude sums instead of canonical 21-by-384 relative linear power",
                "it selected a target row once from an energy argmax and reused it across events instead of using the canonical zero-offset row",
                "it truncated each parent to at most ten assumed three-frame events and aggregated at parent scope",
            ],
            "holdout_consequence": (
                "the current prepare does not access test payloads, but the original V16 "
                "test is not a pristine final holdout for the iso feature concept"
            ),
        },
        "paired_protocol": {
            "seed": SEED,
            "baseline_config": str(source_paths["baseline_config"]),
            "baseline_config_sha256": source_hashes["baseline_config"],
            "baseline_summary": str(source_paths["baseline_summary"]),
            "baseline_summary_sha256": source_hashes["baseline_summary"],
            "baseline_checkpoint": str(source_paths["baseline_checkpoint"]),
            "baseline_checkpoint_sha256": source_hashes["baseline_checkpoint"],
            "baseline_initial_model_state_sha256": rebuilt_initial_sha,
            "initialization": (
                "same seed43 random Baseline initialization; copy all shared tensors and "
                "zero only the one new Fusion column; do not fine-tune the trained Baseline"
            ),
            "epochs": 20,
            "full_parameter_training": True,
            "all_epoch_parent_orders_must_match": True,
        },
        "evaluation": {
            "primary": "development validation parent-final KnownGate macro_f1 delta",
            "adoption_delta_strictly_greater_than": 0.02,
            "research_question": (
                "does current-event canonical target-row energy isolation improve the "
                "ClutterTarget versus DroneTarget KnownGate boundary?"
            ),
            "primary_hypothesis_labels": ["DroneTarget", "ClutterTarget"],
            "safety_labels": ["BirdTarget", "BalloonTarget"],
            "primary_pair_must_have_positive_net_rescue": True,
            "each_primary_label_must_have_non_negative_net_rescue": True,
            "each_safety_label_must_have_non_negative_net_rescue": True,
            "clutter_validation_support": 9,
            "drone_validation_support": 113,
            "parent_stratified_bootstrap_replicates": 5000,
            "test": "未执行",
            "OOF": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "full_cascade": "未执行",
        },
        "old_seed42_candidate": {
            "config": str(source_paths["old_seed42_config"]),
            "config_sha256": source_hashes["old_seed42_config"],
            "formal_training_allowed": False,
            "reason": "selected development Baseline is seed43; cross-seed attribution is invalid",
        },
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    config_payload = _config_payload(
        baseline_config_payload,
        manifest_hash=source_hashes["feature_manifest"],
        train_cache_path=str(train_binding["path"]),
        train_cache_hash=str(train_binding["sha256"]),
        validation_cache_path=str(validation_binding["path"]),
        validation_cache_hash=str(validation_binding["sha256"]),
        experiment_contract_hash=contract_hash,
    )
    _write_json_new_or_identical(root / CONFIG, config_payload)
    config = load_known_gate_config(root / CONFIG)
    treatment, paired_baseline, paired = build_known_gate_model_for_config(root, config)
    if (
        config.variant != V21C_VARIANT
        or config.train.seed != SEED
        or config.train.paired_baseline_config_path != BASELINE_CONFIG.as_posix()
        or paired_baseline is None
        or paired.get("shared_parameters_and_buffers_exactly_equal") is not True
        or paired.get("new_fusion_columns_exactly_zero") is not True
        or int(paired.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V21c generated config failed its selected-seed pair contract")
    del treatment, paired_baseline

    checks = {
        "selected_development_baseline_is_seed43_and_sha_bound": True,
        "current_code_reconstructs_formal_seed43_initial_state_sha": True,
        "iso_formula_matches_the_frozen_v21_core_contract": True,
        "zero_offset_target_row_and_current_event_frames_are_the_only_rd_scope": True,
        "relative_linear_power_fraction_is_bounded_without_label_conditioning": True,
        "tr_missingness_does_not_silently_invalidate_iso": True,
        "core_cache_is_sha_reused_without_raw_rebuild": True,
        "train_cache_sample_order_is_exact": True,
        "train_only_physical_diagnostic_is_recorded_without_becoming_a_performance_gate": True,
        "validation_feature_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_cache_and_training_population": True,
        "test_payload_access_count_is_zero": True,
        "treatment_copies_shared_initialization_and_zeroes_only_one_new_column": True,
        "full_parameter_20_epoch_same_seed_pair_is_required": True,
        "old_seed42_candidate_is_explicitly_non_comparable": True,
        "clutter_drone_are_the_only_primary_hypothesis_labels": True,
        "bird_balloon_are_safety_labels_only": True,
        "historical_probe_test_exposure_and_formula_mismatch_are_disclosed": True,
    }
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_V21C_READINESS_SCHEMA,
        "checks": checks,
        "config": {
            "path": str(root / CONFIG),
            "sha256": sha256_file(root / CONFIG),
        },
        "selected_baseline": {
            "seed": SEED,
            "selection_path": str(source_paths["selection"]),
            "selection_sha256": source_hashes["selection"],
            "config_path": str(source_paths["baseline_config"]),
            "config_sha256": source_hashes["baseline_config"],
            "summary_path": str(source_paths["baseline_summary"]),
            "summary_sha256": source_hashes["baseline_summary"],
            "checkpoint_path": str(source_paths["baseline_checkpoint"]),
            "checkpoint_sha256": source_hashes["baseline_checkpoint"],
            "initial_model_state_sha256": rebuilt_initial_sha,
        },
        "contract_hashes": {
            "index_sha256": source_hashes["index"],
            "core_contract_sha256": source_hashes["core_contract"],
            "normalization_sha256": source_hashes["normalization"],
            "experiment_contract_sha256": contract_hash,
            "feature_manifest_sha256": source_hashes["feature_manifest"],
            "train_feature_cache_sha256": str(train_binding["sha256"]),
            "validation_feature_cache_sha256": str(validation_binding["sha256"]),
        },
        "train_only_feature_diagnostic": diagnostic,
        "validation_used_for_readiness_gate": False,
        "validation_feature_values_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training_at_report_generation": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / READINESS, readiness)
    return {
        "status": "passed",
        "schema_version": KNOWN_GATE_V21C_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "core_feature_cache_reused": True,
        "paired_baseline_seed": SEED,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_support": 9,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the selected-seed43 V21c target-row iso paired contract."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
