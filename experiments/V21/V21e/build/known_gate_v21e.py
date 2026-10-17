from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MANIFEST_SCHEMA,
    KNOWN_GATE_V21E_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V21E_COURSE_PREFIX_VARIANT,
    extract_causal_course_circular_incoherence,
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
    "docs/radar_ln3_v21e_causal_course_consistency_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-"
    "feature-cache-manifest.json"
)
TRAIN_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-train-causal-course-consistency-cache.npz"
)
VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-val-causal-course-consistency-cache.npz"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal"
)

KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
EXPECTED_SUPPORT = {
    "train": {
        "DroneTarget": 476,
        "BirdTarget": 239,
        "BalloonTarget": 238,
        "ClutterTarget": 33,
    },
    "val": {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    },
}
SEED = 43
MINIMUM_VALID_NONFIRST_COURSES = 3


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
        raise ValueError("V21e AUC requires two non-empty parent populations")
    greater = (pos[:, None] > neg[None, :]).sum(dtype=np.int64)
    equal = (pos[:, None] == neg[None, :]).sum(dtype=np.int64)
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def _selected_index(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "sample_id",
            "parent_sample_id",
            "track_id",
            "split_name",
            "label_name",
            "causal_event_index",
            "tr_feature_values_json",
        ],
        low_memory=False,
    )
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V21e {split} index repeats sample_id")
    support = (
        selected[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")["label_name"]
        .value_counts()
        .to_dict()
    )
    support = {label: int(support.get(label, 0)) for label in KNOWN_LABELS}
    if support != EXPECTED_SUPPORT[split]:
        raise ValueError(
            f"V21e {split} support changed: expected={EXPECTED_SUPPORT[split]}, "
            f"actual={support}"
        )
    return selected


def _course_value(payload: Any, *, sample_id: str) -> float | None:
    values = json.loads(str(payload))
    if not isinstance(values, dict):
        raise ValueError(f"V21e TR payload is not an object: {sample_id}")
    raw = values.get("course_deg")
    if raw is None:
        return None
    course = float(raw)
    if not math.isfinite(course):
        raise ValueError(f"V21e course is non-finite: {sample_id}")
    return course


def _build_split(
    selected: pd.DataFrame,
    *,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), 1), dtype=np.float32)
    valid = np.zeros((len(selected), 1), dtype=np.bool_)
    parent_rows: list[dict[str, Any]] = []
    first_event_excluded_count = 0
    first_event_zero_count = 0
    nonfirst_missing_count = 0
    nonfirst_zero_count = 0
    carried_valid_prefix_count = 0
    cursor = 0
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        labels = parent["label_name"].astype(str).unique().tolist()
        tracks = parent["track_id"].astype(int).unique().tolist()
        event_indices = parent["causal_event_index"].astype(int).tolist()
        if len(labels) != 1 or len(tracks) != 1:
            raise ValueError(f"V21e parent identity is not unique: {parent_id}")
        if event_indices != list(range(len(parent))):
            raise ValueError(f"V21e causal event order is not contiguous: {parent_id}")
        courses = [
            _course_value(row.tr_feature_values_json, sample_id=str(row.sample_id))
            for row in parent.itertuples(index=False)
        ]
        first_event_excluded_count += 1
        first_event_zero_count += int(
            courses[0] is not None and float(courses[0]) == 0.0
        )
        nonfirst_missing_count += sum(value is None for value in courses[1:])
        nonfirst_zero_count += sum(
            value is not None and float(value) == 0.0 for value in courses[1:]
        )
        parent_values, parent_valid = extract_causal_course_circular_incoherence(
            courses,
            minimum_valid_nonfirst_points=MINIMUM_VALID_NONFIRST_COURSES,
        )
        length = len(parent)
        values[cursor : cursor + length] = parent_values
        valid[cursor : cursor + length] = parent_valid
        for event_index in range(1, length):
            if (
                courses[event_index] is None
                and bool(parent_valid[event_index, 0])
            ):
                carried_valid_prefix_count += 1
        selected_values = parent_values[parent_valid]
        if selected_values.size == 0 or not bool(parent_valid[-1, 0]):
            raise ValueError(f"V21e parent lacks a valid final course prefix: {parent_id}")
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "track_id": int(tracks[0]),
                "label_name": labels[0],
                "event_count": length,
                "valid_nonfirst_course_count": int(
                    sum(value is not None for value in courses[1:])
                ),
                "valid_prefix_count": int(parent_valid.sum()),
                "prefix_mean_incoherence": float(np.mean(selected_values)),
                "prefix_median_incoherence": float(np.median(selected_values)),
                "final_incoherence": float(parent_values[-1, 0]),
                "first_event_course_was_zero": bool(
                    courses[0] is not None and float(courses[0]) == 0.0
                ),
                "later_zero_course_count": int(
                    sum(
                        value is not None and float(value) == 0.0
                        for value in courses[1:]
                    )
                ),
            }
        )
        cursor += length
    if cursor != len(selected) or np.any(values[~valid] != 0.0):
        raise ValueError("V21e cache construction did not preserve event/mask semantics")
    parents = pd.DataFrame(parent_rows)
    statistics = {
        "split": split,
        "event_count": len(selected),
        "parent_count": len(parents),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_event_fraction": float(valid.mean()),
        "first_event_excluded_count": int(first_event_excluded_count),
        "first_event_zero_but_excluded_count": int(first_event_zero_count),
        "nonfirst_missing_course_count": int(nonfirst_missing_count),
        "nonfirst_true_zero_course_included_count": int(nonfirst_zero_count),
        "missing_event_carried_valid_prefix_count": int(carried_valid_prefix_count),
        "minimum_valid_nonfirst_courses": MINIMUM_VALID_NONFIRST_COURSES,
        "test_payload_access_count": 0,
    }
    return {"sample_ids": sample_ids, "values": values, "valid": valid}, statistics, parents


def _train_diagnostic(parents: pd.DataFrame) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        group = parents[parents["label_name"].eq(label)]
        distributions[label] = {
            "parent_support": int(len(group)),
            "final_incoherence": _summary(
                group["final_incoherence"].to_numpy(dtype=np.float64)
            ),
            "prefix_mean_incoherence": _summary(
                group["prefix_mean_incoherence"].to_numpy(dtype=np.float64)
            ),
            "valid_nonfirst_course_count": _summary(
                group["valid_nonfirst_course_count"].to_numpy(dtype=np.float64)
            ),
            "first_event_course_zero_fraction_but_excluded": float(
                group["first_event_course_was_zero"].mean()
            ),
            "later_true_zero_course_count": int(group["later_zero_course_count"].sum()),
        }
    clutter = parents[parents["label_name"].eq("ClutterTarget")]
    target = parents[parents["label_name"].ne("ClutterTarget")]
    separations: dict[str, Any] = {}
    for label in ("KNOWN_TARGET_COMBINED", "DroneTarget", "BirdTarget", "BalloonTarget"):
        other = target if label == "KNOWN_TARGET_COMBINED" else parents[
            parents["label_name"].eq(label)
        ]
        clutter_values = clutter["final_incoherence"].to_numpy(dtype=np.float64)
        other_values = other["final_incoherence"].to_numpy(dtype=np.float64)
        clutter_higher = _auc(clutter_values, other_values)
        separations[f"ClutterTarget_vs_{label}"] = {
            "clutter_higher_auc": clutter_higher,
            "other_higher_auc": float(1.0 - clutter_higher),
            "direction_free_separation_auc": float(
                max(clutter_higher, 1.0 - clutter_higher)
            ),
            "clutter_minus_other_mean": float(
                np.mean(clutter_values) - np.mean(other_values)
            ),
        }
    length_correlation = float(
        parents[["final_incoherence", "valid_nonfirst_course_count"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    return {
        "role": (
            "post-hoc train-only feature plausibility and shortcut audit; these "
            "statistics do not become a readiness or performance gate"
        ),
        "parent_label_distributions": distributions,
        "parent_separation": separations,
        "final_incoherence_spearman_with_valid_nonfirst_course_count": (
            length_correlation
        ),
        "feature_formula_changed_after_this_diagnostic": False,
    }


def _config_payload(
    baseline: Mapping[str, Any],
    *,
    manifest_hash: str,
    train_cache_hash: str,
    validation_cache_hash: str,
    contract_hash: str,
) -> dict[str, Any]:
    payload = _clone(baseline)
    payload["config_name"] = (
        "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43"
    )
    payload["variant"] = V21E_COURSE_PREFIX_VARIANT
    payload["task"]["feature_names"] = list(COURSE_PREFIX_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": TRAIN_CACHE.as_posix(),
            "validation_feature_cache_path": VALIDATION_CACHE.as_posix(),
            "expected_train_feature_cache_sha256": train_cache_hash,
            "expected_validation_feature_cache_sha256": validation_cache_hash,
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = 1
    artifact = ARTIFACT.as_posix()
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V21-v21e_course_prefix-seed43-known-gate-checkpoint-v1"
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
        "baseline_config": root / BASELINE_CONFIG,
        "baseline_summary": root / BASELINE_ARTIFACT / "training-summary.json",
        "baseline_checkpoint": root / BASELINE_ARTIFACT / "best_model.pt",
        "selection": root / BASELINE_SELECTION,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21e source files are missing: {missing}")
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
        raise ValueError("V21e selected seed43 Baseline binding is stale")
    baseline_payload = json.loads(
        source_paths["baseline_config"].read_text(encoding="utf-8")
    )
    baseline_config = load_known_gate_config(source_paths["baseline_config"])
    rebuilt, reference, pair = build_known_gate_model_for_config(root, baseline_config)
    rebuilt_sha = _state_dict_sha256(rebuilt.state_dict())
    if (
        reference is not None
        or pair.get("paired_baseline") is not False
        or rebuilt_sha != str(baseline_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError("V21e current code cannot reconstruct seed43 Baseline initial state")
    del rebuilt

    core_contract = json.loads(
        source_paths["core_contract"].read_text(encoding="utf-8")
    )
    if core_contract.get("schema_version") != "RADAR-LN3-V21-known-gate-data-contract-v1":
        raise ValueError("V21e core KnownGate contract changed")
    selected = {
        split: _selected_index(source_paths["index"], split)
        for split in ("train", "val")
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    split_stats: dict[str, Any] = {}
    parent_tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        arrays[split], split_stats[split], parent_tables[split] = _build_split(
            selected[split], split=split
        )
    train_diagnostic = _train_diagnostic(parent_tables["train"])

    contract = {
        "schema_version": (
            "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
            "experiment-contract-v1"
        ),
        "experiment_name": "V21e causal trajectory course consistency",
        "internal_variant": V21E_COURSE_PREFIX_VARIANT,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "source": "V16 index current/prior tr_feature_values_json.course_deg",
        "formula": "1 - abs(mean(exp(i * radians(course_deg))))",
        "causal_scope": (
            "current and prior events only; the first reported event is always "
            "excluded and no future or parent-final value is read"
        ),
        "minimum_evidence": "three finite non-first course observations",
        "missing_policy": (
            "before minimum support value=0 and valid=false; a later missing course "
            "adds no observation and carries an already-valid prefix; later true 0-degree "
            "course is valid"
        ),
        "first_event_policy": (
            "unconditionally excluded even when non-zero because CQ08 first-event "
            "speed/course initialization semantics are unconfirmed"
        ),
        "existing_backbone_overlap": (
            "the baseline already consumes pointwise course sin/cos and causal course "
            "increments; V21e tests whether explicit long-prefix circular dispersion "
            "adds residual information"
        ),
        "experiment_document": str(source_paths["experiment_document"]),
        "experiment_document_sha256": source_hashes["experiment_document"],
        "post_hoc_feature_discovery": True,
        "look_elsewhere_risk": {
            "validation_errors_inspected_before_feature_selection": True,
            "prior_v21_feature_families_evaluated": ["V21a", "V21b", "V21c", "V21d"],
            "same_development_validation_is_not_an_unbiased_confirmation_set": True,
            "positive_result_requires_multiseed_train_parent_grouped_oof_and_new_holdout": True,
        },
        "core_data_contract": str(source_paths["core_contract"]),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "source_index": str(source_paths["index"]),
        "source_index_sha256": source_hashes["index"],
        "raw_echo_payload_access_count": 0,
        "mat_payload_access_count": 0,
        "rd_tensor_payload_access_count": 0,
        "test_payload_access_count": 0,
        "validation_feature_values_constructed_for_evaluation": True,
        "validation_feature_values_used_for_readiness_gate": False,
        "train_only_feature_diagnostic": train_diagnostic,
        "paired_protocol": {
            "seed": SEED,
            "baseline_config": str(source_paths["baseline_config"]),
            "baseline_config_sha256": source_hashes["baseline_config"],
            "baseline_summary": str(source_paths["baseline_summary"]),
            "baseline_summary_sha256": source_hashes["baseline_summary"],
            "baseline_checkpoint": str(source_paths["baseline_checkpoint"]),
            "baseline_checkpoint_sha256": source_hashes["baseline_checkpoint"],
            "baseline_initial_model_state_sha256": rebuilt_sha,
            "initialization": (
                "same seed43 random Baseline initialization; copy every shared tensor "
                "and zero only the one new Fusion column; do not fine-tune the trained "
                "Baseline checkpoint"
            ),
            "epochs": 20,
            "full_parameter_training": True,
            "all_epoch_parent_orders_must_match": True,
        },
        "evaluation": {
            "primary": "development validation parent-final KnownGate macro_f1 delta",
            "adoption_delta_strictly_greater_than": 0.02,
            "research_question": (
                "does explicit causal-prefix course circular incoherence improve the "
                "KnownTarget versus ClutterTarget boundary beyond existing TR features?"
            ),
            "primary_hypothesis_labels": list(KNOWN_LABELS),
            "safety_labels": [],
            "combined_net_rescue_must_be_positive": True,
            "each_base_label_net_rescue_must_be_non_negative": True,
            "known_target_recall_must_not_decrease": True,
            "known_clutter_recall_must_not_decrease": True,
            "fixed_treatment_weight_zero_feature_diagnostic_required": True,
            "clutter_validation_support": 9,
            "parent_stratified_bootstrap_replicates": 5000,
            "test": "未执行",
            "OOF": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "full_cascade": "未执行",
        },
        "validation_is_pristine_for_feature_conception": False,
        "test_is_unbiased_for_project_level_release": False,
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

    cache_paths = {"train": root / TRAIN_CACHE, "val": root / VALIDATION_CACHE}
    for split in ("train", "val"):
        _write_npz_new_or_identical(
            cache_paths[split],
            tuple((name, value) for name, value in arrays[split].items()),
        )
    cache_hashes = {split: sha256_file(path) for split, path in cache_paths.items()}
    manifest = {
        "schema_version": KNOWN_GATE_V21E_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "cache_files": {
            split: {
                "path": (TRAIN_CACHE if split == "train" else VALIDATION_CACHE).as_posix(),
                "sha256": cache_hashes[split],
                "fields": ["sample_ids", "values", "valid"],
                **split_stats[split],
            }
            for split in ("train", "val")
        },
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "unknown_excluded": True,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_json_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)

    config_payload = _config_payload(
        baseline_payload,
        manifest_hash=manifest_hash,
        train_cache_hash=cache_hashes["train"],
        validation_cache_hash=cache_hashes["val"],
        contract_hash=contract_hash,
    )
    _write_json_new_or_identical(root / CONFIG, config_payload)
    config = load_known_gate_config(root / CONFIG)
    treatment, paired_baseline, paired = build_known_gate_model_for_config(root, config)
    if (
        config.variant != V21E_COURSE_PREFIX_VARIANT
        or config.train.seed != SEED
        or config.train.paired_baseline_config_path != BASELINE_CONFIG.as_posix()
        or paired_baseline is None
        or paired.get("shared_parameters_and_buffers_exactly_equal") is not True
        or paired.get("new_fusion_columns_exactly_zero") is not True
        or int(paired.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V21e generated config failed the seed43 pair contract")
    del treatment, paired_baseline

    checks = {
        "selected_development_baseline_is_seed43_and_sha_bound": True,
        "current_code_reconstructs_formal_seed43_initial_state_sha": True,
        "feature_uses_only_current_and_prior_course": True,
        "first_event_is_unconditionally_excluded": True,
        "later_true_zero_degree_course_is_included": True,
        "minimum_three_valid_nonfirst_courses_is_enforced": True,
        "invalid_feature_values_are_exactly_zero": True,
        "independent_train_validation_cache_aligns_every_known_event": True,
        "no_raw_mat_or_rd_tensor_payload_is_read_during_prepare": True,
        "train_only_auc_is_recorded_without_becoming_a_readiness_gate": True,
        "validation_post_hoc_and_look_elsewhere_risk_are_disclosed": True,
        "validation_feature_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_cache_and_training_population": True,
        "test_payload_access_count_is_zero": True,
        "treatment_copies_shared_initialization_and_zeroes_only_one_new_column": True,
        "full_parameter_20_epoch_same_seed_pair_is_required": True,
        "fixed_weight_zero_feature_diagnostic_is_required": True,
    }
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_V21E_READINESS_SCHEMA,
        "checks": checks,
        "config": {"path": str(root / CONFIG), "sha256": sha256_file(root / CONFIG)},
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
            "initial_model_state_sha256": rebuilt_sha,
        },
        "contract_hashes": {
            "index_sha256": source_hashes["index"],
            "core_contract_sha256": source_hashes["core_contract"],
            "normalization_sha256": source_hashes["normalization"],
            "experiment_contract_sha256": contract_hash,
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": cache_hashes["train"],
            "validation_feature_cache_sha256": cache_hashes["val"],
        },
        "split_cache_statistics": split_stats,
        "train_only_feature_diagnostic": train_diagnostic,
        "post_hoc_feature_discovery": True,
        "validation_used_for_feature_selection_before_v21e": True,
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
        "schema_version": KNOWN_GATE_V21E_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(root / TRAIN_CACHE),
        "validation_cache": str(root / VALIDATION_CACHE),
        "paired_baseline_seed": SEED,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_support": 9,
        "post_hoc_feature_discovery": True,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the selected-seed43 V21e causal-course experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
