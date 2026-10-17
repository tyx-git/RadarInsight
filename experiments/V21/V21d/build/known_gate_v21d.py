from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _as_bool,
    _parse_indices,
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V21D_MANIFEST_SCHEMA,
    KNOWN_GATE_V21D_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    D_RANGE_PREFIX_FEATURE_NAMES,
    D_RANGE_PREFIX_FEATURE_SCHEMA,
    D_RANGE_STD_CLIP_M,
    V21D_PREFIX_VARIANT,
)


SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
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
CORE_TRAIN_CACHE = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-train-feature-cache.npz"
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
HISTORICAL_PROBE = Path("tmp/v3_probe_archive/tmp_mat_feature_mining.m")
EXPERIMENT_DOCUMENT = Path(
    "docs/radar_ln3_v21d_causal_prefix_d_range_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-causal-prefix-d-range-clutter-bird-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-causal-prefix-d-range-feature-cache-manifest.json"
)
TRAIN_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-train-causal-prefix-d-range-cache.npz"
)
VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-val-causal-prefix-d-range-cache.npz"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21d-cq08-liquid-known-gate-causal-prefix-d-range-clutter-bird-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V21d-cq08-liquid-known-gate-causal-prefix-d-range-clutter-bird-"
    "seed43-formal"
)
OLD_SEED42_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21d-cq08-liquid-known-gate-d-range-stability-seed42-config.json"
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
        raise ValueError("V21d AUC requires two non-empty parent populations")
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
            "track_point_index",
            "tr_feature_values_json",
            "rd_present",
            "rd_frame_count",
            "rd_tensor_path",
            "rd_tensor_frame_indices_json",
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
    support = (
        selected[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")["label_name"]
        .value_counts()
        .to_dict()
    )
    support = {label: int(support.get(label, 0)) for label in KNOWN_LABELS}
    if support != EXPECTED_SUPPORT[split]:
        raise ValueError(
            f"V21d {split} support changed: expected={EXPECTED_SUPPORT[split]}, actual={support}"
        )
    return selected


def _frame_lookup(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    required = [
        "schema_version",
        "parent_sample_id",
        "track_id",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "track_point_index",
        "rd_tensor_path",
        "target_range_reference_m",
    ]
    frame = pd.read_csv(path, usecols=required, low_memory=False)
    if frame.empty or set(frame["schema_version"].astype(str)) != {
        "cq08_rd_physical_canonical_v3_1"
    }:
        raise ValueError("V21d canonical frame provenance schema changed")
    if not bool(frame["is_tensor_frame_kept"].map(_as_bool).all()):
        raise ValueError("V21d canonical frame provenance contains dropped rows")
    if frame.duplicated(["parent_sample_id", "tensor_frame_index"]).any():
        raise ValueError("V21d canonical frame provenance repeats a tensor frame")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        records[(str(row.parent_sample_id), int(row.tensor_frame_index))] = {
            "track_id": int(row.track_id),
            "track_point_index": int(row.track_point_index),
            "rd_tensor_path": str(row.rd_tensor_path),
            "detected_range_m": float(row.target_range_reference_m),
        }
    return records, {
        "schema_version": "cq08_rd_physical_canonical_v3_1",
        "row_count": len(records),
        "feature_source_field": "target_range_reference_m",
        "source_field_semantics": "MAT debug_info.detected_range_m",
    }


def _build_split(
    selected: pd.DataFrame,
    frame_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), 1), dtype=np.float32)
    valid = np.zeros((len(selected), 1), dtype=np.bool_)
    parent_rows: list[dict[str, Any]] = []
    residual_frame_count = 0
    skipped_missing_tr_range_frame_count = 0
    stale_prefix_event_count = 0
    clipped_event_count = 0
    clipped_parent_ids: set[str] = set()
    maximum_raw_prefix_std_m = 0.0
    cursor = 0
    grouped = selected.groupby("parent_sample_id", sort=True)
    for parent_id, parent in grouped:
        labels = parent["label_name"].astype(str).unique().tolist()
        tracks = parent["track_id"].astype(int).unique().tolist()
        if len(labels) != 1 or len(tracks) != 1:
            raise ValueError(f"V21d parent identity is not unique: {parent_id}")
        residuals: list[float] = []
        parent_values_m: list[float] = []
        parent_model_input_values_m: list[float] = []
        used_tensor_indices: set[int] = set()
        for row in parent.itertuples(index=False):
            previous_count = len(residuals)
            tr = json.loads(str(row.tr_feature_values_json))
            tr_range = tr.get("range_m")
            if _as_bool(row.rd_present):
                tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
                if len(tensor_indices) != int(row.rd_frame_count) or not tensor_indices:
                    raise ValueError(f"V21d event frame count changed: {row.sample_id}")
                if used_tensor_indices & set(tensor_indices):
                    raise ValueError(f"V21d reuses a tensor frame: {row.sample_id}")
                used_tensor_indices.update(tensor_indices)
                for tensor_index in tensor_indices:
                    try:
                        source = dict(frame_lookup[(str(parent_id), int(tensor_index))])
                    except KeyError as exc:
                        raise ValueError(
                            f"V21d frame provenance lacks {parent_id}/{tensor_index}"
                        ) from exc
                    if (
                        int(source["track_id"]) != int(tracks[0])
                        or int(source["track_point_index"])
                        != int(row.track_point_index)
                        or Path(str(source["rd_tensor_path"])).as_posix()
                        != Path(str(row.rd_tensor_path)).as_posix()
                    ):
                        raise ValueError(f"V21d event/frame identity differs: {row.sample_id}")
                    if tr_range is None:
                        skipped_missing_tr_range_frame_count += 1
                    else:
                        residuals.append(
                            float(source["detected_range_m"]) - float(tr_range)
                        )
                        residual_frame_count += 1
            if len(residuals) >= 2:
                value_m = float(np.std(np.asarray(residuals), ddof=1))
                maximum_raw_prefix_std_m = max(maximum_raw_prefix_std_m, value_m)
                normalized = value_m / D_RANGE_STD_CLIP_M
                clipped_event_count += int(normalized > 1.0)
                if normalized > 1.0:
                    clipped_parent_ids.add(str(parent_id))
                values[cursor, 0] = np.float32(np.clip(normalized, 0.0, 1.0))
                valid[cursor, 0] = True
                parent_values_m.append(value_m)
                parent_model_input_values_m.append(
                    float(min(value_m, D_RANGE_STD_CLIP_M))
                )
                stale_prefix_event_count += int(len(residuals) == previous_count)
            cursor += 1
        if not parent_values_m:
            raise ValueError(f"V21d parent has no valid causal prefix: {parent_id}")
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": labels[0],
                "prefix_mean_std_m": float(np.mean(parent_values_m)),
                "prefix_median_std_m": float(np.median(parent_values_m)),
                "final_std_m": float(parent_values_m[-1]),
                "maximum_prefix_std_m": float(np.max(parent_values_m)),
                "model_input_prefix_mean_std_m": float(
                    np.mean(parent_model_input_values_m)
                ),
                "model_input_final_std_m": float(parent_model_input_values_m[-1]),
                "valid_prefix_count": len(parent_values_m),
                "residual_frame_count": len(residuals),
            }
        )
    if cursor != len(selected) or np.any(values[~valid] != 0.0):
        raise ValueError("V21d cache construction did not preserve event/mask semantics")
    parents = pd.DataFrame(parent_rows)
    statistics = {
        "split": split,
        "event_count": len(selected),
        "parent_count": len(parents),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_event_fraction": float(valid.mean()),
        "residual_frame_count": int(residual_frame_count),
        "skipped_missing_tr_range_frame_count": int(
            skipped_missing_tr_range_frame_count
        ),
        "stale_but_valid_prefix_event_count": int(stale_prefix_event_count),
        "clipped_event_count": int(clipped_event_count),
        "clipped_parent_ids": sorted(clipped_parent_ids),
        "maximum_raw_prefix_std_m": float(maximum_raw_prefix_std_m),
        "test_payload_access_count": 0,
    }
    return {"sample_ids": sample_ids, "values": values, "valid": valid}, statistics, parents


def _train_diagnostic(
    selected: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    parents: pd.DataFrame,
) -> dict[str, Any]:
    values_m = np.asarray(arrays["values"], dtype=np.float64)[:, 0] * D_RANGE_STD_CLIP_M
    valid = np.asarray(arrays["valid"], dtype=np.bool_)[:, 0]
    distributions: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        event_mask = selected["label_name"].astype(str).eq(label).to_numpy()
        group = parents[parents["label_name"].eq(label)]
        distributions[label] = {
            "parent_support": int(len(group)),
            "valid_causal_prefix_event": _summary(values_m[event_mask & valid]),
            "parent_prefix_mean_std_m": _summary(
                group["prefix_mean_std_m"].to_numpy(dtype=np.float64)
            ),
            "parent_final_std_m": _summary(
                group["final_std_m"].to_numpy(dtype=np.float64)
            ),
            "parent_model_input_prefix_mean_std_m": _summary(
                group["model_input_prefix_mean_std_m"].to_numpy(dtype=np.float64)
            ),
            "parent_model_input_final_std_m": _summary(
                group["model_input_final_std_m"].to_numpy(dtype=np.float64)
            ),
        }
    clutter = parents[parents["label_name"].eq("ClutterTarget")]
    separations: dict[str, Any] = {}
    for other_label in ("DroneTarget", "BirdTarget", "BalloonTarget"):
        other = parents[parents["label_name"].eq(other_label)]
        rows: dict[str, Any] = {}
        for field in ("prefix_mean_std_m", "final_std_m"):
            clutter_values = clutter[field].to_numpy(dtype=np.float64)
            other_values = other[field].to_numpy(dtype=np.float64)
            clutter_higher = _auc(clutter_values, other_values)
            rows[field] = {
                "clutter_higher_auc": clutter_higher,
                "other_higher_auc": float(1.0 - clutter_higher),
                "direction_free_separation_auc": float(
                    max(clutter_higher, 1.0 - clutter_higher)
                ),
                "clutter_minus_other_mean_m": float(
                    np.mean(clutter_values) - np.mean(other_values)
                ),
            }
        separations[f"ClutterTarget_vs_{other_label}"] = rows
    bird_mean = float(
        distributions["BirdTarget"]["parent_model_input_final_std_m"]["mean"]
    )
    clutter_mean = float(
        distributions["ClutterTarget"]["parent_model_input_final_std_m"]["mean"]
    )
    return {
        "role": (
            "train-only physical attribution; direction is disclosed but does not "
            "become a readiness or performance gate"
        ),
        "parent_label_distributions": distributions,
        "parent_separation": separations,
        "interpretation_frozen_before_treatment_validation": {
            "historical_representative_bird_higher_direction_reproduced_on_train_final_means": (
                bird_mean > clutter_mean
            ),
            "train_model_input_final_bird_mean_m_after_75m_clip": bird_mean,
            "train_model_input_final_clutter_mean_m_after_75m_clip": clutter_mean,
            "train_direction_is_opposite_to_historical_two_track_probe": (
                bird_mean < clutter_mean
            ),
            "clutter_bird_final_direction_free_auc": float(
                separations["ClutterTarget_vs_BirdTarget"]["final_std_m"][
                    "direction_free_separation_auc"
                ]
            ),
            "formula_changed_after_this_diagnostic": False,
        },
    }


def _old_proxy_train_diagnostic(
    selected: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
        values = np.asarray(payload["values"], dtype=np.float64)
        valid = np.asarray(payload["valid"], dtype=np.bool_)
    if sample_ids != selected["sample_id"].astype(str).tolist() or values.shape[1] != 6:
        raise ValueError("V21d old core proxy cache no longer aligns train events")
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        mask = selected["label_name"].astype(str).eq(label).to_numpy() & valid[:, 5]
        result[label] = _summary(values[mask, 5] * D_RANGE_STD_CLIP_M)
    return {
        "scope": "train current-event proxy retained only for mismatch attribution",
        "formula": (
            "std(MTD detected range - one constant current-event TR range) within each event; "
            "mathematically equal to current-event std(MTD detected range)"
        ),
        "event_distributions_m": result,
        "superseded_for_formal_training": True,
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
        "RADAR-LN3-V21d-cq08-liquid-known-gate-causal-prefix-d-range-"
        "clutter-bird-seed43"
    )
    payload["variant"] = V21D_PREFIX_VARIANT
    payload["task"]["feature_names"] = list(D_RANGE_PREFIX_FEATURE_NAMES)
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
                "RADAR-LN3-V21-v21d_prefix-clutter-bird-seed43-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": BASELINE_CONFIG.as_posix(),
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "core_feature_manifest": root / CORE_FEATURE_MANIFEST,
        "core_train_cache": root / CORE_TRAIN_CACHE,
        "baseline_config": root / BASELINE_CONFIG,
        "baseline_summary": root / BASELINE_ARTIFACT / "training-summary.json",
        "baseline_checkpoint": root / BASELINE_ARTIFACT / "best_model.pt",
        "selection": root / BASELINE_SELECTION,
        "historical_probe": root / HISTORICAL_PROBE,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
        "old_seed42_config": root / OLD_SEED42_CONFIG,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21d source files are missing: {missing}")
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
        raise ValueError("V21d selected seed43 Baseline binding is stale")
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
        raise ValueError("V21d current code cannot reconstruct seed43 Baseline initial state")
    del rebuilt

    core_contract = json.loads(
        source_paths["core_contract"].read_text(encoding="utf-8")
    )
    if core_contract.get("schema_version") != "RADAR-LN3-V21-known-gate-data-contract-v1":
        raise ValueError("V21d core KnownGate contract changed")
    selected = {
        split: _selected_index(source_paths["index"], split)
        for split in ("train", "val")
    }
    frame_lookup, frame_audit = _frame_lookup(source_paths["canonical_frame_index"])
    arrays: dict[str, dict[str, np.ndarray]] = {}
    split_stats: dict[str, Any] = {}
    parent_tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        arrays[split], split_stats[split], parent_tables[split] = _build_split(
            selected[split], frame_lookup, split=split
        )
    train_diagnostic = _train_diagnostic(
        selected["train"], arrays["train"], parent_tables["train"]
    )
    old_proxy = _old_proxy_train_diagnostic(
        selected["train"], source_paths["core_train_cache"]
    )

    contract = {
        "schema_version": (
            "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-"
            "clutter-bird-experiment-contract-v1"
        ),
        "experiment_name": "V21d causal-prefix d-range stability for Clutter versus Bird",
        "internal_variant": V21D_PREFIX_VARIANT,
        "feature_names": list(D_RANGE_PREFIX_FEATURE_NAMES),
        "source": (
            "sealed canonical frame provenance target_range_reference_m, equivalent to "
            "MAT debug_info.detected_range_m, plus current V16 TR filtered range_m"
        ),
        "formula": (
            "sample std(ddof=1) of every observed frame-level "
            "(MTD detected_range_m - its current-event TR filtered range_m) in the causal prefix; "
            "divide by 75 m and clip to [0,1]"
        ),
        "causal_scope": (
            "current and prior events only; each event receives the statistic after adding all "
            "valid frames from that event; no future event or parent-final value is read"
        ),
        "minimum_evidence": "two valid frame-level residual observations",
        "missing_policy": (
            "missing RD or TR range contributes no new residual; an already-valid prefix statistic "
            "is carried forward; before two valid residuals the value is zero and invalid"
        ),
        "normalization": {"clip_m": D_RANGE_STD_CLIP_M, "ddof": 1},
        "old_v21d_proxy": old_proxy,
        "experiment_document": str(source_paths["experiment_document"]),
        "experiment_document_sha256": source_hashes["experiment_document"],
        "historical_probe": {
            "script": str(source_paths["historical_probe"]),
            "script_sha256": source_hashes["historical_probe"],
            "not_formal_population_evidence": True,
            "bird_track": 333,
            "bird_split": "val",
            "bird_first_nine_mat_frame_sample_std_m": 12.414397991498015,
            "clutter_track": 1868,
            "clutter_split": "test",
            "clutter_first_nine_mat_frame_sample_std_m": 1.9254072238454454,
            "manual_reproduction_unique_test_parent_count": 1,
            "manual_reproduction_unique_test_mat_payload_count": 54,
            "manual_reproduction_model_test_evaluation_executed": False,
            "disclosure": (
                "the quoted 12.4/1.9 values are two representative tracks, not class means; "
                "validation and test were observed during feature conception, and this audit "
                "re-read all 54 MAT frames of the already-exposed test track 1868 only to "
                "verify field semantics; no model test evaluation was run"
            ),
        },
        "core_data_contract": str(source_paths["core_contract"]),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "source_index": str(source_paths["index"]),
        "source_index_sha256": source_hashes["index"],
        "source_canonical_frame_index": str(source_paths["canonical_frame_index"]),
        "source_canonical_frame_index_sha256": source_hashes["canonical_frame_index"],
        "raw_echo_payload_access_count": 0,
        "mat_payload_access_count": 0,
        "rd_tensor_payload_access_count": 0,
        "validation_feature_payload_access_count_for_readiness": 0,
        "test_payload_access_count": 0,
        "test_payload_access_count_scope": (
            "automated prepare only; historical/manual feature-conception audit exposure is "
            "reported separately and makes the original V16 test non-pristine"
        ),
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
                "same seed43 random Baseline initialization; copy every shared tensor and zero "
                "only the one new Fusion column; do not fine-tune the trained Baseline"
            ),
            "epochs": 20,
            "full_parameter_training": True,
            "all_epoch_parent_orders_must_match": True,
        },
        "evaluation": {
            "primary": "development validation parent-final KnownGate macro_f1 delta",
            "adoption_delta_strictly_greater_than": 0.02,
            "research_question": (
                "does causal-prefix MTD-minus-TR range sample standard deviation improve the "
                "ClutterTarget versus BirdTarget KnownGate boundary?"
            ),
            "primary_hypothesis_labels": ["BirdTarget", "ClutterTarget"],
            "safety_labels": ["DroneTarget", "BalloonTarget"],
            "primary_pair_must_have_positive_net_rescue": True,
            "each_primary_label_must_have_non_negative_net_rescue": True,
            "each_safety_label_must_have_non_negative_net_rescue": True,
            "clutter_validation_support": 9,
            "bird_validation_support": 56,
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
            "reason": (
                "it is cross-seed and uses the mathematically degenerate current-event proxy"
            ),
        },
        "validation_is_pristine_for_feature_conception": False,
        "test_is_pristine_for_feature_conception": False,
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
        "schema_version": KNOWN_GATE_V21D_MANIFEST_SCHEMA,
        "feature_schema": D_RANGE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(D_RANGE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
        "source_canonical_frame_index_sha256": source_hashes[
            "canonical_frame_index"
        ],
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
        "frame_index_is_model_input": False,
        "valid_mask_is_model_input": False,
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
        config.variant != V21D_PREFIX_VARIANT
        or config.train.seed != SEED
        or config.train.paired_baseline_config_path != BASELINE_CONFIG.as_posix()
        or paired_baseline is None
        or paired.get("shared_parameters_and_buffers_exactly_equal") is not True
        or paired.get("new_fusion_columns_exactly_zero") is not True
        or int(paired.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V21d generated config failed the seed43 pair contract")
    del treatment, paired_baseline

    checks = {
        "selected_development_baseline_is_seed43_and_sha_bound": True,
        "current_code_reconstructs_formal_seed43_initial_state_sha": True,
        "historical_12_4_vs_1_9_values_are_disclosed_as_two_tracks_not_class_means": True,
        "historical_validation_and_test_feature_conception_exposure_is_disclosed": True,
        "manual_test_mat_field_semantics_recheck_is_disclosed_separately_from_prepare": True,
        "old_current_event_proxy_is_mathematically_a_different_quantity": True,
        "old_seed42_v21d_candidate_is_explicitly_non_comparable": True,
        "formal_feature_is_frame_level_mtd_minus_its_event_tr_range": True,
        "sample_standard_deviation_uses_ddof_1_and_frozen_75m_clip": True,
        "prefix_feature_uses_only_current_and_prior_events": True,
        "fewer_than_two_valid_residuals_is_invalid_zero": True,
        "independent_train_validation_cache_aligns_every_known_event": True,
        "no_raw_mat_or_rd_tensor_payload_is_read_during_prepare": True,
        "train_only_direction_diagnostic_is_recorded_without_becoming_a_gate": True,
        "validation_feature_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_cache_and_training_population": True,
        "test_payload_access_count_is_zero": True,
        "treatment_copies_shared_initialization_and_zeroes_only_one_new_column": True,
        "full_parameter_20_epoch_same_seed_pair_is_required": True,
        "clutter_bird_are_the_only_primary_hypothesis_labels": True,
        "drone_balloon_are_safety_labels_only": True,
    }
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_V21D_READINESS_SCHEMA,
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
            "canonical_frame_index_sha256": source_hashes["canonical_frame_index"],
        },
        "frame_provenance_audit": frame_audit,
        "split_cache_statistics": split_stats,
        "train_only_feature_diagnostic": train_diagnostic,
        "old_proxy_train_only_diagnostic": old_proxy,
        "validation_used_for_readiness_gate": False,
        "validation_feature_values_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "test_payload_access_count_scope": "automated prepare only",
        "manual_feature_audit_exposure": {
            "test_parent_ids": ["cq08|track|1868"],
            "unique_test_mat_payload_count": 54,
            "model_test_evaluation_executed": False,
        },
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
        "schema_version": KNOWN_GATE_V21D_READINESS_SCHEMA,
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
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the selected-seed43 V21d causal-prefix d-range experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
