from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _as_bool,
    _parse_indices,
    _resolve,
    _sha256,
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_GATE_V21A_MANIFEST_SCHEMA,
    KNOWN_GATE_V21A_READINESS_SCHEMA,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    INTERFRAME_SPECTRAL_FEATURE_NAMES,
    INTERFRAME_SPECTRAL_FEATURE_SCHEMA,
    V21A_INTERFRAME_VARIANT,
    extract_interframe_spectral_temporal_consistency,
)


SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
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
    "docs/radar_ln3_v21a_interframe_spectral_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21a-cq08-known-gate-interframe-spectral-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V21a-cq08-known-gate-interframe-spectral-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21a-cq08-liquid-known-gate-interframe-spectral-consistency-seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21a-known-gate-interframe-spectral-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V21a-cq08-liquid-known-gate-interframe-spectral-consistency-seed43-formal"
)
SPLITS = ("train", "val")
SEED = 43


def feature_cache_relative(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"V21a cache split must be train/val: {split!r}")
    return Path(
        "data/processed/"
        f"RADAR-LN3-V21a-cq08-known-gate-{split}-interframe-spectral-cache.npz"
    )


def _load_provenance(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        str(row["parent_sample_id"]): dict(row)
        for row in payload.get("records") or []
    }
    if (
        payload.get("schema_version") != "RADAR-LN3-V16"
        or int(payload.get("cross_label_content_conflict_count", -1)) != 0
        or len(records) != int(payload.get("parent_count", -1))
    ):
        raise ValueError("V21a source provenance is not the clean complete V16 contract")
    return records, payload


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"amplitude", "velocity_mps"}.issubset(payload.files):
            raise ValueError(f"V21a canonical tensor lacks amplitude/velocity: {path}")
        amplitude = np.asarray(payload["amplitude"])
        velocity = np.asarray(payload["velocity_mps"])
    if (
        amplitude.ndim != 3
        or amplitude.shape[1:] != (21, 384)
        or not np.isfinite(amplitude).all()
    ):
        raise ValueError(f"V21a canonical amplitude is not finite [frames,21,384]: {path}")
    if velocity.shape != (384,) or not np.isfinite(velocity).all():
        raise ValueError(f"V21a canonical velocity is not finite [384]: {path}")
    return amplitude, velocity


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not pos.size or not neg.size:
        return None
    combined = np.concatenate([pos, neg])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum = float(ranks[: pos.size].sum())
    return float(
        (rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


def _summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "support": int(data.size),
        "mean": float(np.mean(data)) if data.size else None,
        "median": float(np.median(data)) if data.size else None,
        "q25": float(np.quantile(data, 0.25)) if data.size else None,
        "q75": float(np.quantile(data, 0.75)) if data.size else None,
    }


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _historical_exploration_split_audit(index: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the original exploratory sample without reading any payload."""

    rows: list[dict[str, Any]] = []
    for label, count in (
        ("ClutterTarget", 33),
        ("DroneTarget", 25),
        ("BirdTarget", 25),
        ("BalloonTarget", 25),
    ):
        rng = np.random.RandomState(3)
        tracks = index[index["label_name"].astype(str).eq(label)]["track_id"].astype(int).unique()
        for track_id in rng.choice(tracks, size=min(count, len(tracks)), replace=False):
            matches = index[pd.to_numeric(index["track_id"], errors="raise").astype(int).eq(int(track_id))]
            splits = matches["split_name"].astype(str).unique().tolist()
            if len(splits) != 1:
                raise ValueError("V21a historical exploration track crosses split metadata")
            rows.append(
                {"label_name": label, "track_id": int(track_id), "split_name": splits[0]}
            )
    frame = pd.DataFrame(rows)
    counts = {
        str(key): int(value)
        for key, value in frame["split_name"].value_counts().sort_index().items()
    }
    by_label_split = {
        label: {
            str(key): int(value)
            for key, value in group["split_name"].value_counts().sort_index().items()
        }
        for label, group in frame.groupby("label_name", sort=True)
    }
    expected_by_label_split = {
        "BalloonTarget": {"test": 4, "train": 18, "val": 3},
        "BirdTarget": {"test": 3, "train": 21, "val": 1},
        "ClutterTarget": {"test": 6, "train": 20, "val": 7},
        "DroneTarget": {"test": 1, "train": 18, "val": 6},
    }
    if counts != {"test": 14, "train": 77, "val": 17}:
        raise ValueError(
            "V21a historical feature-conception exposure audit changed: "
            f"expected train/val/test=77/17/14, actual={counts}"
        )
    if by_label_split != expected_by_label_split:
        raise ValueError(
            "V21a historical feature-conception label/split audit changed: "
            f"expected={expected_by_label_split}, actual={by_label_split}"
        )
    return {
        "source": (
            "reconstruction of the original exploratory sample() function, which resets "
            "RandomState(3) independently for each label"
        ),
        "metadata_only_reconstruction": True,
        "payload_read_during_current_prepare": False,
        "sampled_track_count": int(len(frame)),
        "split_track_counts": counts,
        "label_split_track_counts": by_label_split,
        "sampled_test_track_ids": sorted(
            frame.loc[frame["split_name"].eq("test"), "track_id"].astype(int).tolist()
        ),
        "consequence": (
            "the original V16 test split is feature-conception-exposed for V21a and cannot serve "
            "as an unbiased final holdout for this feature family"
        ),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    first = pd.Series(np.asarray(left)).rank(method="average").to_numpy(dtype=np.float64)
    second = pd.Series(np.asarray(right)).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _statistics(
    selected: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    comparison_pair_counts: np.ndarray,
    *,
    include_label_diagnostics: bool,
) -> dict[str, Any]:
    pair_counts = np.asarray(comparison_pair_counts, dtype=np.int64)
    if pair_counts.shape != (len(selected),) or np.any(pair_counts < 0):
        raise ValueError("V21a comparison-pair counts must be non-negative [events]")
    active = valid.all(axis=1)
    if np.any((pair_counts > 0) != active):
        raise ValueError("V21a feature validity must exactly match positive comparison-pair support")
    result: dict[str, Any] = {
        "event_count": int(len(selected)),
        "valid_event_count": int(active.sum()),
        "invalid_event_count": int((~active).sum()),
        "valid_event_fraction": float(active.mean()) if len(active) else 0.0,
        "feature_names": list(INTERFRAME_SPECTRAL_FEATURE_NAMES),
        "comparison_pair_count": _summary(pair_counts[active]),
    }
    if not include_label_diagnostics:
        return result

    labels = selected["label_name"].astype(str).to_numpy()
    frame_counts = (
        pd.to_numeric(selected["rd_frame_count"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    event_distributions: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        mask = active & (labels == label)
        event_distributions[label] = {
            "event_support": int(mask.sum()),
            "event_valid_fraction": float(mask.sum() / max(1, int((labels == label).sum()))),
            "rd_frame_count": _summary(frame_counts[mask]),
            "comparison_pair_count": _summary(pair_counts[mask]),
            "features": {
                name: _summary(values[mask, index])
                for index, name in enumerate(INTERFRAME_SPECTRAL_FEATURE_NAMES)
            },
        }

    parent_rows: list[dict[str, Any]] = []
    for parent_id, group in selected.assign(_row=np.arange(len(selected))).groupby(
        "parent_sample_id", sort=True
    ):
        row_indices = group["_row"].to_numpy(dtype=np.int64)
        parent_active = active[row_indices]
        row: dict[str, Any] = {
            "parent_sample_id": str(parent_id),
            "label_name": str(group["label_name"].iloc[0]),
            "valid_event_count": int(parent_active.sum()),
            "mean_rd_frame_count": float(frame_counts[row_indices][parent_active].mean())
            if parent_active.any()
            else None,
            "mean_comparison_pair_count": float(
                pair_counts[row_indices][parent_active].mean()
            )
            if parent_active.any()
            else None,
        }
        for feature_index, name in enumerate(INTERFRAME_SPECTRAL_FEATURE_NAMES):
            row[name] = (
                float(values[row_indices[parent_active], feature_index].mean())
                if parent_active.any()
                else None
            )
        parent_rows.append(row)
    parents = pd.DataFrame(parent_rows)
    if parents[list(INTERFRAME_SPECTRAL_FEATURE_NAMES)].isna().any(axis=None):
        raise ValueError("V21a train contains a parent without valid inter-frame evidence")

    parent_distributions: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        group = parents[parents["label_name"].eq(label)]
        parent_distributions[label] = {
            "parent_support": int(len(group)),
            "features": {
                name: _summary(group[name].to_numpy(dtype=np.float64))
                for name in INTERFRAME_SPECTRAL_FEATURE_NAMES
            },
        }

    separation: dict[str, Any] = {}
    clutter = parents[parents["label_name"].eq("ClutterTarget")]
    expected_clutter_direction = {
        INTERFRAME_SPECTRAL_FEATURE_NAMES[0]: "higher",
        INTERFRAME_SPECTRAL_FEATURE_NAMES[1]: "higher",
        INTERFRAME_SPECTRAL_FEATURE_NAMES[2]: "lower",
        INTERFRAME_SPECTRAL_FEATURE_NAMES[3]: "lower",
        INTERFRAME_SPECTRAL_FEATURE_NAMES[4]: "lower",
        INTERFRAME_SPECTRAL_FEATURE_NAMES[5]: "higher",
    }
    for other_label in ("BirdTarget", "BalloonTarget"):
        other = parents[parents["label_name"].eq(other_label)]
        rows: dict[str, Any] = {}
        for name in INTERFRAME_SPECTRAL_FEATURE_NAMES:
            auc = _auc(
                clutter[name].to_numpy(dtype=np.float64),
                other[name].to_numpy(dtype=np.float64),
            )
            rows[name] = {
                "clutter_higher_auc": auc,
                "direction_free_separation_auc": None
                if auc is None
                else float(max(auc, 1.0 - auc)),
                "expected_clutter_direction": expected_clutter_direction[name],
                "expected_direction_auc": None
                if auc is None
                else float(
                    auc
                    if expected_clutter_direction[name] == "higher"
                    else 1.0 - auc
                ),
                "clutter_minus_other_mean": float(clutter[name].mean() - other[name].mean()),
            }
        separation[f"ClutterTarget_vs_{other_label}"] = rows

    result["train_only_event_label_diagnostics"] = event_distributions
    result["train_only_parent_label_diagnostics"] = parent_distributions
    result["train_only_parent_separation"] = separation
    result["train_only_event_frame_count_confounding"] = {
        name: {
            "spearman_with_current_event_rd_frame_count": _spearman(
                frame_counts[active], values[active, index]
            ),
            "spearman_with_actual_adjacent_comparison_pair_count": _spearman(
                pair_counts[active], values[active, index]
            ),
        }
        for index, name in enumerate(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    }
    return result


def _build_split(
    root: Path,
    split: str,
    index: pd.DataFrame,
    provenance: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V21a {split} known-event population is empty or duplicated")
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    feature_dim = len(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    values = np.zeros((len(selected), feature_dim), dtype=np.float32)
    valid = np.zeros((len(selected), feature_dim), dtype=np.bool_)
    comparison_pair_counts = np.zeros((len(selected),), dtype=np.int64)
    cursor = 0
    canonical_access_count = 0
    for parent_number, (parent_id_value, parent) in enumerate(
        selected.groupby("parent_sample_id", sort=True), start=1
    ):
        parent_id = str(parent_id_value)
        labels = parent["label_name"].astype(str).unique().tolist()
        paths = [
            str(value)
            for value in parent.loc[parent["rd_present"].map(_as_bool), "rd_tensor_path"]
            .dropna()
            .unique()
            if str(value).strip()
        ]
        if len(labels) != 1 or len(paths) > 1:
            raise ValueError(f"V21a parent label/RD identity is not unique: {parent_id}")
        source = provenance.get(parent_id)
        if (
            source is None
            or source.get("v16_split_name") != split
            or source.get("label_name") != labels[0]
        ):
            raise ValueError(f"V21a parent provenance differs: {parent_id}")
        amplitude: np.ndarray | None = None
        velocity: np.ndarray | None = None
        if paths:
            rd_path = _resolve(root, paths[0])
            if (
                rd_path != Path(str(source["rd_tensor_path"])).resolve()
                or _sha256(rd_path) != str(source["rd_sha256"])
            ):
                raise ValueError(f"V21a canonical RD path/hash differs: {parent_id}")
            amplitude, velocity = _load_canonical(rd_path)
            canonical_access_count += 1
        used_tensor_indices: set[int] = set()
        previous_valid_frame: np.ndarray | None = None
        for row in parent.itertuples(index=False):
            if not _as_bool(row.rd_present):
                event_values = np.zeros((feature_dim,), dtype=np.float32)
                event_valid = np.zeros((feature_dim,), dtype=np.bool_)
                previous_valid_frame = None
            else:
                tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
                if (
                    amplitude is None
                    or velocity is None
                    or len(tensor_indices) != int(row.rd_frame_count)
                    or not tensor_indices
                    or min(tensor_indices) < 0
                    or max(tensor_indices) >= amplitude.shape[0]
                ):
                    raise ValueError(f"V21a event tensor mapping is invalid: {row.sample_id}")
                overlap = used_tensor_indices & set(tensor_indices)
                if overlap:
                    raise ValueError(
                        f"V21a tensor frame reused across events: {parent_id}/{sorted(overlap)}"
                    )
                used_tensor_indices.update(tensor_indices)
                current_frames = amplitude[np.asarray(tensor_indices, dtype=np.int64)]
                comparison_frames = (
                    current_frames
                    if previous_valid_frame is None
                    else np.concatenate(
                        [previous_valid_frame[np.newaxis, ...], current_frames], axis=0
                    )
                )
                event_values, event_valid = extract_interframe_spectral_temporal_consistency(
                    comparison_frames,
                    velocity,
                )
                comparison_pair_counts[cursor] = max(0, int(comparison_frames.shape[0]) - 1)
                previous_valid_frame = np.asarray(current_frames[-1]).copy()
            values[cursor] = event_values
            valid[cursor] = event_valid
            cursor += 1
        if parent_number == 1 or parent_number % 100 == 0:
            print(
                f"[V21a prepare] split={split} parent={parent_number} events={cursor}",
                flush=True,
            )
    if cursor != len(selected):
        raise ValueError("V21a extraction did not consume every selected event")
    statistics = {
        "parent_count": int(selected["parent_sample_id"].nunique()),
        "parent_label_counts": {
            str(key): int(value)
            for key, value in selected.drop_duplicates("parent_sample_id")[
                "label_name"
            ].value_counts().sort_index().items()
        },
        "canonical_rd_tensor_access_count": canonical_access_count,
        "raw_echo_access_count": 0,
        "test_payload_access_count": 0,
        **_statistics(
            selected,
            values,
            valid,
            comparison_pair_counts,
            include_label_diagnostics=(split == "train"),
        ),
    }
    return {"sample_ids": sample_ids, "values": values, "valid": valid}, statistics


def _clone(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _config_payload(
    baseline_payload: Mapping[str, Any],
    *,
    contract_hash: str,
    manifest_hash: str,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = _clone(baseline_payload)
    payload["config_name"] = (
        "RADAR-LN3-V21a-cq08-liquid-known-gate-"
        "interframe-spectral-consistency-seed43"
    )
    payload["variant"] = V21A_INTERFRAME_VARIANT
    payload["task"]["feature_names"] = list(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": feature_cache_relative("train").as_posix(),
            "validation_feature_cache_path": feature_cache_relative("val").as_posix(),
            "expected_train_feature_cache_sha256": cache_hashes["train"],
            "expected_validation_feature_cache_sha256": cache_hashes["val"],
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = len(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    artifact = ARTIFACT.as_posix()
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V21-v21a_interframe-seed43-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": BASELINE_CONFIG.as_posix(),
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "provenance": root / SOURCE_PROVENANCE,
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
        raise FileNotFoundError(f"V21a source files are missing: {missing}")
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
        raise ValueError("V21a selected seed43 Baseline binding is stale or incomplete")
    baseline_config_payload = json.loads(
        source_paths["baseline_config"].read_text(encoding="utf-8")
    )
    baseline_config = load_known_gate_config(source_paths["baseline_config"])
    if baseline_config.variant != "baseline" or baseline_config.train.seed != SEED:
        raise ValueError("V21a paired reference is not the selected seed43 Baseline")
    rebuilt_baseline, rebuilt_reference, rebuilt_evidence = (
        build_known_gate_model_for_config(root, baseline_config)
    )
    rebuilt_initial_state_sha256 = _state_dict_sha256(rebuilt_baseline.state_dict())
    if (
        rebuilt_reference is not None
        or rebuilt_evidence.get("paired_baseline") is not False
        or rebuilt_initial_state_sha256
        != str(baseline_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError(
            "V21a current code no longer reconstructs the formal seed43 Baseline initial state"
        )
    del rebuilt_baseline

    index = pd.read_csv(source_paths["index"], low_memory=False)
    if set(index["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V21a source index schema changed")
    historical_exploration = _historical_exploration_split_audit(index)
    provenance, provenance_payload = _load_provenance(source_paths["provenance"])
    test_policy = dict(provenance_payload.get("test_payload_use") or {})
    if (
        test_policy.get("test_values_used_for_normalization") is not False
        or test_policy.get("test_used_for_model_selection") is not False
    ):
        raise ValueError("V21a source provenance does not keep test sealed")

    cache_hashes: dict[str, str] = {}
    split_statistics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        print(f"[V21a prepare] extracting {split} inter-frame spectral cache", flush=True)
        arrays, statistics = _build_split(root, split, index, provenance)
        path = root / feature_cache_relative(split)
        _write_npz_new_or_identical(
            path, tuple((name, value) for name, value in arrays.items())
        )
        cache_hashes[split] = sha256_file(path)
        split_statistics[split] = statistics

    expected_parent_counts = {"train": 986, "val": 232}
    if {
        split: int(split_statistics[split]["parent_count"]) for split in SPLITS
    } != expected_parent_counts:
        raise ValueError("V21a known parent support changed")
    train_diagnostic = split_statistics["train"]
    separation = dict(train_diagnostic["train_only_parent_separation"])
    separation_gate: dict[str, Any] = {}
    for comparison in (
        "ClutterTarget_vs_BirdTarget",
        "ClutterTarget_vs_BalloonTarget",
    ):
        feature_rows = dict(separation[comparison])
        values = {
            name: float(row["expected_direction_auc"])
            for name, row in feature_rows.items()
            if row.get("expected_direction_auc") is not None
        }
        best_name, best_auc = max(values.items(), key=lambda item: item[1])
        separation_gate[comparison] = {
            "threshold": 0.65,
            "best_feature": best_name,
            "best_expected_direction_auc": best_auc,
            "passed": best_auc >= 0.65,
        }
    if not all(row["passed"] for row in separation_gate.values()):
        raise ValueError(
            "V21a train-only expected-direction separation gate failed: "
            f"{separation_gate}"
        )
    checks = {
        "selected_development_baseline_is_seed43_and_sha_bound": True,
        "current_code_reconstructs_formal_seed43_initial_state_sha": True,
        "feature_matches_original_adjacent_rd_frame_semantics": True,
        "feature_uses_only_current_and_immediately_previous_canonical_rd_frames": True,
        "missing_rd_event_resets_previous_frame_bridge": True,
        "intra_frame_stft_window_surrogate_is_not_used": True,
        "events_without_two_causal_comparison_frames_are_invalid_not_fabricated": True,
        "train_and_validation_cache_sample_order_is_exact": True,
        "only_predeclared_train_auc_gate_is_binding_other_diagnostics_are_advisory": True,
        "train_only_clutter_bird_expected_direction_auc_at_least_0_65": True,
        "train_only_clutter_balloon_expected_direction_auc_at_least_0_65": True,
        "unknown_is_excluded_from_cache_and_training_population": True,
        "raw_echo_payload_access_count_is_zero": True,
        "validation_feature_values_are_not_a_readiness_gate": True,
        "test_payload_access_count_is_zero": True,
        "historical_feature_conception_test_exposure_is_disclosed": True,
        "full_parameter_20_epoch_same_seed_pair_is_required": True,
    }
    contract = {
        "schema_version": (
            "RADAR-LN3-V21a-known-gate-interframe-spectral-experiment-contract-v1"
        ),
        "experiment_name": "V21a inter-frame spectral-shape temporal consistency",
        "internal_variant": V21A_INTERFRAME_VARIANT,
        "feature_schema": INTERFRAME_SPECTRAL_FEATURE_SCHEMA,
        "feature_names": list(INTERFRAME_SPECTRAL_FEATURE_NAMES),
        "formula": (
            "prepend the immediately preceding valid event's final canonical RD frame to the "
            "current event (reset on an RD-missing event), convert dB amplitude to relative linear "
            "power and sum 21 range rows into a 384-bin velocity profile; assign adjacent-frame "
            "cosine mean/minimum, adjacent peak-velocity drift mean/maximum divided by the "
            "canonical velocity span, and near-zero (+/-2 m/s) energy-fraction std/minimum to the "
            "current causal event"
        ),
        "causal_scope": (
            "current event plus at most the immediately preceding valid event's final RD frame; "
            "no future event or parent-final aggregation"
        ),
        "minimum_support": (
            "at least two comparison frames from current frames plus the optional causal bridge"
        ),
        "invalid_policy": "zero values plus internal gating mask; invalid mask is not concatenated",
        "supersedes_candidate": {
            "variant": "v21a",
            "reason": (
                "the seed42 candidate measured intra-frame adjacent STFT-window affinity and did "
                "not implement the original inter-frame spectral-consistency diagnostic"
            ),
            "formal_training_allowed": False,
        },
        "source_hashes": source_hashes,
        "historical_feature_conception_exposure": historical_exploration,
        "cache_hashes": cache_hashes,
        "split_statistics": split_statistics,
        "paired_protocol": {
            "seed": SEED,
            "baseline_config": str(source_paths["baseline_config"]),
            "baseline_config_sha256": source_hashes["baseline_config"],
            "baseline_summary": str(source_paths["baseline_summary"]),
            "baseline_summary_sha256": source_hashes["baseline_summary"],
            "baseline_checkpoint": str(source_paths["baseline_checkpoint"]),
            "baseline_checkpoint_sha256": source_hashes["baseline_checkpoint"],
            "baseline_initial_model_state_sha256": rebuilt_initial_state_sha256,
            "initialization": (
                "same seed43 random baseline initialization; copy all shared tensors and zero only "
                "the new Fusion columns; do not fine-tune the trained Baseline checkpoint"
            ),
            "epochs": 20,
            "full_parameter_training": True,
        },
        "train_only_feature_diagnostic": {
            "role": (
                "train-only physical readiness and attribution; no validation value or learned "
                "decision threshold is consumed"
            ),
            "expected_direction_separation_gate": separation_gate,
            "statistics": {
                key: value
                for key, value in train_diagnostic.items()
                if key.startswith("train_only_")
            },
        },
        "evaluation": {
            "primary": "development validation parent-final KnownGate macro_f1 delta",
            "adoption_delta_strictly_greater_than": 0.02,
            "expected_labels": ["ClutterTarget", "BirdTarget", "BalloonTarget"],
            "clutter_validation_support": 9,
            "parent_stratified_bootstrap_replicates": 5000,
            "test": "未执行",
            "OOF": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "full_cascade": "未执行",
        },
    }
    _write_json_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    manifest = {
        "schema_version": KNOWN_GATE_V21A_MANIFEST_SCHEMA,
        "feature_schema": INTERFRAME_SPECTRAL_FEATURE_SCHEMA,
        "feature_names": list(INTERFRAME_SPECTRAL_FEATURE_NAMES),
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": source_hashes["core_contract"],
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "cache_files": {
            split: {
                "path": feature_cache_relative(split).as_posix(),
                "sha256": cache_hashes[split],
                "event_count": int(split_statistics[split]["event_count"]),
                "fields": ["sample_ids", "values", "valid"],
            }
            for split in SPLITS
        },
        "sample_id_is_model_input": False,
        "frame_index_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_json_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)
    config_payload = _config_payload(
        baseline_config_payload,
        contract_hash=contract_hash,
        manifest_hash=manifest_hash,
        cache_hashes=cache_hashes,
    )
    _write_json_new_or_identical(root / CONFIG, config_payload)
    config = load_known_gate_config(root / CONFIG)
    if (
        config.variant != V21A_INTERFRAME_VARIANT
        or config.train.seed != SEED
        or config.train.paired_baseline_config_path != BASELINE_CONFIG.as_posix()
    ):
        raise ValueError("V21a generated config failed its selected-seed pair contract")
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_V21A_READINESS_SCHEMA,
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
            "initial_model_state_sha256": rebuilt_initial_state_sha256,
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
        "train_only_feature_diagnostic": contract["train_only_feature_diagnostic"],
        "historical_feature_conception_exposure": historical_exploration,
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
        "schema_version": KNOWN_GATE_V21A_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(root / feature_cache_relative("train")),
        "validation_cache": str(root / feature_cache_relative("val")),
        "paired_baseline_seed": SEED,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_support": 9,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare corrected V21a current-event inter-frame spectral-consistency caches and "
            "the selected-seed43 paired training contract."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    build_known_gate_model_for_config,
