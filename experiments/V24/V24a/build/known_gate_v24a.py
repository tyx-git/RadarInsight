"""Prepare V24a's current-event RD peak-drift counter-evidence experiment.

V24a is a strict one-column append to the completed V24 KnownGate control.
It rebuilds the new column from canonical RD tensors and preserves V24's
course column byte-for-byte.  Unlike the historical V21a cache, no preceding
event frame may be used: events with fewer than two *current* RD frames are
invalid rather than receiving an invented observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from build.micro_doppler_v17 import (
    _as_bool,
    _parse_indices,
    _resolve,
    _sha256,
    _write_npz_new_or_identical,
)
from build.known_gate_v24 import (
    CORE_CONTRACT,
    CORE_NORMALIZATION,
    EXPECTED_CACHE_EVENT_COUNTS,
    SOURCE_INDEX,
    TRAIN_PARENT_COUNT,
    V21E_TRAIN_CACHE,
    V21E_VALIDATION_CACHE,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
    VALIDATION_PARENT_COUNT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V24A_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24A_MANIFEST_SCHEMA,
    KNOWN_GATE_V24A_READINESS_SCHEMA,
    KNOWN_LABELS,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES,
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
    extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction,
)


SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
CONTROL_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-"
    "seed43-config.json"
)
CONTROL_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-contract.json"
)
CONTROL_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-"
    "feature-cache-manifest.json"
)
CONTROL_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness.json"
)
CONTROL_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development.json"
)
CONTROL_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-"
    "baseline-seed43-formal"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v24a_rd_peak_drift_counterevidence_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24a-cq08-known-gate-course-rd-peak-drift-mean-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V24a-cq08-known-gate-course-rd-peak-drift-mean-"
    "feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24a-cq08-liquid-known-gate-course-rd-peak-drift-mean-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24a-cq08-liquid-known-gate-course-rd-peak-drift-"
    "mean-seed43-formal"
)
SEED = 43
SPLITS = ("train", "val")


def feature_cache_relative(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"V24a cache split must be train/val: {split!r}")
    return Path(
        "data/processed/"
        f"RADAR-LN3-V24a-cq08-known-gate-{split}-course-rd-peak-drift-mean-cache.npz"
    )


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V24a preparation artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"amplitude", "velocity_mps"}.issubset(payload.files):
            raise ValueError(f"V24a canonical tensor lacks amplitude/velocity: {path}")
        amplitude = np.asarray(payload["amplitude"])
        velocity = np.asarray(payload["velocity_mps"])
    if (
        amplitude.ndim != 3
        or amplitude.shape[1:] != (21, 384)
        or not np.isfinite(amplitude).all()
        or velocity.shape != (384,)
        or not np.isfinite(velocity).all()
    ):
        raise ValueError(f"V24a canonical tensor is malformed: {path}")
    return amplitude, velocity


def _load_provenance(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        str(row["parent_sample_id"]): dict(row)
        for row in list(payload.get("records") or ())
    }
    if (
        payload.get("schema_version") != "RADAR-LN3-V16"
        or int(payload.get("cross_label_content_conflict_count", -1)) != 0
        or len(records) != int(payload.get("parent_count", -1))
    ):
        raise ValueError("V24a source provenance is not the clean complete V16 contract")
    return records


def _load_course_cache(path: Path, *, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_events = EXPECTED_CACHE_EVENT_COUNTS[split]
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V24a {split} source course cache fields differ")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if (
        sample_ids.shape != (expected_events,)
        or sample_ids.dtype.kind not in {"U", "S"}
        or len({str(value) for value in sample_ids.tolist()}) != expected_events
        or values.shape != (expected_events, 1)
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or not np.isfinite(values).all()
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or bool((values[~valid] != 0.0).any())
    ):
        raise ValueError(f"V24a {split} source course cache is malformed")
    return sample_ids, values, valid


def _summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "support": int(data.size),
        "mean": float(np.mean(data)) if data.size else None,
        "median": float(np.median(data)) if data.size else None,
        "q25": float(np.quantile(data, 0.25)) if data.size else None,
        "q75": float(np.quantile(data, 0.75)) if data.size else None,
    }


def _build_split(
    root: Path,
    split: str,
    index: pd.DataFrame,
    provenance: Mapping[str, Mapping[str, Any]],
    course_cache: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V24a {split} known-event population is empty or duplicated")
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    course_ids, course_values, course_valid = course_cache
    if [str(value) for value in course_ids.tolist()] != sample_ids.tolist():
        raise ValueError(f"V24a {split} source course cache order differs from V16 index")
    values = np.zeros((len(selected), 2), dtype=np.float32)
    valid = np.zeros((len(selected), 2), dtype=np.bool_)
    values[:, 0] = course_values[:, 0]
    valid[:, 0] = course_valid[:, 0]
    rd_frame_counts = pd.to_numeric(
        selected["rd_frame_count"], errors="raise"
    ).to_numpy(dtype=np.int64)
    current_event_frame_count = np.zeros((len(selected),), dtype=np.int64)
    cursor = 0
    canonical_access_count = 0
    for parent_number, (parent_value, parent) in enumerate(
        selected.groupby("parent_sample_id", sort=True), start=1
    ):
        parent_id = str(parent_value)
        labels = parent["label_name"].astype(str).unique().tolist()
        source = provenance.get(parent_id)
        if (
            len(labels) != 1
            or source is None
            or source.get("v16_split_name") != split
            or source.get("label_name") != labels[0]
        ):
            raise ValueError(f"V24a parent provenance differs: {parent_id}")
        paths = [
            str(value)
            for value in parent.loc[
                parent["rd_present"].map(_as_bool), "rd_tensor_path"
            ].dropna().unique()
            if str(value).strip()
        ]
        if len(paths) > 1:
            raise ValueError(f"V24a parent RD identity is not unique: {parent_id}")
        amplitude: np.ndarray | None = None
        velocity: np.ndarray | None = None
        if paths:
            rd_path = _resolve(root, paths[0])
            if (
                rd_path != Path(str(source["rd_tensor_path"])).resolve()
                or _sha256(rd_path) != str(source["rd_sha256"])
            ):
                raise ValueError(f"V24a canonical RD path/hash differs: {parent_id}")
            amplitude, velocity = _load_canonical(rd_path)
            canonical_access_count += 1
        used_tensor_indices: set[int] = set()
        for row in parent.itertuples(index=False):
            expected_count = int(row.rd_frame_count)
            if not _as_bool(row.rd_present):
                if expected_count != 0:
                    raise ValueError(f"V24a RD-missing event has nonzero count: {row.sample_id}")
                event_values = np.zeros((1,), dtype=np.float32)
                event_valid = np.zeros((1,), dtype=np.bool_)
                current_event_frame_count[cursor] = 0
            else:
                tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
                if (
                    amplitude is None
                    or velocity is None
                    or len(tensor_indices) != expected_count
                    or not tensor_indices
                    or min(tensor_indices) < 0
                    or max(tensor_indices) >= amplitude.shape[0]
                ):
                    raise ValueError(f"V24a event tensor mapping is invalid: {row.sample_id}")
                overlap = used_tensor_indices & set(tensor_indices)
                if overlap:
                    raise ValueError(
                        f"V24a tensor frame reused across events: {parent_id}/{sorted(overlap)}"
                    )
                used_tensor_indices.update(tensor_indices)
                current_event_frame_count[cursor] = len(tensor_indices)
                # The full current event is the only possible comparison unit.
                current_frames = amplitude[np.asarray(tensor_indices, dtype=np.int64)]
                event_values, event_valid = (
                    extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction(
                        current_frames, velocity
                    )
                )
            values[cursor, 1] = event_values[0]
            valid[cursor, 1] = event_valid[0]
            cursor += 1
        if parent_number == 1 or parent_number % 100 == 0:
            print(
                f"[V24a prepare] split={split} parent={parent_number} events={cursor}",
                flush=True,
            )
    if cursor != len(selected):
        raise ValueError("V24a extraction did not consume every selected event")
    if not np.array_equal(current_event_frame_count, rd_frame_counts):
        raise AssertionError("V24a current RD frame count audit differs from V16 index")
    drift_valid = valid[:, 1]
    if not np.array_equal(drift_valid, current_event_frame_count >= 2):
        raise AssertionError("V24a invalid policy does not exactly match current RD frame support")
    if bool((values[~valid] != 0.0).any()):
        raise AssertionError("V24a invalid evidence is not the required zero sentinel")
    if not np.array_equal(values[:, 0], course_values[:, 0]) or not np.array_equal(
        valid[:, 0], course_valid[:, 0]
    ):
        raise AssertionError("V24a course column is not byte-for-byte array identical")
    labels = selected["label_name"].astype(str).to_numpy()
    label_stats = {
        label: {
            "event_support": int((labels == label).sum()),
            "drift_valid_event_count": int((drift_valid & (labels == label)).sum()),
            "drift_valid_event_fraction": float(
                (drift_valid & (labels == label)).sum()
                / max(1, int((labels == label).sum()))
            ),
            "drift_mean_fraction": _summary(values[drift_valid & (labels == label), 1]),
        }
        for label in KNOWN_LABELS
    }
    parent_valid = (
        selected.assign(_drift_valid=drift_valid)
        .groupby("parent_sample_id", sort=True)["_drift_valid"]
        .any()
    )
    if not bool(parent_valid.all()):
        missing = parent_valid[~parent_valid].index.astype(str).tolist()
        raise AssertionError(f"V24a parent lacks all valid RD-drift events: {missing[:3]}")
    return {
        "sample_ids": sample_ids,
        "values": values,
        "valid": valid,
    }, {
        "event_count": int(len(selected)),
        "parent_count": int(selected["parent_sample_id"].nunique()),
        "parent_label_counts": {
            str(key): int(value)
            for key, value in selected.drop_duplicates("parent_sample_id")["label_name"]
            .value_counts()
            .sort_index()
            .items()
        },
        "feature_names": list(COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES),
        "course_column_reused_byte_for_byte": True,
        "drift_feature": {
            "current_event_only": True,
            "preceding_event_frame_bridge_used": False,
            "event_rd_frame_count": _summary(current_event_frame_count),
            "valid_event_count": int(drift_valid.sum()),
            "invalid_event_count": int((~drift_valid).sum()),
            "valid_event_fraction": float(drift_valid.mean()),
            "invalid_values_exactly_zero": True,
            "valid_zero_drift_event_count": int(
                (drift_valid & (values[:, 1] == 0.0)).sum()
            ),
            "label_diagnostics": label_stats if split == "train" else {},
        },
        "canonical_rd_tensor_access_count": canonical_access_count,
        "raw_echo_payload_access_count": 0,
        "mat_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


def _config_payload(
    control_payload: Mapping[str, Any],
    *,
    manifest_hash: str,
    contract_hash: str,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = _clone(control_payload)
    artifact = ARTIFACT.as_posix()
    payload["schema_version"] = "RADAR-LN3-V24a"
    payload["config_name"] = (
        "RADAR-LN3-V24a-cq08-liquid-known-gate-course-rd-peak-drift-mean-seed43"
    )
    payload["variant"] = V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
    payload["task"]["feature_names"] = list(COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES)
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
    payload["model"]["evidence_dim"] = len(COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES)
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V24a-v24a_course_rd_peak_drift_mean-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": CONTROL_CONFIG.as_posix(),
        }
    )
    return payload


def _initialization_audit(root: Path, config_path: Path) -> dict[str, Any]:
    control_config = load_known_gate_config(root / CONTROL_CONFIG)
    treatment_config = load_known_gate_config(config_path)
    treatment, control, pairing = build_known_gate_model_for_config(root, treatment_config)
    if control is None:
        raise AssertionError("V24a is missing its fresh V24 paired reference")
    control_state_sha = _state_dict_sha256(control.state_dict())
    treatment_state_sha = _state_dict_sha256(treatment.state_dict())
    if (
        control_config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or treatment_config.variant != V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
        or treatment_config.feature_names
        != COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES
        or treatment_config.model.evidence_dim != 2
        or pairing.get("paired_baseline") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
        or control_state_sha
        != str(
            json.loads(
                (root / CONTROL_ARTIFACT / "training-summary.json").read_text(
                    encoding="utf-8"
                )
            ).get("initial_model_state_sha256", "")
        )
    ):
        raise AssertionError("V24a paired fresh initialization audit failed")
    course_column = treatment.fusion[0].weight[:, -2].detach()
    drift_column = treatment.fusion[0].weight[:, -1].detach()
    if not torch.equal(course_column, torch.zeros_like(course_column)) or not torch.equal(
        drift_column, torch.zeros_like(drift_column)
    ):
        raise AssertionError("V24a course/drift columns are not both exactly zero")
    del treatment, control
    return {
        "v24_reference_initial_model_state_sha256": control_state_sha,
        "v24a_initial_model_state_sha256": treatment_state_sha,
        "fresh_v24_reference_reconstructed": True,
        "trained_v24_checkpoint_loaded": False,
        "trained_v24_checkpoint_accessed": False,
        "course_fusion_column_exactly_zero": True,
        "drift_fusion_column_exactly_zero": True,
        "new_drift_fusion_column_count": 1,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "index": root / SOURCE_INDEX,
        "provenance": root / SOURCE_PROVENANCE,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "course_manifest": root / CONTROL_MANIFEST,
        "course_contract": root / CONTROL_CONTRACT,
        "course_train_cache": root / V21E_TRAIN_CACHE,
        "course_validation_cache": root / V21E_VALIDATION_CACHE,
        "control_config": root / CONTROL_CONFIG,
        "control_summary": root / CONTROL_ARTIFACT / "training-summary.json",
        "control_checkpoint": root / CONTROL_ARTIFACT / "best_model.pt",
        "control_readiness": root / CONTROL_READINESS,
        "control_report": root / CONTROL_REPORT,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V24a frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    control_config_payload = json.loads(paths["control_config"].read_text(encoding="utf-8"))
    control_summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    control_readiness = json.loads(paths["control_readiness"].read_text(encoding="utf-8"))
    control_report = json.loads(paths["control_report"].read_text(encoding="utf-8"))
    course_manifest = json.loads(paths["course_manifest"].read_text(encoding="utf-8"))
    course_contract = json.loads(paths["course_contract"].read_text(encoding="utf-8"))
    if (
        control_config_payload.get("schema_version") != "RADAR-LN3-V24"
        or control_config_payload.get("variant")
        != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or control_summary.get("status") != "completed"
        or control_summary.get("variant")
        != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or int(control_summary.get("seed", -1)) != SEED
        or len(list(control_summary.get("history") or ())) != 20
        or not str(control_summary.get("initial_model_state_sha256", ""))
        or control_readiness.get("status") != "passed"
        or control_report.get("status") != "completed_development_baseline_only"
        or course_manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or course_contract.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or int(course_contract.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V24a V24 control binding is stale or incomplete")
    index = pd.read_csv(paths["index"], low_memory=False)
    if set(index["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V24a source index schema changed")
    provenance = _load_provenance(paths["provenance"])
    cache_hashes: dict[str, str] = {}
    split_statistics: dict[str, dict[str, Any]] = {}
    for split, course_path in (
        ("train", paths["course_train_cache"]),
        ("val", paths["course_validation_cache"]),
    ):
        print(f"[V24a prepare] rebuilding {split} current-event RD peak-drift cache", flush=True)
        arrays, statistics = _build_split(
            root,
            split,
            index,
            provenance,
            _load_course_cache(course_path, split=split),
        )
        output = root / feature_cache_relative(split)
        _write_npz_new_or_identical(
            output, tuple((name, value) for name, value in arrays.items())
        )
        cache_hashes[split] = sha256_file(output)
        split_statistics[split] = statistics
    expected_counts = {"train": TRAIN_PARENT_COUNT, "val": VALIDATION_PARENT_COUNT}
    if {
        split: int(split_statistics[split]["parent_count"]) for split in SPLITS
    } != expected_counts:
        raise ValueError("V24a known parent support changed")
    contract = {
        "schema_version": KNOWN_GATE_V24A_EXPERIMENT_SCHEMA,
        "experiment_name": "V24a current-event adjacent RD peak-velocity drift mean",
        "internal_variant": V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
        "scope": "development-only KnownGate L1 single-factor RD counter-evidence",
        "fresh_training_not_finetune": True,
        "feature_names": list(COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES),
        "feature_semantics": {
            "feature": "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
            "formula": (
                "mean(abs(diff(argmax_v(sum_r(10**((A_f-r,max(A_f))/10)))))) "
                "/ (v_max-v_min) over adjacent canonical RD frames within the current event"
            ),
            "current_event_only": True,
            "preceding_event_frame_bridge": False,
            "future_frame_or_event": False,
            "minimum_current_event_rd_frames": 2,
            "invalid_policy": "value=0.0 and valid=false; valid physical zero drift remains valid",
            "valid_mask_is_model_input": False,
        },
        "control": {
            "config": _relative(root, paths["control_config"]),
            "config_sha256": hashes["control_config"],
            "summary": _relative(root, paths["control_summary"]),
            "summary_sha256": hashes["control_summary"],
            "checkpoint": _relative(root, paths["control_checkpoint"]),
            "checkpoint_sha256": hashes["control_checkpoint"],
            "readiness": _relative(root, paths["control_readiness"]),
            "readiness_sha256": hashes["control_readiness"],
            "report": _relative(root, paths["control_report"]),
            "report_sha256": hashes["control_report"],
            "initial_model_state_sha256": str(control_summary["initial_model_state_sha256"]),
            "trained_checkpoint_loaded": False,
        },
        "data_contract": {
            "index": _relative(root, paths["index"]),
            "index_sha256": hashes["index"],
            "core_contract": _relative(root, paths["core_contract"]),
            "core_contract_sha256": hashes["core_contract"],
            "normalization": _relative(root, paths["normalization"]),
            "normalization_sha256": hashes["normalization"],
            "source_provenance": _relative(root, paths["provenance"]),
            "source_provenance_sha256": hashes["provenance"],
            "known_train_parent_count": TRAIN_PARENT_COUNT,
            "known_validation_parent_count": VALIDATION_PARENT_COUNT,
            "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "unknown_training_exposure": 0,
        },
        "core_data_contract_sha256": hashes["core_contract"],
        "course_cache_reuse": {
            "source_manifest": _relative(root, paths["course_manifest"]),
            "source_manifest_sha256": hashes["course_manifest"],
            "source_contract": _relative(root, paths["course_contract"]),
            "source_contract_sha256": hashes["course_contract"],
            "train_cache": _relative(root, paths["course_train_cache"]),
            "train_cache_sha256": hashes["course_train_cache"],
            "validation_cache": _relative(root, paths["course_validation_cache"]),
            "validation_cache_sha256": hashes["course_validation_cache"],
            "course_column_reused_byte_for_byte": True,
        },
        "new_rd_cache": {
            "cache_hashes": cache_hashes,
            "split_statistics": split_statistics,
            "canonical_rd_tensor_payload_access_count": int(
                split_statistics["train"]["canonical_rd_tensor_access_count"]
                + split_statistics["val"]["canonical_rd_tensor_access_count"]
            ),
            "raw_echo_payload_access_count": 0,
            "mat_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        "training": {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "batch_size": 2,
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "checkpoint_selection": "parent_min_precision_recall",
            "class_weight_normalization": "batch_weight_sum",
            "all_nonfeature_training_hyperparameters_frozen_to_v24": True,
        },
        "evaluation": {
            "primary": (
                "reduce high-course-incoherence BirdTarget-to-ClutterTarget and total "
                "KnownTarget-to-ClutterTarget errors without Clutter-to-KnownTarget, "
                "KnownTarget recall, or eligible-prefix safety regression"
            ),
            "required_reports": [
                "parent_final",
                "eligible_prefix",
                "parent_balanced_eligible_prefix",
                "window",
                "paired_parent_bootstrap",
                "Clutter_Wilson_intervals_with_support_9",
                "course_incoherence_strata",
                "RD_drift_validity_strata",
                "capture_strata",
                "per_parent_changes",
            ],
            "validation_used_for_readiness_gate": False,
            "primary_hypothesis_labels": [
                "DroneTarget",
                "BirdTarget",
                "BalloonTarget",
                "ClutterTarget",
            ],
            "safety_labels": [],
        },
        "experiment_document": _relative(root, paths["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    manifest = {
        "schema_version": KNOWN_GATE_V24A_MANIFEST_SCHEMA,
        "feature_names": list(COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES),
        "feature_schema": "cq08-current-event-course-plus-rd-peak-drift-mean-v1",
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": hashes["index"],
        "source_course_manifest": CONTROL_MANIFEST.as_posix(),
        "source_course_manifest_sha256": hashes["course_manifest"],
        "cache_files": {
            split: {
                "path": feature_cache_relative(split).as_posix(),
                "sha256": cache_hashes[split],
                "fields": ["sample_ids", "values", "valid"],
                "event_count": int(split_statistics[split]["event_count"]),
                "feature_width": 2,
                "course_column_reused_byte_for_byte": True,
            }
            for split in SPLITS
        },
        "sample_id_is_model_input": False,
        "frame_index_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "unknown_excluded": True,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)
    config_payload = _config_payload(
        control_config_payload,
        manifest_hash=manifest_hash,
        contract_hash=contract_hash,
        cache_hashes=cache_hashes,
    )
    _write_new_or_identical(root / CONFIG, config_payload)
    initialization = _initialization_audit(root, root / CONFIG)
    checks = {
        "v24_control_is_completed_sha_bound_and_never_loaded_as_trained_weights": True,
        "only_current_event_adjacent_rd_frames_contribute_to_drift": True,
        "no_preceding_event_bridge_or_future_event_is_used": True,
        "events_with_fewer_than_two_current_rd_frames_are_invalid_not_valid_zero": True,
        "course_column_reused_byte_for_byte_and_new_drift_column_is_appended": True,
        "current_event_frame_indices_are_ordered_unique_and_non_overlapping_per_parent": True,
        "new_drift_cache_is_aligned_to_v16_train_validation_event_order": True,
        "v24_fresh_reference_and_v24a_shared_initialization_are_exact": True,
        "both_course_and_drift_fusion_columns_start_exactly_zero": True,
        "v16_index_v21_normalization_and_v24_control_are_sha_bound": True,
        "unknown_is_excluded_and_test_payload_access_is_zero": True,
        "validation_values_are_not_a_readiness_gate": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V24A_READINESS_SCHEMA,
        "variant": V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
        "checks": checks,
        "config": {"path": _relative(root, root / CONFIG), "sha256": sha256_file(root / CONFIG)},
        "feature_manifest_path": _relative(root, root / FEATURE_MANIFEST),
        "feature_manifest_sha256": manifest_hash,
        "contract_path": _relative(root, root / EXPERIMENT_CONTRACT),
        "contract_sha256": contract_hash,
        "contract_hashes": {
            "index_sha256": hashes["index"],
            "core_contract_sha256": hashes["core_contract"],
            "normalization_sha256": hashes["normalization"],
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": cache_hashes["train"],
            "validation_feature_cache_sha256": cache_hashes["val"],
            "experiment_contract_sha256": contract_hash,
        },
        "control": contract["control"],
        "initialization": initialization,
        "split_statistics": split_statistics,
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": KNOWN_GATE_V24A_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(root / feature_cache_relative("train")),
        "validation_cache": str(root / feature_cache_relative("val")),
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V24a current-event RD peak-drift KnownGate experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
