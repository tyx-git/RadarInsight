from __future__ import annotations

"""Canonical train-parent manifests for TR-only track pretraining."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_SCHEMA_VERSION = "tr-only-pretrain-parent-manifest-v1"
PARENT_SPLIT_MANIFEST_SCHEMA_VERSION = "tr-only-pretrain-parent-split-manifest-v1"
PARENT_SPLIT_ASSIGNMENT_ALGORITHM = "source-stratified-session-group-search-v1"


def canonical_parent_ids_from_index(index_df: pd.DataFrame) -> list[str]:
    parent_column = "parent_sample_id" if "parent_sample_id" in index_df.columns else "sample_id"
    if parent_column not in index_df.columns:
        raise ValueError("pretrain parent manifest requires parent_sample_id or sample_id")
    parent_ids = sorted(
        {
            str(value).strip()
            for value in index_df[parent_column].tolist()
            if str(value).strip() and str(value).strip().lower() not in {"nan", "none"}
        }
    )
    if not parent_ids:
        raise ValueError("pretrain parent manifest received no parent IDs")
    return parent_ids


def parent_ids_sha256(parent_sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(parent_sample_ids).encode("utf-8")).hexdigest()


def build_pretrain_parent_manifest(
    index_df: pd.DataFrame,
    *,
    source_index: str,
    split_name: str = "train",
    duplicate_policy: str = "content_component_canonical_only",
    source_index_sha256: str = "",
    source_split_manifest: str = "",
    source_split_manifest_sha256: str = "",
    selected_window_count: int | None = None,
    selection_policy: str = "",
    rd_metadata_contract: str = "",
) -> dict[str, object]:
    parent_sample_ids = canonical_parent_ids_from_index(index_df)
    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_index": str(source_index),
        "split_name": str(split_name),
        "duplicate_policy": str(duplicate_policy),
        "parent_sample_ids": parent_sample_ids,
        "parent_count": len(parent_sample_ids),
        "parent_ids_sha256": parent_ids_sha256(parent_sample_ids),
    }
    optional_fields: dict[str, object] = {
        "source_index_sha256": str(source_index_sha256),
        "source_split_manifest": str(source_split_manifest),
        "source_split_manifest_sha256": str(source_split_manifest_sha256),
        "selection_policy": str(selection_policy),
        "rd_metadata_contract": str(rd_metadata_contract),
    }
    payload.update({key: value for key, value in optional_fields.items() if value})
    if selected_window_count is not None:
        payload["selected_window_count"] = int(selected_window_count)
    return payload


def write_pretrain_parent_manifest(
    output_path: str | Path,
    index_df: pd.DataFrame,
    *,
    source_index: str,
    split_name: str = "train",
    duplicate_policy: str = "content_component_canonical_only",
    source_index_sha256: str = "",
    source_split_manifest: str = "",
    source_split_manifest_sha256: str = "",
    selected_window_count: int | None = None,
    selection_policy: str = "",
    rd_metadata_contract: str = "",
) -> dict[str, object]:
    payload = build_pretrain_parent_manifest(
        index_df,
        source_index=source_index,
        split_name=split_name,
        duplicate_policy=duplicate_policy,
        source_index_sha256=source_index_sha256,
        source_split_manifest=source_split_manifest,
        source_split_manifest_sha256=source_split_manifest_sha256,
        selected_window_count=selected_window_count,
        selection_policy=selection_policy,
        rd_metadata_contract=rd_metadata_contract,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _required_index_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        raise ValueError(f"missing required boolean metadata: {field_name}")
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean metadata for {field_name}: {value!r}")


def _source_validation_groups(
    parent_df: pd.DataFrame,
    *,
    source_name: str,
    validation_ratio: float,
    seed: int,
    search_trials: int,
) -> list[str]:
    labels = sorted(parent_df["label_name"].astype(str).unique().tolist())
    groups: list[tuple[str, np.ndarray, int, int]] = []
    for group_id, group in parent_df.groupby("split_group_id", sort=True):
        label_counts = np.asarray(
            [int((group["label_name"].astype(str) == label).sum()) for label in labels],
            dtype=np.float64,
        )
        groups.append(
            (
                str(group_id),
                label_counts,
                int(len(group)),
                int(group["window_count"].sum()),
            )
        )
    if len(groups) < 2:
        raise ValueError(f"pretrain validation requires at least two groups for source={source_name}")
    total_labels = np.asarray(
        [int((parent_df["label_name"].astype(str) == label).sum()) for label in labels],
        dtype=np.float64,
    )
    label_group_support = np.asarray(
        [
            sum(int(group_counts[index] > 0) for _, group_counts, _, _ in groups)
            for index in range(len(labels))
        ],
        dtype=np.int64,
    )
    if bool((label_group_support < 2).any()):
        unsupported = [
            labels[index]
            for index in range(len(labels))
            if int(label_group_support[index]) < 2
        ]
        raise ValueError(
            "source-label support cannot be isolated into pretrain train/validation: "
            f"source={source_name}, labels={unsupported}"
        )
    target_labels = total_labels * float(validation_ratio)
    target_parents = float(len(parent_df)) * float(validation_ratio)
    target_windows = float(parent_df["window_count"].sum()) * float(validation_ratio)

    def assess(indices: list[int]) -> tuple[float, str, list[int]] | None:
        if not indices or len(indices) == len(groups):
            return None
        validation_labels = sum(
            (groups[index][1] for index in indices),
            np.zeros((len(labels),), dtype=np.float64),
        )
        if bool((validation_labels < 1).any()) or bool(
            ((total_labels - validation_labels) < 1).any()
        ):
            return None
        validation_parents = float(sum(groups[index][2] for index in indices))
        validation_windows = float(sum(groups[index][3] for index in indices))
        label_error = float(
            np.mean(
                np.square(
                    (validation_labels - target_labels)
                    / np.maximum(target_labels, 1.0)
                )
            )
        )
        parent_error = float(
            ((validation_parents - target_parents) / max(target_parents, 1.0)) ** 2
        )
        window_error = float(
            ((validation_windows - target_windows) / max(target_windows, 1.0)) ** 2
        )
        score = label_error + 0.35 * parent_error + 0.15 * window_error
        group_text = "\n".join(groups[index][0] for index in indices)
        return score, hashlib.sha256(group_text.encode("utf-8")).hexdigest(), indices

    best: tuple[float, str, list[int]] | None = None
    if len(groups) <= 16:
        for mask in range(1, 1 << len(groups)):
            candidate = assess(
                [index for index in range(len(groups)) if (mask >> index) & 1]
            )
            if candidate is not None and (best is None or candidate[:2] < best[:2]):
                best = candidate
    else:
        source_seed = int(
            hashlib.sha256(f"{seed}:{source_name}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        rng = np.random.default_rng(source_seed)
        base_group_count = max(1, int(round(len(groups) * float(validation_ratio))))
        for _ in range(max(1, int(search_trials))):
            group_count = int(
                np.clip(base_group_count + int(rng.integers(-5, 6)), 1, len(groups) - 1)
            )
            indices = sorted(
                rng.choice(len(groups), size=group_count, replace=False).tolist()
            )
            candidate = assess(indices)
            if candidate is not None and (best is None or candidate[:2] < best[:2]):
                best = candidate
    if best is None:
        raise ValueError(f"unable to build a valid pretrain validation split for source={source_name}")
    return sorted(groups[index][0] for index in best[2])


def _partition_summary(parent_df: pd.DataFrame) -> dict[str, object]:
    source_label_counts = (
        parent_df.groupby(["source_name", "label_name"], sort=True)
        .size()
        .to_dict()
    )
    return {
        "parent_count": int(len(parent_df)),
        "window_count": int(parent_df["window_count"].sum()),
        "group_count": int(parent_df["split_group_id"].nunique()),
        "source_parent_counts": {
            str(key): int(value)
            for key, value in parent_df["source_name"].astype(str).value_counts().sort_index().items()
        },
        "label_parent_counts": {
            str(key): int(value)
            for key, value in parent_df["label_name"].astype(str).value_counts().sort_index().items()
        },
        "source_label_parent_counts": {
            f"{source_name}::{label_name}": int(value)
            for (source_name, label_name), value in source_label_counts.items()
        },
    }


def build_pretrain_parent_split_manifest(
    index_df: pd.DataFrame,
    *,
    supervised_train_parent_ids: set[str],
    source_parent_manifest: str,
    source_parent_manifest_sha256: str,
    source_index: str,
    source_index_sha256: str,
    source_split_manifest: str,
    source_split_manifest_sha256: str,
    validation_ratio: float = 0.15,
    seed: int = 42,
    search_trials: int = 30000,
) -> dict[str, object]:
    if not 0.0 < float(validation_ratio) < 0.5:
        raise ValueError("pretrain validation_ratio must be within (0, 0.5)")
    required_columns = {
        "sample_id",
        "parent_sample_id",
        "source_name",
        "label_name",
        "timestamp",
        "split_group_id",
        "content_group_id",
        "split_name",
        "is_trainable",
    }
    missing_columns = sorted(required_columns - set(index_df.columns))
    if missing_columns:
        raise ValueError(f"pretrain split index is missing required columns: {missing_columns}")
    trainable = index_df["is_trainable"].map(
        lambda value: _required_index_bool(value, field_name="is_trainable")
    )
    split_names = index_df["split_name"].astype(str).str.strip().str.lower()
    selected = index_df[
        (split_names == "train")
        & trainable
        & index_df["parent_sample_id"].astype(str).isin(supervised_train_parent_ids)
    ].copy()
    if selected.empty:
        raise ValueError("pretrain split received no supervised-train windows")
    actual_parent_ids = set(selected["parent_sample_id"].astype(str))
    if actual_parent_ids != set(supervised_train_parent_ids):
        raise ValueError(
            "pretrain split index/parent manifest mismatch: "
            f"missing={len(set(supervised_train_parent_ids) - actual_parent_ids)}, "
            f"unexpected={len(actual_parent_ids - set(supervised_train_parent_ids))}"
        )
    parent_consistency_columns = [
        "source_name",
        "label_name",
        "timestamp",
        "split_group_id",
        "content_group_id",
    ]
    for column in parent_consistency_columns:
        counts = selected.groupby("parent_sample_id", sort=True)[column].nunique(dropna=False)
        if bool((counts != 1).any()):
            raise ValueError(f"pretrain parent contains inconsistent {column}")
    parent_df = (
        selected.sort_values("sample_id", kind="stable")
        .groupby("parent_sample_id", as_index=False, sort=True)
        .agg(
            source_name=("source_name", "first"),
            label_name=("label_name", "first"),
            timestamp=("timestamp", "first"),
            split_group_id=("split_group_id", "first"),
            content_group_id=("content_group_id", "first"),
            window_count=("sample_id", "size"),
        )
    )
    group_sources = parent_df.groupby("split_group_id", sort=True)["source_name"].nunique()
    if bool((group_sources != 1).any()):
        raise ValueError("pretrain split_group_id spans multiple sources")

    validation_group_ids: set[str] = set()
    for source_name, source_df in parent_df.groupby("source_name", sort=True):
        validation_group_ids.update(
            _source_validation_groups(
                source_df,
                source_name=str(source_name),
                validation_ratio=float(validation_ratio),
                seed=int(seed),
                search_trials=int(search_trials),
            )
        )
    validation_df = parent_df[
        parent_df["split_group_id"].astype(str).isin(validation_group_ids)
    ].copy()
    train_df = parent_df[
        ~parent_df["split_group_id"].astype(str).isin(validation_group_ids)
    ].copy()
    train_parent_ids = sorted(train_df["parent_sample_id"].astype(str).tolist())
    validation_parent_ids = sorted(validation_df["parent_sample_id"].astype(str).tolist())
    if set(train_parent_ids) & set(validation_parent_ids):
        raise ValueError("pretrain train/validation parent sets overlap")
    if set(train_parent_ids) | set(validation_parent_ids) != set(supervised_train_parent_ids):
        raise ValueError("pretrain train/validation parent coverage mismatch")
    assignments = {
        str(group_id): (
            "pretrain_validation" if str(group_id) in validation_group_ids else "pretrain_train"
        )
        for group_id in sorted(parent_df["split_group_id"].astype(str).unique().tolist())
    }
    session_splits = (
        parent_df.assign(
            _partition=parent_df["split_group_id"].astype(str).map(assignments),
            _session=parent_df["source_name"].astype(str) + "::" + parent_df["timestamp"].astype(str),
        )
        .groupby("_session", sort=True)["_partition"]
        .nunique()
    )
    content_splits = (
        parent_df.assign(_partition=parent_df["split_group_id"].astype(str).map(assignments))
        .groupby("content_group_id", sort=True)["_partition"]
        .nunique()
    )
    if bool((session_splits > 1).any()) or bool((content_splits > 1).any()):
        raise ValueError("pretrain split breaks session or content-group isolation")
    assignment_text = "\n".join(
        f"{group_id}\t{partition}" for group_id, partition in assignments.items()
    )
    return {
        "schema_version": PARENT_SPLIT_MANIFEST_SCHEMA_VERSION,
        "assignment_algorithm": PARENT_SPLIT_ASSIGNMENT_ALGORITHM,
        "seed": int(seed),
        "validation_ratio_requested": float(validation_ratio),
        "search_trials": int(search_trials),
        "label_scope": "supervised_train_only",
        "supervised_split_name": "train",
        "split_group_key": "split_group_id",
        "session_group_key": "source_name+timestamp",
        "source_parent_manifest": str(source_parent_manifest),
        "source_parent_manifest_sha256": str(source_parent_manifest_sha256),
        "source_index": str(source_index),
        "source_index_sha256": str(source_index_sha256),
        "source_split_manifest": str(source_split_manifest),
        "source_split_manifest_sha256": str(source_split_manifest_sha256),
        "rd_metadata_contract": "not_applicable_tr_only",
        "scope_parent_count": len(supervised_train_parent_ids),
        "scope_parent_ids_sha256": parent_ids_sha256(sorted(supervised_train_parent_ids)),
        "train_parent_sample_ids": train_parent_ids,
        "train_parent_count": len(train_parent_ids),
        "train_parent_ids_sha256": parent_ids_sha256(train_parent_ids),
        "validation_parent_sample_ids": validation_parent_ids,
        "validation_parent_count": len(validation_parent_ids),
        "validation_parent_ids_sha256": parent_ids_sha256(validation_parent_ids),
        "group_assignments": assignments,
        "group_assignments_sha256": hashlib.sha256(
            assignment_text.encode("utf-8")
        ).hexdigest(),
        "train_summary": _partition_summary(train_df),
        "validation_summary": _partition_summary(validation_df),
    }


def write_pretrain_parent_split_manifest(
    output_path: str | Path,
    index_df: pd.DataFrame,
    **kwargs: object,
) -> dict[str, object]:
    payload = build_pretrain_parent_split_manifest(index_df, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PARENT_SPLIT_ASSIGNMENT_ALGORITHM",
    "PARENT_SPLIT_MANIFEST_SCHEMA_VERSION",
    "build_pretrain_parent_manifest",
    "build_pretrain_parent_split_manifest",
    "canonical_parent_ids_from_index",
    "parent_ids_sha256",
    "write_pretrain_parent_manifest",
    "write_pretrain_parent_split_manifest",
]
