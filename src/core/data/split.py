from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def stratified_split(
    index_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _, group in index_df.groupby("label_name", sort=True):
        sample_ids = group["sample_id"].astype(str).tolist()
        shuffled = np.array(sample_ids, dtype=object)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            if n_train < 1:
                n_train = 1
            if n_val < 1:
                n_val = 1
            if n_train + n_val >= n:
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
        train_ids.extend(shuffled[:n_train].tolist())
        val_ids.extend(shuffled[n_train:n_train + n_val].tolist())
        test_ids.extend(shuffled[n_train + n_val:n_train + n_val + n_test].tolist())

    return SplitResult(
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        test_ids=sorted(test_ids),
    )


def _split_text(value: object) -> str:
    if value is None:
        return ''
    if not isinstance(value, (str, bytes, Mapping)) and not isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ''
    text = str(value).strip()
    return '' if text.lower() in {'', 'nan', 'none', '<na>'} else text


def canonical_group_id(row: Mapping[str, object]) -> str:
    parent_id = _split_text(row.get('parent_sample_id'))
    if parent_id:
        identity = parent_id
    else:
        sample_id = _split_text(row.get('sample_id'))
        if not sample_id:
            raise ValueError('sample_id must be non-empty')
        prefix, marker, _ = sample_id.partition('|win')
        identity = prefix if marker and prefix else sample_id
    source_name = _split_text(row.get('source_name'))
    return f'{source_name}::{identity}' if source_name else identity


def _partitions(split: SplitResult) -> dict[str, list[str]]:
    return {
        'train': [str(value) for value in split.train_ids],
        'val': [str(value) for value in split.val_ids],
        'test': [str(value) for value in split.test_ids],
    }


def validate_grouped_split(index_df: pd.DataFrame, split: SplitResult) -> None:
    if not {'sample_id', 'label_name'}.issubset(index_df.columns):
        raise ValueError('index must contain sample_id and label_name')
    sample_ids = index_df['sample_id'].astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('duplicate sample_id in index')
    partitions = _partitions(split)
    assigned_ids = [sample_id for ids in partitions.values() for sample_id in ids]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError('split partitions overlap')
    if set(assigned_ids) != set(sample_ids):
        raise ValueError('split coverage is incomplete or mismatched')
    assignment = {sample_id: name for name, ids in partitions.items() for sample_id in ids}
    group_splits: dict[str, set[str]] = {}
    group_labels: dict[str, set[str]] = {}
    for row in index_df.to_dict(orient='records'):
        group_id = canonical_group_id(row)
        group_splits.setdefault(group_id, set()).add(assignment[str(row['sample_id'])])
        group_labels.setdefault(group_id, set()).add(str(row['label_name']))
    if any(len(names) > 1 for names in group_splits.values()):
        raise ValueError('groups cross splits')
    if any(len(labels) > 1 for labels in group_labels.values()):
        raise ValueError('groups have conflicting labels')


def stratified_group_split(
    index_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    if not {'sample_id', 'label_name'}.issubset(index_df.columns):
        raise ValueError('index must contain sample_id and label_name')
    group_labels: dict[str, set[str]] = {}
    group_samples: dict[str, list[str]] = {}
    for row in index_df.to_dict(orient='records'):
        group_id = canonical_group_id(row)
        group_labels.setdefault(group_id, set()).add(str(row['label_name']))
        group_samples.setdefault(group_id, []).append(str(row['sample_id']))
    if any(len(labels) > 1 for labels in group_labels.values()):
        raise ValueError('groups have conflicting labels')
    group_df = pd.DataFrame(
        {
            'sample_id': list(group_samples),
            'label_name': [next(iter(group_labels[group_id])) for group_id in group_samples],
        }
    )
    group_split = stratified_split(
        group_df,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    split = SplitResult(
        train_ids=sorted(
            sample_id
            for group_id in group_split.train_ids
            for sample_id in group_samples[group_id]
        ),
        val_ids=sorted(
            sample_id
            for group_id in group_split.val_ids
            for sample_id in group_samples[group_id]
        ),
        test_ids=sorted(
            sample_id
            for group_id in group_split.test_ids
            for sample_id in group_samples[group_id]
        ),
    )
    validate_grouped_split(index_df, split)
    return split


__all__ = [
    "SplitResult",
    "canonical_group_id",
    "validate_grouped_split",
    "stratified_split",
    "stratified_group_split",
]
