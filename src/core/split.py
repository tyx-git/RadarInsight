from __future__ import annotations

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
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        train_ids.extend(shuffled[:n_train].tolist())
        val_ids.extend(shuffled[n_train:n_train + n_val].tolist())
        test_ids.extend(shuffled[n_train + n_val:n_train + n_val + n_test].tolist())

    return SplitResult(
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        test_ids=sorted(test_ids),
    )
