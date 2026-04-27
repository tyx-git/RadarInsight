from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_grouped_split(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    usable_df = manifest_df[manifest_df["is_usable"] == True].copy()
    if usable_df.empty:
        raise ValueError("no usable CQ-08 groups found")

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _, group in usable_df.groupby("label_name", sort=True):
        split_groups = group["split_group_id"].astype(str).drop_duplicates().tolist()
        shuffled = np.array(split_groups, dtype=object)
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

    return {
        "train_group_ids": sorted(train_ids),
        "val_group_ids": sorted(val_ids),
        "test_group_ids": sorted(test_ids),
    }


def write_cq08_grouped_split(
    project_root: Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    manifest_path = project_root / "data" / "manifests" / "cq08_unified_manifest.json"
    output_path = project_root / "data" / "manifests" / "cq08_grouped_split.json"
    manifest_df = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8")))
    split = build_grouped_split(
        manifest_df,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    usable_df = manifest_df[manifest_df["is_usable"] == True].copy()
    train_df = usable_df[usable_df["split_group_id"].astype(str).isin(split["train_group_ids"])].copy()
    val_df = usable_df[usable_df["split_group_id"].astype(str).isin(split["val_group_ids"])].copy()
    test_df = usable_df[usable_df["split_group_id"].astype(str).isin(split["test_group_ids"])].copy()

    payload = {
        "seed": seed,
        "group_unit": "track_id",
        "frame_level_split_forbidden": True,
        **split,
        "counts": {
            "train_groups": int(len(split["train_group_ids"])),
            "val_groups": int(len(split["val_group_ids"])),
            "test_groups": int(len(split["test_group_ids"])),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
        },
        "label_counts": {
            "train": train_df["label_name"].value_counts().to_dict(),
            "val": val_df["label_name"].value_counts().to_dict(),
            "test": test_df["label_name"].value_counts().to_dict(),
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    payload = write_cq08_grouped_split(project_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
