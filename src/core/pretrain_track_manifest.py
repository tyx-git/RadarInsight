from __future__ import annotations

"""Canonical train-parent manifests for TR-only track pretraining."""

import hashlib
import json
from pathlib import Path

import pandas as pd


MANIFEST_SCHEMA_VERSION = "tr-only-pretrain-parent-manifest-v1"


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
) -> dict[str, object]:
    parent_sample_ids = canonical_parent_ids_from_index(index_df)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_index": str(source_index),
        "split_name": str(split_name),
        "duplicate_policy": str(duplicate_policy),
        "parent_sample_ids": parent_sample_ids,
        "parent_count": len(parent_sample_ids),
        "parent_ids_sha256": parent_ids_sha256(parent_sample_ids),
    }


def write_pretrain_parent_manifest(
    output_path: str | Path,
    index_df: pd.DataFrame,
    *,
    source_index: str,
    split_name: str = "train",
    duplicate_policy: str = "content_component_canonical_only",
) -> dict[str, object]:
    payload = build_pretrain_parent_manifest(
        index_df,
        source_index=source_index,
        split_name=split_name,
        duplicate_policy=duplicate_policy,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "build_pretrain_parent_manifest",
    "canonical_parent_ids_from_index",
    "parent_ids_sha256",
    "write_pretrain_parent_manifest",
]
