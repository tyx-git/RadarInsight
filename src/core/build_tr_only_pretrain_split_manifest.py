from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .pretrain_track_manifest import (
    MANIFEST_SCHEMA_VERSION,
    parent_ids_sha256,
    write_pretrain_parent_split_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scope_parent_manifest(
    manifest_path: Path,
    *,
    window_index_path: Path,
) -> tuple[set[str], dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("scope parent manifest schema mismatch")
    if payload.get("split_name") != "train":
        raise ValueError("scope parent manifest must cover the supervised train split")
    if payload.get("rd_metadata_contract") != "not_applicable_tr_only":
        raise ValueError("scope parent manifest is not strict TR-only")
    raw_ids = payload.get("parent_sample_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("scope parent manifest parent_sample_ids must be a list")
    normalized_ids = [str(value).strip() for value in raw_ids if str(value).strip()]
    if normalized_ids != sorted(set(normalized_ids)):
        raise ValueError("scope parent manifest parent IDs must be sorted and unique")
    if int(payload.get("parent_count", -1)) != len(normalized_ids):
        raise ValueError("scope parent manifest parent_count mismatch")
    if payload.get("parent_ids_sha256") != parent_ids_sha256(normalized_ids):
        raise ValueError("scope parent manifest parent hash mismatch")
    if Path(str(payload.get("source_index", ""))).name != window_index_path.name:
        raise ValueError("scope parent manifest source_index does not match window index")
    if str(payload.get("source_index_sha256", "")).lower() != _sha256_file(
        window_index_path
    ):
        raise ValueError("scope parent manifest source index hash mismatch")
    source_split_path = _resolve_path(str(payload.get("source_split_manifest", "")))
    if str(payload.get("source_split_manifest_sha256", "")).lower() != _sha256_file(
        source_split_path
    ):
        raise ValueError("scope parent manifest source split hash mismatch")
    split_payload = json.loads(source_split_path.read_text(encoding="utf-8"))
    if split_payload.get("split_evidence_contract") != "tr_only_v1":
        raise ValueError("scope source split manifest is not strict TR-only")
    return set(normalized_ids), payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split strict TR-only supervised-train parents into session/content-isolated "
            "pretraining train and validation subsets."
        )
    )
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--scope-parent-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=30000)
    args = parser.parse_args()

    window_index_path = _resolve_path(args.window_index)
    parent_manifest_path = _resolve_path(args.scope_parent_manifest)
    parent_ids, parent_payload = _load_scope_parent_manifest(
        parent_manifest_path,
        window_index_path=window_index_path,
    )
    index_df = pd.read_csv(window_index_path, low_memory=False)
    payload = write_pretrain_parent_split_manifest(
        args.output,
        index_df,
        supervised_train_parent_ids=parent_ids,
        source_parent_manifest=_display_path(parent_manifest_path),
        source_parent_manifest_sha256=_sha256_file(parent_manifest_path),
        source_index=_display_path(window_index_path),
        source_index_sha256=_sha256_file(window_index_path),
        source_split_manifest=str(parent_payload["source_split_manifest"]),
        source_split_manifest_sha256=str(parent_payload["source_split_manifest_sha256"]),
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
        search_trials=int(args.search_trials),
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    print(
        json.dumps(
            {
                "output": _display_path(output_path),
                "output_sha256": _sha256_file(output_path),
                "scope_parent_count": payload["scope_parent_count"],
                "train_parent_count": payload["train_parent_count"],
                "validation_parent_count": payload["validation_parent_count"],
                "train_window_count": payload["train_summary"]["window_count"],
                "validation_window_count": payload["validation_summary"]["window_count"],
                "group_assignments_sha256": payload["group_assignments_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
