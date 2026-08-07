from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from config.configmanager import load_expert1_config
from .data.track_dataset import parse_required_bool
from .expert1_trainer import _build_expert1_loaders
from .pretrain_track_manifest import write_pretrain_parent_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TR_ONLY_SPLIT_CONTRACT = "tr_only_v1"
_EMPTY_TEXT_VALUES = {"", "nan", "none", "null"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_input_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _nonempty_text_mask(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.strip().str.lower().isin(_EMPTY_TEXT_VALUES)


def _load_trainable_split_index(
    window_index_path: Path,
    split_manifest_path: Path,
    *,
    split_name: str = "train",
) -> tuple[pd.DataFrame, dict[str, object]]:
    index_df = pd.read_csv(window_index_path, low_memory=False)
    required_columns = {
        "sample_id",
        "parent_sample_id",
        "split_name",
        "is_trainable",
        "is_content_canonical",
        "content_group_id",
        "split_group_id",
    }
    missing_columns = sorted(required_columns - set(index_df.columns))
    if missing_columns:
        raise ValueError(f"TR-only window index is missing required columns: {missing_columns}")
    persisted_splits = index_df["split_name"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(persisted_splits) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"TR-only window index has invalid split_name values: {invalid_splits}")
    trainable = index_df["is_trainable"].map(
        lambda value: parse_required_bool(value, field_name="is_trainable")
    )
    canonical = index_df["is_content_canonical"].map(
        lambda value: parse_required_bool(value, field_name="is_content_canonical")
    )
    normalized_split = str(split_name).strip().lower()
    selected = index_df[(persisted_splits == normalized_split) & trainable].copy()
    if selected.empty:
        raise ValueError(f"TR-only window index has no trainable rows for split={normalized_split}")
    if not bool(canonical.loc[selected.index].all()):
        raise ValueError("trainable TR-only manifest rows must all be content-canonical")
    for column in ("sample_id", "parent_sample_id", "content_group_id", "split_group_id"):
        if not bool(_nonempty_text_mask(selected[column]).all()):
            raise ValueError(f"TR-only manifest selection contains missing {column}")
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("TR-only manifest selection contains duplicate sample_id values")

    selected_parent_ids = set(selected["parent_sample_id"].astype(str))
    selected_parent_rows = index_df[
        index_df["parent_sample_id"].astype(str).isin(selected_parent_ids)
    ].copy()
    parent_split_counts = (
        selected_parent_rows.assign(_split=persisted_splits.loc[selected_parent_rows.index])
        .groupby("parent_sample_id", sort=True)["_split"]
        .nunique()
    )
    if bool((parent_split_counts > 1).any()):
        raise ValueError("selected TR-only parent appears in multiple persisted splits")
    for group_column in ("content_group_id", "split_group_id"):
        group_split_counts = (
            index_df.assign(_split=persisted_splits)
            .groupby(group_column, sort=True)["_split"]
            .nunique()
        )
        if bool((group_split_counts > 1).any()):
            raise ValueError(f"TR-only {group_column} appears in multiple persisted splits")

    for column in ("rd_path", "rd_summary_csv_path", "rd_tensor_path", "rd_content_fingerprint"):
        if column in index_df.columns and bool(_nonempty_text_mask(index_df[column]).any()):
            raise ValueError(f"strict TR-only window index contains usable {column} values")

    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_payload.get("split_evidence_contract") != _TR_ONLY_SPLIT_CONTRACT:
        raise ValueError("split manifest is not declared as strict TR-only")
    if split_payload.get("rd_content_fingerprint_algorithm") != "not_applicable_tr_only":
        raise ValueError("split manifest does not declare RD fingerprint not applicable")
    expected_ids = {str(value) for value in split_payload.get(f"{normalized_split}_ids", [])}
    actual_ids = set(selected["sample_id"].astype(str))
    if expected_ids != actual_ids:
        raise ValueError(
            "window index and split manifest selected IDs differ: "
            f"missing={len(expected_ids - actual_ids)}, unexpected={len(actual_ids - expected_ids)}"
        )
    all_manifest_ids: list[str] = []
    for key in ("train_ids", "val_ids", "test_ids", "diagnostic_ids"):
        values = split_payload.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"split manifest field {key} must be a list")
        all_manifest_ids.extend(str(value) for value in values)
    if len(all_manifest_ids) != len(set(all_manifest_ids)):
        raise ValueError("split manifest ID lists overlap")
    if set(index_df["sample_id"].astype(str)) != set(all_manifest_ids):
        raise ValueError("window index and split manifest do not contain the same complete ID set")
    return selected.reset_index(drop=True), split_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the exact duplicate-safe TR-only train-parent manifest for track pretraining."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--expert1-config")
    source_group.add_argument("--window-index")
    parser.add_argument(
        "--window-split-manifest",
        help="Required with --window-index; must declare split_evidence_contract=tr_only_v1.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_index_sha256 = ""
    source_split_manifest = ""
    source_split_manifest_sha256 = ""
    selection_policy = ""
    rd_metadata_contract = ""
    if args.window_index:
        if not args.window_split_manifest:
            parser.error("--window-split-manifest is required with --window-index")
        source_index_path = _resolve_input_path(args.window_index)
        split_manifest_path = _resolve_input_path(args.window_split_manifest)
        index_df, _ = _load_trainable_split_index(
            source_index_path,
            split_manifest_path,
            split_name="train",
        )
        source_index = _display_path(source_index_path)
        source_index_sha256 = _sha256_file(source_index_path)
        source_split_manifest = _display_path(split_manifest_path)
        source_split_manifest_sha256 = _sha256_file(split_manifest_path)
        selection_policy = "persisted_train_and_strict_is_trainable_and_content_canonical_v1"
        rd_metadata_contract = "not_applicable_tr_only"
    else:
        config = load_expert1_config(args.expert1_config)
        loaders = _build_expert1_loaders(config, validate_csv_files=False)
        index_df = loaders.train_loader.dataset.index_df
        source_index_path = _resolve_input_path(Path(config.data_root) / "processed" / config.index_name)
        source_index = _display_path(source_index_path)
        source_index_sha256 = _sha256_file(source_index_path)

    payload = write_pretrain_parent_manifest(
        args.output,
        index_df,
        source_index=source_index,
        source_index_sha256=source_index_sha256,
        source_split_manifest=source_split_manifest,
        source_split_manifest_sha256=source_split_manifest_sha256,
        selected_window_count=len(index_df),
        selection_policy=selection_policy,
        rd_metadata_contract=rd_metadata_contract,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).as_posix()),
                "parent_count": payload["parent_count"],
                "parent_ids_sha256": payload["parent_ids_sha256"],
                "source_index": payload["source_index"],
                "source_index_sha256": payload.get("source_index_sha256", ""),
                "selected_window_count": payload.get("selected_window_count", 0),
                "source_split_manifest": payload.get("source_split_manifest", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
