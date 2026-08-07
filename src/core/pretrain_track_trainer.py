from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .device import resolve_cuda_device
from .features import (
    track_sequence_binary_feature_indices,
    track_sequence_continuous_feature_indices,
    track_sequence_feature_columns,
    track_sequence_feature_dim,
)
from .pretrain_track_config import TrackPretrainConfig
from .pretrain_track_data import build_pretrain_track_loader
from .tr_only_contract import TR_ONLY_ALLOWED_SOURCES
from .pretrain_track_manifest import (
    MANIFEST_SCHEMA_VERSION,
    PARENT_SPLIT_ASSIGNMENT_ALGORITHM,
    PARENT_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .trajectory import TEMPORAL_POSITION_REFERENCE, TransformerTrackEncoder
from .trajectory import (
    TEMPORAL_POOLING_MODES,
    TrajectorySequencePool,
)


PRETRAIN_CHECKPOINT_SCHEMA_VERSION_V2 = "tr-only-track-pretrain-checkpoint-v2"
PRETRAIN_CHECKPOINT_SCHEMA_VERSION = "tr-only-track-pretrain-checkpoint-v3"
SOURCE_STATISTICS_CONTRACT_VERSION_V2 = "tr-only-source-statistics-v2"
SOURCE_STATISTICS_CONTRACT_VERSION = "tr-only-source-statistics-v3"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _mean_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (hidden * valid).sum(dim=1) / denom


def _mtm_reconstruction_mask(
    mtm_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    feature_missing_mask: torch.Tensor,
) -> torch.Tensor:
    """Return element-level targets valid for MTM reconstruction."""
    if mtm_mask.ndim != 2 or padding_mask.shape != mtm_mask.shape:
        raise ValueError("mtm_mask and padding_mask must have shape [batch, time]")
    if feature_missing_mask.ndim != 3 or feature_missing_mask.shape[:2] != mtm_mask.shape:
        raise ValueError("feature_missing_mask must have shape [batch, time, feature]")
    return (
        mtm_mask.unsqueeze(-1)
        & ~padding_mask.unsqueeze(-1)
        & ~feature_missing_mask.bool()
    )


def _mtm_reconstruction_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mtm_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    feature_missing_mask: torch.Tensor,
) -> torch.Tensor:
    valid_mask = _mtm_reconstruction_mask(
        mtm_mask,
        padding_mask,
        feature_missing_mask,
    )
    if not bool(valid_mask.any()):
        return reconstruction.sum() * 0.0
    return torch.nn.functional.mse_loss(
        reconstruction[valid_mask],
        original[valid_mask],
    )


def _mtm_reconstruction_loss_per_sample(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mtm_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    feature_missing_mask: torch.Tensor,
) -> torch.Tensor:
    valid_mask = _mtm_reconstruction_mask(
        mtm_mask,
        padding_mask,
        feature_missing_mask,
    )
    squared_error = torch.square(reconstruction - original)
    reduce_dims = tuple(range(1, squared_error.ndim))
    numerator = (squared_error * valid_mask.to(squared_error.dtype)).sum(dim=reduce_dims)
    denominator = valid_mask.sum(dim=reduce_dims)
    return numerator / denominator.clamp_min(1).to(numerator.dtype)


def _contrastive_loss_from_logits(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("contrastive logits and positive_mask must have matching shape [batch, batch]")
    if not bool(positive_mask.any(dim=-1).all()):
        raise ValueError("contrastive positive mask must contain at least one positive per row")
    denominator = torch.logsumexp(logits, dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=-1)
    return denominator - numerator


def _contrastive_positive_mask(
    batch_size: int,
    *,
    device: torch.device,
    parent_sample_ids: list[str] | None,
    pairing_mode: str,
) -> torch.Tensor:
    normalized_mode = str(pairing_mode).strip().lower()
    if normalized_mode == "diagonal_v1":
        return torch.eye(batch_size, dtype=torch.bool, device=device)
    if normalized_mode != "parent_multi_positive_v1":
        raise ValueError(
            "unsupported contrastive_pairing_mode: "
            f"{pairing_mode!r}; expected 'diagonal_v1' or 'parent_multi_positive_v1'"
        )
    if parent_sample_ids is None or len(parent_sample_ids) != batch_size:
        raise ValueError("parent_multi_positive_v1 requires one parent_sample_id per embedding")
    values = [str(value) for value in parent_sample_ids]
    return torch.tensor(
        [[left == right for right in values] for left in values],
        dtype=torch.bool,
        device=device,
    )


def _info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    *,
    parent_sample_ids: list[str] | None = None,
    pairing_mode: str = "diagonal_v1",
) -> torch.Tensor:
    return _info_nce_per_sample(
        z1,
        z2,
        temperature,
        parent_sample_ids=parent_sample_ids,
        pairing_mode=pairing_mode,
    ).mean()


def _info_nce_per_sample(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    *,
    parent_sample_ids: list[str] | None = None,
    pairing_mode: str = "diagonal_v1",
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    positive_mask = _contrastive_positive_mask(
        int(z1.shape[0]),
        device=z1.device,
        parent_sample_ids=parent_sample_ids,
        pairing_mode=pairing_mode,
    )
    loss_a = _contrastive_loss_from_logits(logits, positive_mask)
    loss_b = _contrastive_loss_from_logits(logits.T, positive_mask.T)
    return 0.5 * (loss_a + loss_b)


class TrackPretrainModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        dropout: float = 0.1,
        temporal_encoding_mode: str = "none",
        temporal_pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = TransformerTrackEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            dropout=dropout,
            temporal_encoding_mode=temporal_encoding_mode,
        )
        self.temporal_encoding_mode = self.encoder.temporal_encoding_mode
        self.temporal_pooling_mode = str(temporal_pooling_mode).strip().lower()
        if self.temporal_pooling_mode not in TEMPORAL_POOLING_MODES:
            raise ValueError(
                "unsupported temporal pooling mode: "
                f"{temporal_pooling_mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
            )
        self.sequence_pool = TrajectorySequencePool(
            model_dim,
            mode=self.temporal_pooling_mode,
            attention_hidden_dim=max(32, model_dim // 2),
        )
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.reconstruct_head = nn.Linear(model_dim, input_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, masked: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor, view_a: torch.Tensor, view_a_padding_mask: torch.Tensor, view_b: torch.Tensor, view_b_padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        effective_mask = mask.bool() & ~padding_mask.bool()
        mask_token = self.mask_token.to(device=masked.device, dtype=masked.dtype).view(1, 1, -1)
        masked_input = torch.where(effective_mask.unsqueeze(-1), mask_token, masked)
        hidden_masked = self.encoder(masked_input, padding_mask=padding_mask)
        recon = self.reconstruct_head(hidden_masked)
        hidden_a = self.encoder(view_a, padding_mask=view_a_padding_mask)
        hidden_b = self.encoder(view_b, padding_mask=view_b_padding_mask)
        pooled_a = self.proj_head(self.sequence_pool(hidden_a, view_a_padding_mask))
        pooled_b = self.proj_head(self.sequence_pool(hidden_b, view_b_padding_mask))
        return {
            "reconstruction": recon,
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
        }


def _parent_ids_sha256(parent_ids: set[str]) -> str:
    payload = "\n".join(sorted(parent_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_reference(path_value: object, manifest_path: Path) -> Path:
    normalized = str(path_value or "").strip()
    if not normalized:
        raise ValueError("pretraining parent manifest contains an empty referenced path")
    path = Path(normalized)
    if path.is_absolute():
        candidates = [path]
    else:
        project_root = Path(__file__).resolve().parents[2]
        candidates = [Path.cwd() / path, project_root / path, manifest_path.parent / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"pretraining parent manifest referenced asset not found: {path}")


def _validate_parent_manifest_provenance(payload: dict[str, object], manifest_path: Path) -> None:
    expected_values = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_name": "train",
        "duplicate_policy": "content_component_canonical_only",
        "selection_policy": "persisted_train_and_strict_is_trainable_and_content_canonical_v1",
        "rd_metadata_contract": "not_applicable_tr_only",
    }
    for key, expected in expected_values.items():
        if str(payload.get(key, "")) != expected:
            raise ValueError(
                f"pretraining parent manifest provenance mismatch for {key}: "
                f"actual={payload.get(key)!r}, expected={expected!r}"
            )
    if int(payload.get("selected_window_count", 0)) <= 0:
        raise ValueError("pretraining parent manifest selected_window_count must be positive")
    resolved_assets: dict[str, Path] = {}
    for path_key, hash_key in (
        ("source_index", "source_index_sha256"),
        ("source_split_manifest", "source_split_manifest_sha256"),
    ):
        referenced_path = _resolve_manifest_reference(payload.get(path_key), manifest_path)
        resolved_assets[path_key] = referenced_path
        declared_hash = str(payload.get(hash_key, "")).strip().lower()
        actual_hash = _sha256_file(referenced_path)
        if not declared_hash or declared_hash != actual_hash:
            raise ValueError(
                f"pretraining parent manifest {hash_key} does not match {path_key}"
            )
    split_payload = json.loads(
        resolved_assets["source_split_manifest"].read_text(encoding="utf-8")
    )
    if split_payload.get("split_evidence_contract") != "tr_only_v1":
        raise ValueError("pretraining source split manifest is not strict TR-only")
    if split_payload.get("rd_content_fingerprint_algorithm") != "not_applicable_tr_only":
        raise ValueError("pretraining source split manifest does not mark RD not applicable")


def _load_allowed_parent_manifest(
    path_value: str,
    *,
    required: bool,
    require_provenance: bool = False,
) -> tuple[set[str] | None, dict[str, object]]:
    normalized = str(path_value or "").strip()
    if not normalized:
        if required:
            raise ValueError("require_train_parent_manifest=true requires allowed_parent_manifest_path")
        return None, {"enabled": False, "parent_count": 0, "parent_ids_sha256": ""}
    path = Path(normalized)
    if not path.exists():
        raise FileNotFoundError(f"pretraining parent manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_ids = payload.get("parent_sample_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("pretraining parent manifest parent_sample_ids must be a list")
    normalized_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
    parent_ids = set(normalized_ids)
    if not parent_ids:
        raise ValueError("pretraining parent manifest contains no parent_sample_ids")
    computed_hash = _parent_ids_sha256(parent_ids)
    declared_hash = str(payload.get("parent_ids_sha256", ""))
    if declared_hash and declared_hash != computed_hash:
        raise ValueError("pretraining parent manifest hash does not match parent_sample_ids")
    if require_provenance:
        if normalized_ids != sorted(parent_ids) or len(normalized_ids) != len(parent_ids):
            raise ValueError("strict pretraining parent manifest IDs must be sorted and unique")
        if int(payload.get("parent_count", -1)) != len(parent_ids):
            raise ValueError("strict pretraining parent manifest parent_count mismatch")
        if not declared_hash:
            raise ValueError("strict pretraining parent manifest requires parent_ids_sha256")
        _validate_parent_manifest_provenance(payload, path)
    metadata_keys = (
        "schema_version",
        "source_index",
        "source_index_sha256",
        "source_split_manifest",
        "source_split_manifest_sha256",
        "split_name",
        "duplicate_policy",
        "selected_window_count",
        "selection_policy",
        "rd_metadata_contract",
    )
    metadata = {
        "enabled": True,
        "path": path.as_posix(),
        "manifest_file_sha256": _sha256_file(path),
        "parent_count": len(parent_ids),
        "parent_ids_sha256": computed_hash,
    }
    metadata.update({key: payload[key] for key in metadata_keys if key in payload})
    return parent_ids, metadata


def _strict_manifest_parent_ids(
    payload: dict[str, object],
    *,
    ids_key: str,
    count_key: str,
    hash_key: str,
) -> tuple[list[str], set[str]]:
    raw_ids = payload.get(ids_key)
    if not isinstance(raw_ids, list):
        raise ValueError(f"pretraining parent split manifest {ids_key} must be a list")
    normalized_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
    parent_ids = set(normalized_ids)
    if not parent_ids:
        raise ValueError(f"pretraining parent split manifest {ids_key} is empty")
    if normalized_ids != sorted(parent_ids) or len(normalized_ids) != len(parent_ids):
        raise ValueError(f"pretraining parent split manifest {ids_key} must be sorted and unique")
    if int(payload.get(count_key, -1)) != len(parent_ids):
        raise ValueError(f"pretraining parent split manifest {count_key} mismatch")
    if str(payload.get(hash_key, "")) != _parent_ids_sha256(parent_ids):
        raise ValueError(f"pretraining parent split manifest {hash_key} mismatch")
    return normalized_ids, parent_ids


def _strict_index_bool_series(series: pd.Series, *, field_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    valid_values = {"true", "false", "1", "0", "yes", "no"}
    invalid = ~normalized.isin(valid_values)
    if bool(invalid.any()):
        values = sorted(normalized.loc[invalid].unique().tolist())
        raise ValueError(f"invalid {field_name} values in pretraining source index: {values}")
    return normalized.isin({"true", "1", "yes"})


def _load_pretrain_parent_split_manifest(
    path_value: str,
    *,
    required: bool,
    allowed_parent_ids: set[str] | None,
    parent_manifest: dict[str, object],
) -> tuple[set[str] | None, set[str] | None, dict[str, object]]:
    normalized = str(path_value or "").strip()
    if not normalized:
        if required:
            raise ValueError(
                "require_parent_validation=true requires parent_split_manifest_path"
            )
        return None, None, {"enabled": False}
    if allowed_parent_ids is None or not bool(parent_manifest.get("enabled", False)):
        raise ValueError("pretraining parent validation requires an enabled full parent manifest")
    path = Path(normalized)
    if not path.exists():
        raise FileNotFoundError(f"pretraining parent split manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pretraining parent split manifest must contain a JSON object")
    expected_values = {
        "schema_version": PARENT_SPLIT_MANIFEST_SCHEMA_VERSION,
        "assignment_algorithm": PARENT_SPLIT_ASSIGNMENT_ALGORITHM,
        "label_scope": "supervised_train_only",
        "supervised_split_name": "train",
        "split_group_key": "split_group_id",
        "session_group_key": "source_name+timestamp",
        "rd_metadata_contract": "not_applicable_tr_only",
    }
    for key, expected in expected_values.items():
        if str(payload.get(key, "")) != expected:
            raise ValueError(
                f"pretraining parent split manifest contract mismatch for {key}: "
                f"actual={payload.get(key)!r}, expected={expected!r}"
            )

    _, train_parent_ids = _strict_manifest_parent_ids(
        payload,
        ids_key="train_parent_sample_ids",
        count_key="train_parent_count",
        hash_key="train_parent_ids_sha256",
    )
    _, validation_parent_ids = _strict_manifest_parent_ids(
        payload,
        ids_key="validation_parent_sample_ids",
        count_key="validation_parent_count",
        hash_key="validation_parent_ids_sha256",
    )
    if train_parent_ids & validation_parent_ids:
        raise ValueError("pretraining parent split train/validation parent sets overlap")
    scope_parent_ids = train_parent_ids | validation_parent_ids
    if scope_parent_ids != allowed_parent_ids:
        raise ValueError("pretraining parent split scope does not match the full parent manifest")
    if int(payload.get("scope_parent_count", -1)) != len(scope_parent_ids):
        raise ValueError("pretraining parent split scope_parent_count mismatch")
    if str(payload.get("scope_parent_ids_sha256", "")) != _parent_ids_sha256(scope_parent_ids):
        raise ValueError("pretraining parent split scope_parent_ids_sha256 mismatch")

    declared_parent_manifest_hash = str(
        payload.get("source_parent_manifest_sha256", "")
    ).strip().lower()
    actual_parent_manifest_hash = str(
        parent_manifest.get("manifest_file_sha256", "")
    ).strip().lower()
    referenced_parent_manifest = _resolve_manifest_reference(
        payload.get("source_parent_manifest"),
        path,
    )
    if (
        not declared_parent_manifest_hash
        or declared_parent_manifest_hash != _sha256_file(referenced_parent_manifest)
        or declared_parent_manifest_hash != actual_parent_manifest_hash
    ):
        raise ValueError("pretraining parent split source_parent_manifest_sha256 mismatch")

    resolved_assets: dict[str, Path] = {}
    for path_key, hash_key in (
        ("source_index", "source_index_sha256"),
        ("source_split_manifest", "source_split_manifest_sha256"),
    ):
        referenced_path = _resolve_manifest_reference(payload.get(path_key), path)
        resolved_assets[path_key] = referenced_path
        declared_hash = str(payload.get(hash_key, "")).strip().lower()
        if not declared_hash or declared_hash != _sha256_file(referenced_path):
            raise ValueError(f"pretraining parent split {hash_key} mismatch")
        parent_declared_hash = str(parent_manifest.get(hash_key, "")).strip().lower()
        if parent_declared_hash and declared_hash != parent_declared_hash:
            raise ValueError(
                f"pretraining parent split {hash_key} does not match full parent manifest"
            )
    split_payload = json.loads(
        resolved_assets["source_split_manifest"].read_text(encoding="utf-8")
    )
    if split_payload.get("split_evidence_contract") != "tr_only_v1":
        raise ValueError("pretraining parent split source manifest is not strict TR-only")
    if split_payload.get("rd_content_fingerprint_algorithm") != "not_applicable_tr_only":
        raise ValueError("pretraining parent split source manifest does not mark RD not applicable")

    assignments = payload.get("group_assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("pretraining parent split group_assignments must be a non-empty mapping")
    normalized_assignments = {
        str(group_id): str(partition)
        for group_id, partition in assignments.items()
    }
    if set(normalized_assignments.values()) - {"pretrain_train", "pretrain_validation"}:
        raise ValueError("pretraining parent split group_assignments contains an unknown partition")
    assignment_text = "\n".join(
        f"{group_id}\t{partition}"
        for group_id, partition in sorted(normalized_assignments.items())
    )
    if str(payload.get("group_assignments_sha256", "")) != hashlib.sha256(
        assignment_text.encode("utf-8")
    ).hexdigest():
        raise ValueError("pretraining parent split group_assignments_sha256 mismatch")

    index_columns = [
        "parent_sample_id",
        "source_name",
        "timestamp",
        "split_group_id",
        "content_group_id",
        "split_name",
        "is_trainable",
    ]
    index_df = pd.read_csv(resolved_assets["source_index"], usecols=index_columns, low_memory=False)
    index_df["parent_sample_id"] = index_df["parent_sample_id"].astype(str)
    selected = index_df[
        index_df["split_name"].astype(str).str.strip().str.lower().eq("train")
        & _strict_index_bool_series(index_df["is_trainable"], field_name="is_trainable")
        & index_df["parent_sample_id"].isin(scope_parent_ids)
    ].copy()
    if set(selected["parent_sample_id"].unique().tolist()) != scope_parent_ids:
        raise ValueError("pretraining parent split source index parent coverage mismatch")
    selected["_partition"] = selected["parent_sample_id"].map(
        lambda parent_id: (
            "pretrain_train" if parent_id in train_parent_ids else "pretrain_validation"
        )
    )
    actual_group_ids = set(selected["split_group_id"].astype(str).unique().tolist())
    if actual_group_ids != set(normalized_assignments):
        raise ValueError("pretraining parent split group assignment coverage mismatch")
    expected_partitions = selected["split_group_id"].astype(str).map(normalized_assignments)
    if not bool(expected_partitions.eq(selected["_partition"]).all()):
        raise ValueError("pretraining parent split group assignment conflicts with parent IDs")
    session_partition_counts = (
        selected.assign(
            _session=selected["source_name"].astype(str) + "::" + selected["timestamp"].astype(str)
        )
        .groupby("_session", sort=True)["_partition"]
        .nunique()
    )
    content_partition_counts = selected.groupby("content_group_id", sort=True)["_partition"].nunique()
    if bool((session_partition_counts > 1).any()) or bool((content_partition_counts > 1).any()):
        raise ValueError("pretraining parent split breaks session/content isolation")

    metadata = dict(payload)
    metadata.update(
        {
            "enabled": True,
            "path": path.as_posix(),
            "manifest_file_sha256": _sha256_file(path),
        }
    )
    return train_parent_ids, validation_parent_ids, metadata


def _serialized_source_stats(dataset) -> dict[str, dict[str, list[float]]]:
    return {
        str(source_name): {
            "center": center.astype(np.float32, copy=False).tolist(),
            "scale": scale.astype(np.float32, copy=False).tolist(),
        }
        for source_name, (center, scale) in dataset.source_stats.items()
    }


def _source_statistics_contract(
    dataset,
    serialized_stats: dict[str, dict[str, list[float]]],
    *,
    checkpoint_schema_version: str = PRETRAIN_CHECKPOINT_SCHEMA_VERSION_V2,
) -> dict[str, object]:
    weighting = str(dataset.source_statistics_weighting)
    estimator = (
        "source_parent_equal_mean_second_moment_population_std_v1"
        if weighting == "parent_equal_v1"
        else "source_point_mean_population_std_v1"
    )
    stats_payload = json.dumps(
        serialized_stats,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    is_v3 = checkpoint_schema_version == PRETRAIN_CHECKPOINT_SCHEMA_VERSION
    contract = {
        "schema_version": (
            SOURCE_STATISTICS_CONTRACT_VERSION
            if is_v3
            else SOURCE_STATISTICS_CONTRACT_VERSION_V2
        ),
        "weighting": weighting,
        "estimator": estimator,
        "fit_split": (
            "pretrain_train"
            if is_v3
            else str(dataset.sampling_metadata.get("split_name", ""))
        ),
        "sample_count": int(len(dataset)),
        "parent_count": int(len(dataset.actual_parent_ids)),
        "feature_schema": str(dataset.trajectory_feature_schema),
        "feature_columns": track_sequence_feature_columns(dataset.trajectory_feature_schema),
        "stats_sha256": hashlib.sha256(stats_payload).hexdigest(),
    }
    if is_v3:
        contract["parent_ids_sha256"] = _parent_ids_sha256(
            set(dataset.actual_parent_ids)
        )
        contract["dataset_role"] = str(dataset.dataset_role)
        contract["source_statistics_origin"] = str(dataset.source_statistics_origin)
    return contract


def _runtime_code_contract() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "config/configmanager.py",
        "src/core/encoding.py",
        "src/core/expert1_trainer.py",
        "src/core/features.py",
        "src/core/pretrain_track_config.py",
        "src/core/pretrain_track_data.py",
        "src/core/pretrain_track_manifest.py",
        "src/core/pretrain_track_trainer.py",
        "src/core/trajectory.py",
    )
    files = {
        relative_path: _sha256_file(project_root / relative_path)
        for relative_path in relative_paths
    }
    aggregate = hashlib.sha256(
        "\n".join(f"{path}:{digest}" for path, digest in sorted(files.items())).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "tr-only-pretrain-code-contract-v1",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _encoder_state_dict(model: TrackPretrainModel) -> dict[str, torch.Tensor]:
    return {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }


def _parent_equal_loss_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "loss": 0.0,
            "mtm_loss": 0.0,
            "contrastive_loss": 0.0,
            "window_count": 0,
            "parent_count": 0,
        }
    parent_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        parent_id = str(row["parent_sample_id"])
        values = parent_values.setdefault(
            parent_id,
            {"loss": [], "mtm_loss": [], "contrastive_loss": []},
        )
        for key in values:
            values[key].append(float(row[key]))
    return {
        key: float(
            np.mean(
                [np.mean(parent_row[key]) for parent_row in parent_values.values()]
            )
        )
        for key in ("loss", "mtm_loss", "contrastive_loss")
    } | {
        "window_count": len(rows),
        "parent_count": len(parent_values),
    }


def _pretrain_length_bucket(length: int) -> str:
    if length <= 8:
        return "01_08"
    if length <= 16:
        return "09_16"
    return "17_plus"


def _evaluate_pretrain_parent_validation(
    model: TrackPretrainModel,
    loader,
    *,
    device: torch.device,
    contrastive_temperature: float,
    mtm_loss_weight: float,
    contrastive_loss_weight: float,
    contrastive_pairing_mode: str = "diagonal_v1",
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, object]] = []
    try:
        with torch.no_grad():
            for batch in loader:
                masked = batch.masked.to(device)
                original = batch.original.to(device)
                mask = batch.mask.to(device)
                padding_mask = batch.padding_mask.to(device)
                view_a = batch.view_a.to(device)
                view_b = batch.view_b.to(device)
                view_a_padding_mask = batch.view_a_padding_mask.to(device)
                view_b_padding_mask = batch.view_b_padding_mask.to(device)
                feature_missing_mask = batch.feature_missing_mask.to(device)
                output = model(
                    masked,
                    mask,
                    padding_mask,
                    view_a,
                    view_a_padding_mask,
                    view_b,
                    view_b_padding_mask,
                )
                mtm_losses = _mtm_reconstruction_loss_per_sample(
                    output["reconstruction"],
                    original,
                    mask,
                    padding_mask,
                    feature_missing_mask,
                )
                nce_losses = _info_nce_per_sample(
                    output["pooled_a"],
                    output["pooled_b"],
                    contrastive_temperature,
                    parent_sample_ids=[str(value) for value in batch.parent_sample_ids],
                    pairing_mode=contrastive_pairing_mode,
                )
                total_losses = (
                    float(mtm_loss_weight) * mtm_losses
                    + float(contrastive_loss_weight) * nce_losses
                )
                if not (
                    len(batch.sample_ids)
                    == len(batch.parent_sample_ids)
                    == len(batch.source_names)
                    == int(total_losses.shape[0])
                ):
                    raise ValueError("pretrain validation batch metadata/tensor size mismatch")
                for index, sample_id in enumerate(batch.sample_ids):
                    rows.append(
                        {
                            "sample_id": str(sample_id),
                            "parent_sample_id": str(batch.parent_sample_ids[index]),
                            "source_name": str(batch.source_names[index]),
                            "length_bucket": _pretrain_length_bucket(
                                int(batch.lengths[index].item())
                            ),
                            "loss": float(total_losses[index].detach().cpu().item()),
                            "mtm_loss": float(mtm_losses[index].detach().cpu().item()),
                            "contrastive_loss": float(
                                nce_losses[index].detach().cpu().item()
                            ),
                        }
                    )
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("pretrain parent validation loader produced no samples")
    parent_sources: dict[str, set[str]] = {}
    for row in rows:
        parent_sources.setdefault(str(row["parent_sample_id"]), set()).add(
            str(row["source_name"])
        )
    mixed_source_parents = sorted(
        parent_id for parent_id, sources in parent_sources.items() if len(sources) != 1
    )
    if mixed_source_parents:
        raise ValueError(
            "pretrain parent validation contains parents spanning multiple sources: "
            f"{mixed_source_parents[:5]}"
        )
    overall = _parent_equal_loss_summary(rows)
    source_names = sorted({str(row["source_name"]) for row in rows})
    length_buckets = ("01_08", "09_16", "17_plus")
    return {
        **overall,
        "aggregation": "window_mean_within_parent_then_parent_equal_v1",
        "info_nce_reduction": (
            "per_sample_symmetric_parent_multi_positive_v1"
            if str(contrastive_pairing_mode).strip().lower() == "parent_multi_positive_v1"
            else "per_sample_symmetric_v1"
        ),
        "contrastive_pairing_mode": str(contrastive_pairing_mode).strip().lower(),
        "mtm_reduction": "per_sample_reconstructable_feature_mean_v1",
        "by_source": {
            source_name: _parent_equal_loss_summary(
                [row for row in rows if str(row["source_name"]) == source_name]
            )
            for source_name in source_names
        },
        "by_length": {
            bucket: _parent_equal_loss_summary(
                [row for row in rows if str(row["length_bucket"]) == bucket]
            )
            for bucket in length_buckets
        },
    }


def _checkpoint_selection_metric(
    record: dict[str, object],
    *,
    parent_validation_enabled: bool,
) -> tuple[str, float]:
    metric_name = (
        "pretrain_parent_validation_loss"
        if parent_validation_enabled
        else "training_loss"
    )
    record_key = "pretrain_parent_validation_loss" if parent_validation_enabled else "loss"
    if record_key not in record:
        raise ValueError(f"checkpoint selection record is missing {record_key}")
    value = float(record[record_key])
    if not np.isfinite(value):
        raise ValueError(f"checkpoint selection metric {metric_name} is non-finite")
    return metric_name, value


def pretrain_track_encoder(config: TrackPretrainConfig) -> dict[str, object]:
    config.validate_active_training_contract()
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    effective_config_path = run_paths.output_dir / "effective_config.json"
    effective_config_path.write_text(
        json.dumps(
            {
                "schema_version": "pretrain-track-effective-config-v1",
                "pretrain_track": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    allowed_parent_ids, parent_manifest = _load_allowed_parent_manifest(
        config.allowed_parent_manifest_path,
        required=bool(config.require_train_parent_manifest),
        require_provenance=bool(config.require_manifest_provenance),
    )
    if bool(config.require_manifest_provenance):
        manifest_index_name = Path(str(parent_manifest.get("source_index", ""))).name
        if manifest_index_name != str(config.index_name):
            raise ValueError(
                "pretraining config index_name does not match parent manifest source_index: "
                f"config={config.index_name}, manifest={manifest_index_name}"
            )
    train_parent_ids, validation_parent_ids, parent_split_manifest = (
        _load_pretrain_parent_split_manifest(
            config.parent_split_manifest_path,
            required=bool(config.require_parent_validation),
            allowed_parent_ids=allowed_parent_ids,
            parent_manifest=parent_manifest,
        )
    )
    parent_validation_enabled = train_parent_ids is not None
    if parent_validation_enabled != (validation_parent_ids is not None):
        raise ValueError("pretraining parent split must provide both train and validation parents")
    optimization_parent_ids = train_parent_ids if parent_validation_enabled else allowed_parent_ids
    train_loader = build_pretrain_track_loader(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mask_ratio=config.mask_ratio,
        trajectory_feature_schema=config.trajectory_feature_schema,
        index_name=config.index_name,
        allowed_parent_ids=optimization_parent_ids,
        sampling_mode=config.sampling_mode,
        source_statistics_weighting=config.source_statistics_weighting,
        randomization_mode="stochastic",
        random_seed=config.seed,
        dataset_role="pretrain_train",
        shuffle=True,
        parent_sampling_mode=config.parent_sampling_mode,
        allowed_source_names=TR_ONLY_ALLOWED_SOURCES,
    )
    actual_train_parent_ids = set(
        getattr(train_loader.dataset, "actual_parent_ids", set())
    )
    if optimization_parent_ids is not None and actual_train_parent_ids != optimization_parent_ids:
        raise ValueError(
            "pretraining optimization dataset parent IDs do not exactly match the requested scope"
        )
    validation_loader = None
    actual_validation_parent_ids: set[str] = set()
    if parent_validation_enabled:
        validation_loader = build_pretrain_track_loader(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            mask_ratio=config.mask_ratio,
            trajectory_feature_schema=config.trajectory_feature_schema,
            index_name=config.index_name,
            allowed_parent_ids=validation_parent_ids,
            sampling_mode=config.sampling_mode,
            source_statistics_weighting=config.source_statistics_weighting,
            source_stats_override=train_loader.dataset.source_stats,
            randomization_mode="deterministic_v1",
            random_seed=config.seed,
            dataset_role="pretrain_validation",
            shuffle=False,
            parent_sampling_mode="window_shuffle_v1",
            allowed_source_names=TR_ONLY_ALLOWED_SOURCES,
        )
        actual_validation_parent_ids = set(
            getattr(validation_loader.dataset, "actual_parent_ids", set())
        )
        if actual_validation_parent_ids != validation_parent_ids:
            raise ValueError(
                "pretraining validation dataset parent IDs do not exactly match the split manifest"
            )
        if actual_train_parent_ids & actual_validation_parent_ids:
            raise ValueError("pretraining optimization/validation dataset parents overlap")
    actual_scope_parent_ids = actual_train_parent_ids | actual_validation_parent_ids
    if allowed_parent_ids is not None and actual_scope_parent_ids != allowed_parent_ids:
        raise ValueError("pretraining dataset parent scope does not match the full parent manifest")
    parent_manifest = {
        **parent_manifest,
        "actual_parent_count": len(actual_scope_parent_ids),
        "actual_parent_ids_sha256": _parent_ids_sha256(actual_scope_parent_ids),
    }
    checkpoint_schema_version = (
        PRETRAIN_CHECKPOINT_SCHEMA_VERSION
        if parent_validation_enabled
        else PRETRAIN_CHECKPOINT_SCHEMA_VERSION_V2
    )
    serialized_source_stats = _serialized_source_stats(train_loader.dataset)
    source_statistics_contract = _source_statistics_contract(
        train_loader.dataset,
        serialized_source_stats,
        checkpoint_schema_version=checkpoint_schema_version,
    )
    if parent_validation_enabled:
        if int(source_statistics_contract["parent_count"]) != int(
            parent_split_manifest["train_parent_count"]
        ) or str(source_statistics_contract.get("parent_ids_sha256", "")) != str(
            parent_split_manifest["train_parent_ids_sha256"]
        ):
            raise ValueError(
                "pretraining source statistics optimization scope does not match parent split train scope"
            )
    code_contract = _runtime_code_contract()
    model = TrackPretrainModel(
        input_dim=track_sequence_feature_dim(config.trajectory_feature_schema),
        model_dim=config.model_dim,
        dropout=config.dropout,
        temporal_encoding_mode=config.temporal_encoding_mode,
        temporal_pooling_mode=config.temporal_pooling_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    best_selection_value = float("inf")
    best_path = run_paths.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_values: list[float] = []
        mtm_values: list[float] = []
        nce_values: list[float] = []
        masked_token_counts: list[float] = []
        reconstructable_feature_counts: list[float] = []
        for batch in train_loader:
            masked = batch.masked.to(device)
            original = batch.original.to(device)
            mask = batch.mask.to(device)
            padding_mask = batch.padding_mask.to(device)
            view_a = batch.view_a.to(device)
            view_b = batch.view_b.to(device)
            view_a_padding_mask = batch.view_a_padding_mask.to(device)
            view_b_padding_mask = batch.view_b_padding_mask.to(device)
            feature_missing_mask = batch.feature_missing_mask.to(device)

            output = model(masked, mask, padding_mask, view_a, view_a_padding_mask, view_b, view_b_padding_mask)
            recon = output["reconstruction"]
            mtm_loss = _mtm_reconstruction_loss(
                recon,
                original,
                mask,
                padding_mask,
                feature_missing_mask,
            )
            nce_loss = _info_nce(
                output["pooled_a"],
                output["pooled_b"],
                config.contrastive_temperature,
                parent_sample_ids=[str(value) for value in batch.parent_sample_ids],
                pairing_mode=config.contrastive_pairing_mode,
            )
            loss = config.mtm_loss_weight * mtm_loss + config.contrastive_loss_weight * nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_values.append(float(loss.detach().cpu().item()))
            mtm_values.append(float(mtm_loss.detach().cpu().item()))
            nce_values.append(float(nce_loss.detach().cpu().item()))

            valid_mtm_mask = _mtm_reconstruction_mask(
                mask,
                padding_mask,
                feature_missing_mask,
            )
            masked_token_count = float((mask & ~padding_mask).sum().detach().cpu().item())
            reconstructable_feature_count = float(valid_mtm_mask.sum().detach().cpu().item())
            masked_token_counts.append(masked_token_count)
            reconstructable_feature_counts.append(reconstructable_feature_count)

        record = {
            "epoch": epoch,
            "loss": float(np.mean(loss_values)),
            "mtm_loss": float(np.mean(mtm_values)),
            "contrastive_loss": float(np.mean(nce_values)),
            "masked_token_count": float(np.mean(masked_token_counts)) if masked_token_counts else 0.0,
            "reconstructable_feature_count": float(np.mean(reconstructable_feature_counts)) if reconstructable_feature_counts else 0.0,
        }
        validation_metrics: dict[str, object] | None = None
        if validation_loader is not None:
            validation_metrics = _evaluate_pretrain_parent_validation(
                model,
                validation_loader,
                device=device,
                contrastive_temperature=config.contrastive_temperature,
                mtm_loss_weight=config.mtm_loss_weight,
                contrastive_loss_weight=config.contrastive_loss_weight,
                contrastive_pairing_mode=config.contrastive_pairing_mode,
            )
            record["pretrain_parent_validation_loss"] = float(
                validation_metrics["loss"]
            )
            record["pretrain_parent_validation_mtm_loss"] = float(
                validation_metrics["mtm_loss"]
            )
            record["pretrain_parent_validation_contrastive_loss"] = float(
                validation_metrics["contrastive_loss"]
            )
            record["pretrain_parent_validation"] = validation_metrics
        selection_metric, selection_value = _checkpoint_selection_metric(
            record,
            parent_validation_enabled=parent_validation_enabled,
        )
        record["checkpoint_selection_metric"] = selection_metric
        record["checkpoint_selection_value"] = selection_value
        history.append(record)
        if selection_value < best_selection_value:
            best_selection_value = selection_value
            torch.save(
                {
                    "checkpoint_schema_version": checkpoint_schema_version,
                    "code_contract": code_contract,
                    "model_state_dict": _encoder_state_dict(model),
                    "pretrain_model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "trajectory_feature_schema": config.trajectory_feature_schema,
                    "feature_contract": {
                        "feature_columns": track_sequence_feature_columns(config.trajectory_feature_schema),
                        "continuous_indices": track_sequence_continuous_feature_indices(config.trajectory_feature_schema),
                        "binary_missing_indices": track_sequence_binary_feature_indices(config.trajectory_feature_schema),
                        "source_amplitude_transform": "cq08_signed_log1p_v1"
                        if str(config.trajectory_feature_schema).strip().lower() == "v3"
                        else "legacy_none",
                    },
                    "sampling_contract": dict(train_loader.dataset.sampling_metadata),
                    "validation_sampling_contract": (
                        dict(validation_loader.dataset.sampling_metadata)
                        if validation_loader is not None
                        else {}
                    ),
                    "temporal_contract": {
                        "encoding_mode": config.temporal_encoding_mode,
                        "pooling_mode": config.temporal_pooling_mode,
                        "position_reference": TEMPORAL_POSITION_REFERENCE,
                    },
                    "parent_manifest": parent_manifest,
                    "pretrain_parent_split": parent_split_manifest,
                    "source_feature_stats": serialized_source_stats,
                    "source_statistics_contract": source_statistics_contract,
                    "selection_metric": selection_metric,
                    "selection_scope": (
                        "pretrain_validation_parent_equal"
                        if parent_validation_enabled
                        else "pretrain_train_batch_mean"
                    ),
                    "selection_value": float(selection_value),
                    "epoch": int(epoch),
                    "training_loss_at_selection": float(record["loss"]),
                    "validation_metrics_at_selection": validation_metrics or {},
                    **(
                        {"best_pretrain_parent_validation_loss": float(best_selection_value)}
                        if parent_validation_enabled
                        else {"best_training_loss": float(best_selection_value)}
                    ),
                },
                best_path,
            )

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint_schema_version": checkpoint_schema_version,
                "code_contract": code_contract,
                "mode": "pretrain_track",
                "num_epochs": config.epochs,
                "trajectory_feature_schema": config.trajectory_feature_schema,
                "feature_columns": track_sequence_feature_columns(config.trajectory_feature_schema),
                "index_name": config.index_name,
                "sampling_contract": dict(train_loader.dataset.sampling_metadata),
                "validation_sampling_contract": (
                    dict(validation_loader.dataset.sampling_metadata)
                    if validation_loader is not None
                    else {}
                ),
                "temporal_contract": {
                    "encoding_mode": config.temporal_encoding_mode,
                    "pooling_mode": config.temporal_pooling_mode,
                    "position_reference": TEMPORAL_POSITION_REFERENCE,
                },
                "parent_manifest": parent_manifest,
                "pretrain_parent_split": parent_split_manifest,
                "source_statistics_contract": source_statistics_contract,
                "sample_count": len(train_loader.dataset),
                "validation_sample_count": (
                    len(validation_loader.dataset) if validation_loader is not None else 0
                ),
                "selection_metric": (
                    "pretrain_parent_validation_loss"
                    if parent_validation_enabled
                    else "training_loss"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    evaluation_report = {
        "summary": {
            "loss": float(
                history[-1].get(
                    "pretrain_parent_validation_loss",
                    history[-1]["loss"],
                )
            ),
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "route_accuracy": None,
        },
        "evaluation_scope": (
            "pretrain_parent_validation"
            if parent_validation_enabled
            else "pretrain_training_only"
        ),
        "classification_metrics_status": "not_applicable_pretraining",
        "label_names": [],
        "classification_confusion_matrix": [],
        "per_class": [],
        "route": {"label_names": [], "confusion_matrix": [], "per_route": []},
        "by_source": {},
        "pretrain_validation_loss_by_source": (
            dict(history[-1]["pretrain_parent_validation"]["by_source"])
            if parent_validation_enabled
            else {}
        ),
        "device": device_info,
        "predictions": [],
    }
    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path.as_posix()),
        "history_file": str(history_path.as_posix()),
        "manifest_file": str(manifest_path.as_posix()),
        "evaluation_report": str(evaluation_report_path.as_posix()),
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "effective_config": str(effective_config_path.as_posix()),
        "device": device_info,
        "trainable_parameters": _count_trainable_parameters(model),
        "checkpoint_schema_version": checkpoint_schema_version,
        "code_contract": code_contract,
        "parent_manifest": parent_manifest,
        "pretrain_parent_split": parent_split_manifest,
        "source_statistics_contract": source_statistics_contract,
        "selection_metric": (
            "pretrain_parent_validation_loss"
            if parent_validation_enabled
            else "training_loss"
        ),
        "best_selection_value": float(best_selection_value),
        "final_loss": float(history[-1]["loss"]),
        "final_mtm_loss": float(history[-1]["mtm_loss"]),
        "final_contrastive_loss": float(history[-1]["contrastive_loss"]),
        "final_pretrain_parent_validation": (
            dict(history[-1]["pretrain_parent_validation"])
            if parent_validation_enabled
            else {}
        ),
    }
    summary_path = run_paths.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_paths = write_real_experiment_report(
        output_dir=run_paths.output_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=run_paths.timestamp_label,
    )
    summary["real_report_markdown"] = str(report_paths.markdown_path.as_posix())
    summary["real_report_json"] = str(report_paths.json_path.as_posix())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
