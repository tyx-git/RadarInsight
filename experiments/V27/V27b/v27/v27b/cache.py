"""Materialized frozen V27a features used by V27b.

The cache is deliberately a V27b-owned artifact.  It contains only causal
outputs of the frozen V27a model and the small set of quality facts needed for
evaluation; it never contains Unknown or test rows.  Keeping this boundary
explicit makes it possible to train the new correction head without invoking
the historical V27a loader's old manifest paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.runtime import resolve_project_path, write_torch_if_absent_or_identical

from .protocol import V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION, V27BContext, V27B_L2_CLASS_NAMES, V27B_STABLE8_INDICES, V27B_STABLE8_NAMES, expected_parent_label_counts, v27b_artifact_path


def _source_cache_hashes(context: V27BContext) -> dict[str, str]:
    values = context.input_binding.get("cache_payload_sha256s")
    if not isinstance(values, Mapping) or set(values) != {"train", "val"}:
        raise ValueError("V27b input binding lacks train/val cache payload hashes")
    result = {str(key): str(value).lower() for key, value in values.items()}
    if any(len(value) != 64 for value in result.values()):
        raise ValueError("V27b source cache payload hash is malformed")
    return result


def _array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"V27b feature cache field is an object array: {name}")
    return result


def cache_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a V27b payload using the stable V27a semantic cache hash scheme."""

    from v27.v27a.cache import cache_payload_sha256 as _semantic_hash

    return _semantic_hash(payload)


def _validate_parent_layout(payload: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    parent_ids = [str(value) for value in payload.get("parent_sample_ids", ())]
    parent_labels = [str(value) for value in payload.get("parent_labels", ())]
    labels = _array(payload.get("labels"), name="labels")
    lengths = _array(payload.get("lengths"), name="lengths")
    padding = _array(payload.get("padding_mask"), name="padding_mask")
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27b feature cache parent IDs must be unique and non-empty")
    if len(parent_labels) != len(parent_ids):
        raise ValueError("V27b feature cache parent labels are misaligned")
    if any(value not in V27B_L2_CLASS_NAMES for value in parent_labels):
        raise ValueError("V27b feature cache contains an unsupported parent label")
    if labels.shape != (len(parent_ids),) or labels.dtype.kind not in "iu":
        raise ValueError("V27b feature cache labels have an invalid shape/dtype")
    expected_labels = np.asarray(
        [V27B_L2_CLASS_NAMES.index(value) for value in parent_labels], dtype=np.int64
    )
    if not np.array_equal(labels.astype(np.int64), expected_labels):
        raise ValueError("V27b feature cache numeric labels differ from parent labels")
    if lengths.shape != (len(parent_ids),) or lengths.dtype.kind not in "iu":
        raise ValueError("V27b feature cache lengths have an invalid shape/dtype")
    if np.any(lengths <= 0):
        raise ValueError("V27b feature cache contains an empty parent")
    if padding.ndim != 2 or padding.shape[0] != len(parent_ids) or padding.dtype != np.bool_:
        raise ValueError("V27b feature cache padding has an invalid shape/dtype")
    width = padding.shape[1]
    if np.any(lengths > width):
        raise ValueError("V27b feature cache length exceeds configured width")
    expected_padding = np.ones_like(padding, dtype=np.bool_)
    for row, length in enumerate(lengths.tolist()):
        expected_padding[row, : int(length)] = False
    if not np.array_equal(padding, expected_padding):
        raise ValueError("V27b feature cache padding must be a suffix")
    return parent_ids, labels.astype(np.int64), lengths.astype(np.int64), padding


def _required_source_fields(context: V27BContext, split: str) -> dict[str, str]:
    files = context.input_binding.get("source_files")
    if not isinstance(files, Mapping):
        raise ValueError("V27b input binding lacks source files")
    checkpoint = files.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V27b input binding lacks checkpoint source")
    source_hashes = _source_cache_hashes(context)
    return {
        "source_checkpoint_file_sha256": str(checkpoint.get("sha256", "")).lower(),
        "source_checkpoint_state_sha256": str(
            context.config.base.expected_v27a_checkpoint_state_sha256
        ).lower(),
        "source_cache_payload_sha256": source_hashes[split],
        "source_archived_manifest_sha256": str(
            context.config.base.expected_v27a_checkpoint_training_manifest_sha256
        ).lower(),
        "source_archived_config_identity_sha256": str(
            context.config.base.expected_v27a_checkpoint_training_config_identity_sha256
        ).lower(),
        "v27b_config_identity_sha256": context.config.config_identity_sha256,
        "v27b_manifest_sha256": context.manifest_sha256,
    }


def _validate_source_fields(
    payload: Mapping[str, Any],
    context: V27BContext,
    split: str,
) -> None:
    expected = _required_source_fields(context, split)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"V27b feature cache source provenance differs at {key}: "
                f"expected={value!r}, actual={payload.get(key)!r}"
            )


def _validate_feature_payload(
    payload: Mapping[str, Any],
    context: V27BContext,
    split: str,
) -> dict[str, Any]:
    expected_schema = context.config.data.feature_cache_schema
    if payload.get("schema_version") != expected_schema:
        raise ValueError("V27b feature cache schema differs")
    if payload.get("experiment_id") != "V27b":
        raise ValueError("V27b feature cache experiment id differs")
    if payload.get("split_name") != split:
        raise ValueError("V27b feature cache split differs")
    declared = payload.get("cache_payload_sha256")
    if not isinstance(declared, str) or declared != cache_payload_sha256(payload):
        raise ValueError("V27b feature cache payload hash differs")
    _validate_source_fields(payload, context, split)
    parent_ids, labels, lengths, padding = _validate_parent_layout(payload)
    expected_counts = expected_parent_label_counts(split)
    observed_counts = {
        name: int(sum(value == name for value in payload["parent_labels"]))
        for name in V27B_L2_CLASS_NAMES
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"V27b feature cache {split} parent support differs: "
            f"expected={expected_counts}, actual={observed_counts}"
        )
    width = int(padding.shape[1])
    n = len(parent_ids)
    fields = {
        "frozen_l2_logits": ((n, width, 3), np.float32),
        "v27a_base_logits": ((n, width, 3), np.float32),
        "v27a_residual_logits": ((n, width, 3), np.float32),
        "parent_state": ((n, width, context.config.model.parent_state_dim), np.float32),
        "event_representation": (
            (n, width, context.config.model.event_representation_dim),
            np.float32,
        ),
        "event_summary_valid": ((n, width), np.bool_),
        "event_reliable_frame_count": ((n, width), None),
        "event_rd_present": ((n, width), np.bool_),
        "event_reliable_frame_fraction": ((n, width), np.float32),
    }
    direct_frames = context.config.model.evidence_carrier == "direct_frame_gru"
    if direct_frames:
        fields.update(
            {
                "event_frame_values": (
                    (n, width, context.config.data.max_frames_per_event, context.config.data.stable8_dim),
                    np.float32,
                ),
                "event_frame_mask": (
                    (n, width, context.config.data.max_frames_per_event),
                    np.bool_,
                ),
                "event_frame_count": ((n, width), None),
            }
        )
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in fields.items():
        value = _array(payload.get(name), name=name)
        if value.shape != shape:
            raise ValueError(f"V27b feature cache {name} shape differs")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"V27b feature cache {name} dtype differs")
        if dtype != np.bool_ and not np.isfinite(value).all():
            raise ValueError(f"V27b feature cache {name} contains non-finite values")
        arrays[name] = value
    if arrays["event_reliable_frame_count"].dtype.kind not in "iu":
        raise ValueError("V27b feature cache frame counts must be integer")
    if np.any(arrays["event_reliable_frame_count"] < 0):
        raise ValueError("V27b feature cache frame counts cannot be negative")
    if not np.allclose(
        arrays["frozen_l2_logits"],
        arrays["v27a_base_logits"] + arrays["v27a_residual_logits"],
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("V27b frozen logits are not V27a base plus residual")
    for name, value in arrays.items():
        if value.ndim == 2:
            if np.any(value[padding] != 0):
                raise ValueError(f"V27b padded {name} values must be zero")
        else:
            if np.any(value[padding] != 0.0):
                raise ValueError(f"V27b padded {name} values must be zero")
    if np.any(arrays["event_summary_valid"][padding]):
        raise ValueError("V27b padded summary validity must be false")
    if np.any(arrays["event_reliable_frame_fraction"] < 0.0) or np.any(
        arrays["event_reliable_frame_fraction"] > 1.0
    ):
        raise ValueError("V27b reliable frame fractions are out of bounds")
    if direct_frames:
        frame_values = arrays["event_frame_values"]
        frame_mask = arrays["event_frame_mask"]
        frame_count = arrays["event_frame_count"]
        if frame_count.dtype.kind not in "iu":
            raise ValueError("V27b direct frame counts must be integer")
        if np.any(frame_count < 0) or np.any(
            frame_count > context.config.data.max_frames_per_event
        ):
            raise ValueError("V27b direct frame counts are out of bounds")
        if not np.array_equal(frame_count, frame_mask.sum(axis=-1)):
            raise ValueError("V27b direct frame counts are inconsistent")
        if not np.array_equal(
            frame_count.astype(np.int64),
            arrays["event_reliable_frame_count"].astype(np.int64),
        ):
            raise ValueError("V27b direct frame counts disagree with quality facts")
        if np.any(frame_values[~frame_mask] != 0.0):
            raise ValueError("V27b inactive direct frame values must be zero")
        for row in range(n):
            for event in range(width):
                mask = frame_mask[row, event]
                inactive = np.flatnonzero(~mask)
                if inactive.size and bool(mask[int(inactive[0]) :].any()):
                    raise ValueError("V27b direct frame masks must be suffix-padded")
        if not np.array_equal(
            arrays["event_summary_valid"],
            frame_mask.any(axis=-1) & ~padding,
        ):
            raise ValueError("V27b direct frame validity is inconsistent")
        if np.any(
            arrays["event_reliable_frame_fraction"][padding] != 0.0
        ):
            raise ValueError("V27b padded direct frame fractions must be zero")
        if payload.get("stable8_indices") != list(V27B_STABLE8_INDICES):
            raise ValueError("V27b direct frame stable8 indices differ")
        if payload.get("stable8_names") != list(V27B_STABLE8_NAMES):
            raise ValueError("V27b direct frame stable8 names differ")
        if payload.get("normalized_frame_values") is not True:
            raise ValueError("V27b direct frame values must declare normalization")
    return {
        **dict(payload),
        "parent_sample_ids": parent_ids,
        "parent_labels": [str(value) for value in payload["parent_labels"]],
        "labels": labels,
        "lengths": lengths,
        "padding_mask": padding,
        **arrays,
    }


def load_feature_cache(
    path: str | Path,
    *,
    context: V27BContext,
    split: str,
) -> dict[str, Any]:
    canonical_split = str(split).strip().lower()
    if canonical_split not in {"train", "val"}:
        raise ValueError(f"unsupported V27b feature cache split: {split}")
    target = resolve_project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27b feature cache must be a mapping")
    return _validate_feature_payload(payload, context, canonical_split)


class V27BFeatureCacheDataset(Dataset[dict[str, Any]]):
    """Parent-indexed tensor view over one immutable V27b feature cache."""

    def __init__(self, path: str | Path, *, context: V27BContext, split: str) -> None:
        self.path = resolve_project_path(context.project_root, path, must_exist=True)
        self.payload = load_feature_cache(self.path, context=context, split=split)
        self.parent_ids = list(self.payload["parent_sample_ids"])

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("V27b feature cache index must be an integer")
        row = int(index)
        if not 0 <= row < len(self):
            raise IndexError("V27b feature cache index is out of range")
        payload = self.payload
        result: dict[str, Any] = {
            "parent_sample_id": self.parent_ids[row],
            "parent_label": payload["parent_labels"][row],
        }
        for name in (
            "labels",
            "lengths",
            "padding_mask",
            "frozen_l2_logits",
            "v27a_base_logits",
            "v27a_residual_logits",
            "parent_state",
            "event_representation",
            "event_summary_valid",
            "event_reliable_frame_count",
            "event_rd_present",
            "event_reliable_frame_fraction",
        ):
            value = payload[name][row]
            if isinstance(value, np.ndarray):
                result[name] = torch.from_numpy(value.copy())
            else:
                result[name] = value
        if self.payload.get("schema_version") == V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION:
            for name in ("event_frame_values", "event_frame_mask", "event_frame_count"):
                value = self.payload[name][row]
                result[name] = torch.from_numpy(value.copy())
        result["label"] = result.pop("labels")
        result["length"] = result.pop("lengths")
        return result


def build_feature_dataloader(
    dataset: V27BFeatureCacheDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("V27b feature dataloader batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _extract_split_features(
    context: V27BContext,
    *,
    split: str,
    model: torch.nn.Module,
    source_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    from v27.v27a.cache import V27AEvidenceCacheDataset, build_cache_dataloader

    source_dataset = V27AEvidenceCacheDataset(source_path)
    source_loader = build_cache_dataloader(
        source_dataset,
        batch_size=context.config.training.batch_size,
        shuffle=False,
    )
    source_payload = source_dataset.payload
    n = len(source_dataset)
    width = int(source_payload["padding_mask"].shape[1])
    shapes = {
        "frozen_l2_logits": (n, width, 3),
        "v27a_base_logits": (n, width, 3),
        "v27a_residual_logits": (n, width, 3),
        "parent_state": (n, width, context.config.model.parent_state_dim),
        "event_representation": (
            n,
            width,
            context.config.model.event_representation_dim,
        ),
    }
    arrays = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()
    }
    direct_frames = context.config.model.evidence_carrier == "direct_frame_gru"
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in source_loader:
            inputs = {
                name: value.to(device=device)
                for name, value in (
                    ("baseline_logits", batch["baseline_logits"]),
                    ("summary_tokens", batch["summary_tokens"]),
                    ("summary_valid", batch["summary_valid"]),
                    ("frame_values", batch["frame_values"]),
                    ("frame_mask", batch["frame_mask"]),
                    ("padding_mask", batch["padding_mask"]),
                )
            }
            output = model(**inputs)
            size = int(inputs["baseline_logits"].shape[0])
            if cursor + size > n:
                raise ValueError("V27b feature extraction exceeded source cache")
            rows = slice(cursor, cursor + size)
            arrays["v27a_base_logits"][rows] = inputs["baseline_logits"].cpu().numpy()
            arrays["v27a_residual_logits"][rows] = output["residual_logits"].cpu().numpy()
            arrays["frozen_l2_logits"][rows] = output["sequence_logits"].cpu().numpy()
            arrays["parent_state"][rows] = output["parent_sequence"].cpu().numpy()
            arrays["event_representation"][rows] = output[
                "event_representation"
            ].cpu().numpy()
            expected_ids = [str(value) for value in source_payload["parent_sample_ids"][cursor : cursor + size]]
            actual_ids = [str(value) for value in batch["parent_sample_id"]]
            if actual_ids != expected_ids:
                raise ValueError("V27b source cache parent order changed during extraction")
            cursor += size
    if cursor != n:
        raise ValueError("V27b feature extraction did not traverse the source cache")
    payload: dict[str, Any] = {
        "schema_version": context.config.data.feature_cache_schema,
        "experiment_id": "V27b",
        "split_name": split,
        "parent_sample_ids": [str(value) for value in source_payload["parent_sample_ids"]],
        "parent_labels": [str(value) for value in source_payload["parent_labels"]],
        "labels": np.asarray(source_payload["labels"], dtype=np.int64),
        "lengths": np.asarray(source_payload["lengths"], dtype=np.int64),
        "padding_mask": np.asarray(source_payload["padding_mask"], dtype=np.bool_),
        **arrays,
        "event_summary_valid": np.asarray(
            source_payload["event_summary_valid"], dtype=np.bool_
        ),
        "event_reliable_frame_count": np.asarray(
            source_payload["event_reliable_frame_count"], dtype=np.int64
        ),
        "event_rd_present": np.asarray(
            source_payload["event_rd_present"], dtype=np.bool_
        ),
        "event_reliable_frame_fraction": np.asarray(
            source_payload["event_reliable_frame_fraction"], dtype=np.float32
        ),
        **_required_source_fields(context, split),
    }
    if direct_frames:
        source_frame_values = np.asarray(
            source_payload["event_frame_values"], dtype=np.float32
        )
        source_frame_mask = np.asarray(
            source_payload["event_frame_mask"], dtype=np.bool_
        )
        source_frame_count = np.asarray(
            source_payload["event_frame_count"], dtype=np.int64
        )
        expected_shape = (
            n,
            width,
            context.config.data.max_frames_per_event,
            context.config.data.stable8_dim,
        )
        if source_frame_values.shape != expected_shape:
            raise ValueError("V27b source direct frame values shape differs")
        if source_frame_mask.shape != expected_shape[:3]:
            raise ValueError("V27b source direct frame mask shape differs")
        if source_frame_count.shape != expected_shape[:2]:
            raise ValueError("V27b source direct frame count shape differs")
        payload.update(
            {
                "event_frame_values": source_frame_values,
                "event_frame_mask": source_frame_mask,
                "event_frame_count": source_frame_count,
                "stable8_indices": list(V27B_STABLE8_INDICES),
                "stable8_names": list(V27B_STABLE8_NAMES),
                "normalized_frame_values": True,
            }
        )
    payload["cache_payload_sha256"] = cache_payload_sha256(payload)
    return payload


def materialize_v27b_feature_caches(
    context: V27BContext,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the frozen V27a main encoder and persist train/validation features."""

    from v27.v27a.model import build_v27a_model
    from v27.v27a.protocol import load_v27a_config

    root = context.project_root
    v27a_config = load_v27a_config(
        root,
        resolve_project_path(root, context.config.base.v27a_runtime_config_path),
    )
    model = build_v27a_model(
        v27a_config,
        arm=context.config.base.expected_v27a_arm,
        seed=context.config.base.expected_v27a_seed,
    ).to(device)
    checkpoint_path = resolve_project_path(
        root, context.config.base.v27a_checkpoint_path, must_exist=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    results: dict[str, Any] = {}
    source_paths = {
        "train": resolve_project_path(
            root, context.config.base.v27a_train_cache_path, must_exist=True
        ),
        "val": resolve_project_path(
            root, context.config.base.v27a_validation_cache_path, must_exist=True
        ),
    }
    targets = {
        "train": v27b_artifact_path(
            root, context.config.artifacts.train_feature_cache_path
        ),
        "val": v27b_artifact_path(
            root, context.config.artifacts.validation_feature_cache_path
        ),
    }
    for split in ("train", "val"):
        payload = _extract_split_features(
            context,
            split=split,
            model=model,
            source_path=source_paths[split],
            device=device,
        )
        write_torch_if_absent_or_identical(targets[split], payload)
        stored = load_feature_cache(targets[split], context=context, split=split)
        if stored["cache_payload_sha256"] != payload["cache_payload_sha256"]:
            raise ValueError("V27b serialized feature cache hash differs")
        results[split] = {
            "path": str(targets[split]),
            "parent_count": len(stored["parent_sample_ids"]),
            "cache_payload_sha256": stored["cache_payload_sha256"],
        }
    return {
        "status": "materialized_v27b_frozen_v27a_features",
        "checkpoint_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "splits": results,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


__all__ = ["V27BFeatureCacheDataset", "build_feature_dataloader", "cache_payload_sha256", "load_feature_cache", "materialize_v27b_feature_caches"]