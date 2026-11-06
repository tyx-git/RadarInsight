"""独立 L3 knownness head 的表示物化、训练和 checkpoint API。"""

from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from radar_ln3.checkpoint import Checkpoint, canonical_hash, sha256_file
from radar_ln3.data import DatasetManifest, load_manifest
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.data.paths import resolve_project_artifact_path, resolve_project_path
from radar_ln3.features import (
    FeatureCache,
    build_model_batch,
    load_feature_cache_bundle,
)
from radar_ln3.model import (
    KNOWNNESS_HEAD_SCHEMA,
    KNOWNNESS_HEAD_SCHEMA_V3,
    KNOWNNESS_HEAD_SCHEMA_V4,
    KNOWNNESS_HEAD_SCHEMA_V5,
    KNOWNNESS_HEAD_SCHEMA_V6,
    BirdProfileResidualKnownnessHead,
    BirdProfileResidualFusionKnownnessHead,
    BirdProfileResidualEvidenceFusionKnownnessHead,
    BirdTemporalStabilityResidualKnownnessHead,
    ProfileResidualKnownnessHead,
    TemporalKnownnessHead,
    build_profile_knownness_features,
    load_model_config,
    knownness_representation_contract,
)

from .l3 import (
    _l1_inputs,
    _l2_inputs,
    _l2_reference_inputs,
    _load_endpoint_models,
)
from radar_ln3.cache import CachedParentSequenceDataset
from radar_ln3.logger import log_classification_metrics
from .common import _metrics


HEAD_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
L1_ROUTES = ("target", "clutter")
L2_ROUTES = ("DroneTarget", "BirdTarget", "BalloonTarget")
EXTRA_FEATURE_DIM = 32
MICRO_DOPPLER_EVIDENCE_SUMMARY_DIM = 47
REPRESENTATION_CACHE_SCHEMA = "RADAR-LN3-ENDPOINT-REPRESENTATION-V5"


@dataclass(frozen=True)
class FrozenEndpointRepresentation:
    task: str
    parent_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    representation: torch.Tensor
    padding_mask: torch.Tensor
    lengths: torch.Tensor
    source_manifest_sha256: str
    endpoint_sha256: Mapping[str, str]
    feature_cache_manifest_sha256: str
    view_hash: str
    endpoint_logits: torch.Tensor
    profile_representation: torch.Tensor | None = None
    bird_evidence_representation: torch.Tensor | None = None
    test_payload_access_count: int = 0

    def __post_init__(self) -> None:
        if self.task not in {"l1", "l2"}:
            raise ValueError("frozen representation task 必须为 l1 或 l2")
        if len(self.parent_ids) == 0 or len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError("frozen representation parent ids 必须非空且唯一")
        if len(self.label_names) != len(self.parent_ids):
            raise ValueError("frozen representation labels 未对齐")
        if self.representation.ndim != 3 or self.padding_mask.shape != self.representation.shape[:2]:
            raise ValueError("frozen representation tensor shape 无效")
        if self.padding_mask.dtype != torch.bool or self.lengths.shape != (len(self.parent_ids),):
            raise ValueError("frozen representation mask/length shape 无效")
        if (
            self.endpoint_logits.ndim != 2
            or self.endpoint_logits.shape[0] != len(self.parent_ids)
            or self.endpoint_logits.shape[1] != (2 if self.task == "l1" else 3)
            or not self.endpoint_logits.is_floating_point()
            or not torch.isfinite(self.endpoint_logits).all()
        ):
            raise ValueError("frozen endpoint logits shape/value 无效")
        if bool((self.lengths <= 0).any()):
            raise ValueError("frozen representation 不接受空 parent")
        if not self.representation.is_floating_point() or not torch.isfinite(self.representation).all():
            raise ValueError("frozen representation 必须是有限浮点值")
        actual_lengths = (~self.padding_mask).long().sum(dim=1).cpu()
        if not torch.equal(actual_lengths, self.lengths.cpu()):
            raise ValueError("frozen representation lengths 与 padding 不一致")
        if bool(self.representation.masked_select(self.padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("frozen representation padding 必须严格为零")
        if self.profile_representation is not None:
            if self.profile_representation.shape[:2] != self.representation.shape[:2] or not self.profile_representation.is_floating_point() or not torch.isfinite(self.profile_representation).all():
                raise ValueError("frozen profile representation shape/value 无效")
            if bool(self.profile_representation.masked_select(self.padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError("frozen profile representation padding 必须严格为零")
        if self.bird_evidence_representation is not None:
            evidence = self.bird_evidence_representation
            if (
                evidence.shape[:2] != self.representation.shape[:2]
                or evidence.shape[-1] != MICRO_DOPPLER_EVIDENCE_SUMMARY_DIM
                or not evidence.is_floating_point()
                or not torch.isfinite(evidence).all()
            ):
                raise ValueError("frozen Bird evidence representation shape/value 无效")
            if bool(evidence.masked_select(self.padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError("frozen Bird evidence representation padding 必须严格为零")


@dataclass(frozen=True)
class FrozenEndpointEventEvidence:
    """冻结 endpoint 的逐事件分类 logits。

    该对象只用于探索性 L3 回放；knownness head 仍复用同一份 frozen
    representation，避免为每个事件重复构造主干状态。
    """

    representation: FrozenEndpointRepresentation
    sequence_logits: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.sequence_logits.ndim != 3
            or self.sequence_logits.shape[0] != len(self.representation.parent_ids)
            or self.sequence_logits.shape[1] != self.representation.representation.shape[1]
            or self.sequence_logits.shape[2] != (2 if self.representation.task == "l1" else 3)
            or not self.sequence_logits.is_floating_point()
            or not torch.isfinite(self.sequence_logits).all()
        ):
            raise ValueError("frozen event evidence logits shape/value 无效")


@dataclass(frozen=True)
class _EndpointRepresentationBatch:
    representation: torch.Tensor
    profile_representation: torch.Tensor | None
    padding_mask: torch.Tensor
    endpoint_logits: torch.Tensor
    parent_ids: list[str]
    label_names: list[str]
    bird_evidence_representation: torch.Tensor | None
    sequence_logits: torch.Tensor | None


def _device(value: str | torch.device) -> torch.device:
    target = torch.device(str(value))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("knownness head 配置要求 CUDA，但当前环境不可用")
    return target


def _epoch_permutation(
    length: int,
    *,
    seed: int,
    epoch: int,
    phase_offset: int = 0,
) -> torch.Tensor:
    """Return the deterministic shuffle for one training epoch.

    Base and profile training are separate phases.  Keeping the phase offset
    explicit makes profile epoch ``n`` use a new seed instead of accidentally
    reusing the base-history length after the base phase has finished.
    """

    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(phase_offset) + int(epoch)
    )
    return torch.randperm(int(length), generator=generator)


def _feature_caches(
    root: Path,
    manifest: DatasetManifest,
    dataset: CachedParentSequenceDataset,
    path: str | Path,
) -> tuple[dict[str, FeatureCache], str]:
    target = resolve_project_path(root, path, must_exist=True)
    caches = load_feature_cache_bundle(
        root,
        target,
        expected_view=dataset.view,
        expected_source_manifest_hash=data_manifest_hash(manifest),
        verify_integrity=True,
    )
    return caches, sha256_file(target)


def _micro_doppler_evidence_summary(
    sequences: Sequence[Any],
    caches: Mapping[str, FeatureCache],
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """按事件构造因果的完整 23 维微多普勒摘要。

    每个事件只消费当前事件的帧：23 维均值、23 维标准差和有效帧比例。
    不读取后续事件，也不把无效帧的零值混入统计量。
    """

    cache = caches.get("micro_doppler_evidence")
    if cache is None:
        raise ValueError("L2 Bird evidence residual 需要 micro_doppler_evidence cache")
    event_ids_by_parent = [
        [str(event.sample_id) for event in sequence.events[: int(width)]]
        for sequence in sequences
    ]
    return build_micro_doppler_evidence_summary(
        event_ids_by_parent,
        cache,
        width=width,
        device=device,
    )


def build_micro_doppler_evidence_summary(
    event_ids_by_parent: Sequence[Sequence[str]],
    cache: FeatureCache,
    *,
    width: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """从已对齐的事件 ID 构造完整微多普勒证据摘要。"""

    values = np.zeros(
        (len(event_ids_by_parent), int(width), MICRO_DOPPLER_EVIDENCE_SUMMARY_DIM),
        dtype=np.float32,
    )
    positions = cache.sample_positions
    for row, event_ids in enumerate(event_ids_by_parent):
        for event_index, sample_id in enumerate(event_ids[: int(width)]):
            position = positions.get(str(sample_id))
            if position is None:
                raise ValueError(f"micro_doppler_evidence cache 缺少事件：{sample_id}")
            event_values = np.asarray(cache.values[position], dtype=np.float32)
            frame_valid = np.asarray(cache.valid[position], dtype=np.bool_)
            if event_values.ndim != 2 or event_values.shape[1] != 23:
                raise ValueError(f"micro_doppler_evidence values shape 无效：{sample_id}")
            if frame_valid.shape != (event_values.shape[0],):
                raise ValueError(f"micro_doppler_evidence valid shape 无效：{sample_id}")
            if bool(frame_valid.any()):
                selected = event_values[frame_valid]
                values[row, event_index, :23] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
                values[row, event_index, 23:46] = selected.std(axis=0, dtype=np.float64).astype(np.float32)
                values[row, event_index, 46] = float(frame_valid.sum()) / float(frame_valid.size)
    result = torch.from_numpy(values).to(device=torch.device(device))
    if not torch.isfinite(result).all():
        raise FloatingPointError("micro-doppler evidence summary 包含非有限值")
    return result


def build_head_fit_dataset(
    root: Path,
    *,
    manifest_path: str | Path,
    manifest: DatasetManifest,
    validate_event_cache: bool = True,
) -> CachedParentSequenceDataset:
    """构造仅 train、允许 Unknown 的 auxiliary view。"""

    return CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        manifest=manifest,
        split_name="train",
        include_unknown=True,
        training_view=True,
        allowed_labels=None,
        task_name="l3_head_fit",
        allow_unknown_head_training=True,
        validate_event_cache=validate_event_cache,
    )


def build_head_validation_dataset(
    root: Path,
    *,
    manifest_path: str | Path,
    manifest: DatasetManifest,
    validate_event_cache: bool = True,
) -> CachedParentSequenceDataset:
    return CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        manifest=manifest,
        split_name="validation",
        include_unknown=True,
        training_view=True,
        allowed_labels=None,
        task_name="l3_validation",
        validate_event_cache=validate_event_cache,
    )


def _endpoint_representation(
    task: str,
    *,
    l1: Any,
    l2: Any,
    sequences: list[Any],
    caches: Mapping[str, FeatureCache],
    config: Any,
    device: torch.device,
    return_event_outputs: bool = False,
) -> _EndpointRepresentationBatch:
    if task == "l1":
        model_batch = build_model_batch(
            sequences,
            task="l1",
            max_rd_frames=config.data.max_rd_frames_per_point,
            max_sequence_length=config.data.max_sequence_length,
            feature_caches=caches,
            allow_unmapped_labels=True,
            pad_to_max_sequence_length=True,
        )
        output = l1(
            base_inputs=_l1_inputs(model_batch, device),
            observation_increment_values=model_batch["observation_increment_values"].to(device=device),
            return_representation=True,
        )
        extra = torch.zeros(
            (*output["summary_features"].shape[:2], EXTRA_FEATURE_DIM),
            dtype=output["summary_features"].dtype,
            device=device,
        )
    else:
        model_batch = build_model_batch(
            sequences,
            task="l2",
            max_rd_frames=config.data.max_rd_frames_per_point,
            max_sequence_length=config.data.max_sequence_length,
            stable_median=np.asarray(config.data.l2.stable8_median, dtype=np.float32),
            stable_scale=np.asarray(config.data.l2.stable8_scale, dtype=np.float32),
            stable_indices=config.data.l2.stable8_indices,
            clipping={
                "bounds": {
                    name: {"lower": bounds[0], "upper": bounds[1]}
                    for name, bounds in config.data.l2.tr_clipping_bounds.items()
                }
            },
            feature_caches=caches,
            stable_input_mode=("raw" if l2.mode in {"direct", "direct_grouped"} else "normalized"),
            allow_unmapped_labels=True,
            pad_to_max_sequence_length=True,
        )
        inputs = _l2_inputs(model_batch, device)
        if l2.mode in {"direct", "direct_grouped"}:
            output = l2.direct_logits(base_inputs=inputs, return_representation=True)
        else:
            reference_inputs = _l2_reference_inputs(model_batch, inputs, device=device)
            anchor = l2.frozen_reference_logits(base_inputs=reference_inputs).detach().clone()
            anchor = anchor.masked_fill(inputs["tr_padding_mask"].unsqueeze(-1), 0.0)
            output = l2(
                base_inputs=inputs,
                reference_base_inputs=reference_inputs,
                baseline_logits=anchor,
                stable_summary_tokens=model_batch["stable_summary_tokens"].to(device=device),
                stable_summary_valid=model_batch["stable_summary_valid"].to(device=device),
                stable_frame_values=model_batch["stable_frame_values"].to(device=device),
                stable_frame_mask=model_batch["stable_frame_mask"].to(device=device),
                motion_values=model_batch["motion_values"].to(device=device),
                return_representation=True,
            )
        extra = torch.cat(
            (
                inputs["rd_physical_features"],
                inputs["rd_physical_valid"].unsqueeze(-1).to(dtype=torch.float32),
                inputs["micro_doppler_features"],
                inputs["micro_doppler_valid"].unsqueeze(-1).to(dtype=torch.float32),
                model_batch["stable_summary_tokens"].to(device=device),
                model_batch["stable_summary_valid"].to(device=device).unsqueeze(-1).to(dtype=torch.float32),
                model_batch["motion_values"].to(device=device),
            ),
            dim=-1,
        )
    padding = model_batch["tr_padding_mask"].to(device=device)
    bird_evidence = (
        _micro_doppler_evidence_summary(
            sequences,
            caches,
            width=int(padding.shape[1]),
            device=device,
        )
        if task == "l2"
        else None
    )
    representation = torch.cat(
        (
            output["summary_features"],
            output["event_trajectory"],
            output["task_facts_sequence"],
            extra,
        ),
        dim=-1,
    ).masked_fill(padding.unsqueeze(-1), 0.0)
    if not torch.isfinite(representation).all():
        raise FloatingPointError("knownness representation contains non-finite values")
    lengths = (~padding).long().sum(dim=1)
    rows = torch.arange(lengths.shape[0], device=device)
    endpoint_logits = (
        output["final_logits"]
        if task == "l1"
        else output["sequence_logits"][rows, lengths - 1]
    )
    profile = None
    if task == "l2":
        profile = build_profile_knownness_features(
            inputs["rd_physical_features"],
            inputs["rd_physical_valid"],
            inputs["micro_doppler_features"],
            inputs["micro_doppler_valid"],
            padding,
        ).detach().cpu()
    return _EndpointRepresentationBatch(
        representation=representation.detach().cpu(),
        profile_representation=profile,
        padding_mask=padding.detach().cpu(),
        endpoint_logits=endpoint_logits.detach().cpu(),
        parent_ids=[str(value) for value in model_batch["parent_sample_ids"]],
        label_names=[str(value) for value in model_batch["label_names"]],
        bird_evidence_representation=(
            bird_evidence.detach().cpu() if bird_evidence is not None else None
        ),
        sequence_logits=(
            output["sequence_logits"].detach().cpu() if return_event_outputs else None
        ),
    )


def materialize_frozen_endpoint_representation(
    project_root: str | Path,
    *,
    task: str,
    manifest_path: str | Path,
    feature_cache_manifest_path: str | Path,
    split_name: str,
    training_view: bool,
    allow_unknown_head_training: bool = False,
    device: str = "cpu",
    batch_size: int = 4,
    validate_event_cache: bool = True,
    verify_files: bool = True,
    require_training_manifest_match: bool = True,
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    return_event_outputs: bool = False,
) -> FrozenEndpointRepresentation | FrozenEndpointEventEvidence:
    if task not in {"l1", "l2"}:
        raise ValueError("materialize task 必须为 l1 或 l2")
    root = Path(project_root).resolve()
    target_device = _device(device)
    config = load_model_config(root, model_config_path)
    manifest = load_manifest(
        root / manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    dataset = CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        manifest=manifest,
        split_name=split_name,
        include_unknown=True,
        training_view=bool(training_view),
        allowed_labels=None,
        task_name=(
            "l3_head_fit"
            if allow_unknown_head_training
            else ("l3_audit" if split_name == "test" else f"l3_{split_name}")
        ),
        allow_unknown_head_training=allow_unknown_head_training,
        validate_event_cache=validate_event_cache,
    )
    selected_labels = {
        str(record["label_name"])
        for record in manifest.tr_records
        if str(record["sample_id"]) in set(dataset.view.sample_ids)
    }
    if selected_labels - set(HEAD_LABELS):
        raise ValueError(f"knownness view 含未登记标签：{sorted(selected_labels - set(HEAD_LABELS))}")
    caches, cache_hash = _feature_caches(root, manifest, dataset, feature_cache_manifest_path)
    from radar_ln3.training.l3 import _load_endpoint_models

    l1, l2, l1_checkpoint, l2_checkpoint = _load_endpoint_models(
        root,
        device=target_device,
        config=config,
        source_manifest_sha256=(data_manifest_hash(manifest) if require_training_manifest_match else None),
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: items,
    )
    reps: list[torch.Tensor] = []
    profiles: list[torch.Tensor] = []
    bird_evidence: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    endpoint_logits: list[torch.Tensor] = []
    sequence_logits: list[torch.Tensor] = []
    parent_ids: list[str] = []
    labels: list[str] = []
    with torch.inference_mode():
        for sequences in loader:
            endpoint_result = _endpoint_representation(
                task,
                l1=l1,
                l2=l2,
                sequences=sequences,
                caches=caches,
                config=config,
                device=target_device,
                return_event_outputs=return_event_outputs,
            )
            reps.append(endpoint_result.representation)
            if endpoint_result.profile_representation is not None:
                profiles.append(endpoint_result.profile_representation)
            if endpoint_result.bird_evidence_representation is not None:
                bird_evidence.append(endpoint_result.bird_evidence_representation)
            masks.append(endpoint_result.padding_mask)
            endpoint_logits.append(endpoint_result.endpoint_logits)
            parent_ids.extend(endpoint_result.parent_ids)
            labels.extend(endpoint_result.label_names)
            if endpoint_result.sequence_logits is not None:
                sequence_logits.append(endpoint_result.sequence_logits)
    if not reps:
        raise ValueError("knownness representation 为空")
    representation = torch.cat(reps, dim=0)
    profile_representation = torch.cat(profiles, dim=0) if profiles else None
    bird_evidence_representation = torch.cat(bird_evidence, dim=0) if bird_evidence else None
    padding = torch.cat(masks, dim=0)
    endpoint_logits_tensor = torch.cat(endpoint_logits, dim=0)
    lengths = (~padding).long().sum(dim=1)
    endpoint_hash = {
        "l1": sha256_file(l1_checkpoint.path),
        "l2": sha256_file(l2_checkpoint.path),
    }
    representation = FrozenEndpointRepresentation(
        task=task,
        parent_ids=tuple(parent_ids),
        label_names=tuple(labels),
        representation=representation,
        padding_mask=padding,
        lengths=lengths,
        profile_representation=profile_representation,
        bird_evidence_representation=bird_evidence_representation,
        source_manifest_sha256=data_manifest_hash(manifest),
        endpoint_sha256=endpoint_hash,
        feature_cache_manifest_sha256=cache_hash,
        view_hash=dataset.view.view_hash,
        endpoint_logits=endpoint_logits_tensor,
        test_payload_access_count=(len(dataset) if dataset.view.split_name == "test" else 0),
    )
    if return_event_outputs:
        return FrozenEndpointEventEvidence(
            representation=representation,
            sequence_logits=torch.cat(sequence_logits, dim=0),
        )
    return representation


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def endpoint_representation_path(
    project_root: str | Path,
    *,
    task: str,
    split_name: str,
    representation_root: str | Path = "artifacts/radar_ln3/l3/representations",
) -> Path:
    if task not in {"l1", "l2"} or not str(split_name):
        raise ValueError("endpoint representation task/split 无效")
    root = Path(project_root).resolve()
    return resolve_project_artifact_path(
        root,
        Path(representation_root) / f"{task}-{split_name}.pt",
    )


def save_frozen_endpoint_representation(
    project_root: str | Path,
    representation: FrozenEndpointRepresentation,
    *,
    config: Any,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """保存可复用的 endpoint representation 和完整 provenance。"""

    root = Path(project_root).resolve()
    target = resolve_project_artifact_path(root, output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hidden = int(config.model.liquid_hidden_dim)
    contract = knownness_representation_contract(
        task=representation.task,
        summary_dim=14 * hidden,
        event_trajectory_dim=7 * hidden + 19,
        task_fact_dim=8 if representation.task == "l1" else 7,
        extra_dim=EXTRA_FEATURE_DIM,
    )
    if int(representation.representation.shape[-1]) != sum(contract["dimensions"].values()):
        raise ValueError("endpoint representation 与 contract 输入宽度不一致")
    tensors = {
        "representation": representation.representation.detach().cpu().contiguous(),
        "padding_mask": representation.padding_mask.detach().cpu().contiguous(),
        "lengths": representation.lengths.detach().cpu().contiguous(),
        "endpoint_logits": representation.endpoint_logits.detach().cpu().contiguous(),
    }
    if representation.profile_representation is not None:
        tensors["profile_representation"] = representation.profile_representation.detach().cpu().contiguous()
    if representation.bird_evidence_representation is not None:
        tensors["bird_evidence_representation"] = representation.bird_evidence_representation.detach().cpu().contiguous()
    metadata = {
        "schema_version": REPRESENTATION_CACHE_SCHEMA,
        "task": representation.task,
        "config_sha256": sha256_file(config.path),
        "source_manifest_sha256": representation.source_manifest_sha256,
        "endpoint_sha256": dict(representation.endpoint_sha256),
        "feature_cache_manifest_sha256": representation.feature_cache_manifest_sha256,
        "view_hash": representation.view_hash,
        "parent_ids": list(representation.parent_ids),
        "parent_ids_sha256": canonical_hash(sorted(representation.parent_ids)),
        "label_names": list(representation.label_names),
        "representation_contract": contract,
        "representation_contract_sha256": canonical_hash(contract),
        "tensor_sha256": {name: _tensor_digest(value) for name, value in tensors.items()},
        "test_payload_access_count": int(representation.test_payload_access_count),
    }
    sealed = {**metadata, "artifact_sha256": canonical_hash(metadata)}
    payload = {
        "metadata": sealed,
        "tensors": tensors,
    }
    if target.exists() and not overwrite:
        existing = load_frozen_endpoint_representation(root, target)
        if (
            existing.parent_ids == representation.parent_ids
            and existing.view_hash == representation.view_hash
            and dict(existing.endpoint_sha256) == dict(representation.endpoint_sha256)
            and existing.source_manifest_sha256 == representation.source_manifest_sha256
            and existing.feature_cache_manifest_sha256 == representation.feature_cache_manifest_sha256
        ):
            return target
        raise FileExistsError(f"拒绝覆盖不同的 endpoint representation：{target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_frozen_endpoint_representation(
    project_root: str | Path,
    path: str | Path,
    *,
    expected_task: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_feature_cache_manifest_sha256: str | None = None,
    expected_view_hash: str | None = None,
    expected_endpoint_sha256: Mapping[str, str] | None = None,
    expected_config_sha256: str | None = None,
) -> FrozenEndpointRepresentation:
    root = Path(project_root).resolve()
    target = resolve_project_artifact_path(root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError(f"endpoint representation 文件格式无效：{target}")
    metadata = dict(payload["metadata"])
    artifact_hash = metadata.pop("artifact_sha256", None)
    if artifact_hash != canonical_hash(metadata):
        raise ValueError(f"endpoint representation artifact hash 不匹配：{target}")
    if metadata.get("schema_version") != REPRESENTATION_CACHE_SCHEMA:
        raise ValueError(f"endpoint representation schema 不支持：{target}")
    task = str(metadata.get("task") or "")
    if expected_task is not None and task != expected_task:
        raise ValueError("endpoint representation task 不匹配")
    if expected_source_manifest_sha256 is not None and metadata.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("endpoint representation source manifest hash 不匹配")
    if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError("endpoint representation config hash 不匹配")
    if expected_feature_cache_manifest_sha256 is not None and metadata.get("feature_cache_manifest_sha256") != expected_feature_cache_manifest_sha256:
        raise ValueError("endpoint representation feature cache hash 不匹配")
    if expected_view_hash is not None and metadata.get("view_hash") != expected_view_hash:
        raise ValueError("endpoint representation view hash 不匹配")
    if expected_endpoint_sha256 is not None and dict(metadata.get("endpoint_sha256") or {}) != dict(expected_endpoint_sha256):
        raise ValueError("endpoint representation endpoint hash 不匹配")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("endpoint representation 缺少 tensors")
    tensor_hashes = metadata.get("tensor_sha256")
    if not isinstance(tensor_hashes, Mapping):
        raise ValueError("endpoint representation 缺少 tensor hash")
    for name, digest in tensor_hashes.items():
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or _tensor_digest(value) != digest:
            raise ValueError(f"endpoint representation tensor hash 不匹配：{name}")
    contract = metadata.get("representation_contract")
    if not isinstance(contract, Mapping) or metadata.get("representation_contract_sha256") != canonical_hash(contract):
        raise ValueError("endpoint representation contract hash 不匹配")
    if contract.get("task") != task:
        raise ValueError("endpoint representation contract task 不匹配")
    representation = FrozenEndpointRepresentation(
        task=task,
        parent_ids=tuple(str(value) for value in metadata["parent_ids"]),
        label_names=tuple(str(value) for value in metadata["label_names"]),
        representation=tensors["representation"],
        padding_mask=tensors["padding_mask"].bool(),
        lengths=tensors["lengths"].long(),
        source_manifest_sha256=str(metadata["source_manifest_sha256"]),
        endpoint_sha256=dict(metadata["endpoint_sha256"]),
        feature_cache_manifest_sha256=str(metadata["feature_cache_manifest_sha256"]),
        view_hash=str(metadata["view_hash"]),
        endpoint_logits=tensors["endpoint_logits"],
        profile_representation=tensors.get("profile_representation"),
        bird_evidence_representation=tensors.get("bird_evidence_representation"),
        test_payload_access_count=int(metadata.get("test_payload_access_count", 0)),
    )
    if metadata.get("parent_ids_sha256") != canonical_hash(sorted(representation.parent_ids)):
        raise ValueError("endpoint representation parent hash 不匹配")
    if int(representation.representation.shape[-1]) != sum(contract["dimensions"].values()):
        raise ValueError("endpoint representation contract width 不匹配")
    return representation


def _archive(path: Path) -> None:
    if not path.exists():
        return
    archive = path.with_name(f"{path.name}.stale-before-{sha256_file(path)[:16]}")
    if archive.exists():
        archive.unlink()
    shutil.move(str(path), str(archive))


def _save_summary(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == dict(payload):
            return
        raise FileExistsError(f"拒绝覆盖不同的 knownness summary：{path}")
    if path.exists():
        _archive(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _route_targets(
    task: str,
    labels: Sequence[str],
    time_length: int,
    device: torch.device,
    *,
    l2_classwise: bool = False,
) -> torch.Tensor:
    if task == "l1":
        values = [[label in L2_ROUTES, label == "ClutterTarget"] for label in labels]
    elif l2_classwise:
        values = [
            [label == route_name for route_name in L2_ROUTES]
            for label in labels
        ]
    else:
        values = [[label in L2_ROUTES] for label in labels]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, time_length, -1)


def _knownness_metrics(task: str, labels: Sequence[str], scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    known = np.asarray(
        [label != "UnknownTarget" if task == "l1" else label in L2_ROUTES for label in labels],
        dtype=bool,
    )
    unknown = np.asarray([label == "UnknownTarget" for label in labels], dtype=bool)
    if task == "l1":
        route_names = L1_ROUTES
        route_ids = np.argmax(values, axis=1)
        accepted = values.max(axis=1) >= 0.5
        expected = np.asarray(["target" if label in L2_ROUTES else "clutter" for label in labels], dtype=object)
        expected_ids = np.asarray(
            [route_names.index(route) if known[row] else len(route_names) for row, route in enumerate(expected)],
            dtype=np.int64,
        )
        correct = accepted & np.asarray([route_names[int(index)] == expected[row] for row, index in enumerate(route_ids)], dtype=bool)
        per_route = {}
        for index, route in enumerate(route_names):
            route_rows = known & accepted & (route_ids == index)
            per_route[route] = {
                "support": int(route_rows.sum()),
                "score_mean": float(values[:, index].mean()),
                "accepted_precision": float((correct & route_rows).sum() / route_rows.sum()) if route_rows.any() else 0.0,
            }
        known_correct = known & correct
        minimum_precision = min(item["accepted_precision"] for item in per_route.values())
    elif values.shape[1] == len(L2_ROUTES):
        route_names = L2_ROUTES
        route_ids = np.argmax(values, axis=1)
        accepted = values.max(axis=1) >= 0.5
        expected_ids = np.asarray(
            [L2_ROUTES.index(label) if label in L2_ROUTES else len(route_names) for label in labels],
            dtype=np.int64,
        )
        correct = accepted & known & (route_ids == expected_ids)
        per_route = {}
        for index, route in enumerate(route_names):
            route_rows = known & accepted & (route_ids == index)
            per_route[route] = {
                "support": int(route_rows.sum()),
                "score_mean": float(values[:, index].mean()),
                "accepted_precision": float((correct & route_rows).sum() / route_rows.sum()) if route_rows.any() else 0.0,
            }
        known_correct = known & correct
        minimum_precision = min(item["accepted_precision"] for item in per_route.values())
    else:
        route_names = ("target",)
        route_ids = np.zeros(len(labels), dtype=np.int64)
        expected_ids = np.where(known, 0, len(route_names))
        accepted = values[:, 0] >= 0.5
        known_correct = known & accepted
        per_route = {
            "target": {
                "support": int(known.sum()),
                "score_mean": float(values[:, 0].mean()),
                "accepted_precision": float(known_correct.sum() / (known & accepted).sum()) if (known & accepted).any() else 0.0,
            }
        }
        minimum_precision = per_route["target"]["accepted_precision"]
    accepted_known = known & accepted
    return {
        "known_accepted_precision": float(known_correct.sum() / accepted_known.sum()) if accepted_known.any() else 0.0,
        "minimum_route_accepted_precision": float(minimum_precision),
        "known_coverage": float(accepted_known.sum() / known.sum()) if known.any() else 0.0,
        "unknown_false_acceptance_rate": float((unknown & accepted).sum() / unknown.sum()) if unknown.any() else None,
        "accepted_support": int(accepted_known.sum()),
        "known_support": int(known.sum()),
        "unknown_support": int(unknown.sum()),
        "per_route": per_route,
        # 拒识是 knownness 的实际输出；Unknown 误接受也计入对应 route 的 FP。
        # 此诊断固定使用 0.5，不改变 operating point 拟合或 checkpoint 选择。
        "per_class": _metrics(
            np.where(accepted, route_ids, len(route_names)),
            expected_ids,
            (*route_names, "rejected"),
        )["per_class"],
    }


def _l1_operating_point(
    labels: Sequence[str],
    scores: np.ndarray,
    *,
    unknown_far_limit: float,
) -> dict[str, Any]:
    """在 Target route 分数上选择一个 validation operating point。"""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(L1_ROUTES):
        raise ValueError("L1 operating point 需要 [N,2] route scores")
    target_scores = values[:, 0]
    label_array = np.asarray(labels, dtype=object)
    target_rows = np.isin(label_array, L2_ROUTES)
    bird_rows = label_array == "BirdTarget"
    unknown_rows = label_array == "UnknownTarget"
    known_rows = label_array != "UnknownTarget"
    candidates = np.unique(np.concatenate((target_scores, np.asarray([0.0, 1.0]))))
    feasible: list[dict[str, Any]] = []
    for threshold in candidates:
        accepted = target_scores >= float(threshold)
        accepted_known = accepted & known_rows
        point = {
            "threshold": float(threshold),
            "target_recall": float(accepted[target_rows].mean()) if target_rows.any() else 0.0,
            "bird_recall": float(accepted[bird_rows].mean()) if bird_rows.any() else 0.0,
            "unknown_false_acceptance_rate": float(accepted[unknown_rows].mean()) if unknown_rows.any() else None,
            "known_accepted_precision": float(
                accepted[target_rows].sum() / accepted_known.sum()
            ) if accepted_known.any() else 0.0,
            "target_accepted_support": int(accepted[target_rows].sum()),
            "known_accepted_support": int(accepted_known.sum()),
            "unknown_accepted_support": int(accepted[unknown_rows].sum()),
            "unknown_support": int(unknown_rows.sum()),
        }
        if (
            point["unknown_false_acceptance_rate"] is None
            or float(point["unknown_false_acceptance_rate"]) <= float(unknown_far_limit)
        ):
            feasible.append(point)
    if not feasible:
        feasible = [
            {
                "threshold": 1.0,
                "target_recall": 0.0,
                "bird_recall": 0.0,
                "unknown_false_acceptance_rate": 0.0,
                "known_accepted_precision": 0.0,
                "target_accepted_support": 0,
                "known_accepted_support": 0,
                "unknown_accepted_support": 0,
                "unknown_support": int(unknown_rows.sum()),
            }
        ]
    return max(
        feasible,
        key=lambda item: (
            float(item["bird_recall"]),
            float(item["target_recall"]),
            float(item["known_accepted_precision"]),
            -float(item["unknown_false_acceptance_rate"] or 0.0),
            -float(item["threshold"]),
        ),
    )


def _train_knownness_head(
    project_root: str | Path,
    *,
    task: str,
    fit: FrozenEndpointRepresentation,
    validation: FrozenEndpointRepresentation,
    config: Any,
    output_path: str | Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    projection_dim: int,
    min_events: int,
    seed: int,
    overwrite: bool,
    l1_route_loss: bool = False,
    l1_target_weight: float = 1.0,
    l1_clutter_weight: float = 0.75,
    l1_endpoint_weight: float = 0.0,
    l1_bce_unknown_weight: float = 1.0,
    l1_bird_loss_weight: float = 1.0,
    l1_loss_mode: str = "prefix_endpoint",
    l1_tail_weight: float = 0.0,
    l1_tail_events: int = 4,
    l1_target_prefix_weight: float = 0.0,
    l1_clutter_prefix_weight: float = 0.0,
    unknown_loss_weight: float | None = None,
    l2_unknown_loss_weight: float = 1.0,
    l2_classwise: bool = False,
    bird_loss_weight: float = 1.0,
    unknown_far_limit: float = 0.65,
    l1_ignore_unknown: bool = False,
    l1_selection_mode: str = "fixed",
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if task == "l2" and (fit.profile_representation is None or validation.profile_representation is None):
        raise ValueError("L2 profile residual requires profile representation")
    target_device = _device(device)
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if l1_ignore_unknown and task != "l1":
        raise ValueError("l1_ignore_unknown 只适用于 L1 knownness head")
    if l1_selection_mode not in {"fixed", "operating_point"}:
        raise ValueError("l1_selection_mode 必须为 fixed 或 operating_point")
    if task != "l1" and l1_selection_mode != "fixed":
        raise ValueError("l1_selection_mode 只适用于 L1 knownness head")
    if float(l1_bird_loss_weight) != 1.0 and (task != "l1" or l1_route_loss or l1_ignore_unknown):
        raise ValueError("l1_bird_loss_weight 只适用于 L1 balanced BCE 损失")
    if task == "l2" and not l2_classwise and float(bird_loss_weight) != 1.0:
        raise ValueError(
            "L2 非 classwise knownness 没有 Bird 类别权重；"
            "请将 bird_loss_weight 设为 1.0，或启用 classwise"
        )
    if task == "l2" and unknown_loss_weight not in (None, 0.15):
        raise ValueError("L2 不使用 L1 route 的 unknown_loss_weight；请使用 l2_unknown_loss_weight")
    effective_unknown_loss_weight = (
        0.15 if unknown_loss_weight is None else float(unknown_loss_weight)
    )
    if task == "l1" and (
        float(l2_unknown_loss_weight) != 1.0
        or l2_classwise
        or float(bird_loss_weight) != 1.0
    ):
        raise ValueError("L1 不接受 L2 专用的 Unknown、classwise 或 Bird 权重参数")
    if l1_loss_mode not in {"prefix_endpoint", "endpoint_tail"}:
        raise ValueError("l1_loss_mode 必须为 prefix_endpoint 或 endpoint_tail")
    if task != "l1" and l1_loss_mode != "prefix_endpoint":
        raise ValueError("l1_loss_mode 只适用于 L1 knownness head")
    if l1_loss_mode == "endpoint_tail" and (l1_route_loss or l1_ignore_unknown):
        raise ValueError("endpoint_tail 只适用于 L1 balanced BCE 损失")
    if int(l1_tail_events) < 1:
        raise ValueError("l1_tail_events 必须为正整数")
    for name, value in (
        ("l1_endpoint_weight", l1_endpoint_weight),
        ("l1_tail_weight", l1_tail_weight),
        ("l1_target_prefix_weight", l1_target_prefix_weight),
        ("l1_clutter_prefix_weight", l1_clutter_prefix_weight),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} 必须为有限非负数")
    if l1_loss_mode == "endpoint_tail" and (
        float(l1_endpoint_weight) <= 0.0 or float(l1_tail_weight) <= 0.0
    ):
        raise ValueError("endpoint_tail 必须设置正的 endpoint 和 tail 损失权重")
    if l1_loss_mode == "prefix_endpoint" and float(l1_tail_weight) != 0.0:
        raise ValueError("l1_tail_weight 只适用于 endpoint_tail 模式")
    route_prefix_enabled = bool(
        float(l1_target_prefix_weight) > 0.0 or float(l1_clutter_prefix_weight) > 0.0
    )
    if route_prefix_enabled and (
        task != "l1" or l1_loss_mode != "endpoint_tail" or l1_route_loss or l1_ignore_unknown
    ):
        raise ValueError("route 前缀辅助权重只适用于 L1 endpoint_tail balanced BCE 损失")
    experimental_selection = bool(
        l1_route_loss
        or l2_classwise
        or l1_ignore_unknown
        or (task == "l1" and l1_selection_mode == "operating_point")
        or (task == "l2" and float(l2_unknown_loss_weight) < 1.0)
    )
    route_names = (
        L1_ROUTES
        if task == "l1"
        else (L2_ROUTES if l2_classwise else ("target",))
    )
    for name, value in (
        ("l1_target_weight", l1_target_weight),
        ("l1_clutter_weight", l1_clutter_weight),
        ("l1_bce_unknown_weight", l1_bce_unknown_weight),
        ("l1_bird_loss_weight", l1_bird_loss_weight),
        ("unknown_loss_weight", effective_unknown_loss_weight),
        ("l2_unknown_loss_weight", l2_unknown_loss_weight),
        ("bird_loss_weight", bird_loss_weight),
        ("unknown_far_limit", unknown_far_limit),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须为正数")
    representation_contract = knownness_representation_contract(
        task=task,
        summary_dim=14 * int(config.model.liquid_hidden_dim),
        event_trajectory_dim=7 * int(config.model.liquid_hidden_dim) + 19,
        task_fact_dim=8 if task == "l1" else 7,
        extra_dim=EXTRA_FEATURE_DIM,
    )
    if int(fit.representation.shape[-1]) != sum(representation_contract["dimensions"].values()):
        raise ValueError("knownness representation input width 与 contract 不一致")
    if int(validation.representation.shape[-1]) != sum(representation_contract["dimensions"].values()):
        raise ValueError("knownness validation representation input width 与 contract 不一致")
    if fit.source_manifest_sha256 != validation.source_manifest_sha256:
        raise ValueError("knownness fit/validation source manifest 不一致")
    if dict(fit.endpoint_sha256) != dict(validation.endpoint_sha256):
        raise ValueError("knownness fit/validation endpoint checkpoint 不一致")
    base = TemporalKnownnessHead(
        int(fit.representation.shape[-1]),
        route_names=route_names,
        hidden_dim=max(1, int(hidden_dim)),
        projection_dim=max(1, int(projection_dim)),
    ).to(target_device)
    fit_rep = fit.representation.to(device=target_device)
    fit_pad = fit.padding_mask.to(device=target_device)
    val_rep = validation.representation.to(device=target_device)
    val_pad = validation.padding_mask.to(device=target_device)
    fit_profile = None if fit.profile_representation is None else fit.profile_representation.to(device=target_device)
    val_profile = None if validation.profile_representation is None else validation.profile_representation.to(device=target_device)
    fit_targets = _route_targets(
        task,
        fit.label_names,
        time_length=1,
        device=torch.device("cpu"),
        l2_classwise=l2_classwise,
    )
    positive_counts = fit_targets.sum(dim=(0, 1)).tolist()
    weights = []
    for route_index, route in enumerate(route_names):
        positive = int(positive_counts[route_index])
        if task == "l2" and not l2_classwise and positive <= 0:
            raise ValueError("L2 非 classwise knownness 没有已知类正样本，拒绝训练")
        weights.append(
            max(0.5, min(20.0, (len(fit.label_names) - positive) / max(1, positive)))
        )
    positive_weights = torch.tensor(weights, dtype=torch.float32, device=target_device)
    class_loss_weights = torch.ones(len(route_names), dtype=torch.float32, device=target_device)
    if task == "l1":
        class_loss_weights = torch.tensor(
            [float(l1_target_weight), float(l1_clutter_weight)],
            dtype=torch.float32,
            device=target_device,
        )
    elif task == "l2" and l2_classwise:
        class_loss_weights[L2_ROUTES.index("BirdTarget")] = float(bird_loss_weight)
    route_prefix_weights = (
        torch.tensor(
            [float(l1_target_prefix_weight), float(l1_clutter_prefix_weight)],
            dtype=torch.float32,
            device=target_device,
        )
        if route_prefix_enabled else None
    )

    def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        profile: bool,
    ) -> tuple[float, dict[str, float]]:
        model.train()
        order = _epoch_permutation(
            len(fit.parent_ids),
            seed=seed,
            epoch=epoch,
            phase_offset=int(epochs) if profile else 0,
        )
        losses: list[float] = []
        component_totals: dict[str, torch.Tensor] = {}
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            # 表示已经驻留在目标设备；index_select 的索引也必须位于同一设备。
            device_indices = indices.to(device=target_device)
            reps = fit_rep.index_select(0, device_indices)
            pads = fit_pad.index_select(0, device_indices)
            labels = [fit.label_names[int(index)] for index in indices.tolist()]
            targets = _route_targets(
                task,
                labels,
                reps.shape[1],
                target_device,
                l2_classwise=l2_classwise,
            )
            if profile:
                if fit_profile is None:
                    raise ValueError("L2 profile representation 未驻留")
                output = model(
                    reps,
                    fit_profile.index_select(0, device_indices),
                    pads,
                )
                logits = output["route_logits"]
            else:
                output = model(reps, pads)
                logits = output["route_logits"]
            active = ~pads
            prefix = active & (active.long().cumsum(dim=1) >= int(min_events))
            route_loss_values = None
            if task == "l1" and (l1_route_loss or l1_ignore_unknown):
                route_ids = torch.tensor(
                    [0 if label in L2_ROUTES else 1 for label in labels],
                    dtype=torch.long,
                    device=target_device,
                ).unsqueeze(1).expand(-1, reps.shape[1])
                known_rows = torch.tensor(
                    [label != "UnknownTarget" for label in labels],
                    dtype=torch.bool,
                    device=target_device,
                ).unsqueeze(1).expand(-1, reps.shape[1])
                unknown_rows = ~known_rows
                route_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    route_ids.reshape(-1),
                    weight=class_loss_weights,
                    reduction="none",
                ).reshape_as(route_ids)
                unknown_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    torch.zeros_like(logits),
                    reduction="none",
                ).mean(dim=-1)
                loss_values = route_loss * known_rows.to(dtype=route_loss.dtype)
                if not l1_ignore_unknown:
                    loss_values = loss_values + effective_unknown_loss_weight * unknown_loss * unknown_rows.to(dtype=unknown_loss.dtype)
            else:
                loss_values = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=positive_weights.view(1, 1, -1),
                    reduction="none",
                )
                if task == "l1":
                    loss_values = loss_values * class_loss_weights.view(1, 1, -1)
                if task == "l2" and l2_classwise:
                    loss_values = loss_values * class_loss_weights.view(1, 1, -1)
                if task == "l2" and float(l2_unknown_loss_weight) != 1.0:
                    unknown_rows = torch.tensor(
                        [label == "UnknownTarget" for label in labels],
                        dtype=torch.bool,
                        device=target_device,
                    ).view(-1, 1, 1)
                    loss_values = loss_values * torch.where(
                        unknown_rows,
                        torch.full_like(loss_values, float(l2_unknown_loss_weight)),
                        torch.ones_like(loss_values),
                    )
                if task == "l1" and float(l1_bce_unknown_weight) != 1.0:
                    unknown_rows = torch.tensor(
                        [label == "UnknownTarget" for label in labels],
                        dtype=torch.bool,
                        device=target_device,
                    ).view(-1, 1, 1)
                    loss_values = loss_values * torch.where(
                        unknown_rows,
                        torch.full_like(loss_values, float(l1_bce_unknown_weight)),
                        torch.ones_like(loss_values),
                    )
                if task == "l1" and float(l1_bird_loss_weight) != 1.0:
                    # Bird 同时监督 Target 正例与 Clutter 负例，只调整训练损失。
                    bird_weights = torch.tensor(
                        [float(l1_bird_loss_weight) if label == "BirdTarget" else 1.0 for label in labels],
                        dtype=loss_values.dtype,
                        device=target_device,
                    ).view(-1, 1, 1)
                    loss_values = loss_values * bird_weights
                if route_prefix_enabled:
                    # 保留同一份已加权 BCE，给指定 route 增加整段因果前缀监督。
                    route_loss_values = loss_values
                loss_values = loss_values.mean(dim=-1)
            if loss_values.ndim == 3:
                loss_values = loss_values.mean(dim=-1)
            endpoint_loss = None
            if task == "l1" and float(l1_endpoint_weight) > 0.0:
                last_indices = active.long().sum(dim=1).sub(1).clamp_min(0)
                endpoint_loss = loss_values.gather(
                    1, last_indices.unsqueeze(1)
                ).squeeze(1).mean()
            if task == "l1" and l1_loss_mode == "endpoint_tail":
                assert endpoint_loss is not None
                time_indices = torch.arange(
                    loss_values.shape[1], device=target_device
                ).unsqueeze(0)
                # 只监督最终事件之前的成熟因果前缀，排除 padding 和最终事件本身。
                tail_start = (last_indices - int(l1_tail_events)).unsqueeze(1)
                tail_mask = (
                    prefix
                    & (time_indices >= tail_start)
                    & (time_indices < last_indices.unsqueeze(1))
                )
                tail_loss = (
                    loss_values.masked_fill(~tail_mask, 0.0).sum(dim=1)
                    / tail_mask.sum(dim=1).clamp_min(1)
                ).mean()
                loss = float(l1_endpoint_weight) * endpoint_loss + float(l1_tail_weight) * tail_loss
            else:
                prefix_loss = (
                    loss_values.masked_fill(~prefix, 0.0).sum(dim=1)
                    / prefix.sum(dim=1).clamp_min(1)
                ).mean()
                loss = prefix_loss
                if endpoint_loss is not None:
                    loss = loss + float(l1_endpoint_weight) * endpoint_loss
            if route_prefix_enabled:
                assert route_loss_values is not None and route_prefix_weights is not None
                prefix_losses_by_route = (
                    route_loss_values.masked_fill(~prefix.unsqueeze(-1), 0.0).sum(dim=1)
                    / prefix.sum(dim=1).clamp_min(1).unsqueeze(-1)
                )
                # 主损失对两条 route 取平均，辅助项沿用同一归一化，避免额外放大。
                route_prefix_loss = (
                    prefix_losses_by_route * route_prefix_weights.view(1, -1)
                ).mean()
                loss = loss + route_prefix_loss
            if task == "l1":
                # 同一训练损失拆项落盘，便于核对最终事件和前缀的实际权重。
                components = {"final_event_loss": endpoint_loss} if endpoint_loss is not None else {}
                if l1_loss_mode == "endpoint_tail":
                    components["tail_prefix_loss"] = tail_loss
                else:
                    components["all_prefix_loss"] = prefix_loss
                if route_prefix_enabled:
                    route_means = prefix_losses_by_route.mean(dim=0)
                    components["target_all_prefix_loss"] = route_means[0]
                    components["clutter_all_prefix_loss"] = route_means[1]
                    components["weighted_route_prefix_loss"] = route_prefix_loss
                for name, value in components.items():
                    detached = value.detach()
                    component_totals[name] = (
                        component_totals[name] + detached
                        if name in component_totals else detached
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return (
            float(np.mean(losses)) if losses else 0.0,
            {name: float((total / max(1, len(losses))).cpu()) for name, total in component_totals.items()},
        )

    history: list[dict[str, Any]] = []
    best_rank: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_operating_point: dict[str, Any] | None = None
    best_epoch = 0
    optimizer = torch.optim.AdamW(base.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for epoch in range(1, int(epochs) + 1):
        loss, loss_components = train_epoch(base, optimizer, epoch=epoch, profile=False)
        base.eval()
        with torch.inference_mode():
            output = base(val_rep, val_pad)
            rows = torch.arange(val_rep.shape[0], device=target_device)
            scores = output["route_probabilities"][rows, validation.lengths.to(device=target_device) - 1].cpu().numpy()
        metrics = _knownness_metrics(task, validation.label_names, scores)
        operating_point = (
            _l1_operating_point(
                validation.label_names,
                scores,
                unknown_far_limit=unknown_far_limit,
            )
            if task == "l1"
            else None
        )
        history_item = {"epoch": epoch, "train_loss": loss, "validation": metrics}
        if loss_components:
            history_item["loss_mode"] = str(l1_loss_mode)
            history_item["loss_components"] = loss_components
        if operating_point is not None:
            history_item["operating_point"] = operating_point
        history.append(history_item)
        if progress_callback is not None:
            progress_callback({"stage": "knownness_epoch", "task": task, "phase": "base", **history_item})
        log_classification_metrics(
            metrics, task=f"L3/{task}-knownness", epoch=epoch,
            phase="validation phase=base threshold=0.5",
        )
        unknown_far = float(metrics["unknown_false_acceptance_rate"] or 0.0)
        if task == "l1" and l1_selection_mode == "operating_point":
            assert operating_point is not None
            rank = (
                float(
                    operating_point["unknown_false_acceptance_rate"] is None
                    or float(operating_point["unknown_false_acceptance_rate"])
                    <= float(unknown_far_limit)
                ),
                float(operating_point["bird_recall"]),
                float(operating_point["target_recall"]),
                float(operating_point["known_accepted_precision"]),
                -float(operating_point["unknown_false_acceptance_rate"] or 0.0),
                -float(epoch),
            )
        elif experimental_selection:
            correct_coverage = float(metrics["known_coverage"]) * float(
                metrics["known_accepted_precision"]
            )
            rank = (
                float(unknown_far <= float(unknown_far_limit)),
                correct_coverage,
                float(metrics["known_accepted_precision"]),
                -unknown_far,
                -float(epoch),
            )
        else:
            rank = (
                -unknown_far,
                float(metrics["known_coverage"]),
                float(metrics["known_accepted_precision"]),
                -float(epoch),
            )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}
            best_metrics = metrics
            best_epoch = epoch
            best_operating_point = operating_point
    if best_state is None or best_metrics is None:
        raise RuntimeError("knownness head did not produce a checkpoint")
    base.load_state_dict(best_state, strict=True)
    model: nn.Module = base
    profile_history: list[dict[str, Any]] = []
    if task == "l2":
        model = ProfileResidualKnownnessHead(base).to(target_device)
        profile_optimizer = torch.optim.AdamW(model.profile_branch.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        profile_best = None
        profile_best_state = None
        for epoch in range(1, int(epochs) + 1):
            loss, _ = train_epoch(model, profile_optimizer, epoch=epoch, profile=True)
            model.eval()
            with torch.inference_mode():
                if val_profile is None:
                    raise ValueError("L2 validation profile representation 未驻留")
                output = model(val_rep, val_profile, val_pad)
                rows = torch.arange(val_rep.shape[0], device=target_device)
                scores = output["route_probabilities"][rows, validation.lengths.to(device=target_device) - 1].cpu().numpy()
            metrics = _knownness_metrics(task, validation.label_names, scores)
            profile_history.append({"epoch": epoch, "train_loss": loss, "validation": metrics})
            if progress_callback is not None:
                progress_callback({"stage": "knownness_epoch", "task": task, "phase": "profile", **profile_history[-1]})
            log_classification_metrics(
                metrics, task=f"L3/{task}-knownness", epoch=epoch,
                phase="validation phase=profile threshold=0.5",
            )
            unknown_far = float(metrics["unknown_false_acceptance_rate"] or 0.0)
            if experimental_selection:
                correct_coverage = float(metrics["known_coverage"]) * float(
                    metrics["known_accepted_precision"]
                )
                rank = (
                    float(unknown_far <= float(unknown_far_limit)),
                    correct_coverage,
                    float(metrics["known_accepted_precision"]),
                    -unknown_far,
                    -float(epoch),
                )
            else:
                rank = (
                    -unknown_far,
                    float(metrics["known_coverage"]),
                    float(metrics["known_accepted_precision"]),
                    -float(epoch),
                )
            if profile_best is None or rank > profile_best:
                profile_best = rank
                profile_best_state = {name: value.detach().cpu().clone() for name, value in model.profile_branch.state_dict().items()}
                best_metrics = metrics
                best_epoch = epoch
        if profile_best_state is not None:
            model.profile_branch.load_state_dict(profile_best_state, strict=True)
    contract = model.model_contract()
    body = {
        "checkpoint_schema": KNOWNNESS_HEAD_SCHEMA,
        "schema_version": KNOWNNESS_HEAD_SCHEMA,
        "task": f"l3_knownness_{task}",
        "head_task": task,
        "model_contract": contract,
        "model_contract_sha256": canonical_hash(contract),
        "representation_contract": representation_contract,
        "representation_contract_sha256": canonical_hash(representation_contract),
        "endpoint_sha256": dict(fit.endpoint_sha256),
        "config_sha256": sha256_file(config.path),
        "source_manifest_sha256": fit.source_manifest_sha256,
        "feature_cache_manifest_sha256": {"fit": fit.feature_cache_manifest_sha256, "validation": validation.feature_cache_manifest_sha256},
        "fit_view_hash": fit.view_hash,
        "validation_view_hash": validation.view_hash,
        "fit_parent_ids_sha256": canonical_hash(sorted(fit.parent_ids)),
        "validation_parent_ids_sha256": canonical_hash(sorted(validation.parent_ids)),
        "fit_label_support": {label: int(fit.label_names.count(label)) for label in HEAD_LABELS},
        "validation_label_support": {label: int(validation.label_names.count(label)) for label in HEAD_LABELS},
        "seed": int(seed),
        "route_names": list(route_names),
        "training_variant": {
            "l1_objective": (
                "balanced_bce_endpoint"
                if task == "l1" and float(l1_endpoint_weight) > 0.0
                else "route_ce"
                if task == "l1" and (l1_route_loss or l1_ignore_unknown)
                else "balanced_bce"
                if task == "l1"
                else "l2_knownness"
            ),
            "l1_route_loss": bool(l1_route_loss),
            "l1_target_weight": float(l1_target_weight),
            "l1_clutter_weight": float(l1_clutter_weight),
            "l1_endpoint_weight": float(l1_endpoint_weight),
            "l1_bce_unknown_weight": float(l1_bce_unknown_weight),
            "l1_bird_loss_weight": float(l1_bird_loss_weight),
            "l1_loss_mode": str(l1_loss_mode),
            "l1_tail_weight": float(l1_tail_weight),
            "l1_tail_events": int(l1_tail_events),
            "l1_target_prefix_weight": float(l1_target_prefix_weight),
            "l1_clutter_prefix_weight": float(l1_clutter_prefix_weight),
            "unknown_loss_weight": (
                float(effective_unknown_loss_weight) if task == "l1" else None
            ),
            "unknown_loss_weight_applied": bool(
                task == "l1" and (l1_route_loss or l1_ignore_unknown)
            ),
            "l2_unknown_loss_weight": float(l2_unknown_loss_weight),
            "l2_unknown_loss_weight_applied": bool(task == "l2"),
            "l2_classwise": bool(l2_classwise),
            "bird_loss_weight": float(bird_loss_weight),
            "bird_loss_weight_applied": bool(task == "l2" and l2_classwise),
            "effective_bird_loss_weight": float(
                bird_loss_weight if task == "l2" and l2_classwise else 1.0
            ),
            "unknown_far_limit": float(unknown_far_limit),
            "l1_ignore_unknown": bool(l1_ignore_unknown),
            "experimental_selection": bool(experimental_selection),
            "l1_selection_mode": str(l1_selection_mode),
        },
        "selected_epoch": int(best_epoch),
        "selection": {
            "priority": (
                [
                    "l1_operating_point_bird_recall",
                    "l1_operating_point_target_recall",
                    "l1_operating_point_known_precision",
                ]
                if task == "l1" and l1_selection_mode == "operating_point"
                else [
                    "unknown_far_within_limit",
                    "correct_known_coverage",
                    "known_accepted_precision",
                    "lowest_unknown_far",
                    "earliest_epoch",
                ]
                if experimental_selection
                else [
                    "unknown_rejection",
                    "known_coverage",
                    "known_accepted_precision",
                    "earliest_epoch",
                ]
            ),
            "selected_validation": best_metrics,
            "selected_operating_point": best_operating_point,
        },
        "history": history,
        "profile_history": profile_history,
        "test_payload_access_count": 0,
    }
    body["training_parameters"] = {
        "device": str(target_device),
        "epochs": int(epochs),
        "head_batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "projection_dim": int(projection_dim),
        "min_events": int(min_events),
        "seed": int(seed),
    }
    target = resolve_project_artifact_path(Path(project_root).resolve(), output_path)
    if target.exists() and overwrite:
        _archive(target)
    checkpoint, _ = Checkpoint.save(project_root, output_path, {**body, "model_state": model.state_dict()})
    summary = {
        "schema_version": "RADAR-LN3-KNOWNNESS-SUMMARY-V2",
        "task": task,
        "training_variant": body["training_variant"],
        "training_parameters": body["training_parameters"],
        "checkpoint": checkpoint.provenance(kind="l3_knownness_head", task=f"l3_knownness_{task}"),
        "selected_epoch": int(best_epoch),
        "selection": body["selection"],
        "history": history,
        "profile_history": profile_history,
        "fit_parent_count": len(fit.parent_ids),
        "validation_parent_count": len(validation.parent_ids),
        "test_payload_access_count": 0,
    }
    _save_summary(target.with_name(f"{target.stem}-summary.json"), summary, overwrite=overwrite)
    return summary


def train_l3_knownness_head(
    project_root: str | Path,
    *,
    task: str,
    fit: FrozenEndpointRepresentation,
    validation: FrozenEndpointRepresentation,
    config: Any,
    output_path: str | Path,
    device: str = "cpu",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    hidden_dim: int = 64,
    projection_dim: int = 32,
    min_events: int = 1,
    seed: int = 42,
    overwrite: bool = False,
    l1_route_loss: bool = False,
    l1_target_weight: float = 1.0,
    l1_clutter_weight: float = 0.75,
    l1_endpoint_weight: float = 0.0,
    l1_bce_unknown_weight: float = 1.0,
    l1_bird_loss_weight: float = 1.0,
    l1_loss_mode: str = "prefix_endpoint",
    l1_tail_weight: float = 0.0,
    l1_tail_events: int = 4,
    l1_target_prefix_weight: float = 0.0,
    l1_clutter_prefix_weight: float = 0.0,
    unknown_loss_weight: float | None = None,
    l2_unknown_loss_weight: float = 1.0,
    l2_classwise: bool = False,
    bird_loss_weight: float = 1.0,
    unknown_far_limit: float = 0.65,
    l1_ignore_unknown: bool = False,
    l1_selection_mode: str = "fixed",
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return _train_knownness_head(
        project_root,
        task=task,
        fit=fit,
        validation=validation,
        config=config,
        output_path=output_path,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        min_events=max(3, int(min_events)),
        seed=seed,
        overwrite=overwrite,
        l1_route_loss=l1_route_loss,
        l1_target_weight=l1_target_weight,
        l1_clutter_weight=l1_clutter_weight,
        l1_endpoint_weight=l1_endpoint_weight,
        l1_bce_unknown_weight=l1_bce_unknown_weight,
        l1_bird_loss_weight=l1_bird_loss_weight,
        l1_loss_mode=l1_loss_mode,
        l1_tail_weight=l1_tail_weight,
        l1_tail_events=l1_tail_events,
        l1_target_prefix_weight=l1_target_prefix_weight,
        l1_clutter_prefix_weight=l1_clutter_prefix_weight,
        unknown_loss_weight=unknown_loss_weight,
        l2_unknown_loss_weight=l2_unknown_loss_weight,
        l2_classwise=l2_classwise,
        bird_loss_weight=bird_loss_weight,
        unknown_far_limit=unknown_far_limit,
        l1_ignore_unknown=l1_ignore_unknown,
        l1_selection_mode=l1_selection_mode,
        progress_callback=progress_callback,
    )
def load_l3_knownness_head(
    project_root: str | Path,
    path: str | Path,
    *,
    task: str,
    expected_endpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> tuple[TemporalKnownnessHead | ProfileResidualKnownnessHead, Checkpoint]:
    checkpoint = Checkpoint.load(
        project_root,
        path,
    )
    payload = checkpoint.payload
    if payload.get("head_task") != task or payload.get("task") != f"l3_knownness_{task}":
        raise ValueError("knownness head task 不匹配")
    contract = payload.get("model_contract")
    if not isinstance(contract, Mapping) or payload.get("model_contract_sha256") != canonical_hash(contract):
        raise ValueError("knownness head model contract hash 不一致")
    if expected_endpoint_sha256 is not None:
        endpoints = payload.get("endpoint_sha256")
        if (
            not isinstance(endpoints, Mapping)
            or str(endpoints.get(task) or "") != str(expected_endpoint_sha256)
        ):
            raise ValueError("knownness head endpoint hash 不匹配")
    if expected_config_sha256 is not None and payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("knownness head config hash 不匹配")
    schema = str(payload.get("checkpoint_schema") or payload.get("schema_version") or "")
    if schema not in {
        KNOWNNESS_HEAD_SCHEMA,
        KNOWNNESS_HEAD_SCHEMA_V3,
        KNOWNNESS_HEAD_SCHEMA_V4,
        KNOWNNESS_HEAD_SCHEMA_V5,
        KNOWNNESS_HEAD_SCHEMA_V6,
    }:
        raise ValueError(
            "knownness head schema 不支持或属于旧产物，请重新训练当前 L3 head："
            f"{schema}"
        )
    class_path = str(contract.get("class_path") or "")
    if class_path.endswith("BirdTemporalStabilityResidualKnownnessHead"):
        baseline_contract = contract.get("baseline_contract")
        if not isinstance(baseline_contract, Mapping):
            raise ValueError("Bird temporal stability head 缺少 baseline contract")
        baseline = TemporalKnownnessHead.from_contract(baseline_contract)
        model = BirdTemporalStabilityResidualKnownnessHead(
            baseline,
            profile_dim=int(contract.get("profile_feature_dim", 44)),
            extra_dim=int(contract.get("extra_feature_dim", 32)),
            evidence_dim=int(contract.get("evidence_feature_dim", MICRO_DOPPLER_EVIDENCE_SUMMARY_DIM)),
            standardization_mean=contract.get("temporal_residual_standardization_mean", ()),
            standardization_scale=contract.get("temporal_residual_standardization_scale", ()),
            ridge_alpha=float(contract.get("temporal_residual_ridge_alpha", 0.0)),
            cv_fold_count=int(contract.get("temporal_residual_cv_fold_count", 0)),
        )
    elif class_path.endswith("BirdProfileResidualEvidenceFusionKnownnessHead"):
        baseline_contract = contract.get("baseline_contract")
        if not isinstance(baseline_contract, Mapping):
            raise ValueError("Bird evidence fusion head 缺少 baseline contract")
        baseline = TemporalKnownnessHead.from_contract(baseline_contract)
        model = BirdProfileResidualEvidenceFusionKnownnessHead(
            baseline,
            profile_dim=int(contract.get("profile_feature_dim", 44)),
            extra_dim=int(contract.get("extra_feature_dim", 32)),
            evidence_dim=int(contract.get("evidence_feature_dim", MICRO_DOPPLER_EVIDENCE_SUMMARY_DIM)),
        )
    elif class_path.endswith("BirdProfileResidualFusionKnownnessHead"):
        baseline_contract = contract.get("baseline_contract")
        if not isinstance(baseline_contract, Mapping):
            raise ValueError("Bird fusion head 缺少 baseline contract")
        baseline = TemporalKnownnessHead.from_contract(baseline_contract)
        model = BirdProfileResidualFusionKnownnessHead(
            baseline,
            profile_dim=int(contract.get("profile_feature_dim", 44)),
            extra_dim=int(contract.get("extra_feature_dim", 32)),
        )
    elif class_path.endswith("BirdProfileResidualKnownnessHead"):
        baseline_contract = contract.get("baseline_contract")
        if not isinstance(baseline_contract, Mapping):
            raise ValueError("Bird profile head 缺少 baseline contract")
        baseline = TemporalKnownnessHead.from_contract(baseline_contract)
        model = BirdProfileResidualKnownnessHead(
            baseline,
            profile_dim=int(contract.get("profile_feature_dim", 44)),
            extra_dim=int(contract.get("extra_feature_dim", 32)),
        )
    elif class_path.endswith("ProfileResidualKnownnessHead"):
        baseline_contract = contract.get("baseline_contract")
        if not isinstance(baseline_contract, Mapping):
            raise ValueError("profile head 缺少 baseline contract")
        baseline = TemporalKnownnessHead.from_contract(baseline_contract)
        model = ProfileResidualKnownnessHead(
            baseline,
            profile_dim=int(contract.get("profile_feature_dim", 44)),
        )
    elif class_path.endswith("TemporalKnownnessHead"):
        model = TemporalKnownnessHead.from_contract(contract)
    else:
        raise ValueError(f"knownness head class contract 不支持：{class_path}")
    model.load_state_dict(dict(checkpoint.state), strict=True)
    model.eval()
    return model, checkpoint


def score_frozen_endpoint_representation(
    representation: FrozenEndpointRepresentation,
    *,
    project_root: str | Path,
    path: str | Path,
    config_path: str | Path = "src/config/model.json",
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, str]:
    """在冻结表示上计算 parent-final route knownness。"""

    root = Path(project_root).resolve()
    config = load_model_config(root, config_path)
    model, checkpoint = load_l3_knownness_head(
        root,
        path,
        task=representation.task,
        expected_endpoint_sha256=representation.endpoint_sha256[representation.task],
        expected_config_sha256=sha256_file(config.path),
    )
    target_device = (
        next(model.parameters()).device
        if device is None
        else _device(device)
    )
    model = model.to(device=target_device)
    with torch.inference_mode():
        rep = representation.representation.to(device=target_device)
        padding = representation.padding_mask.to(device=target_device)
        if isinstance(model, ProfileResidualKnownnessHead):
            if representation.profile_representation is None:
                raise ValueError("profile head 缺少 profile representation")
            if isinstance(model, BirdProfileResidualEvidenceFusionKnownnessHead):
                if representation.bird_evidence_representation is None:
                    raise ValueError("V5 Bird head 缺少 micro-doppler evidence representation")
                output = model(
                    rep,
                    representation.profile_representation.to(device=target_device),
                    padding,
                    representation.bird_evidence_representation.to(device=target_device),
                )
            else:
                output = model(rep, representation.profile_representation.to(device=target_device), padding)
        elif isinstance(model, TemporalKnownnessHead):
            output = model(rep, padding)
        rows = output["route_probabilities"][
            torch.arange(len(representation.parent_ids), device=target_device),
            representation.lengths.to(device=target_device) - 1,
        ]
        if representation.task == "l2" and rows.shape[-1] == 1:
            rows = rows.expand(-1, len(L2_ROUTES))
    return rows.detach().cpu().numpy().astype(np.float64), sha256_file(checkpoint.path)


def score_frozen_endpoint_event_evidence(
    evidence: FrozenEndpointEventEvidence,
    *,
    project_root: str | Path,
    path: str | Path,
    config_path: str | Path = "src/config/model.json",
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, str]:
    """计算冻结 endpoint 每个 causal event 的 knownness route 分数。"""

    root = Path(project_root).resolve()
    config = load_model_config(root, config_path)
    model, checkpoint = load_l3_knownness_head(
        root,
        path,
        task=evidence.representation.task,
        expected_endpoint_sha256=evidence.representation.endpoint_sha256[evidence.representation.task],
        expected_config_sha256=sha256_file(config.path),
    )
    target_device = next(model.parameters()).device if device is None else _device(device)
    model = model.to(device=target_device)
    rep = evidence.representation
    with torch.inference_mode():
        features = rep.representation.to(device=target_device)
        padding = rep.padding_mask.to(device=target_device)
        if isinstance(model, ProfileResidualKnownnessHead):
            if rep.profile_representation is None:
                raise ValueError("profile event knownness scoring 缺少 profile representation")
            if isinstance(model, BirdProfileResidualEvidenceFusionKnownnessHead):
                if rep.bird_evidence_representation is None:
                    raise ValueError("V5 Bird head 缺少 micro-doppler evidence representation")
                output = model(
                    features,
                    rep.profile_representation.to(device=target_device),
                    padding,
                    rep.bird_evidence_representation.to(device=target_device),
                )
            else:
                output = model(
                    features,
                    rep.profile_representation.to(device=target_device),
                    padding,
                )
        elif isinstance(model, TemporalKnownnessHead):
            output = model(features, padding)
        else:
            raise TypeError("不支持的 knownness head 类型")
        scores = output["route_probabilities"]
        if evidence.representation.task == "l2" and scores.shape[-1] == 1:
            scores = scores.expand(-1, -1, len(L2_ROUTES))
    return scores.detach().cpu().numpy().astype(np.float64), sha256_file(checkpoint.path)


__all__ = [
    "FrozenEndpointRepresentation",
    "FrozenEndpointEventEvidence",
    "HEAD_LABELS",
    "L1_ROUTES",
    "L2_ROUTES",
    "build_head_fit_dataset",
    "build_head_validation_dataset",
    "build_micro_doppler_evidence_summary",
    "endpoint_representation_path",
    "load_frozen_endpoint_representation",
    "load_l3_knownness_head",
    "materialize_frozen_endpoint_representation",
    "save_frozen_endpoint_representation",
    "score_frozen_endpoint_representation",
    "score_frozen_endpoint_event_evidence",
    "train_l3_knownness_head",
]
