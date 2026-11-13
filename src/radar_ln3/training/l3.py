"""Frozen-endpoint L3 calibration from the registered validation view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_ln3.checkpoint import Checkpoint, canonical_hash, load_model_state, sha256_file
from radar_ln3.cache import CachedParentSequenceDataset
from radar_ln3.data import ParentSequenceDataset, load_manifest
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.derived import register_product_artifacts
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.data.paths import (
    relative_project_path,
    resolve_project_artifact_path,
    resolve_project_path,
)
from radar_ln3.device import resolve_device
from radar_ln3.features import (
    FeatureCache,
    build_model_batch,
    load_feature_cache_bundle,
    stable8_normalization_hash,
)
from radar_ln3.model import ModelConfig, build_l1_model, build_l2_model, load_model_config

from .common import (
    l1_model_inputs,
    l2_model_inputs,
    l2_reference_inputs,
    model_batch_kwargs,
)
from radar_ln3.decision.l3 import (
    L3CalibrationPolicy,
    audit_l3_calibration,
    fit_l3_calibration,
    save_l3_calibration,
)


DEFAULT_FEATURE_CACHE_MANIFEST = Path(
    "data/cq08/cache/features/feature-cache-manifest.json"
)
AUDIT_EVIDENCE_SCHEMA = "RADAR-LN3-AUDIT-EVIDENCE-V2"
REQUIRED_L3_FEATURES = frozenset(
    {
        "tr_observation",
        "tr_observation_increment",
        "course_circular_incoherence",
        "rd_quality",
        "rd_physical4",
        "micro_doppler_stable8",
        "micro_doppler_stable8_frame",
    }
)


def _endpoint_hashes(
    root: Path,
    config: ModelConfig,
    *,
    l1_checkpoint_path: str | Path | None,
    l2_checkpoint_path: str | Path | None,
    provided: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if provided is not None:
        return dict(provided)
    return {
        "l1": sha256_file(resolve_project_path(root, l1_checkpoint_path or config.artifacts.l1_output_path, must_exist=True)),
        "l2": sha256_file(resolve_project_path(root, l2_checkpoint_path or config.artifacts.l2_output_path, must_exist=True)),
    }


@dataclass(frozen=True)
class L3ValidationEvidence:
    parent_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    l1_logits: np.ndarray
    l2_logits: np.ndarray
    source_manifest_sha256: str
    endpoint_sha256: Mapping[str, str]
    feature_cache_manifest_sha256: str
    head_checkpoint_sha256: Mapping[str, str] | None = None
    role: str = "exploratory_validation"
    test_payload_access_count: int = 0
    manifest: DatasetManifest | None = field(default=None, repr=False, compare=False)
    config: ModelConfig | None = field(default=None, repr=False, compare=False)
    config_sha256: str | None = field(default=None, repr=False, compare=False)
    feature_caches: Mapping[str, FeatureCache] | None = field(default=None, repr=False, compare=False)
    dataset: CachedParentSequenceDataset | None = field(default=None, repr=False, compare=False)
    # 只有调用方显式要求复用表示时才保留冻结 representation。两份大型 tensor
    # 会从收集阶段一直存活到 head scoring 结束；默认不保留以维持最小驻留。
    representations: tuple[Any, Any] | None = field(default=None, repr=False, compare=False)


def _l2_mode_from_contract(contract: Mapping[str, Any]) -> str:
    mode = contract.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    if (
        contract.get("baseline_source") == "event_baseline_logits_anchor"
        and contract.get("output_equation")
        == "baseline_logits + (current_logits - frozen_reference_logits) + stable8_residual"
    ):
        return "residual"
    raise ValueError("L3 L2 checkpoint contract 缺少可验证的 mode")


def _load_endpoint_models(
    project_root: Path,
    *,
    device: torch.device,
    config: Any,
    source_manifest_sha256: str | None,
    config_sha256: str | None = None,
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, Checkpoint, Checkpoint]:
    l1_checkpoint = Checkpoint.load(
        project_root,
        l1_checkpoint_path or config.artifacts.l1_output_path,
        expected_task="l1",
    )
    l2_checkpoint = Checkpoint.load(
        project_root,
        l2_checkpoint_path or config.artifacts.l2_output_path,
        expected_task="l2",
    )
    if config_sha256 is not None and l1_config_path is None and l2_config_path is None:
        expected_config_hashes = {"l1": str(config_sha256), "l2": str(config_sha256)}
    else:
        l1_config_target = l1_config_path or config.path
        l2_config_target = l2_config_path or config.path
        l1_hash = sha256_file(l1_config_target)
        l2_hash = l1_hash if l2_config_target == l1_config_target else sha256_file(l2_config_target)
        expected_config_hashes = {"l1": l1_hash, "l2": l2_hash}
    for label, checkpoint in (("l1", l1_checkpoint), ("l2", l2_checkpoint)):
        trained_config_hash = checkpoint.payload.get("config_sha256")
        if trained_config_hash != expected_config_hashes[label]:
            raise ValueError(
                f"L3 {label} checkpoint config hash 与绑定配置不一致："
                f"checkpoint={trained_config_hash or '<missing>'}, "
                f"expected={expected_config_hashes[label]}。"
                "请传入该 checkpoint 对应的训练配置，或在当前配置下重新训练。"
            )
        if (
            source_manifest_sha256 is not None
            and checkpoint.payload.get("manifest_sha256") != source_manifest_sha256
        ):
            raise ValueError(f"L3 {label} checkpoint data manifest hash 不一致")
        if checkpoint.payload.get("task") != label:
            raise ValueError(f"L3 {label} checkpoint task 不一致")
        contract = checkpoint.payload.get("model_contract")
        if (
            not isinstance(contract, Mapping)
            or checkpoint.payload.get("model_contract_sha256")
            != canonical_hash(contract)
        ):
            raise ValueError(f"L3 {label} checkpoint model contract hash 不一致")

    l1_contract = l1_checkpoint.payload.get("model_contract")
    if not isinstance(l1_contract, Mapping):
        raise ValueError("L3 L1 checkpoint 缺少 model contract")
    l1 = build_l1_model(
        config,
        arm=str(l1_contract.get("arm", "")),
        seed=int(l1_contract.get("seed", config.training.seed)),
        device=device,
    )
    if canonical_hash(l1_contract) != canonical_hash(l1.model_contract()):
        raise ValueError("L3 L1 checkpoint model contract 与当前运行时不匹配")
    load_model_state(l1, l1_checkpoint.state)
    l1.eval()

    l2_contract = l2_checkpoint.payload.get("model_contract")
    if not isinstance(l2_contract, Mapping):
        raise ValueError("L3 L2 checkpoint 缺少 model contract")
    base = Checkpoint.load(project_root, config.artifacts.l2_base_weight_path)
    stable = Checkpoint.load(project_root, config.artifacts.l2_stable_weight_path)
    l2_mode = _l2_mode_from_contract(l2_contract)
    l2 = build_l2_model(
        config,
        arm=str(l2_contract.get("arm", "")),
        mode=l2_mode,
        base_state=base.state,
        stable_state=stable.state,
        seed=config.training.seed,
    ).to(device)
    compatible_l2_contract = dict(l2_contract)
    compatible_l2_contract.setdefault("mode", l2_mode)
    if canonical_hash(compatible_l2_contract) != canonical_hash(l2.model_contract()):
        raise ValueError("L3 L2 checkpoint model contract 与当前运行时不匹配")
    load_model_state(l2, l2_checkpoint.state)
    l2.eval()
    return l1, l2, l1_checkpoint, l2_checkpoint


def _validate_feature_cache_coverage(
    caches: Mapping[str, FeatureCache],
    *,
    manifest: DatasetManifest,
    dataset: ParentSequenceDataset,
) -> None:
    if not caches:
        raise ValueError("L3 缺少 feature cache")
    missing_features = REQUIRED_L3_FEATURES - set(caches)
    if missing_features:
        raise ValueError(f"L3 feature cache 缺少正式输入：{sorted(missing_features)}")
    records = {str(row["sample_id"]): row for row in manifest.tr_records}
    expected_ids = set(dataset.view.sample_ids)
    for name, cache in caches.items():
        positions = {sample_id: index for index, sample_id in enumerate(cache.sample_ids)}
        if len(positions) != len(cache.sample_ids):
            raise ValueError(f"L3 feature cache sample_id 重复：{name}")
        missing = expected_ids - set(positions)
        if missing:
            raise ValueError(f"L3 feature cache 缺少 validation 事件：{name}:{sorted(missing)[:3]}")
        for sample_id in expected_ids:
            record = records.get(sample_id)
            index = positions[sample_id]
            if record is None or (
                cache.parent_sample_ids[index] != str(record["parent_sample_id"])
                or int(cache.causal_event_indices[index]) != int(record["causal_event_index"])
                or cache.split_names[index] != str(record["split_name"])
            ):
                raise ValueError(f"L3 feature cache event identity 不一致：{name}:{sample_id}")


def _collect_l3_evidence(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_cache_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    split_name: str,
    source_role: str,
    training_view: bool,
    expected_source_manifest_sha256: str | None,
    require_training_manifest_match: bool,
    manifest: DatasetManifest | None = None,
    config: ModelConfig | None = None,
    manifest_sha256: str | None = None,
    config_sha256: str | None = None,
    feature_caches: Mapping[str, FeatureCache] | None = None,
    feature_cache_sha256: str | None = None,
    endpoint_sha256: Mapping[str, str] | None = None,
    validate_event_cache: bool = True,
    device: str = "cpu",
    batch_size: int | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    retain_representations: bool = False,
) -> L3ValidationEvidence:
    """读取一个已注册 L3 role 的冻结 endpoint evidence。

    ``retain_representations=True`` 时 evidence 会持有两份冻结表示，供后续
    knownness head scoring 直接复用，避免重复 torch.load 与 tensor hash 校验。
    """

    root = Path(project_root).resolve()
    config = config or load_model_config(root, model_config_path)
    current_config_hash = str(config_sha256 or sha256_file(config.path))
    if batch_size is not None and (
        isinstance(batch_size, bool) or int(batch_size) <= 0
    ):
        raise ValueError("L3 batch_size 必须是正整数")
    target_device = resolve_device(device)
    manifest = manifest or load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    manifest_hash = str(manifest_sha256 or data_manifest_hash(manifest))
    if (
        expected_source_manifest_sha256 is not None
        and manifest_hash == str(expected_source_manifest_sha256).lower()
    ):
        raise ValueError("L3 post-selection audit 必须使用新的 data manifest")
    if representation_root is not None:
        from radar_ln3.training.l3_heads import (
            endpoint_representation_path,
            load_frozen_endpoint_representation,
            materialize_frozen_endpoint_representation,
            save_frozen_endpoint_representation,
        )

        cache_manifest_path = resolve_project_path(
            root, feature_cache_manifest_path, must_exist=True
        )
        cache_hash = str(feature_cache_sha256 or sha256_file(cache_manifest_path))
        representation_caches = feature_caches
        endpoint_hashes = _endpoint_hashes(
            root, config, l1_checkpoint_path=l1_checkpoint_path,
            l2_checkpoint_path=l2_checkpoint_path, provided=endpoint_sha256,
        )
        endpoint_models = None
        view_dataset = CachedParentSequenceDataset(
            root,
            manifest_path=manifest_path,
            manifest=manifest,
            split_name=split_name,
            include_unknown=True,
            training_view=bool(training_view),
            task_name=("l3_audit" if source_role == "post_selection_audit" else "l3_validation"),
            allow_unknown_head_training=False,
            validate_event_cache=bool(validate_event_cache),
        )
        expected_view_hash = view_dataset.view.view_hash

        def load_or_materialize(task: str, allow_unknown: bool):
            nonlocal representation_caches, endpoint_models
            target = endpoint_representation_path(
                root,
                task=task,
                split_name=split_name,
                representation_root=representation_root,
            )
            if target.exists():
                try:
                    return load_frozen_endpoint_representation(
                        root,
                        target,
                        expected_task=task,
                        expected_source_manifest_sha256=manifest_hash,
                        expected_feature_cache_manifest_sha256=cache_hash,
                        expected_view_hash=expected_view_hash,
                        expected_endpoint_sha256=endpoint_hashes,
                        expected_config_sha256=current_config_hash,
                    )
                except (OSError, RuntimeError, ValueError):
                    pass
            if representation_caches is None:
                representation_caches = load_feature_cache_bundle(
                    root,
                    cache_manifest_path,
                    expected_view=view_dataset.view,
                    expected_source_manifest_hash=manifest_hash,
                    verify_integrity=True,
                )
            if endpoint_models is None:
                endpoint_models = _load_endpoint_models(
                    root,
                    device=target_device,
                    config=config,
                    source_manifest_sha256=(manifest_hash if require_training_manifest_match else None),
                    config_sha256=current_config_hash,
                    l1_checkpoint_path=l1_checkpoint_path,
                    l2_checkpoint_path=l2_checkpoint_path,
                    l1_config_path=l1_config_path,
                    l2_config_path=l2_config_path,
                )
            representation = materialize_frozen_endpoint_representation(
                root,
                task=task,
                manifest_path=manifest_path,
                feature_cache_manifest_path=feature_cache_manifest_path,
                split_name=split_name,
                training_view=training_view,
                allow_unknown_head_training=allow_unknown,
                device=device,
                batch_size=int(batch_size or config.training.batch_size),
                # The supplied dataset was constructed above and has already
                # performed the event-cache check when requested.
                validate_event_cache=False,
                verify_files=bool(verify_files),
                require_training_manifest_match=bool(require_training_manifest_match),
                model_config_path=model_config_path,
                l1_checkpoint_path=l1_checkpoint_path,
                l2_checkpoint_path=l2_checkpoint_path,
                l1_config_path=l1_config_path,
                l2_config_path=l2_config_path,
                manifest=manifest,
                config=config,
                endpoint_sha256=endpoint_hashes,
                feature_caches=representation_caches,
                feature_cache_sha256=cache_hash,
                manifest_sha256=manifest_hash,
                dataset=view_dataset,
                endpoint_models=endpoint_models,
            )
            save_frozen_endpoint_representation(
                root,
                representation,
                config=config,
                config_sha256=current_config_hash,
                output_path=target,
                overwrite=True,
            )
            return representation

        # validation 和 audit 都不是 knownness head fit view。Unknown 只能
        # 在显式的 l3_head_fit/train view 中开启，不能把 training_view=True
        # 直接当成 allow_unknown_head_training。
        l1_representation = load_or_materialize("l1", False)
        l2_representation = load_or_materialize("l2", False)
        if (
            l1_representation.parent_ids != l2_representation.parent_ids
            or l1_representation.label_names != l2_representation.label_names
        ):
            raise ValueError("L3 cached endpoint representation parent 顺序不一致")
        if dict(l1_representation.endpoint_sha256) != dict(l2_representation.endpoint_sha256):
            raise ValueError("L3 cached endpoint representation checkpoint binding 不一致")
        return L3ValidationEvidence(
            parent_ids=l1_representation.parent_ids,
            label_names=l1_representation.label_names,
            l1_logits=l1_representation.endpoint_logits.numpy(),
            l2_logits=l2_representation.endpoint_logits.numpy(),
            source_manifest_sha256=manifest_hash,
            endpoint_sha256=dict(l1_representation.endpoint_sha256),
            feature_cache_manifest_sha256=cache_hash,
            role=source_role,
            test_payload_access_count=max(
                l1_representation.test_payload_access_count,
                l2_representation.test_payload_access_count,
            ),
            manifest=manifest,
            config=config,
            config_sha256=current_config_hash,
            feature_caches=representation_caches,
            dataset=view_dataset,
            representations=(
                (l1_representation, l2_representation)
                if retain_representations
                else None
            ),
        )
    dataset = CachedParentSequenceDataset(
        root,
        manifest=manifest,
        split_name=split_name,
        include_unknown=True,
        training_view=bool(training_view),
        task_name=("l3_audit" if source_role == "post_selection_audit" else "l3_validation"),
        validate_event_cache=bool(validate_event_cache),
    )
    cache_manifest_path = resolve_project_path(
        root,
        feature_cache_manifest_path,
        must_exist=True,
    )
    caches = feature_caches or load_feature_cache_bundle(
        root,
        cache_manifest_path,
        expected_view=dataset.view,
        expected_source_manifest_hash=manifest_hash,
        verify_integrity=True,
    )
    endpoint_hashes = _endpoint_hashes(
        root, config, l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path, provided=endpoint_sha256,
    )
    _validate_feature_cache_coverage(caches, manifest=manifest, dataset=dataset)
    expected_normalization_hash = stable8_normalization_hash(
        config.data.l2.stable8_median,
        config.data.l2.stable8_scale,
    )
    for feature_name in ("micro_doppler_stable8", "micro_doppler_stable8_frame"):
        actual_normalization_hash = caches[feature_name].descriptor.normalization_hash
        if actual_normalization_hash != expected_normalization_hash:
            raise ValueError(
                "L3 feature cache stable8 normalization hash 与当前模型配置不一致："
                f"feature={feature_name}, "
                f"cache={actual_normalization_hash or '<missing>'}, "
                f"current={expected_normalization_hash}。"
                "请先使用 build_l3_cache.py --overwrite 重建对应 role cache。"
            )
    l1, l2, l1_checkpoint, l2_checkpoint = _load_endpoint_models(
        root,
        device=target_device,
        config=config,
        config_sha256=current_config_hash,
        source_manifest_sha256=(
            manifest_hash if require_training_manifest_match else None
        ),
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config.training.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: items,
    )
    parent_ids: list[str] = []
    label_names: list[str] = []
    l1_logits_rows: list[np.ndarray] = []
    l2_logits_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for sequences in loader:
            l1_batch = build_model_batch(
                sequences,
                **model_batch_kwargs(config, "l1", allow_unmapped_labels=True),
                feature_caches=caches,
            )
            l2_batch = build_model_batch(
                sequences,
                **model_batch_kwargs(
                    config,
                    "l2",
                    stable_input_mode=(
                        "raw" if l2.mode in {"direct", "direct_grouped"} else "normalized"
                    ),
                    allow_unmapped_labels=True,
                ),
                feature_caches=caches,
            )
            l1_output = l1(
                base_inputs=l1_model_inputs(l1_batch, target_device),
                observation_increment_values=l1_batch["observation_increment_values"].to(
                    device=target_device
                ),
            )
            l2_inputs = l2_model_inputs(l2_batch, target_device)
            if l2.mode in {"direct", "direct_grouped"}:
                l2_output = l2.direct_logits(base_inputs=l2_inputs)
            else:
                reference_inputs = l2_reference_inputs(
                    l2_batch,
                    target_device,
                    base_inputs=l2_inputs,
                )
                baseline_logits = l2.frozen_reference_logits(
                    base_inputs=reference_inputs
                ).detach().clone().masked_fill(
                    l2_inputs["tr_padding_mask"].unsqueeze(-1), 0.0
                )
                l2_output = l2(
                    base_inputs=l2_inputs,
                    reference_base_inputs=reference_inputs,
                    baseline_logits=baseline_logits,
                    stable_summary_tokens=l2_batch["stable_summary_tokens"].to(device=target_device),
                    stable_summary_valid=l2_batch["stable_summary_valid"].to(device=target_device),
                    stable_frame_values=l2_batch["stable_frame_values"].to(device=target_device),
                    stable_frame_mask=l2_batch["stable_frame_mask"].to(device=target_device),
                    motion_values=l2_batch["motion_values"].to(device=target_device),
                )
            l1_logits_rows.append(l1_output["final_logits"].detach().cpu().numpy())
            l2_length = (~l2_inputs["tr_padding_mask"]).long().sum(dim=1)
            l2_row = torch.arange(l2_length.shape[0], device=target_device)
            l2_logits_rows.append(
                l2_output["sequence_logits"][l2_row, l2_length - 1]
                .detach()
                .cpu()
                .numpy()
            )
            parent_ids.extend(str(value) for value in l1_batch["parent_sample_ids"])
            label_names.extend(str(value) for value in l1_batch["label_names"])
    cache_manifest_sha256 = str(feature_cache_sha256 or sha256_file(cache_manifest_path))
    return L3ValidationEvidence(
        parent_ids=tuple(parent_ids),
        label_names=tuple(label_names),
        l1_logits=np.concatenate(l1_logits_rows, axis=0),
        l2_logits=np.concatenate(l2_logits_rows, axis=0),
        source_manifest_sha256=manifest_hash,
        endpoint_sha256=endpoint_hashes,
        feature_cache_manifest_sha256=cache_manifest_sha256,
        role=source_role,
        test_payload_access_count=(
            len(dataset) if source_role == "post_selection_audit" and split_name == "test" else 0
        ),
        manifest=manifest,
        config=config,
        config_sha256=current_config_hash,
        feature_caches=caches,
        dataset=dataset,
    )


def collect_l3_validation_evidence(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_cache_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    device: str = "cpu",
    batch_size: int | None = None,
    verify_files: bool = True,
    validate_event_cache: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    config: ModelConfig | None = None,
    retain_representations: bool = False,
) -> L3ValidationEvidence:
    """收集完整 validation labels；结果仅供 exploratory 校准。

    ``retain_representations=True`` 供调用方随后直接复用冻结表示做 head scoring。
    """

    return _collect_l3_evidence(
        project_root,
        manifest_path=manifest_path,
        feature_cache_manifest_path=feature_cache_manifest_path,
        split_name="validation",
        source_role="exploratory_validation",
        training_view=True,
        expected_source_manifest_sha256=None,
        require_training_manifest_match=True,
        validate_event_cache=bool(validate_event_cache),
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
        config=config,
        retain_representations=bool(retain_representations),
    )


def collect_l3_audit_evidence(
    project_root: str | Path,
    calibration: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    feature_cache_manifest_path: str | Path,
    thresholds_frozen_before_access: bool,
    split_name: str = "test",
    device: str = "cpu",
    batch_size: int | None = None,
    verify_files: bool = True,
    validate_event_cache: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    config: ModelConfig | None = None,
    retain_representations: bool = False,
) -> L3ValidationEvidence:
    """读取已由入口加载器校验过的独立审计 calibration。

    ``retain_representations=True`` 供调用方随后直接复用冻结表示做 head scoring。
    """

    if thresholds_frozen_before_access is not True:
        raise ValueError("L3 audit 前必须冻结阈值")
    if not str(manifest_path).strip() or not str(feature_cache_manifest_path).strip():
        raise ValueError("L3 audit manifest/cache path 不能为空")
    return _collect_l3_evidence(
        project_root,
        manifest_path=manifest_path,
        feature_cache_manifest_path=feature_cache_manifest_path,
        split_name=split_name,
        source_role="post_selection_audit",
        training_view=False,
        expected_source_manifest_sha256=str(calibration["source_manifest_sha256"]),
        require_training_manifest_match=False,
        validate_event_cache=bool(validate_event_cache),
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
        config=config,
        retain_representations=bool(retain_representations),
    )


def save_and_register_l3_calibration(
    project_root: str | Path,
    calibration: Mapping[str, Any],
    *,
    output_path: str | Path = "artifacts/radar_ln3/l3/radar_ln3_calibration.json",
    manifest_path: str | Path = "src/config/manifests.json",
    overwrite: bool = False,
    return_sha256: bool = False,
    manifest: DatasetManifest | None = None,
    register: bool = True,
) -> Path | tuple[Path, str]:
    """保存 sealed calibration，并按需登记到唯一产品总 manifest。"""

    root = Path(project_root).resolve()
    saved_path, saved_file_sha256 = save_l3_calibration(
        root,
        output_path,
        calibration,
        overwrite=overwrite,
        return_sha256=True,
    )
    if register:
        audit = calibration.get("audit")
        register_product_artifacts(
            root,
            manifest_path=manifest_path,
            overwrite=True,
            manifest=manifest,
            l3_calibration_descriptor={
                "path": relative_project_path(root, saved_path),
                "kind": "l3_calibration",
                "file_size": int(saved_path.stat().st_size),
                "sha256": saved_file_sha256,
                "status": str(calibration["status"]),
                "deployment_eligible": bool(calibration["deployment_eligible"]),
                "artifact_sha256": str(calibration["artifact_sha256"]),
                "source_role": str(calibration["source_role"]),
                "audit_role": None if not isinstance(audit, Mapping) else audit.get("role"),
                "audit_source_manifest_sha256": (
                    None
                    if not isinstance(audit, Mapping)
                    else audit.get("source_manifest_sha256")
                ),
            },
        )
    return (saved_path, saved_file_sha256) if return_sha256 else saved_path


def _collect_l3_head_scores(
    project_root: str | Path,
    *,
    manifest_path: str | Path,
    feature_cache_manifest_path: str | Path,
    split_name: str,
    training_view: bool,
    l1_head_path: str | Path,
    l2_head_path: str | Path,
    device: str,
    batch_size: int | None,
    require_training_manifest_match: bool,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    manifest: DatasetManifest | None = None,
    config: ModelConfig | None = None,
    config_sha256: str | None = None,
    feature_caches: Mapping[str, FeatureCache] | None = None,
    feature_cache_sha256: str | None = None,
    dataset: CachedParentSequenceDataset | None = None,
    endpoint_sha256: Mapping[str, str] | None = None,
    validate_event_cache: bool = True,
    source_manifest_sha256: str | None = None,
    representations: tuple[Any, Any] | None = None,
    l1_head_checkpoint: Checkpoint | None = None,
    l2_head_checkpoint: Checkpoint | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, str], int]:
    from .l3_heads import (
        endpoint_representation_path,
        load_frozen_endpoint_representation,
        materialize_frozen_endpoint_representation,
        load_l3_knownness_head,
        score_frozen_endpoint_representation,
    )

    root = Path(project_root).resolve()
    manifest = manifest or load_manifest(
        root / manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    source_hash = str(source_manifest_sha256 or data_manifest_hash(manifest))
    config = config or load_model_config(root, model_config_path)
    config_hash = str(config_sha256 or sha256_file(config.path))
    endpoint_hashes = _endpoint_hashes(
        root, config, l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path, provided=endpoint_sha256,
    )

    common = {
        "project_root": project_root,
        "manifest_path": manifest_path,
        "feature_cache_manifest_path": feature_cache_manifest_path,
        "split_name": split_name,
        "training_view": training_view,
        "device": device,
        "batch_size": int(batch_size or 4),
        "validate_event_cache": bool(validate_event_cache),
        "require_training_manifest_match": require_training_manifest_match,
        "verify_files": bool(verify_files),
        "model_config_path": model_config_path,
        "l1_checkpoint_path": l1_checkpoint_path,
        "l2_checkpoint_path": l2_checkpoint_path,
        "l1_config_path": l1_config_path,
        "l2_config_path": l2_config_path,
        "manifest": manifest,
        "config": config,
        "endpoint_sha256": endpoint_hashes,
        "feature_caches": feature_caches,
        "feature_cache_sha256": feature_cache_sha256,
        "manifest_sha256": source_hash,
        "dataset": dataset,
        "endpoint_models": None,
    }
    view_dataset = dataset
    if view_dataset is None:
        view_dataset = CachedParentSequenceDataset(
            root,
            manifest_path=manifest_path,
            manifest=manifest,
            split_name=split_name,
            include_unknown=True,
            training_view=bool(training_view),
            task_name=("l3_audit" if split_name == "test" and not training_view else "l3_validation"),
            allow_unknown_head_training=False,
            validate_event_cache=bool(validate_event_cache),
        )
    elif validate_event_cache and not view_dataset.event_cache_validated:
        view_dataset.validate_event_cache()
    expected_view_hash = view_dataset.view.view_hash
    common["dataset"] = view_dataset
    l1_rep = None
    l2_rep = None
    cache_hash = str(feature_cache_sha256 or "")
    if representations is not None:
        l1_rep, l2_rep = representations
    if representation_root is not None and l1_rep is None:
        cache_hash = cache_hash or sha256_file(
            resolve_project_path(root, feature_cache_manifest_path, must_exist=True)
        )
        try:
            l1_rep = load_frozen_endpoint_representation(
                root,
                endpoint_representation_path(root, task="l1", split_name=split_name, representation_root=representation_root),
                expected_task="l1",
                expected_source_manifest_sha256=source_hash,
                expected_feature_cache_manifest_sha256=cache_hash,
                expected_view_hash=expected_view_hash,
                expected_endpoint_sha256=endpoint_hashes,
                expected_config_sha256=config_hash,
            )
            l2_rep = load_frozen_endpoint_representation(
                root,
                endpoint_representation_path(root, task="l2", split_name=split_name, representation_root=representation_root),
                expected_task="l2",
                expected_source_manifest_sha256=source_hash,
                expected_feature_cache_manifest_sha256=cache_hash,
                expected_view_hash=expected_view_hash,
                expected_endpoint_sha256=endpoint_hashes,
                expected_config_sha256=config_hash,
            )
        except (OSError, RuntimeError, ValueError):
            l1_rep = None
            l2_rep = None
    if l1_rep is None or l2_rep is None:
        if common["feature_caches"] is None:
            cache_path = resolve_project_path(
                root, feature_cache_manifest_path, must_exist=True
            )
            common["feature_caches"] = load_feature_cache_bundle(
                root,
                cache_path,
                expected_view=view_dataset.view,
                expected_source_manifest_hash=source_hash,
                verify_integrity=True,
            )
            common["feature_cache_sha256"] = cache_hash or sha256_file(cache_path)
        if common.get("endpoint_models") is None:
            common["endpoint_models"] = _load_endpoint_models(
                root,
                device=resolve_device(device),
                config=config,
                config_sha256=config_hash,
                source_manifest_sha256=(source_hash if require_training_manifest_match else None),
                l1_checkpoint_path=l1_checkpoint_path,
                l2_checkpoint_path=l2_checkpoint_path,
                l1_config_path=l1_config_path,
                l2_config_path=l2_config_path,
            )
        l1_rep = materialize_frozen_endpoint_representation(task="l1", **common)
        l2_rep = materialize_frozen_endpoint_representation(task="l2", **common)
    if l1_rep.parent_ids != l2_rep.parent_ids or l1_rep.label_names != l2_rep.label_names:
        raise ValueError("L3 knownness head representation L1/L2 parent 顺序不一致")
    l1_model, l1_checkpoint = load_l3_knownness_head(
        root,
        l1_head_path,
        task="l1",
        expected_endpoint_sha256=endpoint_hashes["l1"],
        expected_config_sha256=config_hash,
        _checkpoint=l1_head_checkpoint,
    )
    l2_model, l2_checkpoint = load_l3_knownness_head(
        root,
        l2_head_path,
        task="l2",
        expected_endpoint_sha256=endpoint_hashes["l2"],
        expected_config_sha256=config_hash,
        _checkpoint=l2_head_checkpoint,
    )
    l1_scores, l1_hash = score_frozen_endpoint_representation(
        l1_rep, project_root=project_root, path=l1_head_path,
        config_path=model_config_path, device=device, config=config,
        config_sha256=config_hash,
        checkpoint=l1_checkpoint,
        model=l1_model,
    )
    l2_scores, l2_hash = score_frozen_endpoint_representation(
        l2_rep, project_root=project_root, path=l2_head_path,
        config_path=model_config_path, device=device, config=config,
        config_sha256=config_hash,
        checkpoint=l2_checkpoint,
        model=l2_model,
    )
    return (
        l1_rep.parent_ids,
        l1_scores,
        l2_scores,
        {"l1": l1_hash, "l2": l2_hash},
        max(l1_rep.test_payload_access_count, l2_rep.test_payload_access_count),
    )


def fit_l3_validation_calibration(
    project_root: str | Path,
    *,
    policy: L3CalibrationPolicy,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_cache_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    device: str = "cpu",
    batch_size: int | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    l1_head_path: str | Path | None = None,
    l2_head_path: str | Path | None = None,
    l1_head_checkpoint: Checkpoint | None = None,
    l2_head_checkpoint: Checkpoint | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    validate_event_cache: bool = True,
    l1_router_from_head: bool = False,
    l2_knownness_threshold_mode: str = "shared",
    decision_profile: Mapping[str, Any] | None = None,
    config: ModelConfig | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """拟合 validation 诊断 artifact；该数据角色不会获得部署资格。"""

    evidence = collect_l3_validation_evidence(
        project_root,
        manifest_path=manifest_path,
        feature_cache_manifest_path=feature_cache_manifest_path,
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        validate_event_cache=validate_event_cache,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
        config=config,
        retain_representations=l1_head_path is not None and l2_head_path is not None,
    )
    head_scores = None
    if (l1_head_path is None) != (l2_head_path is None):
        raise ValueError("L3 L1/L2 knownness head path 必须成对提供")
    if (l1_head_checkpoint is None) != (l2_head_checkpoint is None):
        raise ValueError("L3 L1/L2 knownness head checkpoint 必须成对提供")
    if l1_head_path is not None and l2_head_path is not None:
        head_ids, l1_scores, l2_scores, head_hashes, head_test_access = _collect_l3_head_scores(
            project_root,
            manifest_path=manifest_path,
            feature_cache_manifest_path=feature_cache_manifest_path,
            split_name="validation",
            training_view=True,
            l1_head_path=l1_head_path,
            l2_head_path=l2_head_path,
            device=device,
            batch_size=batch_size,
            require_training_manifest_match=True,
            verify_files=verify_files,
            validate_event_cache=validate_event_cache,
            representation_root=representation_root,
            model_config_path=model_config_path,
            l1_checkpoint_path=l1_checkpoint_path,
            l2_checkpoint_path=l2_checkpoint_path,
            l1_config_path=l1_config_path,
            l2_config_path=l2_config_path,
            manifest=evidence.manifest,
            config=evidence.config,
            config_sha256=evidence.config_sha256,
            feature_caches=evidence.feature_caches,
            feature_cache_sha256=evidence.feature_cache_manifest_sha256,
            dataset=evidence.dataset,
            endpoint_sha256=evidence.endpoint_sha256,
            source_manifest_sha256=evidence.source_manifest_sha256,
            representations=evidence.representations,
            l1_head_checkpoint=l1_head_checkpoint,
            l2_head_checkpoint=l2_head_checkpoint,
        )
        if head_ids != evidence.parent_ids:
            raise ValueError("L3 knownness head validation parent binding 与 endpoint evidence 不一致")
        head_scores = (l1_scores, l2_scores, head_hashes)
        if head_test_access:
            raise ValueError("L3 validation knownness head 读取了 test payload")
    artifact = fit_l3_calibration(
        parent_ids=evidence.parent_ids,
        labels=evidence.label_names,
        l1_logits=evidence.l1_logits,
        l2_logits=evidence.l2_logits,
        source_manifest_sha256=evidence.source_manifest_sha256,
        endpoint_sha256=evidence.endpoint_sha256,
        source_role=evidence.role,
        policy=policy,
        l1_knownness_scores=None if head_scores is None else head_scores[0],
        l2_knownness_scores_by_route=None if head_scores is None else head_scores[1],
        head_checkpoint_sha256=None if head_scores is None else head_scores[2],
        l1_router_from_head=bool(l1_router_from_head),
        l2_knownness_threshold_mode=str(l2_knownness_threshold_mode),
    )
    if decision_profile is not None:
        profile = dict(decision_profile)
        if profile.get("schema_version") not in {"RADAR-LN3-DECISION-PROFILE", "RADAR-LN3-L3-DECISION-PROFILE-V1"}:
            raise ValueError("L3 decision profile schema 不支持")
        for name in (
            "l1_target_conformal_floor_override",
            "l1_target_knownness_override",
            "bird_recovery_gate",
            "l1_clutter_bird_recovery_gate",
            "l1_target_bird_recovery_gate",
        ):
            if name in profile:
                artifact[name] = profile[name]
        artifact["decision_profile"] = profile
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    body["feature_cache_manifest_sha256"] = evidence.feature_cache_manifest_sha256
    artifact = {**body, "artifact_sha256": canonical_hash(body)}
    if output_path is not None:
        save_and_register_l3_calibration(
            project_root,
            artifact,
            output_path=output_path,
            manifest_path=manifest_path,
            overwrite=overwrite,
            manifest=evidence.manifest,
            register=register,
        )
    return artifact


def _audit_evidence_rows(
    evidence: L3ValidationEvidence,
    head_scores: tuple[np.ndarray, np.ndarray] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, parent_id in enumerate(evidence.parent_ids):
        result.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": str(evidence.label_names[index]),
                "l1_logits": np.asarray(
                    evidence.l1_logits[index], dtype=np.float64
                ).tolist(),
                "l2_logits": np.asarray(
                    evidence.l2_logits[index], dtype=np.float64
                ).tolist(),
                "l1_knownness_scores": (
                    None
                    if head_scores is None
                    else np.asarray(head_scores[0][index], dtype=np.float64).tolist()
                ),
                "l2_knownness_scores": (
                    None
                    if head_scores is None
                    else np.asarray(head_scores[1][index], dtype=np.float64).tolist()
                ),
            }
        )
    return result


def _audit_evidence_binding(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """构造审计证据与 calibration 共享的数据绑定。

    这里只绑定数据身份，不绑定发布状态或阈值派生的 artifact hash，因此重新
    封存 calibration 后同一份 audit evidence 仍可复用。
    """

    audit = calibration.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("calibration 缺少 audit 段，无法绑定 audit evidence")
    endpoint = calibration.get("endpoint_sha256")
    if not isinstance(endpoint, Mapping) or not endpoint:
        raise ValueError("calibration 缺少 endpoint_sha256，无法绑定 audit evidence")
    knownness_source = calibration.get("knownness_source")
    head_binding: dict[str, str] | None = None
    if knownness_source == "independent_l3_head":
        heads = calibration.get("head_checkpoint_sha256")
        if not isinstance(heads, Mapping) or set(heads) != {"l1", "l2"}:
            raise ValueError(
                "calibration 缺少 head_checkpoint_sha256，无法绑定 audit evidence"
            )
        head_binding = {str(key): str(value) for key, value in heads.items()}
    binding = {
        "source_manifest_sha256": audit.get("source_manifest_sha256"),
        "endpoint_sha256": dict(endpoint),
        "feature_cache_manifest_sha256": calibration.get(
            "audit_feature_cache_manifest_sha256"
        ),
        "knownness_source": knownness_source,
        "head_checkpoint_sha256": head_binding,
        "parent_ids_sha256": audit.get("parent_ids_sha256"),
    }
    for name in (
        "source_manifest_sha256",
        "feature_cache_manifest_sha256",
        "parent_ids_sha256",
    ):
        value = binding[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"calibration 缺少有效的 {name}，无法绑定 audit evidence")
    return binding


def _write_audit_evidence(
    project_root: str | Path,
    path: str | Path,
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_artifact_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(
        (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for item in (header, *rows)
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _load_audit_evidence(
    project_root: str | Path,
    path: str | Path,
) -> tuple[
    dict[str, Any],
    L3ValidationEvidence,
    tuple[np.ndarray, np.ndarray] | None,
]:
    root = Path(project_root).resolve()
    target = resolve_project_artifact_path(root, path, must_exist=True)
    try:
        lines = [
            line
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"L3 audit evidence 无法读取：{target}") from exc
    if not lines:
        raise ValueError(f"L3 audit evidence 为空：{target}")
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as exc:
        raise ValueError(f"L3 audit evidence 格式无效：{target}") from exc
    if not isinstance(header, dict) or header.get("schema_version") != AUDIT_EVIDENCE_SCHEMA:
        raise ValueError("L3 audit evidence schema 不支持")
    if len(rows) != int(header.get("record_count", -1)):
        raise ValueError("L3 audit evidence record_count 不一致")
    binding = header.get("data_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("L3 audit evidence 缺少 data_binding")
    missing = sorted(
        {
            "source_manifest_sha256",
            "endpoint_sha256",
            "feature_cache_manifest_sha256",
            "knownness_source",
            "head_checkpoint_sha256",
            "parent_ids_sha256",
        }
        - set(binding)
    )
    if missing:
        raise ValueError(f"L3 audit evidence data_binding 缺少字段：{missing}")
    if not isinstance(binding["endpoint_sha256"], Mapping):
        raise ValueError("L3 audit evidence data_binding endpoint_sha256 必须是对象")
    parent_ids = tuple(str(row["parent_sample_id"]) for row in rows)
    label_names = tuple(str(row["label_name"]) for row in rows)
    l1_logits = np.asarray([row["l1_logits"] for row in rows], dtype=np.float64)
    l2_logits = np.asarray([row["l2_logits"] for row in rows], dtype=np.float64)
    if l1_logits.shape != (len(rows), 2) or l2_logits.shape != (len(rows), 3):
        raise ValueError("L3 audit evidence logits shape 无效")
    if any(row.get("l1_knownness_scores") is None for row in rows):
        head_scores = None
    else:
        l1_scores = np.asarray(
            [row["l1_knownness_scores"] for row in rows], dtype=np.float64
        )
        l2_scores = np.asarray(
            [row["l2_knownness_scores"] for row in rows], dtype=np.float64
        )
        if l1_scores.ndim != 2 or l2_scores.ndim != 2:
            raise ValueError("L3 audit evidence knownness scores shape 无效")
        head_scores = (l1_scores, l2_scores)
    evidence = L3ValidationEvidence(
        parent_ids=parent_ids,
        label_names=label_names,
        l1_logits=l1_logits,
        l2_logits=l2_logits,
        source_manifest_sha256=str(binding["source_manifest_sha256"]),
        endpoint_sha256=dict(binding["endpoint_sha256"]),
        feature_cache_manifest_sha256=str(
            binding["feature_cache_manifest_sha256"]
        ),
        role=str(header.get("role", "post_selection_audit")),
        test_payload_access_count=int(header.get("test_payload_access_count", 0)),
    )
    return header, evidence, head_scores


def finalize_l3_audit(
    project_root: str | Path,
    calibration: Mapping[str, Any],
    *,
    audit_manifest_path: str | Path,
    audit_feature_cache_manifest_path: str | Path,
    thresholds_frozen_before_access: bool,
    audit_split_name: str = "test",
    product_manifest_path: str | Path = "src/config/manifests.json",
    device: str = "cpu",
    batch_size: int | None = None,
    output_path: str | Path = "artifacts/radar_ln3/l3/radar_ln3_calibration.json",
    overwrite: bool = True,
    formal_available: bool = False,
    l1_head_path: str | Path | None = None,
    l2_head_path: str | Path | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    validate_event_cache: bool = True,
    audit_evidence_path: str | Path | None = None,
    config: ModelConfig | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """冻结阈值后执行独立 audit，并按需把审计结果登记到产品总 manifest。

    ``formal_available=True`` 时在封存和登记前完成发布状态提升，保证
    manifest 中登记的文件 hash/size 与唯一一次落盘写入一致。
    """

    evidence = collect_l3_audit_evidence(
        project_root,
        calibration,
        manifest_path=audit_manifest_path,
        feature_cache_manifest_path=audit_feature_cache_manifest_path,
        thresholds_frozen_before_access=thresholds_frozen_before_access,
        split_name=audit_split_name,
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        validate_event_cache=validate_event_cache,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
        config=config,
        retain_representations=calibration.get("knownness_source") == "independent_l3_head",
    )
    root = Path(project_root).resolve()
    if resolve_project_path(root, product_manifest_path) == resolve_project_path(
        root, audit_manifest_path
    ):
        # 同一份 manifest 已在 evidence 收集时装载并校验，避免重复解析。
        if evidence.manifest is None:
            raise ValueError("L3 audit evidence 缺少已装载 manifest")
        product_manifest = evidence.manifest
    else:
        product_manifest = load_manifest(
            product_manifest_path,
            root,
            verify_files=False,
            allow_unmaterialized=False,
        )
    expected_endpoints = dict(calibration.get("endpoint_sha256") or {})
    if dict(evidence.endpoint_sha256) != expected_endpoints:
        raise ValueError("L3 audit endpoint hash 与阈值拟合时不同")
    head_scores = None
    if calibration.get("knownness_source") == "independent_l3_head":
        l1_head_path = l1_head_path or "artifacts/radar_ln3/l3/heads/l1-knownness.pt"
        l2_head_path = l2_head_path or "artifacts/radar_ln3/l3/heads/l2-knownness.pt"
        head_ids, l1_scores, l2_scores, head_hashes, _ = _collect_l3_head_scores(
            project_root,
            manifest_path=audit_manifest_path,
            feature_cache_manifest_path=audit_feature_cache_manifest_path,
            split_name=audit_split_name,
            training_view=False,
            l1_head_path=l1_head_path,
            l2_head_path=l2_head_path,
            device=device,
            batch_size=batch_size,
            require_training_manifest_match=False,
            verify_files=verify_files,
            validate_event_cache=validate_event_cache,
            representation_root=representation_root,
            model_config_path=model_config_path,
            l1_checkpoint_path=l1_checkpoint_path,
            l2_checkpoint_path=l2_checkpoint_path,
            l1_config_path=l1_config_path,
            l2_config_path=l2_config_path,
            manifest=evidence.manifest,
            config=evidence.config,
            config_sha256=evidence.config_sha256,
            feature_caches=evidence.feature_caches,
            feature_cache_sha256=evidence.feature_cache_manifest_sha256,
            dataset=evidence.dataset,
            endpoint_sha256=evidence.endpoint_sha256,
            source_manifest_sha256=evidence.source_manifest_sha256,
            representations=evidence.representations,
        )
        if head_ids != evidence.parent_ids:
            raise ValueError("L3 audit knownness head parent binding 与 endpoint evidence 不一致")
        expected_heads = calibration.get("head_checkpoint_sha256")
        if isinstance(expected_heads, Mapping) and dict(expected_heads) != head_hashes:
            raise ValueError("L3 audit knownness head hash 与 calibration 不一致")
        head_scores = (l1_scores, l2_scores)
    rows = [
        {
            "l1_logits": evidence.l1_logits[index],
            "l2_logits": evidence.l2_logits[index],
            "l1_knownness_scores": None if head_scores is None else head_scores[0][index],
            "l2_knownness_scores": None if head_scores is None else head_scores[1][index],
        }
        for index in range(len(evidence.parent_ids))
    ]
    artifact = audit_l3_calibration(
        calibration,
        parent_ids=evidence.parent_ids,
        labels=evidence.label_names,
        evidences=rows,
        audit_role="post_selection_audit",
        audit_source_manifest_sha256=evidence.source_manifest_sha256,
        thresholds_frozen_before_access=thresholds_frozen_before_access,
    )
    if formal_available:
        artifact["formal_promotion"] = {
            "kind": "formal-audit-promotion",
            "source_status": str(artifact.get("status")),
            "audit_passed": bool((artifact.get("audit") or {}).get("passed")),
        }
        artifact["status"] = "formal_available"
        artifact["deployment_eligible"] = True
        artifact["research_only"] = False
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    body["audit_feature_cache_manifest_sha256"] = evidence.feature_cache_manifest_sha256
    body["audit_payload_access_count"] = int(evidence.test_payload_access_count)
    artifact = {**body, "artifact_sha256": canonical_hash(body)}
    saved_path, saved_file_sha256 = save_and_register_l3_calibration(
        project_root,
        artifact,
        output_path=output_path,
        manifest_path=product_manifest_path,
        overwrite=overwrite,
        return_sha256=True,
        manifest=product_manifest,
        register=register,
    )
    if audit_evidence_path is not None:
        _write_audit_evidence(
            project_root,
            audit_evidence_path,
            {
                "schema_version": AUDIT_EVIDENCE_SCHEMA,
                "data_binding": _audit_evidence_binding(artifact),
                "role": str(evidence.role),
                "split_name": str(audit_split_name),
                "record_count": len(evidence.parent_ids),
                "test_payload_access_count": int(evidence.test_payload_access_count),
                "calibration_artifact_sha256": str(artifact["artifact_sha256"]),
            },
            _audit_evidence_rows(evidence, head_scores),
        )
    return {
        "path": relative_project_path(project_root, saved_path),
        "file_sha256": saved_file_sha256,
        "artifact_sha256": artifact["artifact_sha256"],
        "status": artifact["status"],
        "deployment_eligible": bool(artifact["deployment_eligible"]),
        "audit": artifact["audit"],
    }


__all__ = [
    "L3ValidationEvidence",
    "collect_l3_audit_evidence",
    "collect_l3_validation_evidence",
    "finalize_l3_audit",
    "fit_l3_validation_calibration",
    "save_and_register_l3_calibration",
]
