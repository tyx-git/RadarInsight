"""Frozen-endpoint L3 calibration from the registered validation view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_ln3.checkpoint import Checkpoint, canonical_hash, load_model_state, sha256_file
from radar_ln3.data import ParentSequenceDataset, load_manifest
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.derived import register_product_artifacts
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.data.paths import relative_project_path, resolve_project_path
from radar_ln3.features import (
    FeatureCache,
    build_model_batch,
    load_feature_cache_bundle,
    stable8_normalization_hash,
)
from radar_ln3.model import build_l1_model, build_l2_model, load_model_config

from radar_ln3.decision.l3 import (
    L3CalibrationPolicy,
    audit_l3_calibration,
    fit_l3_calibration,
    save_l3_calibration,
)


DEFAULT_FEATURE_CACHE_MANIFEST = Path(
    "data/cq08/cache/features/feature-cache-manifest.json"
)
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


@dataclass(frozen=True)
class L3ValidationEvidence:
    parent_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    l1_logits: np.ndarray
    l2_logits: np.ndarray
    source_manifest_sha256: str
    endpoint_sha256: Mapping[str, str]
    feature_cache_manifest_sha256: str
    l1_knownness_scores: np.ndarray | None = None
    l2_knownness_scores_by_route: np.ndarray | None = None
    head_checkpoint_sha256: Mapping[str, str] | None = None
    role: str = "exploratory_validation"
    test_payload_access_count: int = 0


def _model_inputs(batch: Mapping[str, Any], names: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    missing = [name for name in names if name not in batch]
    if missing:
        raise ValueError(f"L3 model batch 缺少输入：{missing}")
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"L3 model 输入必须是 tensor：{name}")
        result[name] = value.to(device=device)
    return result


def _l1_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return _model_inputs(
        batch,
        (
            "tr_features",
            "tr_missing_mask",
            "tr_dt_seconds",
            "tr_dt_valid",
            "tr_dt_reason",
            "tr_padding_mask",
            "rd_frames",
            "rd_frame_mask",
            "rd_quality",
            "course_features",
            "course_valid",
            "course_support",
        ),
        device,
    )


def _l2_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return _model_inputs(
        batch,
        (
            "tr_features",
            "tr_missing_mask",
            "tr_dt_seconds",
            "tr_dt_valid",
            "tr_dt_reason",
            "tr_padding_mask",
            "rd_frames",
            "rd_frame_mask",
            "rd_quality",
            "rd_physical_features",
            "rd_physical_valid",
            "micro_doppler_features",
            "micro_doppler_valid",
        ),
        device,
    )


def _l2_reference_inputs(
    batch: Mapping[str, Any],
    inputs: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = dict(inputs)
    for source, target in (
        ("reference_tr_features", "tr_features"),
        ("reference_tr_missing_mask", "tr_missing_mask"),
    ):
        value = batch.get(source)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"L3 L2 batch 缺少 reference 输入：{source}")
        result[target] = value.to(device=device)
    return result


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
    expected_config_hashes = {
        "l1": sha256_file(l1_config_path or config.path),
        "l2": sha256_file(l2_config_path or config.path),
    }
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
    device: str = "cpu",
    batch_size: int | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
) -> L3ValidationEvidence:
    """读取一个已注册 L3 role 的冻结 endpoint evidence。"""

    root = Path(project_root).resolve()
    config = load_model_config(root, model_config_path)
    if batch_size is not None and (
        isinstance(batch_size, bool) or int(batch_size) <= 0
    ):
        raise ValueError("L3 batch_size 必须是正整数")
    target_device = torch.device(str(device))
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L3 配置要求 CUDA，但当前环境不可用")
    manifest = load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    manifest_hash = data_manifest_hash(manifest)
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
        cache_hash = sha256_file(cache_manifest_path)
        endpoint_hashes = {
            "l1": sha256_file(
                resolve_project_path(
                    root,
                    l1_checkpoint_path or config.artifacts.l1_output_path,
                    must_exist=True,
                )
            ),
            "l2": sha256_file(
                resolve_project_path(
                    root,
                    l2_checkpoint_path or config.artifacts.l2_output_path,
                    must_exist=True,
                )
            ),
        }

        def load_or_materialize(task: str, allow_unknown: bool):
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
                        expected_endpoint_sha256=endpoint_hashes,
                        expected_config_sha256=sha256_file(config.path),
                    )
                except (OSError, RuntimeError, ValueError):
                    pass
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
                validate_event_cache=bool(verify_files),
                verify_files=bool(verify_files),
                require_training_manifest_match=bool(require_training_manifest_match),
                model_config_path=model_config_path,
                l1_checkpoint_path=l1_checkpoint_path,
                l2_checkpoint_path=l2_checkpoint_path,
                l1_config_path=l1_config_path,
                l2_config_path=l2_config_path,
            )
            save_frozen_endpoint_representation(
                root,
                representation,
                config=config,
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
        )
    dataset = ParentSequenceDataset(
        root,
        manifest=manifest,
        split_name=split_name,
        include_unknown=True,
        training_view=bool(training_view),
        task_name=("l3_audit" if source_role == "post_selection_audit" else "l3_validation"),
        verify_files=False,
    )
    cache_manifest_path = resolve_project_path(
        root,
        feature_cache_manifest_path,
        must_exist=True,
    )
    caches = load_feature_cache_bundle(
        root,
        cache_manifest_path,
        expected_source_manifest_hash=manifest_hash,
        verify_integrity=True,
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
                task="l1",
                max_rd_frames=config.data.max_rd_frames_per_point,
                max_sequence_length=config.data.max_sequence_length,
                feature_caches=caches,
                allow_unmapped_labels=True,
            )
            l2_batch = build_model_batch(
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
            )
            l1_output = l1(
                base_inputs=_l1_inputs(l1_batch, target_device),
                observation_increment_values=l1_batch["observation_increment_values"].to(
                    device=target_device
                ),
            )
            l2_inputs = _l2_inputs(l2_batch, target_device)
            if l2.mode in {"direct", "direct_grouped"}:
                l2_output = l2.direct_logits(base_inputs=l2_inputs)
            else:
                reference_inputs = _l2_reference_inputs(
                    l2_batch,
                    l2_inputs,
                    device=target_device,
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
    cache_manifest_sha256 = sha256_file(cache_manifest_path)
    return L3ValidationEvidence(
        parent_ids=tuple(parent_ids),
        label_names=tuple(label_names),
        l1_logits=np.concatenate(l1_logits_rows, axis=0),
        l2_logits=np.concatenate(l2_logits_rows, axis=0),
        source_manifest_sha256=manifest_hash,
        endpoint_sha256={
            "l1": sha256_file(l1_checkpoint.path),
            "l2": sha256_file(l2_checkpoint.path),
        },
        feature_cache_manifest_sha256=cache_manifest_sha256,
        role=source_role,
        test_payload_access_count=(
            len(dataset) if source_role == "post_selection_audit" and split_name == "test" else 0
        ),
    )


def collect_l3_validation_evidence(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_cache_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    device: str = "cpu",
    batch_size: int | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
) -> L3ValidationEvidence:
    """收集完整 validation labels；结果仅供 exploratory 校准。"""

    return _collect_l3_evidence(
        project_root,
        manifest_path=manifest_path,
        feature_cache_manifest_path=feature_cache_manifest_path,
        split_name="validation",
        source_role="exploratory_validation",
        training_view=True,
        expected_source_manifest_sha256=None,
        require_training_manifest_match=True,
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
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
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
) -> L3ValidationEvidence:
    """读取已由入口加载器校验过的独立审计 calibration。"""

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
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
    )


def save_and_register_l3_calibration(
    project_root: str | Path,
    calibration: Mapping[str, Any],
    *,
    output_path: str | Path = "artifacts/radar_ln3/l3/radar_ln3_calibration.json",
    manifest_path: str | Path = "src/config/manifests.json",
    overwrite: bool = False,
) -> Path:
    """保存 sealed calibration，并登记到唯一产品总 manifest。"""

    root = Path(project_root).resolve()
    saved_path = save_l3_calibration(
        root,
        output_path,
        calibration,
        overwrite=overwrite,
    )
    audit = calibration.get("audit")
    register_product_artifacts(
        root,
        manifest_path=manifest_path,
        overwrite=True,
        l3_calibration_descriptor={
            "path": relative_project_path(root, saved_path),
            "kind": "l3_calibration",
            "file_size": int(saved_path.stat().st_size),
            "sha256": sha256_file(saved_path),
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
    return saved_path


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
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, str], int]:
    from .l3_heads import (
        endpoint_representation_path,
        load_frozen_endpoint_representation,
        materialize_frozen_endpoint_representation,
        score_frozen_endpoint_representation,
    )

    common = {
        "project_root": project_root,
        "manifest_path": manifest_path,
        "feature_cache_manifest_path": feature_cache_manifest_path,
        "split_name": split_name,
        "training_view": training_view,
        "device": device,
        "batch_size": int(batch_size or 4),
        "validate_event_cache": bool(verify_files),
        "require_training_manifest_match": require_training_manifest_match,
        "verify_files": bool(verify_files),
        "model_config_path": model_config_path,
        "l1_checkpoint_path": l1_checkpoint_path,
        "l2_checkpoint_path": l2_checkpoint_path,
        "l1_config_path": l1_config_path,
        "l2_config_path": l2_config_path,
    }
    l1_rep = None
    l2_rep = None
    if representation_root is not None:
        root = Path(project_root).resolve()
        manifest = load_manifest(root / manifest_path, root, verify_files=bool(verify_files), allow_unmaterialized=False)
        source_hash = data_manifest_hash(manifest)
        cache_hash = sha256_file(resolve_project_path(root, feature_cache_manifest_path, must_exist=True))
        config = load_model_config(root, model_config_path)
        endpoint_hashes = {
            "l1": sha256_file(
                resolve_project_path(
                    root,
                    l1_checkpoint_path or config.artifacts.l1_output_path,
                    must_exist=True,
                )
            ),
            "l2": sha256_file(
                resolve_project_path(
                    root,
                    l2_checkpoint_path or config.artifacts.l2_output_path,
                    must_exist=True,
                )
            ),
        }
        try:
            l1_rep = load_frozen_endpoint_representation(
                root,
                endpoint_representation_path(root, task="l1", split_name=split_name, representation_root=representation_root),
                expected_task="l1",
                expected_source_manifest_sha256=source_hash,
                expected_feature_cache_manifest_sha256=cache_hash,
                expected_endpoint_sha256=endpoint_hashes,
                expected_config_sha256=sha256_file(config.path),
            )
            l2_rep = load_frozen_endpoint_representation(
                root,
                endpoint_representation_path(root, task="l2", split_name=split_name, representation_root=representation_root),
                expected_task="l2",
                expected_source_manifest_sha256=source_hash,
                expected_feature_cache_manifest_sha256=cache_hash,
                expected_endpoint_sha256=endpoint_hashes,
                expected_config_sha256=sha256_file(config.path),
            )
        except (OSError, RuntimeError, ValueError):
            l1_rep = None
            l2_rep = None
    if l1_rep is None or l2_rep is None:
        l1_rep = materialize_frozen_endpoint_representation(task="l1", **common)
        l2_rep = materialize_frozen_endpoint_representation(task="l2", **common)
    if l1_rep.parent_ids != l2_rep.parent_ids or l1_rep.label_names != l2_rep.label_names:
        raise ValueError("L3 knownness head representation L1/L2 parent 顺序不一致")
    l1_scores, l1_hash = score_frozen_endpoint_representation(
        l1_rep, project_root=project_root, path=l1_head_path,
        config_path=model_config_path, device=device
    )
    l2_scores, l2_hash = score_frozen_endpoint_representation(
        l2_rep, project_root=project_root, path=l2_head_path,
        config_path=model_config_path, device=device
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
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
    l1_router_from_head: bool = False,
    l2_knownness_threshold_mode: str = "shared",
    decision_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """拟合 validation 诊断 artifact；该数据角色不会获得部署资格。"""

    evidence = collect_l3_validation_evidence(
        project_root,
        manifest_path=manifest_path,
        feature_cache_manifest_path=feature_cache_manifest_path,
        device=device,
        batch_size=batch_size,
        verify_files=verify_files,
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
    )
    head_scores = None
    if (l1_head_path is None) != (l2_head_path is None):
        raise ValueError("L3 L1/L2 knownness head path 必须成对提供")
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
            representation_root=representation_root,
            model_config_path=model_config_path,
            l1_checkpoint_path=l1_checkpoint_path,
            l2_checkpoint_path=l2_checkpoint_path,
            l1_config_path=l1_config_path,
            l2_config_path=l2_config_path,
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
        artifact = {**artifact, "artifact_sha256": canonical_hash({key: value for key, value in artifact.items() if key != "artifact_sha256"})}
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
        )
    return artifact


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
    l1_head_path: str | Path | None = None,
    l2_head_path: str | Path | None = None,
    verify_files: bool = True,
    representation_root: str | Path | None = "artifacts/radar_ln3/l3/representations",
    model_config_path: str | Path = "src/config/model.json",
    l1_checkpoint_path: str | Path | None = None,
    l2_checkpoint_path: str | Path | None = None,
    l1_config_path: str | Path | None = None,
    l2_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """冻结阈值后执行独立 audit，并把审计结果登记到产品总 manifest。"""

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
        representation_root=representation_root,
        model_config_path=model_config_path,
        l1_checkpoint_path=l1_checkpoint_path,
        l2_checkpoint_path=l2_checkpoint_path,
        l1_config_path=l1_config_path,
        l2_config_path=l2_config_path,
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
            representation_root=representation_root,
            model_config_path=model_config_path,
            l1_checkpoint_path=l1_checkpoint_path,
            l2_checkpoint_path=l2_checkpoint_path,
            l1_config_path=l1_config_path,
            l2_config_path=l2_config_path,
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
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    body["audit_feature_cache_manifest_sha256"] = evidence.feature_cache_manifest_sha256
    body["audit_payload_access_count"] = int(evidence.test_payload_access_count)
    artifact = {**body, "artifact_sha256": canonical_hash(body)}
    saved_path = save_and_register_l3_calibration(
        project_root,
        artifact,
        output_path=output_path,
        manifest_path=product_manifest_path,
        overwrite=overwrite,
    )
    return {
        "path": relative_project_path(project_root, saved_path),
        "file_sha256": sha256_file(saved_path),
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
