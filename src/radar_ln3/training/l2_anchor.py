"""维护 L2 residual 训练所需的事件级 baseline anchor。"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_ln3.cache import (
    CachedParentSequenceDataset,
    load_event_logit_anchor,
    save_event_logit_anchor,
)
from radar_ln3.checkpoint import sha256_file
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.data.paths import resolve_project_path
from radar_ln3.features import build_model_batch, load_feature_cache_bundle
from radar_ln3.model import load_model_config

from .common import _identity_batch, _l2_inputs, _l2_reference_inputs
from .l2 import load_l2_base_model


DEFAULT_FEATURE_CACHE_MANIFEST = "data/cq08/cache/features/feature-cache-manifest.json"
DEFAULT_ANCHOR_PATH = "data/cq08/cache/l2-baseline-anchor.pt"
L2_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")


def _device(value: str | torch.device) -> torch.device:
    target = torch.device(str(value))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L2 anchor 配置要求 CUDA，但当前环境不可用")
    return target


def _archive_existing(path: Path) -> Path | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    archive = path.with_name(f"{path.name}.stale-before-{digest}")
    if not archive.exists():
        shutil.copy2(path, archive)
    return archive


def _load_existing(root: Path, path: str | Path) -> Any | None:
    target = resolve_project_path(root, path, must_exist=False)
    if not target.is_file():
        return None
    try:
        return load_event_logit_anchor(root, path)
    except (OSError, RuntimeError, ValueError):
        return None


def ensure_l2_baseline_anchor(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    manifest: DatasetManifest | None = None,
    feature_cache_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    output_path: str | Path = DEFAULT_ANCHOR_PATH,
    device: str = "cpu",
    batch_size: int = 8,
    verify_files: bool = True,
    validate_event_cache: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    """确保 anchor 覆盖当前 train/validation L2 view。

    旧 anchor 中仍然存在的事件沿用原 logits；新增事件通过当前冻结
    reference base 计算。这样数据视图扩展时不需要伪造 logits，也不需要
    重新访问 test 数据。
    """

    from radar_ln3.data import load_manifest

    root = Path(project_root).resolve()
    config = load_model_config(root)
    checked = manifest or load_manifest(
        root / manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    source_hash = data_manifest_hash(checked)
    dataset = CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        manifest=checked,
        split_name=None,
        include_unknown=False,
        training_view=True,
        allowed_labels=L2_LABELS,
        task_name="l2_anchor",
        validate_event_cache=bool(validate_event_cache),
    )
    expected_sample_ids = tuple(dataset.view.sample_ids)
    expected_set = set(expected_sample_ids)
    anchor_path = resolve_project_path(root, output_path, must_exist=False)
    existing = _load_existing(root, output_path)
    if (
        not bool(rebuild)
        and existing is not None
        and existing.payload.get("source_manifest_hash") == source_hash
        and set(existing.sample_ids) == expected_set
        and len(existing.sample_ids) == len(expected_sample_ids)
    ):
        return {
            "path": str(anchor_path.relative_to(root).as_posix()),
            "status": "reused",
            "source_manifest_hash": source_hash,
            "parent_count": len(existing.parent_ids),
            "record_count": len(existing.sample_ids),
            "computed_event_count": 0,
        }

    cache_path = resolve_project_path(root, feature_cache_manifest_path, must_exist=True)
    caches = load_feature_cache_bundle(
        root,
        cache_path,
        expected_source_manifest_hash=source_hash if verify_files else None,
        verify_integrity=bool(verify_files),
    )
    target_device = _device(device)
    _, model = load_l2_base_model(
        root,
        device=target_device,
        arm="robust_control",
        mode="residual",
    )
    model.eval()
    old_values: dict[str, np.ndarray] = {}
    old_identity: dict[str, tuple[str, int, str]] = {}
    if existing is not None:
        for index, sample_id in enumerate(existing.sample_ids):
            key = str(sample_id)
            old_values[key] = existing.logits[index].numpy().astype(
                np.float32,
                copy=True,
            )
            old_identity[key] = (
                str(existing.parent_sample_ids[index]),
                int(existing.causal_event_indices[index]),
                str(existing.split_names[index]),
            )

    current_records = {
        str(row["sample_id"]): row
        for row in checked.tr_records
        if str(row["sample_id"]) in expected_set
    }
    reusable = set()
    if not bool(rebuild):
        reusable = {
            sample_id
            for sample_id in expected_set
            if sample_id in old_values
            and sample_id in current_records
            and old_identity.get(sample_id)
            == (
                str(current_records[sample_id]["parent_sample_id"]),
                int(current_records[sample_id]["causal_event_index"]),
                str(current_records[sample_id]["split_name"]),
            )
        }
    missing = expected_set - reusable
    computed: dict[str, np.ndarray] = {}
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=_identity_batch,
    )
    with torch.inference_mode():
        for sequences in loader:
            if not missing:
                break
            batch = build_model_batch(
                sequences,
                task="l2",
                max_rd_frames=config.data.max_rd_frames_per_point,
                # 训练/推理 batch 仍按产品 contract 截断到 29 点；anchor
                # 必须覆盖完整 view 的事件身份，因此重建时保留当前
                # parent batch 的完整长度，避免访问被截断后的 logits。
                max_sequence_length=max(
                    int(config.data.max_sequence_length),
                    max(len(sequence.events) for sequence in sequences),
                ),
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
                stable_input_mode="normalized",
                allow_unmapped_labels=True,
            )
            inputs = _l2_inputs(batch, target_device)
            reference_inputs = _l2_reference_inputs(
                batch,
                target_device,
                base_inputs=inputs,
            )
            logits = model.frozen_reference_logits(base_inputs=reference_inputs)
            for row, sequence in enumerate(sequences):
                for event_index, event in enumerate(sequence.events):
                    sample_id = str(event.sample_id)
                    if sample_id in missing:
                        computed[sample_id] = (
                            logits[row, event_index].detach().cpu().numpy().astype(
                                np.float32,
                                copy=True,
                            )
                        )
    still_missing = sorted(missing - set(computed))
    if still_missing:
        raise ValueError(f"无法为新增 L2 anchor 事件计算 reference logits：{still_missing[:3]}")

    records = current_records
    sample_ids: list[str] = []
    parent_sample_ids: list[str] = []
    causal_indices: list[int] = []
    split_names: list[str] = []
    logits_rows: list[np.ndarray] = []
    parent_ids = list(dataset.view.parent_ids)
    parent_lengths: list[int] = []
    for parent_id in parent_ids:
        event_ids = list(dataset.view.event_ids_by_parent[parent_id])
        parent_lengths.append(len(event_ids))
        for sample_id in event_ids:
            record = records.get(str(sample_id))
            if record is None:
                raise ValueError(f"当前 L2 view 缺少 TR 事件：{sample_id}")
            value = (
                old_values.get(str(sample_id))
                if str(sample_id) in reusable
                else computed.get(str(sample_id))
            )
            if value is None:
                raise ValueError(f"当前 L2 anchor 缺少事件：{sample_id}")
            sample_ids.append(str(sample_id))
            parent_sample_ids.append(str(parent_id))
            causal_indices.append(int(record["causal_event_index"]))
            split_names.append(str(record["split_name"]))
            logits_rows.append(np.asarray(value, dtype=np.float32))

    if anchor_path.is_file():
        _archive_existing(anchor_path)
    payload = {
        "schema_version": "RADAR-LN3-EVENT-BASELINE-LOGIT-ANCHOR-V1",
        "anchor_name": "event_baseline_logits",
        "dataset": "CQ08",
        "class_names": list(L2_LABELS),
        "view_name": "l2_train_validation",
        "source_manifest_hash": source_hash,
        "split_source_sha256": str(checked.split.get("source_sha256") or ""),
        "sample_ids": sample_ids,
        "parent_sample_ids": parent_sample_ids,
        "causal_event_indices": causal_indices,
        "split_names": split_names,
        "parent_ids": parent_ids,
        "parent_lengths": parent_lengths,
        "baseline_logits": torch.from_numpy(np.stack(logits_rows, axis=0)),
        "provenance": {
            "reference_model": "l2base1_reference_residual_model",
            "previous_anchor_sha256": (
                None if existing is None else sha256_file(existing.path)
            ),
            "computed_event_count": len(computed),
            "computed_parent_ids": sorted(
                {str(records[sample_id]["parent_sample_id"]) for sample_id in computed}
            ),
        },
    }
    saved = save_event_logit_anchor(root, output_path, payload, overwrite=True)
    from radar_ln3.data.derived import register_product_artifacts

    register_product_artifacts(
        root,
        manifest_path=manifest_path,
        manifest=checked,
        overwrite=True,
        validate_manifest_payload=False,
    )
    return {
        "path": str(saved.path.relative_to(root).as_posix()),
        "status": "rebuilt",
        "source_manifest_hash": source_hash,
        "parent_count": len(parent_ids),
        "record_count": len(sample_ids),
        "computed_event_count": len(computed),
        "computed_parent_ids": sorted(
            {str(records[sample_id]["parent_sample_id"]) for sample_id in computed}
        ),
    }


__all__ = ["ensure_l2_baseline_anchor"]
