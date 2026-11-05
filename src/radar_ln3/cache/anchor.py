"""事件级模型输出缓存的安全读写。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from radar_ln3.data.paths import resolve_project_path


ANCHOR_SCHEMA_VERSION = "RADAR-LN3-EVENT-BASELINE-LOGIT-ANCHOR-V1"
ANCHOR_NAME = "event_baseline_logits"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def anchor_payload_hash(payload: Mapping[str, Any]) -> str:
    """计算不依赖 torch 序列化布局的 anchor 内容 hash。"""

    digest = hashlib.sha256()
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"payload_sha256", "baseline_logits", "baseline_logits_padded"}
    }
    digest.update(_canonical_bytes(metadata))
    for key in ("baseline_logits", "baseline_logits_padded"):
        value = payload.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary: Path | None = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if anchor_payload_hash(existing) == anchor_payload_hash(payload):
                return
            if not overwrite:
                raise FileExistsError(f"拒绝覆盖内容不同的 anchor：{path}")
            os.replace(temporary, path)
            temporary = None
            return
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if anchor_payload_hash(existing) != anchor_payload_hash(payload):
                if not overwrite:
                    raise FileExistsError(f"拒绝覆盖竞态产生的不同 anchor：{path}")
                os.replace(temporary, path)
                temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class EventLogitAnchor:
    """按稳定事件身份提供 baseline logits 的缓存。"""

    path: Path
    sample_ids: tuple[str, ...]
    parent_sample_ids: tuple[str, ...]
    causal_event_indices: torch.Tensor
    split_names: tuple[str, ...]
    logits: torch.Tensor
    parent_ids: tuple[str, ...]
    parent_lengths: torch.Tensor
    payload: Mapping[str, Any]
    _sample_position: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        parent_ids = tuple(str(value) for value in self.parent_sample_ids)
        split_names = tuple(str(value) for value in self.split_names)
        indices = torch.as_tensor(self.causal_event_indices, dtype=torch.long).cpu()
        logits = torch.as_tensor(self.logits, dtype=torch.float32).cpu()
        lengths = torch.as_tensor(self.parent_lengths, dtype=torch.long).cpu()
        count = len(sample_ids)
        if count <= 0 or len(set(sample_ids)) != count:
            raise ValueError("anchor sample_id 必须非空且唯一")
        if len(parent_ids) != count or len(split_names) != count:
            raise ValueError("anchor 事件身份字段长度不一致")
        if indices.shape != (count,) or bool((indices < 0).any()):
            raise ValueError("anchor causal_event_indices 无效")
        if logits.ndim != 2 or logits.shape[0] != count or logits.shape[1] <= 0:
            raise ValueError("anchor logits 必须是 [N,C]")
        if not torch.isfinite(logits).all():
            raise ValueError("anchor logits 必须有限")
        if len(parent_ids) != int(lengths.sum().item()):
            raise ValueError("anchor parent lengths 与事件数量不一致")
        if not self.parent_ids or len(self.parent_ids) != len(lengths):
            raise ValueError("anchor parent 身份与 lengths 不一致")
        cursor = 0
        for parent_id, length in zip(self.parent_ids, lengths.tolist(), strict=True):
            if length <= 0:
                raise ValueError("anchor parent 不能没有事件")
            parent_slice = parent_ids[cursor : cursor + length]
            index_slice = indices[cursor : cursor + length].tolist()
            if any(value != parent_id for value in parent_slice):
                raise ValueError(f"anchor parent 事件未连续排列：{parent_id}")
            if index_slice != list(range(length)):
                raise ValueError(f"anchor event index 不连续：{parent_id}")
            cursor += length
        if len(set(split_names)) > 1:
            split_by_parent: dict[str, str] = {}
            for parent_id, split_name in zip(parent_ids, split_names, strict=True):
                previous = split_by_parent.setdefault(parent_id, split_name)
                if previous != split_name:
                    raise ValueError(f"anchor parent 跨越 split：{parent_id}")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "parent_sample_ids", parent_ids)
        object.__setattr__(self, "split_names", split_names)
        object.__setattr__(self, "causal_event_indices", indices)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "parent_lengths", lengths)
        object.__setattr__(
            self,
            "_sample_position",
            {sample_id: index for index, sample_id in enumerate(sample_ids)},
        )

    @property
    def class_count(self) -> int:
        return int(self.logits.shape[1])

    @property
    def sample_position(self) -> dict[str, int]:
        return self._sample_position

    def values_for_events(
        self,
        sample_ids: Sequence[str],
        parent_sample_ids: Sequence[str],
        causal_event_indices: Sequence[int],
        split_names: Sequence[str],
    ) -> np.ndarray:
        if not (
            len(sample_ids)
            == len(parent_sample_ids)
            == len(causal_event_indices)
            == len(split_names)
        ):
            raise ValueError("anchor 请求事件身份长度不一致")
        positions = self.sample_position
        rows: list[np.ndarray] = []
        for sample_id, parent_id, event_index, split_name in zip(
            sample_ids,
            parent_sample_ids,
            causal_event_indices,
            split_names,
            strict=True,
        ):
            position = positions.get(str(sample_id))
            if position is None:
                raise ValueError(f"anchor 缺少事件：{sample_id}")
            if self.parent_sample_ids[position] != str(parent_id):
                raise ValueError(f"anchor parent 不一致：{sample_id}")
            if int(self.causal_event_indices[position]) != int(event_index):
                raise ValueError(f"anchor event index 不一致：{sample_id}")
            if self.split_names[position] != str(split_name):
                raise ValueError(f"anchor split 不一致：{sample_id}")
            rows.append(self.logits[position].numpy())
        if not rows:
            return np.zeros((0, self.class_count), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


def _validate_payload(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if payload.get("schema_version") != ANCHOR_SCHEMA_VERSION:
        raise ValueError(f"anchor schema_version 不匹配：{path}")
    if payload.get("anchor_name") != ANCHOR_NAME:
        raise ValueError(f"anchor_name 不匹配：{path}")
    declared_hash = str(payload.get("payload_sha256") or "")
    if len(declared_hash) != 64 or declared_hash != anchor_payload_hash(payload):
        raise ValueError(f"anchor payload hash 不匹配：{path}")
    required = {
        "sample_ids", "parent_sample_ids", "causal_event_indices", "split_names",
        "parent_ids", "parent_lengths", "baseline_logits",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"anchor 缺少字段：{missing}")
    return dict(payload)


def load_event_logit_anchor(
    project_root: str | Path,
    path: str | Path,
) -> EventLogitAnchor:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=True)
    if target.suffix.lower() != ".pt":
        raise ValueError(f"anchor 必须是 .pt 文件：{target}")
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"anchor 根节点必须是对象：{target}")
    value = _validate_payload(payload, target)
    anchor = EventLogitAnchor(
        path=target,
        sample_ids=tuple(str(item) for item in value["sample_ids"]),
        parent_sample_ids=tuple(str(item) for item in value["parent_sample_ids"]),
        causal_event_indices=torch.as_tensor(value["causal_event_indices"], dtype=torch.long),
        split_names=tuple(str(item) for item in value["split_names"]),
        logits=torch.as_tensor(value["baseline_logits"], dtype=torch.float32),
        parent_ids=tuple(str(item) for item in value["parent_ids"]),
        parent_lengths=torch.as_tensor(value["parent_lengths"], dtype=torch.long),
        payload=value,
    )
    return anchor


def save_event_logit_anchor(
    project_root: str | Path,
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> EventLogitAnchor:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, path)
    value = dict(payload)
    value["schema_version"] = ANCHOR_SCHEMA_VERSION
    value["anchor_name"] = ANCHOR_NAME
    value.pop("payload_sha256", None)
    value["payload_sha256"] = anchor_payload_hash(value)
    _validate_payload(value, target)
    _atomic_save(target, value, overwrite=bool(overwrite))
    return load_event_logit_anchor(root, target)


__all__ = [
    "ANCHOR_NAME",
    "ANCHOR_SCHEMA_VERSION",
    "EventLogitAnchor",
    "anchor_payload_hash",
    "load_event_logit_anchor",
    "save_event_logit_anchor",
]
