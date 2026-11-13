"""正式模型 checkpoint 的统一读写和 provenance 工具。

该模块只处理可信项目内的 PyTorch artifact，不解释数据集字段，也不包含
任何层级模型或训练策略。所有写入都通过同目录临时文件完成，并拒绝静默
覆盖内容不同的已有文件。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any

import torch

from radar_ln3.data.paths import resolve_project_artifact_path
from radar_ln3.data.raw_reader import sha256_file as _data_sha256_file
from radar_ln3.data.contracts import DataError


class CheckpointError(RuntimeError):
    """checkpoint 结构、来源或完整性错误。"""


class _HashingWriter:
    """Forward a sequential write while recording the exact serialized bytes."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()

    def write(self, value: bytes) -> int:
        self.digest.update(value)
        return self.handle.write(value)

    def tell(self) -> int:
        return self.handle.tell()

    def flush(self) -> None:
        self.handle.flush()

    def fileno(self) -> int:
        return self.handle.fileno()


def canonical_hash(value: Any) -> str:
    """对可 JSON 序列化的 contract 计算稳定 hash。"""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint contract 无法规范化") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"不支持的 checkpoint JSON 类型：{type(value).__name__}")


def sha256_file(path: str | Path) -> str:
    """以流式方式计算文件 SHA-256。"""

    target = Path(path).resolve()
    if not target.is_file():
        raise CheckpointError(f"checkpoint 文件不存在：{target}")
    try:
        return _data_sha256_file(target)
    except (DataError, OSError) as exc:
        raise CheckpointError(f"checkpoint 文件无法计算 SHA-256：{target}") from exc


def state_hash(state: Mapping[str, Any]) -> str:
    """对 state mapping 的名称、dtype、shape 和原始字节计算稳定 hash。"""

    if not isinstance(state, Mapping) or not state:
        raise CheckpointError("checkpoint model_state 必须是非空对象")
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise CheckpointError(f"checkpoint state 不是 tensor：{name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    return canonical_hash(parts)


def _payload_equal(left: Any, right: Any) -> bool:
    """递归比较 checkpoint payload，避免比较 torch.save 的容器字节。"""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or type(left) is not type(right)
            or len(left) != len(right)
        ):
            return False
        return all(_payload_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Path) or isinstance(right, Path):
        return isinstance(left, Path) and isinstance(right, Path) and left == right
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _atomic_save(target: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            writer = _HashingWriter(handle)
            torch.save(dict(payload), writer)
            writer.flush()
            serialized_hash = writer.digest.hexdigest()
        if target.exists():
            if not target.is_file():
                raise CheckpointError(f"拒绝替换非文件 checkpoint：{target}")
            existing = _load_file(target)
            candidate = _load_file(temporary)
            if _payload_equal(existing, candidate):
                return "reused_identical", sha256_file(target)
            raise FileExistsError(f"拒绝覆盖内容不同的 checkpoint：{target}")
        # no-overwrite 写入必须使用原子创建，避免检查和替换之间的竞态覆盖。
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _load_file(target)
            candidate = _load_file(temporary)
            if _payload_equal(existing, candidate):
                return "reused_identical", sha256_file(target)
            raise FileExistsError(f"拒绝覆盖内容不同的 checkpoint：{target}")
        return "created", serialized_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_file(target: Path) -> dict[str, Any]:
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        pickle.UnpicklingError,
    ) as exc:
        raise CheckpointError(f"checkpoint 无法安全读取：{target}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint 根节点必须是对象：{target}")
    return dict(payload)


@dataclass(frozen=True)
class Checkpoint:
    """已经通过结构校验的 checkpoint。"""

    project_root: Path
    path: Path
    payload: Mapping[str, Any]
    _file_sha256: str | None = field(default=None, repr=False, compare=False)
    state_sha256: str | None = None

    @property
    def file_sha256(self) -> str:
        """Return the file digest, computing it once for this checkpoint object."""

        digest = self._file_sha256
        if digest is None:
            digest = sha256_file(self.path)
            object.__setattr__(self, "_file_sha256", digest)
        return digest

    @property
    def state(self) -> Mapping[str, torch.Tensor]:
        value = self.payload.get("model_state")
        if not isinstance(value, Mapping):
            raise CheckpointError(f"checkpoint 缺少 model_state：{self.path}")
        state = dict(value)
        return state

    @property
    def declared_state_hash(self) -> str | None:
        value = self.payload.get("model_state_sha256")
        return None if value is None else str(value)

    def verify_state_hash(self) -> str:
        actual = self.state_sha256 or state_hash(self.state)
        declared = self.declared_state_hash
        if declared is not None and declared != actual:
            raise CheckpointError(f"checkpoint model_state hash 不一致：{self.path}")
        return actual

    def provenance(self, *, kind: str, task: str | None = None) -> dict[str, Any]:
        """返回可登记到总 manifest 的文件 provenance。"""

        result: dict[str, Any] = {
            "path": self.path.relative_to(self.project_root).as_posix(),
            "kind": str(kind),
            "file_size": int(self.path.stat().st_size),
            "sha256": self.file_sha256,
            "schema_version": self.payload.get("checkpoint_schema")
            or self.payload.get("schema_version"),
        }
        if task is not None:
            result["task"] = str(task)
        for key in ("experiment_id", "epoch", "seed", "arm"):
            if key in self.payload:
                result[key] = self.payload[key]
        if self.declared_state_hash is not None:
            result["model_state_sha256"] = self.declared_state_hash
        return result

    @classmethod
    def load(
        cls,
        project_root: str | Path,
        path: str | Path,
        *,
        expected_schema: str | None = None,
        expected_task: str | None = None,
    ) -> "Checkpoint":
        root = Path(project_root).resolve()
        target = resolve_project_artifact_path(root, path, must_exist=True)
        if target.suffix.lower() != ".pt":
            raise CheckpointError(f"checkpoint 必须是 .pt 文件：{target}")
        payload = _load_file(target)
        if expected_schema is not None:
            actual_schema = payload.get("checkpoint_schema") or payload.get("schema_version")
            if actual_schema != expected_schema:
                raise CheckpointError(
                    f"checkpoint schema 不匹配：expected={expected_schema}, actual={actual_schema}"
                )
        if expected_task is not None and str(payload.get("task", "")) != str(expected_task):
            raise CheckpointError("checkpoint task 不匹配")
        state_digest = None
        if "model_state" in payload:
            state_value = payload.get("model_state")
            if not isinstance(state_value, Mapping):
                raise CheckpointError(f"checkpoint model_state 必须是对象：{target}")
            state_digest = state_hash(state_value)
            declared = payload.get("model_state_sha256")
            if declared is not None and str(declared) != state_digest:
                raise CheckpointError(f"checkpoint model_state hash 不一致：{target}")
        result = cls(
            project_root=root,
            path=target,
            payload=payload,
            _file_sha256=None,
            state_sha256=state_digest,
        )
        return result

    @classmethod
    def load_payload(cls, path: str | Path) -> dict[str, Any]:
        """读取已由调用方完成路径边界校验的 checkpoint payload。

        该入口只用于迁移 smoke 等显式指定的历史文件；正式运行时应优先
        使用 ``load``，由项目根目录边界校验配置路径。
        """

        target = Path(path).resolve()
        if target.suffix.lower() != ".pt" or not target.is_file():
            raise CheckpointError(f"checkpoint 必须是存在的 .pt 文件：{target}")
        return _load_file(target)

    @classmethod
    def save(
        cls,
        project_root: str | Path,
        path: str | Path,
        payload: Mapping[str, Any],
    ) -> tuple["Checkpoint", str]:
        root = Path(project_root).resolve()
        target = resolve_project_artifact_path(root, path)
        if target.suffix.lower() != ".pt":
            raise CheckpointError(f"checkpoint 必须是 .pt 文件：{target}")
        value = dict(payload)
        state = value.get("model_state")
        state_digest = None
        if isinstance(state, Mapping):
            state_digest = state_hash(state)
            value["model_state_sha256"] = state_digest
        status, serialized_hash = _atomic_save(target, value)
        return cls(
            project_root=root,
            path=target,
            payload=value,
            _file_sha256=serialized_hash,
            state_sha256=state_digest,
        ), status


def load_model_state(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    """加载模型状态，并忽略已退役且不参与训练/推理的 quality head 权重。"""

    current_keys = set(model.state_dict())
    filtered_state = {
        name: value
        for name, value in state.items()
        if not (
            name not in current_keys
            and ".trajectory_heads.quality_head." in f".{name}."
        )
    }
    model.load_state_dict(filtered_state, strict=True)


__all__ = [
    "Checkpoint",
    "CheckpointError",
    "canonical_hash",
    "load_model_state",
    "sha256_file",
    "state_hash",
]
