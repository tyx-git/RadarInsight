"""Task-isolated V27 datasets and model-input contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from radar_ln3.data import CQ08PointSequenceDataset, PointSequence, collate_point_sequences
from radar_ln3.features import parse_feature_json
from radar_ln3.known_gate_features import load_known_gate_feature_cache
from radar_ln3.runtime import resolve_project_path

from .protocol import (
    V27Config,
    V27_L1_TASK_NAME,
    V27_L2_TASK_NAME,
    task_allowed_labels,
    task_class_names,
    task_label_to_id,
)


V27_COMMON_MODEL_INPUT_KEYS = (
    "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason",
    "tr_padding_mask", "rd_frames", "rd_frame_mask", "rd_quality",
)
V27_L1_MODEL_INPUT_KEYS = (*V27_COMMON_MODEL_INPUT_KEYS, "course_features", "course_valid", "course_support")
V27_L2_MODEL_INPUT_KEYS = (*V27_COMMON_MODEL_INPUT_KEYS, "rd_physical_features", "rd_physical_valid", "micro_doppler_features", "micro_doppler_valid")


@dataclass(frozen=True)
class V27TaskSequence:
    base: PointSequence
    task: str
    course_features: torch.Tensor | None = None
    course_valid: torch.Tensor | None = None
    course_support: torch.Tensor | None = None


def _task_data(config: V27Config, task: str) -> Any:
    if task == V27_L1_TASK_NAME:
        return config.data.l1
    if task == V27_L2_TASK_NAME:
        return config.data.l2
    raise ValueError(f"unsupported V27 task: {task}")


class V27TaskDataset(Dataset[V27TaskSequence]):
    """One L1 or L2 known-class parent view; test/Unknown payloads stay sealed."""

    def __init__(self, project_root: str | Path, config: V27Config, *, task: str, split_name: str, parent_ids: Sequence[str] | None = None, parent_limit: int | None = None) -> None:
        if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
            raise ValueError(f"unsupported V27 task: {task}")
        if split_name not in {config.data.train_split_name, config.data.validation_split_name}:
            raise PermissionError("V27 datasets permit train/validation only; test is sealed")
        if parent_ids is not None and parent_limit is not None:
            raise ValueError("V27 cannot combine parent_ids and parent_limit")
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.task = task
        self.split_name = str(split_name)
        data = _task_data(config, task)
        if task == V27_L1_TASK_NAME:
            physical_path = None
            micro_path = None
            micro_dim = 0
        else:
            physical_path = data.train_rd_physical_cache_path if split_name == config.data.train_split_name else data.validation_rd_physical_cache_path
            micro_path = data.train_micro_doppler_cache_path if split_name == config.data.train_split_name else data.validation_micro_doppler_cache_path
            micro_dim = data.micro_doppler_dim
        self.base = CQ08PointSequenceDataset(
            self.project_root,
            index_path=config.data.index_path,
            split_name=split_name,
            max_sequence_length=config.data.max_sequence_length,
            max_rd_frames_per_point=config.data.max_rd_frames_per_point,
            rd_amplitude_center=data.rd_amplitude_center,
            rd_amplitude_scale=data.rd_amplitude_scale,
            exclude_anomalous_parents=False,
            sample_limit=None,
            tensor_cache_size=1,
            expected_schema_version=config.data.index_schema_version,
            rd_physical_cache_path=physical_path,
            micro_doppler_cache_path=micro_path,
            # The shared loader only accepts registered micro-Doppler widths.
            # L1 does not pass micro-Doppler to its model, so the historical
            # stable8 cache must not be opened merely to satisfy that loader.
            micro_doppler_value_dim=8,
            allowed_labels=task_allowed_labels(task),
            forbidden_labels={"UnknownTarget"} if task == V27_L1_TASK_NAME else {"ClutterTarget", "UnknownTarget"},
        )
        available = list(self.base.parent_ids)
        if parent_ids is not None:
            requested = [str(value) for value in parent_ids]
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("V27 parent_ids must be non-empty and unique")
            missing = sorted(set(requested) - set(available))
            if missing:
                raise ValueError(f"V27 requested parents are unavailable: {missing[:3]}")
            available = [parent_id for parent_id in available if parent_id in set(requested)]
        elif parent_limit is not None:
            if int(parent_limit) <= 0:
                raise ValueError("V27 parent_limit must be positive")
            available = available[:int(parent_limit)]
        if not available:
            raise ValueError(f"V27 {task}/{split_name} has no parents")
        self.parent_ids = available
        positions = {parent_id: index for index, parent_id in enumerate(self.base.parent_ids)}
        self._base_indices = [positions[parent_id] for parent_id in available]
        self.class_names = task_class_names(task)
        self._course_values: dict[str, np.ndarray] | None = None
        self._course_valid: dict[str, np.ndarray] | None = None
        if task == V27_L1_TASK_NAME:
            cache = data.train_course_cache_path if split_name == config.data.train_split_name else data.validation_course_cache_path
            # Cache order is historical-parent order, not the original CSV
            # order.  Load the full known-only cache; values are aligned to
            # this task view by sample id below.
            with np.load(resolve_project_path(self.project_root, cache, must_exist=True), allow_pickle=False) as archive:
                cache_ids = [str(value) for value in archive["sample_ids"].tolist()]
            values, valid, metadata = load_known_gate_feature_cache(
                resolve_project_path(self.project_root, cache, must_exist=True),
                expected_sample_ids=cache_ids,
                variant=data.course_feature_variant,
                allow_superset=False,
            )
            required_ids = self.base.frame["sample_id"].astype(str).tolist()
            if not set(required_ids).issubset(values) or not set(required_ids).issubset(valid):
                raise ValueError("V27 L1 course cache does not cover the task Dataset")
            if metadata["feature_names"] != [data.course_feature_name]:
                raise ValueError("V27 L1 course cache feature name differs")
            self._course_values = values
            self._course_valid = valid

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> V27TaskSequence:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("V27 dataset index must be an integer")
        position = int(index)
        if not 0 <= position < len(self._base_indices):
            raise IndexError("V27 dataset index is out of range")
        base = self.base[self._base_indices[position]]
        mapping = task_label_to_id(self.task)
        if base.label_name not in mapping:
            raise ValueError(f"V27 {self.task} received an unexpected label")
        if self.task != V27_L1_TASK_NAME:
            return V27TaskSequence(base=base, task=self.task)
        if self._course_values is None or self._course_valid is None:
            raise RuntimeError("V27 L1 course cache was not initialized")
        sample_ids = self.base._groups[base.parent_sample_id]["sample_id"].astype(str).tolist()[:base.event_count]
        values = torch.from_numpy(np.stack([self._course_values[item] for item in sample_ids], axis=0).astype(np.float32))
        valid = torch.from_numpy(np.stack([self._course_valid[item] for item in sample_ids], axis=0).astype(np.bool_))
        if values.shape != (base.event_count, 1) or valid.shape != values.shape or bool((values[~valid] != 0.0).any()):
            raise ValueError("V27 L1 course value/valid contract differs")
        raw_rows = self.base._groups[base.parent_sample_id].iloc[:base.event_count]
        support = _course_support_from_index_rows(raw_rows["tr_feature_values_json"].astype(str).tolist())
        expected_valid = support.ge(self.config.data.l1.course_minimum_valid_nonfirst_points).unsqueeze(-1)
        if not torch.equal(valid, expected_valid):
            raise ValueError("V27 L1 course cache validity differs from causal support")
        return V27TaskSequence(base=base, task=self.task, course_features=values, course_valid=valid, course_support=support)

    def label_counts(self) -> dict[str, int]:
        result = {name: 0 for name in self.class_names}
        mapping = task_label_to_id(self.task)
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            result[self.class_names[mapping[label]]] += 1
        return result


class V27OpenSetDataset(Dataset[V27TaskSequence]):
    """A role-scoped mixed-label view for frozen V27 open-set stages.

    Unlike :class:`V27TaskDataset`, this dataset may include ``UnknownTarget``
    after classification has frozen.  It has no classifier label mapping and
    therefore cannot accidentally be used by the known-class classification
    loss.  The caller supplies versioned feature caches whose sample set must
    exactly equal the selected mixed role population.
    """

    _ALLOWED_LABELS = frozenset(
        {
            "DroneTarget",
            "BirdTarget",
            "BalloonTarget",
            "ClutterTarget",
            "UnknownTarget",
        }
    )

    def __init__(
        self,
        project_root: str | Path,
        config: V27Config,
        *,
        task: str,
        split_name: str,
        parent_ids: Sequence[str],
        course_cache_path: str | Path | None = None,
        rd_physical_cache_path: str | Path | None = None,
        micro_doppler_cache_path: str | Path | None = None,
    ) -> None:
        if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
            raise ValueError("V27 open-set dataset requires a canonical task")
        if split_name not in {config.data.train_split_name, config.data.validation_split_name}:
            raise PermissionError("V27 open-set datasets permit train/validation only; test is sealed")
        requested = tuple(str(value) for value in parent_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("V27 open-set parent_ids must be non-empty and unique")
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.task = task
        self.split_name = str(split_name)
        self.parent_ids = list(requested)
        if task == V27_L1_TASK_NAME:
            if course_cache_path is None or any(
                value is not None
                for value in (rd_physical_cache_path, micro_doppler_cache_path)
            ):
                raise ValueError("V27 L1 open-set dataset requires only a course cache")
            physical_path = micro_path = None
            micro_dim = 0
        else:
            if course_cache_path is not None or rd_physical_cache_path is None or micro_doppler_cache_path is None:
                raise ValueError("V27 L2 open-set dataset requires physical4 and stable8 caches")
            physical_path = resolve_project_path(
                self.project_root, rd_physical_cache_path, must_exist=True
            )
            micro_path = resolve_project_path(
                self.project_root, micro_doppler_cache_path, must_exist=True
            )
            micro_dim = config.data.l2.micro_doppler_dim
        data = _task_data(config, task)
        self.base = CQ08PointSequenceDataset(
            self.project_root,
            index_path=config.data.index_path,
            split_name=split_name,
            max_sequence_length=config.data.max_sequence_length,
            max_rd_frames_per_point=config.data.max_rd_frames_per_point,
            rd_amplitude_center=data.rd_amplitude_center,
            rd_amplitude_scale=data.rd_amplitude_scale,
            exclude_anomalous_parents=False,
            sample_limit=None,
            tensor_cache_size=1,
            expected_schema_version=config.data.index_schema_version,
            rd_physical_cache_path=physical_path,
            micro_doppler_cache_path=micro_path,
            micro_doppler_value_dim=8,
            allowed_labels=self._ALLOWED_LABELS,
            forbidden_labels=(),
            allowed_parent_ids=requested,
        )
        if tuple(self.base.parent_ids) != tuple(sorted(requested)):
            raise ValueError("V27 open-set base dataset parent coverage/order differs")
        # Preserve the caller registry's canonical parent order rather than
        # silently inheriting the shared dataset's lexical ordering.
        positions = {parent_id: index for index, parent_id in enumerate(self.base.parent_ids)}
        self._base_indices = [positions[parent_id] for parent_id in requested]
        self._course_values: dict[str, np.ndarray] | None = None
        self._course_valid: dict[str, np.ndarray] | None = None
        if task == V27_L1_TASK_NAME:
            cache = resolve_project_path(self.project_root, course_cache_path, must_exist=True)
            expected_ids = self.base.frame["sample_id"].astype(str).tolist()
            values, valid, metadata = load_known_gate_feature_cache(
                cache,
                expected_sample_ids=expected_ids,
                variant=config.data.l1.course_feature_variant,
                # The role cache is materialized once for the complete mixed
                # role.  A loader may receive a deterministic parent subset
                # (for batching, smoke tests, or a later split), so select
                # that subset while preserving the caller's event ordering.
                allow_superset=True,
            )
            if metadata["feature_names"] != [config.data.l1.course_feature_name]:
                raise ValueError("V27 L1 open-set course feature differs")
            self._course_values, self._course_valid = values, valid

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> V27TaskSequence:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("V27 open-set dataset index must be an integer")
        position = int(index)
        if not 0 <= position < len(self._base_indices):
            raise IndexError("V27 open-set dataset index is out of range")
        base = self.base[self._base_indices[position]]
        if base.label_name not in self._ALLOWED_LABELS:
            raise ValueError("V27 open-set dataset received an unsupported label")
        if self.task != V27_L1_TASK_NAME:
            return V27TaskSequence(base=base, task=self.task)
        if self._course_values is None or self._course_valid is None:
            raise RuntimeError("V27 open-set L1 course cache was not initialized")
        sample_ids = self.base._groups[base.parent_sample_id]["sample_id"].astype(str).tolist()[:base.event_count]
        values = torch.from_numpy(
            np.stack([self._course_values[item] for item in sample_ids], axis=0).astype(np.float32)
        )
        valid = torch.from_numpy(
            np.stack([self._course_valid[item] for item in sample_ids], axis=0).astype(np.bool_)
        )
        support = _course_support_from_index_rows(
            self.base._groups[base.parent_sample_id]
            .iloc[:base.event_count]["tr_feature_values_json"]
            .astype(str)
            .tolist()
        )
        expected_valid = support.ge(
            self.config.data.l1.course_minimum_valid_nonfirst_points
        ).unsqueeze(-1)
        if (
            values.shape != (base.event_count, 1)
            or valid.shape != values.shape
            or bool((values[~valid] != 0.0).any())
            or not torch.equal(valid, expected_valid)
        ):
            raise ValueError("V27 open-set L1 course value/valid/support differs")
        return V27TaskSequence(
            base=base,
            task=self.task,
            course_features=values,
            course_valid=valid,
            course_support=support,
        )

    def base_label_counts(self) -> dict[str, int]:
        """Return mixed ground-truth counts for auditable non-classification use."""

        result = {name: 0 for name in sorted(self._ALLOWED_LABELS)}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            result[label] += 1
        return result


def _course_support_from_index_rows(raw_feature_json: Sequence[str]) -> torch.Tensor:
    """Return exact V21e non-first causal course support per event.

    The cache carries the circular-dispersion value/validity but not its
    support count.  Reconstructing the count from the processed index's
    causal TR JSON is deterministic, remains inside the V27 runtime boundary,
    and avoids treating a carried-valid prefix as a new observation.
    """

    values = list(raw_feature_json)
    support = torch.zeros((len(values),), dtype=torch.long)
    count = 0
    for index, raw in enumerate(values):
        parsed = parse_feature_json(str(raw))
        course = parsed.get("course_deg")
        if index > 0 and course is not None:
            try:
                numeric = float(course)
            except (TypeError, ValueError) as exc:
                raise ValueError("V27 L1 course support cannot parse course_deg") from exc
            if not math.isfinite(numeric):
                raise ValueError("V27 L1 course support received non-finite course_deg")
            count += 1
        support[index] = count
    return support


def _collate_v27_task_sequences(
    items: list[V27TaskSequence],
    *,
    include_classifier_labels: bool,
) -> dict[str, Any]:
    if not items:
        raise ValueError("V27 cannot collate an empty batch")
    task = items[0].task
    if any(item.task != task for item in items):
        raise ValueError("V27 batch cannot mix L1 and L2")
    batch = collate_point_sequences([item.base for item in items])
    batch["task"] = task
    batch["base_label_names"] = [item.base.label_name for item in items]
    if include_classifier_labels:
        mapping = task_label_to_id(task)
        labels = torch.tensor(
            [mapping[item.base.label_name] for item in items], dtype=torch.long
        )
        batch["labels"] = labels
        batch["task_label_names"] = [
            task_class_names(task)[int(label)] for label in labels
        ]
    if task == V27_L1_TASK_NAME:
        width = 1
        course = torch.zeros((len(items), int(batch["tr_features"].shape[1]), width), dtype=torch.float32)
        valid = torch.zeros_like(course, dtype=torch.bool)
        support = torch.zeros((len(items), int(batch["tr_features"].shape[1])), dtype=torch.long)
        for row, item in enumerate(items):
            if item.course_features is None or item.course_valid is None:
                raise RuntimeError("V27 L1 item lacks course evidence")
            length = item.base.event_count
            if item.course_features.shape != (length, width) or item.course_valid.shape != (length, width):
                raise ValueError("V27 L1 course tensor shape differs")
            course[row, :length] = item.course_features
            valid[row, :length] = item.course_valid
            if item.course_support is None or item.course_support.shape != (length,):
                raise ValueError("V27 L1 item lacks exact causal course support")
            support[row, :length] = item.course_support
        if bool(valid.masked_select(batch["tr_padding_mask"].unsqueeze(-1)).any()) or bool(course.masked_select(batch["tr_padding_mask"].unsqueeze(-1)).ne(0.0).any()):
            raise ValueError("V27 padded L1 course evidence must be zero/invalid")
        batch["course_features"] = course
        batch["course_valid"] = valid
        batch["course_support"] = support
    return batch


def collate_v27(items: list[V27TaskSequence]) -> dict[str, Any]:
    """Collate a known-class classification batch with task labels."""

    return _collate_v27_task_sequences(items, include_classifier_labels=True)


def collate_v27_open_set(items: list[V27TaskSequence]) -> dict[str, Any]:
    """Collate a mixed-label frozen-backbone batch without classifier labels."""

    batch = _collate_v27_task_sequences(items, include_classifier_labels=False)
    # ``collate_point_sequences`` necessarily emits the raw V0 label tensor and
    # label names because it is shared with the classifier loader.  Keeping
    # those fields on a mixed open-set batch makes an accidental classification
    # loss deceptively easy to write.  Preserve only the explicitly named
    # audit labels; model_input_kwargs never consumes them.
    batch.pop("labels", None)
    batch.pop("label_names", None)
    if "task_label_names" in batch:
        raise AssertionError("V27 open-set collate unexpectedly exposed task labels")
    if "base_label_names" not in batch:
        raise AssertionError("V27 open-set collate lost audit labels")
    return batch








def model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    task = str(batch.get("task", ""))
    keys = V27_L1_MODEL_INPUT_KEYS if task == V27_L1_TASK_NAME else V27_L2_MODEL_INPUT_KEYS if task == V27_L2_TASK_NAME else ()
    if not keys:
        raise ValueError("V27 batch has no canonical task")
    missing = [key for key in keys if key not in batch]
    if missing:
        raise ValueError(f"V27 batch is missing model inputs: {missing}")
    result: dict[str, Any] = {}
    for key in keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"V27 model input {key} must be a non-scalar tensor")
        result[key] = value.to(device)
    return result



__all__ = ["V27OpenSetDataset", "V27TaskDataset", "V27TaskSequence", "V27_COMMON_MODEL_INPUT_KEYS", "V27_L1_MODEL_INPUT_KEYS", "V27_L2_MODEL_INPUT_KEYS", "collate_v27", "collate_v27_open_set", "model_input_kwargs"]