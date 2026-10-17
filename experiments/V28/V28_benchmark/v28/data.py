"""V28 datasets with explicit known-only and frozen role boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.data import (
    CQ08PointSequenceDataset,
    PointSequence,
    collate_point_sequences,
)
from radar_ln3.features import (
    TR_OBSERVATION_INCREMENT_CLIP,
    encode_causal_tr_observation_increment_sequence,
    encode_tr_feature_values,
    parse_feature_json,
)
from radar_ln3.known_gate_features import (
    extract_causal_course_circular_incoherence,
    load_known_gate_feature_cache,
)

from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK

L1_SOURCE_LABELS = frozenset({"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"})
L2_SOURCE_LABELS = frozenset({"DroneTarget", "BirdTarget", "BalloonTarget"})
ALL_LABELS = frozenset((*L1_SOURCE_LABELS, "UnknownTarget"))
L1_KEYS = (
    "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
    "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
    "rd_quality", "course_features", "course_valid", "course_support",
    "observation_increment",
)
L2_KEYS = (
    "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
    "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
    "rd_quality", "rd_physical_features", "rd_physical_valid",
    "micro_doppler_features", "micro_doppler_valid", "observation_increment",
)
L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")


@dataclass(frozen=True)
class V28Sequence:
    base: PointSequence
    task: str
    course_features: torch.Tensor | None = None
    course_valid: torch.Tensor | None = None
    course_support: torch.Tensor | None = None
    observation_increment: torch.Tensor | None = None
    role: str | None = None

    @property
    def open_set(self) -> bool:
        return self.role is not None


def _course_support(raw: Sequence[str]) -> torch.Tensor:
    count = 0
    values: list[int] = []
    for index, item in enumerate(raw):
        if index > 0 and parse_feature_json(item).get("course_deg") is not None:
            count += 1
        values.append(count)
    return torch.tensor(values, dtype=torch.long)


def _course_maps(context: V28Context, base: Any, *, role: str | None, split: str, data: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the frozen V21e course cache and causally fill Unknown rows.

    The source cache is known-only.  Missing Unknown values are generated from
    the processed TR JSON after the endpoint is frozen; no Unknown label is
    used by the feature function itself.
    """

    cache_key = "validation" if role == "calibration" else "train"
    frame = base.frame
    known_ids = frame.loc[frame["label_name"].astype(str).ne("UnknownTarget"), "sample_id"].astype(str).tolist()
    values: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    # The held-out test split has no train-fitted course cache by design.  It
    # is reconstructed causally from the raw TR JSON after all classifier
    # endpoints are frozen; train/validation continue to use the sealed V21e
    # cache where available.
    use_sealed_cache = split != str(context.config.data["test_split_name"])
    if known_ids and use_sealed_cache:
        cache_path = data[cache_key]["course_cache"]
        cached_values, cached_valid, metadata = load_known_gate_feature_cache(
            context.project_root / cache_path,
            expected_sample_ids=known_ids,
            variant=str(data["course_feature_variant"]),
            allow_superset=True,
        )
        if metadata["feature_names"] != [str(data["course_feature_name"])]:
            raise ValueError("V28 course cache feature name differs")
        values.update(cached_values)
        valid.update(cached_valid)
    minimum = int(data["course_minimum_valid_nonfirst_points"])
    for parent_id, group in base._groups.items():
        courses = [parse_feature_json(str(value)).get("course_deg") for value in group["tr_feature_values_json"].astype(str)]
        generated_values, generated_valid = extract_causal_course_circular_incoherence(courses, minimum_valid_nonfirst_points=minimum)
        for sample_id, generated_value, generated_ok in zip(group["sample_id"].astype(str), generated_values, generated_valid, strict=True):
            sample_id = str(sample_id)
            if sample_id not in values:
                values[sample_id] = generated_value
                valid[sample_id] = generated_ok
            elif not np.array_equal(np.asarray(valid[sample_id]), np.asarray(generated_ok)):
                raise ValueError(f"V28 cached course validity differs from causal support: {sample_id}")
    return values, valid


def _task_data(context: V28Context, task: str) -> Mapping[str, Any]:
    if task == V28_L1_TASK:
        return context.config.data["l1"]
    if task == V28_L2_TASK:
        return context.config.data["l2"]
    raise ValueError(f"V28 unsupported task: {task}")


def _classification_parent_ids(context: V28Context, task: str, *, split_name: str, bucket: int | None) -> list[str]:
    """Return one known-only provenance bucket for classifier fitting/selection."""

    if bucket not in {None, 0, 1}:
        raise ValueError("V28 classifier provenance bucket must be 0 or 1")

    import hashlib
    import pandas as pd

    frame = pd.read_csv(context.project_root / context.config.data["index_path"], low_memory=False, usecols=["parent_sample_id", "provenance_component_id", "split_name", "label_name"])
    parents = frame.drop_duplicates("parent_sample_id")
    allowed = L1_SOURCE_LABELS if task == V28_L1_TASK else L2_SOURCE_LABELS
    selected = parents[
        parents["split_name"].astype(str).eq(str(split_name))
        & parents["label_name"].astype(str).isin(allowed)
        & (parents["provenance_component_id"].astype(str).map(
            lambda value: int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % 2 == bucket
        ) if bucket is not None else True)
    ]
    result = selected.sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist()
    if not result:
        raise ValueError(f"V28 {task} classifier split/bucket is empty: {split_name}/{bucket}")
    return result


class V28TaskDataset(Dataset[V28Sequence]):
    """Known-only classifier dataset.

    There is intentionally no ``open_set`` switch here.  Unknown is not
    representable by this class, which makes accidental classifier exposure a
    construction error rather than a caller convention.
    """

    def __init__(self, context: V28Context, *, task: str, split: str, parent_ids: Sequence[str] | None = None) -> None:
        if task not in {V28_L1_TASK, V28_L2_TASK} or split not in {"train", "validation", "test"}:
            raise ValueError("V28 classifier task or split is invalid")
        data = _task_data(context, task)
        split_name = context.config.data["train_split_name"] if split in {"train", "validation"} else context.config.data["test_split_name"]
        if task == V28_L1_TASK:
            allowed, forbidden = L1_SOURCE_LABELS, {"UnknownTarget"}
            physical = micro = None
            micro_dim = 8
        else:
            allowed, forbidden = L2_SOURCE_LABELS, {"ClutterTarget", "UnknownTarget"}
            # The validation Dataset is the second provenance bucket of the
            # train split, so it must use the separately materialized
            # classifier-development cache.  The test cache is only used by
            # the post-selection replay Dataset.
            source = data[split] if split in {"train", "validation"} else data["test"]
            physical, micro = source["rd_physical_cache"], source["micro_doppler_cache"]
            micro_dim = 8
        self.context, self.task, self.split, self.role = context, task, split, None
        self.base = CQ08PointSequenceDataset(
            context.project_root,
            index_path=context.config.data["index_path"],
            split_name=split_name,
            max_sequence_length=int(context.config.data["max_sequence_length"]),
            max_rd_frames_per_point=int(context.config.data["max_rd_frames_per_point"]),
            rd_amplitude_center=float(data["rd_amplitude_center"]),
            rd_amplitude_scale=float(data["rd_amplitude_scale"]),
            exclude_anomalous_parents=False,
            expected_schema_version=context.config.data["index_schema_version"],
            rd_physical_cache_path=physical,
            micro_doppler_cache_path=micro,
            micro_doppler_value_dim=micro_dim,
            allowed_labels=allowed,
            forbidden_labels=forbidden,
            allowed_parent_ids=parent_ids if parent_ids is not None else _classification_parent_ids(
                context,
                task,
                split_name=(context.config.data["train_split_name"] if split in {"train", "validation"} else context.config.data["test_split_name"]),
                bucket=(1 if split == "validation" else 0 if split == "train" else None),
            ),
        )
        self.parent_ids = list(self.base.parent_ids)
        self.class_names = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE") if task == V28_L1_TASK else L2_CLASSES
        self._course_values: dict[str, np.ndarray] = {}
        self._course_valid: dict[str, np.ndarray] = {}
        if task == V28_L1_TASK:
            self._course_values, self._course_valid = _course_maps(context, self.base, role=None, split=split, data=data)
        self._positions = {value: index for index, value in enumerate(self.base.parent_ids)}

    def __len__(self) -> int:
        return len(self.parent_ids)

    def label_counts(self) -> dict[str, int]:
        result = {name: 0 for name in self.class_names}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            key = "KNOWN_CLUTTER_EVIDENCE" if label == "ClutterTarget" else "KNOWN_TARGET_EVIDENCE" if self.task == V28_L1_TASK else label
            if key not in result:
                raise ValueError(f"V28 classifier label escaped task classes: {label}")
            result[key] += 1
        return result

    def __getitem__(self, index: int) -> V28Sequence:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("V28 dataset index must be integer")
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError("V28 dataset index out of range")
        base = self.base[self._positions[self.parent_ids[position]]]
        group = self.base._groups[base.parent_sample_id].iloc[: base.event_count]
        return _decorate_sequence(self.context, base, group=group, task=self.task, course_values=self._course_values, course_valid=self._course_valid)




class V28ReplayDataset(Dataset[V28Sequence]):
    """All test parents for post-selection endpoint replay only."""

    def __init__(self, context: V28Context, *, task: str) -> None:
        if task not in {V28_L1_TASK, V28_L2_TASK}:
            raise ValueError("V28 replay task is invalid")
        data = _task_data(context, task)
        split_name = str(context.config.data["test_split_name"])
        allowed = ALL_LABELS if task == V28_L1_TASK else L2_SOURCE_LABELS | {"UnknownTarget", "ClutterTarget"}
        source = data.get("test", {})
        self.context, self.task, self.role, self.split = context, task, "post_selection_replay", "test"
        self.base = CQ08PointSequenceDataset(
            context.project_root,
            index_path=context.config.data["index_path"],
            split_name=split_name,
            max_sequence_length=int(context.config.data["max_sequence_length"]),
            max_rd_frames_per_point=int(context.config.data["max_rd_frames_per_point"]),
            rd_amplitude_center=float(data["rd_amplitude_center"]),
            rd_amplitude_scale=float(data["rd_amplitude_scale"]),
            exclude_anomalous_parents=False,
            expected_schema_version=context.config.data["index_schema_version"],
            rd_physical_cache_path=source["rd_physical_cache"] if task == V28_L2_TASK else None,
            micro_doppler_cache_path=source["micro_doppler_cache"] if task == V28_L2_TASK else None,
            micro_doppler_value_dim=8,
            allowed_labels=allowed,
            forbidden_labels=set(),
        )
        self.parent_ids = list(self.base.parent_ids)
        self.class_names = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE") if task == V28_L1_TASK else L2_CLASSES
        self._course_values: dict[str, np.ndarray] = {}
        self._course_valid: dict[str, np.ndarray] = {}
        if task == V28_L1_TASK:
            self._course_values, self._course_valid = _course_maps(context, self.base, role=self.role, split="test", data=data)
        self._positions = {value: index for index, value in enumerate(self.base.parent_ids)}

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> V28Sequence:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("V28 replay dataset index must be integer")
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError("V28 replay dataset index out of range")
        base = self.base[self._positions[self.parent_ids[position]]]
        group = self.base._groups[base.parent_sample_id].iloc[: base.event_count]
        return _decorate_sequence(self.context, base, group=group, task=self.task, course_values=self._course_values, course_valid=self._course_valid, role=self.role)

    def label_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            result[label] = result.get(label, 0) + 1
        return result


def _decorate_sequence(context: V28Context, base: PointSequence, *, group: Any, task: str, course_values: Mapping[str, np.ndarray], course_valid: Mapping[str, np.ndarray], role: str | None = None) -> V28Sequence:
    raw = [parse_feature_json(str(value)) for value in group["tr_feature_values_json"]]
    if task == V28_L2_TASK:
        clip = context.config.data["l2"]["tr_clipping"]
        for value in raw:
            for name, lower, upper in (("range_m", clip["range_lower_m"], clip["range_upper_m"]), ("speed_mps", clip["speed_lower_mps"], clip["speed_upper_mps"])):
                if value.get(name) is not None:
                    value[name] = float(np.clip(float(value[name]), float(lower), float(upper)))
    encoded = [encode_tr_feature_values(value) for value in raw]
    features = torch.from_numpy(np.stack([value[0] for value in encoded]).astype(np.float32))
    missing = torch.from_numpy(np.stack([value[1] for value in encoded]).astype(np.float32))
    observation, _ = encode_causal_tr_observation_increment_sequence(raw)
    if np.any(np.abs(observation) > TR_OBSERVATION_INCREMENT_CLIP + 1.0e-6):
        raise ValueError("V28 observation increment exceeds registered clip")
    base = replace(base, tr_features=features, tr_missing_mask=missing)
    observation_tensor = torch.from_numpy(observation.astype(np.float32))
    if task == V28_L2_TASK:
        return V28Sequence(base, task, observation_increment=observation_tensor, role=role)
    sample_ids = group["sample_id"].astype(str).tolist()
    values = torch.from_numpy(np.stack([course_values[sample_id] for sample_id in sample_ids]).astype(np.float32))
    valid = torch.from_numpy(np.stack([course_valid[sample_id] for sample_id in sample_ids]).astype(bool))
    support = _course_support(group["tr_feature_values_json"].astype(str).tolist())
    expected = support.ge(int(context.config.data["l1"]["course_minimum_valid_nonfirst_points"])).unsqueeze(-1)
    if not torch.equal(valid, expected):
        raise ValueError("V28 causal course validity differs from support")
    return V28Sequence(base, task, values, valid, support, observation_tensor, role)


def collate_v28(items: list[V28Sequence]) -> dict[str, Any]:
    if not items or len({(item.task, item.role) for item in items}) != 1:
        raise ValueError("V28 batch is empty or mixes task/role semantics")
    task, role = items[0].task, items[0].role
    batch = collate_point_sequences([item.base for item in items])
    batch.update(task=task, role=role, base_label_names=[item.base.label_name for item in items], task_label_names=[item.base.label_name for item in items])
    width = int(batch["tr_features"].shape[1])
    observation = torch.zeros((len(items), width, 9), dtype=torch.float32)
    for row, item in enumerate(items):
        observation[row, : item.base.event_count] = item.observation_increment
    batch["observation_increment"] = observation
    if task == V28_L1_TASK:
        course = torch.zeros((len(items), width, 1), dtype=torch.float32)
        valid = torch.zeros_like(course, dtype=torch.bool)
        support = torch.zeros((len(items), width), dtype=torch.long)
        for row, item in enumerate(items):
            length = item.base.event_count
            course[row, :length] = item.course_features
            valid[row, :length] = item.course_valid
            support[row, :length] = item.course_support
        batch.update(course_features=course, course_valid=valid, course_support=support)
        if role is None:
            batch["labels"] = torch.tensor([0 if item.base.label_name != "ClutterTarget" else 1 for item in items], dtype=torch.long)
    elif role is None:
        batch["labels"] = torch.tensor([L2_CLASSES.index(item.base.label_name) for item in items], dtype=torch.long)
    if role is not None:
        # The audit label is retained under an explicit name; no classifier
        # label tensor is emitted for role datasets.
        batch.pop("labels", None)
        batch.pop("label_names", None)
    return batch


def model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    task = batch.get("task")
    keys = L1_KEYS if task == V28_L1_TASK else L2_KEYS if task == V28_L2_TASK else ()
    if not keys:
        raise ValueError("V28 batch task is invalid")
    missing = [key for key in keys if key not in batch]
    if missing:
        raise ValueError(f"V28 batch is missing model inputs: {missing}")
    return {key: batch[key].to(device) for key in keys}


def build_dataloader(dataset: Dataset[V28Sequence], *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V28 batch size must be positive")
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_v28, generator=torch.Generator(device="cpu").manual_seed(int(seed)))


__all__ = ["ALL_LABELS", "L1_KEYS", "L2_CLASSES", "L2_KEYS", "V28ReplayDataset", "V28Sequence", "V28TaskDataset", "build_dataloader", "collate_v28", "model_input_kwargs"]