from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .losses import parent_final_task_ce_with_prefix_auxiliary
from .task_models import summarize_task_records


V19_SCHEMA_VERSION = "RADAR-LN3-V19-micro-doppler-late-residual-v1"
V19_CACHE_SCHEMA_VERSION = (
    "RADAR-LN3-V19-cq08-dbb-frozen-control-logits-causal-stable8-cache-v1"
)
V19_CHECKPOINT_SCHEMA_VERSION = (
    "RADAR-LN3-V19-dbb-micro-doppler-late-residual-checkpoint-v1"
)
V19_CONFIG_NAME = (
    "RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42"
)
V19_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V19_FEATURE_DIM = 8
V19_CLASS_COUNT = 3
V19_RESIDUAL_BOUND = 3.0
V19_TRAINABLE_PARAMETER_COUNT = V19_FEATURE_DIM * V19_CLASS_COUNT

V19_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42-config.json"
)
V19_NORMALIZATION_RELATIVE_PATH = Path(
    "data/processed/RADAR-LN3-V19-cq08-dbb-micro-doppler-late-residual-normalization.json"
)
V19_TRAIN_CACHE_RELATIVE_PATH = Path(
    "data/processed/RADAR-LN3-V19-cq08-dbb-train-micro-doppler-late-residual-cache.npz"
)
V19_VALIDATION_CACHE_RELATIVE_PATH = Path(
    "data/processed/RADAR-LN3-V19-cq08-dbb-val-micro-doppler-late-residual-cache.npz"
)
V19_CONTRACT_RELATIVE_PATH = Path(
    "data/processed/RADAR-LN3-V19-cq08-dbb-micro-doppler-late-residual-contract.json"
)
V19_READINESS_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19-micro-doppler-late-residual-train-readiness.json"
)
V19_REPORT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19-micro-doppler-late-residual-validation.json"
)
V19_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/"
    "RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42-formal"
)

V19_CACHE_FIELDS = (
    "schema_version",
    "split_name",
    "parent_sample_ids",
    "track_ids",
    "source_names",
    "label_names",
    "labels",
    "lengths",
    "padding_mask",
    "micro_doppler_valid",
    "valid_counts",
    "baseline_logits",
    "causal_features",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def causal_valid_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean of valid observations available at every causal prefix."""

    if values.ndim != 3 or values.shape[-1] != V19_FEATURE_DIM:
        raise ValueError("V19 stable8 values must have shape [batch,sequence,8]")
    if valid.shape != values.shape[:2] or padding_mask.shape != values.shape[:2]:
        raise ValueError("V19 stable8 masks do not match the value sequence")
    if valid.dtype != torch.bool or padding_mask.dtype != torch.bool:
        raise ValueError("V19 stable8 masks must be boolean")
    if not torch.isfinite(values).all():
        raise ValueError("V19 stable8 values must be finite")
    if bool(valid.masked_select(padding_mask).any()):
        raise ValueError("V19 padded events cannot contain valid stable8 evidence")
    active = valid & ~padding_mask
    selected = values * active.to(dtype=values.dtype).unsqueeze(-1)
    cumulative_sum = selected.cumsum(dim=1)
    counts = active.long().cumsum(dim=1)
    result = cumulative_sum / counts.clamp_min(1).to(dtype=values.dtype).unsqueeze(-1)
    result = torch.where(counts.unsqueeze(-1) > 0, result, torch.zeros_like(result))
    result = result.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    counts = counts.masked_fill(padding_mask, 0)
    if not torch.isfinite(result).all():
        raise ValueError("V19 causal stable8 mean is non-finite")
    return result, counts


def normalize_stable8(
    values: torch.Tensor,
    valid: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    median: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if median.shape != (V19_FEATURE_DIM,) or scale.shape != (V19_FEATURE_DIM,):
        raise ValueError("V19 normalization vectors must have shape [8]")
    if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
        raise ValueError("V19 normalization vectors must be finite")
    if bool((scale <= 0.0).any()):
        raise ValueError("V19 normalization scale must be positive")
    normalized = (values - median) / scale
    normalized = normalized.masked_fill(~valid.unsqueeze(-1), 0.0)
    return causal_valid_mean(normalized, valid, padding_mask)


class MicroDopplerLateResidual(nn.Module):
    """A zero-initialized, bounded, zero-sum correction to frozen DBB logits."""

    def __init__(self, *, residual_bound: float = V19_RESIDUAL_BOUND) -> None:
        super().__init__()
        self.feature_dim = V19_FEATURE_DIM
        self.class_count = V19_CLASS_COUNT
        self.residual_bound = float(residual_bound)
        if not np.isfinite(self.residual_bound) or self.residual_bound <= 0.0:
            raise ValueError("V19 residual bound must be finite and positive")
        self.linear = nn.Linear(self.feature_dim, self.class_count, bias=False)
        nn.init.zeros_(self.linear.weight)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def residual_logits(
        self,
        causal_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if causal_features.ndim != 3 or causal_features.shape[-1] != self.feature_dim:
            raise ValueError("V19 causal features must have shape [batch,sequence,8]")
        if padding_mask.shape != causal_features.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("V19 padding mask does not match causal features")
        if not torch.isfinite(causal_features).all():
            raise ValueError("V19 causal features must be finite")
        raw = self.linear(causal_features)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        maximum = centered.abs().amax(dim=-1, keepdim=True)
        multiplier = torch.clamp(
            self.residual_bound / maximum.clamp_min(torch.finfo(centered.dtype).tiny),
            max=1.0,
        )
        bounded = centered * multiplier
        bounded = bounded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(bounded).all():
            raise ValueError("V19 residual logits are non-finite")
        tolerance = 2.0e-6 if bounded.dtype == torch.float32 else 1.0e-10
        if bool((bounded.abs() > self.residual_bound + tolerance).any()):
            raise ValueError("V19 residual logits exceed the frozen bound")
        if bool((bounded.sum(dim=-1).abs() > tolerance).any()):
            raise ValueError("V19 residual logits violate the zero-sum contract")
        return bounded

    def forward(
        self,
        baseline_logits: torch.Tensor,
        causal_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if baseline_logits.shape != (*causal_features.shape[:2], self.class_count):
            raise ValueError("V19 frozen baseline logits do not match the causal features")
        if not torch.isfinite(baseline_logits).all():
            raise ValueError("V19 frozen baseline logits must be finite")
        residual = self.residual_logits(causal_features, padding_mask)
        combined = baseline_logits + residual
        if not torch.isfinite(combined).all():
            raise ValueError("V19 combined logits are non-finite")
        return {
            "baseline_logits": baseline_logits,
            "residual_logits": residual,
            "sequence_logits": combined,
        }


class ResidualCacheDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            if tuple(payload.files) != V19_CACHE_FIELDS:
                raise ValueError(
                    f"V19 cache fields differ: expected={V19_CACHE_FIELDS}, actual={payload.files}"
                )
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        schema = str(self.arrays["schema_version"].item())
        if schema != V19_CACHE_SCHEMA_VERSION:
            raise ValueError(f"V19 cache schema differs: {schema}")
        self.split_name = str(self.arrays["split_name"].item())
        self.parent_sample_ids = self.arrays["parent_sample_ids"].astype(str)
        self.track_ids = self.arrays["track_ids"].astype(str)
        self.source_names = self.arrays["source_names"].astype(str)
        self.label_names = self.arrays["label_names"].astype(str)
        self.labels = np.asarray(self.arrays["labels"], dtype=np.int64)
        self.lengths = np.asarray(self.arrays["lengths"], dtype=np.int64)
        self.padding_mask = np.asarray(self.arrays["padding_mask"], dtype=np.bool_)
        self.micro_doppler_valid = np.asarray(
            self.arrays["micro_doppler_valid"], dtype=np.bool_
        )
        self.valid_counts = np.asarray(self.arrays["valid_counts"], dtype=np.int64)
        self.baseline_logits = np.asarray(
            self.arrays["baseline_logits"], dtype=np.float32
        )
        self.causal_features = np.asarray(
            self.arrays["causal_features"], dtype=np.float32
        )
        self._validate()

    def _validate(self) -> None:
        count = len(self.parent_sample_ids)
        vectors = (
            self.track_ids,
            self.source_names,
            self.label_names,
            self.labels,
            self.lengths,
        )
        if any(len(value) != count for value in vectors):
            raise ValueError("V19 cache parent metadata lengths differ")
        if len(set(self.parent_sample_ids.tolist())) != count:
            raise ValueError("V19 cache repeats a parent_sample_id")
        if self.padding_mask.ndim != 2:
            raise ValueError("V19 cache padding mask must have shape [parent,sequence]")
        expected_sequence = self.padding_mask.shape
        if (
            self.micro_doppler_valid.shape != expected_sequence
            or self.valid_counts.shape != expected_sequence
            or self.baseline_logits.shape != (*expected_sequence, V19_CLASS_COUNT)
            or self.causal_features.shape != (*expected_sequence, V19_FEATURE_DIM)
        ):
            raise ValueError("V19 cache tensor shapes differ")
        if not np.isfinite(self.baseline_logits).all() or not np.isfinite(
            self.causal_features
        ).all():
            raise ValueError("V19 cache contains non-finite tensors")
        if np.any(self.labels < 0) or np.any(self.labels >= V19_CLASS_COUNT):
            raise ValueError("V19 cache labels are outside [0,2]")
        active_lengths = (~self.padding_mask).sum(axis=1)
        if not np.array_equal(active_lengths, self.lengths):
            raise ValueError("V19 cache lengths differ from padding masks")
        if np.any(self.lengths <= 0):
            raise ValueError("V19 cache contains an empty parent")
        if np.any(self.micro_doppler_valid & self.padding_mask):
            raise ValueError("V19 cache marks padded micro-Doppler evidence as valid")
        if np.any(self.valid_counts[self.padding_mask] != 0):
            raise ValueError("V19 cache padded valid counts must be zero")
        expected_counts = np.cumsum(
            self.micro_doppler_valid & ~self.padding_mask, axis=1, dtype=np.int64
        )
        expected_counts[self.padding_mask] = 0
        if not np.array_equal(expected_counts, self.valid_counts):
            raise ValueError("V19 cache valid counts are not causal cumulative counts")
        if np.any(self.causal_features[self.valid_counts == 0] != 0.0):
            raise ValueError("V19 cache must use zero before the first valid observation")

    def __len__(self) -> int:
        return len(self.parent_sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "parent_sample_id": str(self.parent_sample_ids[index]),
            "track_id": str(self.track_ids[index]),
            "source_name": str(self.source_names[index]),
            "label_name": str(self.label_names[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "length": torch.tensor(int(self.lengths[index]), dtype=torch.long),
            "padding_mask": torch.from_numpy(self.padding_mask[index].copy()),
            "micro_doppler_valid": torch.from_numpy(
                self.micro_doppler_valid[index].copy()
            ),
            "valid_counts": torch.from_numpy(self.valid_counts[index].copy()),
            "baseline_logits": torch.from_numpy(self.baseline_logits[index].copy()),
            "causal_features": torch.from_numpy(self.causal_features[index].copy()),
        }


def build_residual_dataloader(
    dataset: ResidualCacheDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def residual_objective(
    sequence_logits: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    class_weights: torch.Tensor,
    prefix_auxiliary_weight: float,
    prefix_auxiliary_min_events: int,
) -> dict[str, torch.Tensor]:
    return parent_final_task_ce_with_prefix_auxiliary(
        sequence_logits,
        batch["label"],
        batch["padding_mask"],
        class_weights=class_weights,
        prefix_auxiliary_weight=prefix_auxiliary_weight,
        prefix_auxiliary_min_events=prefix_auxiliary_min_events,
    )


def evaluate_residual(
    model: MicroDopplerLateResidual | None,
    dataset: ResidualCacheDataset,
    *,
    device: torch.device,
    batch_size: int,
    prefix_auxiliary_min_events: int,
) -> dict[str, Any]:
    if model is not None:
        model.eval()
    loader = build_residual_dataloader(
        dataset, batch_size=batch_size, shuffle=False
    )
    window_records: list[dict[str, Any]] = []
    eligible_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    residual_abs_max = 0.0
    residual_sum_abs_max = 0.0
    residual_pairwise_max = 0.0
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            baseline = tensors["baseline_logits"]
            if model is None:
                logits = baseline
                residual = torch.zeros_like(baseline)
            else:
                output = model(
                    baseline,
                    tensors["causal_features"],
                    tensors["padding_mask"],
                )
                logits = output["sequence_logits"]
                residual = output["residual_logits"]
            residual_abs_max = max(residual_abs_max, float(residual.abs().max().item()))
            residual_sum_abs_max = max(
                residual_sum_abs_max,
                float(residual.sum(dim=-1).abs().max().item()),
            )
            pairwise = residual.amax(dim=-1) - residual.amin(dim=-1)
            residual_pairwise_max = max(
                residual_pairwise_max, float(pairwise.max().item())
            )
            probabilities = torch.softmax(logits, dim=-1)
            for batch_index, parent_id in enumerate(batch["parent_sample_id"]):
                length = int(batch["length"][batch_index].item())
                true_id = int(batch["label"][batch_index].item())
                final: dict[str, Any] | None = None
                for event_index in range(length):
                    event_probability = probabilities[batch_index, event_index]
                    pred_id = int(event_probability.argmax().item())
                    final = {
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "confidence": float(event_probability[pred_id].item()),
                        "parent_sample_id": str(parent_id),
                        "track_id": str(batch["track_id"][batch_index]),
                        "source_name": str(batch["source_name"][batch_index]),
                        "label_name": str(batch["label_name"][batch_index]),
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "product_action_eligible": (
                            event_index + 1 >= int(prefix_auxiliary_min_events)
                        ),
                    }
                    window_records.append(dict(final))
                    if (
                        event_index + 1 >= int(prefix_auxiliary_min_events)
                        and event_index + 1 < length
                    ):
                        eligible_records.append(dict(final))
                if final is None:
                    raise ValueError(f"V19 evaluation received an empty parent: {parent_id}")
                parent_records.append(final)
    return {
        "task_name": "dbb_specialist",
        "class_names": list(V19_CLASS_NAMES),
        "parent_count": len(parent_records),
        "window_count": len(window_records),
        "window_metrics": summarize_task_records(
            window_records,
            class_names=V19_CLASS_NAMES,
            scope="causal_prefix_windows",
        ),
        "eligible_prefix_metrics": summarize_task_records(
            eligible_records,
            class_names=V19_CLASS_NAMES,
            scope="eligible_causal_prefix_auxiliary_excluding_final",
        ),
        "parent_metrics": summarize_task_records(
            parent_records,
            class_names=V19_CLASS_NAMES,
            scope="parent_final",
        ),
        "prefix_action_contract": {
            "minimum_evidence_events": int(prefix_auxiliary_min_events),
            "events_before_minimum": "diagnostic_only_pending",
            "eligible_prefix_excludes_parent_final": True,
            "eligible_prefix_count": len(eligible_records),
        },
        "residual_contract": {
            "absolute_max": residual_abs_max,
            "zero_sum_absolute_max": residual_sum_abs_max,
            "maximum_pairwise_correction": residual_pairwise_max,
            "configured_absolute_bound": V19_RESIDUAL_BOUND,
            "configured_pairwise_bound": 2.0 * V19_RESIDUAL_BOUND,
        },
        "records": {
            "window": window_records,
            "eligible_prefix": eligible_records,
            "parent": parent_records,
        },
        "test": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }


def strip_evaluation_records(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "records"}


def save_residual_checkpoint(
    path: Path,
    model: MicroDopplerLateResidual,
    optimizer: torch.optim.Optimizer,
    *,
    config_name: str,
    config_sha256: str,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema": V19_CHECKPOINT_SCHEMA_VERSION,
        "config_name": str(config_name),
        "config_sha256": str(config_sha256),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_contract": {
            "feature_dim": V19_FEATURE_DIM,
            "class_count": V19_CLASS_COUNT,
            "bias": False,
            "zero_sum": True,
            "residual_bound": V19_RESIDUAL_BOUND,
            "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "training_state": dict(training_state),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_residual_checkpoint(
    path: Path,
    model: MicroDopplerLateResidual,
    *,
    expected_config_name: str,
    expected_config_sha256: str,
    expected_provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_schema") != V19_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V19 checkpoint schema differs")
    if payload.get("config_name") != expected_config_name:
        raise ValueError("V19 checkpoint config name differs")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("V19 checkpoint config hash differs")
    if dict(payload.get("provenance") or {}) != dict(expected_provenance):
        raise ValueError("V19 checkpoint provenance differs")
    contract = dict(payload.get("model_contract") or {})
    expected_contract = {
        "feature_dim": V19_FEATURE_DIM,
        "class_count": V19_CLASS_COUNT,
        "bias": False,
        "zero_sum": True,
        "residual_bound": V19_RESIDUAL_BOUND,
        "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
    }
    if contract != expected_contract:
        raise ValueError("V19 checkpoint model contract differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def require_exact_parent_order(
    actual: Sequence[str], expected_sha256: str, *, epoch: int
) -> str:
    digest = hashlib.sha256()
    for parent_id in actual:
        encoded = str(parent_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    value = digest.hexdigest()
    if value != str(expected_sha256):
        raise ValueError(
            f"V19 epoch {epoch} parent order differs from frozen V18 Control: "
            f"actual={value}, expected={expected_sha256}"
        )
    return value


__all__ = [
    "MicroDopplerLateResidual",
    "ResidualCacheDataset",
    "V19_ARTIFACT_RELATIVE_PATH",
    "V19_CACHE_FIELDS",
    "V19_CACHE_SCHEMA_VERSION",
    "V19_CHECKPOINT_SCHEMA_VERSION",
    "V19_CLASS_COUNT",
    "V19_CLASS_NAMES",
    "V19_CONFIG_NAME",
    "V19_CONFIG_RELATIVE_PATH",
    "V19_CONTRACT_RELATIVE_PATH",
    "V19_FEATURE_DIM",
    "V19_NORMALIZATION_RELATIVE_PATH",
    "V19_READINESS_RELATIVE_PATH",
    "V19_REPORT_RELATIVE_PATH",
    "V19_RESIDUAL_BOUND",
    "V19_SCHEMA_VERSION",
    "V19_TRAINABLE_PARAMETER_COUNT",
    "V19_TRAIN_CACHE_RELATIVE_PATH",
    "V19_VALIDATION_CACHE_RELATIVE_PATH",
    "build_residual_dataloader",
    "canonical_json_sha256",
    "causal_valid_mean",
    "evaluate_residual",
    "load_residual_checkpoint",
    "normalize_stable8",
    "require_exact_parent_order",
    "residual_objective",
    "save_residual_checkpoint",
    "sha256_file",
    "strip_evaluation_records",
    "tensor_state_sha256",
]
