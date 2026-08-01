from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch

from .data.rd_segment_store import resolve_window_rd_time_sync_status
from .encoding import encode_track_sequence, prepare_track_sequence_features
from .features import TRACK_FEATURE_COLUMNS, track_sequence_continuous_feature_indices
from .expert1_config import Expert1TrainConfig
from .experts import LOW_ALTITUDE_CLASSES
from .physical import extract_physical_features


SAFE_RD_SYNC_STATUSES = {"exact_time_sync", "within_tolerance"}


@dataclass(frozen=True)
class TrajectoryEvent:
    target_id: str
    event_time_ns: int
    arrival_time_ns: int
    sequence_id: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RDFrameEvent:
    target_id: str
    event_time_ns: int
    arrival_time_ns: int
    sequence_id: int
    sync_status: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalWindow:
    target_id: str
    window_start_ns: int
    window_end_ns: int
    tr_events: tuple[TrajectoryEvent, ...]
    rd_events: tuple[RDFrameEvent, ...]
    rd_time_sync_status: str
    rd_fusion_eligible: bool
    decision_state: str
    window_mode: str


@dataclass(frozen=True)
class StreamingWindowConfig:
    window_duration_ns: int = 10_000_000_000
    update_interval_ns: int = 2_000_000_000
    max_lateness_ns: int = 0
    min_tr_points: int = 5
    min_unique_event_times: int = 2
    prefix_stride_points: int = 5
    max_tr_events_per_target: int = 2_048
    max_rd_events_per_target: int = 2_048
    target_retention_ns: int = 30_000_000_000


@dataclass(frozen=True)
class StableDecisionConfig:
    min_confirmations: int = 3
    switch_confirmations: int = 2
    min_switch_margin: float = 0.05


@dataclass(frozen=True)
class StreamingDecision:
    target_id: str
    decision_time_ns: int
    predicted_label: str
    confidence: float
    decision_state: str
    is_stable: bool
    used_fallback: bool
    first_decision_latency_ns: int | None
    stable_decision_latency_ns: int | None
    prediction_flip_count: int
    candidate_label: str
    candidate_count: int
    rd_time_sync_status: str


@dataclass(frozen=True)
class StreamingInferenceResult:
    window: CausalWindow
    decision: StreamingDecision
    model_version: str
    checkpoint_id: str
    preprocess_latency_ns: int
    model_latency_ns: int
    postprocess_latency_ns: int

    @property
    def total_latency_ns(self) -> int:
        return int(
            self.preprocess_latency_ns
            + self.model_latency_ns
            + self.postprocess_latency_ns
        )


@dataclass(frozen=True)
class StreamingModelPrediction:
    probabilities: dict[str, float]
    used_fallback: bool = False


@dataclass
class _DecisionTargetState:
    first_window_start_ns: int | None = None
    first_decision_time_ns: int | None = None
    stable_decision_time_ns: int | None = None
    stable_label: str | None = None
    stable_confidence: float = 0.0
    candidate_label: str | None = None
    candidate_count: int = 0
    last_emitted_label: str | None = None
    prediction_flip_count: int = 0


class Expert1StableDecisionStateMachine:
    """Deterministic online confirmation and switch control for window logits."""

    def __init__(self, config: StableDecisionConfig | None = None) -> None:
        self.config = config or StableDecisionConfig()
        if self.config.min_confirmations < 1:
            raise ValueError("min_confirmations must be positive")
        if self.config.switch_confirmations < 1:
            raise ValueError("switch_confirmations must be positive")
        if not 0.0 <= self.config.min_switch_margin <= 1.0:
            raise ValueError("min_switch_margin must be within [0, 1]")
        self._targets: dict[str, _DecisionTargetState] = {}

    def update(
        self,
        window: CausalWindow,
        probabilities: dict[str, float],
        *,
        force_fallback: bool = False,
    ) -> StreamingDecision:
        if not probabilities:
            raise ValueError("probabilities must be non-empty")
        normalized = self._normalized_probabilities(probabilities)
        predicted_label, confidence = max(
            normalized.items(),
            key=lambda item: (item[1], item[0]),
        )
        target_id = str(window.target_id)
        state = self._targets.setdefault(target_id, _DecisionTargetState())
        if state.first_window_start_ns is None:
            state.first_window_start_ns = int(window.window_start_ns)
        if state.first_decision_time_ns is None:
            state.first_decision_time_ns = int(window.window_end_ns)
        if state.candidate_label == predicted_label:
            state.candidate_count += 1
        else:
            state.candidate_label = predicted_label
            state.candidate_count = 1

        required_count = (
            self.config.min_confirmations
            if state.stable_label is None or predicted_label == state.stable_label
            else self.config.switch_confirmations
        )
        candidate_ready = state.candidate_count >= required_count
        stable_label_probability = normalized.get(state.stable_label or "", 0.0)
        if state.stable_label is None and candidate_ready:
            state.stable_label = predicted_label
            state.stable_confidence = confidence
            state.stable_decision_time_ns = int(window.window_end_ns)
        elif (
            state.stable_label is not None
            and predicted_label != state.stable_label
            and candidate_ready
            and confidence >= stable_label_probability + self.config.min_switch_margin
        ):
            state.stable_label = predicted_label
            state.stable_confidence = confidence
            state.stable_decision_time_ns = int(window.window_end_ns)
        elif predicted_label == state.stable_label:
            state.stable_confidence = confidence

        emitted_label = state.stable_label or predicted_label
        if state.last_emitted_label is not None and state.last_emitted_label != emitted_label:
            state.prediction_flip_count += 1
        state.last_emitted_label = emitted_label
        used_fallback = bool(force_fallback or not window.rd_fusion_eligible)
        if used_fallback:
            decision_state = "fallback"
        elif state.stable_label is not None:
            decision_state = "stable"
        else:
            decision_state = "provisional"
        first_latency = (
            int(state.first_decision_time_ns - state.first_window_start_ns)
            if state.first_decision_time_ns is not None and state.first_window_start_ns is not None
            else None
        )
        stable_latency = (
            int(state.stable_decision_time_ns - state.first_window_start_ns)
            if state.stable_decision_time_ns is not None and state.first_window_start_ns is not None
            else None
        )
        return StreamingDecision(
            target_id=target_id,
            decision_time_ns=int(window.window_end_ns),
            predicted_label=emitted_label,
            confidence=float(confidence),
            decision_state=decision_state,
            is_stable=state.stable_label is not None,
            used_fallback=used_fallback,
            first_decision_latency_ns=first_latency,
            stable_decision_latency_ns=stable_latency,
            prediction_flip_count=state.prediction_flip_count,
            candidate_label=state.candidate_label or predicted_label,
            candidate_count=state.candidate_count,
            rd_time_sync_status=window.rd_time_sync_status,
        )

    def end_target(self, target_id: str, decision_time_ns: int) -> StreamingDecision | None:
        state = self._targets.get(str(target_id))
        if state is None or state.last_emitted_label is None:
            return None
        first_latency = (
            int(state.first_decision_time_ns - state.first_window_start_ns)
            if state.first_decision_time_ns is not None and state.first_window_start_ns is not None
            else None
        )
        stable_latency = (
            int(state.stable_decision_time_ns - state.first_window_start_ns)
            if state.stable_decision_time_ns is not None and state.first_window_start_ns is not None
            else None
        )
        return StreamingDecision(
            target_id=str(target_id),
            decision_time_ns=int(decision_time_ns),
            predicted_label=state.last_emitted_label,
            confidence=float(state.stable_confidence),
            decision_state="ended",
            is_stable=state.stable_label is not None,
            used_fallback=False,
            first_decision_latency_ns=first_latency,
            stable_decision_latency_ns=stable_latency,
            prediction_flip_count=state.prediction_flip_count,
            candidate_label=state.candidate_label or state.last_emitted_label,
            candidate_count=state.candidate_count,
            rd_time_sync_status="unavailable",
        )

    def reassociate(self, old_target_id: str, new_target_id: str) -> None:
        old_target_id = str(old_target_id)
        new_target_id = str(new_target_id)
        if old_target_id == new_target_id:
            return
        state = self._targets.pop(old_target_id, None)
        if state is None:
            return
        if new_target_id in self._targets:
            raise ValueError("new target already has decision state")
        self._targets[new_target_id] = state

    @staticmethod
    def _normalized_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
        cleaned = {
            str(label): max(0.0, float(value))
            for label, value in probabilities.items()
        }
        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("probabilities must contain a positive finite value")
        return {label: value / total for label, value in cleaned.items()}


class Expert1StreamingInferenceEngine:
    """Connect causal windows, a predictor callback, and stable decisions."""

    def __init__(
        self,
        predictor: Callable[[Any], dict[str, float] | StreamingModelPrediction],
        *,
        preprocess: Callable[[CausalWindow], Any] | None = None,
        window_config: StreamingWindowConfig | None = None,
        decision_config: StableDecisionConfig | None = None,
        model_version: str = "unknown",
        checkpoint_id: str = "unknown",
    ) -> None:
        self._predictor = predictor
        self._preprocess = preprocess or (lambda window: window)
        self.windows = Expert1StreamingWindowEngine(window_config)
        self.decisions = Expert1StableDecisionStateMachine(decision_config)
        self.model_version = str(model_version)
        self.checkpoint_id = str(checkpoint_id)

    def create_target(self, target_id: str) -> None:
        self.windows.create_target(target_id)

    def ingest_trajectory(self, event: TrajectoryEvent) -> list[StreamingInferenceResult]:
        return self._infer_windows(self.windows.ingest_trajectory(event))

    def ingest_rd(self, event: RDFrameEvent) -> list[StreamingInferenceResult]:
        return self._infer_windows(self.windows.ingest_rd(event))

    def end_target(self, target_id: str, event_time_ns: int) -> list[StreamingInferenceResult | StreamingDecision]:
        results: list[StreamingInferenceResult | StreamingDecision] = list(
            self._infer_windows(self.windows.end_target(target_id, event_time_ns))
        )
        ended = self.decisions.end_target(target_id, event_time_ns)
        if ended is not None:
            results.append(ended)
        return results

    def reassociate(self, old_target_id: str, new_target_id: str) -> None:
        self.decisions.reassociate(old_target_id, new_target_id)
        old_state = self.windows._targets.pop(str(old_target_id), None)
        if old_state is None:
            return
        if str(new_target_id) in self.windows._targets:
            raise ValueError("new target already has streaming state")
        self.windows._targets[str(new_target_id)] = old_state

    def target_stats(self, target_id: str) -> dict[str, int | None]:
        return self.windows.target_stats(target_id)

    def collect_expired_targets(self, now_ns: int) -> list[str]:
        expired = self.windows.collect_expired_targets(now_ns)
        for target_id in expired:
            self.decisions._targets.pop(target_id, None)
        return expired

    def _infer_windows(self, windows: list[CausalWindow]) -> list[StreamingInferenceResult]:
        results: list[StreamingInferenceResult] = []
        for window in windows:
            preprocess_start = perf_counter_ns()
            prepared = self._preprocess(window)
            preprocess_latency_ns = perf_counter_ns() - preprocess_start
            model_start = perf_counter_ns()
            prediction = self._predictor(prepared)
            model_latency_ns = perf_counter_ns() - model_start
            postprocess_start = perf_counter_ns()
            if isinstance(prediction, StreamingModelPrediction):
                probabilities = prediction.probabilities
                force_fallback = bool(prediction.used_fallback)
            else:
                probabilities = prediction
                force_fallback = False
            decision = self.decisions.update(
                window,
                probabilities,
                force_fallback=force_fallback,
            )
            postprocess_latency_ns = perf_counter_ns() - postprocess_start
            results.append(
                StreamingInferenceResult(
                    window=window,
                    decision=decision,
                    model_version=self.model_version,
                    checkpoint_id=self.checkpoint_id,
                    preprocess_latency_ns=int(preprocess_latency_ns),
                    model_latency_ns=int(model_latency_ns),
                    postprocess_latency_ns=int(postprocess_latency_ns),
                )
            )
        return results


def _max_event_gap_seconds(event_times_ns: Iterable[int]) -> float:
    ordered = sorted({int(value) for value in event_times_ns})
    if len(ordered) < 2:
        return 0.0
    return float(max(abs(right - left) for left, right in zip(ordered[1:], ordered[:-1])) / 1_000_000_000.0)


def _raw_track_numeric_finite_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    values = frame[TRACK_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return float(np.isfinite(values).mean()) if values.size else 0.0


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean trajectory metadata value: {value!r}")


def _streaming_source_name(
    payloads: list[dict[str, Any]],
    *,
    source_normalization: bool,
    source_feature_stats: dict[str, object],
) -> str:
    values = [str(payload.get("source_name", "")).strip() for payload in payloads]
    registered = {value for value in values if value}
    if len(registered) > 1:
        raise ValueError("causal window mixes trajectory sources")
    source_name = next(iter(registered), "")
    if source_normalization:
        if not source_name or any(not value for value in values):
            raise ValueError("source_normalization requires source_name on every trajectory event")
        if source_name not in source_feature_stats:
            raise ValueError(f"source_normalization has no registered statistics for source={source_name}")
    return source_name


def _streaming_height_missing(payloads: list[dict[str, Any]], frame: pd.DataFrame) -> bool:
    metadata = [
        _coerce_optional_bool(payload["height_missing"])
        for payload in payloads
        if "height_missing" in payload
    ]
    if metadata:
        if len(metadata) != len(payloads) or len(set(metadata)) != 1:
            raise ValueError("height_missing metadata must be present and consistent on every trajectory event")
        return bool(metadata[0])
    if "height_m" not in frame.columns:
        return True
    values = pd.to_numeric(frame["height_m"], errors="coerce").to_numpy(dtype=np.float64)
    return not bool(np.isfinite(values).any())


class Expert1CheckpointTrajectoryPredictor:
    """Run a saved Expert1 checkpoint through its validated trajectory-only path.

    RD fusion is deliberately disabled here. The current assets provide exact RD
    time synchronization for only a tiny raw-source subset, so treating this
    adapter as a trajectory fallback prevents unsafe CQ08 RD fusion.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        raw_config = checkpoint.get("config", {})
        allowed_config_keys = {item.name for item in fields(Expert1TrainConfig)}
        config_values = (
            {key: value for key, value in raw_config.items() if key in allowed_config_keys}
            if isinstance(raw_config, dict)
            else {}
        )
        self.config = Expert1TrainConfig(**config_values)

        from .expert1_trainer import (
            _apply_source_physical_norm,
            _build_expert1_modules,
            _load_checkpoint_into_modules,
            _track_attribution_quality,
        )

        self._track_attribution_quality = _track_attribution_quality
        self._apply_source_physical_norm = _apply_source_physical_norm

        self.representation_builder, self.rd_proj, self.model = _build_expert1_modules(
            self.config,
            self.device,
        )
        loaded_checkpoint = _load_checkpoint_into_modules(
            self.checkpoint_path,
            self.representation_builder,
            self.rd_proj,
            self.model,
            self.device,
        )
        self.representation_builder.eval()
        self.rd_proj.eval()
        self.model.eval()
        source_feature_stats = loaded_checkpoint.get("source_feature_stats", {})
        self.source_feature_stats = source_feature_stats if isinstance(source_feature_stats, dict) else {}
        if self.config.physical_representation_contract_version == "source_robust_v3":
            missing_physical = [
                source_name
                for source_name, values in self.source_feature_stats.items()
                if not isinstance(values, dict)
                or "physical_center" not in values
                or "physical_scale" not in values
            ]
            if missing_physical:
                raise ValueError(
                    "source_robust_v3 streaming checkpoint is missing physical source stats: "
                    f"sources={sorted(missing_physical)[:5]}"
                )
        self.model_version = str(self.config.output_dir or "expert1_checkpoint")
        self.checkpoint_id = self.checkpoint_path.as_posix()

    def __call__(self, window: CausalWindow) -> StreamingModelPrediction:
        payloads = [dict(event.payload) for event in window.tr_events]
        if not payloads:
            raise ValueError("causal window contains no trajectory payloads")
        source_name = _streaming_source_name(
            payloads,
            source_normalization=bool(self.config.source_normalization),
            source_feature_stats=self.source_feature_stats,
        )
        frame = pd.DataFrame(payloads)
        raw_numeric_finite_ratio = _raw_track_numeric_finite_ratio(frame)
        encoded = encode_track_sequence(frame, feature_schema=self.config.trajectory_feature_schema)
        prepared_features = prepare_track_sequence_features(
            encoded.features,
            source_name=source_name,
            feature_schema=self.config.trajectory_feature_schema,
        )
        sequence_features = torch.from_numpy(prepared_features).to(self.device).unsqueeze(0)
        if self.config.source_normalization and source_name in self.source_feature_stats:
            stats = self.source_feature_stats[source_name]
            center = stats["center"].to(self.device, dtype=sequence_features.dtype)
            scale = stats["scale"].to(self.device, dtype=sequence_features.dtype).clamp_min(1.0e-3)
            indices = track_sequence_continuous_feature_indices(self.config.trajectory_feature_schema)
            if center.numel() != sequence_features.shape[-1] or scale.numel() != sequence_features.shape[-1]:
                raise ValueError(
                    "streaming source feature statistics dimension does not match trajectory features: "
                    f"features={sequence_features.shape[-1]}, center={center.numel()}, scale={scale.numel()}"
                )
            if indices:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
                sequence_features[:, :, index_tensor] = (
                    sequence_features[:, :, index_tensor] - center[index_tensor]
                ) / scale[index_tensor]
        height_missing = _streaming_height_missing(payloads, frame)
        physical_features = torch.from_numpy(
            extract_physical_features(
                frame,
                height_missing=height_missing,
                phase_missing=False,
                feature_schema=self.config.trajectory_feature_schema,
            ).values
        ).to(self.device).unsqueeze(0)
        if self.config.physical_representation_contract_version == "source_robust_v3":
            physical_features = self._apply_source_physical_norm(
                physical_features,
                [source_name],
                self.source_feature_stats,
                self.config.trajectory_feature_schema,
            )
        trajectory_length = int(sequence_features.shape[1])
        trajectory_padding_mask = torch.zeros(
            (1, trajectory_length),
            dtype=torch.bool,
            device=self.device,
        )
        max_time_gap_seconds = torch.tensor(
            _max_event_gap_seconds(event.event_time_ns for event in window.tr_events),
            dtype=torch.float32,
            device=self.device,
        )
        duration_seconds = max(1.0e-3, (int(window.window_end_ns) - int(window.window_start_ns)) / 1_000_000_000.0)
        track_quality = self._track_attribution_quality(
            torch.tensor([trajectory_length], dtype=torch.long, device=self.device),
            torch.zeros((1,), dtype=torch.float32, device=self.device),
            torch.tensor([duration_seconds], dtype=torch.float32, device=self.device),
            max_time_gap_seconds.reshape(1),
            torch.tensor([raw_numeric_finite_ratio], dtype=torch.float32, device=self.device),
            self.config,
        )
        rd_features = torch.zeros((1, 1, 512, 4), dtype=torch.float32, device=self.device)
        rd_patch_features = torch.zeros((1, 1, 64, 64, 3), dtype=torch.float32, device=self.device)
        rd_is_multimodal = torch.zeros((1,), dtype=torch.bool, device=self.device)
        rd_padding_mask = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        rd_summary = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            representation = self.representation_builder(
                sequence_features=sequence_features,
                physical_features=physical_features,
                padding_mask=trajectory_padding_mask,
            )
            output = self.model(
                representation.fused_features,
                rd_features=rd_features,
                traj_sequence_features=representation.sequence_features,
                rd_patch_features=rd_patch_features,
                rd_deep_patch_features=rd_patch_features,
                rd_context=self.rd_proj(rd_summary),
                rd_is_multimodal=rd_is_multimodal,
                rd_alignment_quality=torch.zeros((1,), dtype=torch.float32, device=self.device),
                traj_padding_mask=trajectory_padding_mask,
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_padding_mask,
                rd_deep_patch_padding_mask=rd_padding_mask,
                traj_physical_features=representation.physical_features,
                track_quality=track_quality,
                enable_reject=bool(self.config.enable_reject_in_eval),
                disable_tr_rd_fusion=True,
            )
            values = torch.softmax(output.logits[0], dim=-1).detach().cpu().tolist()
        return StreamingModelPrediction(
            probabilities={label: float(value) for label, value in zip(LOW_ALTITUDE_CLASSES, values)},
            used_fallback=True,
        )


def summarize_streaming_decisions(
    decisions: Iterable[tuple[StreamingDecision, str]],
) -> dict[str, object]:
    """Summarize target-level online behavior against fixed replay labels."""
    grouped: dict[str, list[tuple[StreamingDecision, str]]] = {}
    for decision, true_label in decisions:
        if decision.decision_state == "ended":
            continue
        grouped.setdefault(str(decision.target_id), []).append((decision, str(true_label)))
    target_records: list[dict[str, object]] = []
    for target_id, values in grouped.items():
        values.sort(key=lambda item: int(item[0].decision_time_ns))
        labels = {label for _decision, label in values}
        if len(labels) != 1:
            raise ValueError(f"target has conflicting replay labels: {target_id}")
        true_label = next(iter(labels))
        final_decision = values[-1][0]
        first_correct = next(
            (decision for decision, _label in values if decision.predicted_label == true_label),
            None,
        )
        stable_correct = next(
            (
                decision
                for decision, _label in values
                if decision.is_stable and decision.predicted_label == true_label
            ),
            None,
        )
        target_records.append(
            {
                "target_id": target_id,
                "true_label": true_label,
                "final_predicted_label": final_decision.predicted_label,
                "final_correct": final_decision.predicted_label == true_label,
                "window_decision_count": len(values),
                "prediction_flip_count": final_decision.prediction_flip_count,
                "used_fallback": any(decision.used_fallback for decision, _label in values),
                "first_correct_latency_ns": (
                    first_correct.first_decision_latency_ns if first_correct is not None else None
                ),
                "stable_correct_latency_ns": (
                    stable_correct.stable_decision_latency_ns if stable_correct is not None else None
                ),
            }
        )
    if not target_records:
        return {"target_count": 0, "decision_count": 0}
    labels = sorted({str(record["true_label"]) for record in target_records})
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        records = [record for record in target_records if record["true_label"] == label]
        support = len(records)
        correct = sum(bool(record["final_correct"]) for record in records)
        per_class[label] = {
            "support": support,
            "final_recall": float(correct / max(1, support)),
        }
    first_latencies = [
        int(record["first_correct_latency_ns"])
        for record in target_records
        if record["first_correct_latency_ns"] is not None
    ]
    stable_latencies = [
        int(record["stable_correct_latency_ns"])
        for record in target_records
        if record["stable_correct_latency_ns"] is not None
    ]
    flip_counts = [int(record["prediction_flip_count"]) for record in target_records]
    correct_count = sum(bool(record["final_correct"]) for record in target_records)
    return {
        "target_count": len(target_records),
        "decision_count": int(sum(int(record["window_decision_count"]) for record in target_records)),
        "final_target_accuracy": float(correct_count / len(target_records)),
        "fallback_target_count": int(sum(bool(record["used_fallback"]) for record in target_records)),
        "flipped_target_count": int(sum(count > 0 for count in flip_counts)),
        "flipped_target_rate": float(sum(count > 0 for count in flip_counts) / len(target_records)),
        "prediction_flip_count_mean": float(sum(flip_counts) / len(flip_counts)),
        "first_correct_target_count": len(first_latencies),
        "first_correct_latency_ns_p50": _quantile(first_latencies, 0.50),
        "stable_correct_target_count": len(stable_latencies),
        "stable_correct_latency_ns_p50": _quantile(stable_latencies, 0.50),
        "per_class": per_class,
        "target_records": target_records,
    }


def _quantile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


@dataclass
class _TargetState:
    tr_events: list[TrajectoryEvent] = field(default_factory=list)
    rd_events: list[RDFrameEvent] = field(default_factory=list)
    max_event_time_ns: int | None = None
    first_event_time_ns: int | None = None
    watermark_ns: int | None = None
    last_window_end_ns: int | None = None
    last_committed_end_ns: int | None = None
    trajectory_event_count: int = 0
    pending_prefix_ends_ns: list[int] = field(default_factory=list)
    dropped_late_events: int = 0
    dropped_buffer_events: int = 0
    ended_at_ns: int | None = None


def _event_order(event: TrajectoryEvent | RDFrameEvent) -> tuple[int, int, int]:
    return int(event.event_time_ns), int(event.sequence_id), int(event.arrival_time_ns)


class Expert1StreamingWindowEngine:
    """Build bounded, event-time causal Expert1 windows for one or more targets."""

    def __init__(self, config: StreamingWindowConfig | None = None) -> None:
        self.config = config or StreamingWindowConfig()
        if self.config.window_duration_ns <= 0:
            raise ValueError("window_duration_ns must be positive")
        if self.config.update_interval_ns <= 0:
            raise ValueError("update_interval_ns must be positive")
        if self.config.min_tr_points < 1:
            raise ValueError("min_tr_points must be positive")
        if self.config.min_unique_event_times < 1:
            raise ValueError("min_unique_event_times must be positive")
        self._targets: dict[str, _TargetState] = {}

    def create_target(self, target_id: str) -> None:
        self._targets.setdefault(str(target_id), _TargetState())

    def end_target(self, target_id: str, event_time_ns: int) -> list[CausalWindow]:
        state = self._targets.get(str(target_id))
        if state is None:
            return []
        state.ended_at_ns = int(event_time_ns)
        return self._advance(str(target_id), state, force=True)

    def ingest_trajectory(self, event: TrajectoryEvent) -> list[CausalWindow]:
        state = self._targets.setdefault(str(event.target_id), _TargetState())
        if not self._accept_event(state, event):
            return []
        state.tr_events.append(event)
        state.tr_events.sort(key=_event_order)
        state.trajectory_event_count += 1
        self._queue_completed_prefix_end(state)
        self._enforce_buffer_limits(state)
        return self._advance(str(event.target_id), state)

    def ingest_rd(self, event: RDFrameEvent) -> list[CausalWindow]:
        state = self._targets.setdefault(str(event.target_id), _TargetState())
        if not self._accept_event(state, event):
            return []
        state.rd_events.append(event)
        state.rd_events.sort(key=_event_order)
        self._enforce_buffer_limits(state)
        return self._advance(str(event.target_id), state)

    def target_stats(self, target_id: str) -> dict[str, int | None]:
        state = self._targets.get(str(target_id))
        if state is None:
            return {
                "tr_buffer_size": 0,
                "rd_buffer_size": 0,
                "watermark_ns": None,
                "last_window_end_ns": None,
                "dropped_late_events": 0,
                "dropped_buffer_events": 0,
            }
        return {
            "tr_buffer_size": len(state.tr_events),
            "rd_buffer_size": len(state.rd_events),
            "watermark_ns": state.watermark_ns,
            "last_window_end_ns": state.last_window_end_ns,
            "dropped_late_events": state.dropped_late_events,
            "dropped_buffer_events": state.dropped_buffer_events,
        }

    def collect_expired_targets(self, now_ns: int) -> list[str]:
        expired = [
            target_id
            for target_id, state in self._targets.items()
            if state.ended_at_ns is not None
            and int(now_ns) - int(state.ended_at_ns) >= self.config.target_retention_ns
        ]
        for target_id in expired:
            self._targets.pop(target_id, None)
        return sorted(expired)

    def _accept_event(self, state: _TargetState, event: TrajectoryEvent | RDFrameEvent) -> bool:
        event_time_ns = int(event.event_time_ns)
        if state.last_committed_end_ns is not None and event_time_ns < state.last_committed_end_ns:
            state.dropped_late_events += 1
            return False
        if state.max_event_time_ns is None or event_time_ns > state.max_event_time_ns:
            state.max_event_time_ns = event_time_ns
        if state.first_event_time_ns is None or event_time_ns < state.first_event_time_ns:
            state.first_event_time_ns = event_time_ns
        state.watermark_ns = int(state.max_event_time_ns - self.config.max_lateness_ns)
        return True

    def _trim_events(self, state: _TargetState) -> None:
        watermark_ns = state.watermark_ns
        if watermark_ns is not None:
            keep_after_ns = int(watermark_ns - self.config.window_duration_ns)
            state.tr_events = [event for event in state.tr_events if event.event_time_ns >= keep_after_ns]
            state.rd_events = [event for event in state.rd_events if event.event_time_ns >= keep_after_ns]
        self._enforce_buffer_limits(state)

    def _queue_completed_prefix_end(self, state: _TargetState) -> None:
        """Mirror offline point-prefix generation once a later point proves it is not the tail."""
        candidate_index = state.trajectory_event_count - 2
        first_prefix_index = self.config.min_tr_points - 1
        if candidate_index < first_prefix_index:
            return
        if (candidate_index - first_prefix_index) % self.config.prefix_stride_points != 0:
            return
        if len(state.tr_events) < 2:
            return
        candidate_end_ns = int(state.tr_events[-2].event_time_ns)
        first_event_time_ns = state.first_event_time_ns
        if first_event_time_ns is None or candidate_end_ns >= (
            first_event_time_ns + self.config.window_duration_ns
        ):
            return
        if candidate_end_ns not in state.pending_prefix_ends_ns:
            state.pending_prefix_ends_ns.append(candidate_end_ns)

    def _enforce_buffer_limits(self, state: _TargetState) -> None:
        for events, limit in (
            (state.tr_events, self.config.max_tr_events_per_target),
            (state.rd_events, self.config.max_rd_events_per_target),
        ):
            overflow = max(0, len(events) - int(limit))
            if overflow:
                del events[:overflow]
                state.dropped_buffer_events += overflow

    def _advance(self, target_id: str, state: _TargetState, force: bool = False) -> list[CausalWindow]:
        watermark_ns = state.watermark_ns
        if watermark_ns is None and not force:
            return []
        if force:
            if state.max_event_time_ns is None:
                return []
            watermark_ns = int(state.max_event_time_ns)
            state.watermark_ns = watermark_ns
        if watermark_ns is None:
            return []
        if state.last_committed_end_ns is not None and watermark_ns <= state.last_committed_end_ns:
            return []
        first_event_time_ns = int(
            state.first_event_time_ns
            if state.first_event_time_ns is not None
            else watermark_ns
        )
        candidate_specs: dict[int, str] = {}
        committed_end_ns = state.last_committed_end_ns
        next_grid_end = first_event_time_ns + self.config.update_interval_ns
        while next_grid_end <= watermark_ns:
            if committed_end_ns is None or next_grid_end > committed_end_ns:
                mode = "prefix" if next_grid_end < first_event_time_ns + self.config.window_duration_ns else "sliding"
                candidate_specs[int(next_grid_end)] = mode
            next_grid_end += self.config.update_interval_ns
        for prefix_end_ns in state.pending_prefix_ends_ns:
            if (committed_end_ns is None or prefix_end_ns > committed_end_ns) and prefix_end_ns <= watermark_ns:
                candidate_specs[int(prefix_end_ns)] = "prefix"
        if force:
            final_end_ns = int(watermark_ns)
            if committed_end_ns is None or final_end_ns > committed_end_ns:
                if final_end_ns <= first_event_time_ns + self.config.window_duration_ns:
                    candidate_specs[final_end_ns] = "short_full"
                else:
                    regular_end_ns = first_event_time_ns + self.config.window_duration_ns
                    while regular_end_ns + self.config.update_interval_ns <= final_end_ns:
                        regular_end_ns += self.config.update_interval_ns
                    if final_end_ns - regular_end_ns > self.config.update_interval_ns // 2:
                        candidate_specs[final_end_ns] = "tail"
        state.pending_prefix_ends_ns = [
            value for value in state.pending_prefix_ends_ns if value > watermark_ns
        ]
        windows: list[CausalWindow] = []
        for window_end_ns, window_mode in sorted(candidate_specs.items()):
            if window_mode in {"prefix", "short_full"}:
                window_start_ns = first_event_time_ns
            else:
                window_start_ns = int(window_end_ns - self.config.window_duration_ns)
            tr_events = tuple(
                event
                for event in state.tr_events
                if window_start_ns <= event.event_time_ns <= window_end_ns
            )
            unique_times = {event.event_time_ns for event in tr_events}
            if len(tr_events) < self.config.min_tr_points or len(unique_times) < self.config.min_unique_event_times:
                continue
            rd_events = tuple(
                event
                for event in state.rd_events
                if window_start_ns <= event.event_time_ns <= window_end_ns
            )
            if rd_events:
                status = resolve_window_rd_time_sync_status(
                    dict(Counter(event.sync_status for event in rd_events))
                )
            else:
                status = "missing"
            fusion_eligible = status in SAFE_RD_SYNC_STATUSES
            windows.append(
                CausalWindow(
                    target_id=target_id,
                    window_start_ns=window_start_ns,
                    window_end_ns=window_end_ns,
                    tr_events=tr_events,
                    rd_events=rd_events if fusion_eligible else (),
                    rd_time_sync_status=status,
                    rd_fusion_eligible=fusion_eligible,
                    decision_state="provisional" if fusion_eligible else "fallback",
                    window_mode=window_mode,
                )
            )
            state.last_window_end_ns = window_end_ns
            state.last_committed_end_ns = window_end_ns
        self._trim_events(state)
        return windows


__all__ = [
    "CausalWindow",
    "Expert1CheckpointTrajectoryPredictor",
    "Expert1StreamingWindowEngine",
    "Expert1StreamingInferenceEngine",
    "Expert1StableDecisionStateMachine",
    "RDFrameEvent",
    "StreamingWindowConfig",
    "StreamingDecision",
    "StreamingInferenceResult",
    "StreamingModelPrediction",
    "StableDecisionConfig",
    "TrajectoryEvent",
    "summarize_streaming_decisions",
]
