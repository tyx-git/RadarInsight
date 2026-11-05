"""Causal Full Cascade sessions for streaming ASD events."""

from __future__ import annotations

from threading import RLock
from typing import Any

from core.inference.pipeline import OnlineInferenceSession
from radar_ln3.inference.onnx.cascade import CASCADE_ACTION_NAMES


class FullCascadeInferenceFactory:
    """Build independent online histories over a shared product ONNX pipeline."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._inference_lock = RLock()

    def __call__(self) -> "FullCascadeInferenceSession":
        return FullCascadeInferenceSession(self)


class FullCascadeInferenceSession:
    def __init__(self, factory: FullCascadeInferenceFactory) -> None:
        self.factory = factory
        # Directory replay interleaves events from multiple targets in the
        # same radar scan. Keep one causal history per parent instead of
        # replacing the previous target's history on every event.
        self._online_by_parent: dict[str, OnlineInferenceSession] = {}

    def predict(self, event: Any) -> dict[str, Any]:
        model_event = getattr(event, "model_event", None)
        if model_event is None:
            raise ValueError("stream event has no model-facing EventData")
        parent_sample_id = str(model_event.parent_sample_id)
        # A looping replay starts the same parent again at causal index 0.
        # Start a fresh history for that new cycle instead of appending to the
        # previous cycle's sequence.
        online = self._online_by_parent.get(parent_sample_id)
        if online is None or int(model_event.causal_event_index) == 0:
            online = OnlineInferenceSession(
                self.factory.pipeline,
                parent_sample_id=parent_sample_id,
                track_id=str(model_event.track_id),
                label_id=4,
                label_name="UnknownTarget",
            )
            self._online_by_parent[parent_sample_id] = online
        with self.factory._inference_lock:
            result = online.push(model_event)
        action = str(result["action"])
        try:
            action_id = CASCADE_ACTION_NAMES.index(action)
        except ValueError as exc:
            raise ValueError(f"unsupported cascade action: {action}") from exc
        probabilities = result.get("cascade_probabilities") or []
        confidence = float(probabilities[action_id]) if action_id < len(probabilities) else 0.0
        return {
            "track_id": int(event.track_id),
            "class_id": action_id + 1,
            "confidence": confidence,
            "action": action,
            "reason_code": str(result.get("reason_code", "")),
            "event_index": int(result["event_index"]),
            "is_final": bool(
                result["is_final"]
                or int(model_event.causal_event_index) + 1
                >= int(event.metadata.get("parent_event_count", 29))
            ),
        }


class FullCascadeQueueInferenceFactory:
    """Create a batched sliding-window predictor for queue replay mode."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._inference_lock = RLock()

    def __call__(self) -> "FullCascadeQueueInferenceSession":
        return FullCascadeQueueInferenceSession(self)


class FullCascadeQueueInferenceSession:
    """Infer one next point for up to ten targets in a single model call."""

    def __init__(self, factory: FullCascadeQueueInferenceFactory) -> None:
        self.factory = factory
        self._online_by_parent: dict[str, OnlineInferenceSession] = {}

    def predict_batch(self, events: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
        if not events:
            return []
        sequences = []
        prepared: list[tuple[Any, int]] = []
        for event in events:
            model_event = getattr(event, "model_event", None)
            if model_event is None:
                raise ValueError("queue event has no model-facing EventData")
            parent_sample_id = str(model_event.parent_sample_id)
            online = self._online_by_parent.get(parent_sample_id)
            if online is None or int(model_event.causal_event_index) == 0:
                online = OnlineInferenceSession(
                    self.factory.pipeline,
                    parent_sample_id=parent_sample_id,
                    track_id=str(model_event.track_id),
                    label_id=4,
                    label_name="UnknownTarget",
                )
                self._online_by_parent[parent_sample_id] = online
            sequence, event_index = online.append_event(
                model_event,
                allow_sliding=True,
            )
            sequences.append(sequence)
            prepared.append((event, event_index))

        with self.factory._inference_lock:
            results = self.factory.pipeline.predict_sequences(sequences)
        if len(results) != len(prepared):
            raise ValueError(
                f"queue inference batch size mismatch: expected={len(prepared)}, "
                f"actual={len(results)}"
            )

        converted: list[dict[str, Any]] = []
        for (event, event_index), result in zip(prepared, results, strict=True):
            action = str(result.get("action", ""))
            if not action:
                action_id = int(result.get("action_id", 0))
                action = CASCADE_ACTION_NAMES[action_id]
            try:
                action_id = CASCADE_ACTION_NAMES.index(action)
            except ValueError as exc:
                raise ValueError(f"unsupported cascade action: {action}") from exc
            probabilities = result.get("cascade_probabilities") or []
            confidence = (
                float(probabilities[action_id])
                if action_id < len(probabilities)
                else 0.0
            )
            converted.append({
                "track_id": int(event.track_id),
                "class_id": action_id + 1,
                "confidence": confidence,
                "action": action,
                "reason_code": str(result.get("reason_code", "")),
                "event_index": int(event_index),
                "is_final": bool(
                    int(event_index) + 1
                    >= int(event.metadata.get("parent_event_count", 29))
                ),
            })
        return converted


class FullCascadePackageInferenceFactory:
    """Classify each complete ASD parent once and reuse it during replay."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._inference_lock = RLock()
        self._decisions: dict[str, dict[str, Any]] = {}

    def __call__(self) -> "FullCascadePackageInferenceSession":
        return FullCascadePackageInferenceSession(self)


class FullCascadePackageInferenceSession:
    def __init__(self, factory: FullCascadePackageInferenceFactory) -> None:
        self.factory = factory

    def predict(self, event: Any) -> dict[str, Any]:
        parent_sample_id = str(event.parent_sample_id)
        with self.factory._inference_lock:
            decision = self.factory._decisions.get(parent_sample_id)
            if decision is None:
                sequence = getattr(event, "model_sequence", None)
                if sequence is None:
                    raise ValueError("stream event has no model-facing EventSequence")
                result = self.factory.pipeline.predict_sequence(sequence)
                action = str(result["action"])
                try:
                    action_id = CASCADE_ACTION_NAMES.index(action)
                except ValueError as exc:
                    raise ValueError(f"unsupported cascade action: {action}") from exc
                probabilities = result.get("cascade_probabilities") or []
                confidence = (
                    float(probabilities[action_id])
                    if action_id < len(probabilities)
                    else 0.0
                )
                decision = {
                    "class_id": action_id + 1,
                    "confidence": confidence,
                    "action": action,
                    "reason_code": str(result.get("reason_code", "")),
                }
                self.factory._decisions[parent_sample_id] = decision
        return {
            "track_id": int(event.track_id),
            **decision,
            "event_index": int(event.event_index),
            "is_final": True,
        }


__all__ = [
    "FullCascadeInferenceFactory",
    "FullCascadeInferenceSession",
    "FullCascadeQueueInferenceFactory",
    "FullCascadeQueueInferenceSession",
    "FullCascadePackageInferenceFactory",
    "FullCascadePackageInferenceSession",
]
