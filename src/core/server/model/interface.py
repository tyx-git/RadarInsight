"""Stable backend interface for future model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Model backend used by an application or a future result transport.

    RRTSP uses separate PT=96/97 observation and optional PT=98 result tracks.
    Keeping model calls behind this protocol lets future ONNX artifacts replace
    the current cascade without changing the media session implementation.
    """

    @property
    def model_name(self) -> str:
        ...

    @property
    def input_contract(self) -> str:
        ...

    @property
    def metadata(self) -> Mapping[str, str]:
        ...

    @property
    def providers(self) -> Sequence[str]:
        ...

    def readiness(self) -> dict[str, Any]:
        ...

    def predict_sequence(self, sequence: Any) -> dict[str, Any]:
        ...


@runtime_checkable
class StreamingInferenceSession(Protocol):
    """Per-viewer causal predictor consumed by the RRTSP media session."""

    def predict(self, event: Any) -> Mapping[str, Any]:
        ...


@runtime_checkable
class StreamingInferenceFactory(Protocol):
    """Create independent causal state over a shared inference backend."""

    def __call__(self) -> StreamingInferenceSession:
        ...


__all__ = ["InferenceBackend", "StreamingInferenceFactory", "StreamingInferenceSession"]
