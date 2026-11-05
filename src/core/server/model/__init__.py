"""Model interfaces kept independent from the R-RTSP transport."""

from .interface import (
    InferenceBackend,
    StreamingInferenceFactory,
    StreamingInferenceSession,
)
from .onnx import ONNXProductBackend, ONNXRuntimeModel

__all__ = [
    "FullCascadeInferenceFactory",
    "FullCascadeQueueInferenceFactory",
    "FullCascadePackageInferenceFactory",
    "InferenceBackend",
    "ONNXProductBackend",
    "ONNXRuntimeModel",
    "StreamingInferenceFactory",
    "StreamingInferenceSession",
]


def __getattr__(name: str):
    if name in {
        "FullCascadeInferenceFactory",
        "FullCascadeQueueInferenceFactory",
        "FullCascadePackageInferenceFactory",
    }:
        from . import streaming

        return getattr(streaming, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
