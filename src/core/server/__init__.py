"""RadarInsight backend server components.

The R-RTSP server publishes track/RD observations for AeroSight.  Inference is
kept behind ``core.server.model`` so future model artifacts can be introduced
without changing the transport implementation.
"""

from .model import (
    InferenceBackend,
    ONNXProductBackend,
    ONNXRuntimeModel,
    StreamingInferenceFactory,
    StreamingInferenceSession,
)
from .rrtsp import RRTSPServer, RRTSPServerConfig
from .source import (
    ASDDirectoryStreamSource,
    ASDStreamSource,
    MAX_PRODUCT_EVENTS,
    MODEL_CONTEXT_EVENTS,
    QUEUE_MAX_EVENTS,
    RRTSPEvent,
    RRTSPRDFrame,
    StreamSource,
)

__all__ = [
    "ASDDirectoryStreamSource",
    "ASDStreamSource",
    "FullCascadeInferenceFactory",
    "FullCascadeQueueInferenceFactory",
    "FullCascadePackageInferenceFactory",
    "InferenceBackend",
    "ONNXProductBackend",
    "ONNXRuntimeModel",
    "MAX_PRODUCT_EVENTS",
    "MODEL_CONTEXT_EVENTS",
    "QUEUE_MAX_EVENTS",
    "RRTSPEvent",
    "RRTSPRDFrame",
    "RRTSPServer",
    "RRTSPServerConfig",
    "StreamSource",
    "StreamingInferenceFactory",
    "StreamingInferenceSession",
]


def __getattr__(name: str):
    if name in {
        "FullCascadeInferenceFactory",
        "FullCascadeQueueInferenceFactory",
        "FullCascadePackageInferenceFactory",
    }:
        from .model import __getattr__ as model_getattr

        return model_getattr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
