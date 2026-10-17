"""V27b: baseline-conditioned stable8 residual classification experiment."""

from .protocol import (
    V27B_EXPERIMENT_ID,
    V27B_L2_CLASS_NAMES,
    V27B_SCHEMA_VERSION,
)
from .model import V27BConditionalPairResidual, build_v27b_model

__all__ = [
    "V27B_EXPERIMENT_ID",
    "V27B_L2_CLASS_NAMES",
    "V27B_SCHEMA_VERSION",
    "V27BConditionalPairResidual",
    "build_v27b_model",
]
