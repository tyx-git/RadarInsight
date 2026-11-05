"""Radar LN3 formal package.

Checkpoint loading is exposed lazily so data-only deployments, including the
R-RTSP replay server, do not import the optional PyTorch dependency merely by
importing ``radar_ln3.data``.
"""

__all__ = ["Checkpoint", "CheckpointError"]


def __getattr__(name: str):
    if name in __all__:
        from .checkpoint import Checkpoint, CheckpointError

        return {"Checkpoint": Checkpoint, "CheckpointError": CheckpointError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
