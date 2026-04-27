from __future__ import annotations

import torch


def resolve_cuda_device(requested_device: str = "cuda") -> tuple[torch.device, dict[str, object]]:
    normalized = str(requested_device or "cuda").strip().lower()
    if normalized == "cpu":
        raise RuntimeError("CPU training is disabled for this project; set device to cuda or cuda:N.")
    if not normalized.startswith("cuda"):
        raise RuntimeError(f"unsupported training device '{requested_device}'; only cuda/cuda:N is allowed.")
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError("CUDA training was requested but torch.cuda.is_available() is false.")
    device = torch.device(normalized)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested CUDA device {device.index} is unavailable; "
            f"visible CUDA device count is {torch.cuda.device_count()}."
        )
    active_index = 0 if device.index is None else int(device.index)
    info: dict[str, object] = {
        "requested_device": requested_device,
        "resolved_device": str(device),
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(active_index),
        "cpu_training_allowed": False,
    }
    return device, info


__all__ = ["resolve_cuda_device"]
