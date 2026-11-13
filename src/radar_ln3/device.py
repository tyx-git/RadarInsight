"""设备解析：把 CLI / 配置里的 device 文本统一成可执行的 ``torch.device``。"""

from __future__ import annotations

import torch


def resolve_device(value: str | torch.device) -> torch.device:
    """规范化设备：``cpu`` / ``cuda`` / ``cuda:N`` / ``torch.device``。

    任意 cuda 目标都会先检查可用性，避免把错误推迟到第一次 ``.to(device)``。
    """

    if isinstance(value, torch.device):
        target = value
    else:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("device 不能为空")
        if text == "cpu":
            target = torch.device("cpu")
        elif text == "cuda":
            target = torch.device("cuda:0")
        elif text.startswith("cuda:"):
            index = text.removeprefix("cuda:")
            if not index.isdigit():
                raise ValueError(f"不支持的 device：{value}")
            target = torch.device(f"cuda:{int(index)}")
        else:
            raise ValueError(f"不支持的 device：{value}")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用")
    return target


__all__ = ["resolve_device"]
