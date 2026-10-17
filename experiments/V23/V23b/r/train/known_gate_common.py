from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import KNOWN_GATE_SELECTION_METRIC








def _parent_order_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _capture_training_state(
    generator: torch.Generator,
    *,
    run_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator_state": generator.get_state(),
        "run_state": dict(run_state),
    }


def _restore_training_state(
    payload: Mapping[str, Any],
    generator: torch.Generator,
) -> dict[str, Any]:
    state = payload.get("training_state")
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "run_state",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("V21 checkpoint lacks complete epoch-boundary recovery state")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(state["torch_cuda_rng_states"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("V21 checkpoint CUDA RNG state count differs from this host")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, device="cpu") for value in cuda_states])
    generator.set_state(torch.as_tensor(state["dataloader_generator_state"], device="cpu"))
    run_state = state["run_state"]
    if not isinstance(run_state, Mapping):
        raise ValueError("V21 checkpoint training_state lacks run_state")
    return dict(run_state)




















def _selection(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(evaluation.get("parent_metrics") or {})
    rows = list(metrics.get("per_class") or [])
    if len(rows) != 2 or any(int(row.get("support", 0)) <= 0 for row in rows):
        raise ValueError("V21 checkpoint selection requires two non-zero-support classes")
    weak = min(
        float(row[field])
        for row in rows
        for field in ("precision", "recall")
    )
    macro_f1 = float(metrics["macro_f1"])
    if not np.isfinite(weak) or not np.isfinite(macro_f1):
        raise ValueError("V21 checkpoint selection metric is non-finite")
    return {
        "metric": KNOWN_GATE_SELECTION_METRIC,
        "primary_metric": "parent_min_class_precision_recall",
        "primary_value": weak,
        "tie_break_metric": "parent_macro_f1",
        "tie_break_value": macro_f1,
        "ranking_key": [weak, macro_f1],
    }


def _is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return tuple(float(value) for value in candidate["ranking_key"]) > tuple(
        float(value) for value in incumbent["ranking_key"]
    )








def _model_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tr_features": tensors["tr_features"],
        "tr_missing_mask": tensors["tr_missing_mask"],
        "tr_dt_seconds": tensors["tr_dt_seconds"],
        "tr_dt_valid": tensors["tr_dt_valid"],
        "tr_dt_reason": tensors["tr_dt_reason"],
        "rd_frames": tensors["rd_frames"],
        "rd_frame_mask": tensors["rd_frame_mask"],
        "rd_quality": tensors["rd_quality"],
        "tr_padding_mask": tensors["tr_padding_mask"],
        "known_gate_features": tensors.get("known_gate_features"),
        "known_gate_feature_valid": tensors.get("known_gate_feature_valid"),
    }


def _performance(evaluation: Mapping[str, Any], target: float) -> dict[str, Any]:
    rows = list(dict(evaluation.get("parent_metrics") or {}).get("per_class") or [])
    guardrails: list[dict[str, Any]] = []
    values: list[float] = []
    for row in rows:
        metrics = {field: float(row[field]) for field in ("precision", "recall", "f1")}
        values.extend(metrics.values())
        guardrails.append(
            {
                "class": str(row["class"]),
                "support": int(row["support"]),
                **metrics,
                "target": float(target),
                "passed": all(value >= target for value in metrics.values()),
            }
        )
    passed = bool(guardrails) and all(row["passed"] for row in guardrails)
    return {
        "performance_status": "passed" if passed else "failed",
        "target_min_per_class_precision_recall_f1": float(target),
        "observed_min_per_class_precision_recall_f1": min(values) if values else None,
        "guardrails": guardrails,
    }





