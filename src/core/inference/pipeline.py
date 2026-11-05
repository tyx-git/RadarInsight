"""Offline/online product inference pipeline.

The frontend owns transport. This module owns package adaptation, canonical
RD conversion, shared feature batching and the full three-layer endpoint.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from radar_ln3.data.asd import load_asd
from radar_ln3.data.contracts import EventData, EventSequence
from radar_ln3.data.temporal import annotate_event_data
from radar_ln3.features import build_model_batch
from radar_ln3.inference.onnx.cascade import build_full_cascade_model
from radar_ln3.inference.onnx.runner import ONNXCascadeRunner


class ProductInferencePipeline:
    """Run a complete L1/L2/L3 cascade from an `.asd` package."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        binding_path: str | Path = "src/config/radar_ln3_l3.json",
        device: str = "cpu",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.device = torch.device(device)
        binding = self._read_binding(binding_path)
        self.model, self.binding = build_full_cascade_model(
            self.project_root,
            l1_checkpoint_path=binding["l1_checkpoint_path"],
            l2_checkpoint_path=binding["l2_checkpoint_path"],
            l1_knownness_head_path=binding["l1_knownness_head_path"],
            l2_knownness_head_path=binding["l2_knownness_head_path"],
            calibration_path=binding["l3_calibration_path"],
            model_config_path=binding["model_config_path"],
            device=self.device,
        )
        self.model.eval()

    def _read_binding(self, path: str | Path) -> dict[str, Any]:
        import json
        import hashlib

        target = Path(path)
        if not target.is_absolute():
            target = self.project_root / target
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("cascade binding 必须是 JSON 对象")
        if value.get("schema_version") not in {"RADAR-LN3-L3-CONFIG", "RADAR-LN3-L3-CONFIG-V1"}:
            raise ValueError("L3 config schema 不受支持")
        if str(value.get("status") or "") == "needs_refit":
            raise RuntimeError("L3 config 已绑定新的 L1/L2 endpoint，但 knownness head 和 calibration 尚未重新拟合")
        for field in (
            "model_config_path",
            "l1_checkpoint_path",
            "l2_checkpoint_path",
            "l1_knownness_head_path",
            "l2_knownness_head_path",
            "l3_calibration_path",
        ):
            target = Path(value[field])
            if not target.is_absolute():
                target = self.project_root / target
            try:
                target.resolve().relative_to(self.project_root)
            except ValueError as exc:
                raise ValueError(f"radar_ln3 binding path 越过项目根目录：{target}") from exc
            if not target.is_file():
                raise FileNotFoundError(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            expected = value.get(field.replace("_path", "_sha256"))
            if expected is not None and digest != str(expected).lower():
                raise ValueError(f"radar_ln3 binding hash 不匹配：{field}")
        return value

    def sequence_batch(
        self,
        sequences: Any | Sequence[Any],
    ) -> dict[str, torch.Tensor]:
        from radar_ln3.model import load_model_config

        if isinstance(sequences, (list, tuple)):
            sequence_list = list(sequences)
        else:
            sequence_list = [sequences]
        if not sequence_list:
            raise ValueError("模型 batch 不能为空")

        config = load_model_config(self.project_root)
        l1 = build_model_batch(
            sequence_list,
            task="l1",
            max_rd_frames=config.data.max_rd_frames_per_point,
            max_sequence_length=config.data.max_sequence_length,
            pad_to_max_sequence_length=True,
            allow_unmapped_labels=True,
        )
        l2 = build_model_batch(
            sequence_list,
            task="l2",
            max_rd_frames=config.data.max_rd_frames_per_point,
            max_sequence_length=config.data.max_sequence_length,
            stable_median=np.asarray(config.data.l2.stable8_median, dtype=np.float32),
            stable_scale=np.asarray(config.data.l2.stable8_scale, dtype=np.float32),
            stable_indices=config.data.l2.stable8_indices,
            clipping={
                "bounds": {
                    name: {"lower": bounds[0], "upper": bounds[1]}
                    for name, bounds in config.data.l2.tr_clipping_bounds.items()
                }
            },
            stable_input_mode="normalized",
            pad_to_max_sequence_length=True,
            allow_unmapped_labels=True,
        )
        names = (
            "tr_features",
            "tr_missing_mask",
            "tr_dt_seconds",
            "tr_dt_valid",
            "tr_dt_reason",
            "tr_padding_mask",
            "rd_frames",
            "rd_frame_mask",
            "rd_quality",
            "rd_physical_features",
            "rd_physical_valid",
            "micro_doppler_features",
            "micro_doppler_valid",
            "course_features",
            "course_valid",
            "course_support",
            "observation_increment_values",
            "reference_tr_features",
            "reference_tr_missing_mask",
            "stable_summary_tokens",
            "stable_summary_valid",
            "stable_frame_values",
            "stable_frame_mask",
            "motion_values",
        )
        batch = {
            name: (l1[name] if name in {"course_features", "course_valid", "course_support", "observation_increment_values"} else l2[name])
            for name in names
        }
        # LV28.onnx predates the current full-cascade graph contract and keeps
        # separate L1/L2 track and RD tensors.  Preserve both branches in the
        # shared product batch so the runner can feed either graph version.
        batch.update({
            "l1_tr_features": l1["tr_features"],
            "l2_tr_features": l2["tr_features"],
            "l1_rd_frames": l1["rd_frames"],
            "l2_rd_frames": l2["rd_frames"],
        })
        return {name: value.to(device=self.device) for name, value in batch.items()}

    @torch.inference_mode()
    def predict_sequence(self, sequence: Any) -> dict[str, Any]:
        return self.predict_sequences([sequence])[0]

    @torch.inference_mode()
    def predict_sequences(self, sequences: Sequence[Any]) -> list[dict[str, Any]]:
        sequence_list = list(sequences)
        if not sequence_list:
            raise ValueError("模型 batch 不能为空")
        batch = self.sequence_batch(sequence_list)
        outputs = self.model(*[batch[name] for name in self.model_input_names])
        from radar_ln3.inference.onnx.cascade import CASCADE_ACTION_NAMES, CASCADE_REASON_NAMES

        results = []
        for row, sequence in enumerate(sequence_list):
            action = int(outputs[-2][row].item())
            reason = int(outputs[-1][row].item())
            results.append({
                "parent_sample_id": sequence.parent_sample_id,
                "label_id": sequence.label_id,
                "label_name": sequence.label_name,
                "timestamp_unit": sequence.metadata.get("timestamp_unit"),
                "timestamp_unit_assumed": bool(sequence.metadata.get("timestamp_unit_assumed", False)),
                "event_count": len(sequence.events),
                "model_input_event_count": int(
                    batch["tr_padding_mask"][row].logical_not().sum().item()
                ),
                "sequence_truncated": len(sequence.events) > int(batch["tr_padding_mask"].shape[1]),
                "l1_probabilities": outputs[0][row].detach().cpu().tolist(),
                "l2_probabilities": outputs[1][row].detach().cpu().tolist(),
                "cascade_probabilities": outputs[2][row].detach().cpu().tolist(),
                "rd_available": bool(batch["rd_frame_mask"][row].any().item()),
                "stable8_available": bool(batch["stable_summary_valid"][row].any().item()),
                "micro_doppler_available": bool(batch["micro_doppler_valid"][row].any().item()),
                "action_id": action,
                "action": CASCADE_ACTION_NAMES[action],
                "reason_id": reason,
                "reason_code": CASCADE_REASON_NAMES[reason],
                "binding": self.binding,
            })
        return results

    @property
    def model_input_names(self) -> tuple[str, ...]:
        from radar_ln3.inference.onnx.cascade import FULL_CASCADE_INPUT_NAMES

        return FULL_CASCADE_INPUT_NAMES

    def predict_asd(
        self,
        path: str | Path,
        *,
        timestamp_unit: str | None = None,
        amplitude_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.predict_sequence(
            load_asd(path, timestamp_unit=timestamp_unit, amplitude_mode=amplitude_mode)
        )


class ONNXProductInferencePipeline(ProductInferencePipeline):
    """Same product preprocessing backed by the full-cascade ONNX artifact."""

    def __init__(
        self,
        project_root: str | Path,
        model_path: str | Path,
        *,
        prefer_directml: bool = True,
        binding_path: str | Path = "src/config/radar_ln3_l3.json",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.device = torch.device("cpu")
        self.runner = ONNXCascadeRunner(model_path, prefer_directml=prefer_directml)
        self.binding = self.runner.metadata
        self.binding_path = str(binding_path)

    def predict_sequence(self, sequence: Any) -> dict[str, Any]:
        return self.predict_sequences([sequence])[0]

    def predict_sequences(self, sequences: Sequence[Any]) -> list[dict[str, Any]]:
        sequence_list = list(sequences)
        if not sequence_list:
            raise ValueError("模型 batch 不能为空")
        batch = self.sequence_batch(sequence_list)
        outputs = self.runner.predict(batch)
        actions = outputs.get("actions")
        reasons = outputs.get("reasons")
        results = []
        for row, sequence in enumerate(sequence_list):
            action = str(actions[row]) if actions is not None else str(int(outputs["action_ids"][row]))
            reason = str(reasons[row]) if reasons is not None else str(int(outputs["reason_ids"][row]))
            results.append({
                "parent_sample_id": sequence.parent_sample_id,
                "label_id": sequence.label_id,
                "label_name": sequence.label_name,
                "timestamp_unit": sequence.metadata.get("timestamp_unit"),
                "timestamp_unit_assumed": bool(sequence.metadata.get("timestamp_unit_assumed", False)),
                "event_count": len(sequence.events),
                "model_input_event_count": int(
                    batch["tr_padding_mask"][row].logical_not().sum().item()
                ),
                "sequence_truncated": len(sequence.events) > int(batch["tr_padding_mask"].shape[1]),
                "l1_probabilities": outputs["l1_probabilities"][row].tolist(),
                "l2_probabilities": outputs["l2_probabilities"][row].tolist(),
                "cascade_probabilities": outputs["cascade_probabilities"][row].tolist(),
                "rd_available": bool(batch["rd_frame_mask"][row].any().item()),
                "stable8_available": bool(batch["stable_summary_valid"][row].any().item()),
                "micro_doppler_available": bool(batch["micro_doppler_valid"][row].any().item()),
                "action": action,
                "reason_code": reason,
                "providers": list(self.runner.providers),
                "binding": self.binding,
            })
        return results


__all__ = ["ONNXProductInferencePipeline", "ProductInferencePipeline"]


class OnlineInferenceSession:
    """Transport-neutral causal session for a frontend-owned protocol.

    The session keeps raw EventData and replays the current causal prefix
    through the same product pipeline. Event 30 is rejected by design because
    the current model contract is fixed at 29 events.
    """

    def __init__(
        self,
        pipeline: ProductInferencePipeline,
        *,
        parent_sample_id: str,
        track_id: str,
        label_id: int | None = None,
        label_name: str = "UnknownTarget",
    ) -> None:
        self.pipeline = pipeline
        self.parent_sample_id = str(parent_sample_id)
        self.track_id = str(track_id)
        self.label_id = label_id
        self.label_name = str(label_name)
        self.events: list[EventData] = []

    def push(self, event: EventData) -> dict[str, Any]:
        sequence, expected = self.append_event(event, allow_sliding=False)
        result = self.pipeline.predict_sequence(sequence)
        result["event_index"] = expected
        result["is_final"] = expected == 28
        return result

    def append_event(
        self,
        event: EventData,
        *,
        allow_sliding: bool,
    ) -> tuple[EventSequence, int]:
        """Append one event and return its 29-event model window.

        Causal mode keeps the historical hard limit. Queue mode retains the
        source history for continuity but exposes only the most recent 29
        events to the product model.
        """

        expected = len(self.events)
        if int(event.causal_event_index) != expected:
            raise ValueError(
                f"event_index 不连续：expected={expected}, actual={event.causal_event_index}"
            )
        if not allow_sliding and expected >= 29:
            raise ValueError("EVENT_LIMIT_EXCEEDED: 当前在线模型最多接受 29 个事件")
        if str(event.parent_sample_id) != self.parent_sample_id:
            raise ValueError("PARENT_MISMATCH")
        self.events.append(event)
        self.events = annotate_event_data(self.events)
        window = self.events[-29:] if allow_sliding else self.events
        return EventSequence(
            parent_sample_id=self.parent_sample_id,
            track_id=self.track_id,
            label_id=self.label_id,
            label_name=self.label_name,
            split_name="inference",
            events=list(window),
            source_name="online",
        ), expected


__all__ = [
    "ONNXProductInferencePipeline",
    "OnlineInferenceSession",
    "ProductInferencePipeline",
]
