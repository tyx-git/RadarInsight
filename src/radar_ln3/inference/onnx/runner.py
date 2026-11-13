"""ONNX Runtime runner for the full cascade graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_ln3.model.io import LEGACY_LV28_BATCH_ALIASES, LV28_INPUT_NAMES

from .cascade import CASCADE_ACTION_NAMES, CASCADE_REASON_NAMES, FULL_CASCADE_INPUT_NAMES


# Experimental LV28 was exported before the current full-cascade input
# contract was finalized.  Keep its graph usable as a compatibility endpoint;
# the product batch adapter below supplies the same tensors under these names.


class ONNXCascadeRunner:
    """Run a full-cascade ONNX artifact with DirectML -> CPU fallback."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        prefer_directml: bool = True,
        intra_op_num_threads: int | None = 1,
        inter_op_num_threads: int | None = 1,
        graph_optimization_level: str = "disable",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime 未安装") from exc
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        available = tuple(ort.get_available_providers())
        providers: list[str] = []
        if prefer_directml:
            directml_provider = next(
                (
                    name
                    for name in ("DmlExecutionProvider", "DirectMLExecutionProvider")
                    if name in available
                ),
                None,
            )
            if directml_provider is not None:
                providers.append(directml_provider)
        providers.append("CPUExecutionProvider")
        options: dict[str, Any] = {}
        if intra_op_num_threads is not None:
            options["intra_op_num_threads"] = int(intra_op_num_threads)
        if inter_op_num_threads is not None:
            options["inter_op_num_threads"] = int(inter_op_num_threads)
        optimization_levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        if graph_optimization_level not in optimization_levels:
            raise ValueError(f"不支持的 ONNX graph_optimization_level：{graph_optimization_level}")
        options["graph_optimization_level"] = optimization_levels[graph_optimization_level]
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
                sess_options=self._session_options(ort, options),
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
                sess_options=self._session_options(ort, options),
            )
        self.providers = tuple(self.session.get_providers())
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_names = tuple(item.name for item in self.session.get_outputs())
        if self.input_names == FULL_CASCADE_INPUT_NAMES:
            self.contract = "RADAR-LN3-ONNX-FULL-CASCADE-V1"
        elif self.input_names == LV28_INPUT_NAMES:
            self.contract = "RADAR-LN3-ONNX-LV28-EXPERIMENTAL"
        else:
            raise ValueError(
                "ONNX cascade input contract 不一致："
                f"expected={FULL_CASCADE_INPUT_NAMES} or {LV28_INPUT_NAMES}, "
                f"actual={self.input_names}"
            )

    @staticmethod
    def _session_options(ort: Any, values: Mapping[str, Any]) -> Any:
        options = ort.SessionOptions()
        for name, value in values.items():
            setattr(options, name, value)
        return options

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.session.get_modelmeta().custom_metadata_map)

    @property
    def input_contract(self) -> str:
        return self.metadata.get("contract", self.contract)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _feed_for_lv28(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        aliases = LEGACY_LV28_BATCH_ALIASES
        missing = [source for source in aliases.values() if source not in batch]
        if missing:
            raise ValueError(f"LV28 ONNX 输入缺少特征：{missing}")
        return {
            target: self._to_numpy(batch[source])
            for target, source in aliases.items()
        }

    def predict(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray | list[str]]:
        if self.input_names == LV28_INPUT_NAMES:
            feed = self._feed_for_lv28(batch)
        else:
            missing = [name for name in FULL_CASCADE_INPUT_NAMES if name not in batch]
            if missing:
                raise ValueError(f"ONNX full cascade 缺少输入：{missing}")
            feed = {
                name: self._to_numpy(batch[name])
                for name in FULL_CASCADE_INPUT_NAMES
            }
        outputs = self.session.run(list(self.output_names), feed)
        result = {
            name: np.asarray(value)
            for name, value in zip(self.output_names, outputs, strict=True)
        }
        if "action_ids" in result:
            result["actions"] = np.asarray(
                [CASCADE_ACTION_NAMES[int(value)] for value in result["action_ids"]]
            )
        if "reason_ids" in result:
            result["reasons"] = np.asarray(
                [CASCADE_REASON_NAMES[int(value)] for value in result["reason_ids"]]
            )
        return result


__all__ = ["LV28_INPUT_NAMES", "ONNXCascadeRunner"]
