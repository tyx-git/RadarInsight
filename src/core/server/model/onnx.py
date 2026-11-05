"""ONNX Runtime interfaces.

``ONNXRuntimeModel`` is a generic tensor-level adapter for future ONNX
artifacts. ``ONNXProductBackend`` binds the current full-cascade graph to the
shared ASD/feature preprocessing pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ONNXRuntimeModel:
    """Generic ONNX tensor runner with DirectML-to-CPU fallback."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        prefer_directml: bool = True,
        graph_optimization_level: str = "disable",
        intra_op_num_threads: int | None = 1,
        inter_op_num_threads: int | None = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed") from exc
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        optimization_levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        if graph_optimization_level not in optimization_levels:
            raise ValueError(f"unsupported graph optimization level: {graph_optimization_level}")
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization_levels[graph_optimization_level]
        if intra_op_num_threads is not None:
            options.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads is not None:
            options.inter_op_num_threads = int(inter_op_num_threads)
        available = tuple(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if prefer_directml and "DirectMLExecutionProvider" in available:
            providers.insert(0, "DirectMLExecutionProvider")
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
                sess_options=options,
            )
        except Exception:
            session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
        self.session = session
        self.providers = tuple(session.get_providers())
        self.input_names = tuple(item.name for item in session.get_inputs())
        self.output_names = tuple(item.name for item in session.get_outputs())

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.session.get_modelmeta().custom_metadata_map)

    @property
    def model_name(self) -> str:
        return self.metadata.get("model_name", self.model_path.stem)

    @property
    def input_contract(self) -> str:
        return self.metadata.get("contract", self.metadata.get("input_spec_version", ""))

    def readiness(self) -> dict[str, Any]:
        metadata = self.metadata
        formal_available = str(metadata.get("formal_available", "false")).lower() == "true"
        return {
            "ready": True,
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "input_contract": self.input_contract,
            "providers": list(self.providers),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "formal_available": formal_available,
            "metadata": metadata,
        }

    def predict(self, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise ValueError(f"ONNX inputs missing: {missing}")
        feed = {name: _to_numpy(inputs[name]) for name in self.input_names}
        values = self.session.run(list(self.output_names), feed)
        return {
            name: np.asarray(value)
            for name, value in zip(self.output_names, values, strict=True)
        }


class ONNXProductBackend:
    """Current full-cascade product backend behind the stable interface."""

    def __init__(
        self,
        project_root: str | Path,
        model_path: str | Path,
        *,
        prefer_directml: bool = True,
        binding_path: str | Path = "src/config/radar_ln3_l3.json",
    ) -> None:
        # Import lazily so an R-RTSP-only deployment does not require torch.
        from core.inference.pipeline import ONNXProductInferencePipeline

        self.pipeline = ONNXProductInferencePipeline(
            project_root,
            model_path,
            prefer_directml=prefer_directml,
            binding_path=binding_path,
        )

    @property
    def runner(self):
        return self.pipeline.runner

    @property
    def model_name(self) -> str:
        metadata = self.metadata
        return metadata.get("model_name", Path(self.runner.model_path).stem)

    @property
    def input_contract(self) -> str:
        return self.runner.input_contract

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.pipeline.binding)

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self.runner.providers)

    def readiness(self) -> dict[str, Any]:
        metadata = self.metadata
        return {
            "ready": True,
            "model_path": str(self.runner.model_path),
            "model_name": self.model_name,
            "input_contract": self.input_contract,
            "providers": list(self.providers),
            "input_names": list(self.runner.input_names),
            "output_names": list(self.runner.output_names),
            "formal_available": str(metadata.get("formal_available", "false")).lower() == "true",
            "metadata": metadata,
        }

    def predict_sequence(self, sequence: Any) -> dict[str, Any]:
        return self.pipeline.predict_sequence(sequence)

    def predict_asd(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return self.pipeline.predict_asd(path, **kwargs)


__all__ = ["ONNXProductBackend", "ONNXRuntimeModel"]
