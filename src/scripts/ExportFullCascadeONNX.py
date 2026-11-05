"""Export the complete L1/L2/L3 soft-cascade ONNX endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from radar_ln3.inference.onnx.cascade import (  # noqa: E402
    CASCADE_ACTION_NAMES,
    CASCADE_REASON_NAMES,
    FULL_CASCADE_INPUT_NAMES,
    build_full_cascade_model,
)


CONTRACT = "RADAR-LN3-ONNX-FULL-CASCADE-V1"
DEFAULT_BINDING = Path("src/config/radar_ln3_l3.json")
DEFAULT_OUTPUT = Path("artifacts/radar_ln3/onnx/radarln3.onnx")
OUTPUT_NAMES = (
    "l1_probabilities",
    "l2_probabilities",
    "cascade_probabilities",
    "action_ids",
    "reason_ids",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export RadarInsight full cascade ONNX")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--binding-path", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--dummy-batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("full ONNX 导出需要 torch、onnx、onnxruntime 和 numpy") from exc
    return torch, np, onnx, ort


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _dummy_inputs(config: Any, *, batch: int, torch: Any) -> tuple[Any, ...]:
    length = int(config.data.max_sequence_length)
    frames = int(config.data.max_rd_frames_per_point)
    dtype = torch.float32
    values: dict[str, Any] = {
        "tr_features": torch.zeros((batch, length, config.model.tr_value_dim), dtype=dtype),
        "tr_missing_mask": torch.zeros((batch, length, config.model.tr_missing_dim), dtype=dtype),
        "tr_dt_seconds": torch.zeros((batch, length), dtype=dtype),
        "tr_dt_valid": torch.zeros((batch, length), dtype=torch.bool),
        "tr_dt_reason": torch.ones((batch, length), dtype=torch.long),
        "tr_padding_mask": torch.zeros((batch, length), dtype=torch.bool),
        "rd_frames": torch.zeros((batch, length, frames, 2, config.model.rd_height, config.model.rd_width), dtype=dtype),
        "rd_frame_mask": torch.zeros((batch, length, frames), dtype=torch.bool),
        "rd_quality": torch.zeros((batch, length, config.model.rd_quality_dim), dtype=dtype),
        "rd_physical_features": torch.zeros((batch, length, config.data.l2.rd_physical_dim), dtype=dtype),
        "rd_physical_valid": torch.zeros((batch, length), dtype=torch.bool),
        "micro_doppler_features": torch.zeros((batch, length, config.data.l2.micro_doppler_dim), dtype=dtype),
        "micro_doppler_valid": torch.zeros((batch, length), dtype=torch.bool),
        "course_features": torch.zeros((batch, length, 1), dtype=dtype),
        "course_valid": torch.zeros((batch, length, 1), dtype=torch.bool),
        "course_support": torch.zeros((batch, length), dtype=torch.long),
        "observation_increment_values": torch.zeros((batch, length, config.data.motion_dim), dtype=dtype),
        "reference_tr_features": torch.zeros((batch, length, config.model.tr_value_dim), dtype=dtype),
        "reference_tr_missing_mask": torch.zeros((batch, length, config.model.tr_missing_dim), dtype=dtype),
        "stable_summary_tokens": torch.zeros((batch, length, config.data.stable8_dim), dtype=dtype),
        "stable_summary_valid": torch.zeros((batch, length), dtype=torch.bool),
        "stable_frame_values": torch.zeros((batch, length, frames, config.data.stable8_dim), dtype=dtype),
        "stable_frame_mask": torch.zeros((batch, length, frames), dtype=torch.bool),
        "motion_values": torch.zeros((batch, length, config.data.motion_dim), dtype=dtype),
    }
    # Keep one active event in the dummy graph so final-event gathers are
    # exercised during export validation.
    values["tr_padding_mask"][:, 1:] = True
    values["tr_dt_reason"][:, 0] = 1
    values["tr_dt_reason"][:, 1:] = 0
    values["rd_quality"][:, 0, 0] = 1.0
    values["rd_frame_mask"][:, 0, 0] = True
    values["rd_frames"][:, 0, 0, 0] = 55.90625
    values["course_valid"][:, 0, 0] = True
    values["rd_physical_valid"][:, 0] = True
    values["micro_doppler_valid"][:, 0] = True
    return tuple(values[name] for name in FULL_CASCADE_INPUT_NAMES)


def _metadata(binding: dict[str, Any], config: Any, *, opset: int) -> dict[str, str]:
    body = {
        "contract": CONTRACT,
        "model_name": "radar-ln3-full-soft-cascade",
        "model_version": "radar_ln3-independent-head-v1",
        "input_spec_version": CONTRACT,
        "input_names": list(FULL_CASCADE_INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "input_sequence_length": int(config.data.max_sequence_length),
        "rd_shape": [int(config.model.rd_height), int(config.model.rd_width)],
        "rd_max_frames": int(config.data.max_rd_frames_per_point),
        "l1_classes": ["KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE"],
        "l2_classes": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "cascade_classes": ["DroneTarget", "BirdTarget", "BalloonTarget", "Clutter", "UNKNOWN_REJECTED", "PENDING_ABSTAIN"],
        "action_names": list(CASCADE_ACTION_NAMES),
        "reason_names": list(CASCADE_REASON_NAMES),
        "stateful_streaming": False,
        "formal_available": True,
        "binding": binding,
        "opset_version": int(opset),
    }
    return {
        key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, (dict, list, bool, int, float))
        else str(value)
        for key, value in body.items()
    }


def export(
    *,
    project_root: Path,
    binding_path: Path,
    output_path: Path,
    opset_version: int,
    dummy_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    torch, np, onnx, ort = _load_runtime()
    if opset_version < 17:
        raise ValueError("opset_version 必须 >= 17")
    if dummy_batch_size <= 0:
        raise ValueError("dummy_batch_size 必须为正整数")
    root = project_root.resolve()
    binding = json.loads(_resolve(root, binding_path).read_text(encoding="utf-8"))
    if not isinstance(binding, dict):
        raise ValueError("cascade binding 必须是 JSON 对象")
    model, resolved_binding = build_full_cascade_model(
        root,
        l1_checkpoint_path=binding["l1_checkpoint_path"],
        l2_checkpoint_path=binding["l2_checkpoint_path"],
        l1_knownness_head_path=binding["l1_knownness_head_path"],
        l2_knownness_head_path=binding["l2_knownness_head_path"],
        calibration_path=binding["l3_calibration_path"],
        model_config_path=binding["model_config_path"],
        device="cpu",
    )
    config = __import__("radar_ln3.model", fromlist=["load_model_config"]).load_model_config(root)
    target = _resolve(root, output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"ONNX 已存在：{target}")
    dummy = _dummy_inputs(config, batch=dummy_batch_size, torch=torch)
    descriptor = _metadata(resolved_binding, config, opset=opset_version)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with torch.inference_mode():
        reference = model(*dummy)
        torch.onnx.export(
            model,
            dummy,
            str(temporary),
            input_names=list(FULL_CASCADE_INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={name: {0: "N"} for name in (*FULL_CASCADE_INPUT_NAMES, *OUTPUT_NAMES)},
            opset_version=int(opset_version),
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
    proto = onnx.load(str(temporary))
    onnx.checker.check_model(proto)
    proto.ClearField("metadata_props")
    for key, value in descriptor.items():
        item = proto.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(proto, str(temporary))
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    actual_inputs = tuple(value.name for value in session.get_inputs())
    if actual_inputs != FULL_CASCADE_INPUT_NAMES:
        raise RuntimeError(f"ONNX input contract 不一致：{actual_inputs}")
    for batch_size in (1, 2, 8):
        values = _dummy_inputs(config, batch=batch_size, torch=torch)
        feed = {name: value.detach().cpu().numpy() for name, value in zip(FULL_CASCADE_INPUT_NAMES, values, strict=True)}
        outputs = session.run(list(OUTPUT_NAMES), feed)
        if tuple(outputs[0].shape) != (batch_size, 2) or tuple(outputs[1].shape) != (batch_size, 3) or tuple(outputs[2].shape) != (batch_size, 6):
            raise RuntimeError("ONNX full cascade 概率输出 shape 无效")
        if not all(np.isfinite(np.asarray(value)).all() for value in outputs):
            raise RuntimeError("ONNX full cascade 输出包含非有限值")
        if not np.allclose(np.asarray(outputs[2]).sum(axis=1), 1.0, atol=1.0e-5):
            raise RuntimeError("ONNX cascade_probabilities 行和不是 1")
        with torch.inference_mode():
            reference_outputs = model(*values)
        for index, (actual, expected) in enumerate(zip(outputs, reference_outputs, strict=True)):
            expected_array = expected.detach().cpu().numpy()
            actual_array = np.asarray(actual)
            tolerance = 0.0 if index >= 3 else 2.0e-4
            if not np.allclose(actual_array, expected_array, atol=tolerance, rtol=2.0e-4):
                raise RuntimeError(
                    f"ONNX 与 PyTorch full cascade 输出不一致：output={OUTPUT_NAMES[index]}"
                )
    os.replace(temporary, target)
    return {
        "path": str(target),
        "contract": CONTRACT,
        "inputs": list(FULL_CASCADE_INPUT_NAMES),
        "outputs": list(OUTPUT_NAMES),
        "formal_available": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = export(
            project_root=args.project_root,
            binding_path=args.binding_path,
            output_path=args.output_path,
            opset_version=int(args.opset_version),
            dummy_batch_size=int(args.dummy_batch_size),
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"full cascade ONNX export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
