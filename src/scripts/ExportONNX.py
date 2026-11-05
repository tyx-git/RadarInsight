"""Export the current L2 direct endpoint to an ONNX artifact.

This exporter targets the RadarInsight streaming/offline endpoint contract. It
does not produce the separate AeroSight single-input ``[N, 8]`` model contract.
The distinction is written to ONNX metadata so a client cannot silently use
the wrong input protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


EXPORT_CONTRACT = "RADAR-LN3-ONNX-STREAMING-L2-DIRECT-V1"
DEFAULT_CHECKPOINT = Path(
    "artifacts/radar_ln3/l2-direct-grouped/l2-direct-grouped.pt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/radar_ln3/onnx/radar_ln3_l2_direct_grouped.onnx"
)
INPUT_NAMES = (
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
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the RadarInsight L2 direct endpoint to ONNX"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="RadarInsight project root",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="L2 direct/direct_grouped checkpoint path relative to project root",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="ONNX output path relative to project root",
    )
    parser.add_argument(
        "--model-version",
        default="radar-ln3-l2-direct-grouped-onnx-v1",
        help="Version written into ONNX metadata",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dummy-batch-size",
        type=int,
        default=2,
        help="Dummy batch size used during export; runtime batch remains dynamic",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Static exported sequence length; defaults to model max_sequence_length",
    )
    parser.add_argument(
        "--provider",
        choices=("CPUExecutionProvider", "DirectMLExecutionProvider"),
        default="CPUExecutionProvider",
        help="ONNX Runtime provider used for post-export validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing ONNX file",
    )
    return parser


def _resolve_project_path(root: Path, value: str | Path, *, must_exist: bool) -> Path:
    from radar_ln3.data.paths import resolve_project_path

    return resolve_project_path(root, value, must_exist=must_exist)


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "ONNX 导出需要当前虚拟环境同时安装 torch、onnx 和 onnxruntime"
        ) from exc
    return torch, nn, (np, onnx, ort)


def _build_model(
    root: Path,
    checkpoint_path: Path,
) -> tuple[Any, Any, Any, str]:
    from radar_ln3.checkpoint import Checkpoint, canonical_hash, load_model_state, sha256_file
    from radar_ln3.model import build_l2_model, load_model_config

    config = load_model_config(root)
    checkpoint = Checkpoint.load(root, checkpoint_path, expected_task="l2")
    contract = checkpoint.payload.get("model_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("L2 checkpoint 缺少 model_contract")
    mode = str(contract.get("mode") or "")
    arm = str(contract.get("arm") or "")
    if mode not in {"direct", "direct_grouped"}:
        raise ValueError(
            "ExportONNX.py 只导出 L2 direct/direct_grouped checkpoint；"
            f"当前 checkpoint mode={mode!r}"
        )
    if not arm:
        raise ValueError("L2 checkpoint model_contract 缺少 arm")

    base = Checkpoint.load(root, config.artifacts.l2_base_weight_path)
    stable = Checkpoint.load(root, config.artifacts.l2_stable_weight_path)
    model = build_l2_model(
        config,
        arm=arm,
        mode=mode,
        base_state=base.state,
        stable_state=stable.state,
        seed=config.training.seed,
    )
    expected_contract = model.model_contract()
    if canonical_hash(expected_contract) != canonical_hash(contract):
        raise ValueError(
            "checkpoint model_contract 与当前配置构造出的模型不一致；"
            "请使用与当前 src/config/model.json 匹配的 checkpoint"
        )
    load_model_state(model, checkpoint.state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, model, checkpoint, sha256_file(config.path)


def _make_dummy_inputs(
    config: Any,
    *,
    batch: int,
    length: int,
    torch: Any,
) -> tuple[Any, ...]:
    if batch <= 0:
        raise ValueError("dummy batch size 必须为正整数")
    if length <= 0 or length > int(config.data.max_sequence_length):
        raise ValueError("sequence length 必须位于 [1, model.max_sequence_length] 内")
    dtype = torch.float32
    tr_features = torch.zeros((batch, length, config.model.tr_value_dim), dtype=dtype)
    tr_missing_mask = torch.zeros(
        (batch, length, config.model.tr_missing_dim), dtype=dtype
    )
    tr_dt_seconds = torch.zeros((batch, length), dtype=dtype)
    tr_dt_valid = torch.zeros((batch, length), dtype=torch.bool)
    tr_dt_reason = torch.full((batch, length), 1, dtype=torch.long)
    if length > 1:
        tr_dt_seconds[:, 1:] = 1.0
        tr_dt_valid[:, 1:] = True
        tr_dt_reason[:, 1:] = 3
    tr_padding_mask = torch.zeros((batch, length), dtype=torch.bool)
    rd_frames = torch.zeros(
        (
            batch,
            length,
            config.data.max_rd_frames_per_point,
            2,
            config.model.rd_height,
            config.model.rd_width,
        ),
        dtype=dtype,
    )
    rd_frame_mask = torch.zeros(
        (batch, length, config.data.max_rd_frames_per_point), dtype=torch.bool
    )
    rd_quality = torch.zeros(
        (batch, length, config.model.rd_quality_dim), dtype=dtype
    )
    rd_physical_features = torch.zeros(
        (batch, length, config.data.l2.rd_physical_dim), dtype=dtype
    )
    rd_physical_valid = torch.zeros((batch, length), dtype=torch.bool)
    micro_doppler_features = torch.zeros(
        (batch, length, config.data.l2.micro_doppler_dim), dtype=dtype
    )
    micro_doppler_valid = torch.zeros((batch, length), dtype=torch.bool)
    return (
        tr_features,
        tr_missing_mask,
        tr_dt_seconds,
        tr_dt_valid,
        tr_dt_reason,
        tr_padding_mask,
        rd_frames,
        rd_frame_mask,
        rd_quality,
        rd_physical_features,
        rd_physical_valid,
        micro_doppler_features,
        micro_doppler_valid,
    )


def _make_wrapper(model: Any, torch: Any, nn: Any) -> Any:
    class ExportWrapper(nn.Module):
        def __init__(self, endpoint: Any) -> None:
            super().__init__()
            self.endpoint = endpoint

        def _rd_embedding(self, rd_frames: Any, rd_frame_mask: Any) -> Any:
            base = self.endpoint.trainable_base_model
            encoder = base.rd_encoder
            flat_frames = rd_frames.reshape(
                -1,
                2,
                encoder.rd_height,
                encoder.rd_width,
            )
            flat_embeddings = encoder.projection(encoder.network(flat_frames))
            embeddings = flat_embeddings.reshape(
                rd_frames.shape[0],
                rd_frames.shape[1],
                rd_frames.shape[2],
                -1,
            )
            mask = rd_frame_mask.to(dtype=rd_frames.dtype).unsqueeze(-1)
            return (embeddings * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

        def _native_events(self, inputs: Mapping[str, Any]) -> Any:
            base = self.endpoint.trainable_base_model
            tr_features = inputs["tr_features"]
            tr_embedding = base.tr_encoder(
                torch.cat((tr_features, inputs["tr_missing_mask"]), dim=-1)
            )
            rd_embedding = self._rd_embedding(
                inputs["rd_frames"], inputs["rd_frame_mask"]
            )
            physical = inputs["rd_physical_features"] * inputs[
                "rd_physical_valid"
            ].to(dtype=tr_features.dtype).unsqueeze(-1)
            micro = inputs["micro_doppler_features"] * inputs[
                "micro_doppler_valid"
            ].to(dtype=tr_features.dtype).unsqueeze(-1)
            return base.fusion(
                torch.cat(
                    (
                        tr_embedding,
                        rd_embedding,
                        inputs["rd_quality"],
                        physical,
                        micro,
                    ),
                    dim=-1,
                )
            )

        def _cell_step_export(
            self,
            event: Any,
            state: Any,
            dt_seconds: Any,
            dt_valid: Any,
            dt_reason: Any,
            zero_ordinal: Any,
        ) -> Any:
            cell = self.endpoint.trainable_base_model.liquid.cell
            if cell.invalid_dt_update_mode not in {
                "floor",
                "bounded_observation",
                "skip",
                "density_weighted",
            }:
                raise RuntimeError(
                    "ONNX exporter 不支持 strict_skip Liquid V2；"
                    f"当前 mode={cell.invalid_dt_update_mode}"
                )
            normalized = cell.compute_norm(state)
            combined = torch.cat((event, normalized), dim=-1)
            candidate = torch.tanh(
                cell.candidate_projection(combined)
                + cell.input_projection(event)
                + cell.state_projection(normalized)
            )
            gate = torch.sigmoid(cell.gate_projection(combined))
            mixed = gate * candidate + (1.0 - gate) * state

            dtype = dt_seconds.dtype
            safe_dt = dt_seconds.clamp(min=0.0, max=cell.dt_max_seconds)
            floor = ((safe_dt <= 1.0e-9) | (~dt_valid)).to(dtype=dtype)
            safe_dt = (
                (1.0 - floor) * safe_dt
                + floor * float(cell.zero_dt_event_floor_seconds)
            )
            tau = cell.tau(dtype=dtype, device=dt_seconds.device)
            tau_batch = tau.unsqueeze(0).expand_as(state)
            alpha = 1.0 - torch.exp(-safe_dt.unsqueeze(-1) / tau_batch)
            valid = dt_valid.to(dtype=alpha.dtype).unsqueeze(-1)
            if cell.invalid_dt_update_mode in {
                "bounded_observation",
                "skip",
                "density_weighted",
            }:
                gain = torch.full_like(alpha, cell.invalid_dt_observation_gain)
                alpha = valid * alpha + (1.0 - valid) * gain

            first = (dt_reason == 1).to(dtype=alpha.dtype).unsqueeze(-1)
            zero = (dt_reason == 2).to(dtype=alpha.dtype).unsqueeze(-1)
            if cell.invalid_dt_update_mode == "skip":
                zero_gain = torch.zeros_like(alpha)
            elif cell.invalid_dt_update_mode == "density_weighted":
                zero_gain = torch.clamp(
                    cell.invalid_dt_observation_gain
                    * zero_ordinal.to(dtype=alpha.dtype),
                    max=3.0 * cell.invalid_dt_observation_gain,
                ).unsqueeze(-1).expand_as(alpha)
            else:
                zero_gain = torch.full_like(alpha, cell.invalid_dt_observation_gain)
            alpha = first * torch.ones_like(alpha) + (1.0 - first) * alpha
            alpha = (zero * zero_gain + (1.0 - zero) * alpha).clamp(0.0, 1.0)
            return (1.0 - alpha) * state + alpha * mixed

        def _liquid_sequence(self, events: Any, inputs: Mapping[str, Any]) -> Any:
            base = self.endpoint.trainable_base_model
            cell = base.liquid.cell
            padding = inputs["tr_padding_mask"]
            dt_seconds = inputs["tr_dt_seconds"]
            dt_valid = inputs["tr_dt_valid"]
            dt_reason = inputs["tr_dt_reason"]
            state = events.new_zeros((events.shape[0], cell.hidden_dim))
            zero_ordinal = torch.ones_like(dt_reason, dtype=torch.long)
            states: list[Any] = []
            for event_index in range(int(events.shape[1])):
                active = ~padding[:, event_index]
                active_long = active.to(dtype=torch.long)
                active_float = active.to(dtype=dt_seconds.dtype)
                reason = (
                    active_long * dt_reason[:, event_index]
                    + (1 - active_long)
                )
                dt = active_float * dt_seconds[:, event_index]
                valid = active & dt_valid[:, event_index]
                previous = zero_ordinal[:, event_index - 1] if event_index else None
                previous_value = (
                    previous + 1 if previous is not None else torch.ones_like(reason)
                )
                zero_flag = (reason == 2).to(dtype=torch.long)
                current = zero_flag * previous_value + (1 - zero_flag)
                step_ordinal = active_long * current + (1 - active_long)
                zero_ordinal[:, event_index] = active_long * current
                next_state = self._cell_step_export(
                    events[:, event_index],
                    state,
                    dt,
                    valid,
                    reason,
                    step_ordinal,
                )
                state = (
                    active_float.unsqueeze(-1) * next_state
                    + (1.0 - active_float.unsqueeze(-1)) * state
                )
                states.append(state)
            return torch.stack(states, dim=1)

        def forward(
            self,
            tr_features: Any,
            tr_missing_mask: Any,
            tr_dt_seconds: Any,
            tr_dt_valid: Any,
            tr_dt_reason: Any,
            tr_padding_mask: Any,
            rd_frames: Any,
            rd_frame_mask: Any,
            rd_quality: Any,
            rd_physical_features: Any,
            rd_physical_valid: Any,
            micro_doppler_features: Any,
            micro_doppler_valid: Any,
        ) -> Any:
            inputs = {
                "tr_features": tr_features,
                "tr_missing_mask": tr_missing_mask,
                "tr_dt_seconds": tr_dt_seconds,
                "tr_dt_valid": tr_dt_valid,
                "tr_dt_reason": tr_dt_reason,
                "tr_padding_mask": tr_padding_mask,
                "rd_frames": rd_frames,
                "rd_frame_mask": rd_frame_mask,
                "rd_quality": rd_quality,
                "rd_physical_features": rd_physical_features,
                "rd_physical_valid": rd_physical_valid,
                "micro_doppler_features": micro_doppler_features,
                "micro_doppler_valid": micro_doppler_valid,
            }
            events = self._native_events(inputs)
            sequence = self._liquid_sequence(events, inputs)
            base = self.endpoint.trainable_base_model
            sequence_logits = base.classifier(base.classifier_norm(sequence)).masked_fill(
                tr_padding_mask.unsqueeze(-1), 0.0
            )
            lengths = (~tr_padding_mask).long().sum(dim=1).clamp_min(1)
            class_count = int(base.classifier.out_features)
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, class_count)
            final_logits = sequence_logits.gather(1, gather_index).squeeze(1)
            probabilities = torch.softmax(final_logits, dim=-1).reshape(-1, class_count)
            # Keep every declared contract input visible in the graph. Some
            # validity masks are all false in the dummy export input and can
            # otherwise be removed by constant folding.
            anchor = probabilities.new_zeros(())
            for value in (
                tr_features,
                tr_missing_mask,
                tr_dt_seconds,
                tr_dt_valid,
                tr_dt_reason,
                tr_padding_mask,
                rd_frames,
                rd_frame_mask,
                rd_quality,
                rd_physical_features,
                rd_physical_valid,
                micro_doppler_features,
                micro_doppler_valid,
            ):
                anchor = anchor + value.to(dtype=probabilities.dtype).reshape(-1).sum() * 0.0
            return probabilities + anchor

    return ExportWrapper(model).eval()


def _shape_for_input(config: Any, name: str, length: int) -> list[Any]:
    batch = "N"
    if name == "tr_features":
        return [batch, length, config.model.tr_value_dim]
    if name == "tr_missing_mask":
        return [batch, length, config.model.tr_missing_dim]
    if name in {"tr_dt_seconds", "tr_dt_valid", "tr_dt_reason", "tr_padding_mask"}:
        return [batch, length]
    if name == "rd_frames":
        return [
            batch,
            length,
            config.data.max_rd_frames_per_point,
            2,
            config.model.rd_height,
            config.model.rd_width,
        ]
    if name == "rd_frame_mask":
        return [batch, length, config.data.max_rd_frames_per_point]
    if name == "rd_quality":
        return [batch, length, config.model.rd_quality_dim]
    if name == "rd_physical_features":
        return [batch, length, config.data.l2.rd_physical_dim]
    if name == "micro_doppler_features":
        return [batch, length, config.data.l2.micro_doppler_dim]
    if name in {"rd_physical_valid", "micro_doppler_valid"}:
        return [batch, length]
    raise ValueError(f"未登记的 ONNX input：{name}")


def _input_metadata(config: Any, length: int) -> dict[str, Any]:
    return {
        name: {
            "dtype": (
                "int64"
                if name == "tr_dt_reason"
                else "bool"
                if name.endswith("_valid") or name.endswith("_mask")
                else "float32"
            ),
            "shape": _shape_for_input(config, name, length),
        }
        for name in INPUT_NAMES
    }


def _metadata(
    *,
    root: Path,
    config: Any,
    checkpoint: Any,
    config_sha256: str,
    checkpoint_path: Path,
    length: int,
    opset_version: int,
    model_version: str,
) -> dict[str, str]:
    from radar_ln3.checkpoint import sha256_file
    from radar_ln3.data.paths import relative_project_path

    class_names = list(config.class_names["l2"])
    body: dict[str, Any] = {
        "contract": EXPORT_CONTRACT,
        "aerosight_8_feature_compatible": False,
        "model_name": "radar-ln3-l2-direct-grouped",
        "model_version": str(model_version),
        "input_spec_version": EXPORT_CONTRACT,
        "task": "l2",
        "class_names": class_names,
        "input_names": list(INPUT_NAMES),
        "input_specs": _input_metadata(config, length),
        "output_name": "probabilities",
        "output_shape": ["N", len(class_names)],
        "output_semantics": (
            "softmax probability over the final event of each padded parent sequence"
        ),
        "sequence_length": int(length),
        "sequence_length_basis": "default_window_from_average_event_count",
        "dynamic_axes": {
            name: {"0": "N"} for name in (*INPUT_NAMES, "probabilities")
        },
        "opset_version": int(opset_version),
        "checkpoint_path": relative_project_path(root, checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint.path),
        "config_path": relative_project_path(root, config.path),
        "config_sha256": str(config_sha256),
        "stateful_streaming": False,
        "streaming_semantics": (
            "offline_prefix_replay; feed the complete causal prefix and use the final row"
        ),
        "l3_calibration_included": False,
        "aerosight_contract_note": (
            "This artifact has multiple sequence inputs and is not the AeroSight [N,8] client model."
        ),
    }
    return {
        key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, (dict, list, bool, int, float))
        else str(value)
        for key, value in body.items()
    }


def _validate_onnx(
    path: Path,
    *,
    config: Any,
    length: int,
    provider: str,
    torch: Any,
    np: Any,
    onnx: Any,
    ort: Any,
) -> None:
    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    graph_shape = model_proto.graph.output[0].type.tensor_type.shape.dim
    if len(graph_shape) != 2:
        raise RuntimeError(f"ONNX graph output rank 无效：{len(graph_shape)}")
    if graph_shape[1].dim_value != len(config.class_names["l2"]):
        raise RuntimeError(
            "ONNX graph output 类别维度不是固定值："
            f"expected={len(config.class_names['l2'])}, actual={graph_shape[1]}"
        )
    session = ort.InferenceSession(str(path), providers=[provider])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    actual_input_names = tuple(item.name for item in inputs)
    if actual_input_names != INPUT_NAMES:
        raise RuntimeError(
            "ONNX input name/order 与导出 contract 不一致："
            f"expected={INPUT_NAMES}, actual={actual_input_names}"
        )
    if len(outputs) != 1 or outputs[0].name != "probabilities":
        raise RuntimeError("ONNX 必须只有一个 probabilities 输出")
    if outputs[0].type != "tensor(float)":
        raise RuntimeError(f"ONNX probabilities 类型无效：{outputs[0].type}")
    if outputs[0].shape[0] == 1:
        raise RuntimeError("ONNX 输出 batch 维度被固定为 1")
    expected_class_count = len(config.class_names["l2"])
    if outputs[0].shape[1] != expected_class_count:
        raise RuntimeError(
            "ONNX 输出类别维度必须是固定值："
            f"expected={expected_class_count}, actual={outputs[0].shape}"
        )

    for batch in (1, 2, 8):
        values = _make_dummy_inputs(config, batch=batch, length=length, torch=torch)
        feed = {
            name: value.detach().cpu().numpy()
            for name, value in zip(INPUT_NAMES, values, strict=True)
        }
        probabilities = np.asarray(session.run(["probabilities"], feed)[0])
        expected = (batch, expected_class_count)
        if probabilities.shape != expected:
            raise RuntimeError(
                f"ONNX 输出 shape 无效：expected={expected}, actual={probabilities.shape}"
            )
        if probabilities.dtype != np.float32:
            raise RuntimeError(f"ONNX 输出 dtype 无效：{probabilities.dtype}")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("ONNX probabilities 包含非有限值")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError("ONNX probabilities 超出 [0,1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0):
            raise RuntimeError("ONNX probabilities 行和不是 1")

    metadata = {item.key: item.value for item in model_proto.metadata_props}
    class_names = json.loads(metadata.get("class_names", "null"))
    if class_names != list(config.class_names["l2"]):
        raise RuntimeError("ONNX class_names metadata 与 L2 contract 不一致")
    if metadata.get("contract") != EXPORT_CONTRACT:
        raise RuntimeError("ONNX contract metadata 缺失或不匹配")


def _export(
    *,
    root: Path,
    checkpoint_path: Path,
    output_path: Path,
    model_version: str,
    opset_version: int,
    dummy_batch_size: int,
    sequence_length: int | None,
    provider: str,
    overwrite: bool,
) -> dict[str, Any]:
    torch, nn, runtime = _load_runtime()
    np, onnx, ort = runtime
    if opset_version < 17:
        raise ValueError("当前 exporter 要求 opset_version >= 17")
    if dummy_batch_size <= 0:
        raise ValueError("dummy batch size 必须为正整数")
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"ONNX Runtime provider 不可用：{provider}; "
            f"available={ort.get_available_providers()}"
        )

    root = root.resolve()
    checkpoint = _resolve_project_path(root, checkpoint_path, must_exist=True)
    target = _resolve_project_path(root, output_path, must_exist=False)
    if target.exists() and not overwrite:
        raise FileExistsError(f"ONNX 文件已存在；如需覆盖请显式指定 --overwrite：{target}")

    config, model, checkpoint_object, config_sha256 = _build_model(root, checkpoint)
    length = int(
        config.data.max_sequence_length
        if sequence_length is None
        else sequence_length
    )
    wrapper = _make_wrapper(model, torch, nn)
    dummy_inputs = _make_dummy_inputs(
        config,
        batch=dummy_batch_size,
        length=length,
        torch=torch,
    )
    with torch.inference_mode():
        direct_inputs = {
            name: value
            for name, value in zip(INPUT_NAMES, dummy_inputs, strict=True)
        }
        reference = torch.softmax(
            model.direct_logits(base_inputs=direct_inputs)["final_logits"],
            dim=-1,
        )
        wrapped = wrapper(*dummy_inputs)
        if not torch.allclose(reference, wrapped, atol=1.0e-5, rtol=1.0e-5):
            raise RuntimeError("ONNX wrapper 与 PyTorch L2 direct endpoint 输出不一致")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _metadata(
        root=root,
        config=config,
        checkpoint=checkpoint_object,
        config_sha256=config_sha256,
        checkpoint_path=checkpoint,
        length=length,
        opset_version=opset_version,
        model_version=model_version,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.",
            suffix=".onnx",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
        temporary = Path(temporary_name)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                str(temporary),
                input_names=list(INPUT_NAMES),
                output_names=["probabilities"],
                dynamic_axes={
                    name: {0: "N"} for name in (*INPUT_NAMES, "probabilities")
                },
                opset_version=int(opset_version),
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )

        model_proto = onnx.load(str(temporary))
        onnx.checker.check_model(model_proto)
        graph_shape = model_proto.graph.output[0].type.tensor_type.shape.dim
        if len(graph_shape) != 2:
            raise RuntimeError("ONNX graph output 必须是二维 tensor")
        graph_shape[0].ClearField("dim_value")
        graph_shape[0].dim_param = "N"
        graph_shape[1].ClearField("dim_param")
        graph_shape[1].dim_value = len(config.class_names["l2"])
        model_proto.ClearField("metadata_props")
        for key, value in descriptor.items():
            item = model_proto.metadata_props.add()
            item.key = key
            item.value = value
        onnx.save(model_proto, str(temporary))
        _validate_onnx(
            temporary,
            config=config,
            length=length,
            provider=provider,
            torch=torch,
            np=np,
            onnx=onnx,
            ort=ort,
        )
        os.replace(temporary, target)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()

    from radar_ln3.checkpoint import sha256_file

    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "contract": EXPORT_CONTRACT,
        "task": "l2",
        "class_names": list(config.class_names["l2"]),
        "sequence_length": length,
        "dynamic_batch": True,
        "provider_checked": provider,
        "aerosight_8_feature_compatible": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _export(
            root=args.project_root,
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            model_version=str(args.model_version),
            opset_version=int(args.opset_version),
            dummy_batch_size=int(args.dummy_batch_size),
            sequence_length=args.sequence_length,
            provider=str(args.provider),
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ONNX export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
