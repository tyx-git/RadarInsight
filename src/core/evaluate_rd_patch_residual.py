from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .device import resolve_cuda_device
from .rd_drone_bird_trainer import RD_ONLY_LABELS, _balanced_smoke_ids, _make_loader, _metric_payload, _select_rd_only_ids
from .rd_patch_residual_config import RDPatchResidualConfig
from .rd_patch_residual_model import FrozenSequencePatchResidual


def evaluate_rd_patch_residual_checkpoint(checkpoint_path: Path, output_path: Path, *, device_name: str = "cuda", num_workers: int = 0, force_zero_residual: bool = False) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = dict(checkpoint.get("config", {}))
    values["device"] = device_name
    values["num_workers"] = int(num_workers)
    values["run_test_evaluation"] = False
    config = RDPatchResidualConfig(**values)
    device, device_info = resolve_cuda_device(config.device)
    root = Path.cwd()
    index = pd.read_csv(root / "data" / "processed" / config.index_name, low_memory=False)
    anchor_map = pd.read_csv(root / "data" / "processed" / config.window_segment_map_name, low_memory=False)
    val_ids = _select_rd_only_ids(index, anchor_map, "val")
    loader_ids = _balanced_smoke_ids(index, val_ids, config.max_val_batches, config.batch_size)
    loader = _make_loader(str(root / "data"), loader_ids, config, shuffle=False)
    model = FrozenSequencePatchResidual(config.base_checkpoint, dropout=config.dropout, residual_scale=config.residual_scale).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logits_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch.rd_features.to(device, non_blocking=True),
                batch.rd_patch_features.to(device, non_blocking=True),
                rd_padding_mask=batch.rd_padding_mask.to(device, non_blocking=True),
                rd_patch_padding_mask=batch.rd_patch_padding_mask.to(device, non_blocking=True),
                force_zero_residual=force_zero_residual,
            )
            logits = output.logits.detach().cpu()
            base_logits = output.base_logits.detach().cpu()
            deltas = output.residual_delta.detach().cpu()
            gates = output.residual_gate.detach().cpu()
            labels = torch.tensor([1 if value == "BirdTarget" else 0 for value in batch.label_names], dtype=torch.long)
            predicted = logits.argmax(dim=-1).tolist()
            for index_in_batch, sample_id in enumerate(batch.sample_ids):
                predictions.append({
                    "sample_id": str(sample_id),
                    "parent_sample_id": str(batch.parent_sample_ids[index_in_batch]),
                    "source_name": str(batch.source_names[index_in_batch]),
                    "label_name_true": str(batch.label_names[index_in_batch]),
                    "label_name_rd_only_pred": RD_ONLY_LABELS[int(predicted[index_in_batch])],
                    "bird_logit_drone": float(logits[index_in_batch, 0]),
                    "bird_logit_bird": float(logits[index_in_batch, 1]),
                    "base_logit_drone": float(base_logits[index_in_batch, 0]),
                    "base_logit_bird": float(base_logits[index_in_batch, 1]),
                    "residual_delta_drone": float(deltas[index_in_batch, 0]),
                    "residual_delta_bird": float(deltas[index_in_batch, 1]),
                    "residual_gate": float(gates[index_in_batch, 0]),
                    "is_correct": bool(RD_ONLY_LABELS[int(predicted[index_in_batch])] == str(batch.label_names[index_in_batch])),
                    "rd_time_sync_status": str(batch.rd_time_sync_status[index_in_batch]),
                    "rd_valid_length": int(batch.rd_lengths[index_in_batch].item()),
                })
            logits_rows.append(logits)
            labels_rows.append(labels)
            parent_ids.extend(batch.parent_sample_ids)
    logits = torch.cat(logits_rows)
    labels = torch.cat(labels_rows)
    payload = {
        "evaluation_scope": "validation_only_frozen_sequence_patch_residual",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": device_info,
        "config": asdict(config),
        "metrics": _metric_payload(logits, labels, parent_ids),
        "predictions": predictions,
        "force_zero_residual": bool(force_zero_residual),
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluator for frozen-sequence RD patch residual checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-zero-residual", action="store_true")
    args = parser.parse_args()
    result = evaluate_rd_patch_residual_checkpoint(Path(args.checkpoint), Path(args.output), device_name=args.device, num_workers=args.num_workers, force_zero_residual=args.force_zero_residual)
    print(json.dumps({"output": args.output, "metrics": result["metrics"], "force_zero_residual": args.force_zero_residual, "test_evaluation_executed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
