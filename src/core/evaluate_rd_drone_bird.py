from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .device import resolve_cuda_device
from .experts import LowAltitudeSuspectExpert
from .metrics import compute_classification_metrics
from .rd_drone_bird_config import RDDroneBirdDiagnosticConfig
from .rd_drone_bird_trainer import _make_loader, _select_rd_only_ids
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def _metric_payload(logits: torch.Tensor, labels: torch.Tensor, parents: list[str]) -> dict[str, object]:
    window = compute_classification_metrics(logits, labels, num_classes=2)
    per_class = [asdict(item) for item in compute_per_class_metrics(logits, labels, RD_ONLY_LABELS)]
    grouped_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_labels: dict[str, list[int]] = defaultdict(list)
    for parent, logit, label in zip(parents, logits, labels.tolist()):
        grouped_logits[str(parent)].append(logit)
        grouped_labels[str(parent)].append(int(label))
    parent_logits = torch.stack([torch.stack(values).mean(dim=0) for values in grouped_logits.values()])
    parent_targets = torch.tensor([values[0] for values in grouped_labels.values()], dtype=torch.long)
    parent = compute_classification_metrics(parent_logits, parent_targets, num_classes=2)
    return {
        "window": asdict(window),
        "per_class": per_class,
        "parent_mean_logit": asdict(parent),
        "parent_count": int(len(grouped_logits)),
        "window_count": int(logits.shape[0]),
    }


def evaluate_rd_drone_bird_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    device_name: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = dict(checkpoint.get("config", {}))
    if num_workers is not None:
        values["num_workers"] = int(num_workers)
    values["device"] = device_name
    values["run_test_evaluation"] = False
    config = RDDroneBirdDiagnosticConfig(**values)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    index = pd.read_csv(project_root / "data" / "processed" / config.index_name, low_memory=False)
    anchor_map = pd.read_csv(project_root / "data" / "processed" / config.window_segment_map_name, low_memory=False)
    val_ids = _select_rd_only_ids(index, anchor_map, "val")
    loader = _make_loader(str(project_root / "data"), val_ids, config, shuffle=False)
    model = LowAltitudeSuspectExpert(
        input_dim=160,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=0,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            rd_patch_features = batch.rd_patch_features.to(device, non_blocking=True)
            rd_patch_padding = batch.rd_patch_padding_mask.to(device, non_blocking=True)
            output = model.rd_drone_bird_diagnostic(
                batch.rd_features.to(device, non_blocking=True),
                rd_patch_features=rd_patch_features if config.use_rd_patch_cnn else None,
                rd_padding_mask=batch.rd_padding_mask.to(device, non_blocking=True),
                rd_patch_padding_mask=rd_patch_padding if config.use_rd_patch_cnn else None,
            )
            logits = output.bird_logits.detach().cpu()
            labels = torch.tensor(
                [1 if value == "BirdTarget" else 0 for value in batch.label_names],
                dtype=torch.long,
            )
            predicted = logits.argmax(dim=-1)
            for index_in_batch, (sample_id, parent_id, source, truth, logit, pred) in enumerate(
                zip(batch.sample_ids, batch.parent_sample_ids, batch.source_names, batch.label_names, logits, predicted.tolist())
            ):
                predictions.append(
                    {
                        "sample_id": str(sample_id),
                        "parent_sample_id": str(parent_id),
                        "source_name": str(source),
                        "label_name_true": str(truth),
                        "label_name_rd_only_pred": RD_ONLY_LABELS[int(pred)],
                        "bird_logit_drone": float(logit[0]),
                        "bird_logit_bird": float(logit[1]),
                        "is_correct": bool(RD_ONLY_LABELS[int(pred)] == str(truth)),
                        "rd_time_sync_status": str(batch.rd_time_sync_status[index_in_batch]),
                        "rd_valid_length": int(batch.rd_lengths[index_in_batch].item()),
                    }
                )
            logits_rows.append(logits)
            label_rows.append(labels)
            parent_ids.extend(batch.parent_sample_ids)
    logits_tensor = torch.cat(logits_rows)
    labels_tensor = torch.cat(label_rows)
    payload = {
        "evaluation_scope": "validation_only_rd_drone_bird_diagnostic",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": device_info,
        "config": asdict(config),
        "metrics": _metric_payload(logits_tensor, labels_tensor, parent_ids),
        "predictions": predictions,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
        "timing_status": "track_point_anchor_diagnostic_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluator for diagnostic RD-only Drone/Bird checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_rd_drone_bird_checkpoint(
        Path(args.checkpoint),
        Path(args.output),
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps({"output": args.output, "metrics": result["metrics"], "test_evaluation_executed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
