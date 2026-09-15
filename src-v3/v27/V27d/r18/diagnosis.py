"""Read-only paired diagnosis for source stratification and radial speed."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from v27.v27_baseline.metrics import summarize_records
from v27.V27d.r17.frozen import load_frozen_models

from .data import R18Dataset, build_r18_loader, r17_l2_inputs
from .protocol import R18_L2_CLASSES, R18_REPORT_SCHEMA, R18Config, artifact_path, load_manifest, seal, write_json_if_identical


def _records(model: torch.nn.Module, dataset: R18Dataset, *, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r18_loader(dataset, batch_size=4, shuffle=False):
            output = model(**r17_l2_inputs(batch, device))
            logits = output["window_logits"].detach().cpu()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[row]
                output_rows.append({"parent_sample_id": str(parent_id), "source_suffix": str(parent_id).split("|")[2], "label_name": str(batch["label_names"][row]), "true_id": int(batch["l2_labels"][row]), "pred_id": int(final.argmax().item()), "probabilities": [float(v) for v in torch.softmax(final, -1).tolist()], "event_count": int(batch["lengths"][row]), "radial_speed_ablation": dataset.radial_speed_ablation})
    if [row["parent_sample_id"] for row in output_rows] != dataset.parent_ids:
        raise ValueError("r18 diagnosis changed parent order")
    return output_rows


def _paired(full: list[dict[str, Any]], ablated: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["parent_sample_id"]: row for row in full}
    right = {row["parent_sample_id"]: row for row in ablated}
    if set(left) != set(right):
        raise ValueError("r18 paired radial-speed records have different parents")
    transitions: dict[str, int] = defaultdict(int)
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        before, after = left[parent_id], right[parent_id]
        before_correct = before["true_id"] == before["pred_id"]
        after_correct = after["true_id"] == after["pred_id"]
        transitions[f"{'correct' if before_correct else 'wrong'}_to_{'correct' if after_correct else 'wrong'}"] += 1
        if before["pred_id"] != after["pred_id"]:
            changed.append({"parent_sample_id": parent_id, "source_suffix": before["source_suffix"], "label_name": before["label_name"], "full_prediction": before["pred_id"], "ablated_prediction": after["pred_id"], "full_correct": before_correct, "ablated_correct": after_correct})
    return {"parent_count": len(left), "correctness_transitions": dict(sorted(transitions.items())), "changed_predictions": changed}


def diagnose(project_root: str | Path, config: R18Config, *, device: str = "cuda") -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    frozen = load_frozen_models(root, config.r17)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("r18 diagnosis requested CUDA but CUDA is unavailable")
    runtime_device = torch.device("cuda:0" if device == "cuda" else "cpu")
    reports: dict[str, Any] = {}
    for split in ("train", "validation"):
        full_dataset = R18Dataset(root, config, split_name=split, radial_speed_ablation=False)
        ablated_dataset = R18Dataset(root, config, split_name=split, radial_speed_ablation=True)
        full = _records(frozen.l2_model.to(runtime_device), full_dataset, device=runtime_device)
        ablated = _records(frozen.l2_model.to(runtime_device), ablated_dataset, device=runtime_device)
        reports[split] = {"full": {"metrics": summarize_records(full, class_names=R18_L2_CLASSES, scope=f"{split}:full_parent_final"), "records": full}, "radial_speed_ablation": {"metrics": summarize_records(ablated, class_names=R18_L2_CLASSES, scope=f"{split}:radial_speed_ablation_parent_final"), "records": ablated}, "paired": _paired(full, ablated)}
    payload = {"schema_version": R18_REPORT_SCHEMA, "experiment_id": "V27d-r18", "status": "completed_read_only_source_stratified_radial_speed_diagnostic", "device": str(runtime_device), "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "split_protocol": manifest["split"], "data_scope": {"source": "OriData/excel", "unknown_payload": "not_created", "test": "not_created", "rd_payload": "not_available", "snr_used": False}, "reports": reports, "training": "not_executed", "l3": "not_executed", "not_proven": ["retrained radial-speed causal effect", "cross-source unseen-domain generalization", "Unknown rejection", "deployment performance"]}
    path = artifact_path(root, config.artifacts["diagnostic_report_path"])
    write_json_if_identical(path, seal(payload))
    return {"report_path": str(path), **payload}


__all__ = ["diagnose"]
