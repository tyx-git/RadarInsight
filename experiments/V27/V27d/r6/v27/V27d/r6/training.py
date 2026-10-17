"""Training and threshold selection for V27d-r6 Bird verifiers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256, relative_project_path, write_torch_if_absent_or_identical
from v27.V27d.logging_utils import close_logger, configure_logger, log_event
from v27.V27d.r5.training import load_checkpoint as load_r5_checkpoint

from .features import load_population_cache
from .model import BirdVerifier, build_verifier
from .protocol import R6_CHECKPOINT_SCHEMA, R6_EXPERIMENT_ID, artifact_path, load_context, read_sealed_json, source_path, write_sealed_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r6 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r6 device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)


def _inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {name: tensors[name][indices].to(device) for name in ("summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "padding_mask")}


def _prefix_mask(payload: Mapping[str, Any]) -> torch.Tensor:
    return payload["tensors"]["prefix_route_mask"].bool() & payload["tensors"]["candidate_population"].unsqueeze(1)


def _parent_balanced_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, positive_rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    values = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    active = mask.sum(dim=1)
    eligible = active.gt(0)
    if not bool(eligible.any()):
        return logits.sum() * 0.0, {"bird": 0.0, "false_positive": 0.0}
    parent_loss = (values * mask.to(values.dtype)).sum(dim=1) / active.clamp_min(1)
    bird = eligible & positive_rows
    false_positive = eligible & ~positive_rows
    components = {
        "bird": float(parent_loss[bird].mean().detach().item()) if bool(bird.any()) else 0.0,
        "false_positive": float(parent_loss[false_positive].mean().detach().item()) if bool(false_positive.any()) else 0.0,
    }
    losses = []
    if bool(bird.any()):
        losses.append(parent_loss[bird].mean())
    if bool(false_positive.any()):
        losses.append(parent_loss[false_positive].mean())
    if len(losses) != 2:
        return parent_loss[eligible].mean(), components
    return 0.5 * (losses[0] + losses[1]), components


def _output(model: BirdVerifier, payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        return model(**_inputs(payload, indices, device))


def _r5_threshold(root: Path, config: Mapping[str, Any]) -> float:
    report = read_sealed_json(source_path(root, config["source"]["r5_control_report_path"], must_exist=True))
    return float(report["arms"]["control"]["calibration_transfer"]["bird_knownness_probability"]["threshold_source"]["threshold"])


def _r5_scores(root: Path, config: Mapping[str, Any], role: str) -> torch.Tensor:
    r5_config_path = source_path(root, config["source"]["r5_control_config_path"], must_exist=True)
    r5_config = str(r5_config_path)
    model, _, _ = load_r5_checkpoint(root, arm="control", seed=42, config_path=r5_config)
    r5_cache_path = source_path(root, config["source"]["r5_v2_cache_template"].format(role=role), must_exist=True)
    r5 = torch.load(r5_cache_path, map_location="cpu", weights_only=False)
    tensors = r5["tensors"]
    indices = torch.arange(len(r5["parent_ids"]))
    with torch.inference_mode():
        output = model(
            summary_features=tensors["summary_features"][indices],
            event_trajectory=tensors["event_trajectory"][indices],
            task_facts_sequence=tensors["task_facts_sequence"][indices],
            stable8_causal_features=tensors["stable8_causal_features"][indices],
            padding_mask=tensors["padding_mask"][indices],
        )
    return torch.sigmoid(output["knownness_logit_final"]).cpu()


def _candidate_masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = tensors["bird_positive"].bool() & candidate
    negative = tensors["route_false_positive_negative"].bool() & candidate
    return candidate, positive, negative


def select_threshold(model: BirdVerifier, payload: Mapping[str, Any], *, device: torch.device, config: Mapping[str, Any], baseline_accepted_positive: int) -> dict[str, Any]:
    scores = _output(model, payload, device)["verifier_score_final"].detach().cpu()
    _, positive, negative = _candidate_masks(payload)
    candidates = sorted(set(float(value) for value in scores[positive].tolist()))
    if not candidates:
        raise ValueError("V27d-r6 threshold pool has no Bird positive")
    candidates.append(float(torch.nextafter(torch.tensor(candidates[-1]), torch.tensor(float("inf"))).item()))
    minimum_positive = max(0, int(baseline_accepted_positive) - int(config["selection"]["max_additional_bird_rejections_per_pool"]))
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = scores.ge(threshold)
        accepted_positive = int((accepted & positive).sum())
        if accepted_positive < minimum_positive:
            continue
        false_positive_rejection = float((~accepted[negative]).float().mean()) if bool(negative.any()) else 0.0
        accepted_candidate = accepted & (positive | negative)
        bird_precision = float(accepted_positive / max(1, int(accepted_candidate.sum())))
        record = {"feasible": True, "threshold": threshold, "higher_is_known": True, "accepted_bird": accepted_positive, "bird_support": int(positive.sum()), "accepted_false_positive": int((accepted & negative).sum()), "false_positive_rejection": false_positive_rejection, "bird_precision": bird_precision}
        ranking = (false_positive_rejection, accepted_positive, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        raise RuntimeError("V27d-r6 found no recall-safe threshold")
    return best[1]


def train_arm(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if arm not in {"gru_control", "liquid_treatment"} or int(seed) not in (42, 43, 44):
        raise ValueError("V27d-r6 arm or seed is not registered")
    fit = load_population_cache(_cache_path(root, config, "representation_fit"))
    dev = load_population_cache(_cache_path(root, config, "representation_development"))
    device = _device(config)
    model = build_verifier(config, arm=arm, seed=seed).to(device)
    model.train()
    selected = torch.where(fit["tensors"]["candidate_population"].bool())[0]
    target = fit["tensors"]["knownness_target"]
    mask = _prefix_mask(fit)
    positive_rows = fit["tensors"]["bird_positive"][selected].bool()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(int(seed))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    log_path = artifact_path(root, config["artifacts"]["log_template"].format(arm=arm, seed=seed))
    logger = configure_logger(log_path, name=f"v27d-r6.{arm}.{seed}")
    baseline_scores = _r5_scores(root, config, "representation_development")
    baseline_candidate = dev["tensors"]["candidate_population"].bool()
    baseline_positive = dev["tensors"]["bird_positive"].bool()
    baseline_threshold = _r5_threshold(root, config)
    baseline_accepted_positive = int((baseline_candidate & baseline_positive & baseline_scores.ge(baseline_threshold)).sum())
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        log_event(logger, "training_start", experiment_id="V27d-r6-bird-liquid-verifier", arm=arm, seed=seed, device=str(device), candidate_population=int(selected.numel()))
        for epoch in range(1, epochs + 1):
            model.train()
            order = selected[torch.randperm(selected.numel(), generator=generator)]
            losses: list[float] = []
            components: list[dict[str, float]] = []
            steps = (len(order) + batch_size - 1) // batch_size
            for batch_no, start in enumerate(range(0, len(order), batch_size), start=1):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(fit, indices, device))
                batch_target = target[indices].to(device).unsqueeze(1).expand_as(output["verifier_logit_sequence"])
                batch_mask = mask[indices].to(device)
                loss, component = _parent_balanced_loss(output["verifier_logit_sequence"], batch_target, batch_mask, fit["tensors"]["bird_positive"][indices].bool().to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r6 loss is non-finite")
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                components.append(component)
                if batch_no == 1 or batch_no == steps:
                    log_event(logger, "batch", epoch=epoch, batch=batch_no, steps=steps, loss=losses[-1], gradient_norm=gradient_norm)
            selection = select_threshold(model, dev, device=device, config=config, baseline_accepted_positive=baseline_accepted_positive)
            history.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "loss_components": {key: sum(item[key] for item in components) / len(components) for key in components[0]}, "selection": selection, "elapsed_seconds": time.perf_counter() - started})
            log_event(logger, "development_selection", epoch=epoch, bird_precision=selection["bird_precision"], false_positive_rejection=selection["false_positive_rejection"], accepted_bird=selection["accepted_bird"])
            if best_selection is None or (selection["false_positive_rejection"], selection["accepted_bird"]) > (best_selection["false_positive_rejection"], best_selection["accepted_bird"]):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r6 found no checkpoint")
        model.load_state_dict(best_state, strict=True)
        contract = model.model_contract()
        checkpoint = {"checkpoint_schema": R6_CHECKPOINT_SCHEMA, "experiment_id": R6_EXPERIMENT_ID, "arm": arm, "seed": int(seed), "manifest_sha256": manifest["manifest_sha256"], "fit_cache_sha256": fit["cache_sha256"], "model_contract": contract, "model_contract_sha256": canonical_sha256(contract), "selected_epoch": best_epoch, "selection": best_selection, "history": history, "classification_logits_changed": False, "classification_logits_input_to_head": False, "softmax_input_to_head": False, "label_identity_input_to_head": False, "formal_l3_enabled": False, "test_payload_access_count": 0, "head_state": best_state}
        checkpoint["head_state_sha256"] = canonical_sha256({name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numpy().tobytes().hex()} for name, value in sorted(best_state.items())})
        checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed))
        write_torch_if_absent_or_identical(checkpoint_path, checkpoint)
        summary = {"schema_version": "RADAR-LN3-V27d-r6-training-summary-v1", "experiment_id": R6_EXPERIMENT_ID, "arm": arm, "seed": int(seed), "selected_epoch": best_epoch, "selection": best_selection, "history": history, "checkpoint_path": relative_project_path(root, checkpoint_path), "log_path": relative_project_path(root, log_path), "baseline_accepted_bird": baseline_accepted_positive, "formal_l3_enabled": False}
        summary_path = artifact_path(root, config["artifacts"]["summary_template"].format(arm=arm, seed=seed))
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", selected_epoch=best_epoch, bird_precision=best_selection["bird_precision"], false_positive_rejection=best_selection["false_positive_rejection"])
        return {"status": "completed", "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, log_path), "selection": best_selection}
    finally:
        close_logger(logger)


def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdVerifier, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != R6_CHECKPOINT_SCHEMA or payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r6 checkpoint provenance/schema differs")
    model = build_verifier(config, arm=arm, seed=seed)
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r6 checkpoint contract differs")
    model.load_state_dict(payload["head_state"], strict=True)
    return model, dict(payload), config


__all__ = ["load_checkpoint", "train_arm"]