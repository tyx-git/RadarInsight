"""Training and checkpoint contracts for the Bird-route head."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256, relative_project_path, write_torch_if_absent_or_identical

from v27.V27d.logging_utils import close_logger, configure_logger, log_event, progress_interval

from .features import load_population_cache
from .model import BirdRouteKnownnessHead, build_head
from .protocol import artifact_path, load_context, write_sealed_json
from .scoring import select_high_known_threshold


CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r5-bird-head-checkpoint-v1"


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r5 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested != "cpu":
        raise ValueError("V27d-r5 device must be cpu or cuda")
    return torch.device("cpu")


def _cache_path(root: Path, config: Mapping[str, Any], *, arm: str, role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(arm=arm, role=role))


def _head_inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _final_output(model: BirdRouteKnownnessHead, payload: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        output = model(**_head_inputs(payload, indices, device))
    return output["knownness_logit_final"].detach().cpu(), output["projection_final"].detach().cpu()


def _prefix_mask(payload: Mapping[str, Any]) -> torch.Tensor:
    tensors = payload["tensors"]
    length = tensors["padding_mask"].shape[1]
    event_number = torch.arange(1, length + 1).unsqueeze(0)
    return (
        ~tensors["padding_mask"]
        & tensors["quality_target_sequence"]
        & event_number.ge(3)
    )


def _parent_balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, positive_weight: float) -> torch.Tensor:
    values = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=logits.new_tensor(float(positive_weight)),
        reduction="none",
    )
    active = mask.sum(dim=1)
    eligible = active.gt(0)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    means = (values * mask.to(dtype=values.dtype)).sum(dim=1) / active.clamp_min(1)
    return means[eligible].mean()


def _selection(model: BirdRouteKnownnessHead, payload: Mapping[str, Any], device: torch.device, recall_floor: float) -> dict[str, Any]:
    logits, _ = _final_output(model, payload, device)
    tensors = payload["tensors"]
    population = tensors["bird_route_population"].bool()
    positive = tensors["bird_route_positive"].bool() & population
    negative = tensors["bird_route_negative"].bool() & population
    score = torch.sigmoid(logits)
    result = select_high_known_threshold(score, positive, negative, higher_is_known=True, recall_floor=recall_floor)
    result["score_name"] = "bird_knownness_probability"
    result["epoch_population"] = int(population.sum())
    return result


def train_arm(project_root: str | Path, *, arm: str, seed: int = 42, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, _, manifest = load_context(root, config_path)
    if arm not in {"control", "stable8_treatment"}:
        raise ValueError("V27d-r5 arm is invalid")
    if int(seed) != int(config["seed"]):
        raise ValueError("V27d-r5 seed differs from the registered diagnostic seed")
    fit_path = _cache_path(root, config, arm=arm, role="representation_fit")
    dev_path = _cache_path(root, config, arm=arm, role="representation_development")
    fit = load_population_cache(fit_path)
    dev = load_population_cache(dev_path)
    if fit.get("r5_manifest_sha256") != manifest["manifest_sha256"] or dev.get("r5_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r5 population cache provenance differs from the active manifest")
    device = _device(config)
    stable8_enabled = arm == "stable8_treatment"
    model = build_head(stable8_enabled=stable8_enabled, seed=seed).to(device)
    model.train()
    fit_population = fit["tensors"]["bird_route_population"].bool()
    selected = torch.where(fit_population)[0]
    target = fit["tensors"]["bird_knownness_target"]
    prefix_mask = _prefix_mask(fit)
    selected_prefix = prefix_mask[selected]
    selected_target = target[selected].unsqueeze(1).expand_as(selected_prefix)
    selected_values = selected_target.masked_select(selected_prefix)
    positive_count = int(selected_values.sum().item())
    negative_count = int(selected_values.numel() - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("V27d-r5 fit population lacks both Bird and Unknown classes")
    positive_weight = negative_count / positive_count
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    logger_path = artifact_path(root, config["artifacts"]["log_template"].format(arm=arm, seed=seed))
    logger = configure_logger(logger_path, name=f"v27d-r5.{arm}.{seed}")
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    started = time.perf_counter()
    try:
        log_event(logger, "training_start", arm=arm, seed=seed, fit_population=int(selected.numel()), device=str(device))
        for epoch in range(1, epochs + 1):
            model.train()
            order = selected[torch.randperm(selected.numel(), generator=generator)]
            losses: list[float] = []
            steps = (len(order) + batch_size - 1) // batch_size
            interval = progress_interval(steps)
            for batch_no, start in enumerate(range(0, len(order), batch_size), start=1):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_head_inputs(fit, indices, device))
                batch_target = target[indices].to(device).unsqueeze(1).expand_as(output["knownness_logit_sequence"])
                batch_mask = prefix_mask[indices].to(device)
                loss = _parent_balanced_bce(
                    output["knownness_logit_sequence"],
                    batch_target,
                    batch_mask,
                    positive_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r5 loss is non-finite")
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                if batch_no == 1 or batch_no % interval == 0:
                    log_event(logger, "batch", epoch=epoch, batch=batch_no, steps=steps, loss=losses[-1], gradient_norm=gradient_norm)
            selection = _selection(model, dev, device, float(config["selection"]["bird_recall_floor"]))
            record = {
                "epoch": epoch,
                "loss": sum(losses) / max(1, len(losses)),
                "elapsed_seconds": time.perf_counter() - started,
                "selection": selection,
            }
            history.append(record)
            log_event(
                logger,
                "development_selection",
                epoch=epoch,
                support=selection["epoch_population"],
                bird_recall=selection["bird_recall"],
                unknown_rejection=selection["unknown_rejection"],
                feasible=selection["feasible"],
            )
            if selection["feasible"] and (best_selection is None or (selection["unknown_rejection"], selection["bird_recall"]) > (best_selection["unknown_rejection"], best_selection["bird_recall"])):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r5 found no Bird-recall-safe checkpoint")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        checkpoint = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "experiment_id": "V27d-r5-bird-route",
            "arm": arm,
            "seed": int(seed),
            "manifest_sha256": manifest["manifest_sha256"],
            "fit_cache_sha256": fit["cache_sha256"],
            "model_contract": model.model_contract(),
            "model_contract_sha256": canonical_sha256(model.model_contract()),
            "head_state": best_state,
            "selected_epoch": best_epoch,
            "selection": best_selection,
            "history": history,
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
        }
        checkpoint["head_state_sha256"] = canonical_sha256({name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numpy().tobytes().hex()} for name, value in sorted(best_state.items())})
        checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed))
        write_torch_if_absent_or_identical(checkpoint_path, checkpoint)
        summary = {
            "schema_version": "RADAR-LN3-V27d-r5-training-summary-v1",
            "experiment_id": "V27d-r5-bird-route",
            "arm": arm,
            "seed": int(seed),
            "fit_cache_sha256": fit["cache_sha256"],
            "selected_epoch": best_epoch,
            "selection": best_selection,
            "positive_weight": positive_weight,
            "history": history,
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "log_path": relative_project_path(root, logger_path),
            "classification_logits_changed": False,
            "classification_logits_input_to_head": False,
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
        }
        summary_path = artifact_path(root, config["artifacts"]["summary_template"].format(arm=arm, seed=seed))
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", arm=arm, selected_epoch=best_epoch, unknown_rejection=best_selection["unknown_rejection"], bird_recall=best_selection["bird_recall"])
        return {"status": "completed", "arm": arm, "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, logger_path), "selection": best_selection}
    finally:
        close_logger(logger)


def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdRouteKnownnessHead, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, _, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("V27d-r5 checkpoint schema differs")
    if payload.get("manifest_sha256") != manifest["manifest_sha256"] or payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r5 checkpoint provenance differs")
    model = build_head(stable8_enabled=arm == "stable8_treatment", seed=seed)
    if payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("V27d-r5 checkpoint model contract differs")
    state = payload.get("head_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r5 checkpoint state is missing")
    model.load_state_dict(state, strict=True)
    return model, dict(payload), config


__all__ = ["CHECKPOINT_SCHEMA", "load_checkpoint", "train_arm"]