"""Training for the single-variable causal-prefix-consistency objective."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256, relative_project_path, write_torch_if_absent_or_identical

from v27.V27d.logging_utils import close_logger, configure_logger, log_event, progress_interval
from v27.V27d.r5.model import BirdRouteKnownnessHead, build_head

from .features import load_population_cache
from .protocol import V2_CHECKPOINT_SCHEMA, V2_EXPERIMENT_ID, artifact_path, load_context, write_sealed_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r5-v2 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r5-v2 device must be cpu or cuda")


def model_contract(model: BirdRouteKnownnessHead) -> dict[str, Any]:
    """Keep the inherited architecture while sealing the v2 experiment identity."""

    contract = dict(model.model_contract())
    contract.update(
        {
            "schema_version": "RADAR-LN3-V27d-r5-bird-route-v2-head-v1",
            "experiment_id": V2_EXPERIMENT_ID,
            "objective": "causal_prefix_consistency",
            "direct_baseline": "V27d-r5-control",
            "stable8_enabled": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
        }
    )
    return contract


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)


def _head_inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _prefix_mask(payload: Mapping[str, Any], minimum_quality_events: int) -> torch.Tensor:
    tensors = payload["tensors"]
    event_number = torch.arange(1, tensors["padding_mask"].shape[1] + 1).unsqueeze(0)
    return ~tensors["padding_mask"] & tensors["quality_target_sequence"] & event_number.ge(int(minimum_quality_events))


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


def _prefix_consistency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    *,
    positive_weight: float,
    positive_lambda: float,
    unknown_lambda: float,
    unknown_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    probabilities = torch.sigmoid(logits)
    rows = torch.arange(logits.shape[0], device=logits.device)
    final = lengths.to(device=logits.device) - 1
    final_probability = probabilities[rows, final]
    positive_rows = target[:, 0].gt(0.5)
    positive_mask = mask & positive_rows.unsqueeze(1)
    positive_count = positive_mask.sum(dim=1)
    positive_eligible = positive_count.gt(0)
    positive_variance = ((probabilities - final_probability.unsqueeze(1)).square() * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    positive_loss = positive_variance[positive_eligible].mean() if bool(positive_eligible.any()) else logits.sum() * 0.0

    negative_rows = ~positive_rows
    negative_mask = mask & negative_rows.unsqueeze(1)
    prefix_before_final = negative_mask & (torch.arange(logits.shape[1], device=logits.device).unsqueeze(0) < final.unsqueeze(1))
    prefix_count = prefix_before_final.sum(dim=1)
    has_prefix = prefix_count.gt(0)
    # Keep the masked reduction finite even for a one-prefix row.  Rows with
    # no pre-final prefix are excluded below; using -inf here would still
    # create an inf-valued intermediate and can poison autograd on CUDA.
    before_final_sum = (probabilities * prefix_before_final.to(dtype=probabilities.dtype)).sum(dim=1)
    before_final_mean = before_final_sum / prefix_count.clamp_min(1)
    terminal_uplift = nn.functional.relu(final_probability - before_final_mean - float(unknown_margin)).square()
    unknown_loss = terminal_uplift[has_prefix].mean() if bool(has_prefix.any()) else logits.sum() * 0.0

    bce = _parent_balanced_bce(logits, target, mask, positive_weight)
    total = bce + float(positive_lambda) * positive_loss + float(unknown_lambda) * unknown_loss
    return total, {
        "bce": float(bce.detach().item()),
        "positive_prefix_consistency": float(positive_loss.detach().item()),
        "unknown_terminal_uplift": float(unknown_loss.detach().item()),
    }


def _output(model: BirdRouteKnownnessHead, payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        return model(**_head_inputs(payload, indices, device))


def _selection(model: BirdRouteKnownnessHead, payload: Mapping[str, Any], device: torch.device, config: Mapping[str, Any]) -> dict[str, Any]:
    output = _output(model, payload, device)
    score = torch.sigmoid(output["knownness_logit_final"].detach().cpu())
    tensors = payload["tensors"]
    positive = tensors["bird_route_positive"].bool()
    negative = tensors["bird_route_negative"].bool()
    values = sorted(set(float(value) for value in score[positive].tolist()))
    if not values:
        raise ValueError("V27d-r5-v2 development has no Bird positives")
    values.append(float(torch.nextafter(torch.tensor(values[-1]), torch.tensor(float("inf"))).item()))
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    max_rejections = int(config["selection"]["max_correct_bird_new_rejections"])
    floor = float(config["selection"]["bird_recall_floor"])
    for threshold in values:
        accepted = score.ge(threshold)
        recall = float(accepted[positive].float().mean())
        rejection = float((~accepted[negative]).float().mean())
        correct_rejections = int((positive & ~accepted).sum())
        if recall < floor or correct_rejections > max_rejections:
            continue
        record = {
            "feasible": True,
            "threshold": threshold,
            "higher_is_known": True,
            "bird_recall": recall,
            "unknown_rejection": rejection,
            "correct_bird_new_rejections": correct_rejections,
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
            "score_name": "bird_knownness_probability",
        }
        ranking = (rejection, recall, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        return {
            "feasible": False,
            "threshold": None,
            "higher_is_known": True,
            "bird_recall": 0.0,
            "unknown_rejection": 0.0,
            "correct_bird_new_rejections": int(positive.sum()),
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
            "score_name": "bird_knownness_probability",
        }
    return best[1]


def train_control(project_root: str | Path, *, seed: int = 42, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) != int(config["seed"]):
        raise ValueError("V27d-r5-v2 seed differs from registered seed")
    fit = load_population_cache(_cache_path(root, config, "representation_fit"))
    dev = load_population_cache(_cache_path(root, config, "representation_development"))
    if fit["v2_manifest_sha256"] != manifest["manifest_sha256"] or dev["v2_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("V27d-r5-v2 cache provenance differs from active manifest")
    device = _device(config)
    model = build_head(stable8_enabled=False, seed=seed).to(device)
    model.train()
    selected = torch.where(fit["tensors"]["bird_route_population"].bool())[0]
    target = fit["tensors"]["bird_knownness_target"]
    prefix_mask = _prefix_mask(fit, int(config["training"]["minimum_quality_events"]))
    selected_mask = prefix_mask[selected]
    selected_target = target[selected].unsqueeze(1).expand_as(selected_mask)
    selected_values = selected_target.masked_select(selected_mask)
    positives = int(selected_values.sum())
    negatives = int(selected_values.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("V27d-r5-v2 fit population lacks both classes")
    positive_weight = negatives / positives
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(int(seed))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    log_path = artifact_path(root, config["artifacts"]["log_path"])
    logger = configure_logger(log_path, name="v27d-r5-bird-route-v2.control.42")
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        log_event(logger, "training_start", experiment_id="V27d-r5-bird-route-v2", seed=seed, fit_population=int(selected.numel()), device=str(device))
        for epoch in range(1, epochs + 1):
            model.train()
            order = selected[torch.randperm(selected.numel(), generator=generator)]
            losses: list[float] = []
            components: list[dict[str, float]] = []
            steps = (len(order) + batch_size - 1) // batch_size
            interval = progress_interval(steps)
            for batch_no, start in enumerate(range(0, len(order), batch_size), start=1):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_head_inputs(fit, indices, device))
                batch_target = target[indices].to(device).unsqueeze(1).expand_as(output["knownness_logit_sequence"])
                batch_mask = prefix_mask[indices].to(device)
                loss, component = _prefix_consistency_loss(
                    output["knownness_logit_sequence"],
                    batch_target,
                    batch_mask,
                    fit["tensors"]["lengths"][indices],
                    positive_weight=positive_weight,
                    positive_lambda=float(config["training"]["prefix_consistency_weight"]),
                    unknown_lambda=float(config["training"]["unknown_terminal_uplift_weight"]),
                    unknown_margin=float(config["training"]["unknown_terminal_uplift_margin"]),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r5-v2 loss is non-finite")
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                components.append(component)
                if batch_no == 1 or batch_no % interval == 0:
                    log_event(logger, "batch", epoch=epoch, batch=batch_no, steps=steps, loss=losses[-1], gradient_norm=gradient_norm)
            selection = _selection(model, dev, device, config)
            history.append({
                "epoch": epoch,
                "loss": sum(losses) / max(1, len(losses)),
                "loss_components": {name: sum(item[name] for item in components) / max(1, len(components)) for name in components[0]},
                "elapsed_seconds": time.perf_counter() - started,
                "selection": selection,
            })
            log_event(logger, "development_selection", epoch=epoch, bird_recall=selection["bird_recall"], unknown_rejection=selection["unknown_rejection"], feasible=selection["feasible"])
            if selection["feasible"] and (best_selection is None or (selection["unknown_rejection"], selection["bird_recall"]) > (best_selection["unknown_rejection"], best_selection["bird_recall"])):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r5-v2 found no Bird-recall-safe checkpoint")
        model.load_state_dict(best_state, strict=True)
        # Threshold selection is an artifact contract. Replay the selected
        # state on CPU before sealing the checkpoint so summary/evaluation do
        # not differ only because CUDA and CPU sigmoid values round at a
        # nearby candidate threshold.
        canonical_model = build_head(stable8_enabled=False, seed=seed)
        canonical_model.load_state_dict(best_state, strict=True)
        canonical_model.eval()
        canonical_selection = _selection(canonical_model, dev, torch.device("cpu"), config)
        if not canonical_selection["feasible"]:
            raise RuntimeError("V27d-r5-v2 CPU canonical replay found no feasible selection")
        contract = model_contract(model)
        checkpoint = {
            "checkpoint_schema": V2_CHECKPOINT_SCHEMA,
            "experiment_id": "V27d-r5-bird-route-v2",
            "direct_baseline": "V27d-r5-control",
            "seed": int(seed),
            "manifest_sha256": manifest["manifest_sha256"],
            "fit_cache_sha256": fit["cache_sha256"],
            "model_contract": contract,
            "model_contract_sha256": canonical_sha256(contract),
            "objective": {
                "name": "causal_prefix_consistency",
                "prefix_consistency_weight": float(config["training"]["prefix_consistency_weight"]),
                "unknown_terminal_uplift_weight": float(config["training"]["unknown_terminal_uplift_weight"]),
                "unknown_terminal_uplift_margin": float(config["training"]["unknown_terminal_uplift_margin"]),
            },
            "head_state": best_state,
            "selected_epoch": best_epoch,
            "selection": canonical_selection,
            "history": history,
            "classification_logits_changed": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
        }
        checkpoint["head_state_sha256"] = canonical_sha256({name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numpy().tobytes().hex()} for name, value in sorted(best_state.items())})
        checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_path"])
        write_torch_if_absent_or_identical(checkpoint_path, checkpoint)
        summary = {
            "schema_version": "RADAR-LN3-V27d-r5-bird-route-v2-training-summary-v1",
            "experiment_id": "V27d-r5-bird-route-v2",
            "direct_baseline": "V27d-r5-control",
            "seed": int(seed),
            "selected_epoch": best_epoch,
            "selection": canonical_selection,
            "objective": {
                "name": "causal_prefix_consistency",
                "prefix_consistency_weight": float(config["training"]["prefix_consistency_weight"]),
                "unknown_terminal_uplift_weight": float(config["training"]["unknown_terminal_uplift_weight"]),
                "unknown_terminal_uplift_margin": float(config["training"]["unknown_terminal_uplift_margin"]),
            },
            "positive_weight": positive_weight,
            "history": history,
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "log_path": relative_project_path(root, log_path),
            "classification_logits_changed": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
        }
        summary_path = artifact_path(root, config["artifacts"]["summary_path"])
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", selected_epoch=best_epoch, bird_recall=canonical_selection["bird_recall"], unknown_rejection=canonical_selection["unknown_rejection"])
        return {"status": "completed", "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, log_path), "selection": canonical_selection}
    finally:
        close_logger(logger)


def load_checkpoint(project_root: str | Path, *, seed: int = 42, config_path: str | Path | None = None) -> tuple[BirdRouteKnownnessHead, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_path"], must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != V2_CHECKPOINT_SCHEMA:
        raise ValueError("V27d-r5-v2 checkpoint schema differs")
    if payload.get("manifest_sha256") != manifest["manifest_sha256"] or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r5-v2 checkpoint provenance differs")
    model = build_head(stable8_enabled=False, seed=seed)
    contract = model_contract(model)
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r5-v2 checkpoint model contract differs")
    state = payload.get("head_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r5-v2 checkpoint state is missing")
    model.load_state_dict(state, strict=True)
    return model, dict(payload), config


__all__ = ["load_checkpoint", "model_contract", "train_control"]