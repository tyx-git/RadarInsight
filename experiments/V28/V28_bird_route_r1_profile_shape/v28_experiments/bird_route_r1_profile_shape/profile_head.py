"""Small causal profile-shape head for V28 Bird-route r1."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from radar_ln3.runtime import atomic_torch_save, atomic_write_text, canonical_sha256, json_text, relative_project_path, sha256_file

from .features import load_role
from .protocol import CHECKPOINT_SCHEMA, R1Context, resolve_artifact


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_causal_features(values: torch.Tensor, padding: torch.Tensor, *, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-1] != 32 or padding.shape != values.shape[:2]:
        raise ValueError("Bird-route r1 profile tensor shape differs")
    normalized = ((values - mean.view(1, 1, -1)) / scale.view(1, 1, -1)).masked_fill(padding.unsqueeze(-1), 0.0)
    active = (~padding).to(dtype=values.dtype)
    count = active.cumsum(dim=1).clamp_min(1.0).unsqueeze(-1)
    prefix_mean = (normalized * active.unsqueeze(-1)).cumsum(dim=1) / count
    second = (normalized.square() * active.unsqueeze(-1)).cumsum(dim=1) / count
    prefix_std = (second - prefix_mean.square()).clamp_min(0.0).sqrt()
    previous = torch.cat((torch.zeros_like(normalized[:, :1]), normalized[:, :-1]), dim=1)
    previous_active = torch.cat((torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1)
    delta = (normalized - previous).abs() * (active * previous_active).unsqueeze(-1)
    norm = normalized.square().mean(dim=-1).sqrt().unsqueeze(-1)
    delta_norm = delta.mean(dim=-1).sqrt().unsqueeze(-1)
    reliable = normalized[..., 31:32]
    features = torch.cat((normalized, prefix_mean, prefix_std, delta, norm, delta_norm, reliable), dim=-1)
    if features.shape[-1] != 131:
        raise RuntimeError(f"Bird-route r1 causal feature width differs: {features.shape[-1]}")
    return features.masked_fill(padding.unsqueeze(-1), 0.0)


class BirdProfileHead(nn.Module):
    def __init__(self, *, input_dim: int = 131, hidden_dim: int = 64, projection_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(projection_dim)),
            nn.LayerNorm(int(projection_dim)),
            nn.SiLU(),
            nn.Linear(int(projection_dim), 1),
        )

    def forward(self, features: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        logits = self.network(features).squeeze(-1)
        return logits.masked_fill(padding, 0.0)


def _load_role_tensors(context: R1Context, role: str) -> dict[str, Any]:
    payload = load_role(context, role)
    tensors = payload["tensors"]
    return {
        "features": tensors["profile_features"].float(),
        "padding": tensors["padding_mask"].bool(),
        "lengths": tensors["lengths"].long(),
        "labels": np.asarray(payload["label_names"], dtype=object),
        "routes": np.asarray(payload["frozen_l2_route_names"], dtype=object),
        "cache_sha256": payload["cache_sha256"],
    }


def _select_threshold(scores: np.ndarray, labels: np.ndarray, routes: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    route = routes == "BirdTarget"
    positive = route & (labels == "BirdTarget")
    unknown = route & (labels == "UnknownTarget")
    values = np.asarray(scores, dtype=np.float64)
    candidates = sorted({0.0, 1.0, *values[route].tolist(), *[float(np.nextafter(v, 1.0)) for v in values[route].tolist()]})
    floor = float(config["selection"]["bird_recall_floor"])
    maximum = int(config["selection"]["max_correct_known_new_rejections"])
    support_sufficient = int(unknown.sum()) >= int(config["selection"]["minimum_unknown_route_support"])
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        accepted = values >= threshold
        recall = float(accepted[positive].mean()) if positive.any() else 0.0
        rejection = float((~accepted[unknown]).mean()) if unknown.any() else None
        correct_rejections = int((positive & ~accepted).sum())
        rows.append({"threshold": float(threshold), "known_route_recall": recall, "unknown_rejection": rejection, "correct_known_new_rejections": correct_rejections, "unknown_route_support": int(unknown.sum()), "safe": bool(positive.any() and support_sufficient and recall >= floor and correct_rejections <= maximum)})
    safe = [row for row in rows if row["safe"]]
    if safe:
        selected = max(safe, key=lambda row: (float(row["unknown_rejection"] or 0.0), float(row["threshold"])))
        status = "fitted"
    else:
        selected = min(rows, key=lambda row: (max(0.0, floor - row["known_route_recall"]), row["correct_known_new_rejections"], -float(row["unknown_rejection"] or 0.0)))
        status = "fallback_infeasible"
    return {"status": status, "selected": selected, "candidate_count": len(rows), "known_route_support": int(positive.sum()), "unknown_route_support": int(unknown.sum()), "support_sufficient": support_sufficient, "recall_floor": floor, "max_correct_known_new_rejections": maximum}


def _parent_bce(logits: torch.Tensor, target: torch.Tensor, padding: torch.Tensor, parent_mask: torch.Tensor, *, positive_weight: float, min_events: int) -> torch.Tensor:
    active = ~padding
    event_number = active.long().cumsum(dim=1)
    eligible = active & (event_number >= int(min_events)) & parent_mask.unsqueeze(1)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    target_events = target.unsqueeze(1).expand_as(logits)
    values = nn.functional.binary_cross_entropy_with_logits(logits, target_events, pos_weight=logits.new_tensor(float(positive_weight)), reduction="none")
    counts = eligible.sum(dim=1)
    selected = counts > 0
    return ((values * eligible).sum(dim=1) / counts.clamp_min(1))[selected].mean()


def _final_scores(head: BirdProfileHead, payload: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> np.ndarray:
    features = payload["features"].to(device)
    padding = payload["padding"].to(device)
    causal = build_causal_features(features, padding, mean=mean, scale=scale)
    with torch.inference_mode():
        logits = head(causal, padding)
    rows = torch.arange(features.shape[0], device=device)
    lengths = payload["lengths"].to(device)
    return torch.sigmoid(logits[rows, lengths - 1]).cpu().numpy()


def fit_profile_head(context: R1Context, *, seed: int = 42) -> dict[str, Any]:
    _seed(seed)
    device_name = str(context.config["training"]["device"]).lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V28 Bird-route r1 requires CUDA but CUDA is unavailable")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    fit = _load_role_tensors(context, "representation_fit")
    development = _load_role_tensors(context, "representation_development")
    fit_known = fit["labels"] != "UnknownTarget"
    fit_active = (~fit["padding"].numpy()).reshape(-1)
    fit_values = fit["features"].reshape(-1, 32)[fit_active & np.repeat(fit_known, fit["features"].shape[1])]
    mean = fit_values.mean(dim=0).to(device)
    scale = fit_values.std(dim=0).clamp_min(1.0e-4).to(device)
    head = BirdProfileHead(hidden_dim=int(context.config["model"]["hidden_dim"]), projection_dim=int(context.config["model"]["projection_dim"]), dropout=float(context.config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(context.config["training"]["learning_rate"]), weight_decay=float(context.config["training"]["weight_decay"]))
    route_fit = fit["routes"] == "BirdTarget"
    positive_fit = route_fit & (fit["labels"] == "BirdTarget")
    negative_fit = route_fit & (fit["labels"] != "BirdTarget")
    fit_mask = route_fit
    target_fit = torch.from_numpy(positive_fit.astype(np.float32)).to(device)
    fit_parent_mask = torch.from_numpy(fit_mask).to(device)
    positive_count = int(positive_fit.sum())
    negative_count = int(negative_fit.sum())
    if not positive_count or not negative_count:
        raise ValueError("Bird-route r1 fit lacks both Bird and negative route parents")
    positive_weight = negative_count / positive_count
    history: list[dict[str, Any]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(1, int(context.config["training"]["epochs"]) + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        causal = build_causal_features(fit["features"].to(device), fit["padding"].to(device), mean=mean, scale=scale)
        logits = head(causal, fit["padding"].to(device))
        loss = _parent_bce(logits, target_fit, fit["padding"].to(device), fit_parent_mask, positive_weight=positive_weight, min_events=int(context.config["training"]["knownness_min_events"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(context.config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        head.eval()
        scores = _final_scores(head, development, mean, scale, device)
        selection = _select_threshold(scores, development["labels"], development["routes"], context.config)
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "selection": selection})
        states[epoch] = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    best = min(history, key=lambda row: (not bool(row["selection"]["status"] == "fitted"), -float(row["selection"]["selected"].get("unknown_rejection") or 0.0), -float(row["selection"]["selected"].get("known_route_recall") or 0.0), int(row["epoch"])))
    head.load_state_dict(states[int(best["epoch"])], strict=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "experiment_id": "V28-bird-route-r1-profile-shape",
        "task": "bird_route_knownness",
        "seed": int(seed),
        "system_manifest_sha256": context.v28_context.manifest_sha256,
        "v28_registry_sha256": context.v28_registry["registry_sha256"],
        "fit_cache_sha256": fit["cache_sha256"],
        "development_cache_sha256": development["cache_sha256"],
        "model_contract": {"input_dim": 131, "knownness_input": "causal_complete_micro_doppler_profile_shape", "forbidden_inputs": context.config["model"]["forbidden_inputs"], "classifier_backbone_frozen": True},
        "normalization_mean": mean.cpu(),
        "normalization_scale": scale.cpu(),
        "state_dict": head.state_dict(),
        "selected_epoch": int(best["epoch"]),
        "selection": best["selection"],
        "history": history,
        "test_access_count": 0,
    }
    payload["state_dict_sha256"] = canonical_sha256({key: value.detach().cpu().numpy().tobytes().hex() for key, value in sorted(head.state_dict().items())})
    payload["model_contract_sha256"] = canonical_sha256(payload["model_contract"])
    path = resolve_artifact(context, "training_checkpoint_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(path, payload)
    summary = {key: value for key, value in payload.items() if key not in {"state_dict", "normalization_mean", "normalization_scale"}}
    summary["checkpoint_path"] = relative_project_path(context.project_root, path)
    summary["checkpoint_sha256"] = sha256_file(path)
    summary_path = resolve_artifact(context, "training_summary_path")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, json_text(summary))
    return summary


def load_profile_head(context: R1Context) -> tuple[BirdProfileHead, dict[str, Any]]:
    path = resolve_artifact(context, "training_checkpoint_path", must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("system_manifest_sha256") != context.v28_context.manifest_sha256 or payload.get("v28_registry_sha256") != context.v28_registry["registry_sha256"]:
        raise ValueError("Bird-route r1 profile head provenance differs")
    head = BirdProfileHead(hidden_dim=int(context.config["model"]["hidden_dim"]), projection_dim=int(context.config["model"]["projection_dim"]), dropout=float(context.config["model"]["dropout"]))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return head, payload


__all__ = ["BirdProfileHead", "build_causal_features", "fit_profile_head", "load_profile_head"]
