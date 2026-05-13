from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data.labels import DEFAULT_VOCAB
from .experts import (
    ConventionalAirExpert,
    ExpertOutput,
    GroundTargetExpert,
    LOW_ALTITUDE_CLASSES,
    LowAltitudeFusionOutput,
    LowAltitudeSuspectExpert,
    RuleBasedHighAltitudeExpert,
)
from .router import DeepGatingRouter, RouteDecision, ROUTE_NAMES
from .trajectory import TrackRepresentation, TrackRepresentationBuilder


GLOBAL_LABEL_TO_INDEX = DEFAULT_VOCAB.label_to_index
UNKNOWN_INDEX = GLOBAL_LABEL_TO_INDEX["UnknownTarget"]
LOW_LOCAL_UNKNOWN_INDEX = LOW_ALTITUDE_CLASSES.index("UnknownTarget")


@dataclass(frozen=True)
class RadarSystemOutput:
    representation: TrackRepresentation
    routing: RouteDecision
    rd_features: torch.Tensor
    low_altitude_detail: LowAltitudeFusionOutput
    conventional_air_detail: ExpertOutput
    high_altitude_detail: ExpertOutput
    ground_detail: ExpertOutput
    low_altitude_logits: torch.Tensor
    conventional_air_logits: torch.Tensor
    high_altitude_logits: torch.Tensor
    ground_logits: torch.Tensor
    reroute_logits: torch.Tensor | None = None
    reroute_diagnostics: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class RerouteConfig:
    max_passes: int = 3
    penalty: float = 3.0
    max_total_penalty: float = 6.0
    low_altitude_conf_threshold: float = 0.35
    conventional_air_conf_threshold: float = 0.35
    ground_conf_threshold: float = 0.50


class RadarClassificationSystem(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        physical_input_dim: int,
        model_dim: int = 128,
        dropout: float = 0.1,
        low_altitude_cosface_scale: float = 16.0,
        low_altitude_cosface_margin: float = 0.2,
        low_altitude_reject_threshold: float = 0.35,
        low_altitude_use_rd_patch_cnn: bool = True,
        low_altitude_rd_patch_input_dim: int = 3,
        rd_bird_logit_scale: float = 0.75,
        rd_unknown_logit_scale: float = 0.75,
        rd_clutter_logit_scale: float = 0.75,
        rd_residual_logit_scale: float = 0.9,
        rd_unknown_clutter_logit_scale: float = 1.1,
        rd_clutter_morphology_logit_scale: float = 0.9,
        low_altitude_joint_agreement_prior_scale: float = 0.25,
        low_altitude_rd_specialist_residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        rd_dim = 16
        fused_dim = model_dim + 32
        self.representation_builder = TrackRepresentationBuilder(
            sequence_input_dim=sequence_input_dim,
            physical_input_dim=physical_input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.rd_proj = nn.Sequential(
            nn.Linear(6, rd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rd_dim, rd_dim),
            nn.LayerNorm(rd_dim),
        )
        self.router = DeepGatingRouter(input_dim=physical_input_dim)
        self.low_altitude_expert = LowAltitudeSuspectExpert(
            input_dim=fused_dim,
            rd_input_dim=4,
            rd_patch_input_dim=low_altitude_rd_patch_input_dim,
            rd_context_dim=rd_dim,
            traj_sequence_dim=model_dim,
            dropout=dropout,
            cosface_scale=low_altitude_cosface_scale,
            cosface_margin=low_altitude_cosface_margin,
            reject_threshold=low_altitude_reject_threshold,
            use_rd_patch_cnn=low_altitude_use_rd_patch_cnn,
            rd_bird_logit_scale=rd_bird_logit_scale,
            rd_unknown_logit_scale=rd_unknown_logit_scale,
            rd_clutter_logit_scale=rd_clutter_logit_scale,
            rd_residual_logit_scale=rd_residual_logit_scale,
            rd_unknown_clutter_logit_scale=rd_unknown_clutter_logit_scale,
            rd_clutter_morphology_logit_scale=rd_clutter_morphology_logit_scale,
            joint_agreement_prior_scale=low_altitude_joint_agreement_prior_scale,
            rd_specialist_residual_scale=low_altitude_rd_specialist_residual_scale,
        )
        self.conventional_air_expert = ConventionalAirExpert(sequence_dim=model_dim, physical_dim=physical_input_dim, dropout=dropout)
        self.high_altitude_expert = RuleBasedHighAltitudeExpert()
        self.ground_expert = GroundTargetExpert(deep_dim=model_dim, raw_physical_dim=10, dropout=dropout)

    def _rd_summary(
        self,
        sequence_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rd_features is not None and rd_features.ndim == 4:
            amp_range = rd_features[..., 0].mean(dim=1)
            amp_dopp = rd_features[..., 2].mean(dim=1)
            return torch.stack(
                [
                    amp_range.mean(dim=-1),
                    amp_range.max(dim=-1).values,
                    amp_range.min(dim=-1).values,
                    amp_dopp.mean(dim=-1),
                    amp_dopp.std(dim=-1),
                    rd_is_multimodal.float()
                    if rd_is_multimodal is not None
                    else torch.ones(
                        (sequence_features.shape[0],),
                        dtype=sequence_features.dtype,
                        device=sequence_features.device,
                    ),
                ],
                dim=-1,
            )
        if rd_features is not None:
            return rd_features.mean(dim=1)
        return torch.zeros(
            (sequence_features.shape[0], 6),
            dtype=sequence_features.dtype,
            device=sequence_features.device,
        )

    def _forward_experts(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        rd_patch_features: torch.Tensor | None = None,
        rd_deep_patch_features: torch.Tensor | None = None,
        low_altitude_targets: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        rd_deep_patch_padding_mask: torch.Tensor | None = None,
        source_names: list[str] | None = None,
        enable_low_altitude_reject: bool = True,
        detach_ground_features: bool = False,
    ) -> RadarSystemOutput:
        representation = self.representation_builder(
            sequence_features=sequence_features,
            physical_features=physical_features,
            padding_mask=padding_mask,
        )
        rd_summary = self._rd_summary(
            sequence_features=sequence_features,
            rd_features=rd_features,
            rd_is_multimodal=rd_is_multimodal,
        )
        rd_embedding = self.rd_proj(rd_summary)
        routing = self.router(
            representation.raw_physical_features,
            rule_features=representation.raw_physical_features,
        )
        low_altitude = self.low_altitude_expert(
            representation.fused_features,
            rd_features=rd_features,
            traj_sequence_features=representation.sequence_features,
            rd_patch_features=rd_patch_features,
            rd_deep_patch_features=rd_deep_patch_features,
            rd_context=rd_embedding,
            low_altitude_targets=low_altitude_targets,
            rd_is_multimodal=rd_is_multimodal,
            traj_padding_mask=padding_mask,
            rd_padding_mask=rd_padding_mask,
            rd_patch_padding_mask=rd_patch_padding_mask,
            rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
            enable_reject=enable_low_altitude_reject,
            inference_mode="v2_joint",
        )
        conventional_air = self.conventional_air_expert(
            representation.sequence_features,
            representation.raw_physical_features,
            padding_mask=padding_mask,
            source_names=source_names,
        )
        high_altitude = self.high_altitude_expert(
            height_mean=physical_features[:, 0],
            speed_mean=physical_features[:, 3],
        )
        ground_deep_features = representation.deep_features.detach() if detach_ground_features else representation.deep_features
        ground = self.ground_expert(ground_deep_features, representation.raw_physical_features)
        return RadarSystemOutput(
            representation=representation,
            routing=routing,
            rd_features=rd_summary,
            low_altitude_detail=low_altitude,
            conventional_air_detail=conventional_air,
            high_altitude_detail=high_altitude,
            ground_detail=ground,
            low_altitude_logits=low_altitude.logits,
            conventional_air_logits=conventional_air.logits,
            high_altitude_logits=high_altitude.logits,
            ground_logits=ground.logits,
        )

    def _forward_reroute(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        rd_patch_features: torch.Tensor | None = None,
        rd_deep_patch_features: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        rd_deep_patch_padding_mask: torch.Tensor | None = None,
        source_names: list[str] | None = None,
        enable_low_altitude_reject: bool = True,
        reroute_config: RerouteConfig | None = None,
        num_global_classes: int | None = None,
    ) -> RadarSystemOutput:
        config = reroute_config or RerouteConfig()
        representation = self.representation_builder(
            sequence_features=sequence_features,
            physical_features=physical_features,
            padding_mask=padding_mask,
        )
        rd_summary = self._rd_summary(
            sequence_features=sequence_features,
            rd_features=rd_features,
            rd_is_multimodal=rd_is_multimodal,
        )
        rd_embedding = self.rd_proj(rd_summary)
        routing = self.router(
            representation.raw_physical_features,
            rule_features=representation.raw_physical_features,
        )
        batch_size = sequence_features.shape[0]
        num_classes = int(num_global_classes or len(DEFAULT_VOCAB.labels))
        route_logits = routing.route_logits.detach().clone()
        device = route_logits.device
        dtype = route_logits.dtype
        final_logits = torch.full((batch_size, num_classes), fill_value=-6.0, dtype=dtype, device=device)
        fallback_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        completed_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        reroute_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
        selected_route = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        initial_route = routing.route_logits.argmax(dim=-1)
        total_penalty = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        last_confidence = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        low_deep_used_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        low_deep_accept_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        low_deep_helped_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        low_deep_harmed_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        low_pass1_pred = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        low_deep_pred = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        low_final_pred = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        low_deep_accept_reason_code = torch.zeros((batch_size,), dtype=torch.long, device=device)
        reject_reason_code = torch.zeros((batch_size,), dtype=torch.long, device=device)
        next_route_override = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        route_path = torch.full((batch_size, int(max(1, config.max_passes)) + 1), fill_value=-1, dtype=torch.long, device=device)
        route_evaluated = torch.zeros((len(ROUTE_NAMES),), dtype=torch.bool, device=device)

        low_altitude: LowAltitudeFusionOutput | None = None
        low_altitude_deep: LowAltitudeFusionOutput | None = None
        conventional_air: ExpertOutput | None = None
        high_altitude: ExpertOutput | None = None
        ground: ExpertOutput | None = None

        def get_low_altitude() -> LowAltitudeFusionOutput:
            nonlocal low_altitude
            if low_altitude is None:
                route_evaluated[ROUTE_NAMES.index("low_altitude_suspect")] = True
                low_altitude = self.low_altitude_expert(
                    representation.fused_features,
                    rd_features=rd_features,
                    traj_sequence_features=representation.sequence_features,
                    rd_patch_features=rd_patch_features,
                    rd_deep_patch_features=rd_deep_patch_features,
                    rd_context=rd_embedding,
                    low_altitude_targets=None,
                    rd_is_multimodal=rd_is_multimodal,
                    traj_padding_mask=padding_mask,
                    rd_padding_mask=rd_padding_mask,
                    rd_patch_padding_mask=rd_patch_padding_mask,
                    rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
                    enable_reject=enable_low_altitude_reject,
                    inference_mode="v2_joint",
                )
            return low_altitude

        def get_low_altitude_deep() -> LowAltitudeFusionOutput:
            nonlocal low_altitude_deep
            if low_altitude_deep is None:
                route_evaluated[ROUTE_NAMES.index("low_altitude_suspect")] = True
                low_altitude_deep = self.low_altitude_expert(
                    representation.fused_features,
                    rd_features=rd_features,
                    traj_sequence_features=representation.sequence_features,
                    rd_patch_features=rd_patch_features,
                    rd_deep_patch_features=rd_deep_patch_features,
                    rd_context=rd_embedding,
                    low_altitude_targets=None,
                    rd_is_multimodal=rd_is_multimodal,
                    traj_padding_mask=padding_mask,
                    rd_padding_mask=rd_padding_mask,
                    rd_patch_padding_mask=rd_patch_padding_mask,
                    rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
                    enable_reject=enable_low_altitude_reject,
                    inference_mode="deep_reroute",
                )
            return low_altitude_deep

        def get_conventional_air() -> ExpertOutput:
            nonlocal conventional_air
            if conventional_air is None:
                route_evaluated[ROUTE_NAMES.index("conventional_air")] = True
                conventional_air = self.conventional_air_expert(
                    representation.sequence_features,
                    representation.raw_physical_features,
                    padding_mask=padding_mask,
                    source_names=source_names,
                )
            return conventional_air

        def get_high_altitude() -> ExpertOutput:
            nonlocal high_altitude
            if high_altitude is None:
                route_evaluated[ROUTE_NAMES.index("high_altitude_filter")] = True
                high_altitude = self.high_altitude_expert(
                    height_mean=physical_features[:, 0],
                    speed_mean=physical_features[:, 3],
                )
            return high_altitude

        def get_ground() -> ExpertOutput:
            nonlocal ground
            if ground is None:
                route_evaluated[ROUTE_NAMES.index("ground_target")] = True
                ground = self.ground_expert(representation.deep_features, representation.raw_physical_features)
            return ground

        def acceptance(route_idx: int, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            route_name = ROUTE_NAMES[route_idx]
            if route_name == "low_altitude_suspect":
                detail = get_low_altitude()
                confidence, accepted = self.low_altitude_expert.acceptance(
                    detail,
                    threshold=float(config.low_altitude_conf_threshold),
                )
                rejected = active_mask & ~accepted
                if bool(rejected.any()):
                    deep_detail = get_low_altitude_deep()
                    deep_confidence, deep_accepted = self.low_altitude_expert.acceptance(
                        deep_detail,
                        threshold=float(config.low_altitude_conf_threshold),
                    )
                    pass1_pred = detail.logits.argmax(dim=-1)
                    deep_pred = deep_detail.logits.argmax(dim=-1)
                    pass1_unknown = pass1_pred == LOW_LOCAL_UNKNOWN_INDEX
                    deep_unknown = deep_pred == LOW_LOCAL_UNKNOWN_INDEX
                    deep_confident = deep_confidence >= float(config.low_altitude_conf_threshold)
                    low_deep_used_mask[:] = low_deep_used_mask | rejected
                    low_deep_accept_mask[:] = low_deep_accept_mask | (rejected & deep_accepted)
                    low_pass1_pred[:] = torch.where(rejected, pass1_pred, low_pass1_pred)
                    low_deep_pred[:] = torch.where(rejected, deep_pred, low_deep_pred)
                    low_deep_helped_mask[:] = low_deep_helped_mask | (rejected & pass1_unknown & deep_accepted & ~deep_unknown)
                    low_deep_harmed_mask[:] = low_deep_harmed_mask | (rejected & ~pass1_unknown & deep_accepted & deep_unknown)
                    reason = torch.where(
                        deep_unknown & deep_accepted,
                        torch.full_like(low_deep_accept_reason_code, 2),
                        torch.where(
                            deep_confident & deep_accepted,
                            torch.full_like(low_deep_accept_reason_code, 1),
                            torch.zeros_like(low_deep_accept_reason_code),
                        ),
                    )
                    low_deep_accept_reason_code[:] = torch.where(rejected & deep_accepted, reason, low_deep_accept_reason_code)
                    confidence = torch.where(rejected, deep_confidence, confidence)
                    accepted = accepted | (rejected & deep_accepted)
                    low_reject_score = detail.reject_score
                    if low_reject_score is not None:
                        unknown_trigger = (detail.reject_reasons or {}).get("unknown_trigger")
                        confusion_trigger = (detail.reject_reasons or {}).get("confusion_pair_trigger")
                        evidence_trigger = (detail.reject_reasons or {}).get("evidence_conflict_trigger")
                        low_margin_trigger = (detail.reject_reasons or {}).get("low_margin_trigger")
                        route_reason = torch.where(
                            unknown_trigger.bool() if unknown_trigger is not None else torch.zeros_like(rejected),
                            torch.full_like(reject_reason_code, 1),
                            torch.where(
                                confusion_trigger.bool() if confusion_trigger is not None else torch.zeros_like(rejected),
                                torch.full_like(reject_reason_code, 2),
                                torch.where(
                                    evidence_trigger.bool() if evidence_trigger is not None else torch.zeros_like(rejected),
                                    torch.full_like(reject_reason_code, 3),
                                    torch.where(
                                        low_margin_trigger.bool() if low_margin_trigger is not None else torch.zeros_like(rejected),
                                        torch.full_like(reject_reason_code, 4),
                                        torch.full_like(reject_reason_code, 5),
                                    ),
                                ),
                            ),
                        )
                        reject_reason_code[:] = torch.where(rejected & (reject_reason_code == 0), route_reason, reject_reason_code)
                return confidence, accepted
            if route_name == "conventional_air":
                detail = get_conventional_air()
                confidence, accepted = self.conventional_air_expert.acceptance(
                    detail,
                    threshold=float(config.conventional_air_conf_threshold),
                )
                rejected = active_mask & ~accepted
                reject_reason_code[:] = torch.where(rejected & (reject_reason_code == 0), torch.full_like(reject_reason_code, 6), reject_reason_code)
                return confidence, accepted
            if route_name == "ground_target":
                detail = get_ground()
                confidence, accepted = self.ground_expert.acceptance(
                    detail,
                    threshold=float(config.ground_conf_threshold),
                )
                rejected = active_mask & ~accepted
                reject_reason_code[:] = torch.where(rejected & (reject_reason_code == 0), torch.full_like(reject_reason_code, 7), reject_reason_code)
                return confidence, accepted
            detail = get_high_altitude()
            confidence, accepted = self.high_altitude_expert.acceptance(detail)
            rejected = active_mask & ~accepted
            reject_reason_code[:] = torch.where(rejected & (reject_reason_code == 0), torch.full_like(reject_reason_code, 8), reject_reason_code)
            return confidence, accepted

        def next_route_for_reject(route_idx: int, reject_mask: torch.Tensor) -> torch.Tensor:
            next_route = torch.full_like(selected_route, fill_value=-1)
            if not bool(reject_mask.any()):
                return next_route
            low_idx = ROUTE_NAMES.index("low_altitude_suspect")
            air_idx = ROUTE_NAMES.index("conventional_air")
            ground_idx = ROUTE_NAMES.index("ground_target")
            if route_idx == low_idx:
                aircraft_detail = get_conventional_air()
                aircraft_aux = aircraft_detail.auxiliary or {}
                aircraft_similarity = aircraft_aux.get("aircraft_similarity")
                aircraft_threshold = aircraft_aux.get("aircraft_threshold")
                if aircraft_similarity is None:
                    aircraft_similarity = aircraft_detail.logits[:, 1] if aircraft_detail.logits.shape[1] > 1 else aircraft_detail.logits[:, 0]
                if aircraft_threshold is None:
                    aircraft_threshold = torch.full_like(aircraft_similarity, float(config.conventional_air_conf_threshold))
                aircraft_evidence = (aircraft_similarity - aircraft_threshold) >= 0.0
                ground_detail = get_ground()
                ground_probability = (ground_detail.auxiliary or {}).get("groundness_probability")
                if ground_probability is None:
                    ground_probability = torch.zeros((batch_size,), dtype=dtype, device=device)
                ground_evidence = ground_probability >= float(config.ground_conf_threshold)
                next_route = torch.where(
                    reject_mask & aircraft_evidence,
                    torch.full_like(next_route, air_idx),
                    next_route,
                )
                next_route = torch.where(
                    reject_mask & (next_route < 0) & ground_evidence,
                    torch.full_like(next_route, ground_idx),
                    next_route,
                )
            elif route_idx == air_idx:
                ground_detail = get_ground()
                ground_probability = (ground_detail.auxiliary or {}).get("groundness_probability")
                if ground_probability is None:
                    ground_probability = torch.zeros((batch_size,), dtype=dtype, device=device)
                ground_evidence = ground_probability >= float(config.ground_conf_threshold)
                has_rd = (
                    rd_is_multimodal.bool()
                    if rd_is_multimodal is not None
                    else torch.ones((batch_size,), dtype=torch.bool, device=device)
                )
                next_route = torch.where(
                    reject_mask & has_rd,
                    torch.full_like(next_route, low_idx),
                    next_route,
                )
                next_route = torch.where(
                    reject_mask & (next_route < 0) & ground_evidence,
                    torch.full_like(next_route, ground_idx),
                    next_route,
                )
            elif route_idx == ground_idx:
                aircraft_detail = get_conventional_air()
                aircraft_aux = aircraft_detail.auxiliary or {}
                aircraft_similarity = aircraft_aux.get("aircraft_similarity")
                aircraft_threshold = aircraft_aux.get("aircraft_threshold")
                if aircraft_similarity is None:
                    aircraft_similarity = aircraft_detail.logits[:, 1] if aircraft_detail.logits.shape[1] > 1 else aircraft_detail.logits[:, 0]
                if aircraft_threshold is None:
                    aircraft_threshold = torch.full_like(aircraft_similarity, float(config.conventional_air_conf_threshold))
                aircraft_evidence = (aircraft_similarity - aircraft_threshold) >= 0.0
                has_rd = (
                    rd_is_multimodal.bool()
                    if rd_is_multimodal is not None
                    else torch.ones((batch_size,), dtype=torch.bool, device=device)
                )
                next_route = torch.where(
                    reject_mask & aircraft_evidence,
                    torch.full_like(next_route, air_idx),
                    next_route,
                )
                next_route = torch.where(
                    reject_mask & (next_route < 0) & has_rd,
                    torch.full_like(next_route, low_idx),
                    next_route,
                )
            return next_route

        def global_logits_for_route(route_idx: int) -> torch.Tensor:
            logits = torch.full((batch_size, num_classes), fill_value=-6.0, dtype=dtype, device=device)
            route_name = ROUTE_NAMES[route_idx]
            if route_name == "low_altitude_suspect":
                local = get_low_altitude().logits
                if low_altitude_deep is not None:
                    deep_local = get_low_altitude_deep().logits
                    local = torch.where(low_deep_used_mask.unsqueeze(-1), deep_local, local)
                logits[:, GLOBAL_LABEL_TO_INDEX["DroneTarget"]] = local[:, 0]
                logits[:, GLOBAL_LABEL_TO_INDEX["BirdTarget"]] = local[:, 1]
                logits[:, GLOBAL_LABEL_TO_INDEX["BalloonTarget"]] = local[:, 2]
                logits[:, GLOBAL_LABEL_TO_INDEX["ClutterTarget"]] = local[:, 3]
                logits[:, GLOBAL_LABEL_TO_INDEX["UnknownTarget"]] = local[:, 4]
            elif route_name == "conventional_air":
                local = get_conventional_air().logits
                aircraft_column = 1 if local.shape[1] > 1 else 0
                logits[:, GLOBAL_LABEL_TO_INDEX["AircraftTarget"]] = local[:, aircraft_column]
            elif route_name == "high_altitude_filter":
                local = get_high_altitude().logits
                aircraft_column = 1 if local.shape[1] > 1 else 0
                high_accept = local[:, aircraft_column] > local[:, 0]
                logits[high_accept, GLOBAL_LABEL_TO_INDEX["AircraftTarget"]] = local[high_accept, aircraft_column]
                logits[~high_accept, GLOBAL_LABEL_TO_INDEX["UnknownTarget"]] = 6.0
            elif route_name == "ground_target":
                local = get_ground().logits
                logits[:, GLOBAL_LABEL_TO_INDEX["PeopleTarget"]] = local[:, 0]
                logits[:, GLOBAL_LABEL_TO_INDEX["CarTarget"]] = local[:, 1]
                logits[:, GLOBAL_LABEL_TO_INDEX["ShipTarget"]] = local[:, 2]
            return logits

        rule_route = routing.route_indices
        high_idx = ROUTE_NAMES.index("high_altitude_filter")
        ground_idx = ROUTE_NAMES.index("ground_target")
        rule_terminal = routing.override_mask & ((rule_route == high_idx) | (rule_route == ground_idx))
        if bool(rule_terminal.any()):
            completed_mask |= rule_terminal
            selected_route = torch.where(rule_terminal, rule_route, selected_route)

        for pass_idx in range(max(1, int(config.max_passes))):
            pending_mask = ~completed_mask & ~fallback_mask
            if not bool(pending_mask.any()):
                break
            route_choice = torch.where(next_route_override >= 0, next_route_override, route_logits.argmax(dim=-1))
            route_path[:, pass_idx] = torch.where(pending_mask, route_choice, route_path[:, pass_idx])
            next_route_override = torch.full_like(next_route_override, fill_value=-1)
            for route_idx in range(len(ROUTE_NAMES)):
                route_mask = pending_mask & (route_choice == route_idx)
                if not bool(route_mask.any()):
                    continue
                confidence, accepted = acceptance(route_idx, route_mask)
                accept_mask = route_mask & accepted
                reject_mask = route_mask & ~accepted
                if bool(accept_mask.any()):
                    completed_mask |= accept_mask
                    selected_route = torch.where(
                        accept_mask,
                        torch.full_like(selected_route, route_idx),
                        selected_route,
                    )
                    last_confidence = torch.where(accept_mask, confidence, last_confidence)
                if bool(reject_mask.any()):
                    reroute_count[reject_mask] += 1
                    directed_next_route = next_route_for_reject(route_idx, reject_mask)
                    next_route_override = torch.where(
                        reject_mask & (directed_next_route >= 0),
                        directed_next_route,
                        next_route_override,
                    )
                    penalty = torch.full_like(total_penalty, float(config.penalty))
                    remaining = float(config.max_total_penalty) - total_penalty
                    clipped_penalty = torch.minimum(penalty, remaining.clamp_min(0.0))
                    route_logits[reject_mask, route_idx] -= clipped_penalty[reject_mask].to(dtype=dtype)
                    total_penalty[reject_mask] += clipped_penalty[reject_mask]
                    last_confidence = torch.where(reject_mask, confidence, last_confidence)
            no_directed_next = (~completed_mask) & (~fallback_mask) & (next_route_override < 0)
            maxed_penalty = total_penalty >= float(config.max_total_penalty)
            exhausted = reroute_count >= int(config.max_passes)
            fallback_mask |= (~completed_mask) & (maxed_penalty | exhausted | no_directed_next)

        fallback_mask |= ~completed_mask
        lazy_expert_eval_count = int(route_evaluated.sum().item())
        for route_idx in range(len(ROUTE_NAMES)):
            route_mask = completed_mask & (selected_route == route_idx)
            if bool(route_mask.any()):
                route_logits_full = global_logits_for_route(route_idx)
                final_logits[route_mask] = route_logits_full[route_mask]
        if bool(fallback_mask.any()):
            final_logits[fallback_mask] = -6.0
            final_logits[fallback_mask, UNKNOWN_INDEX] = 6.0

        # Materialize missing experts after the reroute decision so existing
        # evaluation diagnostics keep working without influencing reroute.
        low_altitude = get_low_altitude()
        if low_altitude_deep is None:
            low_altitude_deep = low_altitude
        conventional_air = get_conventional_air()
        high_altitude = get_high_altitude()
        ground = get_ground()
        low_reject_reasons = low_altitude.reject_reasons or {}
        low_deep_reject_reasons = low_altitude_deep.reject_reasons or {}
        low_route_idx = ROUTE_NAMES.index("low_altitude_suspect")
        low_final_route_mask = completed_mask & (selected_route == low_route_idx)
        if bool(low_final_route_mask.any()):
            low_final_pred = torch.where(
                low_final_route_mask,
                final_logits[:, [
                    GLOBAL_LABEL_TO_INDEX["DroneTarget"],
                    GLOBAL_LABEL_TO_INDEX["BirdTarget"],
                    GLOBAL_LABEL_TO_INDEX["BalloonTarget"],
                    GLOBAL_LABEL_TO_INDEX["ClutterTarget"],
                    GLOBAL_LABEL_TO_INDEX["UnknownTarget"],
                ]].argmax(dim=-1),
                low_final_pred,
            )
        diagnostics = {
            "reroute_count": reroute_count.detach(),
            "reroute_triggered": (reroute_count > 0).detach(),
            "reroute_fallback": fallback_mask.detach(),
            "reroute_selected_route": selected_route.detach(),
            "reroute_initial_route": initial_route.detach(),
            "reroute_path": route_path.detach(),
            "reroute_final_route_logits": route_logits.detach(),
            "reroute_total_penalty": total_penalty.detach(),
            "reroute_accept_confidence": last_confidence.detach(),
            "reroute_reject_reason_code": reject_reason_code.detach(),
            "reroute_rule_terminal": rule_terminal.detach(),
            "reroute_system_forward": torch.ones((batch_size,), dtype=torch.bool, device=device),
            "reroute_lazy_expert_eval_count": torch.full((batch_size,), lazy_expert_eval_count, dtype=torch.long, device=device),
            "reroute_low_altitude_deep_used": low_deep_used_mask.detach(),
            "reroute_low_altitude_deep_accepted": low_deep_accept_mask.detach(),
            "reroute_low_altitude_deep_helped": low_deep_helped_mask.detach(),
            "reroute_low_altitude_deep_harmed": low_deep_harmed_mask.detach(),
            "reroute_low_altitude_pass1_pred": low_pass1_pred.detach(),
            "reroute_low_altitude_deep_pred": low_deep_pred.detach(),
            "reroute_low_altitude_final_pred": low_final_pred.detach(),
            "reroute_low_altitude_deep_accept_reason_code": low_deep_accept_reason_code.detach(),
            "low_reject_score": (
                low_altitude.reject_score.detach()
                if low_altitude.reject_score is not None
                else torch.zeros((batch_size,), dtype=torch.float32, device=device)
            ),
            "low_unknown_bonus": low_reject_reasons.get("unknown_trigger", torch.zeros((batch_size,), dtype=torch.bool, device=device)).float().detach(),
            "low_confusion_pair_bonus": low_reject_reasons.get("confusion_pair_trigger", torch.zeros((batch_size,), dtype=torch.bool, device=device)).float().detach() * 0.75,
            "low_evidence_conflict_bonus": low_reject_reasons.get("evidence_conflict_trigger", torch.zeros((batch_size,), dtype=torch.bool, device=device)).float().detach() * 0.55,
            "low_margin_bonus": low_reject_reasons.get("low_margin_trigger", torch.zeros((batch_size,), dtype=torch.bool, device=device)).float().detach() * 0.35,
            "low_agreement_confidence_credit": low_reject_reasons.get("agreement_confidence_credit", torch.zeros((batch_size,), dtype=torch.bool, device=device)).float().detach() * 0.50,
            "low_high_risk_pair": low_reject_reasons.get("high_risk_pair", torch.zeros((batch_size,), dtype=torch.bool, device=device)).detach(),
            "low_final_margin": low_reject_reasons.get("final_margin", torch.zeros((batch_size,), dtype=torch.float32, device=device)).detach(),
            "low_traj_pred": low_reject_reasons.get("traj_pred", torch.full((batch_size,), -1, dtype=torch.long, device=device)).detach(),
            "low_rd_pred": low_reject_reasons.get("rd_pred", torch.full((batch_size,), -1, dtype=torch.long, device=device)).detach(),
            "low_deep_pair_correction_max": low_deep_reject_reasons.get("deep_pair_correction_max", torch.zeros((batch_size,), dtype=torch.float32, device=device)).detach(),
            "low_deep_unknown_clutter_pred": low_deep_reject_reasons.get("deep_unknown_clutter_pred", torch.full((batch_size,), -1, dtype=torch.long, device=device)).detach(),
            "low_deep_unknown_clutter_confidence": low_deep_reject_reasons.get("deep_unknown_clutter_confidence", torch.zeros((batch_size,), dtype=torch.float32, device=device)).detach(),
        }
        return RadarSystemOutput(
            representation=representation,
            routing=routing,
            rd_features=rd_summary,
            low_altitude_detail=low_altitude,
            conventional_air_detail=conventional_air,
            high_altitude_detail=high_altitude,
            ground_detail=ground,
            low_altitude_logits=low_altitude.logits,
            conventional_air_logits=conventional_air.logits,
            high_altitude_logits=high_altitude.logits,
            ground_logits=ground.logits,
            reroute_logits=final_logits,
            reroute_diagnostics=diagnostics,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        rd_patch_features: torch.Tensor | None = None,
        rd_deep_patch_features: torch.Tensor | None = None,
        low_altitude_targets: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        rd_deep_patch_padding_mask: torch.Tensor | None = None,
        source_names: list[str] | None = None,
        enable_low_altitude_reject: bool = True,
        detach_ground_features: bool = False,
        inference_mode: str = "parallel",
        reroute_config: RerouteConfig | None = None,
        num_global_classes: int | None = None,
    ) -> RadarSystemOutput:
        if inference_mode == "reroute":
            if self.training:
                raise RuntimeError("inference_mode='reroute' is eval-only; keep training on parallel expert outputs")
            return self._forward_reroute(
                sequence_features=sequence_features,
                physical_features=physical_features,
                rd_features=rd_features,
                rd_patch_features=rd_patch_features,
                rd_deep_patch_features=rd_deep_patch_features,
                rd_is_multimodal=rd_is_multimodal,
                padding_mask=padding_mask,
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_patch_padding_mask,
                rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
                source_names=source_names,
                enable_low_altitude_reject=enable_low_altitude_reject,
                reroute_config=reroute_config,
                num_global_classes=num_global_classes,
            )
        output = self._forward_experts(
            sequence_features=sequence_features,
            physical_features=physical_features,
            rd_features=rd_features,
            rd_patch_features=rd_patch_features,
            rd_deep_patch_features=rd_deep_patch_features,
            low_altitude_targets=low_altitude_targets,
            rd_is_multimodal=rd_is_multimodal,
            padding_mask=padding_mask,
            rd_padding_mask=rd_padding_mask,
            rd_patch_padding_mask=rd_patch_padding_mask,
            rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
            source_names=source_names,
            enable_low_altitude_reject=enable_low_altitude_reject,
            detach_ground_features=detach_ground_features,
        )
        return output
