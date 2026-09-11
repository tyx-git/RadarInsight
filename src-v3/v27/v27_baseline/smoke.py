"""Short structural smoke tests for the complete V27 baseline."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper
from radar_ln3.losses import parent_final_fixed_parent_count_ce
from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file

from .data import V27TaskDataset, build_v27_dataloader, model_input_kwargs
from .base_binding import load_frozen_base_model, write_frozen_base_binding
from .l3 import (
    ACTION_SELECTION_METHOD,
    L1_CLASS_NAMES,
    V27Action,
    make_fitted_registry,
    make_unfitted_registry,
    resolve_l3_action,
    verify_l3_registry,
)
from .logging_utils import close_training_logger, configure_training_logger, log_event
from .protocol import (
    V27_L1_TASK_NAME,
    V27_L2_TASK_NAME,
    load_v27_config,
    v27_artifact_path,
    v27_training_log_path,
)
from .runtime import build_v27_model, load_v27_context, write_v27_sealed_json
from .training import run_v27_training_step


def _one_batch(root: Path, context: Any, task: str) -> tuple[Any, dict[str, Any]]:
    dataset = V27TaskDataset(
        root,
        context.config,
        task=task,
        split_name=context.config.data.train_split_name,
        parent_limit=3,
    )
    loader = build_v27_dataloader(dataset, batch_size=3, shuffle=False)
    return dataset, next(iter(loader))


def _assert_liquid_v2_continuation_equivalence(context: Any) -> None:
    """Prove offline, chunked and online paths use the same state equation.

    The test intentionally includes a zero-delta event, suffix padding and
    an explicit continuation state.  This catches a common deployment bug:
    an online implementation that behaves like the offline wrapper only for
    positive timestamps or unpadded full windows.
    """

    model = context.config.model
    cell = LiquidV2Cell(
        input_dim=4,
        hidden_dim=5,
        tau_min_seconds=model.tau_min_seconds,
        tau_max_seconds=model.tau_max_seconds,
        zero_dt_event_floor_seconds=model.zero_dt_event_floor_seconds,
        dt_max_seconds=model.dt_max_seconds,
        invalid_dt_update_mode=model.invalid_dt_update_mode,
        invalid_dt_observation_gain=model.invalid_dt_observation_gain,
    )
    wrapper = LiquidV2SequenceWrapper(cell)
    generator = torch.Generator(device="cpu").manual_seed(27)
    events = torch.randn((1, 4, 4), dtype=torch.float32, generator=generator)
    dt_seconds = torch.tensor([[0.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
    dt_reason = torch.tensor(
        [[
            DT_REASON_FIRST_EVENT,
            DT_REASON_ZERO_DELTA,
            DT_REASON_POSITIVE_DELTA,
            DT_REASON_PADDING,
        ]],
        dtype=torch.long,
    )
    dt_valid = torch.tensor([[False, False, True, False]], dtype=torch.bool)
    padding = torch.tensor([[False, False, False, True]], dtype=torch.bool)
    zero_ordinal = torch.tensor([[1, 2, 1, 0]], dtype=torch.long)

    offline = wrapper(
        events,
        dt_seconds,
        padding,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        zero_ordinal=zero_ordinal,
        return_trace=True,
    )
    first_chunk = wrapper(
        events[:, :2],
        dt_seconds[:, :2],
        padding[:, :2],
        dt_valid=dt_valid[:, :2],
        dt_reason=dt_reason[:, :2],
        zero_ordinal=zero_ordinal[:, :2],
        return_trace=True,
    )
    second_chunk = wrapper(
        events[:, 2:3],
        dt_seconds[:, 2:3],
        padding[:, 2:3],
        dt_valid=dt_valid[:, 2:3],
        dt_reason=dt_reason[:, 2:3],
        initial_state=first_chunk.final_state,
        zero_ordinal=zero_ordinal[:, 2:3],
        return_trace=True,
    )
    if not torch.allclose(
        offline.final_state,
        second_chunk.final_state,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("V27 Liquid V2 chunked continuation differs from offline")

    state: torch.Tensor | None = None
    for index in range(3):
        online = wrapper.step(
            events[:, index],
            dt_seconds[:, index],
            dt_valid=dt_valid[:, index],
            dt_reason=dt_reason[:, index],
            state=state,
            zero_ordinal=zero_ordinal[:, index],
        )
        state = online.next_state
    if state is None or not torch.allclose(
        offline.final_state,
        state,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("V27 Liquid V2 online continuation differs from offline")
    if offline.trace is None or not torch.equal(
        offline.sequence[:, 3], offline.sequence[:, 2]
    ):
        raise AssertionError("V27 Liquid V2 padding changed raw memory")


def _assert_training_log_path(context: Any) -> None:
    """Verify the configured log path is task-scoped and project-contained."""

    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        path = v27_training_log_path(
            context.project_root,
            context.config,
            task=task,
            seed=int(context.config.training.seeds[0]),
        )
        try:
            relative = path.relative_to(context.project_root)
        except ValueError as exc:
            raise AssertionError("V27 training log escapes project root") from exc
        if path.name != "training.log" or task not in relative.as_posix():
            raise AssertionError("V27 training log path is not task-scoped")
        head_root = v27_artifact_path(
            context.project_root,
            context.config.artifacts.head_training_log_template,
            int(context.config.training.seeds[0]),
            task=task,
        )
        for component in ("knownness", "quality"):
            head_path = head_root.parent / component / "training.log"
            if (
                head_path.name != "training.log"
                or component not in head_path.relative_to(context.project_root).as_posix()
            ):
                raise AssertionError("V27 open-set head log path is not component-scoped")

    escaped_artifacts = replace(
        context.config.artifacts,
        training_log_template="../outside/training.log",
    )
    escaped_config = replace(context.config, artifacts=escaped_artifacts)
    try:
        v27_training_log_path(
            context.project_root,
            escaped_config,
            task=V27_L1_TASK_NAME,
            seed=int(context.config.training.seeds[0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("V27 training log path accepted a project escape")


def _assert_logger_lifecycle() -> None:
    """Verify append, reconfiguration, and shutdown without touching run artifacts."""

    logger_name = "radar_ln3.v27.smoke.logger_lifecycle"
    with tempfile.TemporaryDirectory(prefix="v27-logger-smoke-") as directory:
        log_path = Path(directory) / "training.log"
        logger = configure_training_logger(
            log_path,
            logger_name=logger_name,
            console=False,
        )
        log_event(logger, "logger=smoke", stage="first")
        close_training_logger(logger)

        logger = configure_training_logger(
            log_path,
            logger_name=logger_name,
            console=False,
        )
        log_event(logger, "logger=smoke", stage="second")
        close_training_logger(logger)

        lines = log_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 2 or "stage=first" not in lines[0] or "stage=second" not in lines[1]:
            raise AssertionError("V27 logger append/reconfiguration contract differs")


def _assert_prefix_heads_are_causal(
    model: Any,
    inputs: dict[str, torch.Tensor],
) -> None:
    """Prove that a prefix head cannot see a modified future event.

    The test deliberately changes only an active event after the selected
    prefix.  Classifier and head parameters remain fixed; any earlier prefix
    change therefore means an accidental future-derived statistic entered the
    supposedly causal trajectory branch.
    """

    padding = inputs["tr_padding_mask"]
    lengths = (~padding).long().sum(dim=1)
    if int(lengths.max()) < 3:
        raise AssertionError("V27 causal-head smoke requires a three-event parent")
    row = int(torch.nonzero(lengths >= 3, as_tuple=False)[0].item())
    prefix = 1
    future = 2
    with torch.inference_mode():
        before = model(**inputs, return_trace=False)
    modified = {name: value.clone() for name, value in inputs.items()}
    # Alter only future inputs while preserving shape/mask/dtype contracts.
    modified["tr_features"][row, future].add_(0.125)
    modified["rd_quality"][row, future, 0] = 1.0 - modified["rd_quality"][row, future, 0]
    with torch.inference_mode():
        after = model(**modified, return_trace=False)
    for name in (
        "knownness_projection_sequence",
        "knownness_logit_sequence",
        "quality_logit_sequence",
        "task_facts_sequence",
        "trajectory_encoder_sequence",
    ):
        left, right = before[name][row, prefix], after[name][row, prefix]
        if not torch.equal(left, right):
            raise AssertionError(f"V27 prefix trajectory output leaks future evidence: {name}")


def _assert_frozen_base_loading(context: Any) -> None:
    """Verify exact classifier restoration with deliberately fresh heads."""

    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        model, _, binding = load_frozen_base_model(
            context,
            task=task,
            device=torch.device("cpu"),
        )
        record = binding["l1" if task == V27_L1_TASK_NAME else "l2"]
        from .base_binding import classifier_state_hash

        if classifier_state_hash(model.state_dict()) != record["classifier_state_sha256"]:
            raise AssertionError("V27 frozen classifier hash did not restore")
        if not any(parameter.requires_grad for parameter in model.trajectory_heads.parameters()):
            raise AssertionError("V27 fresh trajectory heads are unexpectedly frozen")


def _assert_frozen_binding_write_boundary(context: Any) -> None:
    """Ensure binding writes use an explicit, project-contained root."""

    from .base_binding import load_frozen_base_binding

    binding = load_frozen_base_binding(context.project_root)
    with tempfile.TemporaryDirectory(
        prefix="v27-binding-smoke-", dir=context.project_root
    ) as directory:
        relative_target = Path(directory).relative_to(context.project_root) / "binding.json"
        written = write_frozen_base_binding(
            relative_target,
            binding,
            project_root=context.project_root,
        )
        if written != (context.project_root / relative_target).resolve():
            raise AssertionError("V27 frozen binding write did not resolve under project root")
        if json.loads(written.read_text(encoding="utf-8")) != binding:
            raise AssertionError("V27 frozen binding replay changed its payload")
        try:
            write_frozen_base_binding(
                Path(directory).relative_to(context.project_root) / ".." / ".." / "outside.json",
                binding,
                project_root=context.project_root,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("V27 frozen binding write accepted a project escape")


def _assert_rd_aggregation_is_fail_closed(context: Any) -> None:
    """Reject a config label that the baseline encoder cannot implement."""

    payload = json.loads(context.config.config_path.read_text(encoding="utf-8"))
    payload["model"]["rd_frame_aggregation"] = "causal_gru"
    with tempfile.TemporaryDirectory(
        prefix="v27-config-smoke-", dir=context.project_root
    ) as directory:
        candidate = Path(directory) / "invalid-config.json"
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_v27_config(context.project_root, candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("V27 accepted an unsupported RD aggregation mode")


def _assert_conformal_candidate_contract() -> None:
    """Check dynamic L1/L2 candidate sets without a raw-score action path."""

    from .open_set import _conformal_candidates

    l1_conflict = _conformal_candidates(
        torch.tensor([0.60, 0.55]).numpy(),
        class_names=L1_CLASS_NAMES,
        probability_floor_by_class={
            "KNOWN_TARGET_EVIDENCE": 0.50,
            "KNOWN_CLUTTER_EVIDENCE": 0.50,
        },
    )
    l1_empty = _conformal_candidates(
        torch.tensor([0.49, 0.48]).numpy(),
        class_names=L1_CLASS_NAMES,
        probability_floor_by_class={
            "KNOWN_TARGET_EVIDENCE": 0.50,
            "KNOWN_CLUTTER_EVIDENCE": 0.50,
        },
    )
    l2_single = _conformal_candidates(
        torch.tensor([0.79, 0.12, 0.09]).numpy(),
        class_names=("DroneTarget", "BirdTarget", "BalloonTarget"),
        probability_floor_by_class={
            "DroneTarget": 0.50,
            "BirdTarget": 0.50,
            "BalloonTarget": 0.50,
        },
    )
    if (
        l1_conflict != L1_CLASS_NAMES
        or l1_empty != ()
        or l2_single != ("DroneTarget",)
    ):
        raise AssertionError("V27 dynamic conformal candidate contract differs")


def _assert_fitted_registry_contract(context: Any) -> dict[str, Any]:
    """Exercise registry score-contract and calibration-scope integrity checks."""

    digest = "a" * 64
    section = {
        "class_names": list(L1_CLASS_NAMES),
        "temperature": 1.0,
        "conformal_error_rate": 0.1,
        "conformal_method": "class_conditional_split_conformal_1_minus_calibrated_probability_higher",
        "conformal_quantile_by_class": {
            "KNOWN_TARGET_EVIDENCE": 0.4,
            "KNOWN_CLUTTER_EVIDENCE": 0.4,
        },
        "conformal_candidate_probability_floor_by_class": {
            "KNOWN_TARGET_EVIDENCE": 0.6,
            "KNOWN_CLUTTER_EVIDENCE": 0.6,
        },
        "conformal_support_by_class": {
            "KNOWN_TARGET_EVIDENCE": 1,
            "KNOWN_CLUTTER_EVIDENCE": 1,
        },
        "conformal_coverage_on_calibration_by_class": {
            "KNOWN_TARGET_EVIDENCE": 1.0,
            "KNOWN_CLUTTER_EVIDENCE": 1.0,
        },
        "conformal_support": 2,
        "conformal_coverage_on_calibration": 1.0,
        "knownness_threshold": 0.5,
        "knownness_score_direction": "higher_is_known",
        "knownness_threshold_space": "sigmoid_probability",
        "knownness_positive_support": 1,
        "knownness_negative_support": 1,
        "quality_threshold": 0.5,
        "quality_score_direction": "higher_is_eligible",
        "quality_positive_support": 1,
        "quality_negative_support": 1,
        "quality_gate": {
            "active": False,
            "status": "inactive_insufficient_final_negative_support",
            "required_negative_support": context.config.open_set.quality_min_final_negative_support,
            "observed_negative_support": 1,
            "learned_score_role": "advisory_only",
        },
    }
    l2_section = {
        **section,
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "conformal_quantile_by_class": {
            "DroneTarget": 0.4,
            "BirdTarget": 0.4,
            "BalloonTarget": 0.4,
        },
        "conformal_candidate_probability_floor_by_class": {
            "DroneTarget": 0.6,
            "BirdTarget": 0.6,
            "BalloonTarget": 0.6,
        },
        "conformal_support_by_class": {
            "DroneTarget": 1,
            "BirdTarget": 1,
            "BalloonTarget": 1,
        },
        "conformal_coverage_on_calibration_by_class": {
            "DroneTarget": 1.0,
            "BirdTarget": 1.0,
            "BalloonTarget": 1.0,
        },
        "conformal_support": 3,
    }
    score_contract = {
        "schema_version": "RADAR-LN3-V27-dynamic-calibration-v3",
        "fixed_softmax_threshold": False,
        "single_distance_gate": False,
        "quality_label_policy": "causal_deterministic_facts_v1",
        "classifier_probability_set": "dynamic_temperature_class_conditional_conformal_probability_set",
    }
    scores = {
        "schema_version": "RADAR-LN3-V27-dynamic-calibration-v3",
        "status": "fitted",
        "experiment_id": "V27",
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": digest,
        "calibration_pool_registry_sha256": digest,
        "representation_cache_manifest_sha256": digest,
        "score_contract": score_contract,
        "score_contract_sha256": canonical_sha256(score_contract),
        "fitting_method": "smoke",
        "action_selection": {
            "method": ACTION_SELECTION_METHOD,
            "known_target_action_floor": 0.95,
            "clutter_action_floor": 8.0 / 9.0,
            "knownness_candidate_policy": "smoke",
            "l1_conformal_error_rate_grid": [0.1],
            "l2_conformal_error_rate_grid": [0.1],
            "selection_ranking": ["smoke"],
            "candidates_considered": 1,
            "feasible_candidate_count": 1,
            "low_quality_pending_counted_as_unknown_rejection": False,
            "selected": {
                "l1_conformal_error_rate": 0.1,
                "l2_conformal_error_rate": 0.1,
                "l1_knownness_threshold": 0.5,
                "l2_knownness_threshold": 0.5,
                "action_metrics": {
                    "all_known_target_parent_count": 1,
                    "eligible_known_target_support": 1,
                    "known_target_deterministic_quality_pending_count": 0,
                    "eligible_known_target_dbb_route_recall": 1.0,
                    "all_clutter_parent_count": 1,
                    "eligible_clutter_support": 1,
                    "clutter_deterministic_quality_pending_count": 0,
                    "eligible_clutter_recall": 1.0,
                    "unknown_parent_count": 1,
                },
            },
        },
        "l1": section,
        "l2": l2_section,
        "quality": {
            "threshold": 0.5,
            "threshold_space": "sigmoid_probability",
            "score_direction": "higher_is_eligible",
            "positive_support": 1,
            "negative_support": 1,
            "task_specific_thresholds": {"known_gate": 0.5, "dbb_specialist": 0.5},
            "task_specific_gate_active": {
                "known_gate": False,
                "dbb_specialist": False,
            },
        },
        "head_checkpoints": {
            task: {"sha256": digest, "trajectory_head_state_sha256": digest}
            for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME)
        },
        "calibration_score_cache_sha256": {
            task: digest for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME)
        },
        "calibration_pool_parent_count": 2,
        "calibration_scope": {
            "pool_role": "calibration",
            "head_optimization_independent": True,
            "classifier_checkpoint_selection_independent": False,
            "coverage_guarantee": (
                "not_claimed_after_joint_action_selection_or_classifier_"
                "checkpoint_selection"
            ),
        },
        "unknown_used_for_classifier_training": False,
        "test_payload_access_count": 0,
    }
    fitted = make_fitted_registry(
        system_manifest_sha256=context.manifest_sha256,
        fitted_scores=scores,
    )
    verify_l3_registry(
        fitted, expected_system_manifest_sha256=context.manifest_sha256
    )
    tampered = dict(fitted)
    tampered_scores = dict(scores)
    tampered_scores["score_contract"] = {**score_contract, "single_distance_gate": True}
    tampered_scores["score_contract_sha256"] = canonical_sha256(
        tampered_scores["score_contract"]
    )
    tampered["fitted_scores"] = tampered_scores
    tampered["registry_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "registry_sha256"}
    )
    try:
        verify_l3_registry(
            tampered, expected_system_manifest_sha256=context.manifest_sha256
        )
    except ValueError:
        pass
    else:
        raise AssertionError("V27 score-contract tampering was accepted")
    return fitted


def _assert_fitted_registry_write_is_immutable(
    context: Any, fitted: Mapping[str, Any]
) -> None:
    """A completed calibration result must never be silently overwritten."""

    from .runtime import write_v27_sealed_json_if_absent_or_identical

    scores = fitted.get("fitted_scores")
    if not isinstance(scores, Mapping):
        raise AssertionError("V27 smoke fitted registry lacks scores")
    replacement = make_fitted_registry(
        system_manifest_sha256=context.manifest_sha256,
        fitted_scores={
            **dict(scores),
            "fitting_method": "smoke-different-registry",
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "registry.json"
        write_v27_sealed_json_if_absent_or_identical(path, dict(fitted))
        write_v27_sealed_json_if_absent_or_identical(path, dict(fitted))
        try:
            write_v27_sealed_json_if_absent_or_identical(path, replacement)
        except FileExistsError:
            return
    raise AssertionError("V27 fitted registry overwrite was accepted")


def _assert_unfitted_registry_upgrade_is_allowed(
    context: Any, fitted: Mapping[str, Any]
) -> None:
    """The only mutable registry transition is an unfitted placeholder upgrade."""

    from .runtime import read_v27_sealed_json

    unfitted = make_unfitted_registry(
        system_manifest_sha256=context.manifest_sha256
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "registry.json"
        write_v27_sealed_json(path, unfitted)
        write_v27_sealed_json(path, dict(fitted))
        loaded = read_v27_sealed_json(path)
        if loaded.get("status") != "fitted":
            raise AssertionError("V27 unfitted registry upgrade did not persist")


def _assert_joint_action_calibration_contract(context: Any) -> None:
    """Check that action selection protects known safety before OOD ranking."""

    from .open_set import (
        _action_safety_metrics,
        _joint_action_calibration,
        _replay_l3_records,
    )

    l1 = {
        "known_target": np.asarray([1, 1, 1, 1, 0], dtype=np.int64),
        "knownness_probability": np.asarray([0.9, 0.85, 0.83, 0.8, 0.2]),
        "class_labels": np.asarray([0, 0, 0, 1], dtype=np.int64),
        "parent_ids": ["d1", "b1", "a1", "c1", "u1"],
        "labels": [
            "DroneTarget",
            "BirdTarget",
            "BalloonTarget",
            "ClutterTarget",
            "UnknownTarget",
        ],
        "lengths": np.asarray([3, 3, 3, 3, 3]),
        "classifier_logits": np.asarray(
            [
                [4.0, 0.0],
                [4.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]
        ),
        "deterministic_quality": np.asarray([True, True, True, True, True]),
        "quality_probability": np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
    }
    l2 = {
        "known_target": np.asarray([1, 1, 1, 0, 0], dtype=np.int64),
        "knownness_probability": np.asarray([0.9, 0.85, 0.83, 0.2, 0.2]),
        "class_labels": np.asarray([0, 1, 2], dtype=np.int64),
        "parent_ids": ["d1", "b1", "a1", "c1", "u1"],
        "labels": [
            "DroneTarget",
            "BirdTarget",
            "BalloonTarget",
            "ClutterTarget",
            "UnknownTarget",
        ],
        "lengths": np.asarray([3, 3, 3, 3, 3]),
        "classifier_logits": np.asarray(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        "deterministic_quality": np.asarray([True, True, True, True, True]),
        "quality_probability": np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
    }
    base = {
        "temperature": 1.0,
        "knownness_threshold": 0.8,
        "quality_threshold": 0.5,
        "quality_gate": {"active": False},
    }
    l1_section = {
        **base,
        "class_names": list(L1_CLASS_NAMES),
    }
    l2_section = {
        **base,
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
    }
    l1_selected, l2_selected, selection = _joint_action_calibration(
        context,
        l1=l1,
        l2=l2,
        l1_section=l1_section,
        l2_section=l2_section,
        l1_class_probabilities=torch.softmax(
            torch.tensor(l1["classifier_logits"]), dim=-1
        ).numpy()[:4],
        l2_class_probabilities=torch.softmax(
            torch.tensor(l2["classifier_logits"]), dim=-1
        ).numpy()[:3],
    )
    if (
        not selection["feasible_candidate_count"]
        or l1_selected["knownness_threshold"] > base["knownness_threshold"]
        or l2_selected["knownness_threshold"] > base["knownness_threshold"]
        or selection["selected"]["action_metrics"][
            "eligible_known_target_dbb_route_recall"
        ]
        < context.config.open_set.action_known_target_recall_floor
        or selection["selected"]["action_metrics"]["eligible_clutter_recall"]
        < context.config.open_set.action_clutter_recall_floor
    ):
        raise AssertionError("V27 joint L3 action calibration contract differs")
    if not _action_safety_metrics(
        _replay_l3_records(
            l1=l1,
            l2=l2,
            l1_section=l1_selected,
            l2_section=l2_selected,
            calibration_status="fitted",
            streaming=False,
        )
    ):
        raise AssertionError("V27 joint L3 action replay unexpectedly has no metrics")


def run_smoke(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    context = load_v27_context(root)
    _assert_liquid_v2_continuation_equivalence(context)
    _assert_training_log_path(context)
    _assert_logger_lifecycle()
    _assert_frozen_base_loading(context)
    _assert_frozen_binding_write_boundary(context)
    _assert_rd_aggregation_is_fail_closed(context)
    _assert_conformal_candidate_contract()
    fitted_registry = _assert_fitted_registry_contract(context)
    _assert_fitted_registry_write_is_immutable(context, fitted_registry)
    _assert_unfitted_registry_upgrade_is_allowed(context, fitted_registry)
    _assert_joint_action_calibration_contract(context)
    models: dict[str, Any] = {}
    logits: dict[str, torch.Tensor] = {}
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        dataset, batch = _one_batch(root, context, task)
        model = build_v27_model(context, task=task, device=torch.device("cpu"))
        inputs = model_input_kwargs(batch, torch.device("cpu"))
        output = model(**inputs, return_trace=True)
        _assert_prefix_heads_are_causal(model, inputs)
        weights = torch.tensor(context.manifest["tasks"][task]["class_weights"])
        loss = parent_final_fixed_parent_count_ce(
            output["sequence_logits"],
            batch["labels"],
            batch["tr_padding_mask"],
            class_weights=weights,
        )["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("V27 smoke loss is non-finite")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("V27 smoke gradient is non-finite")
        models[task], logits[task] = model, output["window_logits"].detach().clone()
        if (
            output["knownness_projection"].shape[-1]
            != context.config.model.knownness_projection_dim
            or output["quality_logit"].shape != (len(batch["labels"]),)
            or output["knownness_projection_sequence"].shape[:2]
            != output["sequence_logits"].shape[:2]
            or output["quality_logit_sequence"].shape
            != output["sequence_logits"].shape[:2]
            or output["trace"]["raw_state"].shape
            != output["liquid_sequence"].shape
        ):
            raise ValueError("V27 smoke head/trace contract differs")
    # Quality/knownness outputs are downstream-only. Altering their head
    # parameters must not alter recurrent/classification logits.
    l2_model = models[V27_L2_TASK_NAME]
    _, l2_batch = _one_batch(root, context, V27_L2_TASK_NAME)
    l2_inputs = model_input_kwargs(l2_batch, torch.device("cpu"))
    with torch.inference_mode():
        before = l2_model(**l2_inputs, return_trace=False)["window_logits"].clone()
    with torch.no_grad():
        for parameter in l2_model.trajectory_heads.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    with torch.inference_mode():
        after = l2_model(**l2_inputs, return_trace=False)["window_logits"]
    if not torch.equal(before, after):
        raise AssertionError("V27 quality/knownness head feeds back into classification")
    # Direct object and parameter-key overlap checks catch accidental sharing.
    if models[V27_L1_TASK_NAME].liquid.cell is models[V27_L2_TASK_NAME].liquid.cell:
        raise AssertionError("V27 L1/L2 Liquid V2 cells are unexpectedly shared")
    l1_ids = {id(value) for value in models[V27_L1_TASK_NAME].parameters()}
    l2_ids = {id(value) for value in models[V27_L2_TASK_NAME].parameters()}
    if l1_ids & l2_ids:
        raise AssertionError("V27 L1/L2 parameters are unexpectedly shared")
    training_steps = {
        task: run_v27_training_step(context, task=task)
        for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME)
    }
    registry = make_unfitted_registry(system_manifest_sha256=context.manifest_sha256)
    verify_l3_registry(registry, expected_system_manifest_sha256=context.manifest_sha256)
    sealed_registry_path = (
        root / "src-v3" / "v27" / "calibration" / "V27-l3-registry.json"
    )
    if sealed_registry_path.is_file():
        verify_l3_registry(
            json.loads(sealed_registry_path.read_text(encoding="utf-8")),
            expected_system_manifest_sha256=context.manifest_sha256,
        )
    decisions = {
        "low_quality": resolve_l3_action(
            calibration_status="fitted",
            deterministic_quality_eligible=False,
            learned_quality_eligible=True,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=True,
            l2_knownness_accepted=True,
            dbb_conformal_candidates=("DroneTarget",),
            streaming=False,
        ),
        "unfitted": resolve_l3_action(
            calibration_status="unfitted",
            deterministic_quality_eligible=True,
            learned_quality_eligible=None,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=None,
            l2_knownness_accepted=None,
            dbb_conformal_candidates=(),
            streaming=False,
        ),
        "conflict": resolve_l3_action(
            calibration_status="fitted",
            deterministic_quality_eligible=True,
            learned_quality_eligible=True,
            l1_target_accepted=True,
            l1_clutter_accepted=True,
            l1_knownness_accepted=True,
            l2_knownness_accepted=True,
            dbb_conformal_candidates=("DroneTarget",),
            streaming=False,
        ),
        "clutter": resolve_l3_action(
            calibration_status="fitted",
            deterministic_quality_eligible=True,
            learned_quality_eligible=True,
            l1_target_accepted=False,
            l1_clutter_accepted=True,
            l1_knownness_accepted=True,
            l2_knownness_accepted=None,
            dbb_conformal_candidates=(),
            streaming=False,
        ),
        "clutter_l2_immature": resolve_l3_action(
            calibration_status="fitted",
            l1_deterministic_quality_eligible=True,
            l1_learned_quality_eligible=True,
            l2_deterministic_quality_eligible=False,
            l2_learned_quality_eligible=False,
            l1_target_accepted=False,
            l1_clutter_accepted=True,
            l1_knownness_accepted=True,
            l2_knownness_accepted=None,
            dbb_conformal_candidates=(),
            streaming=False,
        ),
        "target_l2_immature": resolve_l3_action(
            calibration_status="fitted",
            l1_deterministic_quality_eligible=True,
            l1_learned_quality_eligible=True,
            l2_deterministic_quality_eligible=False,
            l2_learned_quality_eligible=False,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=True,
            l2_knownness_accepted=None,
            dbb_conformal_candidates=(),
            streaming=False,
        ),
        "advisory_quality": resolve_l3_action(
            calibration_status="fitted",
            l1_deterministic_quality_eligible=True,
            l1_learned_quality_eligible=None,
            l1_learned_quality_gate_active=False,
            l2_deterministic_quality_eligible=True,
            l2_learned_quality_eligible=None,
            l2_learned_quality_gate_active=False,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=True,
            l2_knownness_accepted=True,
            dbb_conformal_candidates=("DroneTarget",),
            streaming=False,
        ),
        "dbb": resolve_l3_action(
            calibration_status="fitted",
            deterministic_quality_eligible=True,
            learned_quality_eligible=True,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=True,
            l2_knownness_accepted=True,
            dbb_conformal_candidates=("BirdTarget",),
            streaming=False,
        ),
        "ambiguous": resolve_l3_action(
            calibration_status="fitted",
            deterministic_quality_eligible=True,
            learned_quality_eligible=True,
            l1_target_accepted=True,
            l1_clutter_accepted=False,
            l1_knownness_accepted=True,
            l2_knownness_accepted=True,
            dbb_conformal_candidates=("DroneTarget", "BirdTarget"),
            streaming=False,
        ),
    }
    if (
        decisions["low_quality"].action != V27Action.PENDING_ABSTAIN
        or decisions["unfitted"].reason_code != "CALIBRATION_NOT_FITTED"
        or decisions["conflict"].reason_code != "L1_ROUTE_CONFLICT"
        or decisions["clutter"].action != V27Action.CLUTTER
        or decisions["clutter_l2_immature"].action != V27Action.CLUTTER
        or decisions["target_l2_immature"].action != V27Action.PENDING_ABSTAIN
        or decisions["advisory_quality"].action != V27Action.DRONE
        or decisions["dbb"].action != V27Action.BIRD
        or decisions["ambiguous"].reason_code != "KNOWN_CLASS_AMBIGUOUS"
    ):
        raise AssertionError("V27 L3 ordering/reason-code contract differs")
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V27-complete-contract-smoke-v3",
        "experiment_id": "V27",
        "tasks": [V27_L1_TASK_NAME, V27_L2_TASK_NAME],
        "raw_memory_post_update_layer_norm": False,
        "active_time_mode": context.config.model.invalid_dt_update_mode,
        "liquid_v2_offline_chunked_online_equivalent": True,
        "l1_l2_parameters_independent": True,
        "knownness_quality_heads": "separate_output_only_no_classifier_feedback",
        "training_log_path_contract": "project_root_resolved_task_scoped",
        "training_logger_lifecycle": "append_reconfigure_close_verified",
        "classification_training_step": training_steps,
        "l3_unfitted_fails_closed": True,
        "l3_reason_code_paths": {
            name: value.reason_code for name, value in decisions.items()
        },
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "training": "not_executed",
        "deployment_eligible": False,
    }
    report = resolve_project_path(root, context.config.artifacts.contract_smoke_report_path)
    write_v27_sealed_json(report, result)
    return {**result, "report_path": str(report), "report_sha256": sha256_file(report)}


__all__ = ["run_smoke"]
