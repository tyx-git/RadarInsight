"""Bounded V26j contract/model/mask smoke checks; never trains or fits L3."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_oof_dataset,
    load_known_gate_config,
)
from radar_ln3.known_gate_v26j_oof import (
    CONTRACT, CONTRACT_SCHEMA, FOLD_COUNT, L1_EPOCH, L1_LABELS, L1_TRAIN_EPOCHS,
    READINESS, READINESS_SCHEMA, build_v26j_l1_model, fold_record,
    l1_config_path, l1_train_feature_cache_path, model_runtime_binding,
    relative_path, sha256_file, validate_frozen_v26c_sources,
)
from build.known_gate_v26e_oof import _validate_frozen_l1_data, _validate_frozen_l2
from radar_ln3.known_gate_v26c_oof import fold_record as v26c_fold_record
from train.known_gate_v21 import _model_inputs as l1_model_inputs
from train.known_gate_v26j_oof import _runtime_l1_config


EXPECTED_PARENT_COUNT = 986
V26_SOURCE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"V26j smoke expected JSON object: {path}")
    return value


def _assert_rd_mask_and_causality(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    frames = batch["rd_frames"][:1].clone(); mask = batch["rd_frame_mask"][:1].clone()
    # Ensure a synthetically padded frame cannot alter output.  This validates
    # the GRU's conditional state update, even when a real sampled event has no
    # padding position available.
    if frames.shape[2] < 2: raise AssertionError("V26j smoke requires at least two RD frame slots")
    synthetic_mask = mask.clone(); synthetic_mask[:, :, -1] = False
    padded_a = frames.clone(); padded_b = frames.clone(); padded_b[:, :, -1].fill_(987.0)
    with torch.inference_mode():
        a = model.rd_encoder(padded_a, synthetic_mask); b = model.rd_encoder(padded_b, synthetic_mask)
    padded_difference = float((a - b).abs().max().item())
    if padded_difference != 0.0: raise AssertionError("V26j masked RD padding altered causal-GRU output")
    # A later active frame must be able to affect the result.  Force two valid
    # positions in one event using finite real-shaped tensors, then perturb only
    # the second one; the first-event state must remain unchanged.
    causal_mask = torch.zeros_like(mask); causal_mask[:, 0, 0] = True; causal_mask[:, 0, 1] = True
    first = frames.clone(); second = frames.clone(); second[:, 0, 1] = second[:, 0, 1] + 1.0
    with torch.inference_mode():
        first_out = model.rd_encoder(first, causal_mask); second_out = model.rd_encoder(second, causal_mask)
    later_difference = float((first_out[:, 0] - second_out[:, 0]).abs().max().item())
    if not later_difference > 0.0: raise AssertionError("V26j later active RD frame has no causal-GRU effect")
    zero_mask = torch.zeros_like(mask)
    with torch.inference_mode(): zero_output = model.rd_encoder(frames, zero_mask)
    if not torch.isfinite(zero_output).all() or float(zero_output.abs().max().item()) != 0.0: raise AssertionError("V26j zero-RD-frame event is not finite zero evidence")
    return {"padded_frame_max_difference": padded_difference, "later_active_frame_max_difference": later_difference, "zero_frame_output_finite_zero": True}


def _assert_offline_online_equivalence(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure event-internal RD GRU does not alter causal online semantics."""

    from train.known_gate_v21 import _model_inputs

    length = int(batch["lengths"][0].item())
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))["sequence_logits"][:1, :length]
        state = None
        online_logits = []
        for event_index in range(length):
            output = model.forward_online_point(
                tr_features=batch["tr_features"][:1, event_index],
                tr_missing_mask=batch["tr_missing_mask"][:1, event_index],
                tr_dt_seconds=batch["tr_dt_seconds"][:1, event_index],
                tr_dt_valid=batch["tr_dt_valid"][:1, event_index],
                tr_dt_reason=batch["tr_dt_reason"][:1, event_index],
                rd_frames=batch["rd_frames"][:1, event_index],
                rd_frame_mask=batch["rd_frame_mask"][:1, event_index],
                rd_quality=batch["rd_quality"][:1, event_index],
                liquid_state=state,
                known_gate_features=batch["known_gate_features"][:1, event_index],
                known_gate_feature_valid=batch["known_gate_feature_valid"][:1, event_index],
            )
            state = output["liquid_state"]
            online_logits.append(output["window_logits"])
        online = torch.stack(online_logits, dim=1)
    difference = float((offline - online).abs().max().item())
    if difference > 1.0e-5:
        raise AssertionError("V26j offline and online causal logits differ")
    return {"event_count": length, "max_logit_difference": difference, "tolerance": 1.0e-5}


def _assert_config_namespace_exclusivity(root: Path, config_path: Path) -> dict[str, bool]:
    """Prove V26 and V26j cannot silently exchange aggregation modes."""

    source_path = root / V26_SOURCE_CONFIG
    source = json.loads(source_path.read_text(encoding="utf-8"))
    treatment = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        load_known_gate_config(source_path).model.rd_frame_aggregation != "masked_mean"
        or load_known_gate_config(config_path).model.rd_frame_aggregation != "causal_gru"
    ):
        raise AssertionError("V26/V26j positive config namespace pairing differs")
    results: dict[str, bool] = {
        "v26_source_accepts_masked_mean": True,
        "v26j_accepts_causal_gru": True,
    }
    for name, payload, mode in (
        ("v26j_rejects_masked_mean", treatment, "masked_mean"),
        ("v26_rejects_causal_gru", source, "causal_gru"),
    ):
        altered = json.loads(json.dumps(payload))
        altered["model"]["rd_frame_aggregation"] = mode
        with tempfile.TemporaryDirectory(prefix="radar-ln3-v26j-config-") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(altered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                load_known_gate_config(path)
            except ValueError:
                results[name] = True
            else:
                raise AssertionError(f"V26j config namespace accepted an invalid {mode} mode")
    return results


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(); contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract, readiness = _read(contract_path), _read(readiness_path); checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed" or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks or not all(value is True for value in checks.values())
        or int(contract.get("known_train_parent_count", -1)) != EXPECTED_PARENT_COUNT
        or contract.get("single_variable", {}).get("field") != "model.rd_frame_aggregation"
        or contract.get("single_variable", {}).get("control") != "masked_mean"
        or contract.get("single_variable", {}).get("treatment") != "causal_gru"
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
    ): raise AssertionError("V26j prepared contract/readiness is invalid")
    validate_frozen_v26c_sources(root, contract)
    source_contract = _read(root / "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json")
    assignment = _read(root / "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json")
    assignment_rows = {str(row["parent_sample_id"]): dict(row) for row in assignment["assignments"]}
    held_seen, folds = [], []
    all_known = sorted(assignment_rows); all_dbb = sorted(parent for parent,row in assignment_rows.items() if str(row["label_name"]) in {"DroneTarget","BirdTarget","BalloonTarget"})
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id); config, config_path, _ = _runtime_l1_config(root, fold_id, dict(fold["l1"]))
        if config.model.rd_frame_aggregation != "causal_gru" or config.model.liquid_state_normalization != "recurrent": raise AssertionError("V26j config is not causal-GRU/recurrent")
        if fold["l1"].get("config_path") != relative_path(root, config_path) or fold["l1"].get("config_sha256") != sha256_file(config_path): raise AssertionError("V26j config hash binding differs")
        train_components = {assignment_rows[parent]["provenance_component_id"] for parent in fold["train_parent_ids"]}; held_components = {assignment_rows[parent]["provenance_component_id"] for parent in fold["held_out_parent_ids"]}
        if train_components & held_components: raise AssertionError("V26j provenance component crosses fold")
        _validate_frozen_l1_data(root, v26c_fold_record(source_contract, fold_id), all_known_parent_ids=all_known)
        _validate_frozen_l2(root, v26c_fold_record(source_contract, fold_id), assignment=assignment_rows, all_known_parent_ids=all_known, all_dbb_parent_ids=all_dbb, source_contract=source_contract, source_contract_path=root / "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json", source_readiness_path=root / "src-v3/processed/report/RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness.json")
        held_seen.extend(fold["held_out_parent_ids"]); folds.append({"fold_id":fold_id,"train_parent_count":len(fold["train_parent_ids"]),"held_out_parent_count":len(fold["held_out_parent_ids"]),"rd_frame_aggregation":"causal_gru"})
    if sorted(held_seen) != all_known or len(held_seen) != len(set(held_seen)): raise AssertionError("V26j held-out coverage is invalid")
    first = fold_record(contract, 0); config, _, _ = _runtime_l1_config(root, 0, dict(first["l1"]))
    dataset = build_known_gate_oof_dataset(root, config, parent_ids=first["train_parent_ids"], feature_cache_path=root / l1_train_feature_cache_path(0))
    batch = next(iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False, pin_memory=False)))
    model, initialization = build_v26j_l1_model(root, config); runtime = model_runtime_binding(model); model.eval()
    with torch.inference_mode(): output = model(**l1_model_inputs(batch))
    if output["sequence_logits"].shape[:2] != batch["tr_features"].shape[:2] or not torch.isfinite(output["sequence_logits"]).all(): raise AssertionError("V26j first forward invalid")
    return {"status":"passed","schema_version":"RADAR-LN3-V26j-rd-causal-gru-oof-contract-smoke-v1","contract_path":relative_path(root,contract_path),"contract_sha256":sha256_file(contract_path),"fold_count":FOLD_COUNT,"known_parent_count":len(all_known),"fixed_snapshots":{"l1_epoch":L1_EPOCH,"l2_epoch":15},"fold_checks":folds,"actual_rd_runtime":runtime,"initialization":initialization,"first_l1_train_batch_logits_shape":list(output["sequence_logits"].shape),"rd_mask_causality":_assert_rd_mask_and_causality(model,batch),"offline_online_causal_equivalence":_assert_offline_online_equivalence(model,batch),"config_namespace_exclusivity":_assert_config_namespace_exclusivity(root, root/l1_config_path(0)),"unknown_training_exposure":0,"unknown_normalization_exposure":0,"unknown_threshold_fit_exposure":0,"validation_access_count":0,"test_access_count":0,"training":"not_executed","deployment_eligible":False}


def main() -> None:
    parser=argparse.ArgumentParser(description="Smoke-test V26j causal-RD-GRU OOF contract."); parser.add_argument("--project-root",type=Path,required=True); args=parser.parse_args(); print(json.dumps(run(args.project_root.resolve()),ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
