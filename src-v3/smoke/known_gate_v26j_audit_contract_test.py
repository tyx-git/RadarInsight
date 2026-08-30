"""Bounded semantic/cache smoke test for V26j frozen mixed audit."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from radar_ln3.known_gate_v26j_audit import (
    ACTION_NAMES, CONTRACT, CONTRACT_SCHEMA, EXPECTED_PARENT_COUNTS, FOLD_COUNT,
    L1_TEST_CACHE, L1_VALIDATION_CACHE, L2_TEST_MICRO_CACHE, L2_TEST_PHYSICAL_CACHE,
    L2_VALIDATION_MICRO_CACHE, L2_VALIDATION_PHYSICAL_CACHE, READINESS,
    READINESS_SCHEMA, selected_event_sample_ids, validate_feature_cache,
    validate_prepared_contract,
)
from radar_ln3.open_set_architecture import ProductAction, resolve_semantic_action


def _cache_smoke() -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v26j-audit-") as directory:
        path=Path(directory)/"vector.npz"; ids=np.asarray(["s0","s1"],dtype="U2"); values=np.asarray([[0.25,0.0],[0.0,0.75]],dtype=np.float32); valid=np.asarray([[True,False],[False,True]],dtype=np.bool_); np.savez(path,sample_ids=ids,values=values,valid=valid)
        vector=validate_feature_cache(path,expected_sample_ids=["s0","s1"],width=2,valid_rank=2,description="vector")
        # A rank-1 invalid flag invalidates the whole event, so its entire
        # feature row must be zero.  Keep this fixture semantically valid;
        # the validator's rejection path is covered by the vector fixture.
        scalar_values=np.asarray([[0.25,0.0],[0.0,0.0]],dtype=np.float32)
        scalar_path=Path(directory)/"scalar.npz"; np.savez(scalar_path,sample_ids=ids,values=scalar_values,valid=np.asarray([True,False],dtype=np.bool_)); scalar=validate_feature_cache(scalar_path,expected_sample_ids=["s0","s1"],width=2,valid_rank=1,description="scalar")
    return {"vector_valid_event_count":int(vector["valid_event_count"]),"scalar_valid_event_count":int(scalar["valid_event_count"])}


def _sealed_mixed_cache_smoke(root: Path) -> dict[str, dict[str, int]]:
    """Check the real audit cache shape/order without running model forwards."""

    result: dict[str, dict[str, int]] = {}
    for split, l1, physical, micro in (
        ("val", L1_VALIDATION_CACHE, L2_VALIDATION_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE),
        ("test", L1_TEST_CACHE, L2_TEST_PHYSICAL_CACHE, L2_TEST_MICRO_CACHE),
    ):
        ids = selected_event_sample_ids(root, split=split)
        l1_metadata = validate_feature_cache(root / l1, expected_sample_ids=ids, width=1, valid_rank=2, description=f"{split} L1")
        physical_metadata = validate_feature_cache(root / physical, expected_sample_ids=ids, width=4, valid_rank=1, description=f"{split} physical")
        micro_metadata = validate_feature_cache(root / micro, expected_sample_ids=ids, width=8, valid_rank=1, description=f"{split} micro")
        result[split] = {
            "event_count": len(ids),
            "l1_valid_event_count": int(l1_metadata["valid_event_count"]),
            "l2_physical_valid_event_count": int(physical_metadata["valid_event_count"]),
            "l2_micro_valid_event_count": int(micro_metadata["valid_event_count"]),
        }
    return result


def _semantic_smoke() -> dict[str,str]:
    cases={"accepted_target":resolve_semantic_action(observation_eligible=True,known_target_accepted=True,known_clutter_accepted=False,dbb_subclass=ProductAction.DRONE,dbb_subclass_accepted=True).action,"clutter":resolve_semantic_action(observation_eligible=True,known_target_accepted=False,known_clutter_accepted=True).action,"subclass_weak":resolve_semantic_action(observation_eligible=True,known_target_accepted=True,known_clutter_accepted=False,dbb_subclass=ProductAction.DRONE,dbb_subclass_accepted=False).action,"neither":resolve_semantic_action(observation_eligible=True,known_target_accepted=False,known_clutter_accepted=False).action,"dual":resolve_semantic_action(observation_eligible=True,known_target_accepted=True,known_clutter_accepted=True).action,"insufficient":resolve_semantic_action(observation_eligible=False,known_target_accepted=True,known_clutter_accepted=False).action}
    expected={"accepted_target":ProductAction.DRONE,"clutter":ProductAction.CLUTTER,"subclass_weak":ProductAction.UNKNOWN_REJECTED,"neither":ProductAction.UNKNOWN_REJECTED,"dual":ProductAction.PENDING_ABSTAIN,"insufficient":ProductAction.PENDING_ABSTAIN}
    if cases!=expected: raise AssertionError("V26j semantic action boundary differs")
    return {name:value.value for name,value in cases.items()}


def run(project_root: Path) -> dict[str,object]:
    root=project_root.resolve(); contract,readiness=validate_prepared_contract(root)
    if contract.get("schema_version")!=CONTRACT_SCHEMA or readiness.get("schema_version")!=READINESS_SCHEMA or int(contract.get("fold_count",-1))!=FOLD_COUNT or int(contract.get("unknown_training_exposure",-1))!=0 or int(contract.get("unknown_threshold_fit_exposure",-1))!=0 or EXPECTED_PARENT_COUNTS["test"]["ClutterTarget"]!=8: raise AssertionError("V26j audit scope/support invalid")
    if set(ACTION_NAMES)!={"DroneTarget","BirdTarget","BalloonTarget","ClutterTarget","UNKNOWN_REJECTED","PENDING_ABSTAIN"}: raise AssertionError("V26j action space changed")
    return {"status":"passed","schema_version":"RADAR-LN3-V26j-frozen-mixed-open-set-audit-smoke-v1","contract_path":str(root/CONTRACT),"readiness_path":str(root/READINESS),"fold_count":FOLD_COUNT,"validation_parent_counts":EXPECTED_PARENT_COUNTS["val"],"test_parent_counts":EXPECTED_PARENT_COUNTS["test"],"cache_validator":_cache_smoke(),"sealed_mixed_cache_validator":_sealed_mixed_cache_smoke(root),"semantic_actions":_semantic_smoke(),"training":"not_executed","threshold_fitting":"not_executed","mixed_forward":"not_executed","deployment_eligible":False}


def main() -> None:
    parser=argparse.ArgumentParser(description="Smoke-test V26j frozen audit."); parser.add_argument("--project-root",type=Path,required=True); args=parser.parse_args(); print(json.dumps(run(args.project_root.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
