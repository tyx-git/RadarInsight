from copy import deepcopy
import json

import numpy as np
import pytest

from radar_ln3.decision.l3 import (
    L3CalibrationPolicy,
    resolve_l3,
)


@pytest.fixture
def l3_calibration():
    return {
        "policy": L3CalibrationPolicy().to_dict(),
        "artifact_sha256": "calibration-test",
        "status": "exploratory",
        "deployment_eligible": False,
        "knownness_source": "independent_l3_head",
        "l1_router_source": "classifier_endpoint",
        "temperature": {
            "l1": {"temperature": 1.0},
            "l2": {"temperature": 1.0},
        },
        "knownness_thresholds": {
            "l1": {
                "target": {"selected": {"threshold": 0.1}},
                "clutter": {"selected": {"threshold": 0.1}},
            },
            "l2": {
                "DroneTarget": {"selected": {"threshold": 0.1}},
                "BirdTarget": {"selected": {"threshold": 0.1}},
                "BalloonTarget": {"selected": {"threshold": 0.1}},
            },
        },
        "conformal": {
            "l1": {"candidate_probability_floor_by_class": {
                "KNOWN_TARGET_EVIDENCE": 0.1,
                "KNOWN_CLUTTER_EVIDENCE": 0.1,
            }},
            "l2": {"candidate_probability_floor_by_class": {
                "DroneTarget": 0.1,
                "BirdTarget": 0.1,
                "BalloonTarget": 0.1,
            }},
        },
        "l2_top2_margin_thresholds": {
            "DroneTarget": 1.0,
            "BirdTarget": 1.0,
            "BalloonTarget": 1.0,
        },
    }


def test_l3_decision_has_no_separate_quality_gate(l3_calibration):
    result = resolve_l3(
        l3_calibration,
        policy=L3CalibrationPolicy(**l3_calibration["policy"]),
        parent_sample_id="parent-low-quality",
        l1_logits=(8.0, -8.0),
        l2_logits=(8.0, -8.0, -8.0),
        l1_knownness_scores=(0.99, 0.01),
        l2_knownness_scores=(0.99, 0.01, 0.01),
        allow_exploratory=True,
    )

    assert result["action"] == "DroneTarget"
    assert result["reason_code"] == "KNOWN_DBB_CLASS_ACCEPTED"
    assert "l1_quality_score" not in result["details"]
    assert "l2_quality_score" not in result["details"]


@pytest.mark.parametrize("clutter_floor", [0.4, None])
@pytest.mark.parametrize(
    ("target_floor", "override", "expected_target_floor"),
    [
        pytest.param(0.8, None, 0.8, id="no-override"),
        pytest.param(0.8, {"floor": 0.6}, 0.6, id="lower"),
        pytest.param(0.8, {"floor": 0.8}, 0.8, id="equal"),
        pytest.param(0.8, {"floor": 0.8000000000005}, 0.8, id="higher-within-tolerance"),
        pytest.param(0.8, {"floor": 0.0}, 0.0, id="zero"),
        pytest.param(1.0, {"floor": 1.0}, 1.0, id="one"),
        pytest.param(None, None, None, id="unsupported-target"),
    ],
)
def test_effective_l1_conformal_floors(
    l3_calibration, target_floor, clutter_floor, override, expected_target_floor
):
    from radar_ln3.decision.l3 import _effective_l1_conformal_floors

    l3_calibration["conformal"]["l1"]["candidate_probability_floor_by_class"] = {
        "KNOWN_TARGET_EVIDENCE": target_floor,
        "KNOWN_CLUTTER_EVIDENCE": clutter_floor,
    }
    l3_calibration["l1_target_conformal_floor_override"] = override
    original = deepcopy(l3_calibration)

    assert _effective_l1_conformal_floors(l3_calibration) == {
        "KNOWN_TARGET_EVIDENCE": expected_target_floor,
        "KNOWN_CLUTTER_EVIDENCE": clutter_floor,
    }
    assert l3_calibration == original


@pytest.mark.parametrize("router_source", ["classifier_endpoint", "independent_l3_head"])
@pytest.mark.parametrize(
    ("override_floor", "expected_floor", "target_candidates", "target_action", "target_reason"),
    [
        pytest.param(None, 0.8, [], "UNKNOWN_REJECTED", "L1_CONFORMAL_SET_EMPTY", id="no-override"),
        pytest.param(0.6, 0.6, ["KNOWN_TARGET_EVIDENCE"], "DroneTarget", "KNOWN_DBB_CLASS_ACCEPTED", id="lower"),
        pytest.param(0.7, 0.7, ["KNOWN_TARGET_EVIDENCE"], "DroneTarget", "KNOWN_DBB_CLASS_ACCEPTED", id="boundary"),
        pytest.param(0.8, 0.8, [], "UNKNOWN_REJECTED", "L1_CONFORMAL_SET_EMPTY", id="equal"),
    ],
)
def test_l3_audit_report_uses_effective_floors(
    tmp_path, monkeypatch, l3_calibration, router_source,
    override_floor, expected_floor, target_candidates, target_action, target_reason,
):
    import radar_ln3.decision as decision
    import radar_ln3.training as training
    import radar_ln3.training.l3 as l3_training
    from scripts import report_l3_audit

    l3_calibration["l1_router_source"] = router_source
    l3_calibration["conformal"]["l1"]["candidate_probability_floor_by_class"] = {
        "KNOWN_TARGET_EVIDENCE": 0.8,
        "KNOWN_CLUTTER_EVIDENCE": 0.6,
    }
    if override_floor is not None:
        l3_calibration["l1_target_conformal_floor_override"] = {
            "floor": override_floor,
            "source": "validation_l1_knownness_fit",
            "fit_artifact_sha256": "f" * 64,
        }
    l3_calibration["endpoint_sha256"] = {"l1": "a" * 64, "l2": "b" * 64}
    l3_calibration["head_checkpoint_sha256"] = {"l1": "c" * 64, "l2": "d" * 64}
    original = deepcopy(l3_calibration)
    l1_scores = np.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45]])
    l2_scores = np.full((3, 3), 0.95)
    endpoint_probabilities = (
        l1_scores[:, ::-1] if router_source == "independent_l3_head" else l1_scores
    )
    evidence = l3_training.L3ValidationEvidence(
        parent_ids=("synthetic-target", "synthetic-clutter", "synthetic-unknown"),
        label_names=("DroneTarget", "ClutterTarget", "UnknownTarget"),
        l1_logits=np.log(endpoint_probabilities),
        l2_logits=np.log(np.array([[0.9, 0.06, 0.04]] * 3)),
        source_manifest_sha256="e" * 64,
        endpoint_sha256=l3_calibration["endpoint_sha256"],
        feature_cache_manifest_sha256="f" * 64,
        role="post_selection_audit",
    )
    monkeypatch.setattr(decision, "load_l3_calibration", lambda *args: l3_calibration)
    monkeypatch.setattr(training, "collect_l3_audit_evidence", lambda *args, **kwargs: evidence)
    monkeypatch.setattr(
        l3_training, "_collect_l3_head_scores",
        lambda *args, **kwargs: (
            evidence.parent_ids, l1_scores, l2_scores,
            l3_calibration["head_checkpoint_sha256"], 0,
        ),
    )
    args = report_l3_audit._parser().parse_args([
        "--project-root", str(tmp_path),
        "--calibration-path", "synthetic-calibration.json",
        "--audit-manifest-path", "synthetic-manifest.json",
        "--audit-feature-cache-manifest-path", "synthetic-cache.json",
        "--output-path", "synthetic-report.json",
    ])

    result = report_l3_audit.run(args)
    report = json.loads((tmp_path / "synthetic-report.json").read_text(encoding="utf-8"))

    assert result["record_count"] == report["record_count"] == 3
    assert result["path"] == str(tmp_path / "synthetic-report.json")
    assert report["conformal_floors"]["l1"] == {
        "KNOWN_TARGET_EVIDENCE": expected_floor,
        "KNOWN_CLUTTER_EVIDENCE": 0.6,
    }
    assert report["conformal_floors"]["l2"] == {
        "DroneTarget": 0.1, "BirdTarget": 0.1, "BalloonTarget": 0.1,
    }
    assert [(row["action"], row["reason_code"]) for row in report["records"]] == [
        (target_action, target_reason),
        ("Clutter", "L1_CLUTTER_ACCEPTED"),
        ("UNKNOWN_REJECTED", "L1_CONFORMAL_SET_EMPTY"),
    ]
    expected_candidates = [target_candidates, ["KNOWN_CLUTTER_EVIDENCE"], []]
    assert [row["l1_candidate_set"] for row in report["records"]] == expected_candidates
    assert [row["decision_details"]["l1_candidates"] for row in report["records"]] == expected_candidates
    assert l3_calibration == original
