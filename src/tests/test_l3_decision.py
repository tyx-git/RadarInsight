from copy import deepcopy
import json
from types import SimpleNamespace

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


def test_save_l3_calibration_reuse_returns_disk_digest(tmp_path, monkeypatch):
    import hashlib
    import radar_ln3.decision.l3 as decision

    value = {"b": 2, "a": 1}
    target = tmp_path / "calibration.json"
    target.write_bytes(b'{"a": 1, "b": 2}\n')
    monkeypatch.setattr(decision, "_validate_calibration", lambda payload: dict(payload))
    monkeypatch.setattr(decision, "load_l3_calibration", lambda *args: dict(value))

    saved, digest = decision.save_l3_calibration(
        tmp_path,
        target,
        value,
        overwrite=False,
        return_sha256=True,
    )

    assert saved == target
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()


def test_finalize_audit_registers_against_product_manifest(tmp_path, monkeypatch):
    import radar_ln3.training.l3 as training

    evidence = training.L3ValidationEvidence(
        parent_ids=("parent",),
        label_names=("UnknownTarget",),
        l1_logits=np.zeros((1, 2)),
        l2_logits=np.zeros((1, 3)),
        source_manifest_sha256="a" * 64,
        endpoint_sha256={},
        feature_cache_manifest_sha256="b" * 64,
        role="post_selection_audit",
    )
    product_manifest = object()
    captured = {}
    monkeypatch.setattr(training, "collect_l3_audit_evidence", lambda *args, **kwargs: evidence)
    monkeypatch.setattr(training, "load_manifest", lambda *args, **kwargs: product_manifest)
    monkeypatch.setattr(
        training,
        "audit_l3_calibration",
        lambda *args, **kwargs: {
            "status": "exploratory",
            "deployment_eligible": False,
            "artifact_sha256": "c" * 64,
            "audit": {"role": "post_selection_audit"},
        },
    )

    def capture_save(*args, **kwargs):
        captured["manifest"] = kwargs["manifest"]
        return tmp_path / "calibration.json", "d" * 64

    monkeypatch.setattr(training, "save_and_register_l3_calibration", capture_save)
    result = training.finalize_l3_audit(
        tmp_path,
        {"endpoint_sha256": {}, "knownness_source": "classifier_endpoint"},
        audit_manifest_path="audit.json",
        audit_feature_cache_manifest_path="audit-cache.json",
        thresholds_frozen_before_access=True,
        product_manifest_path="product.json",
    )

    assert captured["manifest"] is product_manifest
    assert result["file_sha256"] == "d" * 64


def test_finalize_audit_can_skip_product_registration(tmp_path, monkeypatch):
    import radar_ln3.training.l3 as training

    evidence = training.L3ValidationEvidence(
        parent_ids=("parent",),
        label_names=("UnknownTarget",),
        l1_logits=np.zeros((1, 2)),
        l2_logits=np.zeros((1, 3)),
        source_manifest_sha256="a" * 64,
        endpoint_sha256={},
        feature_cache_manifest_sha256="b" * 64,
        role="post_selection_audit",
    )
    monkeypatch.setattr(
        training, "collect_l3_audit_evidence", lambda *args, **kwargs: evidence
    )
    monkeypatch.setattr(training, "load_manifest", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        training,
        "audit_l3_calibration",
        lambda *args, **kwargs: {
            "status": "exploratory",
            "deployment_eligible": False,
            "artifact_sha256": "c" * 64,
            "audit": {"role": "post_selection_audit"},
        },
    )
    captured = {}

    def capture_save(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "calibration.json", "d" * 64

    monkeypatch.setattr(training, "save_and_register_l3_calibration", capture_save)
    result = training.finalize_l3_audit(
        tmp_path,
        {"endpoint_sha256": {}, "knownness_source": "classifier_endpoint"},
        audit_manifest_path="audit.json",
        audit_feature_cache_manifest_path="audit-cache.json",
        thresholds_frozen_before_access=True,
        product_manifest_path="product.json",
        register=False,
    )

    assert captured["register"] is False
    assert result["file_sha256"] == "d" * 64


def test_save_calibration_can_skip_registration(tmp_path, monkeypatch):
    import radar_ln3.training.l3 as training

    registrations = []
    monkeypatch.setattr(
        training,
        "save_l3_calibration",
        lambda *args, **kwargs: (tmp_path / "calibration.json", "a" * 64),
    )
    monkeypatch.setattr(
        training,
        "register_product_artifacts",
        lambda *args, **kwargs: registrations.append(kwargs),
    )

    path, digest = training.save_and_register_l3_calibration(
        tmp_path,
        {
            "status": "fitted",
            "deployment_eligible": True,
            "artifact_sha256": "c" * 64,
            "source_role": "exploratory_validation",
        },
        output_path="calibration.json",
        register=False,
        return_sha256=True,
    )

    assert registrations == []
    assert path == tmp_path / "calibration.json"
    assert digest == "a" * 64


def test_finalize_audit_promotes_before_single_registration(tmp_path, monkeypatch):
    import radar_ln3.training.l3 as training
    from radar_ln3.checkpoint import canonical_hash

    evidence = training.L3ValidationEvidence(
        parent_ids=("parent",),
        label_names=("UnknownTarget",),
        l1_logits=np.zeros((1, 2)),
        l2_logits=np.zeros((1, 3)),
        source_manifest_sha256="a" * 64,
        endpoint_sha256={},
        feature_cache_manifest_sha256="b" * 64,
        role="post_selection_audit",
    )
    monkeypatch.setattr(
        training, "collect_l3_audit_evidence", lambda *args, **kwargs: evidence
    )
    monkeypatch.setattr(training, "load_manifest", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        training,
        "audit_l3_calibration",
        lambda *args, **kwargs: {
            "status": "fitted",
            "deployment_eligible": True,
            "artifact_sha256": "c" * 64,
            "audit": {"role": "post_selection_audit", "passed": True},
        },
    )
    captured = {}
    registrations = []

    def capture_save(*args, **kwargs):
        registrations.append(kwargs)
        captured.update(kwargs.get("calibration", args[1] if len(args) > 1 else {}))
        return tmp_path / "calibration.json", "d" * 64

    monkeypatch.setattr(training, "save_and_register_l3_calibration", capture_save)
    result = training.finalize_l3_audit(
        tmp_path,
        {"endpoint_sha256": {}, "knownness_source": "classifier_endpoint"},
        audit_manifest_path="audit.json",
        audit_feature_cache_manifest_path="audit-cache.json",
        thresholds_frozen_before_access=True,
        product_manifest_path="product.json",
        formal_available=True,
        audit_evidence_path=None,
    )

    assert len(registrations) == 1
    assert captured["status"] == "formal_available"
    assert captured["deployment_eligible"] is True
    assert captured["research_only"] is False
    assert captured["formal_promotion"] == {
        "kind": "formal-audit-promotion",
        "source_status": "fitted",
        "audit_passed": True,
    }
    body = {key: value for key, value in captured.items() if key != "artifact_sha256"}
    assert captured["artifact_sha256"] == canonical_hash(body)
    assert result["status"] == "formal_available"
    assert result["artifact_sha256"] == captured["artifact_sha256"]


def test_l3_evidence_exposes_explicit_event_cache_switch(monkeypatch):
    import radar_ln3.training.l3 as training

    captured = {}

    def collect(*args, **kwargs):
        captured.update(kwargs)
        return "evidence"

    monkeypatch.setattr(training, "_collect_l3_evidence", collect)
    assert training.collect_l3_validation_evidence(
        "root", validate_event_cache=False, retain_representations=True
    ) == "evidence"
    assert captured["validate_event_cache"] is False
    assert captured["retain_representations"] is True


def test_head_scoring_reuses_provided_representations_and_manifest_hash(monkeypatch):
    import radar_ln3.training.l3 as training
    import radar_ln3.training.l3_heads as heads

    calls = {"manifest_hash": 0, "representation_load": 0}
    l1_representation = SimpleNamespace(
        parent_ids=("parent",),
        label_names=("DroneTarget",),
        test_payload_access_count=0,
    )
    l2_representation = SimpleNamespace(
        parent_ids=("parent",),
        label_names=("DroneTarget",),
        test_payload_access_count=0,
    )

    def manifest_hash(value):
        calls["manifest_hash"] += 1
        return "a" * 64

    def load_representation(*args, **kwargs):
        calls["representation_load"] += 1
        raise AssertionError("不应重复加载 representation")

    monkeypatch.setattr(training, "data_manifest_hash", manifest_hash)
    monkeypatch.setattr(heads, "load_frozen_endpoint_representation", load_representation)
    monkeypatch.setattr(
        heads,
        "load_l3_knownness_head",
        lambda *args, **kwargs: (
            SimpleNamespace(),
            SimpleNamespace(file_sha256="f" * 64),
        ),
    )
    monkeypatch.setattr(
        heads,
        "score_frozen_endpoint_representation",
        lambda *args, **kwargs: (np.zeros((1, 2)), "f" * 64),
    )

    parent_ids, l1_scores, l2_scores, head_hashes, access_count = training._collect_l3_head_scores(
        "root",
        manifest_path="manifest.json",
        feature_cache_manifest_path="cache.json",
        split_name="validation",
        training_view=True,
        l1_head_path="l1-head.pt",
        l2_head_path="l2-head.pt",
        device="cpu",
        batch_size=1,
        require_training_manifest_match=True,
        representation_root="representations",
        manifest=SimpleNamespace(),
        config=SimpleNamespace(path="model.json"),
        config_sha256="b" * 64,
        endpoint_sha256={"l1": "c" * 64, "l2": "d" * 64},
        dataset=SimpleNamespace(
            event_cache_validated=True,
            view=SimpleNamespace(view_hash="view-hash"),
        ),
        source_manifest_sha256="a" * 64,
        representations=(l1_representation, l2_representation),
    )

    assert calls == {"manifest_hash": 0, "representation_load": 0}
    assert parent_ids == ("parent",)
    assert l1_scores.shape == (1, 2) and l2_scores.shape == (1, 2)
    assert head_hashes == {"l1": "f" * 64, "l2": "f" * 64}
    assert access_count == 0


def test_fit_reuses_evidence_representations_and_head_checkpoints(tmp_path, monkeypatch):
    import radar_ln3.training.l3 as training

    l1_representation, l2_representation = object(), object()
    l1_head_checkpoint, l2_head_checkpoint = object(), object()
    evidence = training.L3ValidationEvidence(
        parent_ids=(),
        label_names=(),
        l1_logits=np.zeros((0, 2)),
        l2_logits=np.zeros((0, 3)),
        source_manifest_sha256="a" * 64,
        endpoint_sha256={},
        feature_cache_manifest_sha256="b" * 64,
        representations=(l1_representation, l2_representation),
    )
    captured = {}

    def collect(*args, **kwargs):
        captured["collect"] = kwargs
        return evidence

    def head_scores(*args, **kwargs):
        captured["head_scores"] = kwargs
        return ((), np.zeros((0, 2)), np.zeros((0, 3)), {"l1": "c" * 64, "l2": "d" * 64}, 0)

    monkeypatch.setattr(training, "collect_l3_validation_evidence", collect)
    monkeypatch.setattr(training, "_collect_l3_head_scores", head_scores)
    monkeypatch.setattr(
        training,
        "fit_l3_calibration",
        lambda **kwargs: {"status": "exploratory", "deployment_eligible": False},
    )

    training.fit_l3_validation_calibration(
        tmp_path,
        policy=object(),
        l1_head_path="l1-head.pt",
        l2_head_path="l2-head.pt",
        l1_head_checkpoint=l1_head_checkpoint,
        l2_head_checkpoint=l2_head_checkpoint,
    )

    assert captured["collect"]["retain_representations"] is True
    assert captured["head_scores"]["source_manifest_sha256"] == "a" * 64
    assert captured["head_scores"]["representations"] == (l1_representation, l2_representation)
    assert captured["head_scores"]["l1_head_checkpoint"] is l1_head_checkpoint
    assert captured["head_scores"]["l2_head_checkpoint"] is l2_head_checkpoint


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
    import radar_ln3.model as model
    import radar_ln3.training as training
    import radar_ln3.training.l3 as l3_training
    from scripts import report_l3_audit

    monkeypatch.setattr(
        model,
        "load_model_config",
        lambda *args, **kwargs: SimpleNamespace(
            path=tmp_path / "src/config/model.json",
            artifacts=SimpleNamespace(
                l1_output_path="artifacts/radar_ln3/l1/l1.pt",
                l2_output_path="artifacts/radar_ln3/l2/l2.pt",
            ),
        ),
    )
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
    ])

    result = report_l3_audit.run(args)
    report_path = tmp_path / "artifacts/radar_ln3/l3/radar_ln3_audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["record_count"] == report["record_count"] == 3
    assert result["path"] == str(report_path)
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


def test_l3_audit_report_reuses_saved_evidence(
    tmp_path, monkeypatch, l3_calibration
) -> None:
    import radar_ln3.decision as decision
    import radar_ln3.training as training
    import radar_ln3.training.l3 as l3_training
    from radar_ln3.checkpoint import canonical_hash
    from scripts import report_l3_audit

    parent_ids = ("synthetic-target", "synthetic-clutter", "synthetic-unknown")
    labels = ("DroneTarget", "ClutterTarget", "UnknownTarget")
    endpoint_sha256 = {"l1": "a" * 64, "l2": "b" * 64}
    head_sha256 = {"l1": "c" * 64, "l2": "d" * 64}
    l3_calibration["endpoint_sha256"] = endpoint_sha256
    l3_calibration["head_checkpoint_sha256"] = head_sha256
    l3_calibration["knownness_source"] = "independent_l3_head"
    l3_calibration["audit"] = {
        "role": "post_selection_audit",
        "source_manifest_sha256": "e" * 64,
        "parent_ids_sha256": canonical_hash(sorted(parent_ids)),
    }
    l3_calibration["audit_feature_cache_manifest_sha256"] = "f" * 64
    l1_logits = np.array([[1.0, -1.0], [-1.0, 1.0], [0.2, -0.2]])
    l2_logits = np.log(np.array([[0.9, 0.06, 0.04]] * 3))
    l1_scores = np.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45]])
    l2_scores = np.full((3, 3), 0.95)
    evidence = l3_training.L3ValidationEvidence(
        parent_ids=parent_ids,
        label_names=labels,
        l1_logits=l1_logits,
        l2_logits=l2_logits,
        source_manifest_sha256="e" * 64,
        endpoint_sha256=endpoint_sha256,
        feature_cache_manifest_sha256="f" * 64,
        role="post_selection_audit",
    )
    l3_training._write_audit_evidence(
        tmp_path,
        "artifacts/radar_ln3/l3/audit-evidence.jsonl",
        {
            "schema_version": l3_training.AUDIT_EVIDENCE_SCHEMA,
            "data_binding": {
                "source_manifest_sha256": "e" * 64,
                "endpoint_sha256": endpoint_sha256,
                "feature_cache_manifest_sha256": "f" * 64,
                "knownness_source": "independent_l3_head",
                "head_checkpoint_sha256": head_sha256,
                "parent_ids_sha256": canonical_hash(sorted(parent_ids)),
            },
            "role": "post_selection_audit",
            "split_name": "test",
            "record_count": len(parent_ids),
            "test_payload_access_count": 0,
            "calibration_artifact_sha256": l3_calibration["artifact_sha256"],
        },
        l3_training._audit_evidence_rows(
            evidence,
            (l1_scores, l2_scores),
        ),
    )
    monkeypatch.setattr(decision, "load_l3_calibration", lambda *args: l3_calibration)
    monkeypatch.setattr(
        training,
        "collect_l3_audit_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应重新收集 evidence")
        ),
    )
    monkeypatch.setattr(
        l3_training,
        "_collect_l3_head_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应重新计算 head scores")
        ),
    )
    args = report_l3_audit._parser().parse_args(
        ["--project-root", str(tmp_path)]
    )

    result = report_l3_audit.run(args)

    assert result["record_count"] == 3
    assert (tmp_path / "artifacts/radar_ln3/l3/radar_ln3_audit_report.json").is_file()


def test_audit_evidence_rejects_feature_cache_mismatch(
    tmp_path, monkeypatch, l3_calibration
) -> None:
    import radar_ln3.decision as decision
    import radar_ln3.training as training
    import radar_ln3.training.l3 as l3_training
    from radar_ln3.checkpoint import canonical_hash
    from scripts import report_l3_audit

    parent_ids = ("synthetic-target",)
    endpoint_sha256 = {"l1": "a" * 64, "l2": "b" * 64}
    head_sha256 = {"l1": "c" * 64, "l2": "d" * 64}
    l3_calibration["endpoint_sha256"] = endpoint_sha256
    l3_calibration["head_checkpoint_sha256"] = head_sha256
    l3_calibration["knownness_source"] = "independent_l3_head"
    l3_calibration["audit"] = {
        "role": "post_selection_audit",
        "source_manifest_sha256": "e" * 64,
        "parent_ids_sha256": canonical_hash(sorted(parent_ids)),
    }
    l3_calibration["audit_feature_cache_manifest_sha256"] = "f" * 64
    evidence = l3_training.L3ValidationEvidence(
        parent_ids=parent_ids,
        label_names=("DroneTarget",),
        l1_logits=np.zeros((1, 2)),
        l2_logits=np.zeros((1, 3)),
        source_manifest_sha256="e" * 64,
        endpoint_sha256=endpoint_sha256,
        feature_cache_manifest_sha256="f" * 64,
        role="post_selection_audit",
    )
    l3_training._write_audit_evidence(
        tmp_path,
        "artifacts/radar_ln3/l3/audit-evidence.jsonl",
        {
            "schema_version": l3_training.AUDIT_EVIDENCE_SCHEMA,
            "data_binding": {
                "source_manifest_sha256": "e" * 64,
                "endpoint_sha256": endpoint_sha256,
                "feature_cache_manifest_sha256": "0" * 64,
                "knownness_source": "independent_l3_head",
                "head_checkpoint_sha256": head_sha256,
                "parent_ids_sha256": canonical_hash(sorted(parent_ids)),
            },
            "role": "post_selection_audit",
            "split_name": "test",
            "record_count": 1,
            "test_payload_access_count": 0,
            "calibration_artifact_sha256": l3_calibration["artifact_sha256"],
        },
        l3_training._audit_evidence_rows(evidence, None),
    )
    monkeypatch.setattr(decision, "load_l3_calibration", lambda *args: l3_calibration)
    monkeypatch.setattr(
        training,
        "collect_l3_audit_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("绑定不一致时不应重新收集 evidence")
        ),
    )
    args = report_l3_audit._parser().parse_args(["--project-root", str(tmp_path)])

    with pytest.raises(ValueError, match="feature_cache_manifest_sha256"):
        report_l3_audit.run(args)
