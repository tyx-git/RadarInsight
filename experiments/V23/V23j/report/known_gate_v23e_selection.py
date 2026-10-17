"""Read-only V23e scan of the frozen V23a 20-epoch checkpoint trajectory."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from report.known_gate_common import _ci


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 23051












def _selection_bootstrap(records: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Parent bootstrap for the selected state, retaining all its prefixes."""

    parents = {str(row["parent_sample_id"]): dict(row) for row in records["parent"]}
    prefixes: dict[str, list[dict[str, Any]]] = {}
    for row in records["eligible_prefix"]:
        prefixes.setdefault(str(row["parent_sample_id"]), []).append(dict(row))
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    strata = {
        label: np.asarray(
            [
                parent_id
                for parent_id, row in parents.items()
                if str(row["label_name"]) == label
            ],
            dtype=object,
        )
        for label in labels
    }
    expected = {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9}
    if {label: len(values) for label, values in strata.items()} != expected:
        raise ValueError("V23e selected-state bootstrap support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    parent_clutter_precision: list[float] = []
    parent_clutter_recall: list[float] = []
    parent_clutter_f1: list[float] = []
    parent_macro_f1: list[float] = []
    prefix_clutter_recall: list[float] = []
    prefix_clutter_f1: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            str(parent_id)
            for values in strata.values()
            for parent_id in rng.choice(values, size=len(values), replace=True)
        ]
        sampled_parent = [parents[parent_id] for parent_id in sampled_ids]
        sampled_prefix = [
            {**row, "parent_sample_id": f"{parent_id}#bootstrap{index}"}
            for index, parent_id in enumerate(sampled_ids)
            for row in prefixes.get(parent_id, ())
        ]
        # Distinct synthetic IDs retain the intended multiplicity when a
        # bootstrap draw contains the same original parent more than once.
        from radar_ln3.known_gate import (
            summarize_known_gate_records,
            summarize_parent_balanced_eligible_prefix_records,
        )

        parent_metrics = summarize_known_gate_records(
            sampled_parent, scope="v23e_selected_parent_bootstrap"
        )
        prefix_metrics = summarize_parent_balanced_eligible_prefix_records(
            sampled_prefix, scope="v23e_selected_prefix_bootstrap"
        )
        parent_rows = {row["class"]: row for row in parent_metrics["per_class"]}
        prefix_rows = {row["class"]: row for row in prefix_metrics["per_class"]}
        parent_clutter_precision.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["precision"]))
        parent_clutter_recall.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]))
        parent_clutter_f1.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]))
        parent_macro_f1.append(float(parent_metrics["macro_f1"]))
        prefix_clutter_recall.append(float(prefix_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]))
        prefix_clutter_f1.append(float(prefix_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]))
    return {
        "method": "stratified parent bootstrap; each sampled parent retains all causal prefixes with internal 1/n weight",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": expected,
        "parent_final": {
            "macro_f1": _ci(parent_macro_f1),
            "clutter_precision": _ci(parent_clutter_precision),
            "clutter_recall": _ci(parent_clutter_recall),
            "clutter_f1": _ci(parent_clutter_f1),
        },
        "parent_balanced_eligible_prefix": {
            "clutter_recall": _ci(prefix_clutter_recall),
            "clutter_f1": _ci(prefix_clutter_f1),
        },
    }







