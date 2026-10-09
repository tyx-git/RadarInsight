"""Four-class closed-set cascade: L1 routes, L2 refines, over the L1 validation set.

This produces the 4-class confusion matrix (Drone / Bird / Balloon / Clutter)
that the two per-task scopes cannot express on their own, because L1's own
scope is 2-class and L2's is 3-class over a different population.

SCOPE LIMITS -- read before quoting any number from this script:
  * This is CLOSED SET.  There is no Unknown input and no abstention.
  * The open-set version (Unknown + UNKNOWN_REJECTED + PENDING_ABSTAIN) is NOT
    computable on the current artifacts: the L3 registry fails identity
    verification against the manifest, and the calibration it would supply is
    recorded as unfitted.  Every end-to-end call would return
    REGISTRY_INVALID_OR_UNAVAILABLE or PENDING_ABSTAIN.

Run:
    ./.venv/Scripts/python.exe src-v3/deploy/four_class_cascade.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from v28.benchmark import _load_registry_tolerant  # noqa: E402
from v28.checkpoint import load_checkpoint  # noqa: E402
from v28.data import V28TaskDataset, build_dataloader, model_input_kwargs  # noqa: E402
from v28.metrics import confusion_metrics  # noqa: E402
from v28.model import V28TaskModel  # noqa: E402
from v28.protocol import V28_L1_TASK, V28_L2_TASK, load_context  # noqa: E402

L1_TARGET = "KNOWN_TARGET_EVIDENCE"
L1_CLUTTER = "KNOWN_CLUTTER_EVIDENCE"
FOUR = ("DroneTarget", "BirdTarget", "BalloonTarget", "Clutter")

#: The index stores the clutter truth label as ``ClutterTarget``; L3's *action*
#: for it is ``Clutter``.  The two names are not interchangeable, so the truth
#: side has to be normalised before it can be compared with a predicted action.
TRUTH_ALIAS = {"ClutterTarget": "Clutter"}


def collect(model, dataset, device, config) -> list[dict]:
    """Per-parent final logits argmax, plus the base (4-class) truth label."""
    out: list[dict] = []
    model.eval()
    loader = build_dataloader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
        seed=int(config.training["dataloader_seed"]),
    )
    with torch.inference_mode():
        for batch in loader:
            output = model(**model_input_kwargs(batch, device))
            probs = torch.softmax(output["sequence_logits"], dim=-1)
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][row])
                out.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "base_label": str(batch["base_label_names"][row]),
                        "pred_id": int(probs[row, length - 1].argmax()),
                    }
                )
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    context = load_context(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    registry, _integrity = _load_registry_tolerant(context)
    print(f"registry status: {registry.get('status')}  "
          f"deployment_eligible: {registry.get('deployment_eligible')}")

    models: dict[str, V28TaskModel] = {}
    for task in (V28_L1_TASK, V28_L2_TASK):
        binding = registry["endpoint_binding"][task]
        model = V28TaskModel(context, task=task).to(device)
        payload = load_checkpoint(
            context, model, binding["path"], task=task,
            seed=int(binding["seed"]), map_location=device,
        )
        model.eval()
        models[task] = model
        print(f"  {task}: checkpoint epoch {int(payload['epoch'])}")

    datasets = {
        task: V28TaskDataset(context, task=task, split="validation")
        for task in (V28_L1_TASK, V28_L2_TASK)
    }
    for task, ds in datasets.items():
        print(f"  {task}: {len(ds)} validation parents")

    l1_records = collect(models[V28_L1_TASK], datasets[V28_L1_TASK], device, context.config)
    l2_records = collect(models[V28_L2_TASK], datasets[V28_L2_TASK], device, context.config)

    l1_names = list(datasets[V28_L1_TASK].class_names)
    l2_names = list(datasets[V28_L2_TASK].class_names)
    l2_by_id = {r["parent_sample_id"]: r for r in l2_records}

    pairs: list[tuple[str, str]] = []
    unmatched = 0
    for record in l1_records:
        truth = TRUTH_ALIAS.get(record["base_label"], record["base_label"])
        route = l1_names[record["pred_id"]]
        if route == L1_CLUTTER:
            prediction = "Clutter"
        else:
            peer = l2_by_id.get(record["parent_sample_id"])
            if peer is None:
                unmatched += 1
                continue
            prediction = l2_names[peer["pred_id"]]
        pairs.append((truth, prediction))

    index = {name: position for position, name in enumerate(FOUR)}
    usable = [(t, p) for t, p in pairs if t in index and p in index]
    dropped = len(pairs) - len(usable)

    print()
    print(f"joined pairs      : {len(pairs)}")
    print(f"no L2 peer        : {unmatched}")
    print(f"outside 4 classes : {dropped}")
    print()

    if not usable:
        print("no usable pairs")
        return 1

    matrix = confusion_metrics(
        [{"true_id": index[t], "pred_id": index[p]} for t, p in usable],
        class_names=list(FOUR),
        scope="four_class_cascade_closed_set",
    )
    print(json.dumps(matrix, ensure_ascii=False, indent=2))

    # Show the cascade explicitly so the reader can see where L1 routing shows up.
    route_counts: dict[str, int] = {}
    for record in l1_records:
        key = l1_names[record["pred_id"]]
        route_counts[key] = route_counts.get(key, 0) + 1
    print()
    print("L1 route counts:", json.dumps(route_counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
