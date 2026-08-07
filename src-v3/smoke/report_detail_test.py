from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from radar_ln3.evaluator import write_sample_detail_information


def run() -> dict[str, object]:
    result = {
        "split": "val",
        "parent_metrics": {"macro_f1": 0.75},
        "records": {
            "window": [{"parent_sample_id": "parent-a", "confidence": 0.55}],
            "parent": [{"parent_sample_id": "parent-a", "confidence": 0.55}],
            "stable": [{"parent_sample_id": "parent-a", "confidence": 0.80}],
        },
    }
    with TemporaryDirectory(prefix="radar-ln3-report-") as temporary:
        output_dir = Path(temporary)
        path = write_sample_detail_information(output_dir, result, checkpoint_path="best_model.pt", epoch=3, selection_metric="parent_macro_f1")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != "RADAR-LN3-V2-sample-detail-v1":
            raise AssertionError("sample detail schema mismatch")
        if payload["counts"] != {"window": 1, "parent": 1, "stable": 1, "stable_unconfirmed": 0}:
            raise AssertionError("sample detail records were not preserved")
        if "records" not in payload or len(payload["records"]["stable"]) != 1:
            raise AssertionError("all sample details must be written to the detail artifact")
        result_without_records = dict(result)
        result_without_records.pop("records")
        if "records" in result_without_records:
            raise AssertionError("summary view must not include detailed records")
    output = {
        "status": "passed",
        "detail_file": "sample_detail_information.json",
        "all_records_saved": True,
        "summary_records_removed": True,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    run()
