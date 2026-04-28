from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _structured_complex_to_amplitude(dpl: np.ndarray) -> np.ndarray:
    real = dpl["real"].astype(np.float64)
    imag = dpl["imag"].astype(np.float64)
    return np.sqrt(real * real + imag * imag)


def _structured_complex_to_phase(dpl: np.ndarray) -> np.ndarray:
    real = dpl["real"].astype(np.float64)
    imag = dpl["imag"].astype(np.float64)
    return np.arctan2(imag, real)


def export_raw_rd(project_root: Path) -> List[Dict[str, object]]:
    manifest_path = project_root / "data" / "manifests" / "raw_tr_rd_manifest.json"
    output_dir = project_root / "data" / "processed" / "raw_rd"
    _ensure_clean_dir(output_dir)
    records: List[Dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_rows: List[Dict[str, object]] = []

    for item in records:
        rd_path = project_root / Path(item["rd_file"])
        with h5py.File(rd_path, "r") as handle:
            data_time = np.array(handle["DATA_time"]).reshape(-1).astype(float)
            gps_time = np.array(handle["GPS_time_in_data"]).reshape(-1).astype(float) if "GPS_time_in_data" in handle else np.array([], dtype=float)
            iframecnt = np.array(handle["Iframecnt"]).reshape(-1).astype(float) if "Iframecnt" in handle else np.array([], dtype=float)
            dpl = np.array(handle["DPL"])

        amplitude = _structured_complex_to_amplitude(dpl)
        phase = _structured_complex_to_phase(dpl)
        frame_count = int(amplitude.shape[1])
        bin_count = int(amplitude.shape[0])
        frame_energy = amplitude.mean(axis=0)
        summary_df = pd.DataFrame(
            {
                "pair_id": item["pair_id"],
                "timestamp": item["timestamp"],
                "frame_index": np.arange(frame_count, dtype=int),
                "time_raw": data_time[:frame_count],
                "gps_time_raw": gps_time[:frame_count] if gps_time.size else np.nan,
                "iframecnt": iframecnt[:frame_count] if iframecnt.size else np.nan,
                "rd_bin_count": bin_count,
                "frame_energy_mean": frame_energy.astype(float),
                "frame_energy_max": amplitude.max(axis=0).astype(float),
                "frame_energy_min": amplitude.min(axis=0).astype(float),
            }
        )
        output_path = output_dir / f"{item['pair_id'].replace('|', '__')}_rd_summary.csv"
        summary_df.to_csv(output_path, index=False, encoding="utf-8")
        rd_tensor_path = output_dir / f"{item['pair_id'].replace('|', '__')}_rd_tensor.npz"
        np.savez_compressed(
            rd_tensor_path,
            amplitude=amplitude.astype(np.float32),
            phase=phase.astype(np.float32),
            time_raw=data_time[:frame_count].astype(np.float32),
        )

        index_rows.append(
            {
                "dataset_name": "raw_rd",
                "source_name": "raw_tr_rd",
                "sample_id": item["pair_id"],
                "sample_kind": "raw_rd_pair",
                "timestamp": item["timestamp"],
                "label_name_raw": item["label"],
                "label_name": item["label"],
                "frame_count": frame_count,
                "rd_bin_count": bin_count,
                "time_raw_min": float(data_time[:frame_count].min()) if frame_count else np.nan,
                "time_raw_max": float(data_time[:frame_count].max()) if frame_count else np.nan,
                "frame_energy_mean": float(frame_energy.mean()) if frame_count else np.nan,
                "frame_energy_std": float(frame_energy.std()) if frame_count else np.nan,
                "summary_csv_path": output_path.relative_to(project_root).as_posix(),
                "rd_tensor_path": rd_tensor_path.relative_to(project_root).as_posix(),
                "rd_mat_path": rd_path.relative_to(project_root).as_posix(),
            }
        )

    return index_rows


def export_all(project_root: Path) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    rd_index_rows = export_raw_rd(project_root)
    rd_index_df = pd.DataFrame(rd_index_rows)
    rd_index_df.to_csv(processed_root / "raw_rd_index.csv", index=False, encoding="utf-8")
    summary = {
        "raw_rd_exports": int(len(rd_index_rows)),
        "source_counts": rd_index_df["source_name"].value_counts(dropna=False).to_dict() if not rd_index_df.empty else {},
    }
    (processed_root / "raw_rd_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = export_all(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
