import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.coord_transform.radar_geo import RadarGeoTransformer


class TruthMatcher:
    """基于时间和空间门限将雷达轨迹与 GNSS 真值进行自动匹配。"""

    def __init__(self, session_dir: str, radar_yaw_deg: float = 0.0):
        self.session_dir = Path(session_dir)
        self.track_csv = self.session_dir / "radar/track/track_data.csv"
        self.gnss_dir = self.session_dir / "truth/gnss"
        self.output_dir = self.session_dir / "labels"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geo_transformer = RadarGeoTransformer(radar_yaw_deg=radar_yaw_deg)

    @staticmethod
    def _normalize_track_id(track_id: Any) -> int:
        if hasattr(track_id, "item"):
            track_id = track_id.item()
        return int(str(track_id))

    def _load_truth_frame(self) -> pd.DataFrame | None:
        if not self.gnss_dir.exists():
            return None

        truth_frames: list[pd.DataFrame] = []
        for csv_path in sorted(self.gnss_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path)
            if frame.empty:
                continue
            frame["source_file"] = csv_path.name
            truth_frames.append(frame)

        if not truth_frames:
            return None

        return pd.concat(truth_frames, ignore_index=True, sort=False)

    def match_by_time_and_space(
        self,
        t_thresh: float = 1.0,
        d_thresh: float = 20.0,
        v_thresh: float = 3.0,
        az_thresh: float = 5.0,
    ) -> list[dict[str, Any]]:
        if not self.track_csv.exists():
            print("[!] 雷达轨迹数据缺失。")
            return []

        df_radar = pd.read_csv(self.track_csv)
        df_gnss = self._load_truth_frame()
        if df_gnss is None or df_gnss.empty:
            print("[!] GNSS 数据缺失，跳过自动匹配。")
            return []

        matched_results: list[dict[str, Any]] = []

        for track_id, radar_group in df_radar.groupby("track_id"):
            track_id_int = self._normalize_track_id(track_id)
            t0, t1 = radar_group["t"].min(), radar_group["t"].max()
            gnss_window = df_gnss[(df_gnss["t"] >= t0 - t_thresh) & (df_gnss["t"] <= t1 + t_thresh)]
            if len(gnss_window) < 5:
                continue

            radar_mid = radar_group.iloc[len(radar_group) // 2]
            best_match: dict[str, Any] | None = None
            best_score = float("inf")

            for _, gnss_mid in gnss_window.iterrows():
                required_cols = ["range_m", "radial_speed_mps", "azimuth_deg"]
                if not all(column in radar_mid for column in required_cols) or not all(
                    column in gnss_mid for column in required_cols
                ):
                    dist_diff = abs(float(radar_mid["range_m"]) - float(gnss_mid["range_m"]))
                    score = dist_diff
                    match_ok = dist_diff <= d_thresh
                else:
                    dist_diff = abs(float(radar_mid["range_m"]) - float(gnss_mid["range_m"]))
                    speed_diff = abs(float(radar_mid["radial_speed_mps"]) - float(gnss_mid["radial_speed_mps"]))
                    az_diff = abs(float(radar_mid["azimuth_deg"]) - float(gnss_mid["azimuth_deg"]))
                    if az_diff > 180:
                        az_diff = 360 - az_diff

                    score = dist_diff + speed_diff * 2.0 + az_diff * 0.5
                    match_ok = (dist_diff <= d_thresh) and (speed_diff <= v_thresh) and (az_diff <= az_thresh)

                if not match_ok or score >= best_score:
                    continue

                label = gnss_mid.get("label", "unknown")
                if pd.isna(label):
                    label = "unknown"

                best_score = score
                best_match = {
                    "track_id": track_id_int,
                    "label_main": str(label),
                    "confidence": "gold",
                    "evidence": f"gnss_match:{gnss_mid.get('source_file', 'unknown')}",
                }

            if best_match is not None:
                matched_results.append(best_match)

        labels_file = self.output_dir / "labels_matched.jsonl"
        with open(labels_file, "w", encoding="utf-8") as file:
            for result in matched_results:
                file.write(json.dumps(result) + "\n")

        print(f"[*] 自动匹配完成，生成 {len(matched_results)} 条标注。")
        return matched_results

    def export_to_cvat(self, matched_results: list[dict[str, Any]]):
        pass


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath("."))
    matcher = TruthMatcher("./data/sessions/test_session")
    print("[*] Truth matcher initialized.")
