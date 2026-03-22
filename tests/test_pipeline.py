from src.preprocessing.slicer import DataSlicer
from src.qc.integrity_check import DataIntegrityChecker
from src.labeling.truth_matcher import TruthMatcher
from pathlib import Path

def test_pipeline():
    session_id = "mock_session_20260105"
    session_dir = f"./data/sessions/{session_id}"
    
    print(f"--- 启动流水线测试: {session_id} ---")
    
    # 1. 质检 (QC)
    checker = DataIntegrityChecker(session_dir)
    qc_results = checker.check_integrity()
    print(f"[1/3] QC 完成: {qc_results['status']}, 轨迹数: {qc_results.get('num_tracks')}")
    
    # 2. 数据切片 (Slicing)
    slicer = DataSlicer(session_dir)
    track_clips = slicer.slice_tracks(window_size_s=8.0, overlap=0.5)
    print(f"[2/3] 切片完成: 生成 {len(track_clips)} 个片段")
    
    # 3. 自动匹配 (Matching - 即使没有 GNSS 也会运行逻辑)
    matcher = TruthMatcher(session_dir, radar_yaw_deg=0)
    matches = matcher.match_by_time_and_space()
    print(f"[3/3] 自动匹配完成: 匹配到 {len(matches)} 条真值")
    
    print("--- 流水线测试结束 ---")

if __name__ == "__main__":
    test_pipeline()
