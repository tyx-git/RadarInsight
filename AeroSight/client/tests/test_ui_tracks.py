from __future__ import annotations

import os
import queue
import signal
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsView, QPushButton)

from aerosight.core.radar import RadarFrame, RadarObservation
from aerosight.app_config import load_config
from aerosight.main import _install_shutdown_timer
from aerosight.pipeline.rrtsp_source import (
    RRTSPConnection,
    parse_sdp_update_rate,
)
from aerosight.ui.main_window import MainWindow
from aerosight.ui.map_view import (
    FIXED_RING_VALUES_M,
    MapRadarView,
    RadarTagItem,
    RADAR_SECTOR_WIDTH_DEG,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def observation(track_id: int, step: int) -> RadarObservation:
    return RadarObservation(
        stream_id="stream-a",
        track_id=track_id,
        range_m=700.0 + track_id * 15 + step * 12,
        azimuth_deg=-48.0 + track_id * 7 + step * 0.8,
        ground_range_m=680.0 + track_id * 15 + step * 12,
        altitude_m=90.0 + track_id,
        speed_mps=18.0 + step,
    )


def test_map_retains_ten_trajectories_as_individual_points(app):
    view = MapRadarView()
    view.resize(1100, 760)
    view.begin_stream("stream-a")

    frame_id = 0
    latest = None
    labels = {}
    for track_id in range(1, 13):
        for step in range(5):
            latest = observation(track_id, step)
            labels[("stream-a", track_id)] = ("drone", 0.9)
            view.record_frame(
                [latest],
                frame_id=frame_id,
                timestamp=float(frame_id) / 10,
            )
            frame_id += 1
    view.update_targets(
        [latest], labels, frame_id=frame_id - 1, timestamp=(frame_id - 1) / 10)

    assert view.track_count == 10
    assert {key[1] for key, _obs in view.display_targets()} == set(range(3, 13))
    assert all(len(history) == 5 for history in view.trajectories.values())

    point_items = [
        item for item in view._dynamic_items
        if isinstance(item, QGraphicsEllipseItem)
    ]
    path_lines = [
        item for item in view._dynamic_items
        if isinstance(item, QGraphicsLineItem)
    ]
    assert len(point_items) == 50
    assert not path_lines
    assert view.map_visible
    crosshair_items = [
        item for item in view.scene().items()
        if item.data(0) == "radar-crosshair"
    ]
    assert len(crosshair_items) == 4
    assert all(isinstance(item, QGraphicsLineItem) for item in crosshair_items)
    line_lengths = sorted(item.line().length() for item in crosshair_items)
    assert line_lengths[0] < line_lengths[-1]
    assert not any(
        isinstance(item, QGraphicsLineItem)
        and item.data(0) != "radar-crosshair"
        for item in view.scene().items()
    )

    view.set_trails_visible(False)
    assert not view.trails_visible
    view.set_labels_visible(False)
    assert not view.labels_visible
    view.set_map_visible(False)
    assert not view.map_visible
    assert view.track_count == 10
    assert not view.trails_visible
    assert not view.labels_visible
    view.set_map_visible(True)
    assert view.map_visible
    assert not any(
        isinstance(item, QGraphicsLineItem)
        and item.data(0) != "radar-crosshair"
        for item in view.scene().items()
    )
    view.zoom_by(1.2)
    assert view._zoom > 1.0
    view.center_radar()
    assert view._zoom == 1.0
    view.close()


def test_fixed_north_sector_and_points_are_static(app):
    view = MapRadarView()
    view.resize(1100, 760)
    view.begin_stream("stream-a")
    north = observation(1, 0)
    north.azimuth_deg = 0.0
    east = observation(2, 0)
    east.azimuth_deg = 90.0
    view.record_frame([north, east], frame_id=1, timestamp=1.0)
    view.update_targets(
        [north, east], frame_id=1, timestamp=1.0)

    sector_items = [
        item for item in view.scene().items()
        if item.data(0) == "radar-fixed-sector"
    ]
    assert len(sector_items) == 1
    assert isinstance(sector_items[0], QGraphicsPathItem)
    assert RADAR_SECTOR_WIDTH_DEG == 60.0
    assert sector_items[0].path().elementCount() == 15

    point_items = [
        item for item in view._dynamic_items
        if isinstance(item, QGraphicsEllipseItem)
    ]
    assert len(point_items) == 2

    view.clear_trajectories()
    assert not [
        item for item in view._dynamic_items
        if isinstance(item, QGraphicsEllipseItem)
    ]
    view.close()


def test_target_tag_expands_on_hover_state(app):
    tag = RadarTagItem(
        ("#42  drone  87%", "1200m  18.0m/s"),
        QColor("#4DB6AC"),
        detail_lines=(
            "航迹 #42",
            "来源：stream-a",
            "平面距离（地图距离）：1200.0 m",
            "方位角：12.0°",
        ),
    )
    compact_height = tag.boundingRect().height()
    assert tag.acceptHoverEvents()
    assert not tag.expanded

    tag.set_expanded(True)
    assert tag.expanded
    assert tag.boundingRect().height() > compact_height
    assert "方位角" in tag.toPlainText()

    tag.set_expanded(False)
    assert not tag.expanded
    assert tag.boundingRect().height() == compact_height


def test_sdp_update_rate_is_read_from_track_media():
    sdp = """v=0
m=application 5004 RTP/AVP 96
a=fmtp:96 frameRate=0.333333
m=application 5006 RTP/AVP 97
a=fmtp:97 encoding=tlv;updateRate=0.333333;coord=polar
m=application 5008 RTP/AVP 98
"""
    assert parse_sdp_update_rate(sdp) == pytest.approx(0.333333)
    assert parse_sdp_update_rate("m=application 5006 RTP/AVP 97\n") is None


def test_remote_inference_wait_does_not_timeout():
    connection = RRTSPConnection("rtsp://127.0.0.1:8554/radar")
    connection.has_inference = True
    connection.track_q.put((42, []))
    result = {}
    finished = threading.Event()

    def receive():
        result["value"] = connection.receive_paired(timeout=0.01)
        finished.set()

    worker = threading.Thread(target=receive)
    worker.start()
    assert not finished.wait(0.05)
    connection.inference_q.put((42, {7: ("drone", 0.9)}))
    assert finished.wait(1.0)
    worker.join(timeout=1.0)
    assert result["value"] == (42, [], None)
    assert connection.last_remote_labels == {7: ("drone", 0.9)}
    connection.close()


def test_builtin_map_asset_is_loaded_by_map_view(app):
    cfg = load_config()
    map_path = Path(cfg["instances"][0]["map"]["image_path"])
    assert map_path.name == "map.png"
    assert map_path.is_file()
    assert cfg["instances"][0]["map"]["radar_x"] == 975.0
    assert cfg["instances"][0]["map"]["radar_y"] == 700.0

    view = MapRadarView()
    view.resize(1100, 700)
    view.configure(cfg["instances"][0]["map"], 3000)
    view.show()
    app.processEvents()
    assert view._map_item is not None
    assert view._map_cover_scale >= 1.0
    map_rect = view._map_item.sceneBoundingRect().adjusted(
        -0.001, -0.001, 0.001, 0.001)
    assert map_rect.contains(view._viewport_scene_rect())
    range_rings = [
        item for item in view.scene().items()
        if isinstance(item, QGraphicsEllipseItem)
        and item.zValue() == 2
    ]
    assert range_rings
    assert all(item.pen().widthF() >= 1.0 for item in range_rings)
    assert view._ring_values() == list(FIXED_RING_VALUES_M)
    ring_radii = sorted(
        round(item.rect().width() / 2.0, 3)
        for item in range_rings)
    assert ring_radii == [712.5, 1425.0, 2850.0, 4275.0]
    view.close()


def test_batch_frames_keep_history_for_stable_display_slots(app):
    view = MapRadarView()
    view.resize(1100, 760)
    view.begin_stream("stream-a")

    for batch in range(5):
        observations = [
            observation(track_id, batch)
            for track_id in range(1, 21)
        ]
        view.record_frame(
            observations,
            frame_id=batch,
            timestamp=float(batch) * 3.0,
        )

    assert view.track_count == 10
    assert {key[1] for key, _obs in view.display_targets()} == set(range(1, 11))
    assert all(len(history) == 5 for history in view.trajectories.values())
    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    view.close()


def test_remote_classification_is_rendered_in_target_table(app, tmp_path):
    cfg = {
        "ui": {
            "window_title": "AeroSight test",
            "selected_instance_id": "stream-a",
            "render_fps": 30,
        },
        "instances": [{
            "id": "stream-a",
            "name": "Radar A",
            "enabled": True,
            "map": {},
            "display": {"plane_range_m": 3000},
            "model": {"onnx_path": ""},
            "source": {
                "kind": "rrtsp",
                "rrtsp": {"remote_inference": True},
            },
        }],
    }
    message_q = queue.Queue()
    inference_q = queue.Queue()
    result_q = queue.Queue()
    stop_event = threading.Event()
    window = MainWindow(
        cfg,
        message_q,
        inference_q,
        result_q,
        queue.Queue(),
        queue.Queue(),
        stop_event,
    )
    for step in range(5):
        message_q.put(("frame", RadarFrame(
            stream_id="stream-a",
            stream_name="Radar A",
            frame_id=4 + step,
            timestamp=(4 + step) / 10,
            observations=[observation(42, step)],
            remote_labels={42: ("drone", 0.87)},
        )))

    window._tick()

    window.resize(1120, 680)
    window.show()
    app.processEvents()
    window._set_top_state("正在建立连接")
    app.processEvents()
    assert window.top_state.fontMetrics().horizontalAdvance(
        window.top_state.text()) <= window.top_state.contentsRect().width()
    top_right = window.top_state.mapTo(window, window.top_state.rect().topRight()).x()
    settings_left = window.settings_btn.mapTo(window, window.settings_btn.rect().topLeft()).x()
    assert top_right <= settings_left

    assert window.map_view.track_count == 1
    assert len(window.map_view.trajectories[("stream-a", 42)]) == 5
    assert window.targets_table.item(0, 0).text() == "#42"
    assert window.targets_table.item(0, 1).text() == "drone"
    assert window.targets_table.item(0, 2).text() == "87%"
    assert window.value_model.text() == "\u672a\u6307\u5b9a"
    assert window.start_detection_button.objectName() == "StartDetectionButton"
    assert window.start_detection_button.text() == "\u5f00\u59cb\u63a2\u6d4b"
    assert window.emergency_stop_button.objectName() == "EmergencyStopButton"
    assert window.emergency_stop_button.text() == "\u7d27\u6025\u505c\u673a"
    assert window.start_detection_button.parentWidget() is window.map_surface
    assert window.emergency_stop_button.parentWidget() is window.map_surface
    assert window.start_detection_button.size().width() == 124
    assert window.start_detection_button.size().height() == 36
    assert window.emergency_stop_button.size().width() == 124
    assert window.emergency_stop_button.size().height() == 36
    group_width = (
        window.start_detection_button.width()
        + 14
        + window.emergency_stop_button.width())
    expected_left = (window.map_surface.width() - group_width) // 2
    assert window.start_detection_button.geometry().left() == expected_left
    assert window.emergency_stop_button.geometry().left() == (
        expected_left + window.start_detection_button.width() + 14)
    assert window.start_detection_button.geometry().right() < window.emergency_stop_button.geometry().left()
    assert window.start_detection_button.geometry().bottom() == 462
    assert window.emergency_stop_button.geometry().bottom() == 462
    assert window.map_surface.rect().bottom() - window.start_detection_button.geometry().bottom() == 68

    model_path = tmp_path / "custom-cascade.onnx"
    model_path.write_bytes(b"placeholder")
    window.cfg["instances"][0]["model"]["onnx_path"] = str(model_path)
    window._metrics()
    assert window.value_model.text() == "custom-cascade.onnx"

    window.cfg["instances"][0]["model"]["onnx_path"] = (
        str(tmp_path / "missing.onnx"))
    window._metrics()
    assert window.value_model.text() == "\u672a\u6307\u5b9a"

    window.map_button.click()
    assert not window.map_view.map_visible
    assert window.map_view.trails_visible
    assert window.map_view.labels_visible
    window.map_button.click()
    assert window.map_view.map_visible
    window.trails_button.click()
    assert not window.map_view.trails_visible
    window.labels_button.click()
    assert not window.map_view.labels_visible
    window.findChild(QPushButton, "ZoomIn").click()
    assert window.map_view._zoom > 1.0
    window.findChild(QPushButton, "CenterView").click()
    assert window.map_view._zoom == 1.0
    window.targets_table.selectRow(0)
    assert window.map_view.selected_track == ("stream-a", 42)
    window.close()


def test_sigint_requests_clean_qt_event_loop_exit(app):
    signums = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {signum: signal.getsignal(signum) for signum in signums}
    shutdown_requested, timer = _install_shutdown_timer(app, interval_ms=10)
    quit_events = []
    app.aboutToQuit.connect(lambda: quit_events.append(True))
    QTimer.singleShot(0, lambda: signal.raise_signal(signal.SIGINT))
    watchdog = QTimer(app)
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(app.quit)
    watchdog.start(2000)

    try:
        app.exec()
        assert shutdown_requested.is_set()
        assert quit_events
    finally:
        timer.stop()
        watchdog.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
