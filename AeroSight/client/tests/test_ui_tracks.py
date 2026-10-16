from __future__ import annotations

import os
import math
import queue
import signal
import struct
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsView, QPushButton)

from aerosight.core.radar import RadarFrame, RadarObservation
from aerosight.app_config import default_instance, load_config
from aerosight.main import _install_shutdown_timer
from aerosight.pipeline.rrtsp_source import (
    RRTSPConnection,
    parse_sdp_update_rate,
)
from aerosight.ui.main_window import MainWindow
from aerosight.ui.settings_dialog import BasicSettingsDialog
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
    assert view._radar_display_scale > 0.0
    map_rect = view._map_item.sceneBoundingRect().adjusted(
        -0.001, -0.001, 0.001, 0.001)
    assert map_rect.width() > 0.0
    assert map_rect.height() > 0.0
    calibration = cfg["instances"][0]["map"]["pixels_per_100m"]
    radar_x = cfg["instances"][0]["map"]["radar_x"]
    radar_y = cfg["instances"][0]["map"]["radar_y"]
    origin = view._map_item.mapToScene(QPointF(radar_x, radar_y))
    probe = view._map_item.mapToScene(
        QPointF(radar_x + calibration, radar_y))
    calibrated_distance = math.hypot(
        probe.x() - origin.x(), probe.y() - origin.y())
    initial_display_scale = view._radar_display_scale
    view.resize(1600, 500)
    app.processEvents()
    origin_after_resize = view._map_item.mapToScene(QPointF(radar_x, radar_y))
    probe_after_resize = view._map_item.mapToScene(
        QPointF(radar_x + calibration, radar_y))
    assert view._radar_display_scale == pytest.approx(initial_display_scale)
    resized_distance = math.hypot(
        probe_after_resize.x() - origin_after_resize.x(),
        probe_after_resize.y() - origin_after_resize.y(),
    )
    assert resized_distance == pytest.approx(
        calibrated_distance
    )
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
    assert ring_radii == [
        round(value * view._radar_display_scale, 3)
        for value in FIXED_RING_VALUES_M
    ]
    assert FIXED_RING_VALUES_M == (500.0, 1000.0, 2000.0, 3000.0, 5000.0)
    assert view._effective_plane_range_m() == 5000.0
    outer_ring_radius = ring_radii[-1]
    assert outer_ring_radius * view._screen_pixels_per_meter() == pytest.approx(
        view._view_radius_px())
    assert view._map_edge_distance_m == pytest.approx(outer_ring_radius)
    target = RadarObservation(
        stream_id="stream-a", track_id=99, ground_range_m=1000.0,
        azimuth_deg=0.0,
    )
    target_position = view._target_position(target)
    assert math.hypot(target_position.x(), target_position.y()) == pytest.approx(
        1000.0 * view._radar_display_scale)
    map_pixel = view._map_item.mapFromScene(target_position)
    assert map_pixel.x() == pytest.approx(radar_x)
    assert map_pixel.y() == pytest.approx(
        radar_y - calibration * 10 * view._radar_display_scale)
    view.resize(680, 420)
    app.processEvents()
    resized_target_position = view._target_position(target)
    resized_map_pixel = view._map_item.mapFromScene(resized_target_position)
    assert resized_map_pixel.x() == pytest.approx(radar_x)
    assert resized_map_pixel.y() == pytest.approx(
        radar_y - calibration * 10 * view._radar_display_scale)
    view.set_map_visible(False)
    hidden_map_position = view._target_position(target)
    assert hidden_map_position.x() == pytest.approx(resized_target_position.x())
    assert hidden_map_position.y() == pytest.approx(resized_target_position.y())
    view.set_map_visible(True)
    assert view._target_position(target) == resized_target_position
    view.close()


def test_missing_position_fields_are_rejected_and_large_track_payload_is_safe(app):
    from aerosight.core.radar import RadarObservation
    from aerosight.pipeline.rrtsp_source import (
        TRACK_FRAME_HEAD,
        TRACK_VALUE_HEAD,
        MAX_RTP_DATAGRAM_SIZE,
        RTP_HEAD,
        TLV_HEAD,
        parse_track_payload,
    )
    with pytest.raises(ValueError, match="finite range_m"):
        RadarObservation.from_track(
            "stream-a",
            {"id": 1, "fields": {"range_m": 1000, "elevation_deg": 2}},
        )

    values = struct.pack(">f", 1000.0) * 8
    track_value = TRACK_VALUE_HEAD.pack(1, 0, 5, 0xFF) + values
    tlv = TLV_HEAD.pack(1, len(track_value)) + track_value
    payload = TRACK_FRAME_HEAD.pack(1, 91, 0, 0) + tlv * 91
    assert len(payload) > 4096
    assert len(payload) + RTP_HEAD.size <= MAX_RTP_DATAGRAM_SIZE
    assert len(parse_track_payload(payload)) == 91
    with pytest.raises(ValueError, match="truncated"):
        parse_track_payload(payload[:-1])
    short_value = TRACK_VALUE_HEAD.pack(1, 0, 5, 1) + b"\x00\x00"
    malformed = (
        TRACK_FRAME_HEAD.pack(1, 1, 0, 0)
        + TLV_HEAD.pack(1, len(short_value))
        + short_value
    )
    with pytest.raises(ValueError, match="truncated float field"):
        parse_track_payload(malformed)


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


def test_settings_apply_independent_periods_for_each_instance(app):
    cfg = load_config()
    cfg["instances"] = [
        default_instance("radar-n", "N", "rrtsp"),
        default_instance("radar-t", "T", "rrtsp"),
    ]
    cfg["ui"]["selected_instance_id"] = "radar-n"
    receiver_cmd, inference_cmd = queue.Queue(), queue.Queue()
    dialog = BasicSettingsDialog(cfg, receiver_cmd, inference_cmd)
    dialog.show()
    app.processEvents()
    try:
        assert dialog.backend_send_period.suffix() == " s"
        assert dialog.frontend_scan_period.suffix() == " s"
        assert dialog.backend_send_period.value() == 1.0
        assert dialog.frontend_scan_period.value() == 3.0
        dialog.backend_send_period.setValue(1.25)
        dialog.frontend_scan_period.setValue(7.5)
        dialog.instance_table.selectRow(1)
        assert dialog.backend_send_period.value() == 1.0
        assert dialog.frontend_scan_period.value() == 3.0
        dialog.backend_send_period.setValue(0.75)
        dialog.frontend_scan_period.setValue(4.5)
        dialog.instance_table.selectRow(0)
        assert dialog.backend_send_period.value() == 1.25
        assert dialog.frontend_scan_period.value() == 7.5
        dialog.apply_settings()
        assert not dialog.isVisible()
        for commands in (receiver_cmd, inference_cmd):
            command = commands.get_nowait()
            assert command["cmd"] == "apply"
            instances = command["cfg"]["instances"]
            assert [item["source"]["rrtsp"]["send_interval_seconds"] for item in instances] == [1.25, 0.75]
            assert [item["display"]["frontend_scan_period_seconds"] for item in instances] == [7.5, 4.5]
    finally:
        dialog.close()


def test_scan_period_change_preserves_angle_and_pending_reveal(app, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("aerosight.ui.map_view.time.monotonic", lambda: clock[0])
    view = MapRadarView()
    view.set_display_mode("scan")
    view.set_buffered_reveal(True)
    view.begin_stream("stream-a")
    clock[0] = 101.0
    view.advance_scan()
    point = observation(1, 0)
    point.azimuth_deg = -90.0
    view.record_frame([point], frame_id=1, timestamp=1.0)
    key = ("stream-a", 1)
    crossing = view._pending_points[key][0].reveal_at_position
    assert view._scan_position_degrees == pytest.approx(120.0)
    view.set_scan_period(6.0)
    assert view._scan_position_degrees == pytest.approx(120.0)
    assert view._pending_points[key][0].reveal_at_position == crossing
    clock[0] = 102.0
    view.advance_scan()
    assert view._scan_position_degrees == pytest.approx(180.0)
    assert key not in view.trajectories
    clock[0] = 104.0
    view.advance_scan()
    assert len(view.trajectories[key]) == 1
    view.close()
