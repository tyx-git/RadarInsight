from __future__ import annotations

import os
import queue
import signal
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsView,
    QPushButton)

from aerosight.core.radar import RadarFrame, RadarObservation
from aerosight.main import _install_shutdown_timer
from aerosight.ui.main_window import MainWindow
from aerosight.ui.map_view import MapRadarView


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
    assert not any(
        isinstance(item, QGraphicsLineItem)
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
        for item in view.scene().items()
    )
    view.zoom_by(1.2)
    assert view._zoom > 1.0
    view.center_radar()
    assert view._zoom == 1.0
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
