from __future__ import annotations

import math
import os
import queue
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from aerosight.core.radar import RadarFrame, RadarObservation, StreamStatus
from aerosight.core.track_analysis import TrackAnalysisHistory, TrackAnalysisSample
from aerosight.pipeline.rrtsp_source import RRTSPConnection, RTP_CLOCK_HZ
from aerosight.ui.main_window import MainWindow
from aerosight.ui.map_view import RadarTagItem
from aerosight.ui.track_analysis import TrackAnalysisDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def frame(step: int, *, stream_id="stream-a", track_id=534,
          rtp_timestamp=None, timestamp=None):
    return RadarFrame(
        stream_id=stream_id, stream_name="Radar " + stream_id,
        frame_id=step,
        timestamp=float(step * 3 if timestamp is None else timestamp),
        rtp_timestamp=rtp_timestamp,
        observations=[RadarObservation(
            stream_id=stream_id, track_id=track_id,
            range_m=1000 + step, ground_range_m=990 + step,
            azimuth_deg=3, altitude_m=100 + step,
            radial_speed_mps=-14 + step / 10,
        )],
        remote_labels={track_id: ("drone", 0.9)},
    )


@pytest.fixture
def window(app):
    cfg = {
        "ui": {"render_fps": 30, "selected_instance_id": "stream-a"},
        "instances": [
            {
                "id": stream, "name": "Radar " + stream, "enabled": True,
                "map": {}, "display": {"plane_range_m": 3000},
                "model": {"onnx_path": ""},
                "source": {"kind": "rrtsp", "rrtsp": {"remote_inference": True}},
            }
            for stream in ("stream-a", "stream-b")
        ],
    }
    result = MainWindow(
        cfg, queue.Queue(), queue.Queue(), queue.Queue(),
        queue.Queue(), queue.Queue(), threading.Event(),
    )
    result.timer.stop()
    result.clock.stop()
    yield result
    result.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_history_preserves_all_batches_and_separates_streams_and_tracks():
    history = TrackAnalysisHistory()
    for step in range(40):
        a = frame(step, rtp_timestamp=900_000 + step * 270_000, timestamp=999)
        history.record_frame(a)
        assert history.record_frame(a) == {}
        history.record_frame(frame(step, stream_id="stream-b"))
    a_samples = history.samples_for("stream-a", 534)
    assert len(a_samples) == 40
    assert a_samples[-1].elapsed_seconds == 117
    assert a_samples[0].radial_speed_mps == -14
    assert len(history.samples_for("stream-b", 534)) == 40
    assert history.samples_for("stream-a", 999) == ()
    history.record_frame(frame(40, track_id=999, rtp_timestamp=11_700_000))
    assert history.samples_for("stream-a", 999)[0].elapsed_seconds == 0
    assert len(history.samples_for("stream-a", 534)) == 40


def test_history_handles_rtp_wrap_fallback_invalid_values_and_reset():
    history = TrackAnalysisHistory()
    history.record_frame(frame(0, rtp_timestamp=2**32 - 90_000))
    invalid = frame(1, rtp_timestamp=180_000)
    invalid.observations[0].radial_speed_mps = math.nan
    invalid.observations[0].altitude_m = math.inf
    history.record_frame(invalid)
    sample = history.samples_for("stream-a", 534)[-1]
    assert sample.elapsed_seconds == 3
    assert sample.radial_speed_mps is None
    assert sample.altitude_m is None
    assert history.record_frame(frame(2, rtp_timestamp=90_000)) == {}
    assert len(history.samples_for("stream-a", 534)) == 2
    history.record_frame(frame(0, stream_id="stream-b", timestamp=50))
    history.record_frame(frame(1, stream_id="stream-b", timestamp=53.5))
    assert history.samples_for("stream-b", 534)[-1].elapsed_seconds == 3.5
    history.reset_stream("stream-a")
    history.record_frame(frame(0, rtp_timestamp=900_000))
    assert history.samples_for("stream-a", 534)[0].elapsed_seconds == 0
    assert len(history.samples_for("stream-b", 534)) == 2


def test_chart_draws_lines_only_and_leaves_gaps_for_missing_values(app):
    dialog = TrackAnalysisDialog("stream-a", 534, "Radar A")
    dialog.show()
    app.processEvents()
    dialog.append_samples([TrackAnalysisSample(0, -14, 100)])
    assert dialog.sample_count == 1
    assert not dialog.speed_curve.segments[0].pointsVisible()
    assert not dialog.altitude_curve.segments[0].pointsVisible()
    assert dialog.speed_axis.min() < -14 < dialog.speed_axis.max()
    assert dialog.altitude_axis.min() < 100 < dialog.altitude_axis.max()
    dialog.append_samples([
        TrackAnalysisSample(3, -15, None),
        TrackAnalysisSample(6, None, 100),
        TrackAnalysisSample(9, -13, 100),
    ])
    assert [segment.count() for segment in dialog.speed_curve.segments] == [2, 1]
    assert [segment.count() for segment in dialog.altitude_curve.segments] == [1, 2]
    assert dialog.time_axis.max() == 9
    app.processEvents()
    assert len([marker for marker in dialog.chart.legend().markers()
                if marker.isVisible()]) == 2
    assert not dialog.isModal()
    assert dialog.windowFlags() & Qt.WindowType.WindowMinMaxButtonsHint
    assert dialog.minimumSize().width() == 680
    dialog.reset_history()
    assert dialog.sample_count == 0
    assert dialog.speed_curve.segments[0].count() == 0
    assert len(dialog.altitude_curve.segments) == 1
    dialog.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(dialog)


@pytest.mark.parametrize("track_id", [534, 4_000_000_001])
def test_tag_right_click_opens_live_window_and_reuses_it(window, app, track_id):
    for step in range(35):
        window.message_q.put(("frame", frame(step, track_id=track_id)))
    window._tick()
    window.show()
    app.processEvents()
    tag = next(item for item in window.map_view._dynamic_items
               if isinstance(item, RadarTagItem))
    point = window.map_view.mapFromScene(
        tag.mapToScene(QPointF(20, 10)))
    QTest.mouseMove(window.map_view.viewport(), QPoint(1, 1))
    app.processEvents()
    QTest.mouseMove(window.map_view.viewport(), point)
    app.processEvents()
    assert tag.expanded
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, point,
        window.map_view.viewport().mapToGlobal(point),
    )
    QApplication.sendEvent(window.map_view.viewport(), event)
    app.processEvents()
    key = ("stream-a", track_id)
    assert key in window.analysis_windows
    dialog = window.analysis_windows[key]
    assert dialog.isVisible()
    assert dialog.sample_count == 35
    assert len(window.map_view.trajectories[key]) == 29
    window.open_track_analysis(*key)
    assert window.analysis_windows[key] is dialog
    assert dialog.sample_count == 35
    window._tick()
    assert dialog.sample_count == 35
    for step in (35, 36, 37):
        window.message_q.put(("frame", frame(step, track_id=track_id)))
    window._tick()
    assert dialog.sample_count == 38
    assert dialog.speed_curve.segments[0].at(37).x() == 111
    assert dialog.altitude_curve.segments[0].at(37).y() == 137
    window.message_q.put(("frame", frame(0, stream_id="stream-b", track_id=track_id)))
    window._tick()
    window.open_track_analysis("stream-b", track_id)
    other = window.analysis_windows[("stream-b", track_id)]
    window.instance_combo.setCurrentIndex(1)
    assert dialog.sample_count == 38
    assert other.sample_count == 1
    dialog.close()
    assert key not in window.analysis_windows
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(dialog)
    window.open_track_analysis(*key)
    assert window.analysis_windows[key].sample_count == 38


def test_disconnect_holds_history_and_reconnect_starts_new_segment(window):
    key = ("stream-a", 534)
    window.message_q.put(("status", StreamStatus("stream-a", "Radar A", "connected")))
    window.message_q.put(("frame", frame(0)))
    window.message_q.put(("frame", frame(1)))
    window._tick()
    window.open_track_analysis(*key)
    dialog = window.analysis_windows[key]
    window.message_q.put(("status", StreamStatus("stream-a", "Radar A", "error")))
    window._tick()
    assert dialog.sample_count == 2
    window.message_q.put(("status", StreamStatus("stream-a", "Radar A", "connecting")))
    window._tick()
    assert dialog.sample_count == 2
    window.message_q.put(("status", StreamStatus("stream-a", "Radar A", "connected")))
    window._tick()
    assert dialog.sample_count == 0
    assert window.analysis_history.samples_for(*key) == ()
    window.message_q.put(("frame", frame(0, timestamp=30)))
    window._tick()
    assert dialog.sample_count == 1
    assert dialog.speed_curve.segments[0].at(0).x() == 0
    window.close()
    assert window.analysis_windows == {}
    assert window.stop_event.is_set()


def test_unpaired_tracks_do_not_advance_analysis(window):
    window.open_track_analysis("stream-a", 534)
    dialog = window.analysis_windows[("stream-a", 534)]
    connection = RRTSPConnection("rtsp://127.0.0.1:8554/radar")
    connection.has_inference = True
    ticks = 270_000
    connection.track_q.put((ticks, [{
        "id": 534, "fields": {
            "range_m": 1000, "elevation_deg": 5, "radial_speed_mps": -14,
        },
    }]))
    waiting = threading.Event()
    done = threading.Event()
    errors = []

    def receive():
        try:
            waiting.set()
            timestamp, tracks, _rd = connection.receive_paired(timeout=0.1)
            payload = RadarFrame(
                "stream-a", "Radar A", 0, timestamp / RTP_CLOCK_HZ,
                rtp_timestamp=timestamp,
                observations=[RadarObservation.from_track("stream-a", track)
                              for track in tracks],
                remote_labels=dict(connection.last_remote_labels),
            )
            window.message_q.put(("frame", payload))
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    try:
        assert waiting.wait(1)
        connection.inference_q.put((ticks + 1, {534: ("bird", 0.9)}))
        assert not done.wait(0.05)
        window._tick()
        assert dialog.sample_count == 0
        connection.inference_q.put((ticks, {534: ("drone", 0.9)}))
        assert done.wait(1)
        assert not errors
        window._tick()
        assert dialog.sample_count == 1
        assert dialog.speed_curve.segments[0].at(0).y() == -14
    finally:
        connection.close()
        worker.join(timeout=1)
