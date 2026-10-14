# -*- coding: utf-8 -*-
"""AeroSight main window."""
from __future__ import annotations

import logging
import math
import queue as _q
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from aerosight.core.track_analysis import TrackAnalysisHistory
from aerosight.ui.i18n import t
from aerosight.ui.map_view import MapRadarView
from aerosight.ui.settings_dialog import BasicSettingsDialog
from aerosight.ui.theme import C, category_color
from aerosight.ui.track_analysis import TrackAnalysisDialog
from aerosight.ui.widgets import LedDot

log = logging.getLogger(__name__)

MAP_ACTION_BOTTOM = 68
MAP_ACTION_GAP = 14
MAP_ACTION_WIDTH = 124
MAP_ACTION_HEIGHT = 36


def _action_icon(kind: str) -> QIcon:
    """Create the small neutral control glyphs used by the map actions."""
    pixmap = QPixmap(30, 30)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    icon_color = QColor(31, 43, 61, 235)
    pen = QPen(icon_color, 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if kind == "start":
        painter.drawEllipse(4, 4, 22, 22)
        painter.drawArc(7, 7, 16, 16, 35 * 16, 120 * 16)
        painter.drawArc(10, 10, 10, 10, 35 * 16, 120 * 16)
        painter.setBrush(icon_color)
        painter.drawEllipse(13, 13, 4, 4)
    else:
        painter.drawRoundedRect(4, 4, 22, 22, 4, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(icon_color)
        painter.drawRoundedRect(11, 11, 8, 8, 2, 2)
    painter.end()
    return QIcon(pixmap)


class MapSurface(QWidget):
    """Map viewport with controls anchored directly to its bottom edge."""

    def __init__(self, map_view, start_button, emergency_button, parent=None):
        super().__init__(parent)
        self.map_view = map_view
        self.start_button = start_button
        self.emergency_button = emergency_button
        self.map_view.setParent(self)
        self.start_button.setParent(self)
        self.emergency_button.setParent(self)

    def resizeEvent(self, event):
        self.map_view.setGeometry(self.rect())
        button_height = self.start_button.height()
        button_y = max(self.height() - MAP_ACTION_BOTTOM - button_height, 0)
        group_width = (
            self.start_button.width()
            + MAP_ACTION_GAP
            + self.emergency_button.width())
        group_left = max((self.width() - group_width) // 2, 0)
        self.start_button.move(group_left, button_y)
        self.emergency_button.move(
            self.start_button.x() + self.start_button.width() + MAP_ACTION_GAP,
            button_y)
        self.start_button.raise_()
        self.emergency_button.raise_()
        super().resizeEvent(event)


def _local_onnx_name(model_path: str) -> str:
    """Return the local ONNX filename only when the configured file exists."""
    raw_path = str(model_path or "").strip()
    if not raw_path:
        return t("model_unspecified")

    path = Path(raw_path).expanduser()
    if path.suffix.lower() != ".onnx" or not path.is_file():
        return t("model_unspecified")
    return path.name


class MainWindow(QMainWindow):
    def __init__(self, cfg, message_q, inference_q, result_q,
                 receiver_cmd, inference_cmd, stop_event):
        super().__init__()
        self.cfg = cfg
        self.message_q = message_q
        self.inference_q = inference_q
        self.result_q = result_q
        self.receiver_cmd = receiver_cmd
        self.inference_cmd = inference_cmd
        self.stop_event = stop_event

        self.latest_frames = {}
        self.stream_status = {}
        self.inference_by_instance = {}
        self.inferred = {}
        self.arrival_by_instance: dict[str, deque] = {}
        self.fps_smoothed: dict[str, float] = {}
        self.last_fps_at_by_instance: dict[str, float] = {}
        self.current_instance_id = ""
        self.settings_dialog = None
        self.analysis_history = TrackAnalysisHistory()
        self.analysis_windows: dict[tuple[str, int], TrackAnalysisDialog] = {}
        self._target_table_signature = None

        self.setWindowTitle(cfg["ui"].get("window_title", t("app_title")))
        self.resize(1600, 900)
        self.setMinimumSize(1120, 680)
        self._build_layout()
        self._refresh_instance_selector()
        self._apply_preview()

        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick_clock)
        self.clock.start(1000)
        self._tick_clock()

        fps = int(cfg["ui"].get("render_fps", 30))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(int(1000 / fps), 16))

    def _build_layout(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._topbar())
        self.map_view = MapRadarView()
        self.map_view.trackAnalysisRequested.connect(self.open_track_analysis)
        layout.addWidget(self._viewbar())

        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        workspace_layout.addWidget(self._map_surface(), 1)
        workspace_layout.addWidget(self._target_panel())
        layout.addWidget(workspace, 1)
        self.setCentralWidget(root)

        self.sb_status = QLabel(t("ready"))
        self.sb_status.setMinimumWidth(0)
        self.sb_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.sb_time = QLabel()
        self.statusBar().addWidget(self.sb_status)
        self.statusBar().addPermanentWidget(self.sb_time)

    def _map_surface(self):
        """Build the map viewport with bottom-edge action controls."""
        self.start_detection_button = QPushButton(t("start_detection"))
        self.start_detection_button.setObjectName("StartDetectionButton")
        self.start_detection_button.setFixedSize(
            MAP_ACTION_WIDTH, MAP_ACTION_HEIGHT)
        self.start_detection_button.setIcon(_action_icon("start"))
        self.start_detection_button.setIconSize(QSize(22, 22))
        self.start_detection_button.clicked.connect(
            self._on_start_detection_clicked)

        self.emergency_stop_button = QPushButton(t("emergency_stop"))
        self.emergency_stop_button.setObjectName("EmergencyStopButton")
        self.emergency_stop_button.setFixedSize(
            MAP_ACTION_WIDTH, MAP_ACTION_HEIGHT)
        self.emergency_stop_button.setIcon(_action_icon("stop"))
        self.emergency_stop_button.setIconSize(QSize(22, 22))
        self.emergency_stop_button.clicked.connect(
            self._on_emergency_stop_clicked)

        self.map_surface = MapSurface(
            self.map_view,
            self.start_detection_button,
            self.emergency_stop_button)
        return self.map_surface

    def _on_start_detection_clicked(self):
        # TODO: Bind this placeholder to the future detection-start command.
        pass

    def _on_emergency_stop_clicked(self):
        # TODO: Bind this placeholder to the future emergency-stop command.
        pass

    def _topbar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(10)

        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(0)
        title = QLabel(t("brand"))
        title.setObjectName("Brand")
        subtitle = QLabel(t("brand_sub"))
        subtitle.setObjectName("BrandSub")
        brand_layout.addWidget(title)
        brand_layout.addWidget(subtitle)
        header.addLayout(brand_layout)
        header.addStretch(1)

        header.addWidget(QLabel(t("current_instance")))
        self.instance_combo = QComboBox()
        self.instance_combo.setMinimumWidth(180)
        self.instance_combo.setMaximumWidth(280)
        self.instance_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.instance_combo.currentIndexChanged.connect(self._on_instance_selected)
        header.addWidget(self.instance_combo)

        status_group = QHBoxLayout()
        status_group.setSpacing(5)
        self.top_led = LedDot(C.amber, 10)
        self.top_state = QLabel(t("waiting_data"))
        self.top_state.setMinimumWidth(60)
        self.top_state.setMaximumWidth(230)
        self.top_state.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        status_group.addWidget(self.top_led)
        status_group.addWidget(self.top_state, 1)
        header.addLayout(status_group, 1)

        self.settings_btn = QPushButton(t("settings"))
        self.settings_btn.setObjectName("Primary")
        self.settings_btn.clicked.connect(self.open_settings)
        header.addWidget(self.settings_btn)
        layout.addLayout(header)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.card_online, self.value_online = self._mini(t("online_streams"))
        self.card_tracks, self.value_tracks = self._mini(t("display_targets"))
        self.card_fps, self.value_fps = self._mini(t("receive_fps"))
        self.card_model, self.value_model = self._mini(t("model_state"))
        for card in (self.card_online, self.card_tracks, self.card_fps, self.card_model):
            metrics.addWidget(card, 1)
        layout.addLayout(metrics)
        return bar

    @staticmethod
    def _mini(title):
        card = QFrame()
        card.setObjectName("MiniCard")
        card.setMinimumWidth(120)
        card.setMinimumHeight(48)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(0)
        name = QLabel(title)
        name.setObjectName("MiniTitle")
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        value = QLabel("--")
        value.setObjectName("MiniValue")
        value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(name)
        layout.addWidget(value)
        return card, value

    def _viewbar(self):
        bar = QFrame()
        bar.setObjectName("ViewBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(6)

        self.map_button = QPushButton(t("map_layer"))
        self.map_button.setObjectName("ModeToggle")
        self.map_button.setCheckable(True)
        self.map_button.setChecked(True)
        self.map_button.setToolTip(t("map_layer_hint"))
        self.map_button.toggled.connect(self.map_view.set_map_visible)
        layout.addWidget(self.map_button)

        self.trails_button = QPushButton(t("trajectory_points"))
        self.trails_button.setObjectName("ModeToggle")
        self.trails_button.setCheckable(True)
        self.trails_button.setChecked(True)
        self.trails_button.setToolTip(t("trajectory_points"))
        self.trails_button.toggled.connect(self.map_view.set_trails_visible)
        layout.addWidget(self.trails_button)

        self.labels_button = QPushButton(t("map_labels"))
        self.labels_button.setObjectName("ModeToggle")
        self.labels_button.setCheckable(True)
        self.labels_button.setChecked(True)
        self.labels_button.setToolTip(t("map_labels"))
        self.labels_button.toggled.connect(self.map_view.set_labels_visible)
        layout.addWidget(self.labels_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        layout.addWidget(separator)

        zoom_out = QPushButton("\u2212")
        zoom_out.setObjectName("ZoomOut")
        zoom_out.setFixedWidth(36)
        zoom_out.setToolTip(t("zoom_out"))
        zoom_out.clicked.connect(lambda: self.map_view.zoom_by(1.0 / 1.2))
        layout.addWidget(zoom_out)

        zoom_in = QPushButton("+")
        zoom_in.setObjectName("ZoomIn")
        zoom_in.setFixedWidth(36)
        zoom_in.setToolTip(t("zoom_in"))
        zoom_in.clicked.connect(lambda: self.map_view.zoom_by(1.2))
        layout.addWidget(zoom_in)

        center = QPushButton(t("center_view"))
        center.setObjectName("CenterView")
        center.setToolTip(t("center_view"))
        center.clicked.connect(self.map_view.center_radar)
        layout.addWidget(center)
        layout.addStretch(1)
        return bar

    def _target_panel(self):
        panel = QFrame()
        panel.setObjectName("TargetPanel")
        panel.setFixedWidth(326)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(8)

        heading = QHBoxLayout()
        title = QLabel(t("target_history"))
        title.setObjectName("PanelHeader")
        self.target_count = QLabel("00 / 10")
        self.target_count.setObjectName("TargetCount")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.target_count)
        layout.addLayout(heading)

        self.targets_table = QTableWidget(0, 4)
        self.targets_table.setHorizontalHeaderLabels([
            t("target_id"), t("category"), t("confidence"), t("range_short")])
        self.targets_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.targets_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.targets_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.targets_table.setShowGrid(False)
        self.targets_table.setAlternatingRowColors(False)
        self.targets_table.setWordWrap(False)
        self.targets_table.verticalHeader().setVisible(False)
        self.targets_table.verticalHeader().setDefaultSectionSize(40)
        table_header = self.targets_table.horizontalHeader()
        table_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        table_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for index, width in enumerate((50, 82, 60, 78)):
            self.targets_table.setColumnWidth(index, width)
        self.targets_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.targets_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.targets_table.itemSelectionChanged.connect(self._target_selected)
        layout.addWidget(self.targets_table, 1)
        return panel

    def _tick(self):
        self._drain_messages()
        self._drain_results()
        self._refresh_instance_selector()

        frame = self.latest_frames.get(self.current_instance_id)
        observations = self._current_observations()
        self.map_view.update_targets(
            observations,
            self.inferred,
            frame_id=frame.frame_id if frame else None,
            timestamp=frame.timestamp if frame else None,
        )
        self.inferred = {
            key: value for key, value in self.inferred.items()
            if key in self.map_view.latest_by_track
        }
        self._refresh_target_table()
        self._metrics()

    def _drain_messages(self):
        while True:
            try:
                kind, payload = self.message_q.get_nowait()
            except _q.Empty:
                break
            if kind == "frame":
                self.latest_frames[payload.stream_id] = payload
                added = self.analysis_history.record_frame(payload)
                for key, samples in added.items():
                    dialog = self.analysis_windows.get(key)
                    if dialog is not None:
                        dialog.set_stream_name(payload.stream_name)
                        dialog.append_samples(samples)
                if payload.stream_id == self.current_instance_id:
                    self.map_view.record_frame(
                        payload.observations,
                        frame_id=payload.frame_id,
                        timestamp=payload.timestamp,
                    )
                for track_id, label in payload.remote_labels.items():
                    self.inferred[(payload.stream_id, int(track_id))] = label
                self.arrival_by_instance.setdefault(
                    payload.stream_id, deque(maxlen=300)).append(time.time())
                self._forward(payload)
            elif kind == "status":
                previous = self.stream_status.get(payload.id)
                if payload.state == "connected" and (
                    previous is None or previous.state != "connected"
                ):
                    self.analysis_history.reset_stream(payload.id)
                    for key, dialog in self.analysis_windows.items():
                        if key[0] == payload.id:
                            dialog.reset_history()
                self.stream_status[payload.id] = payload

    def _drain_results(self):
        while True:
            try:
                result = self.result_q.get_nowait()
            except _q.Empty:
                break
            self.inference_by_instance[result.stream_id] = result.status
            for track_id, value in result.labels.items():
                self.inferred[(result.stream_id, track_id)] = value

    def _forward(self, frame):
        try:
            self.inference_q.put_nowait(frame)
        except Exception:
            try:
                self.inference_q.get_nowait()
                self.inference_q.put_nowait(frame)
            except Exception:
                pass

    def _available_instances(self) -> list:
        return [
            item for item in self.cfg.get("instances", [])
            if item.get("enabled", True)
        ]

    def _find_instance(self, instance_id):
        for item in self.cfg.get("instances", []):
            if item["id"] == instance_id:
                return item
        return None

    def _refresh_instance_selector(self):
        available = self._available_instances()
        available_ids = [item["id"] for item in available]
        previous = [self.instance_combo.itemData(i) for i in range(self.instance_combo.count())]
        if previous == available_ids and self.current_instance_id in available_ids:
            return

        preferred = self.current_instance_id or self.cfg.get("ui", {}).get("selected_instance_id", "")
        if preferred not in available_ids:
            preferred = available_ids[0] if available_ids else ""

        self.instance_combo.blockSignals(True)
        self.instance_combo.clear()
        for item in available:
            self.instance_combo.addItem(item["name"], item["id"])
        if preferred:
            self.instance_combo.setCurrentIndex(available_ids.index(preferred))
        self.instance_combo.blockSignals(False)

        if self.current_instance_id != preferred:
            self.current_instance_id = preferred
            self.cfg.setdefault("ui", {})["selected_instance_id"] = preferred
            self.inferred = {
                key: value for key, value in self.inferred.items()
                if key[0] == preferred
            }
            self.map_view.begin_stream(preferred)
            self._apply_preview()

    def _on_instance_selected(self, _index):
        instance_id = self.instance_combo.currentData()
        if not instance_id or instance_id == self.current_instance_id:
            return
        self.current_instance_id = instance_id
        self.cfg.setdefault("ui", {})["selected_instance_id"] = instance_id
        self.inferred = {
            key: value for key, value in self.inferred.items()
            if key[0] == instance_id
        }
        self.map_view.begin_stream(instance_id)
        self._apply_preview()
        frame = self.latest_frames.get(instance_id)
        self.map_view.update_targets(
            self._current_observations(),
            self.inferred,
            frame_id=frame.frame_id if frame else None,
            timestamp=frame.timestamp if frame else None,
        )
        self._refresh_target_table()

    def _current_observations(self):
        frame = self.latest_frames.get(self.current_instance_id)
        if not frame:
            return []
        return sorted(frame.observations, key=lambda obs: obs.track_id)

    def _metrics(self):
        now = time.time()
        instance_id = self.current_instance_id
        arrivals = self.arrival_by_instance.setdefault(
            instance_id, deque(maxlen=300))
        while arrivals and now - arrivals[0] > 1.0:
            arrivals.popleft()
        instant_fps = float(len(arrivals))

        previous = self.fps_smoothed.get(instance_id)
        last_metric_at = self.last_fps_at_by_instance.get(instance_id, 0.0)

        # 瞬时帧数为 0 且没有历史值时，平滑帧率直接初始化为 0。
        # 避免 dt=0 时出现除零或异常初始化。
        if previous is None and instant_fps <= 0.0:
            smoothed_fps = 0.0
        elif previous is None or last_metric_at <= 0.0:
            smoothed_fps = instant_fps
            self.fps_smoothed[instance_id] = smoothed_fps
            self.last_fps_at_by_instance[instance_id] = now
        else:
            dt = max(0.0, min(now - last_metric_at, 1.0))
            alpha = 1.0 - math.exp(-dt / 0.8)
            smoothed_fps = previous + alpha * (instant_fps - previous)
            self.fps_smoothed[instance_id] = smoothed_fps
            self.last_fps_at_by_instance[instance_id] = now

        online = sum(
            1 for status in self.stream_status.values()
            if status.state == "connected")
        instance = self._find_instance(instance_id)
        model_cfg = (instance or {}).get("model", {})
        model_name = _local_onnx_name(model_cfg.get("onnx_path", ""))
        self.value_online.setText(str(online))
        self.value_tracks.setText(str(self.map_view.track_count))
        self.value_fps.setText(f"{smoothed_fps:.1f}")
        model_width = self.value_model.width() or max(self.card_model.width() - 20, 80)
        self.value_model.setText(QFontMetrics(self.value_model.font()).elidedText(
            model_name, Qt.TextElideMode.ElideMiddle, model_width))
        self.value_model.setToolTip(model_name)

        status = self.stream_status.get(self.current_instance_id)
        if not self.current_instance_id:
            self.top_led.set_color(C.amber)
            self._set_top_state(t("no_instance"))
        elif status and status.state == "connected":
            self.top_led.set_color(C.green)
            self._set_top_state(f"{instance['name']} \u5728\u7ebf")
        elif status and status.state == "error":
            self.top_led.set_color(C.red)
            self._set_top_state(status.detail)
        else:
            self.top_led.set_color(C.amber)
            self._set_top_state(status.detail if status else t("waiting_data"))

        status_text = self.inference_by_instance.get(
            self.current_instance_id,
            "\u672a\u9009\u62e9 ONNX\uff0c\u672a\u505a\u76ee\u6807\u5206\u7c7b")
        status_width = self.sb_status.width() or max(self.width() - 260, 320)
        self.sb_status.setText(QFontMetrics(self.sb_status.font()).elidedText(
            status_text, Qt.TextElideMode.ElideRight, status_width))
        self.sb_status.setToolTip(status_text)

    def _set_top_state(self, text: str):
        # Elide to the label's real geometry. Using a larger synthetic width
        # lets QLabel paint into the neighboring settings button.
        width = max(int(self.top_state.contentsRect().width()), 1)
        self.top_state.setText(QFontMetrics(self.top_state.font()).elidedText(
            text, Qt.TextElideMode.ElideRight, width))
        self.top_state.setToolTip(text)

    def _apply_preview(self):
        instance = self._find_instance(self.current_instance_id)
        if not instance:
            self.map_view.configure({}, 3000)
            return
        range_m = instance.get("display", {}).get("plane_range_m", 3000)
        self.map_view.configure(instance.get("map", {}), range_m)

    def _refresh_target_table(self):
        targets = self.map_view.display_targets()
        signature = tuple(
            (key, obs.range_m, obs.ground_range_m, obs.speed_mps,
             self.map_view.inferred.get(key))
            for key, obs in targets
        )
        if signature == self._target_table_signature:
            return
        self._target_table_signature = signature

        selected = self.map_view.selected_track
        self.targets_table.blockSignals(True)
        self.targets_table.setRowCount(len(targets))
        selected_row = -1
        for row, (key, obs) in enumerate(targets):
            inference = self.map_view.inferred.get(key)
            if inference:
                label, confidence = inference
                color = QColor(category_color(label))
                confidence_text = f"{float(confidence) * 100:.0f}%"
            else:
                label = t("uncategorized")
                confidence_text = "--"
                color = QColor(C.text_dim)

            distance = float(obs.ground_range_m)
            distance_text = (
                f"{distance / 1000:.1f}k" if distance >= 1000 else f"{distance:.0f}m")
            values = (f"#{obs.track_id}", label, confidence_text, distance_text)
            for column, value in enumerate(values):
                available_width = max(self.targets_table.columnWidth(column) - 8, 16)
                display_value = QFontMetrics(self.targets_table.font()).elidedText(
                    value, Qt.TextElideMode.ElideRight, available_width)
                item = QTableWidgetItem(display_value)
                item.setForeground(color if column == 1 else QColor(C.text))
                if column in (0, 2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if inference and column == 1:
                    item.setToolTip(f"{label} | {float(confidence) * 100:.1f}%")
                elif column == 3:
                    item.setToolTip(f"{obs.ground_range_m:.1f} m")
                else:
                    item.setToolTip(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, key)
                self.targets_table.setItem(row, column, item)
            self.targets_table.setRowHeight(row, 40)
            if key == selected:
                selected_row = row

        if selected_row >= 0:
            self.targets_table.selectRow(selected_row)
        else:
            self.targets_table.clearSelection()
        self.targets_table.blockSignals(False)
        self.target_count.setText(f"{len(targets):02d} / 10")

    def _target_selected(self):
        row = self.targets_table.currentRow()
        if row < 0:
            self.map_view.select_track(None)
            return
        item = self.targets_table.item(row, 0)
        if item is not None:
            self.map_view.select_track(item.data(Qt.ItemDataRole.UserRole))

    def open_track_analysis(self, stream_id: str, track_id: int):
        key = (str(stream_id), int(track_id))
        dialog = self.analysis_windows.get(key)
        if dialog is None:
            frame = self.latest_frames.get(key[0])
            instance = self._find_instance(key[0]) or {}
            name = frame.stream_name if frame is not None else instance.get("name", key[0])
            dialog = TrackAnalysisDialog(key[0], key[1], name, self)
            dialog.append_samples(self.analysis_history.samples_for(*key))
            self.analysis_windows[key] = dialog
            dialog.finished.connect(
                lambda _result, key=key: self.analysis_windows.pop(key, None))
        if dialog.isMinimized():
            dialog.showNormal()
        else:
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_applied(self, cfg):
        configured = {
            item["id"] for item in cfg.get("instances", [])
            if item.get("enabled", True)
        }
        for stream_id in self.analysis_history.stream_ids:
            if stream_id not in configured:
                self.analysis_history.reset_stream(stream_id)
        for key, dialog in list(self.analysis_windows.items()):
            if key[0] not in configured:
                dialog.close()
        for instance_id in list(self.latest_frames):
            if instance_id not in configured:
                self.latest_frames.pop(instance_id, None)
        for instance_id in list(self.stream_status):
            if instance_id not in configured:
                self.stream_status.pop(instance_id, None)
        for store in (
            self.arrival_by_instance,
            self.fps_smoothed,
            self.last_fps_at_by_instance,
        ):
            for instance_id in list(store):
                if instance_id not in configured:
                    store.pop(instance_id, None)
        self._refresh_instance_selector()
        self._target_table_signature = None
        self._apply_preview()

    def open_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = BasicSettingsDialog(
                self.cfg, self.receiver_cmd, self.inference_cmd, self)
            self.settings_dialog.applied.connect(self._on_applied)
        self.settings_dialog.load_from_cfg()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _tick_clock(self):
        self.sb_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def closeEvent(self, event):
        log.info("UI closed")
        self.timer.stop()
        self.clock.stop()
        self.stop_event.set()
        for dialog in list(self.analysis_windows.values()):
            dialog.close()
        self.analysis_history.clear()
        event.accept()
