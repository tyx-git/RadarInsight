# -*- coding: utf-8 -*-
"""Floating live analysis of measurements already delivered by the backend."""
from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QVBoxLayout

from aerosight.core.track_analysis import TrackAnalysisSample
from aerosight.ui.i18n import t
from aerosight.ui.theme import C, ui_font_family


class _MetricCurve:
    """Use separate line segments so missing values leave a real visual gap."""

    def __init__(self, chart: QChart, x_axis: QValueAxis, y_axis: QValueAxis,
                 name: str, color: str, minimum_span: float):
        self.chart = chart
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.name = name
        self.color = QColor(color)
        self.minimum_span = minimum_span
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.segments: list[QLineSeries] = []
        self._active: QLineSeries | None = self._new_segment()

    def _new_segment(self) -> QLineSeries:
        series = QLineSeries()
        series.setName(self.name)
        series.setPen(QPen(self.color, 1.8))
        series.setPointsVisible(False)
        self.chart.addSeries(series)
        series.attachAxis(self.x_axis)
        series.attachAxis(self.y_axis)
        if self.segments:
            for marker in self.chart.legend().markers(series):
                marker.setVisible(False)
        self.segments.append(series)
        return series

    def append(self, seconds: float, value: float | None):
        if value is None:
            self._active = None
            return
        if self._active is None:
            self._active = (
                self.segments[0] if not self.segments[0].count()
                else self._new_segment()
            )
        self._active.append(QPointF(seconds, value))
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def update_range(self):
        if self.minimum is None or self.maximum is None:
            self.y_axis.setRange(0.0, self.minimum_span)
            return
        padding = max((self.maximum - self.minimum) * 0.08, self.minimum_span / 2)
        self.y_axis.setRange(self.minimum - padding, self.maximum + padding)

    def clear(self):
        for series in self.segments[1:]:
            self.chart.removeSeries(series)
            series.deleteLater()
        self.segments = self.segments[:1]
        self.segments[0].clear()
        self._active = self.segments[0]
        self.minimum = None
        self.maximum = None
        self.update_range()


class TrackAnalysisDialog(QDialog):
    def __init__(self, stream_id: str, track_id: int, stream_name: str, parent=None):
        flags = (
            Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        super().__init__(parent, flags)
        self.stream_id = str(stream_id)
        self.track_id = int(track_id)
        self.sample_count = 0
        self._latest_seconds = 0.0
        self.setObjectName("TrackAnalysisDialog")
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(940, 560)
        self.setMinimumSize(680, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel(t("track_analysis_heading").format(track_id=self.track_id))
        title.setObjectName("AnalysisTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        self.count_label = QLabel()
        self.count_label.setObjectName("AnalysisCount")
        heading.addWidget(self.count_label)
        layout.addLayout(heading)

        metrics = QHBoxLayout()
        self.source_label = QLabel()
        self.source_label.setObjectName("AnalysisCount")
        metrics.addWidget(self.source_label)
        metrics.addStretch(1)
        self.speed_label = QLabel()
        self.speed_label.setObjectName("AnalysisSpeed")
        self.altitude_label = QLabel()
        self.altitude_label.setObjectName("AnalysisAltitude")
        self.altitude_label.setToolTip(t("track_analysis_altitude_hint"))
        metrics.addWidget(self.speed_label)
        metrics.addSpacing(20)
        metrics.addWidget(self.altitude_label)
        layout.addLayout(metrics)

        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self.chart.setDropShadowEnabled(False)
        self.chart.setBackgroundBrush(QBrush(QColor(C.panel)))
        self.chart.setBackgroundPen(QPen(QColor(C.border), 1.0))
        self.chart.setBackgroundRoundness(4.0)
        self.chart.setPlotAreaBackgroundBrush(QBrush(QColor(C.panel)))
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.setMargins(QMargins(10, 8, 10, 8))
        self.chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chart.legend().setLabelColor(QColor(C.text_dim))
        self.chart.legend().setFont(QFont(ui_font_family(), 9))

        self.time_axis = self._axis(t("track_analysis_time"), C.text_dim)
        self.speed_axis = self._axis(t("track_analysis_speed") + " (m/s)", C.cyan)
        self.altitude_axis = self._axis(t("track_analysis_altitude") + " (m)", C.blue)
        self.altitude_axis.setGridLineVisible(False)
        self.chart.addAxis(self.time_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.speed_axis, Qt.AlignmentFlag.AlignLeft)
        self.chart.addAxis(self.altitude_axis, Qt.AlignmentFlag.AlignRight)
        self.speed_curve = _MetricCurve(
            self.chart, self.time_axis, self.speed_axis,
            t("track_analysis_speed"), C.cyan, 1.0,
        )
        self.altitude_curve = _MetricCurve(
            self.chart, self.time_axis, self.altitude_axis,
            t("track_analysis_altitude"), C.blue, 2.0,
        )
        self.chart_view = QChartView(self.chart, self)
        self.chart_view.setFrameShape(QFrame.Shape.NoFrame)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view, 1)
        self.set_stream_name(stream_name)
        self.reset_history()

    @staticmethod
    def _axis(title: str, color: str) -> QValueAxis:
        axis = QValueAxis()
        axis.setTitleText(title)
        axis.setTitleFont(QFont(ui_font_family(), 9))
        axis.setTitleBrush(QBrush(QColor(color)))
        axis.setLabelsFont(QFont(ui_font_family(), 9))
        axis.setLabelsColor(QColor(color))
        axis.setLinePen(QPen(QColor(C.border_h), 1.0))
        axis.setGridLinePen(QPen(QColor(C.border), 0.8, Qt.PenStyle.DashLine))
        axis.setMinorGridLineVisible(False)
        axis.setTickCount(6)
        axis.setLabelFormat("%.1f")
        return axis

    def set_stream_name(self, stream_name: str):
        name = str(stream_name or self.stream_id)
        self.source_label.setText(name)
        self.setWindowTitle(t("track_analysis_title").format(
            stream_name=name, track_id=self.track_id,
        ))

    def append_samples(self, samples: Iterable[TrackAnalysisSample]):
        latest = None
        for sample in samples:
            self.speed_curve.append(sample.elapsed_seconds, sample.radial_speed_mps)
            self.altitude_curve.append(sample.elapsed_seconds, sample.altitude_m)
            self._latest_seconds = sample.elapsed_seconds
            self.sample_count += 1
            latest = sample
        if latest is None:
            return
        self.time_axis.setRange(0.0, max(self._latest_seconds, 1.0))
        self.time_axis.setLabelFormat("%.1f" if self._latest_seconds < 10 else "%.0f")
        self.speed_curve.update_range()
        self.altitude_curve.update_range()
        self.count_label.setText(t("track_analysis_count").format(count=self.sample_count))
        self.speed_label.setText(self._metric_text(
            t("track_analysis_speed"), latest.radial_speed_mps, "m/s"))
        self.altitude_label.setText(self._metric_text(
            t("track_analysis_altitude"), latest.altitude_m, "m"))

    @staticmethod
    def _metric_text(name: str, value: float | None, unit: str) -> str:
        text = "--" if value is None else f"{value:.2f}"
        return f"{name}  {text} {unit}"

    def reset_history(self):
        self.speed_curve.clear()
        self.altitude_curve.clear()
        self.sample_count = 0
        self._latest_seconds = 0.0
        self.time_axis.setRange(0.0, 1.0)
        self.time_axis.setLabelFormat("%.1f")
        self.count_label.setText(t("track_analysis_count").format(count=0))
        self.speed_label.setText(self._metric_text(t("track_analysis_speed"), None, "m/s"))
        self.altitude_label.setText(self._metric_text(t("track_analysis_altitude"), None, "m"))
