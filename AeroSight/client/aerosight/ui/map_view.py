# -*- coding: utf-8 -*-
"""Radar-centred preview with independently controlled display layers."""
from __future__ import annotations

import math
import os
from collections import OrderedDict, deque
from dataclasses import astuple

from PySide6.QtCore import QLineF, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap,
    QPainterPath, QTransform,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from aerosight.ui.theme import C, category_color, ui_font_family

DEFAULT_CAL = {
    "radar_x": 975.0,
    "radar_y": 700.0,
    "pixels_per_100m": 20.0,
    "heading_deg": 0.0,
}
DEFAULT_PLANE_RANGE_M = 3000.0
EMPTY_BACKGROUND = "#111923"
MAX_TRAJECTORIES = 10
MAX_TRAJECTORY_POINTS = 29
FIXED_RING_VALUES_M = (500.0, 1000.0, 2000.0, 3000.0)
RADAR_BLUE_GRAY = (78, 121, 142)
# Keep the range labels unchanged while tuning only the radar-layer spacing.
RADAR_LAYER_SCALE = 1.5 * 0.95
RADAR_SECTOR_WIDTH_DEG = 60.0
RADAR_SECTOR_HALF_WIDTH_DEG = RADAR_SECTOR_WIDTH_DEG / 2.0
RADAR_SECTOR_CENTER_DEG = 0.0


class RadarTagItem(QGraphicsItem):
    """Compact fixed-screen label with a dark, high-contrast background."""

    def __init__(
        self,
        lines: tuple[str, ...],
        color: QColor,
        parent=None,
        detail_lines: tuple[str, ...] | None = None,
    ):
        super().__init__(parent)
        self._color = QColor(color)
        self.font = QFont(ui_font_family(), 8)
        self._compact_lines = tuple(lines)
        self._detail_lines = tuple(detail_lines or lines)
        self._expanded = False
        self._lines: tuple[str, ...] = ()
        self._rect = QRectF()
        self.setAcceptHoverEvents(True)
        self._rebuild_geometry()

    def _rebuild_geometry(self):
        metrics = QFontMetrics(self.font)
        source_lines = self._detail_lines if self._expanded else self._compact_lines
        max_width = 320 if self._expanded else 220
        self._lines = tuple(
            metrics.elidedText(line, Qt.TextElideMode.ElideRight, max_width)
            for line in source_lines
        )
        text_width = max(
            (metrics.horizontalAdvance(line) for line in self._lines),
            default=0,
        )
        text_height = metrics.height() * len(self._lines) + 2
        self._rect = QRectF(
            0.0, 0.0, float(text_width + 28), float(text_height + 8))

    def set_expanded(self, expanded: bool):
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self.prepareGeometryChange()
        self._expanded = expanded
        self._rebuild_geometry()
        self.update()
        if self.scene() is not None:
            self.scene().update()

    @property
    def expanded(self) -> bool:
        return self._expanded

    def boundingRect(self) -> QRectF:
        return self._rect

    def toPlainText(self) -> str:
        return "\n".join(self._lines)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font)
        background = QColor(9, 15, 28, 205)
        painter.setPen(QPen(background, 0))
        painter.setBrush(background)
        painter.drawRoundedRect(self._rect, 3.0, 3.0)

        painter.setPen(QPen(self._color, 0))
        painter.setBrush(self._color)
        metrics = QFontMetrics(self.font)
        painter.drawEllipse(7, int(6 + metrics.height() / 2 - 3), 6, 6)

        painter.setPen(QColor(247, 250, 255))
        metrics = QFontMetrics(self.font)
        y = 6
        for line in self._lines:
            painter.drawText(19, y + metrics.ascent(), line)
            y += metrics.height()
        painter.restore()

    def hoverEnterEvent(self, event):
        self.set_expanded(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.set_expanded(False)
        super().hoverLeaveEvent(event)


def thin_pen(color, alpha: int | None = None) -> QPen:
    """Return a fixed, hairline-width screen pen."""
    qcolor = QColor(color)
    if alpha is not None:
        qcolor.setAlpha(alpha)
    pen = QPen(qcolor)
    pen.setWidth(0)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


class MapRadarView(QGraphicsView):
    """Radar-centred view with an independently transformed map layer."""

    # Calibration clicks report original image pixels, not scene coordinates.
    mapClicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setInteractive(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # Dynamic labels ignore the view transform. Full viewport updates keep
        # their old bounding regions from leaving stale pixels behind when a
        # target moves or a label is replaced.
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.image_path = ""
        self.cal = dict(DEFAULT_CAL)
        self.plane_range_m = DEFAULT_PLANE_RANGE_M
        self.observations = []
        self.inferred: dict[tuple[str, int], tuple[str, float]] = {}
        self.trajectories: OrderedDict[
            tuple[str, int], deque
        ] = OrderedDict()
        self.latest_by_track: dict[tuple[str, int], object] = {}
        self._current_stream_id = ""
        self._last_frame_token = None
        self.selected_track: tuple[str, int] | None = None
        self.trails_visible = True
        self.labels_visible = True
        self.map_visible = True
        self._zoom = 1.0
        self._arm_position = False
        self._map_item: QGraphicsPixmapItem | None = None
        self._map_cover_scale = 1.0
        self._dynamic_items = []
        self._static_text_items = []
        self._scan_sector_item: QGraphicsPathItem | None = None
        self._target_signature = None
        self._build_layers()

    # ------------------------------------------------------------ public
    def configure(self, map_cfg=None, plane_range_m=DEFAULT_PLANE_RANGE_M):
        self.image_path = str((map_cfg or {}).get("image_path", ""))
        self.cal = dict(DEFAULT_CAL)
        self.cal.update(map_cfg or {})
        self.plane_range_m = max(float(plane_range_m), 1.0)
        self._build_layers()

    def set_background(self, path: str):
        self.image_path = path or ""
        self._build_layers()

    def set_calibration(self, cal=None):
        self.cal = dict(DEFAULT_CAL)
        self.cal.update(cal or {})
        self._build_layers()

    def begin_stream(self, stream_id: str):
        stream_id = str(stream_id or "")
        if stream_id == self._current_stream_id:
            return
        self.clear_trajectories()
        self._current_stream_id = stream_id

    def clear_trajectories(self):
        self.trajectories.clear()
        self.latest_by_track.clear()
        self.inferred.clear()
        self.observations = []
        self.selected_track = None
        self._last_frame_token = None
        self._target_signature = None
        self._draw_targets()

    def record_frame(
        self,
        observations,
        *,
        frame_id=None,
        timestamp=None,
    ):
        observations = list(observations or [])
        if observations:
            stream_id = str(observations[0].stream_id)
            self.begin_stream(stream_id)
        if frame_id is None:
            frame_token = tuple(tuple(astuple(obs)) for obs in observations)
        else:
            frame_token = (self._current_stream_id, int(frame_id), timestamp)

        if frame_token == self._last_frame_token:
            return frame_token

        self._last_frame_token = frame_token
        self.observations = observations
        current_keys = {
            (str(obs.stream_id), int(obs.track_id))
            for obs in observations
        }
        for obs in observations:
            key = (str(obs.stream_id), int(obs.track_id))
            history = self.trajectories.get(key)
            if history is None:
                if len(self.trajectories) >= MAX_TRAJECTORIES:
                    # Keep the visible target slots stable while a scan frame
                    # carries more targets than the UI can display. A target
                    # that disappeared can be replaced; active targets keep
                    # their full history across subsequent scan batches.
                    stale_key = next(
                        (
                            candidate
                            for candidate in self.trajectories
                            if candidate not in current_keys
                        ),
                        None,
                    )
                    if stale_key is None:
                        continue
                    self.trajectories.pop(stale_key, None)
                    self.latest_by_track.pop(stale_key, None)
                    self.inferred.pop(stale_key, None)
                    if self.selected_track == stale_key:
                        self.selected_track = None
                history = deque(maxlen=MAX_TRAJECTORY_POINTS)
                self.trajectories[key] = history
            history.append(obs)
            self.latest_by_track[key] = obs
            self.trajectories.move_to_end(key)

        while len(self.trajectories) > MAX_TRAJECTORIES:
            removed, _removed_history = self.trajectories.popitem(last=False)
            self.latest_by_track.pop(removed, None)
            self.inferred.pop(removed, None)
            if self.selected_track == removed:
                self.selected_track = None
        return frame_token

    def update_targets(
        self,
        observations,
        inferred=None,
        *,
        frame_id=None,
        timestamp=None,
    ):
        observations = list(observations or [])
        frame_token = self.record_frame(
            observations,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        for key, value in (inferred or {}).items():
            if key in self.trajectories:
                self.inferred[key] = value

        self.observations = observations
        signature = (
            frame_token,
            tuple(tuple(astuple(obs)) for obs in observations),
            tuple(sorted(self.inferred.items())),
            self.trails_visible,
            self.labels_visible,
            self.selected_track,
        )
        if signature == self._target_signature:
            return
        self._target_signature = signature
        self._draw_targets()

    @property
    def track_count(self) -> int:
        return len(self.trajectories)

    def display_targets(self):
        return [
            (key, self.latest_by_track[key])
            for key in reversed(self.trajectories)
            if key in self.latest_by_track
        ]

    def select_track(self, key):
        self.selected_track = key if key in self.latest_by_track else None
        self._target_signature = None
        self._draw_targets()

    def set_trails_visible(self, visible: bool):
        self.trails_visible = bool(visible)
        self._target_signature = None
        self._draw_targets()

    def set_labels_visible(self, visible: bool):
        self.labels_visible = bool(visible)
        self._target_signature = None
        self._draw_targets()

    def set_map_visible(self, visible: bool):
        """Toggle the local map/radar background without touching targets."""
        visible = bool(visible)
        if visible == self.map_visible:
            return
        self.map_visible = visible
        self._build_layers()

    def arm_radar_position(self):
        self._arm_position = True

    def center_radar(self):
        self._zoom = 1.0
        self._update_view_transform()
        self.centerOn(0.0, 0.0)
        self._target_signature = None
        self._draw_targets()

    def zoom_by(self, factor: float):
        viewport_center = self.viewport().rect().center()
        scene_center = self.mapToScene(viewport_center)
        self._zoom = min(max(self._zoom * float(factor), 0.55), 3.0)
        self._update_view_transform()
        self.centerOn(scene_center)
        self._target_signature = None
        self._draw_targets()

    # Backward-compatible aliases.
    fit_full = center_radar
    reset_view = center_radar

    def showEvent(self, event):
        super().showEvent(event)
        self._build_layers()

    def closeEvent(self, event):
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._build_layers()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.15 if delta > 0 else 1.0 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if (
            self._arm_position
            and event.button() == Qt.MouseButton.LeftButton
            and self._map_item is not None
        ):
            image_point = self._event_to_image_point(event)
            if image_point is not None:
                self._arm_position = False
                self.mapClicked.emit(image_point)
                event.accept()
                return
        super().mousePressEvent(event)

    # ---------------------------------------------------------- layers
    def _build_layers(self):
        scene = self.scene()
        scene.clear()
        self._dynamic_items = []
        self._target_signature = None
        self._map_item = None
        self._scan_sector_item = None
        self._map_cover_scale = 1.0
        self._static_text_items = []
        scene.setBackgroundBrush(QColor(EMPTY_BACKGROUND))

        map_rect = self._add_map_layer(scene)
        if self.map_visible:
            self._draw_radar_layer(scene)

        radius = self.plane_range_m * RADAR_LAYER_SCALE
        radar_rect = QRectF(-radius, -radius, 2 * radius, 2 * radius)
        # Include enough scene area around the origin to fill the whole
        # viewport; otherwise Qt cannot keep the radar origin centred.
        viewport_rect = self._viewport_scene_rect()
        scene.setSceneRect(
            map_rect.united(radar_rect).united(viewport_rect)
            .adjusted(-50, -50, 50, 50))
        self._update_view_transform()
        self.centerOn(0.0, 0.0)
        self._draw_targets()

    def _add_map_layer(self, scene: QGraphicsScene) -> QRectF:
        if (
            not self.map_visible
            or not self.image_path
            or not os.path.exists(self.image_path)
        ):
            return QRectF()

        pixmap = QPixmap(self.image_path)
        if pixmap.isNull():
            return QRectF()

        radar_x = float(self.cal.get("radar_x", 0.0))
        radar_y = float(self.cal.get("radar_y", 0.0))
        pixels_per_100m = max(float(self.cal.get("pixels_per_100m", 20.0)), 1.0)
        heading = float(self.cal.get("heading_deg", 0.0))

        item = QGraphicsPixmapItem(pixmap)
        item.setZValue(-20)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

        # The selected calibration point maps to the radar origin. Image pixels
        # are converted to metres, and the map layer - not the radar layer - is
        # rotated opposite to radar heading.
        scale = 100.0 / pixels_per_100m
        transform = QTransform()
        transform.rotate(-heading)
        transform.scale(scale, scale)
        transform.translate(-radar_x, -radar_y)
        item.setTransform(transform)

        # The map is a background layer, so its finite image bounds must cover
        # the initial viewport. Keep the calibrated radar point at scene origin
        # while applying only the minimum additional scale required to remove
        # exposed scene background around the image.
        base_rect = item.sceneBoundingRect()
        viewport_rect = self._viewport_scene_rect()
        cover_scale = self._required_map_cover_scale(base_rect, viewport_rect)
        if cover_scale > 1.0:
            covered_transform = QTransform()
            covered_transform.rotate(-heading)
            covered_transform.scale(scale * cover_scale, scale * cover_scale)
            covered_transform.translate(-radar_x, -radar_y)
            item.setTransform(covered_transform)
        self._map_cover_scale = cover_scale
        scene.addItem(item)
        self._map_item = item
        return item.sceneBoundingRect()

    @staticmethod
    def _required_map_cover_scale(
        map_rect: QRectF, viewport_rect: QRectF
    ) -> float:
        """Return the minimum scale that makes a map rect cover a viewport."""
        required = 1.0
        for map_start, map_end, view_start, view_end in (
            (map_rect.left(), map_rect.right(), viewport_rect.left(), viewport_rect.right()),
            (map_rect.top(), map_rect.bottom(), viewport_rect.top(), viewport_rect.bottom()),
        ):
            if map_start < 0.0 and view_start < map_start:
                required = max(required, view_start / map_start)
            if map_end > 0.0 and view_end > map_end:
                required = max(required, view_end / map_end)
        return max(required, 1.0)

    def _draw_radar_layer(self, scene: QGraphicsScene):
        radius = self.plane_range_m * RADAR_LAYER_SCALE

        coverage_rect = QRectF(-radius, -radius, 2 * radius, 2 * radius)
        disk = QGraphicsEllipseItem(coverage_rect)
        coverage_color = QColor(*RADAR_BLUE_GRAY, 235)
        coverage_fill = QColor(*RADAR_BLUE_GRAY, 18)
        ring_color = QColor(*RADAR_BLUE_GRAY, 230)
        label_color = QColor(*RADAR_BLUE_GRAY, 245)

        coverage_pen = QPen(coverage_color)
        coverage_pen.setWidthF(1.4)
        coverage_pen.setCosmetic(True)
        disk.setPen(coverage_pen)
        disk.setBrush(QBrush(coverage_fill))
        disk.setZValue(1)
        scene.addItem(disk)

        sector = QGraphicsPathItem()
        sector.setBrush(QBrush(QColor(*RADAR_BLUE_GRAY, 76)))
        sector.setPen(Qt.PenStyle.NoPen)
        sector.setZValue(1.4)
        sector.setData(0, "radar-fixed-sector")
        scene.addItem(sector)
        self._scan_sector_item = sector
        sector.setPath(
            self._fixed_sector_path(
                self.plane_range_m * RADAR_LAYER_SCALE))

        self._draw_radar_crosshair(scene, radius)

        for value in self._ring_values():
            self._draw_range_ring(scene, value, ring_color, label_color)

    @staticmethod
    def _fixed_sector_path(radius: float) -> QPainterPath:
        path = QPainterPath(QPointF(0.0, 0.0))
        steps = 12
        start = RADAR_SECTOR_CENTER_DEG - RADAR_SECTOR_HALF_WIDTH_DEG
        for index in range(steps + 1):
            bearing = start + RADAR_SECTOR_WIDTH_DEG * index / steps
            angle = math.radians(bearing)
            path.lineTo(
                QPointF(radius * math.sin(angle), -radius * math.cos(angle)))
        path.closeSubpath()
        return path

    def _draw_radar_crosshair(self, scene: QGraphicsScene, radius: float):
        """Draw balanced cardinal and shorter diagonal radar reference lines."""
        origin = QPointF(0.0, 0.0)
        light_blue_gray = (112, 148, 165)
        cardinal_color = QColor(*light_blue_gray, 168)
        diagonal_color = QColor(*light_blue_gray, 132)

        self._add_radar_line(
            scene,
            QLineF(-radius, 0.0, radius, 0.0),
            cardinal_color,
            width=1.15,
        )
        self._add_radar_line(
            scene,
            QLineF(0.0, -radius, 0.0, radius),
            cardinal_color,
            width=1.15,
        )

        diagonal_radius = radius * 0.72
        diagonal_offset = diagonal_radius / math.sqrt(2.0)
        self._add_radar_line(
            scene,
            QLineF(-diagonal_offset, -diagonal_offset,
                   diagonal_offset, diagonal_offset),
            diagonal_color,
            width=1.0,
        )
        self._add_radar_line(
            scene,
            QLineF(-diagonal_offset, diagonal_offset,
                   diagonal_offset, -diagonal_offset),
            diagonal_color,
            width=1.0,
        )

    @staticmethod
    def _add_radar_line(
        scene: QGraphicsScene,
        line: QLineF,
        color: QColor,
        *,
        width: float,
    ):
        item = QGraphicsLineItem(line)
        pen = QPen(color)
        pen.setWidthF(width)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        item.setPen(pen)
        item.setZValue(2.1)
        item.setData(0, "radar-crosshair")
        scene.addItem(item)

    def _view_radius_px(self) -> float:
        width = max(self.viewport().width(), 1)
        height = max(self.viewport().height(), 1)
        return max(min(width, height) / 2.0 - 46.0, 20.0)

    def _screen_pixels_per_meter(self) -> float:
        return self._view_radius_px() / self.plane_range_m

    def _ring_values(self):
        return list(FIXED_RING_VALUES_M)

    def _draw_range_ring(
        self,
        scene: QGraphicsScene,
        value: float,
        ring_color: QColor | None = None,
        label_color: QColor | None = None,
    ):
        display_radius = float(value) * RADAR_LAYER_SCALE
        rect = QRectF(
            -display_radius, -display_radius,
            2 * display_radius, 2 * display_radius)
        ring = QGraphicsEllipseItem(rect)
        pen = QPen(ring_color or QColor(197, 213, 224, 105))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        ring.setPen(pen)
        ring.setBrush(Qt.BrushStyle.NoBrush)
        ring.setZValue(2)
        scene.addItem(ring)

        label = QGraphicsTextItem(f"{value / 1000.0:g}km")
        label.setDefaultTextColor(label_color or QColor(188, 204, 216))
        label.setFont(QFont(ui_font_family(), 8))
        label.setZValue(3)
        label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label.setPos(8.0, -display_radius - 8.0)
        scene.addItem(label)
        self._static_text_items.append(label)

    # --------------------------------------------------------- targets
    def _draw_targets(self):
        scene = self.scene()
        for item in self._dynamic_items:
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
        self._dynamic_items = []

        occupied_labels: list[QRectF] = []
        for item in self._static_text_items:
            bounds = item.mapRectToScene(item.boundingRect())
            first = self.mapFromScene(bounds.topLeft())
            last = self.mapFromScene(bounds.bottomRight())
            occupied_labels.append(QRectF(
                min(first.x(), last.x()),
                min(first.y(), last.y()),
                abs(last.x() - first.x()),
                abs(last.y() - first.y()),
            ).adjusted(-3.0, -2.0, 3.0, 2.0))
        for key, obs in self.display_targets():
            pos = self._target_position(obs)
            is_selected = key == self.selected_track
            inferred = self.inferred.get(key)
            if inferred:
                label_text, score = inferred
                color = QColor(category_color(label_text))
                header = f"#{obs.track_id}  {label_text}  {score * 100:.0f}%"
            else:
                color = QColor(C.cyan)
                header = f"#{obs.track_id}  \u672a\u5206\u7c7b"

            history = self.trajectories.get(key, ())
            if self.trails_visible:
                visible_history = list(history)[:-1]
                point_count = len(visible_history)
                for index, point_obs in enumerate(visible_history):
                    point_color = QColor(color)
                    point_color.setAlpha(90 + int(105 * (index + 1) / max(point_count, 1)))
                    radius = 2.7 if is_selected else 2.1
                    dot = QGraphicsEllipseItem(
                        QRectF(-radius, -radius, radius * 2, radius * 2))
                    dot.setPos(self._target_position(point_obs))
                    dot_pen = QPen(QColor(10, 17, 25, 220), 0.8)
                    dot_pen.setCosmetic(True)
                    dot.setPen(dot_pen)
                    dot.setBrush(point_color)
                    dot.setZValue(5 if not is_selected else 6)
                    dot.setFlag(dot.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    dot.setToolTip(self._tooltip(point_obs))
                    scene.addItem(dot)
                    self._dynamic_items.append(dot)

            marker_radius = 6.0 if is_selected else 4.8
            marker = QGraphicsEllipseItem(
                QRectF(-marker_radius, -marker_radius, marker_radius * 2, marker_radius * 2))
            marker.setPos(pos)
            marker.setPen(QPen(QColor("#F8FAFC"), 1.4))
            marker.setBrush(color)
            marker.setZValue(7)
            marker.setToolTip(self._tooltip(obs))
            marker.setFlag(marker.GraphicsItemFlag.ItemIgnoresTransformations, True)
            scene.addItem(marker)
            self._dynamic_items.append(marker)

            if self.labels_visible:
                tag = RadarTagItem(
                    (header, f"{obs.ground_range_m:.0f}m  {obs.speed_mps:.1f}m/s"),
                    color,
                    detail_lines=tuple(self._tooltip(obs).splitlines()),
                )
                tag.setZValue(8 if is_selected else 7)
                tag.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                tag.setToolTip(self._tooltip(obs))
                if self._place_tag(tag, pos, occupied_labels):
                    scene.addItem(tag)
                    self._dynamic_items.append(tag)

        scene.update()
        self.viewport().update()

    def _target_position(self, obs) -> QPointF:
        """Map a target to screen-oriented scene coordinates in metres."""
        plane_range = float(getattr(obs, "ground_range_m", 0.0) or 0.0)
        if plane_range <= 0.0:
            plane_range = math.hypot(
                float(getattr(obs, "x_m", 0.0) or 0.0),
                float(getattr(obs, "y_m", 0.0) or 0.0))
        if plane_range <= 0.0 and float(getattr(obs, "range_m", 0.0) or 0.0) > 0.0:
            plane_range = (
                float(obs.range_m)
                * math.cos(math.radians(float(getattr(obs, "elevation_deg", 0.0) or 0.0))))

        return self._target_position_for_range(plane_range, float(
            getattr(obs, "azimuth_deg", 0.0) or 0.0))

    @staticmethod
    def _target_position_for_range(plane_range: float, azimuth_deg: float) -> QPointF:
        azimuth = math.radians(float(azimuth_deg))
        # Positive azimuth is clockwise/right; scene y grows downward, hence -y.
        return QPointF(
            plane_range * math.sin(azimuth),
            -plane_range * math.cos(azimuth))

    def _place_tag(self, tag, marker_pos: QPointF, occupied: list[QRectF]) -> bool:
        """Place a fixed-pixel label without clipping or colliding with another tag."""
        width = float(tag.boundingRect().width())
        height = float(tag.boundingRect().height())
        view_width = max(int(self.viewport().width()), 1)
        view_height = max(int(self.viewport().height()), 1)
        margin = 6.0
        gap = 10.0
        marker = self.mapFromScene(marker_pos)
        center_x = float(marker.x())
        center_y = float(marker.y())
        candidates = (
            (center_x + gap, center_y - height / 2),
            (center_x - width - gap, center_y - height / 2),
            (center_x - width / 2, center_y - height - gap),
            (center_x - width / 2, center_y + gap),
            (center_x + gap, center_y - height - gap),
            (center_x - width - gap, center_y - height - gap),
            (center_x + gap, center_y + gap),
            (center_x - width - gap, center_y + gap),
        )
        for x, y in candidates:
            rect = QRectF(x, y, width, height)
            if (
                rect.left() < margin
                or rect.top() < margin
                or rect.right() > view_width - margin
                or rect.bottom() > view_height - margin
            ):
                continue
            padded = rect.adjusted(-3.0, -2.0, 3.0, 2.0)
            if any(padded.intersects(other) for other in occupied):
                continue
            occupied.append(padded)
            tag.setPos(self.mapToScene(QPoint(int(round(x)), int(round(y)))))
            return True
        return False

    @staticmethod
    def _tooltip(obs):
        return (
            f"\u822a\u8ff9 #{obs.track_id}\n"
            f"\u6765\u6e90\uff1a{obs.stream_id}\n"
            f"\u5e73\u9762\u8ddd\u79bb\uff08\u5730\u56fe\u8ddd\u79bb\uff09\uff1a{obs.ground_range_m:.1f} m\n"
            f"\u659c\u8ddd\uff08\u96f7\u8fbe\u76f4\u7ebf\u8ddd\u79bb\uff09\uff1a{obs.range_m:.1f} m\n"
            f"\u65b9\u4f4d\u89d2\uff1a{obs.azimuth_deg:.1f}\u00b0\n"
            f"\u4fef\u4ef0\u89d2\uff1a{obs.elevation_deg:.1f}\u00b0\n"
            f"\u9ad8\u5ea6\uff1a{obs.altitude_m:.1f} m\n"
            f"\u901f\u5ea6\uff1a{obs.speed_mps:.1f} m/s\n"
            f"\u5f84\u5411\u901f\u5ea6\uff1a{obs.radial_speed_mps:.1f} m/s\n"
            f"\u822a\u5411\u89d2\uff1a{obs.course_deg:.1f}\u00b0\n"
            f"\u72b6\u6001\uff1a{obs.state or '-'}")

    # --------------------------------------------------------- helpers
    def _update_view_transform(self):
        pixels_per_meter = self._screen_pixels_per_meter() * self._zoom
        self.setTransform(QTransform.fromScale(pixels_per_meter, pixels_per_meter))

    def _viewport_scene_rect(self) -> QRectF:
        ppm = max(self._screen_pixels_per_meter(), 1e-9)
        half_width = self.viewport().width() / 2.0 / ppm
        half_height = self.viewport().height() / 2.0 / ppm
        return QRectF(
            -half_width, -half_height,
            2.0 * half_width, 2.0 * half_height)

    def _event_to_image_point(self, event) -> QPointF | None:
        if self._map_item is None:
            return None
        scene_point = self.mapToScene(event.pos())
        # No pixmap offset is used; item coordinates are image pixels.
        image_point = self._map_item.mapFromScene(scene_point)
        pixmap = self._map_item.pixmap()
        if (
            -1 <= image_point.x() <= pixmap.width() + 1
            and -1 <= image_point.y() <= pixmap.height() + 1
        ):
            return image_point
        return None
