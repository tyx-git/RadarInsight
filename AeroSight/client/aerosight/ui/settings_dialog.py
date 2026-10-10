# -*- coding: utf-8 -*-
"""AeroSight settings dialog: instance manager architecture."""
from __future__ import annotations

import os
import time
from urllib.parse import unquote, urlparse, urlunparse
from uuid import uuid4

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from aerosight.app_config import  default_instance, normalize_map
from aerosight.ui.i18n import t
from aerosight.ui.map_view import MapRadarView
from aerosight.ui.theme import C, ui_font_family


class BasicSettingsDialog(QDialog):
    applied = Signal(dict)

    def __init__(self, cfg, receiver_cmd, inference_cmd, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.receiver_cmd = receiver_cmd
        self.inference_cmd = inference_cmd
        self._building = True
        self.current_editor_id = ""

        self.setWindowTitle(t("settings"))
        self.resize(1360, 880)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(False)

        self._build_ui()
        self.load_from_cfg()
        self._building = False

    def showEvent(self, event):
        super().showEvent(event)
        total = max(self.width(), 1)
        left = int(total * 0.38)
        self.main_splitter.setSizes([left, total - left])

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._instances_page(), t("instances"))
        self.tabs.addTab(self._general_page(), t("general"))
        root.addWidget(self.tabs, 1)
        self._hide_spin_buttons()

        row = QHBoxLayout()
        self.state_label = QLabel(t("state_modified"))
        self.state_label.setStyleSheet("color:#7A8798;")
        row.addWidget(self.state_label)
        row.addStretch(1)
        apply_btn = QPushButton(t("save_apply"))
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_settings)
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.hide)
        row.addWidget(apply_btn)
        row.addWidget(close_btn)
        root.addLayout(row)

    def _instances_page(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self._manager_panel(self.main_splitter))
        self.main_splitter.addWidget(self._instance_editor_panel(self.main_splitter))
        self.main_splitter.setStretchFactor(0, 38)
        self.main_splitter.setStretchFactor(1, 62)
        self.main_splitter.setSizes([520, 840])
        return self.main_splitter

    def _manager_panel(self, parent):
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(12)

        header = QLabel(t("instance_manage"))
        header.setObjectName("PanelHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setMinimumHeight(30)
        layout.addWidget(header)

        self.instance_table = QTableWidget(0, 4)
        self.instance_table.setHorizontalHeaderLabels(
            ["", t("instance_name"), t("source_kind"), t("rrtsp_url")])
        self.instance_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.instance_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        header_view = self.instance_table.horizontalHeader()
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.instance_table.setColumnWidth(0, 42)
        self.instance_table.verticalHeader().setDefaultSectionSize(38)
        self.instance_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.instance_table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.instance_table, 1)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(8)
        buttons.setVerticalSpacing(8)
        actions = [
            (t("add_radar"), self.add_radar_instance),
            (t("add_simulation"), self.add_simulation_instance),
            (t("paste_import"), self.import_clipboard_url),
            (t("remove_selected"), self.remove_selected),
        ]
        for index, (text, handler) in enumerate(actions):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            buttons.addWidget(btn, index // 2, index % 2)
        layout.addLayout(buttons)
        return panel

    def _instance_editor_panel(self, parent):
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.addStretch(1)
        self.editor_title = QLabel("")
        self.editor_title.setObjectName("InstanceName")
        self.editor_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.editor_title.setMinimumHeight(44)
        title_row.addWidget(self.editor_title, 4)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        separator = QFrame()
        separator.setObjectName("HLine")
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        self.editor_tabs = QTabWidget()
        self.editor_tabs.addTab(self._source_page(), t("source"))
        self.editor_tabs.addTab(self._display_page(), t("display"))
        self.editor_tabs.addTab(self._map_page(), t("map_calibration"))
        self.editor_tabs.addTab(self._onnx_page(), t("onnx"))
        layout.addWidget(self.editor_tabs, 1)
        return panel

    @staticmethod
    def _form_label(text: str) -> QLabel:
        label = QLabel(text)
        font = QFont(ui_font_family(), 10)
        font.setWeight(QFont.Weight.DemiBold)
        label.setFont(font)
        label.setStyleSheet(f"color:{C.text};")
        return label

    def _source_page(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(34, 30, 34, 30)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(16)

        self.source_kind = QComboBox()
        self.source_kind.addItem(t("kind_rrtsp"), "rrtsp")
        self.source_kind.addItem(t("kind_simulation"), "simulation")
        self.source_kind.currentIndexChanged.connect(self._on_form_changed)
        form.addRow(self._form_label(t("source_kind")), self.source_kind)

        self.rrtsp_url = QLineEdit()
        self.rrtsp_username = QLineEdit()
        self.rrtsp_password = QLineEdit()
        self.rrtsp_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self._form_label(t("rrtsp_url")), self.rrtsp_url)
        form.addRow(self._form_label(t("username")), self.rrtsp_username)
        form.addRow(self._form_label(t("password")), self.rrtsp_password)

        self.simulation_rate = QDoubleSpinBox()
        self.simulation_rate.setRange(0.1, 30.0)
        self.simulation_rate.setSuffix(" Hz")
        self.simulation_rate.setDecimals(1)
        form.addRow(self._form_label(t("simulation_rate")), self.simulation_rate)

        for widget in (self.rrtsp_url, self.rrtsp_username, self.rrtsp_password):
            widget.textChanged.connect(self._on_form_changed)
        self.simulation_rate.valueChanged.connect(self._on_form_changed)
        return page

    def _display_page(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(34, 30, 34, 30)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(16)

        self.preview_range = QDoubleSpinBox()
        self.preview_range.setRange(50, 100000)
        self.preview_range.setSuffix(" m")
        self.preview_range.setDecimals(0)
        self.preview_range.valueChanged.connect(self._on_form_changed)
        form.addRow(self._form_label(t("preview_range")), self.preview_range)
        return page

    def _map_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        path_row = QHBoxLayout()
        path_row.addWidget(self._form_label(t("map_image")))
        self.map_path = QLineEdit()
        self.map_path.setPlaceholderText(t("image_placeholder"))
        browse = QPushButton(t("browse"))
        browse.clicked.connect(self.browse_map)
        clear = QPushButton(t("clear"))
        clear.clicked.connect(lambda: self.map_path.setText(""))
        path_row.addWidget(self.map_path, 1)
        path_row.addWidget(browse)
        path_row.addWidget(clear)
        layout.addLayout(path_row)

        self.map_preview = MapRadarView()
        self.map_preview.setMinimumHeight(340)
        self.map_preview.mapClicked.connect(self._map_clicked)
        layout.addWidget(self.map_preview, 1)

        bottom = QHBoxLayout()
        position_grid = QGridLayout()
        position_grid.setHorizontalSpacing(10)
        position_grid.setVerticalSpacing(12)

        self.radar_x = self._map_spin(-100000, 100000, " px", 0)
        self.radar_y = self._map_spin(-100000, 100000, " px", 0)
        self.pixels_per_100m = self._map_spin(1, 10000, " px/100m", 1)
        self.heading = self._map_spin(-360, 360, "\u00b0", 1)
        position_grid.addWidget(self._form_label(t("radar_x")), 0, 0)
        position_grid.addWidget(self.radar_x, 0, 1)
        position_grid.addWidget(self._form_label(t("radar_y")), 0, 2)
        position_grid.addWidget(self.radar_y, 0, 3)
        position_grid.addWidget(self._form_label(t("map_scale")), 1, 0)
        position_grid.addWidget(self.pixels_per_100m, 1, 1)
        position_grid.addWidget(self._form_label(t("radar_heading")), 1, 2)
        position_grid.addWidget(self.heading, 1, 3)
        bottom.addLayout(position_grid, 3)

        actions = QVBoxLayout()
        actions.setSpacing(8)
        rotate_row = QHBoxLayout()
        for text, delta in (
            (t("rotate_left_1"), -1.0),
            (t("rotate_left_01"), -0.1),
            (t("rotate_right_01"), 0.1),
            (t("rotate_right_1"), 1.0),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked=False, d=delta: self._rotate_map(d))
            rotate_row.addWidget(btn)
        actions.addLayout(rotate_row)

        action_row = QHBoxLayout()
        reset = QPushButton(t("reset"))
        reset.clicked.connect(self._reset_map_rotation)
        arm = QPushButton(t("pick_position_short"))
        arm.clicked.connect(self.map_preview.arm_radar_position)
        action_row.addWidget(reset)
        action_row.addWidget(arm)
        action_row.addStretch(1)
        actions.addLayout(action_row)
        bottom.addLayout(actions, 2)
        layout.addLayout(bottom)

        hint = QLabel(t("map_hint"))
        hint.setStyleSheet("color:#607186;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.map_path.textChanged.connect(self._on_form_changed)
        for widget in (self.radar_x, self.radar_y, self.pixels_per_100m, self.heading):
            widget.valueChanged.connect(self._on_form_changed)
        return page

    @staticmethod
    def _map_spin(minimum: float, maximum: float, suffix: str, decimals: int):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setDecimals(decimals)
        spin.setFixedWidth(86)
        return spin

    def _onnx_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        row = QHBoxLayout()
        row.addWidget(self._form_label(t("onnx_file")))
        self.onnx_path = QLineEdit()
        browse = QPushButton(t("select_model"))
        browse.clicked.connect(self.browse_model)
        clear = QPushButton(t("clear_model"))
        clear.clicked.connect(lambda: self.onnx_path.setText(""))
        row.addWidget(self.onnx_path, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        layout.addLayout(row)

        self.onnx_hint = QLabel()
        self.onnx_hint.setStyleSheet("color:#607186;")
        self.onnx_hint.setWordWrap(True)
        layout.addWidget(self.onnx_hint)
        layout.addStretch(1)
        self.onnx_path.textChanged.connect(self._on_form_changed)
        return page

    def _general_page(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(34, 30, 34, 30)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(16)

        self.render_fps = QSpinBox()
        self.render_fps.setRange(5, 60)
        self.render_fps.setSuffix(" fps")
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        form.addRow(self._form_label(t("render_fps")), self.render_fps)
        form.addRow(self._form_label(t("log_level")), self.log_level)
        return page

    def _hide_spin_buttons(self):
        for spin in self.findChildren(QAbstractSpinBox):
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    # ------------------------------------------------------------ load
    def load_from_cfg(self):
        self._building = True
        self.instance_table.setRowCount(0)
        for item in self.cfg.get("instances", []):
            self._append_table_row(item)

        selected_id = self.cfg.get("ui", {}).get("selected_instance_id", "")
        row = self._row_for_id(selected_id)
        if row < 0 and self.instance_table.rowCount():
            row = 0
        if row >= 0:
            self.instance_table.selectRow(row)
            self._load_selected_instance()

        self.render_fps.setValue(int(self.cfg.get("ui", {}).get("render_fps", 30)))
        self.log_level.setCurrentText(self.cfg.get("logging", {}).get("level", "INFO"))
        self._hide_spin_buttons()
        self._building = False

    def _append_table_row(self, item):
        row = self.instance_table.rowCount()
        self.instance_table.insertRow(row)

        holder = QWidget()
        box = QHBoxLayout(holder)
        box.setContentsMargins(9, 0, 0, 0)
        check = QCheckBox()
        check.setChecked(bool(item.get("enabled", True)))
        check.stateChanged.connect(
            lambda _v, instance_id=item["id"]: self._sync_enabled(instance_id, _v))
        box.addWidget(check)
        self.instance_table.setCellWidget(row, 0, holder)

        name = QTableWidgetItem(str(item.get("name", "")))
        name.setData(Qt.ItemDataRole.UserRole, item["id"])
        kind = item.get("source", {}).get("kind", "rrtsp")
        kind_item = QTableWidgetItem(
            t("kind_simulation") if kind == "simulation" else t("kind_rrtsp"))
        kind_item.setFlags(kind_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        url_text = item.get("source", {}).get("rrtsp", {}).get("url", "")
        url_item = QTableWidgetItem(str(url_text))
        url_item.setFlags(url_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.instance_table.setItem(row, 1, name)
        self.instance_table.setItem(row, 2, kind_item)
        self.instance_table.setItem(row, 3, url_item)

    def _row_for_id(self, instance_id):
        for row in range(self.instance_table.rowCount()):
            item = self.instance_table.item(row, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == instance_id:
                return row
        return -1

    # -------------------------------------------------------- instance
    def _selected_row(self):
        rows = {index.row() for index in self.instance_table.selectedIndexes()}
        return min(rows) if rows else -1

    def _selected_instance_id(self):
        row = self._selected_row()
        item = self.instance_table.item(row, 1) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _selected_instance(self):
        instance_id = self._selected_instance_id()
        for item in self.cfg.get("instances", []):
            if item["id"] == instance_id:
                return item
        return None

    def _on_selection_changed(self):
        if self._building:
            return
        self._commit_instance(self.current_editor_id)
        self._load_selected_instance()

    def _on_cell_changed(self, row, col):
        if self._building:
            return
        item = self.cfg.get("instances", [])[row]
        if col == 1:
            item["name"] = self.instance_table.item(row, 1).text().strip() or item["id"]
            if row == self._selected_row():
                self.editor_title.setText(item["name"])

    def _sync_enabled(self, instance_id, state):
        for item in self.cfg.get("instances", []):
            if item.get("id") == instance_id:
                item["enabled"] = bool(state)
                return

    def _load_selected_instance(self):
        item = self._selected_instance()
        self._building = True
        if not item:
            self.editor_title.setText("")
            self.current_editor_id = ""
            self._building = False
            return

        source = item["source"]
        self.current_editor_id = item["id"]
        self.editor_title.setText(item["name"])
        kind = source.get("kind", "rrtsp")
        self.source_kind.setCurrentIndex(0 if kind == "rrtsp" else 1)
        rrtsp = source.get("rrtsp", {})
        self.rrtsp_url.setText(rrtsp.get("url", ""))
        self.rrtsp_username.setText(rrtsp.get("username", ""))
        self.rrtsp_password.setText(rrtsp.get("password", ""))
        self.simulation_rate.setValue(source.get("simulation", {}).get("rate_hz", 5))

        self.preview_range.setValue(item.get("display", {}).get("plane_range_m", 3000))
        map_cfg = item.get("map", {})
        self.map_path.setText(map_cfg.get("image_path", ""))
        self.radar_x.setValue(map_cfg.get("radar_x", 975))
        self.radar_y.setValue(map_cfg.get("radar_y", 700))
        self.pixels_per_100m.setValue(map_cfg.get("pixels_per_100m", 20))
        self.heading.setValue(map_cfg.get("heading_deg", 0))
        self.onnx_path.setText(item.get("model", {}).get("onnx_path", ""))
        self._update_model_hint(item)
        self._update_editor_enabled()
        self._update_map_preview()
        self._hide_spin_buttons()
        self._building = False

    def _commit_instance(self, instance_id):
        if not instance_id:
            return
        item = next(
            (candidate for candidate in self.cfg.get("instances", [])
             if candidate["id"] == instance_id),
            None,
        )
        if not item:
            return

        kind = self.source_kind.currentData()
        item["source"] = {
            "kind": kind,
            "rrtsp": {
                "url": self.rrtsp_url.text().strip(),
                "username": self.rrtsp_username.text().strip(),
                "password": self.rrtsp_password.text(),
                "remote_inference": bool(
                    item.get("source", {}).get("rrtsp", {}).get("remote_inference", False)
                ),
            },
            "simulation": {
                "rate_hz": float(self.simulation_rate.value()),
            },
        }
        item["display"] = {"plane_range_m": float(self.preview_range.value())}
        item["map"] = normalize_map({
            "image_path": self.map_path.text().strip(),
            "radar_x": float(self.radar_x.value()),
            "radar_y": float(self.radar_y.value()),
            "pixels_per_100m": float(self.pixels_per_100m.value()),
            "heading_deg": float(self.heading.value()),
        })
        item["model"] = {"onnx_path": self.onnx_path.text().strip()}
        self._update_model_hint(item)
        self._refresh_summary_for_id(instance_id)

    def _refresh_summary_for_id(self, instance_id):
        row = self._row_for_id(instance_id)
        item = next(
            (candidate for candidate in self.cfg.get("instances", [])
             if candidate["id"] == instance_id),
            None,
        )
        if row < 0 or not item:
            return
        self._building = True
        kind = item["source"]["kind"]
        self.instance_table.item(row, 2).setText(
            t("kind_simulation") if kind == "simulation" else t("kind_rrtsp"))
        self.instance_table.item(row, 3).setText(
            item["source"]["rrtsp"]["url"] if kind == "rrtsp" else "")
        self._building = False

    def _on_form_changed(self):
        if self._building:
            return
        self._commit_instance(self.current_editor_id)
        self._update_editor_enabled()
        self._update_map_preview()

    def _update_editor_enabled(self):
        simulation = self.source_kind.currentData() == "simulation"
        for widget in (self.rrtsp_url, self.rrtsp_username, self.rrtsp_password):
            widget.setEnabled(not simulation)
        self.simulation_rate.setEnabled(simulation)

    def _update_model_hint(self, item=None):
        item = item or self._selected_instance()
        source = (item or {}).get("source", {})
        remote = (
            source.get("kind", "rrtsp") == "rrtsp"
            and bool(source.get("rrtsp", {}).get("remote_inference", False))
        )
        if remote:
            self.onnx_hint.setText(t("server_model_hint"))
        elif self.onnx_path.text().strip():
            self.onnx_hint.setText(t("local_model_hint"))
        else:
            self.onnx_hint.setText(t("no_model_hint"))

    # ----------------------------------------------------------- maps
    def _map_clicked(self, point):
        self._building = True
        self.radar_x.setValue(point.x())
        self.radar_y.setValue(point.y())
        self._building = False
        self._on_form_changed()

    def _rotate_map(self, delta):
        value = self.heading.value() + delta
        if value > 360:
            value -= 360
        if value < -360:
            value += 360
        self.heading.setValue(value)

    def _reset_map_rotation(self):
        self.heading.setValue(0.0)

    def _update_map_preview(self):
        item = self._selected_instance()
        if not item:
            return
        self.map_preview.configure(
            item.get("map", {}),
            float(self.preview_range.value()))
        self.map_preview.update_targets([], {})

    # ----------------------------------------------------------- add
    def add_radar_instance(self):
        if len(self.cfg.get("instances", [])) >= 8:
            QMessageBox.information(self, t("limit_title"), t("limit_message"))
            return
        row = len(self.cfg["instances"])
        item = default_instance(uuid4().hex[:12], f"\u96f7\u8fbe {row + 1}", "rrtsp")
        item["source"]["rrtsp"]["url"] = f"rtsp://127.0.0.1:{8554 + row}/radar"
        self.cfg["instances"].append(item)
        self._append_table_row(item)
        self.instance_table.selectRow(row)

    def add_simulation_instance(self):
        if len(self.cfg.get("instances", [])) >= 8:
            QMessageBox.information(self, t("limit_title"), t("limit_message"))
            return
        row = len(self.cfg["instances"])
        item = default_instance(
            uuid4().hex[:12], f"\u5185\u7f6e\u4eff\u771f {row + 1}", "simulation")
        self.cfg["instances"].append(item)
        self._append_table_row(item)
        self.instance_table.selectRow(row)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.instance_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.instance_table.removeRow(row)
            if row < len(self.cfg["instances"]):
                self.cfg["instances"].pop(row)
        selected_row = self._selected_row()
        if selected_row < 0 and self.instance_table.rowCount():
            self.instance_table.selectRow(0)
        self._load_selected_instance()

    def import_clipboard_url(self):
        if len(self.cfg.get("instances", [])) >= 8:
            QMessageBox.information(self, t("limit_title"), t("limit_message"))
            return
        parsed = urlparse(QApplication.clipboard().text().strip())
        if parsed.scheme != "rtsp" or not parsed.hostname:
            QMessageBox.warning(self, t("invalid_url_title"), t("invalid_url_message"))
            return

        port = parsed.port or 8554
        path_value = parsed.path or "/radar"
        url = urlunparse(("rtsp", f"{parsed.hostname}:{port}", path_value, "", "", ""))
        row = len(self.cfg["instances"])
        item = default_instance(uuid4().hex[:12], f"\u96f7\u8fbe {row + 1}", "rrtsp")
        item["source"]["rrtsp"] = {
            "url": url,
            "username": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
        }
        self.cfg["instances"].append(item)
        self._append_table_row(item)
        self.instance_table.selectRow(row)
        self.state_label.setText(t("state_imported"))

    # --------------------------------------------------------- gather
    def gather_cfg(self):
        self._commit_instance(self.current_editor_id)
        self.cfg["ui"]["render_fps"] = int(self.render_fps.value())
        self.cfg["ui"]["selected_instance_id"] = self._selected_instance_id()
        self.cfg["logging"]["level"] = self.log_level.currentText()
        return self.cfg

    def apply_settings(self):
        cfg = self.gather_cfg()
        self.receiver_cmd.put({"cmd": "apply", "cfg": cfg})
        self.inference_cmd.put({"cmd": "apply", "cfg": cfg})
        self.state_label.setText(
            f"{t('state_saved')} \u00b7 {time.strftime('%H:%M:%S')}")
        self.applied.emit(cfg)

    # -------------------------------------------------------- browse
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, t("select_model"), "", "ONNX (*.onnx);;All files (*.*)")
        if path:
            self.onnx_path.setText(os.path.normpath(path))

    def browse_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self, t("map_image"), "",
            "Images (*.png *.jpg *.jpeg *.bmp *.svg);;All files (*.*)")
        if path:
            self.map_path.setText(os.path.normpath(path))
