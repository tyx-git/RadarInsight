# -*- coding: utf-8 -*-
"""AeroSight color theme."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


_UI_FONT_FAMILY = "Microsoft YaHei UI"


def ui_font_family() -> str:
    return _UI_FONT_FAMILY


def _resolve_ui_font() -> str:
    families = set(QFontDatabase.families())
    preferred = (
        "Microsoft YaHei UI", "Microsoft YaHei", "DengXian",
        "SimHei", "SimSun", "Segoe UI", "Arial",
    )
    for family in preferred:
        if family in families:
            return family

    windows_dir = os.environ.get("WINDIR", r"C:\\Windows")
    font_dir = Path(windows_dir) / "Fonts"
    for filename in ("msyh.ttc", "Deng.ttf", "simhei.ttf", "simsun.ttc"):
        path = font_dir / filename
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        registered = QFontDatabase.applicationFontFamilies(font_id)
        if registered:
            return registered[0]
    return "Arial"


class C:
    bg = "#F2F5F8"
    panel = "#FFFFFF"
    panel_alt = "#F5F7FA"
    card = "#FFFFFF"
    border = "#D3DAE3"
    border_h = "#9AA8B8"
    text = "#1B2735"
    text_dim = "#536273"
    text_faint = "#788697"
    blue = "#1769AA"
    blue_dark = "#125584"
    cyan = "#007F86"
    green = "#287A4B"
    amber = "#A85D12"
    purple = "#7351A6"
    slate = "#5D6A77"
    red = "#B83B3B"

    @classmethod
    def qcolor(cls, name: str) -> QColor:
        return QColor(getattr(cls, name))


CATEGORY_COLORS = {
    "drone": C.cyan,
    "bird": C.amber,
    "balloon": C.purple,
    "clutter": C.slate,
    "other": C.blue,
}


def category_color(label: str) -> str:
    text = (label or "").lower()
    for key, value in CATEGORY_COLORS.items():
        if key in text:
            return value
    return C.text_dim


QSS = f"""
* {{ font-size:10pt; color:{C.text}; outline:none; }}
QMainWindow,QDialog {{ background:{C.bg}; }}
QFrame#TopBar {{ background:{C.panel}; border:none; border-bottom:1px solid {C.border}; }}
QFrame#TargetPanel {{ background:{C.panel}; border:none; border-left:1px solid {C.border}; }}
QFrame#ViewBar {{ background:{C.panel}; border:none; border-bottom:1px solid {C.border}; }}
QLabel#Brand {{ font-size:16pt; font-weight:800; color:{C.blue}; }}
QLabel#BrandSub {{ font-size:8.5pt; color:{C.text_faint}; }}
QLabel#PanelHeader {{ color:{C.text}; font-size:13pt; font-weight:800; }}
QLabel#InstanceName {{ color:{C.blue}; font-size:16pt; font-weight:800; }}
QFrame#HLine {{ background:{C.border}; border:none; max-height:1px; }}
QFrame#MiniCard,QFrame#Card {{ background:{C.card}; border:1px solid {C.border}; border-radius:4px; }}
QLabel#MiniTitle {{ color:{C.text_faint}; font-size:8pt; }}
QLabel#MiniValue {{ color:{C.text}; font-size:14pt; font-weight:800; }}
QLabel#TargetCount {{ color:{C.text_dim}; font-size:9pt; font-weight:700; }}
QLabel#MetricValue {{ font-size:19pt; font-weight:800; color:{C.text}; }}
QLabel#MetricUnit,QLabel#MetricName {{ color:{C.text_dim}; }}
QPushButton {{ background:{C.panel}; border:1px solid {C.border}; border-radius:4px; padding:6px 12px; }}
QPushButton:hover {{ background:#F0F6FF; border-color:{C.blue}; }}
QPushButton:pressed {{ background:#E4EEFB; }}
QPushButton:checked {{ background:#E7F1F7; border-color:{C.blue}; color:{C.blue_dark}; }}
QPushButton:disabled {{ color:{C.text_faint}; background:{C.panel_alt}; }}
QPushButton#Primary {{ background:{C.blue}; border-color:{C.blue_dark}; color:white; font-weight:700; }}
QPushButton#Primary:hover {{ background:{C.blue_dark}; }}
QPushButton#ModeToggle {{ min-width:76px; }}
QPushButton#StartDetectionButton,QPushButton#EmergencyStopButton {{ background:rgba(232,235,247,225); border:1px solid rgba(255,255,255,235); border-radius:8px; color:#1F2B3D; font-size:11pt; font-weight:700; padding:0px; }}
QPushButton#StartDetectionButton:hover,QPushButton#EmergencyStopButton:hover {{ background:rgba(245,247,255,240); border-color:rgba(255,255,255,255); }}
QPushButton#StartDetectionButton:pressed,QPushButton#EmergencyStopButton:pressed {{ background:rgba(205,211,230,240); }}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox {{ background:white; border:1px solid {C.border}; border-radius:4px; padding:5px 8px; min-height:18px; }}
QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus {{ border-color:{C.blue}; }}
QSpinBox::up-button,QSpinBox::down-button,QDoubleSpinBox::up-button,QDoubleSpinBox::down-button {{ width:0; height:0; border:none; }}
QComboBox::drop-down {{ border:none; width:22px; }}
QComboBox QAbstractItemView {{ background:white; border:1px solid {C.border}; selection-background-color:#DBEAFE; selection-color:{C.text}; }}
QGroupBox {{ background:white; border:1px solid {C.border}; border-radius:4px; margin-top:12px; padding:12px; font-weight:700; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 5px; color:{C.blue}; }}
QTabWidget::pane {{ border:1px solid {C.border}; border-radius:4px; background:white; }}
QTabBar::tab {{ background:transparent; color:{C.text_dim}; padding:9px 20px; border:1px solid transparent; font-size:10pt; }}
QTabBar::tab:selected {{ color:{C.blue}; background:white; border-color:{C.border}; border-bottom-color:white; }}
QTabBar::tab:hover {{ color:{C.blue}; }}
QTableWidget,QListWidget {{ background:white; border:1px solid {C.border}; border-radius:4px; alternate-background-color:#F7F9FB; font-size:9pt; }}
QTableWidget {{ gridline-color:#E4ECF4; }}
QTableWidget::item {{ padding:4px 3px; border-bottom:1px solid #EEF1F4; }}
QTableWidget::item:selected {{ background:#E8F1F6; color:{C.text}; }}
QHeaderView::section {{ background:white; color:{C.text_dim}; border:none; padding:7px; font-size:10pt; font-weight:700; }}
QTableWidget::item {{ padding:4px; }}
QScrollBar:vertical {{ background:transparent; width:10px; }}
QScrollBar::handle:vertical {{ background:{C.border_h}; border-radius:5px; min-height:28px; }}
QScrollBar:horizontal {{ background:transparent; height:10px; }}
QScrollBar::handle:horizontal {{ background:{C.border_h}; border-radius:5px; min-width:28px; }}
QStatusBar {{ background:white; color:{C.text_dim}; border-top:1px solid {C.border}; }}
QCheckBox {{ spacing:7px; }}
QMenu {{ background:white; border:1px solid {C.border}; }}
QMenu::item:selected {{ background:#DBEAFE; }}
"""


def apply_theme(app: QApplication):
    global _UI_FONT_FAMILY
    _UI_FONT_FAMILY = _resolve_ui_font()
    app.setFont(QFont(_UI_FONT_FAMILY, 10))
    app.setStyleSheet(QSS)
