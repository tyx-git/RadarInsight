"""通用 UI 控件: 卡片 / 分区标题 / 状态灯 / 指标卡 / 类别标签."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QGraphicsDropShadowEffect)

from aerosight.ui.theme import C, category_color


def card(content: QWidget) -> QFrame:
    """把内容包进带圆角边框的卡片."""
    f = QFrame(); f.setObjectName("Card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.addWidget(content)
    return f


def section_title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("SectionHeader")
    lab.setText(text.upper())
    return lab


class LedDot(QWidget):
    """圆形状态灯: 支持发光."""

    def __init__(self, color: str = C.green, size: int = 10):
        super().__init__()
        self._color = QColor(color)
        self.setFixedSize(size + 4, size + 4)
        self._size = size

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, e):
        from PySide6.QtGui import QPainter, QRadialGradient, QPen, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._size
        x = (self.width() - r) / 2
        y = (self.height() - r) / 2
        glow = QRadialGradient(x + r / 2, y + r / 2, r)
        glow.setColorAt(0.5, self._color)
        c2 = QColor(self._color); c2.setAlpha(40)
        glow.setColorAt(1.0, c2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(int(x), int(y), r, r)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(int(x + r * 0.28), int(y + r * 0.28),
                      int(r * 0.44), int(r * 0.44))


class MetricCard(QFrame):
    """指标卡: 名称 / 大数字 / 单位 / 状态色."""

    def __init__(self, name: str, unit: str = "", accent: str = C.cyan):
        super().__init__()
        self.setObjectName("Card")
        self.setFixedHeight(76)
        self.accent = accent

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        top = QHBoxLayout()
        n = QLabel(name); n.setObjectName("MetricName")
        top.addWidget(n)
        top.addStretch(1)
        self.dot = LedDot(accent, 8)
        top.addWidget(self.dot)
        lay.addLayout(top)

        row = QHBoxLayout()
        self.value = QLabel("--")
        self.value.setObjectName("MetricValue")
        row.addWidget(self.value)
        row.addStretch(1)
        u = QLabel(unit); u.setObjectName("MetricUnit")
        row.addWidget(u, alignment=Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)

    def set_value(self, v, color: str | None = None):
        self.value.setText(str(v))
        if color:
            self.value.setStyleSheet(f"color:{color};")
            self.dot.set_color(color)

    def set_accent(self, color: str):
        self.dot.set_color(color)


class Chip(QLabel):
    """类别小标签."""

    def __init__(self, text: str, color: str | None = None):
        super().__init__(" " + text + " ")
        color = color or category_color(text)
        self.setFixedHeight(20)
        self.setStyleSheet(
            f"background:{color}22; color:{color}; "
            f"border:1px solid {color}66; border-radius:9px; "
            f"font-size:8.5pt; font-weight:600; padding:0 4px;")


class InfoRow(QWidget):
    """左标题右值的信息行."""

    def __init__(self, title: str, value: str = "-"):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        self.t = QLabel(title); self.t.setStyleSheet(f"color:{C.text_dim};")
        self.v = QLabel(value)
        lay.addWidget(self.t); lay.addStretch(1); lay.addWidget(self.v)

    def set_value(self, v: str):
        self.v.setText(str(v))


class BlinkDot(LedDot):
    """告警闪烁灯."""
    def __init__(self, color=C.amber, size=10):
        super().__init__(color, size)
        self._on = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._toggle)
        self.timer.start(700)

    def _toggle(self):
        self._on = not self._on
        c = QColor(self._color)
        c.setAlpha(255 if self._on else 60)
        self._color = c
        self.update()

    def stop(self):
        self.timer.stop()
