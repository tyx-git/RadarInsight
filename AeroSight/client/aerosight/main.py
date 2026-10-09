# -*- coding: utf-8 -*-
"""AeroSight 客户端入口：后台接收/推理进程 + 概览 UI。"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import sys
import threading

from aerosight.app_config import load_config
from aerosight.logging_setup import setup_logging
from aerosight.ui.main_window import MainWindow

log = logging.getLogger(__name__)


def _install_shutdown_timer(app, interval_ms: int = 100):
    from PySide6.QtCore import QTimer

    shutdown_requested = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, lambda *_args: shutdown_requested.set())
        except (OSError, ValueError):
            pass

    timer = QTimer(app)
    timer.setInterval(max(int(interval_ms), 1))
    timer.timeout.connect(
        lambda: app.quit() if shutdown_requested.is_set() else None)
    timer.start()
    return shutdown_requested, timer


def main() -> int:
    cfg = load_config()
    setup_logging(cfg["logging"]["level"], cfg["logging"]["dir"])

    mp.freeze_support()
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    message_q = ctx.Queue(maxsize=int(cfg["pipeline"]["message_queue_size"]))
    inference_q = ctx.Queue(maxsize=int(cfg["pipeline"]["inference_queue_size"]))
    result_q = ctx.Queue(maxsize=int(cfg["pipeline"]["result_queue_size"]))
    receiver_cmd = ctx.Queue()
    inference_cmd = ctx.Queue()

    from PySide6.QtWidgets import QApplication
    from aerosight.ui.theme import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
    _shutdown_requested, shutdown_timer = _install_shutdown_timer(app)

    receiver = ctx.Process(
        target=_receiver_entry,
        args=(cfg, message_q, receiver_cmd, stop_event),
        name="aerosight-receiver", daemon=True)
    worker = ctx.Process(
        target=_inference_entry,
        args=(cfg, inference_q, result_q, inference_cmd, stop_event),
        name="aerosight-inference", daemon=True)
    receiver.start()
    worker.start()

    win = MainWindow(
        cfg, message_q, inference_q, result_q,
        receiver_cmd, inference_cmd, stop_event)
    win.show()
    try:
        code = app.exec()
    finally:
        shutdown_timer.stop()
        win.close()
        stop_event.set()
        receiver.join(timeout=3)
        worker.join(timeout=3)
        if receiver.is_alive():
            receiver.terminate()
        if worker.is_alive():
            worker.terminate()
    return code


def _receiver_entry(cfg, message_q, receiver_cmd, stop_event):
    setup_logging(cfg["logging"]["level"], cfg["logging"]["dir"])
    from aerosight.pipeline.receiver import receiver_main
    receiver_main(cfg, message_q, receiver_cmd, stop_event)


def _inference_entry(cfg, inference_q, result_q, inference_cmd, stop_event):
    setup_logging(cfg["logging"]["level"], cfg["logging"]["dir"])
    from aerosight.pipeline.inference_worker import inference_main
    inference_main(cfg, inference_q, result_q, inference_cmd, stop_event)


if __name__ == "__main__":
    sys.exit(main())
