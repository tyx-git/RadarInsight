#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AeroSight R-RTSP 推流工具启动入口。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Windows console / redirected output: force UTF-8 to keep Chinese logs intact
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from rrtsp.gui import run_gui

if __name__ == "__main__":
    sys.exit(run_gui())
