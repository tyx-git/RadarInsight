#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AeroSight 客户端启动入口。

用法: python main.py   (或直接双击)
从任意目录启动都可以, 依赖使用 aerosight-client 环境。
"""
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

from aerosight.main import main

if __name__ == "__main__":
    sys.exit(main())
