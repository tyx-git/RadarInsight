# -*- coding: utf-8 -*-
"""AeroSight R-RTSP 推流工具的紧凑 Tkinter 界面。

设计为单面板手风琴：8 个推流组以卡片纵向排列，头部一行承载常用
控件，点击展开才显示数据包、账号、链接等详情，避免分页式留白。
每个推流实例只引用一个用户选择的 .asd 文件。
"""
from __future__ import annotations

import json
import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import quote

from .asd_format import AsdData, AsdError
from .netutil import detect_lan_ip
from .server import RRTSPServer

MAX_GROUPS = 8
CONFIG_NAME = "tool_config.json"
AUTH_OPEN = "开放模式"
AUTH_DIGEST = "私密模式"

# 配色
ROW_ODD = "#FFFFFF"
ROW_EVEN = "#F3F6FA"
ROW_EXPANDED = "#E5EDF7"
ROW_BORDER = "#D3DAE4"
DETAIL_BG = "#FAFBFD"
TEXT_MUTED = "#7A8798"
TEXT_DARK = "#1F2937"
COLOR_OK = "#047857"
COLOR_ERR = "#B91C1C"
COLOR_WARN = "#B45309"


def tool_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def config_path() -> str:
    return os.path.join(tool_dir(), CONFIG_NAME)


def default_group(index: int) -> dict:
    return {
        "name": f"推流 {index + 1}",
        "port": 8554 + index,
        "asd_path": "",
        "rate_hz": 10.0,
        "auth_mode": "open",
        "username": "viewer",
        "password": "",
        "embed_credentials": False,
    }


class StreamItem:
    """一个推流组：头部行 + 可折叠详情。"""

    def __init__(self, master, app: "RRTSPToolGUI", index: int):
        self.app = app
        self.index = index
        self.server: RRTSPServer | None = None
        self.loading = True
        self.expanded = False

        # 每个条目独立样式，便于切换底色
        self.frame_style = f"item{index}.TFrame"
        self.label_style = f"item{index}.TLabel"
        self.dot_style = f"item{index}.dot.TLabel"
        for style_name in (self.frame_style,):
            app.style.configure(style_name)
        app.style.configure(self.label_style, foreground=TEXT_DARK)
        app.style.configure(self.dot_style, font=("Segoe UI", 11))

        self.container = ttk.Frame(master, style=self.frame_style)
        self.header = ttk.Frame(self.container, style=self.frame_style)
        self.header.pack(fill="x")

        self._build_header()
        self.detail = ttk.Frame(self.container, style="Detail.TFrame")
        self._build_detail()

        self.apply_background(ROW_ODD if index % 2 == 0 else ROW_EVEN)
        self._bind_header_click()
        self.loading = False

    # ---------------------------------------------------------- header
    def _build_header(self):
        h = self.header
        self.toggle_btn = ttk.Button(h, text="启动", command=self.toggle, width=5)
        self.toggle_btn.pack(side="left", padx=(7, 6), pady=3)

        self.dot_label = ttk.Label(h, text="●", style=self.dot_style)
        self.dot_label.pack(side="left")
        self.status_label = ttk.Label(h, text="未启动", style=self.label_style,
                                     width=7, anchor="w")
        self.status_label.pack(side="left", padx=(3, 8))

        ttk.Separator(h, orient="vertical").pack(
            side="left", fill="y", padx=(0, 8))

        self.name_entry = ttk.Entry(h, width=10)
        self.name_entry.pack(side="left")

        ttk.Label(h, text="端口", style=self.label_style).pack(
            side="left", padx=(10, 3))
        self.port_entry = ttk.Entry(h, width=6)
        self.port_entry.pack(side="left")

        ttk.Label(h, text="速率", style=self.label_style).pack(
            side="left", padx=(10, 3))
        self.rate_entry = ttk.Entry(h, width=5)
        self.rate_entry.pack(side="left")
        ttk.Label(h, text="Hz", style=self.label_style).pack(
            side="left", padx=(2, 0))

        ttk.Label(h, text="鉴权", style=self.label_style).pack(
            side="left", padx=(10, 3))
        self.auth_combo = ttk.Combobox(
            h, width=9, state="readonly",
            values=(AUTH_OPEN, AUTH_DIGEST))
        self.auth_combo.pack(side="left")
        self.auth_combo.bind("<<ComboboxSelected>>", self.on_auth_changed)

        ttk.Frame(h, style=self.frame_style).pack(
            side="left", fill="x", expand=True)
        self.chevron = ttk.Label(h, text="详情 ▸", style=self.label_style)
        self.chevron.pack(side="right", padx=9)

        for widget in (self.name_entry, self.port_entry, self.rate_entry):
            widget.bind("<FocusOut>", self.on_control_changed)
            widget.bind("<Return>", self.on_control_changed)

    def _bind_header_click(self):
        def on_click(_event=None):
            self.set_expanded(not self.expanded)
        for widget in (self.header, self.dot_label, self.status_label,
                       self.chevron):
            widget.bind("<Button-1>", on_click)
        for child in self.header.winfo_children():
            if child.winfo_class() in ("TLabel", "TFrame", "TSeparator"):
                child.bind("<Button-1>", on_click)

    # ---------------------------------------------------------- detail
    def _build_detail(self):
        inner = ttk.Frame(self.detail, padding=(8, 5), style="Detail.TFrame")
        inner.pack(fill="x")

        # 数据包行
        row1 = ttk.Frame(inner, style="Detail.TFrame")
        row1.pack(fill="x")
        ttk.Label(row1, text=".asd 数据包", style="Detail.TLabel").pack(side="left")
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(
            row1, textvariable=self.path_var, state="readonly")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(7, 5))
        self.browse_btn = ttk.Button(
            row1, text="选择", command=self.browse, width=5)
        self.browse_btn.pack(side="left", padx=(0, 4))
        self.clear_btn = ttk.Button(
            row1, text="清空", command=self.clear_path, width=5)
        self.clear_btn.pack(side="left")

        self.info_label = ttk.Label(
            inner, text="未选择数据包", style="DetailMuted.TLabel")
        self.info_label.pack(fill="x", pady=(3, 0))

        ttk.Separator(inner).pack(fill="x", pady=4)

        # 账号 + 链接
        self.auth_hint = ttk.Label(
            inner,
            text=("私密模式：观看端必须输入账号密码（账号密码经 Digest "
                    "加密校验，不发送明文）"),
            style="DetailMuted.TLabel")
        self.auth_hint.pack(fill="x", pady=(0, 3))
        row2 = ttk.Frame(inner, style="Detail.TFrame")
        row2.pack(fill="x")
        ttk.Label(row2, text="账号", style="Detail.TLabel").pack(side="left")
        self.username_entry = ttk.Entry(row2, width=14)
        self.username_entry.pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="密码", style="Detail.TLabel").pack(side="left")
        self.password_entry = ttk.Entry(row2, width=14, show="*")
        self.password_entry.pack(side="left", padx=(4, 8))
        self.embed_var = tk.BooleanVar(value=False)
        self.embed_check = ttk.Checkbutton(
            row2, text="链接携带账号信息", variable=self.embed_var,
            command=self.on_control_changed)
        self.embed_check.pack(side="left")

        row3 = ttk.Frame(inner, style="Detail.TFrame")
        row3.pack(fill="x", pady=(4, 0))
        ttk.Label(row3, text="推流链接", style="Detail.TLabel").pack(side="left")
        self.link_var = tk.StringVar()
        ttk.Entry(row3, textvariable=self.link_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(7, 5))
        ttk.Button(row3, text="复制", command=self.copy_link, width=5).pack(
            side="left")

        for widget in (self.username_entry, self.password_entry):
            widget.bind("<FocusOut>", self.on_control_changed)
            widget.bind("<Return>", self.on_control_changed)

    # ------------------------------------------------------ expand/collapse
    def set_expanded(self, expanded: bool):
        if expanded:
            self.app.expand_item(self.index)
        else:
            self.detail.pack_forget()
            self.expanded = False
            self.chevron.configure(text="详情 ▸")
            self.apply_background(
                ROW_ODD if self.index % 2 == 0 else ROW_EVEN)

    def show_detail(self):
        self.detail.pack(fill="x")
        self.expanded = True
        self.chevron.configure(text="收起 ▾")
        self.apply_background(ROW_EXPANDED)

    def apply_background(self, color: str):
        self.app.style.configure(self.frame_style, background=color)
        self.app.style.configure(self.label_style, background=color)
        self.app.style.configure(self.dot_style, background=color)

    # ---------------------------------------------------------- load/save
    def load(self, data: dict):
        self.loading = True
        values = default_group(self.index)
        values.update(data or {})

        self._replace_entry(self.name_entry, str(values["name"]))
        self._replace_entry(self.port_entry, str(values["port"]))
        self._replace_entry(self.rate_entry, str(values["rate_hz"]))
        self.path_var.set(str(values.get("asd_path", "")))

        raw = data or {}
        auth_mode = str(raw.get("auth_mode", "")).strip().lower()
        if not auth_mode:
            auth_label = str(raw.get("auth_label", ""))
            auth_mode = "digest" if "digest" in auth_label.lower() else "open"
        self.auth_combo.set(AUTH_DIGEST if auth_mode == "digest" else AUTH_OPEN)

        self._replace_entry(self.username_entry, str(values["username"]))
        self._replace_entry(self.password_entry, str(values["password"]))
        self.embed_var.set(bool(values.get("embed_credentials", False)))

        self.update_auth_controls()
        self.update_info()
        self.update_link()
        self.set_status("未启动", TEXT_MUTED)
        self.loading = False

    def gather(self) -> dict:
        index = self.index
        port = self._safe_int(self.port_entry.get(), 8554 + index)
        rate = self._safe_float(self.rate_entry.get(), 10.0)
        return {
            "name": self.name_entry.get().strip() or f"推流 {index + 1}",
            "port": port,
            "asd_path": self.path_var.get().strip(),
            "rate_hz": rate,
            "auth_mode": "digest" if self.is_digest() else "open",
            "username": self.username_entry.get().strip(),
            "password": self.password_entry.get(),
            "embed_credentials": bool(self.embed_var.get()),
        }

    @staticmethod
    def _replace_entry(entry: ttk.Entry, value: str):
        entry.delete(0, "end")
        entry.insert(0, value)

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    # --------------------------------------------------------- callbacks
    def on_control_changed(self, event=None):
        if self.loading:
            return
        self.update_link()
        self.app.schedule_save()

    def on_auth_changed(self, event=None):
        self.update_auth_controls()
        self.on_control_changed()

    def update_auth_controls(self):
        editable = self.is_digest()
        self.username_entry.configure(
            state="normal" if editable else "disabled")
        self.password_entry.configure(
            state="normal" if editable else "disabled")
        self.embed_check.configure(
            state="normal" if editable else "disabled")

    def is_digest(self) -> bool:
        return self.auth_combo.get() == AUTH_DIGEST

    # ------------------------------------------------------------ package
    def browse(self):
        current = self.path_var.get().strip()
        initial_dir = (
            os.path.dirname(current)
            if current and os.path.dirname(current)
            else os.path.expanduser("~")
        )
        path = filedialog.askopenfilename(
            parent=self.app.root,
            title="选择 .asd 数据包",
            initialdir=initial_dir,
            filetypes=[("AeroSight 数据包", "*.asd"), ("所有文件", "*.*")],
        )
        if not path:
            return
        path = os.path.normpath(path)
        self.path_var.set(path)
        self.update_info()
        self.update_link()
        self.app.schedule_save()

    def clear_path(self):
        self.path_var.set("")
        self._set_info("未选择数据包", TEXT_MUTED)
        self.update_link()
        self.app.schedule_save()

    def _set_info(self, text: str, color: str):
        self.info_label.configure(text=text)
        self.app.style.configure("DetailMuted.TLabel", foreground=color)

    def update_info(self):
        path = self.path_var.get().strip()
        if not path:
            self._set_info("未选择数据包", TEXT_MUTED)
            return
        if not os.path.exists(path):
            self._set_info("数据包不存在", COLOR_ERR)
            return

        package = None
        try:
            package = AsdData(path)
            self._set_info(package.summary(), "#52647A")
        except AsdError as exc:
            self._set_info(str(exc), COLOR_ERR)
        finally:
            if package is not None:
                package.close()

    # --------------------------------------------------------------- link
    def get_port(self) -> int:
        try:
            port = int(self.port_entry.get().strip())
        except ValueError as exc:
            raise ValueError("端口必须是整数") from exc
        if not 1 <= port <= 65535:
            raise ValueError("端口范围为 1-65535")
        return port

    def get_rate(self) -> float:
        try:
            rate = float(self.rate_entry.get().strip())
        except ValueError as exc:
            raise ValueError("速率必须是数字") from exc
        if not 0.1 <= rate <= 100.0:
            raise ValueError("速率范围为 0.1-100 Hz")
        return rate

    def update_link(self):
        try:
            port = self.get_port()
        except ValueError:
            self.link_var.set("链接无效")
            return

        host = self.app.lan_ip or "127.0.0.1"
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        credentials = ""
        if self.is_digest() and self.embed_var.get() and (username or password):
            credentials = (
                f"{quote(username, safe='')}:{quote(password, safe='')}@")
        self.link_var.set(f"rtsp://{credentials}{host}:{port}/radar")

    def copy_link(self):
        link = self.link_var.get()
        if not link or link == "链接无效":
            messagebox.showinfo(
                "提示", "请先修正推流配置", parent=self.app.root)
            return
        self.app.copy_text(link)
        self.set_status("链接已复制", COLOR_OK)

    # -------------------------------------------------------------- start
    def toggle(self):
        if self.server and self.server.is_running():
            self.server.stop()
            self.server = None
            self.after_server_stopped()
            return

        try:
            path = self.path_var.get().strip()
            if not path:
                raise ValueError("请先选择一个 .asd 数据包")
            port = self.get_port()
            rate = self.get_rate()
            if self.is_digest():
                if not self.username_entry.get().strip():
                    raise ValueError("私密模式下请先设置账号")
                if not self.password_entry.get():
                    raise ValueError("私密模式下请先设置密码")
        except ValueError as exc:
            messagebox.showinfo(
                "提示", str(exc), parent=self.app.root)
            return

        self.app.save_config()
        server = RRTSPServer(
            host="0.0.0.0",
            port=port,
            asd_path=path,
            rate=rate,
            auth_mode="digest" if self.is_digest() else "open",
            username=self.username_entry.get().strip(),
            password=self.password_entry.get(),
            on_log=lambda text: self.app.log_message(self.index, text),
        )
        try:
            server.start()
        except OSError as exc:
            server.stop()
            messagebox.showerror(
                "启动失败", f"端口或网络异常：{exc}", parent=self.app.root)
            self.set_status("启动失败", COLOR_ERR)
            return
        except Exception as exc:
            server.stop()
            messagebox.showerror(
                "启动失败", str(exc), parent=self.app.root)
            self.set_status("启动失败", COLOR_ERR)
            return

        self.server = server
        self.toggle_btn.configure(text="停止")
        self.set_running_controls(False)
        self.set_status("推流中", COLOR_OK)
        self.app.log_message(self.index, "R-RTSP 推流已启动")

    def after_server_stopped(self):
        self.toggle_btn.configure(text="启动")
        self.set_running_controls(True)
        self.set_status("已停止", TEXT_MUTED)
        self.app.log_message(self.index, "R-RTSP 推流已停止")

    def set_running_controls(self, enabled: bool):
        for widget in (self.name_entry, self.port_entry, self.rate_entry,
                       self.auth_combo):
            widget.configure(state="normal" if enabled else "disabled")
        self.browse_btn.configure(state="normal" if enabled else "disabled")
        self.clear_btn.configure(state="normal" if enabled else "disabled")
        self.path_entry.configure(
            state="readonly" if enabled else "disabled")

        if self.is_digest() and enabled:
            self.username_entry.configure(state="normal")
            self.password_entry.configure(state="normal")
            self.embed_check.configure(state="normal")
        else:
            self.username_entry.configure(state="disabled")
            self.password_entry.configure(state="disabled")
            self.embed_check.configure(state="disabled")

    def stop_if_running(self):
        if self.server and self.server.is_running():
            self.server.stop()
        self.server = None

    def set_status(self, text: str, color: str):
        self.status_label.configure(text=text)
        self.app.style.configure(self.dot_style, foreground=color)


class RRTSPToolGUI:
    """主窗口：紧凑的推流组列表 + 运行日志。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.messages: queue.Queue = queue.Queue()
        self.lan_ip = detect_lan_ip()
        self._save_after = None

        root.title("AeroSight R-RTSP 推流工具")
        root.geometry("900x640")
        root.minsize(760, 540)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.style = ttk.Style()
        self.setup_style()
        self.build_widgets()
        self.load_config()
        self.root.after(120, self.poll_messages)

    def setup_style(self):
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure(".", font=("Microsoft YaHei UI", 9))
        self.style.configure("TButton", padding=(4, 2))
        self.style.configure("TEntry", padding=2)
        self.style.configure("TCheckbutton", padding=0)
        self.style.configure("TNotebook.Tab", padding=(9, 4))

        # 详情区
        self.style.configure("Detail.TFrame", background=DETAIL_BG)
        self.style.configure(
            "Detail.TLabel", background=DETAIL_BG, foreground=TEXT_DARK)
        self.style.configure(
            "DetailMuted.TLabel", background=DETAIL_BG,
            foreground=TEXT_MUTED, font=("Microsoft YaHei UI", 8))

    def build_widgets(self):
        outer = ttk.Frame(self.root, padding=(6, 6, 6, 4))
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 6))
        self.ip_label = ttk.Label(
            toolbar, text=f"本机 IP：{self.lan_ip or '未检测到'}")
        self.ip_label.pack(side="left")
        ttk.Button(toolbar, text="刷新 IP", command=self.refresh_ip).pack(
            side="right", padx=(6, 0))
        ttk.Button(toolbar, text="全部停止", command=self.stop_all).pack(
            side="right")

        pane = ttk.PanedWindow(outer, orient="vertical")
        pane.pack(fill="both", expand=True)

        # 可滚动的推流组列表
        list_wrap = ttk.Frame(pane)
        self.canvas = tk.Canvas(
            list_wrap, highlightthickness=0, borderwidth=0,
            background="#FFFFFF")
        scrollbar = ttk.Scrollbar(
            list_wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.list_inner = ttk.Frame(self.canvas)
        self._inner_id = self.canvas.create_window(
            (0, 0), window=self.list_inner, anchor="nw")

        self.items: list[StreamItem] = []
        for i in range(MAX_GROUPS):
            item = StreamItem(self.list_inner, self, i)
            self.items.append(item)
            item.container.pack(fill="x")
            if i < MAX_GROUPS - 1:
                ttk.Frame(self.list_inner, height=1,
                          style="Divider.TFrame").pack(fill="x")
        pane.add(list_wrap, weight=13)

        self.style.configure("Divider.TFrame", background=ROW_BORDER)

        self.list_inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>", self._sync_inner_width)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

        log_box = ttk.Labelframe(pane, text="运行日志", padding=4)
        self.log_text = scrolledtext.ScrolledText(
            log_box, height=7, wrap="word", font=("Consolas", 9),
            state="disabled")
        self.log_text.pack(fill="both", expand=True)
        pane.add(log_box, weight=7)

    def _sync_inner_width(self, event):
        self.canvas.itemconfigure(self._inner_id, width=event.width)

    def _bind_wheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def expand_item(self, index: int):
        for i, item in enumerate(self.items):
            if i == index:
                item.show_detail()
            elif item.expanded:
                item.set_expanded(False)

    def refresh_ip(self):
        self.lan_ip = detect_lan_ip()
        self.ip_label.configure(
            text=f"本机 IP：{self.lan_ip or '未检测到'}")
        for item in self.items:
            item.update_link()

    def stop_all(self):
        for item in self.items:
            if item.server and item.server.is_running():
                item.server.stop()
                item.server = None
                item.after_server_stopped()

    # ---------------------------------------------------------- config IO
    def schedule_save(self):
        if self._save_after:
            self.root.after_cancel(self._save_after)
        self._save_after = self.root.after(400, self.save_config)

    def save_config(self):
        data = {
            "version": 3,
            "lan_ip": self.lan_ip,
            "groups": [item.gather() for item in self.items],
        }
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_config(self):
        if not os.path.exists(config_path()):
            for i, item in enumerate(self.items):
                item.load(default_group(i))
        else:
            try:
                with open(config_path(), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            self.lan_ip = data.get("lan_ip", self.lan_ip)
            self.ip_label.configure(
                text=f"本机 IP：{self.lan_ip or '未检测到'}")
            groups = data.get("groups", [])
            for i, item in enumerate(self.items):
                item.load(groups[i] if i < len(groups) else default_group(i))

        self.expand_item(0)

    # -------------------------------------------------------------- logging
    def log_message(self, group_index: int, text: str):
        self.messages.put((group_index, str(text)))

    def poll_messages(self):
        while True:
            try:
                group_index, text = self.messages.get_nowait()
            except queue.Empty:
                break
            self.append_log(group_index, text)
        self.root.after(120, self.poll_messages)

    def append_log(self, group_index: int, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert(
            "end", f"{timestamp} [推流{group_index + 1}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def copy_text(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_add(text)

    def on_close(self):
        for item in self.items:
            item.stop_if_running()
        try:
            self.save_config()
        except Exception:
            pass
        self.root.destroy()


def run_gui() -> int:
    root = tk.Tk()
    RRTSPToolGUI(root)
    root.mainloop()
    return 0
