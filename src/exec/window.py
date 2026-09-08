"""
exec/window.py —— 窗口管理（枚举 / 查找 / 聚焦 / 取矩形）

参考 windows-computer-use-mcp 的 computer_list_windows / computer_focus_window：
  「点击或输入前先确认目标窗口已聚焦」是桌面自动化的基本纪律。
  全局按键发给的不是你以为的那个窗口，是**当前前台窗口**——
  先 focus 再输入，能消掉一大类"按键去哪了"的玄学问题。

系统 2 决策出坐标后，正确的执行顺序是：
    focus_window(目标) → 取窗口矩形 → 坐标换算 → 输入
而不是直接对屏幕坐标盲发。

UIPI 提示：medium IL 进程无法对 high IL（管理员启动的）窗口可靠地
SetForegroundWindow，此时 focus() 返回 False，上层应给出明确诊断。
"""

from __future__ import annotations

import ctypes
import platform
from typing import List, Optional, Tuple

from ctypes import wintypes

IS_WINDOWS = (platform.system() == "Windows")

SW_RESTORE = 9
SW_SHOW = 5


class WindowManager:
    """Windows 窗口管理；非 Windows 平台所有方法安全降级（返回空/False）。"""

    def __init__(self):
        self._u = ctypes.windll.user32 if IS_WINDOWS else None

    # -- 枚举 --
    def list_windows(self, visible_only: bool = True) -> List[dict]:
        """返回 [{"hwnd","title","pid"}]。"""
        if not IS_WINDOWS:
            return []
        u = self._u
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        out: List[dict] = []

        def _cb(hwnd, _lp):
            if visible_only and not u.IsWindowVisible(hwnd):
                return True
            n = u.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            if not title:
                return True
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            out.append({"hwnd": int(hwnd), "title": title, "pid": int(pid.value)})
            return True

        u.EnumWindows(WNDENUMPROC(_cb), 0)
        return out

    def find(self, title_contains: str) -> Optional[int]:
        """按标题子串找窗口，返回 hwnd（取第一个匹配）。"""
        for w in self.list_windows():
            if title_contains.lower() in w["title"].lower():
                return w["hwnd"]
        return None

    # -- 聚焦 --
    def focus(self, hwnd: int) -> bool:
        """聚焦窗口。返回是否成功（目标 high IL 时可能被 UIPI 拒绝）。"""
        if not IS_WINDOWS or not hwnd:
            return False
        u = self._u
        try:
            if u.IsIconic(hwnd):            # 最小化则先还原
                u.ShowWindow(hwnd, SW_RESTORE)
            else:
                u.ShowWindow(hwnd, SW_SHOW)
            u.SetForegroundWindow(hwnd)
            return u.GetForegroundWindow() == hwnd
        except Exception:
            return False

    def focus_by_title(self, title_contains: str) -> Optional[int]:
        hwnd = self.find(title_contains)
        if hwnd and self.focus(hwnd):
            return hwnd
        return None

    def is_minimized(self, hwnd: int) -> bool:
        """窗口是否最小化（最小化时 GetWindowRect 会返回 -32000 这种伪坐标）。"""
        if not IS_WINDOWS or not hwnd:
            return False
        try:
            return bool(self._u.IsIconic(hwnd))
        except Exception:
            return False

    def foreground(self) -> Optional[int]:
        if not IS_WINDOWS:
            return None
        return int(self._u.GetForegroundWindow())

    def foreground_title(self) -> str:
        if not IS_WINDOWS:
            return ""
        hwnd = self._u.GetForegroundWindow()
        n = self._u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        self._u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    # -- 几何 --
    def rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """窗口矩形 (left, top, right, bottom)。"""
        if not IS_WINDOWS or not hwnd:
            return None
        r = wintypes.RECT()
        if self._u.GetWindowRect(hwnd, ctypes.byref(r)):
            return (r.left, r.top, r.right, r.bottom)
        return None

    def client_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """客户区矩形（去掉标题栏边框）。

        注意：GetClientRect 给的是**相对客户区左上角**的坐标(left/top 恒为 0)，
        不能直接拿去截屏。要屏幕绝对坐标请用 client_screen_rect()。
        """
        if not IS_WINDOWS or not hwnd:
            return None
        r = wintypes.RECT()
        if self._u.GetClientRect(hwnd, ctypes.byref(r)):
            return (r.left, r.top, r.right, r.bottom)
        return None

    def client_screen_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """客户区的**屏幕绝对坐标** (left, top, right, bottom)，可直接喂给截屏。

        做法：GetClientRect 拿宽高，再用 ClientToScreen 把左上角换算成屏幕坐标。
        """
        if not IS_WINDOWS or not hwnd:
            return None
        u = self._u
        r = wintypes.RECT()
        if not u.GetClientRect(hwnd, ctypes.byref(r)):
            return None
        w, h = r.right - r.left, r.bottom - r.top
        if w <= 0 or h <= 0:
            return None
        pt = wintypes.POINT(0, 0)
        if not u.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        return (pt.x, pt.y, pt.x + w, pt.y + h)
