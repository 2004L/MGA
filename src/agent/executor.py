"""
executor.py —— 执行层（MGA ⑥ computer-use 执行：聚焦 + 原生输入注入 + 回执自检）
==============================================================================
坐在 exec/ 之上：WindowManager（先聚焦，消除"按键发到哪去了"）+ InputBackend
（SendInput 原生注入，无 robotjs 依赖）。

三个硬经验（真机踩过坑，写进代码）：
    1. SendInput 把键发给**前台窗口**——不先聚焦，键就发给了别的窗口（之前因此
       "按键无效"整局空转）。所以 act() 前必须 ensure_focus()。
    2. 按下要**长按 ~50ms**再抬起，极短按下游戏可能采样不到。
    3. 执行后要**回执校验**（画面是否变化）——没有回执就没有闭环自愈。

dry=True 时只打印不真按（沙箱/自测安全）。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

_KEYMAP = {"jump": "space", "squat": "down", "wait": None, "none": None}


class Executor:
    """执行层：把决策变成真实的键盘/鼠标动作，并能自检有没有生效。"""

    def __init__(self, prefer: str = "sendinput", dry: bool = False,
                 focus_title: str = "Chrome"):
        self.dry = dry
        self.focus_title = focus_title
        self.backend = None
        self.wm = None
        self.hwnd: Optional[int] = None
        self.backend_name = "dry"
        if dry:
            return
        try:
            from exec.backends import build_backend
            self.backend = build_backend(prefer=prefer)
            self.backend_name = type(self.backend).__name__
        except Exception as e:
            print(f"  [执行] 输入后端不可用：{type(e).__name__}: {e}")
        try:
            from exec.window import WindowManager
            self.wm = WindowManager()
        except Exception as e:
            print(f"  [执行] 窗口管理不可用：{type(e).__name__}: {e}")

    # ---------------- 聚焦：按键能不能生效全看这一步 ----------------
    def ensure_focus(self, title_contains: Optional[str] = None) -> Optional[int]:
        if self.dry or self.wm is None:
            return None
        title = title_contains or self.focus_title
        try:
            hwnd = self.wm.focus_by_title(title)
            if hwnd:
                self.hwnd = hwnd
                return hwnd
        except Exception:
            pass
        try:
            hwnd = self.wm.find(title)
            if hwnd:
                self.wm.focus(hwnd)
                self.hwnd = hwnd
                return hwnd
        except Exception:
            pass
        return None

    # ---------------- 动作执行 ----------------
    def act(self, action: str) -> bool:
        key = _KEYMAP.get(action)
        if not key:
            return False
        if self.dry or self.backend is None:
            print(f"  [执行·dry] 假装按 {action}({key})")
            return True
        try:
            self.backend.key_down(key)
            time.sleep(0.05)          # 长按 50ms，太短游戏采样不到
            self.backend.key_up(key)
            return True
        except Exception as e:
            print(f"  [执行] 按键失败：{type(e).__name__}: {e}")
            return False

    # ---------------- 回执自检：按键到底生效了吗 ----------------
    def probe(self, grab: Callable[[], np.ndarray], action: str = "jump",
              wait: float = 0.4, thr: float = 0.5) -> bool:
        """按一次键，比较前后画面差异；差异过小说明键没送到游戏（多半是没聚焦）。"""
        if self.dry:
            return True
        try:
            a = grab()
            self.act(action)
            time.sleep(wait)
            b = grab()
            diff = float(np.abs(a.astype(int) - b.astype(int)).mean())
            print(f"  [回执自检] 画面差异={diff:.2f} → "
                  f"{'按键有效(游戏有响应)' if diff > thr else '按键无效! 检查窗口聚焦'}")
            return diff > thr
        except Exception as e:
            print(f"  [回执自检] 失败：{type(e).__name__}: {e}")
            return False
